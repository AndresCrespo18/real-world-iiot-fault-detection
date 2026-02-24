# train_compare_final_dashboard_v15.py
# (Modificado para guardar modelos en archivos separados)

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import psycopg
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler


CT_BASE = ["temp_a_c", "temp_b_c", "i_u_a", "i_v_a", "i_w_a"]
CT_EXTRA = ["temp_diff", "i_mean", "i_rms", "i_std", "i_imbalance"]

VIB_FEATURES = [
    "vib_xa_p95", "vib_ya_p95", "vib_xb_p95", "vib_yb_p95",
    "vib_xa_max", "vib_ya_max", "vib_xb_max", "vib_yb_max",
    "vib_xa_rms", "vib_ya_rms", "vib_xb_rms", "vib_yb_rms",
    "vib_xa_kurt", "vib_ya_kurt", "vib_xb_kurt", "vib_yb_kurt",
]


@dataclass
class KeySplit:
    mode: str
    train_keys: List[str]
    val_keys: List[str]
    test_keys: List[str]
    boundaries: Dict[str, Dict[str, int]]
    notes: str


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def orient_score_by_auc(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, int, float]:
    auc = _safe_auc(y, s)
    if np.isnan(auc):
        return s, 1, auc
    if auc < 0.5:
        return -s, -1, float(1.0 - auc)
    return s, 1, float(auc)


def orient_prob_by_auc(y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, int, float]:
    auc = _safe_auc(y, p)
    if np.isnan(auc):
        return p, 1, auc
    if auc < 0.5:
        return 1.0 - p, -1, float(1.0 - auc)
    return p, 1, float(auc)


def thr_target_fp(norm_scores: np.ndarray, target_fp: float) -> float:
    target_fp = float(target_fp)
    target_fp = min(max(target_fp, 1e-6), 0.5)
    return float(np.quantile(norm_scores, 1.0 - target_fp))


def thr_constrained_f1(y: np.ndarray, s: np.ndarray, target_fp: float) -> Tuple[float, str]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    target_fp = float(target_fp)

    qs = np.linspace(0.0, 1.0, 401)
    thrs = np.unique(np.quantile(s, qs))

    best = None
    for thr in thrs:
        pred = (s >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
        if fpr <= target_fp:
            f1 = f1_score(y, pred, zero_division=0)
            if (best is None) or (f1 > best[0]):
                best = (f1, float(thr))
    if best is not None:
        return best[1], "constrained_f1_fpr"

    best2 = None
    for thr in thrs:
        pred = (s >= thr).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if (best2 is None) or (f1 > best2[0]):
            best2 = (f1, float(thr))
    return best2[1], "max_f1_fallback"


def eval_binary(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["avg_precision"] = float(average_precision_score(y_true, score))
    else:
        out["roc_auc"] = float("nan")
        out["avg_precision"] = float("nan")
    return out


def robust_baseline(X: np.ndarray, feature_cols: List[str]) -> Dict[str, object]:
    X = np.asarray(X, dtype=float)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    scale = np.where(mad > 1e-9, mad, 1.0)
    return {"feature_cols": list(feature_cols), "median": med.tolist(), "scale": scale.tolist()}


def add_ct_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in CT_BASE:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["temp_diff"] = df["temp_a_c"] - df["temp_b_c"]
    df["i_mean"] = (df["i_u_a"] + df["i_v_a"] + df["i_w_a"]) / 3.0
    df["i_rms"] = np.sqrt((df["i_u_a"]**2 + df["i_v_a"]**2 + df["i_w_a"]**2) / 3.0)
    df["i_std"] = df[["i_u_a", "i_v_a", "i_w_a"]].std(axis=1)
    df["i_imbalance"] = df["i_std"] / (df["i_mean"].abs() + 1e-6)
    return df


def compute_sec_idx(df: pd.DataFrame, sess_col: str, ts_col: str) -> pd.Series:
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    start = ts.groupby(df[sess_col]).transform("min")
    sec = (ts - start).dt.total_seconds().round().astype("Int64")
    return sec.astype(int)


def fetch_pairs(conn: psycopg.Connection, asset_code: str, load_nm: int) -> Tuple[int, pd.DataFrame]:
    with conn.cursor() as cur:
        cur.execute("SELECT asset_id FROM iiot.assets WHERE asset_code=%s", (asset_code,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"asset_code not found: {asset_code}")
        asset_id = int(row[0])

        sql = r'''
        WITH
        ct AS (
          SELECT
            asset_id,
            load_nm,
            lower(condition::text) AS condition,
            severity::text AS severity,
            session_id AS ct_session_id,
            regexp_replace(regexp_replace(lower(source_file), '^.*[\\\/]', ''), '\.[^.]+$', '') AS file_key
          FROM iiot.sessions
          WHERE asset_id = %s AND load_nm = %s
            AND modality = 'current_temp'::iiot.modality_t
        ),
        vib AS (
          SELECT
            asset_id,
            load_nm,
            lower(condition::text) AS condition,
            severity::text AS severity,
            session_id AS vib_session_id,
            regexp_replace(regexp_replace(lower(source_file), '^.*[\\\/]', ''), '\.[^.]+$', '') AS file_key
          FROM iiot.sessions
          WHERE asset_id = %s AND load_nm = %s
            AND modality = 'vibration'::iiot.modality_t
        )
        SELECT
          ct.asset_id, ct.load_nm, ct.condition, ct.severity, ct.file_key,
          ct.ct_session_id, vib.vib_session_id
        FROM ct
        JOIN vib
          ON vib.asset_id = ct.asset_id
         AND vib.load_nm  = ct.load_nm
         AND vib.condition = ct.condition
         AND vib.severity  = ct.severity
         AND vib.file_key  = ct.file_key
        ORDER BY ct.file_key;
        '''
        cur.execute(sql, (asset_id, int(load_nm), asset_id, int(load_nm)))
        rows = cur.fetchall()
        pairs = pd.DataFrame(rows, columns=[d[0] for d in cur.description])

    return asset_id, pairs


def fetch_ct(conn: psycopg.Connection, asset_id: int, load_nm: int, ct_ids: List[int]) -> pd.DataFrame:
    with conn.cursor() as cur:
        sql = '''
        SELECT asset_id, load_nm, session_id, ts,
               lower(condition::text) AS condition,
               severity::text AS severity,
               temp_a_c, temp_b_c, i_u_a, i_v_a, i_w_a
        FROM iiot.current_temp_1s
        WHERE asset_id = %s AND load_nm = %s
          AND session_id = ANY(%s)
        ORDER BY session_id, ts;
        '''
        cur.execute(sql, (int(asset_id), int(load_nm), ct_ids))
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description])


def fetch_vib(conn: psycopg.Connection, asset_id: int, load_nm: int, vib_ids: List[int]) -> pd.DataFrame:
    with conn.cursor() as cur:
        sql = '''
        SELECT asset_id, load_nm, session_id, ts,
               lower(condition::text) AS condition,
               severity::text AS severity,
               vib_xa_p95, vib_ya_p95, vib_xb_p95, vib_yb_p95,
               vib_xa_max, vib_ya_max, vib_xb_max, vib_yb_max,
               vib_xa_rms, vib_ya_rms, vib_xb_rms, vib_yb_rms,
               vib_xa_kurt, vib_ya_kurt, vib_xb_kurt, vib_yb_kurt
        FROM iiot.vibration_1s_wide_full
        WHERE asset_id = %s AND load_nm = %s
          AND session_id = ANY(%s)
        ORDER BY session_id, ts;
        '''
        cur.execute(sql, (int(asset_id), int(load_nm), vib_ids))
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description])


def choose_split(keys_df: pd.DataFrame, random_state: int) -> KeySplit:
    keys = keys_df["file_key"].astype(str).to_numpy()
    y = keys_df["y"].astype(int).to_numpy()

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    if min(n0, n1) >= 2:
        for k in range(80):
            rs = int(random_state + k * 17)
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=rs)
            for tr, te in sss1.split(keys, y):
                train = keys[tr]; temp = keys[te]; temp_y = y[te]
            if len(np.unique(temp_y)) < 2:
                continue
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=rs + 1)
            for tr2, te2 in sss2.split(temp, temp_y):
                val = temp[tr2]; test = temp[te2]
            notes = f"split=file_key_group(60/20/20) tries={k+1}"
            return KeySplit("file_key_group", train.tolist(), val.tolist(), test.tolist(), {}, notes)

    boundaries = {}
    for _, r in keys_df.iterrows():
        fk = str(r["file_key"])
        max_sec = int(r["max_sec_overlap"])
        n = max_sec + 1
        train_end = int(np.floor(0.6 * n) - 1)
        val_end = int(np.floor(0.8 * n) - 1)
        train_end = max(train_end, 0)
        val_end = max(val_end, train_end)
        boundaries[fk] = {"max_sec": max_sec, "train_end": train_end, "val_end": val_end}
    notes = f"split=within_file_key_time(60/20/20) because normal_keys={n0}, fault_keys={n1}"
    return KeySplit("within_key_time", keys.tolist(), [], [], boundaries, notes)


def apply_split_df(df: pd.DataFrame, split: KeySplit) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split.mode == "file_key_group":
        tr = df[df["file_key"].isin(split.train_keys)]
        va = df[df["file_key"].isin(split.val_keys)]
        te = df[df["file_key"].isin(split.test_keys)]
        cal = pd.concat([tr, va], axis=0)
        return tr, va, te, cal

    def tag_row(r):
        b = split.boundaries.get(r["file_key"])
        if b is None:
            return "train"
        if int(r["sec_idx"]) > int(b["max_sec"]):
            return None
        if int(r["sec_idx"]) <= int(b["train_end"]):
            return "train"
        if int(r["sec_idx"]) <= int(b["val_end"]):
            return "val"
        return "test"

    tags = df.apply(tag_row, axis=1)
    df2 = df.copy()
    df2["_split"] = tags
    df2 = df2[df2["_split"].notna()]
    tr = df2[df2["_split"] == "train"].drop(columns=["_split"])
    va = df2[df2["_split"] == "val"].drop(columns=["_split"])
    te = df2[df2["_split"] == "test"].drop(columns=["_split"])
    cal = pd.concat([tr, va], axis=0)
    return tr, va, te, cal


def train_iforest(X_train: np.ndarray, y_cal: np.ndarray, X_cal: np.ndarray,
                 target_fp: float, random_state: int,
                 n_estimators: int, max_features: float) -> Tuple[Pipeline, Dict[str, object]]:
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
        ("iforest", IsolationForest(
            n_estimators=int(n_estimators),
            max_features=float(max_features),
            contamination="auto",
            random_state=int(random_state),
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train)

    Xc = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(X_cal))
    s_raw = -pipe.named_steps["iforest"].score_samples(Xc)
    s, score_sign, auc = orient_score_by_auc(y_cal, s_raw)

    norm = s[y_cal == 0]
    if norm.size >= 10:
        thr_fp = thr_target_fp(norm, target_fp)
        thr_f1, mode_f1 = thr_constrained_f1(y_cal, s, target_fp)
        f1_fp = f1_score(y_cal, (s >= thr_fp).astype(int), zero_division=0)
        f1_f1 = f1_score(y_cal, (s >= thr_f1).astype(int), zero_division=0)
        if f1_f1 >= f1_fp + 1e-6:
            thr, thr_mode = thr_f1, mode_f1
        else:
            thr, thr_mode = thr_fp, "target_fp_cal_normals"
    else:
        thr, thr_mode = thr_constrained_f1(y_cal, s, target_fp)

    meta = {
        "threshold": float(thr),
        "thr_mode": thr_mode,
        "target_fp": float(target_fp),
        "score_sign": int(score_sign),
        "auc_cal": float(auc) if not np.isnan(auc) else float("nan"),
    }
    return pipe, meta


def train_sgd_epochs(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                     alpha: float, l1_ratio: float,
                     epochs: int, patience: int, random_state: int,
                     sample_weight: np.ndarray) -> Tuple[SGDClassifier, Dict[str, List[float]]]:
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        learning_rate="optimal",
        random_state=int(random_state),
        max_iter=1,
        tol=None,
        warm_start=True,
        fit_intercept=True,
    )
    classes = np.array([0, 1], dtype=int)
    history = {"val_f1": [], "val_auc": [], "val_ap": []}

    best_auc = -1.0
    best_state = None
    bad = 0

    for _ep in range(1, int(epochs) + 1):
        clf.partial_fit(Xtr, ytr, classes=classes, sample_weight=sample_weight)

        p = clf.predict_proba(Xva)[:, 1]
        auc = _safe_auc(yva, p)
        ap = float("nan") if len(np.unique(yva)) < 2 else float(average_precision_score(yva, p))
        f1 = float(f1_score(yva, (p >= 0.5).astype(int), zero_division=0))

        history["val_f1"].append(float(f1))
        history["val_auc"].append(float(auc) if not np.isnan(auc) else float("nan"))
        history["val_ap"].append(float(ap) if not np.isnan(ap) else float("nan"))

        auc_cmp = -1.0 if np.isnan(auc) else float(auc)
        if auc_cmp > best_auc + 1e-6:
            best_auc = auc_cmp
            best_state = (clf.coef_.copy(), clf.intercept_.copy())
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        clf.coef_ = best_state[0]
        clf.intercept_ = best_state[1]

    return clf, history


def train_sgd_with_calibration(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                               Xcal: np.ndarray, ycal: np.ndarray,
                               alpha: float, l1_ratio: float,
                               epochs: int, patience: int, random_state: int,
                               target_fp: float) -> Tuple[SGDClassifier, Dict[str, object], Dict[str, List[float]]]:
    n0 = max(1, int((ytr == 0).sum()))
    n1 = max(1, int((ytr == 1).sum()))
    w0 = 0.5 / n0
    w1 = 0.5 / n1
    sw = np.where(ytr == 0, w0, w1).astype(float)

    clf, hist = train_sgd_epochs(Xtr, ytr, Xva, yva, alpha, l1_ratio, epochs, patience, random_state, sw)

    p_cal_raw = clf.predict_proba(Xcal)[:, 1]
    p_cal, sign, auc = orient_prob_by_auc(ycal, p_cal_raw)

    norm = p_cal[ycal == 0]
    if norm.size >= 10:
        thr_fp = thr_target_fp(norm, target_fp)
        thr_f1, mode_f1 = thr_constrained_f1(ycal, p_cal, target_fp)
        f1_fp = f1_score(ycal, (p_cal >= thr_fp).astype(int), zero_division=0)
        f1_f1 = f1_score(ycal, (p_cal >= thr_f1).astype(int), zero_division=0)
        if f1_f1 >= f1_fp + 1e-6:
            thr, thr_mode = thr_f1, mode_f1
        else:
            thr, thr_mode = thr_fp, "target_fp_cal_normals"
    else:
        thr, thr_mode = thr_constrained_f1(ycal, p_cal, target_fp)

    meta = {
        "threshold": float(thr),
        "thr_mode": thr_mode,
        "target_fp": float(target_fp),
        "score_sign": int(sign),
        "auc_cal": float(auc) if not np.isnan(auc) else float("nan"),
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
    }
    return clf, meta, hist


def build_model_artifact(
    model: Any,
    preprocess: Dict[str, Any],
    model_meta: Dict[str, Any],
    split_info: Dict[str, Any],
    metrics: Dict[str, float],
    global_meta: Dict[str, Any],
    feature_cols: List[str],
    baseline: Dict[str, Any],
    history: Optional[Dict] = None
) -> Dict:
    """Construye un artefacto para un modelo individual."""
    artifact = {
        "meta": global_meta.copy(),
        "feature_cols": feature_cols,
        "model": model,
        "preprocess": preprocess,
        "model_meta": model_meta,
        "split_groups": split_info,
        "metrics": metrics,
        "baseline": baseline,
    }
    if history is not None:
        artifact["history"] = history
    return artifact


def train_modality(mod_name: str,
                   tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, cal: pd.DataFrame,
                   feat_cols: List[str],
                   args: argparse.Namespace,
                   split_info: Dict,
                   global_meta: Dict,
                   asset_id: int) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Entrena IsolationForest y SGD para una modalidad.
    Retorna dos artefactos: (artifact_if, artifact_sgd) o (None, None) si no hay datos.
    """
    if tr.empty or te.empty:
        return None, None

    Xtr = tr[feat_cols].to_numpy()
    ytr = tr["y"].to_numpy()
    Xva = va[feat_cols].to_numpy() if not va.empty else Xtr[:1]
    yva = va["y"].to_numpy() if not va.empty else ytr[:1]
    Xte = te[feat_cols].to_numpy()
    yte = te["y"].to_numpy()
    Xcal = cal[feat_cols].to_numpy()
    ycal = cal["y"].to_numpy()

    # Línea base robusta (para detección de anomalías simple)
    Xnorm = cal.loc[cal["y"] == 0, feat_cols].to_numpy()
    if Xnorm.shape[0] < 10:
        Xnorm = tr.loc[tr["y"] == 0, feat_cols].to_numpy()
    if Xnorm.shape[0] < 10:
        Xnorm = pd.concat([tr, va, te], axis=0)[feat_cols].to_numpy()
    baseline = robust_baseline(Xnorm, feat_cols)

    # --- Modelo 1: IsolationForest ---
    m1, m1_meta = train_iforest(
        Xtr, ycal, Xcal,
        float(args.target_fp), int(args.random_state),
        int(args.if_estimators), float(args.if_max_features)
    )
    m1_meta["baseline"] = baseline  # opcional, lo guardamos aparte

    # Evaluación en test
    Xte_sc = m1.named_steps["scaler"].transform(m1.named_steps["imputer"].transform(Xte))
    s_raw = -m1.named_steps["iforest"].score_samples(Xte_sc)
    s = float(m1_meta["score_sign"]) * s_raw
    pred1 = (s >= float(m1_meta["threshold"])).astype(int)
    met1 = eval_binary(yte, pred1, s)

    # Artefacto para IsolationForest
    preprocess_if = {
        "imputer": m1.named_steps["imputer"],
        "scaler": m1.named_steps["scaler"],
        "feature_cols": feat_cols
    }
    artifact_if = build_model_artifact(
        model=m1,
        preprocess=preprocess_if,
        model_meta=m1_meta,
        split_info=split_info,
        metrics=met1,
        global_meta=global_meta,
        feature_cols=feat_cols,
        baseline=baseline,
        history=None
    )

    # --- Modelo 2: SGD ---
    imp2 = SimpleImputer(strategy="median")
    sc2 = StandardScaler()
    Xtr2 = sc2.fit_transform(imp2.fit_transform(Xtr))
    Xva2 = sc2.transform(imp2.transform(Xva))
    Xte2 = sc2.transform(imp2.transform(Xte))
    Xcal2 = sc2.transform(imp2.transform(Xcal))

    m2, m2_meta, hist2 = train_sgd_with_calibration(
        Xtr2, ytr, Xva2, yva, Xcal2, ycal,
        float(args.sgd_alpha), float(args.sgd_l1_ratio),
        int(args.epochs), int(args.patience), int(args.random_state),
        float(args.target_fp),
    )
    m2_meta["baseline"] = baseline

    # Evaluación en test
    p_raw = m2.predict_proba(Xte2)[:, 1]
    p = (1.0 - p_raw) if int(m2_meta["score_sign"]) == -1 else p_raw
    pred2 = (p >= float(m2_meta["threshold"])).astype(int)
    met2 = eval_binary(yte, pred2, p)

    preprocess_sgd = {
        "imputer": imp2,
        "scaler": sc2,
        "feature_cols": feat_cols
    }
    artifact_sgd = build_model_artifact(
        model=m2,
        preprocess=preprocess_sgd,
        model_meta=m2_meta,
        split_info=split_info,
        metrics=met2,
        global_meta=global_meta,
        feature_cols=feat_cols,
        baseline=baseline,
        history=hist2
    )

    return artifact_if, artifact_sgd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--loads", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--out_dir", default="models_out_final_v15")

    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--target_fp", type=float, default=0.05)

    ap.add_argument("--if_estimators", type=int, default=300)
    ap.add_argument("--if_max_features", type=float, default=0.7)

    ap.add_argument("--sgd_alpha", type=float, default=1e-4)
    ap.add_argument("--sgd_l1_ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument("--ct_extra", action="store_true", default=True)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_metrics = []

    with psycopg.connect(args.dsn) as conn:
        for load_nm in args.loads:
            asset_id, pairs = fetch_pairs(conn, args.asset, int(load_nm))
            if pairs.empty:
                print(f"[WARN] No CT/VIB pairs for load={load_nm}.")
                continue

            ct_ids = pairs["ct_session_id"].astype(int).tolist()
            vib_ids = pairs["vib_session_id"].astype(int).tolist()

            df_ct = fetch_ct(conn, asset_id, int(load_nm), ct_ids)
            df_vib = fetch_vib(conn, asset_id, int(load_nm), vib_ids)

            m_ct = dict(zip(pairs["ct_session_id"].astype(int), pairs["file_key"].astype(str)))
            m_vib = dict(zip(pairs["vib_session_id"].astype(int), pairs["file_key"].astype(str)))
            df_ct["file_key"] = df_ct["session_id"].map(m_ct).astype(str)
            df_vib["file_key"] = df_vib["session_id"].map(m_vib).astype(str)

            df_ct["y"] = (df_ct["condition"] != "normal").astype(int)
            df_vib["y"] = (df_vib["condition"] != "normal").astype(int)
            df_ct["sec_idx"] = compute_sec_idx(df_ct, "session_id", "ts")
            df_vib["sec_idx"] = compute_sec_idx(df_vib, "session_id", "ts")

            if args.ct_extra:
                df_ct = add_ct_features(df_ct)
                ct_feat = CT_BASE + CT_EXTRA
            else:
                ct_feat = CT_BASE
            vib_feat = VIB_FEATURES

            ct_max = df_ct.groupby("file_key")["sec_idx"].max().rename("ct_max_sec")
            vib_max = df_vib.groupby("file_key")["sec_idx"].max().rename("vib_max_sec")
            keys_df = pd.concat([ct_max, vib_max], axis=1).dropna().reset_index()
            keys_df["max_sec_overlap"] = keys_df[["ct_max_sec","vib_max_sec"]].min(axis=1).astype(int)

            key_y = df_ct.groupby("file_key")["y"].max().rename("y")
            keys_df = keys_df.merge(key_y.reset_index(), on="file_key", how="left").dropna()
            keys_df["y"] = keys_df["y"].astype(int)

            split = choose_split(keys_df, int(args.random_state))
            print(f"\n=== LOAD {load_nm} | keys={len(keys_df)} | {split.notes} ===")

            df_ct = df_ct.merge(keys_df[["file_key","max_sec_overlap"]], on="file_key", how="inner")
            df_vib = df_vib.merge(keys_df[["file_key","max_sec_overlap"]], on="file_key", how="inner")
            df_ct = df_ct[df_ct["sec_idx"] <= df_ct["max_sec_overlap"]].copy()
            df_vib = df_vib[df_vib["sec_idx"] <= df_vib["max_sec_overlap"]].copy()
            df_ct = df_ct.drop(columns=["max_sec_overlap"])
            df_vib = df_vib.drop(columns=["max_sec_overlap"])

            # Aplicar splits
            ct_tr, ct_va, ct_te, ct_cal = apply_split_df(df_ct[["file_key","sec_idx","y"] + ct_feat], split)
            vb_tr, vb_va, vb_te, vb_cal = apply_split_df(df_vib[["file_key","sec_idx","y"] + vib_feat], split)

            # Construir dataset fusionado
            ct_keep = ["file_key","sec_idx","y"] + ct_feat
            vib_keep = ["file_key","sec_idx"] + vib_feat
            df_fused = df_ct[ct_keep].merge(df_vib[vib_keep], on=["file_key","sec_idx"], how="inner")
            df_fused["y"] = df_fused["y"].astype(int)
            fused_feat = ct_feat + vib_feat
            fu_tr, fu_va, fu_te, fu_cal = apply_split_df(df_fused[["file_key","sec_idx","y"] + fused_feat], split)

            # Metadatos globales para artefactos
            global_meta = {
                "asset": args.asset,
                "asset_id": int(asset_id),
                "load_nm": int(load_nm),
                "random_state": int(args.random_state),
            }
            split_info = {
                "group_col": "file_key",
                "mode": split.mode,
                "train": split.train_keys,
                "val": split.val_keys,
                "test": split.test_keys,
                "boundaries": split.boundaries,
                "notes": split.notes,
            }

            # ---- Entrenar modelos de CT ----
            artifact_ct_if, artifact_ct_sgd = train_modality(
                "ct", ct_tr, ct_va, ct_te, ct_cal, ct_feat, args, split_info, global_meta, asset_id
            )
            if artifact_ct_if:
                path = os.path.join(args.out_dir, f"ct_isolationforest_load{load_nm}.joblib")
                joblib.dump(artifact_ct_if, path)
                print(f"Saved: {path}")
                all_metrics.append({"modality": "ct", "load_nm": load_nm, "model": "IsolationForest", **artifact_ct_if["metrics"]})
            if artifact_ct_sgd:
                path = os.path.join(args.out_dir, f"ct_sgd_logistic_load{load_nm}.joblib")
                joblib.dump(artifact_ct_sgd, path)
                print(f"Saved: {path}")
                all_metrics.append({"modality": "ct", "load_nm": load_nm, "model": "SGD_Logistic", **artifact_ct_sgd["metrics"]})

            # ---- Entrenar modelos de VIB ----
            artifact_vib_if, artifact_vib_sgd = train_modality(
                "vib", vb_tr, vb_va, vb_te, vb_cal, vib_feat, args, split_info, global_meta, asset_id
            )
            if artifact_vib_if:
                path = os.path.join(args.out_dir, f"vib_isolationforest_load{load_nm}.joblib")
                joblib.dump(artifact_vib_if, path)
                print(f"Saved: {path}")
                all_metrics.append({"modality": "vib", "load_nm": load_nm, "model": "IsolationForest", **artifact_vib_if["metrics"]})
            if artifact_vib_sgd:
                path = os.path.join(args.out_dir, f"vib_sgd_logistic_load{load_nm}.joblib")
                joblib.dump(artifact_vib_sgd, path)
                print(f"Saved: {path}")
                all_metrics.append({"modality": "vib", "load_nm": load_nm, "model": "SGD_Logistic", **artifact_vib_sgd["metrics"]})

            # ---- Entrenar modelo fusionado (SGD) ----
            if not fu_tr.empty and not fu_te.empty:
                imp3 = SimpleImputer(strategy="median")
                sc3 = StandardScaler()
                Xtr3 = sc3.fit_transform(imp3.fit_transform(fu_tr[fused_feat].to_numpy()))
                Xva3 = sc3.transform(imp3.transform(fu_va[fused_feat].to_numpy())) if not fu_va.empty else Xtr3[:1]
                Xte3 = sc3.transform(imp3.transform(fu_te[fused_feat].to_numpy()))
                Xcal3 = sc3.transform(imp3.transform(fu_cal[fused_feat].to_numpy()))
                ytr3 = fu_tr["y"].to_numpy()
                yva3 = fu_va["y"].to_numpy() if not fu_va.empty else ytr3[:1]
                yte3 = fu_te["y"].to_numpy()
                ycal3 = fu_cal["y"].to_numpy()

                m3, m3_meta, m3_hist = train_sgd_with_calibration(
                    Xtr3, ytr3, Xva3, yva3, Xcal3, ycal3,
                    float(args.sgd_alpha), float(args.sgd_l1_ratio),
                    int(args.epochs), int(args.patience), int(args.random_state),
                    float(args.target_fp),
                )

                # Línea base robusta para fused
                Xnorm3 = fu_cal.loc[fu_cal["y"] == 0, fused_feat].to_numpy()
                if Xnorm3.shape[0] < 10:
                    Xnorm3 = fu_tr.loc[fu_tr["y"] == 0, fused_feat].to_numpy()
                if Xnorm3.shape[0] < 10:
                    Xnorm3 = df_fused[fused_feat].to_numpy()
                baseline3 = robust_baseline(Xnorm3, fused_feat)
                m3_meta["baseline"] = baseline3

                p_raw = m3.predict_proba(Xte3)[:, 1]
                p = (1.0 - p_raw) if int(m3_meta["score_sign"]) == -1 else p_raw
                pred = (p >= float(m3_meta["threshold"])).astype(int)
                m3_metrics = eval_binary(yte3, pred, p)

                preprocess3 = {
                    "imputer": imp3,
                    "scaler": sc3,
                    "feature_cols": fused_feat
                }
                artifact_fused = build_model_artifact(
                    model=m3,
                    preprocess=preprocess3,
                    model_meta=m3_meta,
                    split_info=split_info,
                    metrics=m3_metrics,
                    global_meta=global_meta,
                    feature_cols=fused_feat,
                    baseline=baseline3,
                    history=m3_hist
                )
                path = os.path.join(args.out_dir, f"fused_sgd_logistic_load{load_nm}.joblib")
                joblib.dump(artifact_fused, path)
                print(f"Saved: {path}")
                all_metrics.append({"modality": "ctvib", "load_nm": load_nm, "model": "SGD_Fused", **m3_metrics})

    if all_metrics:
        dfm = pd.DataFrame(all_metrics)
        out_csv = os.path.join(args.out_dir, "metrics_summary.csv")
        dfm.to_csv(out_csv, index=False)
        print(f"\nWrote summary: {out_csv}")


if __name__ == "__main__":
    main()