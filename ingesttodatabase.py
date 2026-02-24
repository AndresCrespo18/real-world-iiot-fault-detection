# ingest_schema_aligned.py
# --------------------------------------------------------------------
# Ingest aligned with schema.sql (sessions has NOT NULL: modality, file_ext, fs_hz)
#
# Default behavior (safe + ML-friendly):
#   - CT (TDMS): store 1-second means into iiot.current_temp_samples
#   - VIB (MAT): store per-second features into iiot.vibration_1s_features (+ view vibration_1s_wide already in schema)
#   - AC  (MAT): store per-second features into iiot.acoustic_1s_features (if present)
#
# Optional (VERY LARGE): store raw samples
#   --ct_level raw|both, --vib_level raw|both, --ac_level raw|both
#
# Key robustness:
#   - Fixes time base when files use sample index as "time" (converts using fs)
#   - Normalizes condition labels to match iiot.condition_t enum
#   - Parses severity including unbalance "####mg"
#
# Requirements:
#   pip install psycopg[binary] numpy pandas scipy nptdms tqdm
#
# Example (recommended):
#   python ingest_schema_aligned.py --data_dir "C:\...\datasetr" --dsn "postgresql://..." --asset MOTOR_001 --overwrite
# --------------------------------------------------------------------

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg

try:
    from nptdms import TdmsFile
except Exception:
    TdmsFile = None

try:
    from scipy.io import loadmat
    from scipy.stats import kurtosis
except Exception:
    loadmat = None
    kurtosis = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x


FS_CT_DEFAULT = 25600.0
FS_VIB_DEFAULT = 25600.0
FS_AC_DEFAULT = 51200.0
DURATION_SANITY_MAX = 600.0

# aaaaNm_bbbb_cccc.ext   where cccc can be 03,10,30,01,05, 0583mg, etc.
FNAME_RE = re.compile(
    r"(?P<load>\d+)Nm_(?P<cond>[^_.]+)(?:_(?P<sev>[^.]+))?\.(?P<ext>tdms|mat)$",
    re.IGNORECASE,
)

COND_MAP = {
    "normal": "normal",
    "bpfi": "bpfi",
    "bpfo": "bpfo",
    "misalign": "misalign",
    "misalignment": "misalign",
    "unbalance": "unbalance",
    "unbalalnce": "unbalance",
}


def normalize_condition(token: str) -> str:
    t = str(token).strip().lower()
    return COND_MAP.get(t, t)


def parse_meta(p: Path) -> Optional[Tuple[int, str, str, str]]:
    m = FNAME_RE.search(p.name)
    if not m:
        return None
    load_nm = int(m.group("load"))
    cond = normalize_condition(m.group("cond"))
    sev = (m.group("sev") or "00").strip()
    ext = (m.group("ext") or "").lower()
    # normalize severity (keep mg if present)
    sev = sev.lower()
    return load_nm, cond, sev, ext


def normalize_time_seconds(t: np.ndarray, fs_hint: float) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size == 0:
        return t
    if not np.isfinite(t).all():
        t = np.arange(t.size, dtype=float)
    t = t - float(t[0])
    if t.size < 3:
        return t
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return np.arange(t.size, dtype=float) / float(fs_hint)
    duration = float(t[-1] - t[0])

    # If looks like sample-index time (dt ~ 1, duration huge), convert by fs
    if duration > DURATION_SANITY_MAX and 0.5 <= dt <= 2.0 and fs_hint >= 1000:
        return t / float(fs_hint)

    # If implied hz ~ 1 but vector is huge, also treat as index
    implied_hz = 1.0 / dt
    if implied_hz <= 5.0 and t.size > 100000 and fs_hint >= 1000:
        return t / float(fs_hint)

    return t


def estimate_fs(t_s: np.ndarray, fs_default: float) -> float:
    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    if t_s.size < 3:
        return float(fs_default)
    dt = float(np.median(np.diff(t_s)))
    if not np.isfinite(dt) or dt <= 0:
        return float(fs_default)
    fs = 1.0 / dt
    # sanity clamp
    if fs < 1000:
        return float(fs_default)
    return float(fs)


# ----------------- TDMS CT -----------------
@dataclass
class CtSignals:
    t_s: np.ndarray
    temp_a_c: np.ndarray
    temp_b_c: np.ndarray
    i_u_a: np.ndarray
    i_v_a: np.ndarray
    i_w_a: np.ndarray
    fs_hz: float


def _tdms_infer_wave_time(ch) -> Optional[np.ndarray]:
    props = getattr(ch, "properties", {}) or {}
    dt = props.get("wf_increment") or props.get("WF_INCREMENT")
    if dt is None:
        return None
    try:
        dt = float(dt)
    except Exception:
        return None
    if not np.isfinite(dt) or dt <= 0:
        return None
    start_off = props.get("wf_start_offset") or props.get("WF_START_OFFSET") or 0.0
    try:
        start_off = float(start_off)
    except Exception:
        start_off = 0.0
    n = len(ch[:])
    return start_off + np.arange(n, dtype=float) * dt


def read_tdms_current_temp(tdms_path: Path) -> CtSignals:
    if TdmsFile is None:
        raise RuntimeError("Missing dependency: nptdms. Install: pip install nptdms")

    td = TdmsFile.read(str(tdms_path))
    channels = []
    for g in td.groups():
        for ch in g.channels():
            name = f"{g.name}/{ch.name}".lower()
            arr = np.asarray(ch[:])
            props = getattr(ch, "properties", {}) or {}
            unit = str(props.get("unit_string") or props.get("Unit") or props.get("units") or "").lower()
            channels.append((name, arr, unit, ch))

    if not channels:
        raise RuntimeError(f"No channels found in TDMS: {tdms_path.name}")

    # time channel
    time_arr = None
    for name, arr, unit, _ch in channels:
        if "time" in name or "timestamp" in name:
            if np.issubdtype(np.asarray(arr).dtype, np.number):
                time_arr = np.asarray(arr, dtype=float)
                break

    if time_arr is None:
        for _name, _arr, _unit, ch in channels:
            t = _tdms_infer_wave_time(ch)
            if t is not None:
                time_arr = np.asarray(t, dtype=float)
                break

    numeric = [(name, np.asarray(arr), unit) for (name, arr, unit, _ch) in channels if np.issubdtype(np.asarray(arr).dtype, np.number)]
    nmax = max(len(a) for _, a, _ in numeric)

    if time_arr is None or len(time_arr) != nmax:
        time_arr = np.arange(nmax, dtype=float) / FS_CT_DEFAULT

    t_s = normalize_time_seconds(time_arr, fs_hint=FS_CT_DEFAULT)
    fs_hz = estimate_fs(t_s, FS_CT_DEFAULT)

    # map signals by name
    def pick_by_keywords(keywords: List[str]) -> List[np.ndarray]:
        out = []
        for name, arr, unit in numeric:
            if any(k in name for k in keywords):
                out.append(np.asarray(arr, dtype=float))
        return out

    # Prefer explicit names
    temps = pick_by_keywords(["temperature", "temp"])
    currents = pick_by_keywords(["u-phase", "u phase", "v-phase", "v phase", "w-phase", "w phase", "current"])

    # Fallback: unit + variance
    if len(temps) < 2 or len(currents) < 3:
        pool = [(name, np.asarray(arr, dtype=float), unit) for name, arr, unit in numeric]
        # temps by units containing C
        if len(temps) < 2:
            t2 = [a for n, a, u in pool if ("c" in u or "°c" in u)]
            temps = t2 if len(t2) >= 2 else temps
        # currents by units containing A
        if len(currents) < 3:
            c3 = [a for n, a, u in pool if (u in ("a", "amp", "amps", "ampere", "amperes") or "a" == u)]
            currents = c3 if len(c3) >= 3 else currents

    # Final fallback: variance heuristic
    if len(temps) < 2 or len(currents) < 3:
        pool = [np.asarray(arr, dtype=float) for _, arr, _ in numeric]
        pool = sorted(pool, key=lambda x: float(np.nanvar(x)), reverse=True)
        if len(currents) < 3 and len(pool) >= 3:
            currents = pool[:3]
        if len(temps) < 2 and len(pool) >= 5:
            temps = pool[3:5]

    candidates = temps[:2] + currents[:3]
    n = min([len(t_s)] + [len(x) for x in candidates if len(x) > 0])

    def take(lst: List[np.ndarray], idx: int) -> np.ndarray:
        if len(lst) > idx and len(lst[idx]) >= n:
            return np.asarray(lst[idx][:n], dtype=float)
        return np.full(n, np.nan, dtype=float)

    return CtSignals(
        t_s=np.asarray(t_s[:n], dtype=float),
        temp_a_c=take(temps, 0),
        temp_b_c=take(temps, 1),
        i_u_a=take(currents, 0),
        i_v_a=take(currents, 1),
        i_w_a=take(currents, 2),
        fs_hz=float(fs_hz),
    )


def ct_to_1s_means(sig: CtSignals) -> pd.DataFrame:
    sec = np.floor(sig.t_s).astype(int)
    df = pd.DataFrame({
        "sec": sec,
        "temp_a_c": sig.temp_a_c,
        "temp_b_c": sig.temp_b_c,
        "i_u_a": sig.i_u_a,
        "i_v_a": sig.i_v_a,
        "i_w_a": sig.i_w_a,
    })
    out = df.groupby("sec", sort=True).mean(numeric_only=True).reset_index()
    return out


# ----------------- MAT helpers -----------------
def _walk_numeric(root):
    arrays_1d = []
    arrays_2d = []
    scalars = []

    def walk(obj):
        if hasattr(obj, "_fieldnames"):
            for f in obj._fieldnames:
                try:
                    walk(getattr(obj, f))
                except Exception:
                    pass
            return
        if isinstance(obj, np.void) and obj.dtype.fields:
            for f in obj.dtype.fields:
                try:
                    walk(obj[f])
                except Exception:
                    pass
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).startswith("__"):
                    continue
                walk(v)
            return
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj)
            if arr.dtype == object:
                for idx in np.ndindex(arr.shape):
                    walk(arr[idx])
                return
            if arr.dtype.fields:
                for f in arr.dtype.fields:
                    try:
                        walk(arr[f])
                    except Exception:
                        pass
                return
            if np.issubdtype(arr.dtype, np.number):
                if arr.ndim == 0:
                    try:
                        scalars.append(float(arr))
                    except Exception:
                        pass
                elif arr.ndim == 1 and arr.size > 1000:
                    arrays_1d.append(arr.astype(float))
                elif arr.ndim == 2 and arr.size > 1000:
                    arrays_2d.append(arr.astype(float))
            return
        if isinstance(obj, (int, float)) and np.isfinite(obj):
            scalars.append(float(obj))

    walk(root)
    return arrays_1d, arrays_2d, scalars


def mat_to_vibration_df(mat_path: Path) -> Tuple[pd.DataFrame, float]:
    if loadmat is None:
        raise RuntimeError("Missing dependency: scipy. Install: pip install scipy")
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    root = mat.get("Signal", mat)
    arrays_1d, arrays_2d, scalars = _walk_numeric(root)

    # Nx5 (t + 4 axes)
    for arr in sorted(arrays_2d, key=lambda a: a.size, reverse=True):
        r, c = arr.shape
        if r > 1000 and c >= 5:
            df = pd.DataFrame(arr[:, :5], columns=["t", "xA", "yA", "xB", "yB"])
            t_s = normalize_time_seconds(df["t"].to_numpy(), FS_VIB_DEFAULT)
            df["t_s"] = t_s
            fs_hz = estimate_fs(t_s, FS_VIB_DEFAULT)
            return df[["t_s","xA","yA","xB","yB"]], fs_hz
        if c > 1000 and r == 5:
            df = pd.DataFrame(arr.T[:, :5], columns=["t", "xA", "yA", "xB", "yB"])
            t_s = normalize_time_seconds(df["t"].to_numpy(), FS_VIB_DEFAULT)
            df["t_s"] = t_s
            fs_hz = estimate_fs(t_s, FS_VIB_DEFAULT)
            return df[["t_s","xA","yA","xB","yB"]], fs_hz

    # Nx4 with time elsewhere
    data4 = None
    for arr in sorted(arrays_2d, key=lambda a: a.size, reverse=True):
        r, c = arr.shape
        if r > 1000 and c == 4:
            data4 = arr
            break
        if c > 1000 and r == 4:
            data4 = arr.T
            break
    if data4 is None:
        keys = [k for k in mat.keys() if not str(k).startswith("__")]
        raise RuntimeError(f"Vibration MAT layout not recognized: {mat_path.name}. Keys: {keys}")

    N = int(data4.shape[0])

    tvec = None
    best = -1.0
    for v in arrays_1d:
        if len(v) != N:
            continue
        dv = np.diff(v)
        score = float((dv >= 0).mean())
        if score > best:
            best = score
            tvec = np.asarray(v, dtype=float)

    if tvec is None:
        tvec = np.arange(N, dtype=float) / FS_VIB_DEFAULT

    t_s = normalize_time_seconds(tvec, FS_VIB_DEFAULT)
    fs_hz = estimate_fs(t_s, FS_VIB_DEFAULT)

    df = pd.DataFrame({
        "t_s": t_s.astype(float),
        "xA": data4[:, 0].astype(float),
        "yA": data4[:, 1].astype(float),
        "xB": data4[:, 2].astype(float),
        "yB": data4[:, 3].astype(float),
    })
    return df, fs_hz


def mat_to_acoustic_df(mat_path: Path) -> Tuple[pd.DataFrame, float]:
    if loadmat is None:
        raise RuntimeError("Missing dependency: scipy. Install: pip install scipy")
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    arrays_1d, arrays_2d, scalars = _walk_numeric(mat)
    if "Signal" in mat:
        a1, a2, sc = _walk_numeric(mat["Signal"])
        arrays_1d += a1
        arrays_2d += a2
        scalars += sc

    # Nx2
    for arr in sorted(arrays_2d, key=lambda a: a.size, reverse=True):
        r, c = arr.shape
        if r > 1000 and c >= 2:
            df = pd.DataFrame(arr[:, :2], columns=["t", "value"])
            t_s = normalize_time_seconds(df["t"].to_numpy(), FS_AC_DEFAULT)
            df["t_s"] = t_s
            fs_hz = estimate_fs(t_s, FS_AC_DEFAULT)
            return df[["t_s","value"]], fs_hz
        if c > 1000 and r == 2:
            df = pd.DataFrame(arr.T[:, :2], columns=["t", "value"])
            t_s = normalize_time_seconds(df["t"].to_numpy(), FS_AC_DEFAULT)
            df["t_s"] = t_s
            fs_hz = estimate_fs(t_s, FS_AC_DEFAULT)
            return df[["t_s","value"]], fs_hz

    # fallback: largest vector as value
    candidates = []
    for v in arrays_1d:
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size > 1000:
            candidates.append(v)
    if not candidates:
        keys = [k for k in mat.keys() if not str(k).startswith("__")]
        raise RuntimeError(f"Acoustic MAT layout not recognized: {mat_path.name}. Keys: {keys}")

    value = max(candidates, key=lambda v: v.size)
    N = int(value.size)

    tvec = None
    best = -1.0
    for v in arrays_1d:
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size != N:
            continue
        dv = np.diff(v)
        score = float((dv >= 0).mean())
        if score > best:
            best = score
            tvec = v

    if tvec is None:
        tvec = np.arange(N, dtype=float) / FS_AC_DEFAULT

    t_s = normalize_time_seconds(tvec, FS_AC_DEFAULT)
    fs_hz = estimate_fs(t_s, FS_AC_DEFAULT)
    df = pd.DataFrame({"t_s": t_s.astype(float), "value": value.astype(float)})
    return df, fs_hz


# ----------------- 1s feature extraction -----------------
def vib_1s_features(vib: pd.DataFrame) -> pd.DataFrame:
    if kurtosis is None:
        raise RuntimeError("Missing scipy.stats.kurtosis. Install scipy.")
    sec = np.floor(vib["t_s"].to_numpy(dtype=float)).astype(int)

    rows = []
    for axis in ["xA","yA","xB","yB"]:
        x = vib[axis].to_numpy(dtype=float)
        tmp = pd.DataFrame({"sec": sec, "x": x})
        g = tmp.groupby("sec", sort=True)["x"]

        secs = g.mean().index.to_numpy()
        rms = np.sqrt(g.apply(lambda s: float(np.mean(np.square(s.to_numpy(dtype=float))))).to_numpy())
        mx = g.max().to_numpy(dtype=float)
        p95 = g.quantile(0.95).to_numpy(dtype=float)
        kur = g.apply(lambda s: float(kurtosis(s.to_numpy(dtype=float), fisher=False, bias=False)) if len(s) > 3 else np.nan).to_numpy(dtype=float)

        for i in range(len(secs)):
            rows.append((int(secs[i]), axis, float(rms[i]), float(mx[i]), float(p95[i]), None if np.isnan(kur[i]) else float(kur[i])))
    return pd.DataFrame(rows, columns=["sec","axis","rms_g","max_g","p95_g","kurtosis"])


def ac_1s_features(ac: pd.DataFrame) -> pd.DataFrame:
    if kurtosis is None:
        raise RuntimeError("Missing scipy.stats.kurtosis. Install scipy.")
    sec = np.floor(ac["t_s"].to_numpy(dtype=float)).astype(int)
    x = ac["value"].to_numpy(dtype=float)
    tmp = pd.DataFrame({"sec": sec, "x": x})
    g = tmp.groupby("sec", sort=True)["x"]

    secs = g.mean().index.to_numpy()
    rms = np.sqrt(g.apply(lambda s: float(np.mean(np.square(s.to_numpy(dtype=float))))).to_numpy())
    mx = g.max().to_numpy(dtype=float)
    p95 = g.quantile(0.95).to_numpy(dtype=float)
    kur = g.apply(lambda s: float(kurtosis(s.to_numpy(dtype=float), fisher=False, bias=False)) if len(s) > 3 else np.nan).to_numpy(dtype=float)

    return pd.DataFrame({
        "sec": secs.astype(int),
        "rms_pa": rms.astype(float),
        "max_pa": mx.astype(float),
        "p95_pa": p95.astype(float),
        "kurtosis": [None if np.isnan(k) else float(k) for k in kur],
    })


# ----------------- DB helpers -----------------
def exec_sql_file(cur: psycopg.Cursor, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    # very simple splitter; schema.sql uses BEGIN/COMMIT and DO $$ blocks; execute as one.
    cur.execute(sql)


def ensure_asset(cur: psycopg.Cursor, asset_code: str) -> int:
    cur.execute(
        "INSERT INTO iiot.assets(asset_code) VALUES (%s) "
        "ON CONFLICT(asset_code) DO UPDATE SET asset_code=EXCLUDED.asset_code "
        "RETURNING asset_id;",
        (asset_code,),
    )
    return int(cur.fetchone()[0])


def get_session(cur: psycopg.Cursor, asset_id: int, modality: str, source_file: str) -> Optional[int]:
    cur.execute(
        "SELECT session_id FROM iiot.sessions WHERE asset_id=%s AND modality=%s AND source_file=%s LIMIT 1;",
        (asset_id, modality, source_file),
    )
    r = cur.fetchone()
    return int(r[0]) if r else None


def delete_session(cur: psycopg.Cursor, asset_id: int, session_id: int) -> None:
    cur.execute("DELETE FROM iiot.sessions WHERE asset_id=%s AND session_id=%s;", (asset_id, session_id))


def create_session(
    cur: psycopg.Cursor,
    asset_id: int,
    modality: str,
    source_file: str,
    file_ext: str,
    load_nm: int,
    condition: str,
    severity: str,
    fs_hz: float,
    start_ts: datetime,
    end_ts: datetime,
) -> int:
    cur.execute(
        """
        INSERT INTO iiot.sessions(asset_id, modality, source_file, file_ext, load_nm, condition, severity, fs_hz, start_ts, end_ts)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING session_id;
        """,
        (asset_id, modality, source_file, file_ext, int(load_nm), condition, severity, float(fs_hz), start_ts, end_ts),
    )
    return int(cur.fetchone()[0])


def copy_rows(cur: psycopg.Cursor, table: str, cols: List[str], rows_iter) -> None:
    with cur.copy(f"COPY {table} ({','.join(cols)}) FROM STDIN") as cp:
        for row in rows_iter:
            cp.write_row(row)


# ----------------- Ingest per modality -----------------
def ingest_ct(
    cur: psycopg.Cursor,
    asset_id: int,
    p: Path,
    load_nm: int,
    cond: str,
    sev: str,
    start_ts: datetime,
    overwrite: bool,
    ct_level: str,
) -> Tuple[int, datetime]:
    modality = "current_temp"
    source_file = str(p)
    sid = get_session(cur, asset_id, modality, source_file)
    if sid and overwrite:
        delete_session(cur, asset_id, sid)
        sid = None

    sig = read_tdms_current_temp(p)
    duration_s = float(sig.t_s[-1]) if sig.t_s.size else 0.0
    end_ts = start_ts + timedelta(seconds=max(1.0, duration_s))

    if sid is None:
        sid = create_session(cur, asset_id, modality, source_file, "tdms", load_nm, cond, sev, sig.fs_hz, start_ts, end_ts)

    # RAW
    if ct_level in ("raw", "both"):
        n = len(sig.t_s)
        def rows():
            for i in range(n):
                yield (
                    asset_id, sid, int(i), float(sig.t_s[i]),
                    None if np.isnan(sig.temp_a_c[i]) else float(sig.temp_a_c[i]),
                    None if np.isnan(sig.temp_b_c[i]) else float(sig.temp_b_c[i]),
                    None if np.isnan(sig.i_u_a[i]) else float(sig.i_u_a[i]),
                    None if np.isnan(sig.i_v_a[i]) else float(sig.i_v_a[i]),
                    None if np.isnan(sig.i_w_a[i]) else float(sig.i_w_a[i]),
                    "DATASET",
                )
        copy_rows(cur, "iiot.current_temp_raw",
                  ["asset_id","session_id","sample_idx","t_s","temp_a_c","temp_b_c","i_u_a","i_v_a","i_w_a","source"],
                  rows())

    # 1s (recommended)
    if ct_level in ("1s", "both"):
        agg = ct_to_1s_means(sig)
        if not agg.empty:
            def rows2():
                for sec, ta, tb, iu, iv, iw in agg[["sec","temp_a_c","temp_b_c","i_u_a","i_v_a","i_w_a"]].itertuples(index=False, name=None):
                    ts = start_ts + timedelta(seconds=int(sec))
                    yield (
                        ts, asset_id, sid, int(load_nm), cond, sev,
                        None if pd.isna(ta) else float(ta),
                        None if pd.isna(tb) else float(tb),
                        None if pd.isna(iu) else float(iu),
                        None if pd.isna(iv) else float(iv),
                        None if pd.isna(iw) else float(iw),
                        "DATASET",
                    )
            copy_rows(cur, "iiot.current_temp_samples",
                      ["ts","asset_id","session_id","load_nm","condition","severity","temp_a_c","temp_b_c","i_u_a","i_v_a","i_w_a","source"],
                      rows2())

    return sid, end_ts


def ingest_vib(
    cur: psycopg.Cursor,
    asset_id: int,
    p: Path,
    load_nm: int,
    cond: str,
    sev: str,
    start_ts: datetime,
    overwrite: bool,
    vib_level: str,
) -> Tuple[int, datetime]:
    modality = "vibration"
    source_file = str(p)
    sid = get_session(cur, asset_id, modality, source_file)
    if sid and overwrite:
        delete_session(cur, asset_id, sid)
        sid = None

    vib, fs_hz = mat_to_vibration_df(p)
    duration_s = float(vib["t_s"].iloc[-1]) if len(vib) else 0.0
    end_ts = start_ts + timedelta(seconds=max(1.0, duration_s))

    if sid is None:
        sid = create_session(cur, asset_id, modality, source_file, "mat", load_nm, cond, sev, fs_hz, start_ts, end_ts)

    # RAW (huge)
    if vib_level in ("raw", "both"):
        n = len(vib)
        def rows():
            for i, (t_s, xa, ya, xb, yb) in enumerate(vib[["t_s","xA","yA","xB","yB"]].itertuples(index=False, name=None)):
                yield (
                    asset_id, sid, int(i), float(t_s),
                    float(xa), float(ya), float(xb), float(yb),
                    "DATASET",
                )
        copy_rows(cur, "iiot.vibration_raw",
                  ["asset_id","session_id","sample_idx","t_s","xA_g","yA_g","xB_g","yB_g","source"],
                  rows())

    # 1s features
    if vib_level in ("1s", "both"):
        feat = vib_1s_features(vib)
        if not feat.empty:
            def rows2():
                for sec, axis, rms, mx, p95, kur in feat.itertuples(index=False, name=None):
                    ts = start_ts + timedelta(seconds=int(sec))
                    yield (
                        ts, asset_id, sid, int(load_nm), cond, sev,
                        axis, float(rms), float(mx), float(p95), kur,
                        "DATASET",
                    )
            copy_rows(cur, "iiot.vibration_1s_features",
                      ["ts","asset_id","session_id","load_nm","condition","severity","axis","rms_g","max_g","p95_g","kurtosis","source"],
                      rows2())

    return sid, end_ts


def ingest_ac(
    cur: psycopg.Cursor,
    asset_id: int,
    p: Path,
    load_nm: int,
    cond: str,
    sev: str,
    start_ts: datetime,
    overwrite: bool,
    ac_level: str,
) -> Tuple[int, datetime]:
    modality = "acoustic"
    source_file = str(p)
    sid = get_session(cur, asset_id, modality, source_file)
    if sid and overwrite:
        delete_session(cur, asset_id, sid)
        sid = None

    ac, fs_hz = mat_to_acoustic_df(p)
    duration_s = float(ac["t_s"].iloc[-1]) if len(ac) else 0.0
    end_ts = start_ts + timedelta(seconds=max(1.0, duration_s))

    if sid is None:
        sid = create_session(cur, asset_id, modality, source_file, "mat", load_nm, cond, sev, fs_hz, start_ts, end_ts)

    if ac_level in ("raw", "both"):
        n = len(ac)
        def rows():
            for i, (t_s, v) in enumerate(ac[["t_s","value"]].itertuples(index=False, name=None)):
                yield (asset_id, sid, int(i), float(t_s), float(v), "DATASET")
        copy_rows(cur, "iiot.acoustic_raw",
                  ["asset_id","session_id","sample_idx","t_s","value_pa","source"],
                  rows())

    if ac_level in ("1s", "both"):
        feat = ac_1s_features(ac)
        if not feat.empty:
            def rows2():
                for sec, rms, mx, p95, kur in feat[["sec","rms_pa","max_pa","p95_pa","kurtosis"]].itertuples(index=False, name=None):
                    ts = start_ts + timedelta(seconds=int(sec))
                    yield (
                        ts, asset_id, sid, int(load_nm), cond, sev,
                        float(rms), float(mx), float(p95), kur,
                        "DATASET",
                    )
            copy_rows(cur, "iiot.acoustic_1s_features",
                      ["ts","asset_id","session_id","load_nm","condition","severity","rms_pa","max_pa","p95_pa","kurtosis","source"],
                      rows2())

    return sid, end_ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root folder with current_temp/, vibration/, acoustic/")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--asset", default="MOTOR_001")

    ap.add_argument("--schema_sql", default="schema.sql", help="Path to schema.sql (run once to create tables)")
    ap.add_argument("--init_schema", action="store_true", help="Execute schema_sql before ingest (recommended for fresh DB)")
    ap.add_argument("--overwrite", action="store_true", help="Delete & reinsert session per file")

    ap.add_argument("--ct_level", choices=["1s","raw","both"], default="1s")
    ap.add_argument("--vib_level", choices=["1s","raw","both"], default="1s")
    ap.add_argument("--ac_level", choices=["1s","raw","both"], default="1s")

    ap.add_argument("--only", choices=["all","ct","vib","ac"], default="all")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--start_ts", default="2026-01-01T00:00:00Z")
    ap.add_argument("--gap_s", type=int, default=2)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ct_dir = data_dir / "current_temp"
    vib_dir = data_dir / "vibration"
    ac_dir = data_dir / "acoustic"

    ct_files = sorted(ct_dir.glob("*.tdms")) if ct_dir.exists() else []
    vib_files = sorted(vib_dir.glob("*.mat")) if vib_dir.exists() else []
    ac_files = sorted(ac_dir.glob("*.mat")) if ac_dir.exists() else []

    if args.limit and args.limit > 0:
        ct_files = ct_files[:args.limit]
        vib_files = vib_files[:args.limit]
        ac_files = ac_files[:args.limit]

    start_ts = datetime.fromisoformat(args.start_ts.replace("Z", "+00:00")).astimezone(timezone.utc)

    with psycopg.connect(args.dsn) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        with conn.cursor() as cur:
            if args.init_schema:
                exec_sql_file(cur, Path(args.schema_sql))
            asset_id = ensure_asset(cur, args.asset)
        conn.commit()

        cur_ts = start_ts

        def process(files, kind):
            nonlocal cur_ts
            if not files:
                print(f"[{kind}] No files found.")
                return
            with conn.cursor() as cur:
                for p in tqdm(files, desc=f"Ingest {kind}"):
                    meta = parse_meta(p)
                    if not meta:
                        print(f"[{kind}] skip (cannot parse): {p.name}")
                        continue
                    load_nm, cond, sev, ext = meta
                    try:
                        if kind == "ct":
                            _sid, end_ts = ingest_ct(cur, asset_id, p, load_nm, cond, sev, cur_ts, args.overwrite, args.ct_level)
                        elif kind == "vib":
                            _sid, end_ts = ingest_vib(cur, asset_id, p, load_nm, cond, sev, cur_ts, args.overwrite, args.vib_level)
                        else:
                            _sid, end_ts = ingest_ac(cur, asset_id, p, load_nm, cond, sev, cur_ts, args.overwrite, args.ac_level)
                        conn.commit()
                        cur_ts = end_ts + timedelta(seconds=int(args.gap_s))
                    except Exception as e:
                        conn.rollback()
                        print(f"[{kind}] ERROR {p.name}: {e}")
                        cur_ts = cur_ts + timedelta(seconds=int(args.gap_s))

        if args.only in ("all","ct"):
            process(ct_files, "ct")
        if args.only in ("all","vib"):
            process(vib_files, "vib")
        if args.only in ("all","ac"):
            process(ac_files, "ac")

        # quick audit
        with conn.cursor() as cur:
            cur.execute("""
                SELECT modality, load_nm, condition,
                       COUNT(*) AS n_sessions,
                       ROUND(AVG(EXTRACT(EPOCH FROM (end_ts-start_ts)))::numeric, 2) AS avg_sec
                FROM iiot.sessions
                WHERE asset_id=%s
                GROUP BY modality, load_nm, condition
                ORDER BY modality, load_nm, condition;
            """, (asset_id,))
            rows = cur.fetchall()

        print("\n=== AUDIT sessions (avg_sec should be around 60/120) ===")
        for r in rows:
            print(f"{r[0]:<12} load={r[1]} cond={r[2]:<10} n={r[3]:<3} avg_sec={r[4]}")
        print("\n✅ Ingest completed.")

if __name__ == "__main__":
    main()
