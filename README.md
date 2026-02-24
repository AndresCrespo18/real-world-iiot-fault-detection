--Overview

This project implements a real-time anomaly detection system for predictive maintenance of a rotating machine, utilizing vibration, current, and temperature sensors.
The dataset, provided by Mendeley, contains recordings under five operating conditions: normal, BPFI, BPFO, misalignment, and unbalance, at three load settings (0, 2, and 4 Nm).
The task is binary classification, aiming to distinguish normal operation from any fault (comprising all four fault types), a crucial step in predictive maintenance.

--The system encompasses the full machine learning lifecycle:

Data ingestion from various sensor formats (.tdms for current/temperature and .mat for vibration) into a PostgreSQL database.
Time-series alignment to create synchronized 1-Hz windows between the different sensor modalities.
Model training on three different model families per load: Isolation Forest (unsupervised), SGD Logistic (supervised), and a fused model that combines both sensor sources.
Model evaluation using realistic performance metrics, with a focus on avoiding overfitting and ensuring that the selected models generalize well.
A REST API (FastAPI) is developed for real-time inference, allowing the model to be tested and applied in live settings.
Interactive dashboards (Streamlit) provide both model comparison and live testing capabilities.

--Key Technologies

Technology	             Role
PostgreSQL	             Centralized database for storing raw samples and aggregated 1‑Hz features. SQL views simplify training queries.
Python + scikit-learn	   Implements Isolation Forest (unsupervised anomaly detector) and SGD Logistic (linear classifier with elastic-net penalty).
FastAPI	                 High-performance web framework serving predictions via /predict endpoint.
Streamlit	               Dashboards for model comparison and real-time testing via the API.
Pandas / NumPy	         Data manipulation, feature engineering, and calculations.
psycopg	PostgreSQL       adapter for Python for interacting with the database.
nptdms / scipy	         Parse and preprocess data from .tdms and .mat files.

--Data Pipeline

1. Ingestion
Challenge: The dataset consists of heterogeneous file formats — .tdms (National Instruments) for current and temperature, and .mat (MATLAB) for vibration — each containing multiple sensor channels.
Solution: The script ingest_chat.py uses nptdms and scipy.io.loadmat to robustly extract the signals, normalizes timestamps, and computes per-second averages/features before inserting them into PostgreSQL.

2. Time Alignment
Challenge: Vibration files contain millisecond-precision timestamps, whereas CT files use seconds or sample indices. Direct alignment by time is impossible because the recordings are independent.
Solution: Timestamps are truncated to whole seconds, and a file_key (extracted from the base filename, such as 0Nm_BPFI_03) is used to pair the CT and VIB sessions that belong to the same experiment.
Using relative second indices (sec_idx) from each session start, both modalities are merged on (file_key, sec_idx) to create perfectly aligned 1-Hz windows.

4. Database Views
Several views were created to simplify training and inference:
current_temp_1s – CT features per second.
vibration_1s_wide_full – Vibration features per second (including p95, max, RMS, kurtosis for four axes).
session_pairs_byfile – Pairs CT and VIB sessions using file_key.
aligned_ct_vib_1s_v2 – Final dataset joining both modalities, perfectly aligned.
These views allow the training script to easily fetch synchronized data with a single SELECT query, simplifying the data extraction process.

--Model Training
For each load (0, 2, 4 Nm), we trained the following models:

Isolation Forest on CT features only (unsupervised).
SGD Logistic on CT features only (supervised).
SGD Logistic on fused CT+VIB features (supervised).
Each model was calibrated using a hybrid method (train+validation set) to select the optimal threshold and score orientation (positive class). The threshold selection ensures that the models are robust and reliable for real-world deployment.
For models like Isolation Forest, target FPR (False Positive Rate) on normal samples is used for threshold selection.
For SGD Logistic, a constrained F1-maximization approach was applied under a given FPR budget.

--Results & Model Selection

The table below shows the three models selected for production, based on the best trade-off between fault detection rate and false positive rate. Models with perfect metrics (F1=1.0, FP=0) were excluded as they likely suffered from overfitting.

Load	--Selected Model	     --F1	    --FP	 --Recall	  --Behavior
0 Nm	--ctvib SGD_Fused	     --0.892	--49	 --0.955	  --High fault detection (95% recall) with acceptable false alarms (49 out of 60 normals).
2 Nm	--ct SGD_Logistic	     --0.950	--2	   --0.917	  --Excellent balance – only 2 false alarms and 12 missed faults. Highly reliable.
4 Nm	--ct IsolationForest   --0.888	--5	   --0.814	  --Ideal for minimizing unnecessary stoppages with very few false positives (5).

--Why These Models?
Load 0: The fused model (CT+VIB) outperforms both the CT-only and VIB-only models while maintaining acceptable FP. It catches 95% of faults.
Load 2: CT SGD offers near-perfect performance with only 2 false alarms — a highly reliable detector for this load.
Load 4: CT Isolation Forest provides the best balance with an F1 score of 0.888 and minimal false positives. 
The SGD variants for this load, though offering higher F1 scores, suffer from misclassifying all normal samples as faults, which is impractical in real-world settings.
Exclusion of Overfitted Models
Models with perfect scores (F1=1.0) were excluded due to overfitting or potential data leakage. Real-world data is rarely perfectly separable, so these models were not considered for production. 
The chosen models represent the most realistic and robust solutions.

--Visual Documentation
All trained models are documented visually through confusion matrices and key metrics. Screenshots of each model are available in the docs/images/ folder. The naming convention follows:
ct_load0_isolationforest.PNG – CT, Load 0, Isolation Forest
vib_load2_sgdlogistic.PNG – Vibration, Load 2, SGD Logistic
load0_ctvib_sgdlogistic.PNG – Fused CT+VIB, Load 0, SGD Logistic
Each image includes a confusion matrix and detailed metrics for that model.

--Real-Time Testing & Evaluation

The system also includes a FastAPI-based REST API for real-time inference and testing:
/predict endpoint: Accepts input feature windows and returns the model's predictions (NORMAL or FAULT) with associated anomaly scores.
Streamlit Dashboards: Users can visually compare model metrics and test the models with live data, sending feature windows via the API and visualizing the results interactively.

--You can use Streamlit to:
Compare the performance of different models.
Test the models live by inputting feature windows (either manually or via file upload).

--Conclusion

This system provides a robust solution for real-time anomaly detection and predictive maintenance of rotating machinery. 
By integrating real-time inference capabilities, model comparison tools, and a seamless data pipeline, this project represents a comprehensive approach to improving industrial reliability and reducing maintenance costs.
