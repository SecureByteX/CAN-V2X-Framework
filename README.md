# CAN-Bus and V2X Intrusion Detection Framework for Secure ADAS Perception

## Project Overview
This project presents a domain-separated Intrusion Detection System (IDS) designed to secure connected autonomous vehicles and ADAS perception pipelines. It independently monitors in-vehicle CAN-Bus traffic and external DSRC/C-V2X wireless communication, then visualizes the resulting security alerts in a unified Virtual Cybersecurity Dashboard.

This is a 40% working prototype focused on demonstrating the core architecture, attack simulation, feature engineering, and ML-based detection flow for both domains.

---

## 40% Working Prototype Architecture

```text
+--------------------------------------------------------------------------+
|                 VIRTUAL CYBERSECURITY DASHBOARD (UI)                     |
|          (Chart.js, Bootstrap 5, Dark Cybersecurity Theme)             |
+--------------------------------------------------------------------------+
|                 Flask Web Server & API Layer (app.py)                  |
|                   SQLite Database (database/db.py)                     |
+-----------------------------------+----------------------------------------+
|           CAN IDS MODULE           |           V2X IDS MODULE              |
|  - CAN generator & attack engine   |  - V2X BSM generator & attack engine  |
|  - Timing and frequency extractors |  - Kinematic and sequence extractors  |
|  - CAN ML models (Random Forest)  |  - V2X ML models (Random Forest)      |
+-----------------------------------+----------------------------------------+
```

> [!IMPORTANT]
> **Strict Domain Isolation**: The CAN-Bus and V2X IDS pipelines are implemented as completely separate ML models and feature-engineering modules.

---

## Key Features Completed (40% Milestone)

1. **Synthetic CAN-Bus Traffic and Attack Simulator**
   - Normal ECU traffic streaming, including Engine `0x0C0`, Speed `0x120`, Brake `0x200`, and Steering `0x316`.
   - Simulated attack types include **Injection**, **Spoofing**, **DoS Flooding**, and **Replay**.

2. **Synthetic V2X BSM Wireless Generator and Attack Simulator**
   - DSRC / C-V2X Basic Safety Message (BSM) traffic at 10 Hz.
   - Simulated attack types include **V2X Spoofing (Ghost Vehicles)**, **Replay**, **Channel DoS Flooding**, and **False Emergency Safety Warnings**.

3. **Domain-Specific Feature Extractors**
   - CAN features: inter-arrival time, rolling frequency, payload byte entropy, byte sum, byte standard deviation, and burst rate.
   - V2X features: sequence consistency, GPS displacement speed consistency, acceleration consistency, and heading change rate.

4. **Independent ML Training and Model Benchmarking**
   - Evaluated Logistic Regression, Random Forest, and Gradient Boosting / XGBoost.
   - Metrics include Accuracy, Precision, Recall, F1-score, FPR, and inference time.

5. **Flask Backend and SQLite Event Logging**
   - Real-time endpoints: `/api/can-alerts`, `/api/v2x-alerts`, `/api/statistics`, and `/api/model-performance`.
   - Separate event tables are maintained for CAN and V2X detections.

6. **Virtual Cybersecurity Dashboard**
   - Compact V-SOC overview with backend health, recorded alerts, source-separated distributions, and offline model evaluation.
   - Searchable alert filters and details from persisted database values.
   - CAN and V2X simulator / dataset testing remains on the separate `/can` and `/v2x` pages.

---

## Dashboard Verification Rules

- Simulation records include a ground-truth `label`. A record is counted as a true positive only when its actual label is an attack and the model predicts that exact attack class.
- CAN/V2X alert tables, confirmed-attack counts, severity counts, and attack charts include true positives only. Node inspection tables show both exact-match attack true positives and correctly identified normal true negatives. False positives, false negatives, and incorrect attack-class predictions are retained as event outcomes but are not presented as verified results.
- Records without a ground-truth label are marked as unverified and excluded from verified dashboard totals. This includes older database rows created before ground-truth tracking was introduced.
- Severity is a fixed mapping derived from the predicted class, not a second ML output. Confidence is the classifier probability for the predicted class and is not a guarantee of correctness.
- The model-performance chart reports offline holdout metrics (accuracy and F1) from training evaluation; it is not a count of live true-positive alerts.
- Live real-world traffic without labels cannot be confirmed as a true positive at detection time. This prototype's true-positive-only alerts apply to labeled datasets and simulation modes.

## V-SOC Overview

The `/dashboard` page reads the normalized SOC summary from `/api/soc`. It displays only recorded, labeled true-positive events; it does not create placeholder alerts or infer that a vehicle is under attack. This project does not currently ingest a live CAN or V2X vehicle feed, so database records are explicitly labeled simulator/test outcomes. A real input feed must be connected before the page can represent production vehicle monitoring.

The time-range filter uses the database recording time (`created_at`), not a simulator timestamp. Time range, source, severity, attack type, and search filters apply to the alert list. Clicking an alert displays the persisted model and feature values when available; older rows display “Not available” for fields they did not store. CAN and V2X simulator / dataset tools remain on their separate pages.

`/api/health` reports database and predictor readiness; `/api/ready` returns HTTP 503 while a required dependency is degraded and is used by the Render health check. The model section displays stored offline held-out evaluation artifacts; it is not live accuracy. The VeReMi “Under evaluation” status compares its best stored macro-F1 with `V2X_MACRO_F1_REVIEW_THRESHOLD` (default `0.70`). This threshold is configurable and does not modify or inflate measured scores.

---

## Folder Structure

```text
CAN-V2X-IDS/
│
├── app.py                       # Main Flask application and REST API
├── config.py                    # System configuration and path constants
├── requirements.txt             # Python dependency requirements
├── README.md                    # Technical documentation and setup guide
├── .gitignore                   # Git tracking exclusion rules
├── .env.example                 # Environment variables template
│
├── data/
│   ├── can/
│   │   ├── raw/                 # Synthetic raw CAN CSV logs
│   │   └── processed/           # Preprocessed CAN feature dataset
│   └── v2x/
│       ├── raw/                 # Synthetic raw V2X CSV logs
│       └── processed/           # Preprocessed V2X feature dataset
│
├── ml/
│   ├── preprocessing.py         # Generic cleaning and dataset splitters
│   ├── feature_engineering.py    # Domain feature extractors (CAN and V2X)
│   ├── can/
│   │   ├── train.py             # CAN model training script
│   │   ├── predict.py           # Real-time CAN alert predictor
│   │   └── model.pkl            # Trained CAN model artifact
│   └── v2x/
│       ├── train.py            # V2X model training script
│       ├── predict.py          # Real-time V2X alert predictor
│       └── model.pkl           # Trained V2X model artifact
│
├── simulation/
│   ├── can_generator.py         # Normal CAN stream generator
│   ├── can_attacks.py           # CAN attack injection simulator
│   ├── v2x_generator.py         # Normal V2X BSM generator
│   └── v2x_attacks.py           # V2X wireless attack simulator
│
├── evaluation/
│   ├── can_evaluation.py        # Metrics calculator and report generator
│   └── v2x_evaluation.py        # V2X metrics calculator
│
├── database/
│   ├── db.py                    # SQLite database setup and connection
│   └── models.py                # Data access objects (DAO)
│
├── templates/
│   ├── dashboard.html           # Dual-domain cybersecurity dashboard
│   ├── can.html                 # CAN IDS control-node interface
│   └── v2x.html                 # V2X IDS control-node interface
│
├── static/
│   ├── css/style.css            # Dark cybersecurity UI theme
│   └── js/dashboard.js          # Real-time polling and Chart.js renderer
│
└── tests/
    ├── test_can.py              # CAN unit tests
    └── test_v2x.py              # V2X unit tests
```

---

## Installation and Running Instructions (Windows)

### Prerequisites
- Python 3.11 installed on your system.

### Step 1: Create and Activate a Virtual Environment
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run the Unit Test Suite
```powershell
python -m pytest -q
```

### Step 4: Launch the Production Web Server
```powershell
waitress-serve --listen=0.0.0.0:5000 app:app
```

### Step 5: Access the Web Interface
Open `http://127.0.0.1:5000/` in your browser. The root page opens directly to the dashboard.

To access the app from another device on the same Wi-Fi or LAN, find your computer's local IP address using `ipconfig` and open `http://<computer-ip>:5000/` on that device. Allow Python through the Windows Firewall on a private network if prompted. The dashboard and API have no login requirement, so anyone who can reach the server can use them. Do not expose them to the public internet without adding authentication and access controls.

The dashboard separates historical synthetic simulator events from offline dataset evaluation. The CAN node exposes only HCRL-trained classes. The VeReMi page includes all 20 original `AttackerType` labels from the supplied raw dataset, including `Benign`. Buttons evaluate held-out rows from the same class and only show correct predictions. Absolute receive time and receiver identity are excluded from VeReMi model features. The older five-class synthetic V2X event panel is labeled as a legacy simulator and is not used as VeReMi model evidence.

---

## Dataset Evaluation Method

- **CAN**: Balanced HCRL sample of 500,000 rows (100,000 per class), reservoir-sampled across the supplied captures. Each class's last 20% by timestamp is held out. The selected algorithm is chosen by macro F1.
- **VeReMi**: All 2,269,540 raw rows and all 20 original labels, with the last 20% by receive time held out separately within each label. The 16 numeric position, speed, acceleration, heading, and noise fields are used as model features. `rcvTime`, `ReceiverID`, and `AttackerType` are excluded from feature selection.
- **Model comparison**: Logistic Regression (SGD), Random Forest, and XGBoost are compared on the same split within each dataset. Macro F1 is used for model selection. Weighted and macro precision, recall, F1, accuracy, false-positive rate, and repeated-batch inference time are shown in the dashboard.
- **Current held-out results**: HCRL CAN XGBoost achieves 99.78% accuracy and 99.78% macro F1. Raw VeReMi XGBoost achieves 21.52% accuracy and 22.98% macro F1. The VeReMi result is low and the model is not production-ready; its 20 classes overlap substantially in the supplied features. Do not present its score as 100%.
- **Severity and confidence**: Severity is a rule-based display band derived from the predicted class, not a machine-learning output or a label supplied by either dataset. Confidence is the model's predicted-class probability and is not a guarantee of correctness.

Retrain the raw VeReMi benchmark using:

```powershell
python -m ml.v2x.train_veremi
```

This is separate from the legacy five-class synthetic V2X simulator model.

---

## Deploy as a Public Website

The project includes a Render Blueprint in `render.yaml`. Push the project to a GitHub repository, create a new Blueprint in Render, and select that repository. Commit the three model artifacts and the two compact holdout CSV files explicitly unignored in `.gitignore`; without them the readiness check fails and the dataset testing pages cannot work. Render then builds the app and provides a public HTTPS URL. The production Waitress server binds to Render's assigned port, and Render generates the Flask secret key. The dashboard and API are open to anyone who can access the URL; add authentication and access controls before public deployment.

This prototype uses SQLite at the path configured by `DATABASE_PATH` (default: `database/ids_system.db`). On Render's free web service, local files can be discarded when the service restarts or redeploys, so event history is not durable. Configure `DATABASE_PATH` to a persistent mounted path or migrate to a managed database before relying on long-term alert storage. Free services may also sleep when idle.

---

## Interactive 40% Prototype Demonstration Guide

### DEMO A: HCRL CAN Holdout Evaluation
1. Open the dashboard at `http://127.0.0.1:5000/dashboard`.
2. Click **CAN Node Controls** or navigate to `http://127.0.0.1:5000/can`.
3. The page creates one button for each HCRL class: Normal, DoS, FUZZY, GEAR, and RPM.
4. Select a class. The table is replaced with up to 50 matching chronological holdout rows, showing the actual label, model prediction, outcome, confidence, and rule-based severity.
5. A normal-class result is a true negative. An attack-class result is shown only when it is a true positive.

### DEMO B: Raw VeReMi Evaluation
1. Open **VeReMi Dataset** or navigate to `http://127.0.0.1:5000/v2x`.
2. The page creates one button for `Benign` and each of the 19 raw attack labels, including DoS variants, replay, position, speed, and Sybil classes.
3. Select a label. The table is replaced with up to 50 held-out rows from that class. Only exact-match attack true positives or correctly predicted Benign true negatives are shown.
4. The overview HCRL and raw VeReMi charts show training label distributions, including green Normal/Benign slices. The model table compares candidates only within the same dataset.

---

## Using the Uploaded HCRL CAN Dataset

The HCRL Car-Hacking captures are stored under `data/can/raw/CAN — HCRL Car-Hacking Dataset/`. Prepare a balanced training file and retrain the CAN model with:

```powershell
python -m ml.can.prepare_hcrl --source-dir "data/can/raw/CAN — HCRL Car-Hacking Dataset" --output "data/can/processed/can_hcrl_training.csv"
python -m ml.can.train
```

The preparation step reads the original captures as a stream, maps `R` records in the separate normal capture to `NORMAL`, maps `T` records to the corresponding attack class for each capture, and reservoir-samples up to 100,000 records per class. It discards repeated `R` records embedded in each attack capture and keeps only the five fields used by this CAN pipeline. The uploaded originals are not modified. Training then uses a per-class chronological holdout from this balanced file.

The HCRL upload contains CAN data only. Raw VeReMi is trained separately with all 20 labels and is no longer collapsed into the legacy synthetic V2X classes. The CAN balanced file is a 100,000-per-class sample from larger source captures, not every HCRL row. These are chronological per-class holdouts, but they are still within-dataset evaluations rather than independent cross-vehicle validation.

---

## 40% Review Project Guide Explanation
> "We have completed the domain-specific intrusion detection layer. CAN and V2X are intentionally implemented as separate IDS modules because they use different communication mechanisms, traffic characteristics, features, and attack patterns. The current prototype independently detects attacks in both domains and presents the results through a common virtual cybersecurity dashboard."

---

## Future Scope (Phases Beyond 40%)
- Alert normalization and threat correlation
- Risk assessment engine
- Explainable AI (SHAP / LIME)
- LLM security incident investigation assistant
- MITRE ATT&CK mapping for ICS / automotive systems
- Automated countermeasure and V-SOC integration
