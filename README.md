# SentinelGraph

## Temporal Network Attack Forecasting System

SentinelGraph is a cybersecurity intelligence system designed to move beyond traditional intrusion detection by forecasting future attack risk from evolving network behavior.

Instead of asking:

> "What attack is happening now?"

SentinelGraph focuses on:

> "Given the current network state and its recent history, what is likely to happen next?"

The system transforms network traffic into temporal network states, constructs historical sequences, and uses machine learning and temporal deep learning models to forecast future attack activity.

The project is being developed as a research-oriented MVP during the current semester. Advanced capabilities such as graph-based modeling, latent world models, multi-horizon rollouts, and attack progression modeling form the subsequent research direction.

---

# 1. Problem Statement

Traditional Intrusion Detection Systems primarily focus on identifying whether the current observed network activity is malicious.

SentinelGraph explores a different objective:

```text
Current Network State
        +
Recent Network History
        ↓
Future Attack Forecast
```

The objective is to determine whether temporal patterns in network behavior can provide an early indication of future attack activity.

The system therefore focuses on:

* Network traffic analysis
* Network state representation
* Temporal behavior modeling
* Future attack forecasting
* Attack-risk estimation
* Attack-intensity forecasting
* Explainable security analytics
* Early-warning analysis

---

# 2. Core Idea

The fundamental distinction between conventional detection and SentinelGraph is:

### Traditional IDS

```text
Network Traffic
      ↓
Current Activity
      ↓
Attack / Benign
```

### SentinelGraph

```text
Network Traffic
      ↓
Network State S(t)
      ↓
Recent State History
      ↓
Temporal Forecasting
      ↓
Future Attack Risk
```

The long-term research direction extends this into learned network dynamics:

```text
Network State S(t)
      ↓
Learned Representation
      ↓
State Transition Dynamics
      ↓
Future States
      ↓
Future Attack Risk / Intensity / Progression
```

---

# 3. System Architecture

## Current Implementation

```text
CIC-IDS2018 Network Traffic
            |
            v
      Data Cleaning
            |
            v
     Feature Engineering
            |
            v
    Temporal Network States
            |
            v
     Temporal Sequences
            |
            v
 Chronological Train / Validation / Test
            |
            v
      Forecasting Models
            |
       +----+----+
       |         |
       v         v
 Classical     LSTM
 Baselines
       |         |
       +----+----+
            |
            v
    Future Attack Prediction
            |
            v
        Evaluation
```

## Planned Semester MVP

```text
Network Traffic
      |
      v
Data Processing
      |
      v
Network State Construction
      |
      v
Temporal State History
      |
      v
Attack Forecasting
      |
      +-----------------------+
      |                       |
      v                       v
Future Attack Risk     Attack Intensity
      |                       |
      +-----------+-----------+
                  |
                  v
           Explainability
                  |
                  v
         Early-Warning Analysis
                  |
                  v
              Dashboard
```

---

# 4. Data Pipeline

The current data pipeline is:

```text
CIC-IDS2018 Dataset
        |
        v
Raw Network Flow Data
        |
        v
Data Cleaning
        |
        v
Clean Flow Dataset
        |
        v
Feature Engineering
        |
        v
Cybersecurity Feature Dataset
        |
        v
30-second Network States
        |
        v
Temporal Sequence Generation
        |
        v
Chronological Dataset Split
        |
        v
Forecasting Experiments
```

---

# 5. Pipeline Stages

## 5.1 Data Cleaning

Input:

```text
Raw CIC-IDS2018 CSV files
```

The cleaning pipeline performs operations including:

* Schema normalization
* Datatype correction
* Timestamp conversion
* Missing-value handling
* Corrupted-row handling
* Label normalization

Output:

```text
data/interim/
```

---

## 5.2 Feature Engineering

The feature pipeline converts cleaned flow records into security-oriented features.

### Traffic Behavior

* Total packets
* Total bytes
* Average packet size
* Byte rate
* Packet rate

### Directional Behavior

* Forward packet ratio
* Backward packet ratio

### TCP Behavior

* SYN rate
* ACK rate
* RST rate

### Timing Behavior

* Flow duration
* Inter-arrival-time statistics

Output:

```text
data/features/
```

---

## 5.3 Temporal Network State Generation

Individual network flows are aggregated into temporal network states.

Current aggregation window:

```text
30 seconds
```

Conceptually:

```text
Flow 1
Flow 2
Flow 3
Flow 4
   |
   v
Network State S(t)
```

Each state summarizes network behavior observed during a temporal window.

Output:

```text
data/states/
```

---

## 5.4 Temporal Sequence Generation

The forecasting dataset is constructed using consecutive network states.

Current sequence length:

```text
10 states
```

Conceptually:

```text
S(t-9) → S(t-8) → ... → S(t-1) → S(t)
                                      |
                                      v
                              Future Target
```

Each generated sequence currently contains:

```text
X
y_attack
y_intensity
last_attack
```

Where:

| Field         | Description                                       |
| ------------- | ------------------------------------------------- |
| `X`           | Historical network-state sequence                 |
| `y_attack`    | Future attack indicator                           |
| `y_intensity` | Future attack ratio                               |
| `last_attack` | Attack state of the most recently observed window |

The current sequence representation has:

```text
10 temporal states
×
24 features
```

Therefore the sequence shape is:

```text
(N, 10, 24)
```

---

# 6. Dataset

## CIC-IDS2018

The current implementation uses the Canadian Institute for Cybersecurity Intrusion Detection System Dataset 2018.

The dataset contains:

* Network flow records
* Benign traffic
* Multiple attack scenarios

Attack categories represented in the dataset include:

* Bot
* DoS
* DDoS
* Brute Force
* Web Attacks
* Infiltration

The original dataset is **not included in this repository**.

Download the dataset separately and place the required files under:

```text
data/raw/CIC-IDS2018/
```

Example:

```text
data/
└── raw/
    └── CIC-IDS2018/
        ├── Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
        └── Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
```

---

# 7. Experimental Dataset Splits

SentinelGraph maintains separate dataset configurations for next-state
classification and exploratory attack-onset forecasting.

## Expanded Next-State Classification Split

TRAIN
- Wednesday-14-02-2018
- Thursday-15-02-2018
- Friday-16-02-2018
- Wednesday-21-02-2018

VALIDATION
- Friday-23-02-2018
- Wednesday-28-02-2018

TEST
- Friday-02-03-2018

| Split | Sequences |
|---|---:|
| Train | 2,419 |
| Validation | 2,210 |
| Test | 1,028 |
| Total | 5,657 |

February 22 is not included in this expanded split.

The March 2 test day has a substantially higher attack prevalence
than the training and validation sets. This is a distribution-shift
evaluation on one test day, not broad evidence of cross-day generalization.

Preprocessing is fitted using training data only.

## Exploratory Attack-Onset Split

Training uses February 14, 15, 16, 21, and 22. February 23 is used
for exploratory evaluation.

| Split | Eligible sequences | Onsets | Onset prevalence |
|---|---:|---:|---:|
| Train | 2,688 | 110 | 4.09% |
| Test (Feb 23) | 900 | 86 | 9.56% |

Only sequences whose last observed state is benign are included.
February 23 was previously used for validation in an earlier experiment,
so this is not an untouched independent test set.

See `reports/onset_forecasting_experiment.md` for details.

---

# 8. Current Forecasting Models

The current model progression is:

```text
Majority Baseline
        ↓
Persistence Baseline
        ↓
Logistic Regression
        ↓
XGBoost
        ↓
LSTM
```

The purpose of this model ladder is to establish increasingly capable reference points before introducing more advanced temporal and structural models.

---

# 9. Baseline Models

## Majority Baseline

Predicts the majority class for every sample.

Purpose:

> Establish a trivial reference point.

---

## Persistence Baseline

Uses the most recently observed attack state as the prediction for the future state.

Purpose:

> Determine how much predictive power exists simply because attack states exhibit temporal persistence.

This is an important baseline for a forecasting problem.

---

## Logistic Regression

A linear baseline trained on the flattened temporal representation.

Current input representation:

```text
10 states × 24 features = 240 features
```

Purpose:

> Establish a linear reference model.

---

## XGBoost

A nonlinear tree-based baseline trained on the flattened temporal representation.

Current configuration uses:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

The initial benchmark uses a fixed classification threshold of:

```text
0.50
```

---

## LSTM

The LSTM operates directly on the temporal sequence representation:

```text
(batch, sequence_length, features)
```

Current configuration:

```text
Sequence length = 10
Features = 24
Hidden size = 64
Batch size = 64
Learning rate = 0.001
Optimizer = Adam
Early stopping = enabled
```

The model predicts future attack probability.

---

# 10. Current Experimental Results

Current held-out test results:

| Model               |     F1 | PR-AUC | ROC-AUC | Recall |    FPR |
| ------------------- | -----: | -----: | ------: | -----: | -----: |
| Majority            | 0.0000 | 0.1762 |  0.5000 | 0.0000 | 0.0000 |
| Persistence         | 0.5708 | 0.4017 |  0.7393 | 0.5697 | 0.0911 |
| Logistic Regression | 0.2706 | 0.3124 |  0.4854 | 0.2418 | 0.1166 |
| XGBoost             | 0.3987 | 0.5142 |  0.7670 | 0.2541 | 0.0044 |
| LSTM                | 0.1369 | 0.3876 |  0.7034 | 0.0738 | 0.0009 |

### Interpretation

The current results demonstrate that:

* Attack persistence is a strong reference point.
* XGBoost obtains the highest current PR-AUC and ROC-AUC among the tested models.
* Persistence provides substantial predictive information.
* The LSTM achieves a non-random ROC-AUC but has very low recall at the fixed 0.50 threshold on the held-out test set.
* Validation and test behavior differ substantially for some models.

The validation/test difference motivates further investigation of:

* Distribution shift
* Generalization
* Threshold stability
* Calibration
* Temporal forecasting robustness

These are research questions rather than reasons to discard the models.

---

# 11. Evaluation Metrics

The current experiments use:

## Classification

* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* False Positive Rate
* Confusion Matrix

Accuracy is not treated as the primary metric because of class imbalance.

## Future Forecasting

The MVP will additionally evaluate:

* Future attack intensity
* Intensity MAE
* Intensity MSE
* Forecast horizon degradation
* Early-warning lead time

## Later Research Evaluation

Future research stages will additionally consider:

* Brier score
* Expected Calibration Error
* Reliability diagrams
* Distribution-shift performance
* Attack-family generalization
* Error analysis
* Ablation studies

---

# 12. Current Implementation Status

## Completed

* [x] CIC-IDS2018 dataset integration
* [x] Data cleaning pipeline
* [x] Schema normalization
* [x] Security-focused feature engineering
* [x] 30-second temporal network state generation
* [x] Temporal sequence generation
* [x] Chronological train/validation/test split
* [x] Leakage-safe preprocessing
* [x] Majority baseline
* [x] Persistence baseline
* [x] Logistic Regression baseline
* [x] XGBoost baseline
* [x] LSTM temporal forecasting baseline
* [x] Held-out evaluation
* [x] Probability diagnostics for temporal model

## Current Work

* [ ] Current-state vs temporal-history experiment
* [ ] MLP baseline
* [ ] Attack-intensity forecasting
* [ ] End-to-end MVP inference pipeline
* [ ] Explainability
* [ ] Early-warning analysis
* [ ] Offline inference workflow
* [ ] Dashboard

---

# 13. Semester MVP

The goal for the current semester is to finish a complete working MVP.

The MVP will extend the current forecasting pipeline into an operational offline system:

```text
Offline Network Traffic
        |
        v
Data Processing
        |
        v
Network State Construction
        |
        v
Temporal History
        |
        v
Forecasting Model
        |
        +-------------------------+
        |                         |
        v                         v
Future Attack Risk        Future Intensity
        |                         |
        +------------+------------+
                     |
                     v
               Explanation
                     |
                     v
            Early-Warning Signal
                     |
                     v
                Dashboard
```

The MVP should demonstrate the complete path from network telemetry to an interpretable future-risk forecast.

---

# 14. MVP Outputs

The semester MVP is intended to provide:

## Future Attack Risk

Example:

```text
Future attack probability: 73%
```

## Future Attack Intensity

Example:

```text
Predicted future attack intensity: 0.42
```

## Temporal Risk

The system should be able to show how predicted risk changes as the network state evolves.

## Explanation

The system should provide the major traffic features contributing to a forecast.

Example:

```text
Important contributing signals:

- SYN rate
- Packet rate
- Flow frequency
- Byte rate
- RST rate
```

The exact explanation method will depend on the final model used in the MVP.

---

# 15. SIH Alignment

The project is designed around the requirements of the SIH network attack forecasting problem.

The MVP focuses on the core requirements that can be implemented and demonstrated during the semester:

| Requirement                  | SentinelGraph                                             |
| ----------------------------- | ----------------------------------------------------------- |
| Network traffic ingestion    | CIC-IDS2018 flow data                                     |
| Network-state representation | 30-second network states                                  |
| Temporal modeling            | 10-state temporal sequences                               |
| Future attack prediction     | Current forecasting models                                |
| Attack intensity             | `y_intensity` target and planned forecasting              |
| Explainability               | Planned MVP component                                     |
| Offline operation            | Planned MVP workflow                                      |
| Benchmarking                 | Majority, Persistence, Logistic Regression, XGBoost, LSTM |
| User interface               | Planned MVP dashboard                                     |

More advanced SIH-oriented capabilities will be investigated in subsequent research stages.

---

# 16. Research Direction

The central research question is:

> To what extent can temporal and structural modeling of network telemetry provide reliable early warning of future cyberattack progression?

A second research direction investigates whether a learned latent state-transition model can represent useful network dynamics well enough to forecast future network states and attack risk over multiple horizons.

The project follows the principle:

```text
Question
   ↓
Hypothesis
   ↓
Experiment
   ↓
Result
   ↓
Interpretation
   ↓
Next Decision
```

The project does not assume that a more complex model must outperform a simpler model.

Instead, each additional modeling component should demonstrate measurable predictive or operational value.

---

# 17. Research Hypotheses

## H1 — Temporal Information

Historical network states provide useful information for forecasting future attack activity beyond the current state alone.

## H2 — Structural Information

Network topology can provide additional predictive information beyond vector-based traffic representations.

## H3 — Learned Dynamics

A learned state-transition model can capture useful network dynamics for future-state forecasting.

## H4 — Forecast Horizon

Forecasting performance degrades as the prediction horizon increases.

Planned horizons include:

```text
+30s
+60s
+90s
+120s
+180s
+300s
```

## H5 — Early Warning

A useful forecasting system can provide measurable lead time before future attack activity.

## H6 — Generalization

Forecasting performance changes under distribution shift and across different attack scenarios.

## H7 — Calibration

Reliable attack probabilities require evaluation beyond classification accuracy, including calibration metrics.

---

# 18. Future Research Extensions

## Phase 1 — Packet-Level Telemetry

Introduce packet-derived features alongside flow-level features.

```text
Flow Features
      +
Packet Features
      ↓
Unified Network State
```

---

## Phase 2 — Structural Representation

Represent network entities and interactions as graphs.

```text
Hosts
  |
Connections
  |
Communication Graph
```

This enables comparison between:

```text
Vector representation
        vs
Graph representation
```

---

## Phase 3 — Temporal Graph Modeling

Investigate temporal graph models for evolving network structure.

Potential comparison:

```text
LSTM / GRU
     vs
Temporal GNN
```

---

## Phase 4 — Latent World Model

The longer-term architecture introduces a learned latent network state:

```text
S(t)
 |
 v
Encoder
 |
 v
z(t)
 |
 v
Transition Model
 |
 +----> z(t+1)
 |
 +----> z(t+2)
 |
 +----> ...
 |
 +----> z(t+K)
 |
 v
Decoder / Prediction Heads
```

The world model would be evaluated using:

* Future-state prediction
* Attack-risk prediction
* Attack-intensity prediction
* State reconstruction
* Rollout degradation
* Calibration
* Generalization

---

## Phase 5 — Multi-Horizon Forecasting

Evaluate how forecasting quality changes with increasing prediction horizon.

```text
Current
   |
   +---- +30s
   +---- +60s
   +---- +90s
   +---- +120s
   +---- +180s
   +---- +300s
```

---

## Phase 6 — Attack Progression

Where dataset labels support it, investigate attack-stage progression and mapping to appropriate security frameworks.

The system should not infer unsupported attack stages simply to produce a more complex visualization.

---

## Phase 7 — Research Validation

Planned comparisons include:

```text
Static vs Temporal
Persistence vs ML
Current State vs History
Vector vs Graph
Flow vs Flow + Packet
LSTM vs GRU
LSTM vs Temporal GNN
Temporal Model vs World Model
One-step vs Multi-step
Teacher Forcing vs Free Running
```

Additional evaluation will include:

* Ablation studies
* Error analysis
* Calibration
* Distribution shift
* Attack-family generalization where supported

---

# 19. Long-Term Architecture

The long-term SentinelGraph architecture is:

```text
                 Flow + Packet Telemetry
                           |
                           v
                    Network State S(t)
                           |
              +------------+------------+
              |                         |
              v                         v
       Vector Encoder            Graph Encoder
              |                         |
              +------------+------------+
                           |
                           v
                    Latent State z(t)
                           |
                           v
                  World Model Dynamics
                           |
              +------------+------------+
              |            |            |
              v            v            v
           z(t+1)       z(t+2)       z(t+K)
              |            |            |
              +------------+------------+
                           |
                           v
                 Future State Prediction
                           |
            +--------------+--------------+
            |              |              |
            v              v              v
       Attack Risk     Intensity     Progression
            |              |              |
            +--------------+--------------+
                           |
                           v
                Explainability / Uncertainty
                           |
                           v
                     Early Warning
                           |
                           v
                       Dashboard
```

---

# 20. Development Roadmap

The project follows a staged research and development roadmap.

```text
Level 1 — Data
        ↓
Level 2 — Forecasting
        ↓
Level 3 — Temporal Dynamics
        ↓
Level 4 — Structure
        ↓
Level 5 — World Model
        ↓
Level 6 — Rollout
        ↓
Level 7 — Operational Value
        ↓
Level 8 — Research Validation
        ↓
Level 9 — Productization
```

### Current Position

```text
Level 1 — Data                    COMPLETE
Level 2 — Forecasting             MOSTLY COMPLETE
Level 3 — Temporal Dynamics       NEXT
Level 4 — Structure               FUTURE
Level 5 — World Model             FUTURE
Level 6 — Rollout                 FUTURE
Level 7 — Operational Value       MVP STAGE
Level 8 — Research Validation     FUTURE
Level 9 — Productization          FUTURE
```

---

# 21. Installation

## Clone Repository

```bash
git clone <repository-url>
cd sentinel-graph
```

## Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 22. Dataset Setup

Create the dataset directory.

### macOS / Linux

```bash
mkdir -p data/raw/CIC-IDS2018
```

### Windows PowerShell

```powershell
mkdir data/raw/CIC-IDS2018
```

Download CIC-IDS2018 separately and place the required CSV files inside:

```text
data/raw/CIC-IDS2018/
```

Verify the files:

### macOS / Linux

```bash
ls data/raw/CIC-IDS2018
```

### Windows PowerShell

```powershell
dir data/raw/CIC-IDS2018
```

---

# 23. Running the Data Pipeline

Run the stages sequentially.

## Step 1 — Data Cleaning

```bash
python src/cleaning/clean_dataset.py
```

Verify:

```bash
python src/cleaning/check_clean.py
```

---

## Step 2 — Feature Engineering

```bash
python src/features/build_features.py
```

Verify:

```bash
python src/features/check_features.py
```

---

## Step 3 — Generate Temporal Network States

```bash
python src/windowing/create_states.py
```

Verify:

```bash
python src/windowing/check_states.py
```

---

## Step 4 — Generate Forecasting Sequences

```bash
python src/dataset/build_sequences.py
```

---

## Step 5 — Create Dataset Splits

```bash
python src/dataset/create_splits.py
```

---

# 24. Running Experiments

## Classical Baselines

```bash
python src/experiments/baselines.py
```

This evaluates:

* Majority
* Persistence
* Logistic Regression
* XGBoost

---

## LSTM Temporal Baseline

```bash
python src/experiments/lstm_baseline.py
```

---

# 25. Project Structure

```text
sentinel-graph/
│
├── data/
│   ├── raw/
│   │   └── CIC-IDS2018/
│   │
│   ├── interim/
│   ├── features/
│   ├── states/
│   └── sequences/
│
├── src/
│   │
│   ├── ingestion/
│   │
│   ├── cleaning/
│   │   ├── clean_dataset.py
│   │   └── check_clean.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── check_features.py
│   │
│   ├── windowing/
│   │   ├── create_states.py
│   │   └── check_states.py
│   │
│   ├── dataset/
│   │   ├── build_sequences.py
│   │   └── create_splits.py
│   │
│   └── experiments/
│       ├── preprocessing.py
│       ├── baselines.py
│       └── lstm_baseline.py
│
├── notebooks/
├── reports/
│
├── requirements.txt
├── requirements-lock.txt
├── README.md
└── .gitignore
```

---

# 26. Development Workflow

## Branch Naming

Use:

```text
feature/<name>
bugfix/<name>
experiment/<name>
```

Examples:

```text
feature/mvp-inference
feature/explainability
experiment/current-vs-history
experiment/temporal-gnn
```

## Commit Convention

Use conventional commits:

```text
feat:
fix:
docs:
refactor:
test:
chore:
```

Examples:

```text
feat: add temporal state generation
feat: add forecasting baselines
feat: add LSTM temporal baseline
experiment: compare current state and temporal history
docs: update project README
```

---

# 27. Research Development Principle

SentinelGraph follows:

```text
Question
    ↓
Hypothesis
    ↓
Experiment
    ↓
Result
    ↓
Interpretation
    ↓
Next Decision
```

The goal is not to add increasingly complex technologies simply because they are available.

Each major component should answer a research or operational question.

A more complex model is justified only when it provides measurable value in areas such as:

* Forecasting quality
* Longer-horizon prediction
* Early-warning lead time
* Calibration
* Generalization
* Structural understanding
* Explainability
* Operational usefulness

---

# 28. Project Scope

The semester MVP focuses on establishing a complete temporal forecasting system.

The longer-term research direction extends the MVP toward a network world model capable of learning evolving network dynamics and forecasting future states.

The project deliberately avoids adding infrastructure or technologies solely for complexity or resume keywords.

Examples include:

* Kubernetes
* Kafka
* Spark
* Unnecessary microservices
* LLM agents
* Blockchain
* Cloud infrastructure without a research or deployment requirement

The emphasis remains on:

```text
Data
  ↓
Network State
  ↓
Temporal Dynamics
  ↓
Forecasting
  ↓
Interpretability
  ↓
Validation
```

---

# 29. License

Add the appropriate project license here before public release.

---

# 30. Project Status

**Current Stage:** Temporal forecasting baseline development

**Semester Objective:** Complete the SentinelGraph MVP

**Long-Term Objective:** Investigate temporal, structural, and latent state-transition modeling for reliable multi-horizon network attack forecasting.