# SentinelGraph — Attack Onset Forecasting Experiment

**Status:** Exploratory evaluation  
**Target:** Predict whether an attack begins in the next 30-second state window.

## 1. Objective

Evaluate whether temporal network-flow features can identify an attack onset before the next state window.

The input contains 10 consecutive 30-second states (5 minutes). The target is the immediately following 30-second state.

An onset is defined as:

`y_onset = (y_attack == 1) AND (last_attack == 0)`

Only sequences whose last observed state is benign are included in the risk set. This avoids evaluating onset prediction on sequences where an onset is impossible by definition.

## 2. Data

Eight CIC-IDS2018 traffic days were processed.

- Combined sequence count: 6,734
- Input shape: 10 time steps × 24 features
- Attack-onset positives across all sequences: 199

The exploratory train/test arrangement was:

| Partition | Days | Eligible sequences | Onsets |
|---|---|---:|---:|
| Train | Feb 14, 15, 16, 21, 22 | 2,688 | 110 |
| Test | Feb 23 | 900 | 86 |

Training onset prevalence was 4.09%; test prevalence was 9.56%.

**Important split limitation:** February 23 had previously been used as a validation day in earlier expanded experiments. It is therefore not an untouched independent holdout. These results are exploratory and must not be presented as final generalization performance.

February 28 and March 2 contained only two and one onset examples respectively, making them unsuitable for meaningful onset evaluation.

## 3. Method

Three representations of each five-minute input sequence were compared:

1. **Last window:** Features from the final observed state.
2. **Five-minute mean:** Mean feature values across the ten observed states.
3. **First-to-last delta:** Difference between the final and first observed states.

Two classifiers were evaluated:

- Logistic Regression with standardisation and balanced class weights.
- XGBoost classifier.

The reported classification metrics use a fixed threshold of 0.5. PR-AUC and ROC-AUC are also reported. The no-skill PR-AUC reference is the test prevalence, 0.0956.

## 4. February 23 exploratory test results

| Representation | Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Last window | Logistic Regression | 0.0705 | 0.5930 | 0.1261 | 0.0652 | 0.3004 |
| Last window | XGBoost | 0 | 0 | 0 | 0.0699 | 0.3361 |
| Five-minute mean | Logistic Regression | 0.0721 | 0.5349 | 0.1271 | 0.0746 | 0.3800 |
| Five-minute mean | XGBoost | 0.0294 | 0.0116 | 0.0167 | 0.0773 | 0.4042 |
| First-to-last delta | Logistic Regression | 0.1066 | 0.4535 | 0.1726 | 0.1062 | 0.5435 |
| First-to-last delta | XGBoost | 0.3333 | 0.0116 | 0.0225 | 0.1201 | 0.5240 |

The first-to-last delta representation performed best among the tested representations on PR-AUC for both models. Logistic Regression with this representation achieved the highest F1 score (0.1726).

However, this model identified only 39 of 86 onsets and produced 327 false positives. XGBoost identified only one onset at the selected threshold.

The results do not demonstrate reliable onset forecasting.

## 5. Leave-one-day-out analysis

Logistic Regression using first-to-last delta features was also evaluated by holding out each training day in turn.

| Held-out day | Eligible samples | Onsets | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Feb 14 | 686 | 2 | 0.0068 | 1.0000 | 0.0135 | 0.0064 | 0.6615 |
| Feb 15 | 1,006 | 5 | 0.0089 | 0.8000 | 0.0176 | 0.0090 | 0.6683 |
| Feb 16 | 0 | 0 | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| Feb 21 | 48 | 0 | 0 | 0 | 0 | Not meaningful | Undefined |
| Feb 22 | 948 | 103 | 0.1570 | 0.3398 | 0.2147 | 0.1408 | 0.5622 |

February 16 had no eligible sequences. February 21 had no positive onset examples, so its PR-AUC and ROC-AUC are not meaningful evaluation measures.

The training data is highly concentrated: February 22 contributes 103 of the 110 training onsets. When February 22 is held out, only seven positive examples remain for training. This makes the day-level results unstable and exposes a serious class scarcity problem.

## 6. Interpretation

The current experiments show:

- Onset prediction is substantially harder than predicting whether the next state contains any attack.
- Performance varies considerably across traffic days.
- The first-to-last feature delta is the most promising of the tested representations, but its performance remains weak.
- High recall at the selected threshold comes with many false positives.
- The very small number of positive onset examples on several days prevents reliable day-level evaluation.
- The February 23 result cannot be treated as an independent final test because that day was previously used during validation.

## 7. Data and evaluation limitations

- The dataset currently covers eight traffic days, not the full CIC-IDS2018 collection.
- Several days have very few usable onset examples.
- State windows containing missing values were dropped rather than imputed.
- Sequences are segmented at timestamp discontinuities; they must not cross gaps.
- The 30-second binary target represents whether any attack-labelled flow occurs in the next state. It does not represent attack severity.
- Overlapping temporal sequences and day-level distribution shifts require careful treatment in future evaluation.
- The current results are exploratory and do not establish generalisation to unseen traffic days.

## 8. Next steps

1. Acquire additional CIC-IDS2018 days to obtain a more representative set of attack onsets.
2. Redesign the split around genuinely untouched chronological test days.
3. Report per-day metrics and confidence intervals where sample counts permit.
4. Investigate threshold selection using validation data only.
5. Compare against appropriate onset-specific baselines.
6. Reassess whether more complex temporal or graph models are justified after the evaluation design is strengthened.

## 9. Current conclusion

The onset-forecasting pipeline and initial comparative experiments are implemented. The current models do not yet provide reliable attack-onset predictions.

The completed work is a meaningful engineering and experimental milestone, but it is not evidence that the forecasting objective has been solved. Further data and a stronger evaluation design are required before claiming successful generalisation.
