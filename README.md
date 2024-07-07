# 2024 TikTok TechJam Track9

This is a repo for 2024 TikTok TechJam Track9.

We have 3 tasks:
- Task 1: Classification (normal, offensive, hatespeech)
- Task 2: Classification (Race, Religion, Gender, Sexual Orientation, Miscellaneous, None)
- Task 3: Tagging for each token

## Usage

### Train

```bash
python3 preprocess.py
python3 main.py
```

### Evaluate

#### Performance and Bias

```bash
python3 evaluation.py
```

#### Explainability

```bash
python3 preprocess_for_explainability.py
python3 explainability.py
python3 explainability_get_result_json.py
git clone git@github.com:jayded/eraserbenchmark.git
# manually add `labels +=['normal']` after `eraserbenchmark/rationale_benchmark/metrics.py` line 285    
cd eraserbenchmark
PYTHONPATH=./:$PYTHONPATH python3 rationale_benchmark/metrics.py --split test --strict --data_dir ../explainability_data --results ../explainability_data/results.json --score_file ../explainability_data/score.json
cd ..
python3 print_explainability_result.py
```

## Result

### Performance

```text
===== Report for Task 1 =====
Accuracy Score of Task 1: 0.6871101871101871
F1 Score of Task 1: 0.6693286454954777
ROCAUC Score of Task 1: 0.8334828348428918
Confusion Matrix:
[[580 135  67]
 [162 251 135]
 [ 30  73 491]]
              precision    recall  f1-score   support

           0       0.75      0.74      0.75       782
           1       0.55      0.46      0.50       548
           2       0.71      0.83      0.76       594

    accuracy                           0.69      1924
   macro avg       0.67      0.68      0.67      1924
weighted avg       0.68      0.69      0.68      1924

===== Report for Task 2 =====
Accuracy Score of Task 2: 0.7115384615384616
F1 Score of Task 2: 0.690247099614914
ROCAUC Score of Task 2: 0.9104437788823788
Confusion Matrix:
[[340  28   7   3  24  63]
 [ 11 293   2   5   5   9]
 [ 10   3  68   6   3  25]
 [  3   4   0 141   1   9]
 [ 16  15   2   3  97  43]
 [ 52  48  39  39  77 430]]
              precision    recall  f1-score   support

           0       0.79      0.73      0.76       465
           1       0.75      0.90      0.82       325
           2       0.58      0.59      0.58       115
           3       0.72      0.89      0.79       158
           4       0.47      0.55      0.51       176
           5       0.74      0.63      0.68       685

    accuracy                           0.71      1924
   macro avg       0.67      0.72      0.69      1924
weighted avg       0.72      0.71      0.71      1924
```

### Bias

```text
subgroup_aucs [0.8069547325102882, 0.853749747321609, 0.7849326599326599, 0.8059440559440559, 0.79833984375, 0.7825852657928589]
bpsn_aucs [0.7222749886928992, 0.7638764083776871, 0.8877995642701525, 0.8334729626808836, 0.8666584483892177, 0.884571201054782]
bnsp_aucs [0.9063951301255087, 0.9133496885164563, 0.7511890427453342, 0.8320406445406445, 0.8035592643051771, 0.6898482469911041]
GMB-Subgroup-AUC: 0.803505819894705
GMB-BPSN-AUC: 0.8107981445919924
GMB-BNSP-AUC: 0.7917722180146786
```

### Explainability

```text
Plausibility
IOU F1 : 0.0678713759729247
Token F1 : 0.23093872878968755
AUPRC : 0.3683726271577383

Faithfulness
Comprehensiveness : -0.09838706204541285
Sufficiency -0.02606892689199519
```
