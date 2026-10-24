# Phishing Email Detection (Logistic Regression, Gaussian NB, Decision Tree)

Classifies emails as phishing (1, positive class) or legitimate (0) from numeric
features engineered from raw email text. **Recall is the primary comparison metric**
because a missed phishing email (false negative) costs more than a false alarm.

## Layout
```
phishing_detection/
├── data/
│   ├── raw/emails.csv          # input: columns `email_text` (raw RFC-822), `label`
│   └── processed/              # train.csv / test.csv (features + label)
├── models/                     # trained *.joblib pipelines
├── reports/                    # metrics.csv/json, report.md, confusion_matrices.png
├── src/
│   ├── config.py               # paths, seed, split size, hyper-parameters, grids
│   ├── feature_extraction.py   # email parsing + all numeric features
│   ├── data_processing.py      # load/validate, features, stratified 70/30 split
│   ├── models.py               # LR (scaled), GaussianNB, Decision Tree (gini, depth-limited)
│   ├── train.py                # train (optionally CV-tune on recall) and save
│   ├── evaluate.py             # precision/recall/F1/confusion matrices, ranking, plots
│   ├── pipeline.py             # one-command end-to-end run
│   └── generate_synthetic_data.py  # balanced demo dataset with realistic noise
├── tests/test_features.py
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
**Quick demo (synthetic data, everything in one command):**
```bash
python -m src.pipeline --synthetic 2000
```

**With your own data:** place a CSV at `data/raw/emails.csv` with columns
`email_text` (full raw message incl. headers; HTML and multipart supported) and
`label` (1 = phishing, 0 = legitimate), balanced across classes, then:
```bash
python -m src.pipeline                 # default hyper-parameters
python -m src.pipeline --tune          # 5-fold stratified CV grid search, scored on recall
```

**Step by step:**
```bash
python -m src.generate_synthetic_data --n 2000   # optional
python -m src.data_processing                    # features + 70/30 stratified split (seed 42)
python -m src.train [--tune]
python -m src.evaluate
pytest -q                                        # unit tests for the feature extractors
```

## Features (14)
| Group | Features |
|---|---|
| URL | `num_links`, `num_ip_urls` (raw-IP hosts), `num_http_links` (non-HTTPS), `all_links_https` |
| Header | `sender_replyto_mismatch` (registrable From domain ≠ Reply-To domain; absent Reply-To = 0) |
| Link | `num_mismatched_links`, `has_mismatched_link` (anchor text shows one domain, href goes to another) |
| Content | `kw_verify`, `kw_suspended`, `kw_password` (stem-matched), `kw_suspicious_total`, `urgency_score` (weighted urgency phrases + `!` + ALL-CAPS words) |
| Structure | `body_length`, `has_attachment` |

**Showcase page:** `python -m src.pipeline` also builds `showcase/index.html` (open in a browser). Talking points: `docs/PRESENTATION.md`. Windows one-click: `run_demo.bat`.

## Design notes
- **Reproducibility:** `RANDOM_SEED=42` for the split and all models; split is stratified so both sets stay balanced.
- **Leakage-safe:** scaling lives inside the sklearn `Pipeline`, fitted on training data only; tuning uses CV on the training set only.
- **Overfitting control:** tree uses Gini with `max_depth`, `min_samples_split`, `min_samples_leaf`; tune via `--tune`.
- **Output:** `reports/report.md` ranks models by recall and includes LR coefficients and the tree's text rules for explainability.
- **Registrable domains** use a lightweight heuristic; swap in `tldextract` for production-grade accuracy.

## Caveats
Synthetic data is only for verifying the pipeline - its scores say nothing about real-world performance.
Evaluate on a real corpus (e.g. Nazario / Enron-based phishing sets), keep the test set untouched during
tuning, and consider threshold tuning on `predict_proba` to push recall higher at the cost of precision.
