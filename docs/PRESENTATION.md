# Presentation guide (8 slides, ~8 min)
1. **Problem** - phishing detection; false negative > false positive, so Recall is primary.
2. **Data** - balanced, phishing=1; *synthetic, generated to mimic real patterns* (say this upfront).
3. **Features** - 14 features: URL, header, link, content, structure.
4. **Models** - Logistic Regression (scaled), Gaussian NB, Decision Tree (Gini, depth-limited).
5. **Method** - stratified 70/30 split, seed 42, no leakage (scaler inside pipeline).
6. **Results** - `reports/confusion_matrices.png` + ranked table (`showcase/index.html`).
7. **Insights** - recall gap is small; tree = fewest false alarms, NB/LR = best recall; show tree rules + LR coefficients from `reports/report.md`.
8. **Limitations / future work** - real corpora, text features, threshold tuning.

## Likely questions
- **Why recall, not accuracy?** Missed phishing is the costly error; accuracy hides error type.
- **Why scale for Logistic Regression?** `body_length` would dominate binary flags.
- **Why depth-limit the tree?** Prevents memorising noise (overfitting).
- **Is the data real?** No - synthetic; scores are optimistic; real data is next.
- **Where is the confusion-matrix code?** `src/evaluate.py` -> `plot_confusion_matrices()` -> `reports/confusion_matrices.png`.
