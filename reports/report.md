# Phishing detection - evaluation report

Positive class = phishing (1). **Primary metric: Recall** (false negatives = missed phishing are the costliest error).

| Rank | Model | Recall | Precision | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| 1 | Gaussian Naive Bayes | **0.9300** | 0.9929 | 0.9604 | 298 | 2 | 21 | 279 |
| 2 | Logistic Regression | **0.9267** | 0.9823 | 0.9537 | 295 | 5 | 22 | 278 |
| 3 | Decision Tree | **0.9233** | 1.0000 | 0.9601 | 300 | 0 | 23 | 277 |

**Best by recall:** Gaussian Naive Bayes

## Interpretability

Logistic regression coefficients (standardised features):

```
kw_suspicious_total        2.783
urgency_score              2.469
all_links_https            1.367
num_http_links             1.196
has_mismatched_link        1.057
num_mismatched_links       0.874
body_length               -0.669
num_ip_urls                0.613
num_links                 -0.530
kw_suspended               0.523
kw_password               -0.500
kw_verify                 -0.173
has_attachment             0.077
sender_replyto_mismatch   -0.003
```

Decision tree rules:

```
|--- kw_suspicious_total <= 1.50
|   |--- urgency_score <= 2.75
|   |   |--- body_length <= 1124.50
|   |   |   |--- num_links <= 0.50
|   |   |   |   |--- class: 0
|   |   |   |--- num_links >  0.50
|   |   |   |   |--- num_links <= 3.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- num_links >  3.50
|   |   |   |   |   |--- class: 0
|   |   |--- body_length >  1124.50
|   |   |   |--- body_length <= 1307.00
|   |   |   |   |--- body_length <= 1287.00
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- body_length >  1287.00
|   |   |   |   |   |--- class: 0
|   |   |   |--- body_length >  1307.00
|   |   |   |   |--- class: 0
|   |--- urgency_score >  2.75
|   |   |--- class: 1
|--- kw_suspicious_total >  1.50
|   |--- class: 1

```