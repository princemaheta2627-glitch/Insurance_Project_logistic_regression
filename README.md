# Logistic Regression — Predicting Life Insurance Purchase from Age

A simple, intuitive binary classification project that predicts whether a person will buy life insurance based on a single feature — their **age** — using `scikit-learn`'s Logistic Regression, and walks through the underlying sigmoid math by hand.

---

## 1. Project Objective

Build a logistic regression model to predict whether a person buys life insurance (**Yes/No**) using their age as the sole predictor, and demonstrate the model's internal mechanics by manually reproducing its predictions with the sigmoid function.

This is a **binary classification problem** — there are exactly two possible outcomes (bought insurance / did not).

## 2. Dataset

| Attribute | Detail |
|---|---|
| File | `insurance_data.csv` |
| Observations | 27 people |
| Feature (X) | `age` |
| Target (y) | `bought_insurance` (1 = Yes, 0 = No) |
| Class balance | 14 bought (51.9%) / 13 did not (48.1%) — well balanced |

Raw sample:

| age | bought_insurance |
|---|---|
| 22 | 0 |
| 25 | 0 |
| 47 | 1 |
| 52 | 0 |
| 46 | 1 |
| 56 | 1 |
| 18 | 0 |
| 60 | 1 |

## 3. Tools & Libraries

- **Python 3**
- `pandas` — data handling
- `matplotlib` — visualization
- `scikit-learn` — train/test split and `LogisticRegression`
- `math` — manual sigmoid calculation to verify the model's predictions by hand

## 4. Methodology

1. **Load the dataset** into a DataFrame.
2. **Explore visually** — a scatter plot of age vs. `bought_insurance` shows a clear pattern: younger people cluster at 0 (did not buy), older people cluster at 1 (bought), with a mixed zone in the middle.
3. **Train/test split** — 80% train / 20% test.
4. **Train the model** — `sklearn.linear_model.LogisticRegression` fit on `age` alone.
5. **Predict & evaluate** — generate class predictions and predicted probabilities on the test set, and score overall accuracy.
6. **Reproduce the model by hand** — extract the fitted coefficient (`m`) and intercept (`b`), plug them into the sigmoid function manually, and confirm the hand-calculated probabilities match `model.predict_proba()`.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    df[['age']], df.bought_insurance, train_size=0.8
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
```

## 5. Model Results

| Term | Value |
|---|---|
| Coefficient (m) | 0.1131 |
| Intercept (b) | -4.1531 |
| **Fitted equation** | `z = 0.1131 × age − 4.1531` |

**Model equation:** `P(buys insurance) = sigmoid(0.1131 × age − 4.1531)`

**Decision boundary:** solving for `P = 0.5` gives an age of approximately **36.7 years** — the point at which the model's predicted outcome flips from "No" to "Yes".

## 6. Manual Verification with the Sigmoid Function

The notebook reproduces the model's predictions by hand to build intuition for how logistic regression actually works under the hood:

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(age):
    z = 0.1131 * age - 4.1531
    return sigmoid(z)
```

| Age | Hand-calculated P(buys insurance) | Interpretation |
|---|---|---|
| 35 | ≈ 0.46 | Below 0.5 → predicted **No** |
| 43 | ≈ 0.72 | Above 0.5 → predicted **Yes** |

This confirms the model's internal logic matches `model.predict_proba()` exactly — a useful sanity check and a good demonstration of understanding the math behind the library call, not just the API.

## 7. Model Evaluation

On the 20% held-out test set (6 people):

| Age | Actual | Predicted | Predicted Probability (Yes) |
|---|---|---|---|
| 62 | Yes | Yes | 0.946 |
| 61 | Yes | Yes | 0.940 |
| 29 | No | No | 0.294 |
| 28 | No | No | 0.272 |
| 26 | No | No | 0.229 |
| 22 | No | No | 0.159 |

**Accuracy: 6 / 6 = 100%** on this test split.

> With only 27 total records and a 6-person test set, a perfect score is expected given how cleanly age separates the two classes here, rather than evidence the model would generalize this well on a larger, noisier population. This dataset is best treated as a clear, teaching-scale illustration of logistic regression mechanics rather than a production-ready model.

## 8. Visualization

![Logistic regression curve fitted to the insurance data, showing predicted probability of purchase rising with age, with a decision boundary near 37 years](insurance_logistic_curve.png)

The sigmoid curve shows the predicted probability of purchasing insurance rising smoothly from near 0 in the late teens to near 1 by the early 60s, with the steepest change — and greatest prediction uncertainty — right around the ~37-year decision boundary.

## 9. Key Insights

- **Age alone is a strong, intuitive predictor** of life insurance purchase in this dataset, consistent with real-world intuition that older individuals are more likely to plan for end-of-life financial products.
- The **decision boundary of ~37 years** cleanly separates the observed data, with only a small "gray zone" (roughly ages 25–47) where outcomes are mixed.
- Manually reproducing the model's predictions with the sigmoid function confirms a solid understanding of what logistic regression is doing mathematically, beyond simply calling `.fit()` and `.predict()`.

## 10. Limitations & Next Steps

- **Very small dataset (27 records)** — the perfect test accuracy should not be over-interpreted; results would need validation on a much larger sample.
- **Single feature** — real insurance purchase decisions depend on income, dependents, health status, and other factors; a multivariate model would likely be more realistic.
- **No cross-validation** — a single train/test split can be sensitive to which records land in the test set; k-fold cross-validation would give a more robust accuracy estimate.
- The notebook's own exercise section suggests extending this workflow to a related HR dataset (employee retention vs. salary/department) as further practice — a natural next project applying the same logistic regression skills.

## 11. How to Run

```bash
pip install pandas matplotlib scikit-learn
jupyter notebook Insurance_logistic_regression_Answer.ipynb
```

## 12. Repository Contents

| File | Description |
|---|---|
| `Insurance_logistic_regression_Answer.ipynb` | Main analysis notebook |
| `insurance_data.csv` | Source dataset |
| `insurance_logistic_curve.png` | Fitted sigmoid curve visualization |
| `README.md` | This report |

## 13. Conclusion

This project builds a clean, interpretable logistic regression model that predicts life insurance purchase from age alone, achieving 100% accuracy on a small held-out test set and identifying an intuitive ~37-year decision boundary. Beyond fitting the model, the project reproduces its predictions by hand using the sigmoid function — reinforcing the mathematical foundation of logistic regression rather than treating it as a black box, which is a valuable habit for building toward more complex classification problems.

---
**Tech stack:** Python · pandas · matplotlib · scikit-learn
