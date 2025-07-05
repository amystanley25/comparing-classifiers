
# Term Deposit Subscription Prediction – Classifier Comparison

This project uses data from a Portuguese banking institution to build and evaluate machine learning models that predict whether a customer will subscribe to a term deposit after a marketing call. The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and includes customer demographics, campaign details, and economic context.

---

## Objective

The **business goal** is to predict which clients are most likely to subscribe to a term deposit so that marketing efforts can be focused more effectively—saving time, costs, and improving customer targeting.

---

## Dataset

- File used: `bank-additional-full.csv`
- Format: CSV (semicolon-separated)
- Rows: ~41,000
- Features: 20 input variables (categorical & numeric), 1 binary output variable (`y`: "yes" or "no")
- Campaigns: 17 marketing campaigns between May 2008 and November 2010

---

## Tasks Completed

### Data Preprocessing
- Removed `duration` column (not usable for real-time predictions)
- One-hot encoded categorical variables
- Binary-encoded target column (`yes` → 1, `no` → 0)
- Checked for missing values and data types

### Exploratory Data Analysis (EDA)
- Feature understanding using UCI and academic paper documentation
- Analyzed class imbalance and discussed ethical implications (e.g., gender-based targeting)

### Model Development

**Models Used:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Support Vector Machine (SVM)

**Baseline Model:**  
A dummy classifier that predicts the majority class ("no") was used to establish a baseline accuracy.

### Evaluation Metrics

| Metric              | Why It Matters                                       |
|---------------------|------------------------------------------------------|
| Accuracy            | Quick snapshot but misleading with imbalance         |
| Precision & Recall  | More balanced view—especially recall for this case   |
| F1 Score            | Harmonic mean of precision and recall                |
| ROC-AUC             | Assesses model's ability to discriminate classes     |

---

##  Model Comparison Results (Default Hyperparameters)

| Model              | Train Time (s) | Train Accuracy | Test Accuracy |
|-------------------|----------------|----------------|---------------|
| Logistic Regression | x.xx          | xx.xx%         | xx.xx%        |
| KNN                 | x.xx          | xx.xx%         | xx.xx%        |
| Decision Tree       | x.xx          | xx.xx%         | xx.xx%        |
| SVM                 | x.xx          | xx.xx%         | xx.xx%        |

*(Replace with actual values)*

---

## Improvements Suggested

- **Hyperparameter Tuning:**  
  Used `GridSearchCV` to tune parameters like:
  - `KNN`: `n_neighbors`
  - `Decision Tree`: `max_depth`, `min_samples_split`
  - `SVM`: `C`, `kernel`, `gamma`

- **Metric Adjustment:**  
  Accuracy is misleading due to class imbalance. Future versions of this project should prioritize:
  - **Recall** (to avoid missing potential subscribers)
  - **F1-score** (balance between recall and precision)

- **Ethical Considerations:**  
  While adding features like gender may improve accuracy, it could introduce discrimination. Fairness should be a key consideration in production-level models.

---

## Next Steps

- Perform **feature selection** to reduce dimensionality
- Try **ensemble methods** (e.g., Random Forest, XGBoost)
- Implement **SMOTE** or similar techniques to address class imbalance
- Visualize **confusion matrices** and **ROC curves**


## References

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- [Moro et al. (2014)](https://www.sciencedirect.com/science/article/pii/S0167923614000613): "A Data-Driven Approach to Predict the Success of Bank Telemarketing"

---
