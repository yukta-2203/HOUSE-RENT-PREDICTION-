
# Predictive Modeling of Residential Rental Price Dynamics

## 1. Project Overview

This project presents a supervised machine learning regression approach to model and predict the monthly rental price (in Rupees) for residential properties. Utilizing a multi-feature dataset, the study employs rigorous data preprocessing and a comparative analysis of established regression algorithms to establish a robust predictive framework for the real estate rental market. The ultimate goal is to identify the most effective modeling technique for this domain and provide transparent insights into the factors driving rental valuation.

---

## 2. Methodology

The predictive pipeline followed standard machine learning best practices, encompassing four critical stages:

### A. Data Preprocessing and Feature Engineering
* **Outlier Management:** Outliers in key continuous variables (`Rent`, `Size`, `Bathroom`) were systematically identified and treated using the **Interquartile Range (IQR) method** to mitigate bias and enhance model stability.
* **Categorical Encoding:** Nominal categorical features (e.g., `City`, `Area Type`, `Furnishing Status`) were transformed using **One-Hot Encoding** to prepare them for consumption by the algorithms.
* **Feature Scaling:** Continuous features were normalized using **MinMaxScaler** to ensure that no single feature's magnitude dominated the model training process.

### B. Model Selection and Training
A diverse suite of regression models was trained and evaluated on the preprocessed training data:
1.  **Linear Regression (LR):** Establishing a performance baseline.
2.  **Decision Tree Regressor (DT):** Assessing a non-linear, tree-based baseline.
3.  **Random Forest Regressor (RF):** Utilizing bagging to reduce variance and improve generalization.
4.  **Gradient Boosting Regressor (GBR):** Employing boosting for sequential error correction, which historically excels in complex regression tasks.

---

## 3. Key Findings and Results

The models were evaluated on the held-out test set using two standard metrics: the **Coefficient of Determination ($R^2$)** and the **Mean Squared Error (MSE)**.

| Model | $R^2$ Score | Mean Squared Error (MSE) |
| :--- | :--- | :--- |
| **Gradient Boosting Regressor** | **0.845** | **59,987,041.45** |
| Random Forest Regressor | 0.830 | 66,220,950.84 |
| Decision Tree Regressor | 0.821 | 70,051,751.21 |
| Linear Regression | 0.721 | 108,124,195.02 |

### Conclusion

The **Gradient Boosting Regressor** achieved the highest $R^2$ score of **0.845**, demonstrating its superior ability to capture the complex, non-linear relationships within the rental data. This indicates the model explains approximately **84.5% of the variance** in the monthly rent price, confirming its suitability for high-stakes predictive deployment.

---

## 4. Reproducibility and Requirements

### Dependencies
The project relies on standard Python data science libraries. The primary dependencies are:
* `Python (3.x)`
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

### Setup and Execution
1.  Clone this repository.
2.  Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3.  Run the main analysis notebook: **`Untitled1 (2).ipynb`**
    * *Note:* The complete data preprocessing, model training, and evaluation steps are contained sequentially within this Jupyter notebook.

---

## 5. References

1.  **Machine Learning Fundamentals and Implementation:**
    * Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras & TensorFlow: Concepts, tools, and techniques to build intelligent systems*. O'Reilly Media.

2.  **Ensemble Methods (Gradient Boosting):**
    * Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *The Annals of Statistics*, *29*(5), 1189–1232. [Standard reference for the GBR algorithm]

3.  **Real Estate Price Prediction:**
    * Chau, K. W., & Poon, C. Y. (2003). A comparison of the performance of ANN and traditional econometric models in forecasting house price. *Journal of Applied Soft Computing*, *3*(2), 221–234. [General reference on ML in housing prediction]

4.  **Statistical Methods and Outlier Treatment:**
    * Tukey, J. W. (1977). *Exploratory data analysis*. Addison-Wesley. [Foundational work for the Box Plot and IQR method for outlier detection]

5.  **Software Implementation:**
    * Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*(Oct), 2825-2830. [Citation for the primary machine learning library used]
