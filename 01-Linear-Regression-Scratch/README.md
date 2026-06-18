# Machine Learning From Scratch: Linear Regression

An implementation of a simple Linear Regression model built entirely from scratch using pure Python loops and formulas (no Scikit-Learn). The model uses Gradient Descent to optimize its parameters and is evaluated on a real-world Salary dataset.

## 🚀 Project Overview
The objective of this project is to demonstrate the underlying mechanics of supervised learning and gradient descent. By manually coding the derivatives and update loops, this project bridges the gap between theoretical calculus and working software.

### Key Features
* **Zero Dependencies:** Optimized parameters are calculated without ML libraries.
* **Manual Gradient Descent:** Custom implementation of slope calculation and parameter updates.
* **Data Visualization:** Built-in tracking of the best-fit regression line over raw data points.

---

## 📊 Dataset
The model is trained on a **Annual Salary vs. Years Of Experience** dataset sourced from Kaggle. 
* **Features ($X$):** YearsExperience
* **Target ($y$):** Salary

---

## 🧠 Math & Algorithm Breakdown

The model fits a straight line to the data using the equation:
$$\hat{y} = mx + b$$

### Gradient Descent Optimization
To minimize the Mean Squared Error (MSE), the parameters $m$ (slope) and $b$ (intercept) are iteratively updated using partial derivatives:

$$\frac{\partial J}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$
$$\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

Every epoch, the weights shift against the gradient scaled by the learning rate ($L$):
* $m = m - L \cdot \frac{\partial J}{\partial m}$
* $b = b - L \cdot \frac{\partial J}{\partial b}$

---

## 💻 How to Run the Code

1. Clone this repository:
   ```bash
   git clone [https://github.com/ramiyer-0/Machine-Learning-Portfolio.git](https://github.com/ramiyer-0/Machine-Learning-Portfolio.git)
