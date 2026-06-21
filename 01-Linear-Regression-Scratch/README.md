# 📉 Salary Prediction using Linear Regression

This project contains a pure, framework-free implementation of a simple Linear Regression model optimized via Gradient Descent. Built entirely using basic Python loops, math formulas, and fundamental logic, this project predicts worker salary based on their years of experience without relying on high-level libraries like Scikit-Learn for modeling.

---

## 🎯 Project Objective
The goal is to model the linear relationship between a feature variable X (Years of Experience) and a continuous target variable y (Salary) by manually calculating the optimal slope (m) and intercept (b) that minimizes prediction error.

The underlying mathematical model is:
y = (m * x) + b

---

## 🛠️ The Mathematical Engine: Gradient Descent
Instead of using a shortcut, this implementation iteratively updates parameters using partial derivatives of the Mean Squared Error (MSE) loss function to find the global minimum.

For each iteration (epoch) over a dataset of size n, the gradients are computed manually inside a loop using these partial derivatives:

* **Slope Gradient (m_gradient):** += -(2/n) * x * (y - m_now * x - b_now)
* **Intercept Gradient (b_gradient):** += -(2/n) * (y - m_now * x - b_now)

### Hyperparameters Configured:
* **Learning Rate (L):** 0.0001 (Controls the step size taken down the error gradient)
* **Epochs:** 10,000 (The number of complete mathematical passes through the dataset)

---

## 📊 Dataset & Architecture Overview

The dataset consists of 30 records tracking professional experience relative to compensation structure. 

### Data Snippet:
```text
    YearsExperience    Salary
0               1.2   39344.0
1               1.4   46206.0
2               1.6   37732.0
...
28             10.4  122392.0
29             10.6  121873.0
