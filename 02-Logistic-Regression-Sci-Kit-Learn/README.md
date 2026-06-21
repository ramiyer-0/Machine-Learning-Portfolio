# 📱 Mobile Price Classification using Logistic Regression

This project implements a Supervised Machine Learning classification pipeline using Scikit-Learn to predict whether a mobile phone belongs to an "Expensive" or "Cheap" price bracket. Unlike simple regression tasks, this project leverages feature scaling and optimization techniques to solve a binary classification problem based on real-world technical hardware specifications.

---

## 🎯 Project Objective
The goal is to determine the classification boundary that separates high-tier mobile devices from low/medium-tier devices. The model evaluates 20 distinct hardware attributes (such as RAM, internal memory, battery power, and camera megapixels) to calculate the mathematical probability of a device belonging to the premium class.

The target variable was engineered into a binary format:
* **1 (True):** High / Very High Cost
* **0 (False):** Low / Medium Cost

---

## 🛠️ The Machine Learning Engine: Scikit-Learn
Because Logistic Regression relies on gradient descent to find the optimal decision boundary, feature scaling is applied to normalize data scales and ensure fast, stable model convergence.

### Core Data Pipeline:
* **Feature Scaling (StandardScaler):** Standardizes all technical features so they have a mean of 0 and a variance of 1. This prevents variables with large numeric ranges (like battery capacity in mAh) from dominating variables with small ranges (like clock speed in GHz).
* **Optimization:** Utilizes Scikit-Learn's optimized solver algorithms to fit the logistic sigmoid curve across the multi-dimensional feature space.

---

## 📊 Dataset & Processing Overview

The dataset utilizes 2,000 empirical records of mobile phone hardware attributes split systematically into training and testing datasets.

### Core Code Workflow:
1. **Data Ingestion:** Loaded the target mobile attributes dataset directly via a remote URL stream into a `pandas` DataFrame.
2. **Target Binary Splitting:** Evaluated the multi-class `price_range` attribute and applied a boolean mask (`> 1`) to cast the labels into strict binary classification categories (`is_expensive`).
3. **Data Splitting:** Divided the processed attributes matrix into 80% training data (to teach the model patterns) and 20% test data (to evaluate accuracy).
4. **Data Transformation:** Fit and executed a `StandardScaler` pipeline to ensure all input numeric metrics share a common distribution scale.
5. **Model Evaluation:** Executed the `.fit()` routine and generated final validation classifications using `.predict()`.

---

## 🚀 Classification Performance Results
After completing the preprocessing operations and feeding the transformed arrays into the Logistic Regression engine, the framework yielded highly stable metrics:

* **Classification Training Split:** 80% Train / 20% Test matrix split
* **Overall Model Accuracy:** ~95.50% (Indicating exceptional predictive performance on unseen evaluation data)

The resulting final notebook showcases a streamlined, industry-standard approach to data ingestion, cleaning, scaling, and deployment.
