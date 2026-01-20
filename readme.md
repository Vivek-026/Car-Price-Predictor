# 🚗 Car Price Prediction App

A Machine Learning web application that predicts the selling price of used cars based on various features like brand, year, fuel type, transmission, and mileage.

## 📌 Project Overview
This project uses a **Linear Regression** model wrapped in a Scikit-Learn pipeline. The key to high accuracy in this project was identifying that car prices are **right-skewed** and applying a **Log Transformation** to the target variable, which significantly improved performance compared to complex models like Random Forest.

### 🚀 **Live Demo:** [Click here to use the App](https://car-price-predictor-anpjxcfqeyth6exwzsifer.streamlit.app/)

**Key Features:**
* **User Interface:** Interactive web app built with Streamlit.
* **Machine Learning:** Linear Regression with Log-Target transformation.
* **Preprocessing:** Automated handling of missing values, scaling, and one-hot encoding using `ColumnTransformer`.
* **Accuracy:** Achieved an **R2 Score of ~0.77** (beating Random Forest at 0.67).

## 📂 Project Structure
```bash
Car_ML_project/
├── data/
│   └── car_data.csv        # The dataset used for training
├── models/
│   └── model.pkl           # The saved trained pipeline
├── venv/                   # Virtual Environment (not included in repo)
├── app.py                  # Streamlit frontend application
├── train.py                # Script to train and save the model
├── README.md               # Project documentation
└── requirements.txt        # List of dependencies

