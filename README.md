# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1: Data Collection and Preprocessing

2: Splitting Data and Applying Cross-Validation 

3: Model Training using Multiple Linear Regression

4: Model Evaluation and Final Prediction

## Program:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt

#1 Load and prepare data
data=pd.read_csv('CarPrice_Assignment (1).csv')

#Simple preprocessing
data=data.drop(['car_ID','CarName'],axis=1) #Removes unescessary columns
data=pd.get_dummies(data,drop_first=True)   #Handle categorical variables

#2.Split data
x=data.drop('price',axis=1)
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#3. Create and train model
model=LinearRegression()
model.fit(x_train,y_train)

#4.Evaluate with cross-validation (simple version)
print('Name: SHRIHARI M')
print('Reg. No: 212225230265')
print("\n=== Cross-Validation ===")
cv_scores=cross_val_score(model,x,y,cv=5)
print("Fold R2 scores",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")

#5.Test set evaluation
y_pred = model.predict(x_test)
print("\n=== Test Set Performance ===")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")

#6.Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```

## Output:
<img width="956" height="704" alt="image" src="https://github.com/user-attachments/assets/6efaed40-52dd-4278-82f6-9fbf00b2a1f8" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
