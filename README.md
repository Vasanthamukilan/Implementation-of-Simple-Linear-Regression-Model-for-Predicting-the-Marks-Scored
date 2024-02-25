# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for the marks by using the representation of the graph.
6. Hence we obtained the linear regression for the given dataset.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Vasanthamukilan M
RegisterNumber:212222230167
*/
```
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("/content/score_updated.csv")
df.head(10)
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
x = df.iloc[:,0:1]
y = df.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
x_train
y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(X_train,lr.predict(X_train),color='red')
lr.coef_
lr.intercept_
```
## Output:
### df.head()
![Screenshot 2024-02-25 205647](https://github.com/Vasanthamukilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559694/73abbc99-a88c-45f6-8ee5-75a212cae3a0)

### df.tail()
![Screenshot 2024-02-25 210329](https://github.com/Vasanthamukilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559694/bfa6d3cf-96e5-4f84-a538-3a315c65f08d)

### GRAPH OF PLOTTED DATA
![Screenshot 2024-02-25 210355](https://github.com/Vasanthamukilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559694/a37a11a9-0af3-4236-bcfa-cfcffdcf9621)

### TRAINED DATA
![Screenshot 2024-02-25 210408](https://github.com/Vasanthamukilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559694/03d99fd1-9ea1-4c2c-a153-4dc10cb0bd89)

### PERFORMING LINEAR REGRESSION
![Screenshot 2024-02-25 210419](https://github.com/Vasanthamukilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559694/d34a719d-8515-45e9-93b6-4c5532e02b30)

### PREDICTING LINE OF REGRESSION
![Screenshot 2024-02-25 210452](https://github.com/Vasanthamukilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559694/87e37d91-e436-447b-82e9-a734ae4289bb)

### COEFFICIENT AND INTERCEPT VALUES
![Screenshot 2024-02-25 210502](https://github.com/Vasanthamukilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559694/cde7d44c-94d6-48ca-a89b-3b0b557d1239)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
