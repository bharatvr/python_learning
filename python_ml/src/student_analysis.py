
"""
Linear Regression model predict student grade based on training data.
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

data = pd.read_csv("student-mat.csv", sep=";")


data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())

predict = "G3"

# x axis having data. We drop G3 as we don't want to train model for this grade (will use for prediction)
x = np.array(data.drop([predict], 1))

# y axis having data label
y = np.array(data[predict])

# split 10 % of data into test to predict result (the data which model never see before)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_test, y_test)

# check accuracy of model
acc = linear.score(x_test, y_test)
print("Model Accuracy : ", acc)

# Predict result based on text data
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    # First Col is model prediction Grade, Second student info and Third is real Grade
    print(predictions[x], x_test[x], y_test[x])

