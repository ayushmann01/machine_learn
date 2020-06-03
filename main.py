import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
"""
print(diabetes.keys())
dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
print(diabetes.DESCR)  shows details of the dataset
"""

# diabetes_x = diabetes.data[:, np.newaxis, 2]  # this takes only one feature
diabetes_x = diabetes.data

diabetes_x_train = diabetes_x[: -30]
diabetes_x_test = diabetes_x[-30:]


diabetes_y_train = diabetes.target[: -30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predict = model.predict(diabetes_x_test)
print("Meaned square error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predict))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plt.scatter(diabetes_x_test, diabetes_y_test)
# plt.plot(diabetes_x_test, diabetes_y_predict)
#
# plt.show()

# predictions with single feature
# Meaned square error is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698

