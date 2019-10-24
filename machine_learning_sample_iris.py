# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:44:14 2019

@author: naveen
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset=load_iris()
print("keys of iris_dataset:\n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193]+"\n...")
#the value of the key target_names is an array of strings, containing the species of flower
#that we want to predict
print("Target names:{}".format(iris_dataset['target_names']))
#the value of feature_name is a list of strings giving the description of each feature
print("Feature names:\n{}".format(iris_dataset['feature_names']))
#The data itself is contained in the target and data fields. data containing the numeric
# measurements of sepal length, sepal width, petal length and petal width in numpy array
print("Type of data:{}".format(type(iris_dataset['data'])))
#The rows in the data array corresponds to flowers, while the columns represent the four measurements
#that were taken for each flower
print("Shape of data:{}".format(iris_dataset['data'].shape))
#here is the values for the first five samples
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
#target array contains the species of each of the flowers that were measured, also
# as a numpy array
print("Type of target:\n{}".format(type(iris_dataset['target'])))
#target is a one dimensional array with one entry per flower
print("Shape of target:\n{}".format(iris_dataset['target'].shape))
#the species are encoded as integers from 0 to 2
print("Target:\n{}".format(iris_dataset['target']))
##splitting the data in to tranining data and test data, using function train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

