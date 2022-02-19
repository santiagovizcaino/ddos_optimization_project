# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:30:00 2022

@author: santy
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

features = pd.read_csv(r"C:\Users\santy\Desktop\dataset_003.csv", encoding = 'utf-8')

features.head()
print('The shape of our features is:', features.shape)

features['type'] = pd.factorize(features.type)[0]

labels = np.array(features['type'])

features= features.drop('type', axis = 1)
#features= features.drop('ack', axis = 1)

feature_list = list(features.columns)# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('length')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)# Calculate the absolute errors
errors = abs(predictions - test_labels)# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,precision_score,recall_score,f1_score

# Calculate mean absolute percentage error (MAPE)


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot# Pull out one tree from the forest
tree = rf.estimators_[5]# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot# Pull out one tree from the forest
tree = rf.estimators_[5]# Export the image to a dot file

export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1,filled=True)# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')# Write graph to a png file
graph.write_png('tree.png')