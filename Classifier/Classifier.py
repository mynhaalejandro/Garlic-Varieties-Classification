import numpy as np
import pandas as pd
import graphviz
import pydotplus
import sklearn as sk
import cv2
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("color_features.csv")
# X = dataset[['mean_r','mean_g','mean_b','std_r','std_g','std_b','mean_h','mean_s','mean_v','std_h','std_s','std_v']]
X = dataset[['mean_h','mean_v']]
Y = dataset[['varieties']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 25)
Y_train.describe()

Y_test.describe()

from sklearn.tree import DecisionTreeClassifier

# Create the classifier
decision_tree_classifier = DecisionTreeClassifier(random_state = 0)

# Train the classifier on the training set
decision_tree_classifier.fit(X_train, Y_train)

# Evaluate the classifier on the testing set using classification accuracy
decision_tree_classifier.score(X_test, Y_test)

print("Accuracy on training set: {:.0f}%".format((decision_tree_classifier.score(X_train, Y_train))*100))
print("Accuracy on test set: {:.0f}%".format((decision_tree_classifier.score(X_test, Y_test))*100))

from sklearn import tree

feature_names = list(dataset)[0:-1]
class_names = ['batanes', 'ilocos_pink', 'ilocos_white', 'mexican', 'mmsu_gem', 'tan_bolters', 'vfta']
# print feature_names
# print class_names[0]

# dot_file = tree.export_graphviz(decision_tree_classifier, out_file="tree2.dot", 
#                                 feature_names = feature_names,
#                                 class_names = class_names)

# import graphviz
# with open("tree_a1.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)

# Simplified Tree

# decision_tree_pruned = DecisionTreeClassifier(random_state = 0, max_depth = 2)

# decision_tree_pruned.fit(X_train, Y_train)
# decision_tree_pruned.score(X_test, Y_test)

# pre_pruned_dot_file = tree.export_graphviz(decision_tree_pruned, out_file='tree_pruned1.dot', 
#                                 feature_names = feature_names,
#                                 class_names = class_names)
# with open("tree_pruned.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)