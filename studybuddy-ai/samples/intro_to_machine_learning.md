# Introduction to Machine Learning (sample notes)

## What is Machine Learning?
Machine learning (ML) lets computers learn patterns from data instead of being explicitly programmed. A model is trained on examples and then makes predictions on new, unseen data. The two main families are supervised learning and unsupervised learning.

## Supervised Learning
In supervised learning every training example has a label. The model learns a mapping from inputs (features) to outputs (labels). Regression predicts a continuous number, for example house price. Classification predicts a category, for example spam or not spam. Common algorithms include linear regression, logistic regression, decision trees and k-nearest neighbours.

## Unsupervised Learning
Unsupervised learning works with unlabelled data and looks for structure. Clustering groups similar items together; k-means is the classic algorithm and needs the number of clusters k in advance. Dimensionality reduction, such as PCA, compresses many features into fewer while keeping most of the variance.

## Overfitting and Underfitting
Overfitting happens when a model memorises the training data, including noise, and performs badly on new data. Underfitting happens when the model is too simple to capture the pattern. Signs of overfitting: very low training error but high validation error. Remedies include more data, regularisation, simpler models, early stopping and cross-validation.

## Evaluating Models
Data is split into training, validation and test sets. Accuracy is the fraction of correct predictions but is misleading on imbalanced data. Precision = TP / (TP + FP). Recall = TP / (TP + FN). The F1 score is the harmonic mean of precision and recall. For regression, mean squared error (MSE) is the average of squared differences between predictions and true values.

## Gradient Descent
Gradient descent minimises a loss function by repeatedly moving the parameters in the direction opposite to the gradient. The update rule is w = w - alpha * dL/dw, where alpha is the learning rate. If alpha is too large the algorithm can overshoot and diverge; if it is too small, training is very slow. Stochastic gradient descent uses a small random batch of data for each step.
