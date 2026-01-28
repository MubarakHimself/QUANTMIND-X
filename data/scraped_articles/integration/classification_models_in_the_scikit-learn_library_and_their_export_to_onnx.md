---
title: Classification models in the Scikit-Learn library and their export to ONNX
url: https://www.mql5.com/en/articles/13451
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:04:38.486772
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/13451&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083317953803327808)

MetaTrader 5 / Examples


The development of technology has led to the emergence of a fundamentally new approach to building data processing algorithms. Previously, for solving each specific task, a clear formalization and development of corresponding algorithms were required.

In machine learning, the computer learns to find the best ways to process data on its own. Machine learning models can successfully solve classification tasks (where there is a fixed set of classes and the goal is to find the probabilities of a given set of features belonging to each class) and regression tasks (where the goal is to estimate a numerical value of the target variable based on a given set of features). More complex data processing models can be built based on these fundamental components.

The Scikit-learn library provides a multitude of tools for both classification and regression. The choice of specific methods and models depends on the characteristics of the data since different methods can have varying effectiveness and provide different results depending on the task.

In the press release ["ONNX Runtime is now open source"](https://www.mql5.com/go?link=https://azure.microsoft.com/de-de/blog/onnx-runtime-is-now-open-source/ "https://azure.microsoft.com/de-de/blog/onnx-runtime-is-now-open-source/"), it is claimed that ONNX Runtime also supports the ONNX-ML profile:

ONNX Runtime is the first publicly available inference engine with full support for ONNX 1.2 and higher including the ONNX-ML profile.

The [ONNX-ML](https://www.mql5.com/go?link=https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md "https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md") profile is a part of ONNX designed specifically for machine learning (ML) models. It is intended for describing and representing various types of ML models, such as classification, regression, clustering, and others, in a convenient format that can be used on various platforms and environments that support ONNX. The ONNX-ML profile simplifies the transmission, deployment, and execution of machine learning models, making them more accessible and portable.

In this article, we will explore the application of all classification models in the Scikit-learn package for solving the Fisher's Iris classification task. We will also attempt to convert these models into the ONNX format and use the resulting models in MQL5 programs.

Furthermore, we will compare the accuracy of the original models with their ONNX versions on the complete Iris dataset.

### Table of Contents

- [1\. Fisher's Irises](https://www.mql5.com/en/articles/13451#iris1)

- [2\. Models for Classification](https://www.mql5.com/en/articles/13451#models2)

[List of Scikit-learn Classifiers](https://www.mql5.com/en/articles/13451#models2_1)

[Different Output Representations of Models](https://www.mql5.com/en/articles/13451#models2_2)

[iris.mqh](https://www.mql5.com/en/articles/13451#models2_3)

- [2.1. SVC Classifier](https://www.mql5.com/en/articles/13451#model_2_1)

[2.1.1. Code for Creating the SVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_1_1)

[2.1.2. MQL5 Code for Working with the SVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_1_2)

[2.1.3. ONNX Representation of the SVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_1_3)

- [2.2. LinearSVC Classifier](https://www.mql5.com/en/articles/13451#model_2_2)

[2.2.1. Code for Creating the LinearSVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_2_1)

[2.2.2. MQL5 Code for Working with the LinearSVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_2_2)

[2.2.3. ONNX Representation of the LinearSVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_2_3)

- [2.3. NuSVC Classifier](https://www.mql5.com/en/articles/13451#model_2_3)

[2.3.1. Code for Creating the NuSVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_3_1)

[2.3.2. MQL5 Code for Working with the NuSVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_3_2)

[2.3.3. ONNX Representation of the NuSVC Classifier Model](https://www.mql5.com/en/articles/13451#model_2_3_3)

- [2.4. Radius Neighbors Classifier](https://www.mql5.com/en/articles/13451#model_2_4)

[2.4.1. Code for Creating the Radius Neighbors Classifier Model](https://www.mql5.com/en/articles/13451#model_2_4_1)

[2.4.2. MQL5 Code for Working with the Radius Neighbors Classifier Model](https://www.mql5.com/en/articles/13451#model_2_4_2)

[2.3.3. ONNX Representation of the Radius Neighbors Classifier Model](https://www.mql5.com/en/articles/13451#model_2_4_3)

- [2.5. Ridge Classifier](https://www.mql5.com/en/articles/13451#model_2_5)

[2.5.1. Code for Creating the Ridge Classifier Model](https://www.mql5.com/en/articles/13451#model_2_5_1)

[2.5.2. MQL5 Code for Working with the Ridge Classifier Model](https://www.mql5.com/en/articles/13451#model_2_5_2)

[2.5.3. ONNX Representation of the Ridge Classifier Model](https://www.mql5.com/en/articles/13451#model_2_5_3)

- [2.6. RidgeClassifierCV](https://www.mql5.com/en/articles/13451#model_2_6)

[2.6.1. Code for Creating the Ridge ClassifierCV Model](https://www.mql5.com/en/articles/13451#model_2_6_1)

[2.6.2. MQL5 Code for Working with the Ridge ClassifierCV Model](https://www.mql5.com/en/articles/13451#model_2_6_2)

[2.6.3. ONNX Representation of the Ridge ClassifierCV Model](https://www.mql5.com/en/articles/13451#model_2_6_3)

- [2.7. Random Forest Classifier](https://www.mql5.com/en/articles/13451#model_2_7)

[2.7.1. Code for Creating the Random Forest Classifier Model](https://www.mql5.com/en/articles/13451#model_2_7_1) [2.7.2. MQL5 Code for Working with the Random Forest Classifier Model](https://www.mql5.com/en/articles/13451#model_2_7_2) [2.7.3. ONNX Representation of the Random Forest Classifier Model](https://www.mql5.com/en/articles/13451#model_2_7_3)
- [2.8. Gradient Boosting Classifier](https://www.mql5.com/en/articles/13451#model_2_8)

[2.8.1. Code for Creating the Gradient Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_8_1)

[2.8.2. MQL5 Code for Working with the Gradient Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_8_2)

[2.8.3. ONNX Representation of the Gradient Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_8_3)

- [2.9. Adaptive Boosting Classifier](https://www.mql5.com/en/articles/13451#model_2_9)

[2.9.1. Code for Creating the Adaptive Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_9_1)

[2.9.2. MQL5 Code for Working with the Adaptive Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_9_2)

[2.9.3. ONNX Representation of the Adaptive Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_9_3)

- [2.10. Bootstrap Aggregating Classifier](https://www.mql5.com/en/articles/13451#model_2_10)

[2.10.1. Code for Creating the Bootstrap Aggregating Classifier Model](https://www.mql5.com/en/articles/13451#model_2_10_1)

[2.10.2. MQL5 Code for Working with the Bootstrap Aggregating Classifier Model](https://www.mql5.com/en/articles/13451#model_2_10_2)

[2.10.3. ONNX Representation of the Bootstrap Aggregating Classifier Model](https://www.mql5.com/en/articles/13451#model_2_10_3)

- [2.11. K-Nearest Neighbors (K-NN) Classifier](https://www.mql5.com/en/articles/13451#model_2_11)

[2.11.1. Code for Creating the K-Nearest Neighbors (K-NN) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_11_1)

[2.11.2. MQL5 Code for Working with the K-Nearest Neighbors (K-NN) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_11_2)

[2.11.3. ONNX Representation of the K-Nearest Neighbors (K-NN) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_11_3)

- [2.12. Decision Tree Classifier](https://www.mql5.com/en/articles/13451#model_2_12)

[2.12.1. Code for Creating the Decision Tree Classifier Model](https://www.mql5.com/en/articles/13451#model_2_12_1)

[2.12.2. MQL5 Code for Working with the Decision Tree Classifier Model](https://www.mql5.com/en/articles/13451#model_2_12_2)

[2.12.3. ONNX Representation of the Decision Tree Classifier Model](https://www.mql5.com/en/articles/13451#model_2_12_3)

- [2.13. Logistic Regression Classifier](https://www.mql5.com/en/articles/13451#model_2_13)

[2.13.1. Code for Creating the Logistic Regression Classifier Model](https://www.mql5.com/en/articles/13451#model_2_13_1)

[2.13.2. MQL5 Code for Working with the Logistic Regression Classifier Model](https://www.mql5.com/en/articles/13451#model_2_13_2)

[2.13.3. ONNX Representation of the Logistic Regression Classifier Model](https://www.mql5.com/en/articles/13451#model_2_13_3)

- [2.14. LogisticRegressionCV Classifier](https://www.mql5.com/en/articles/13451#model_2_14)

[2.14.1. Code for Creating the LogisticRegressionCV Classifier Model](https://www.mql5.com/en/articles/13451#model_2_14_1)

[2.14.2. MQL5 Code for Working with the LogisticRegressionCV Classifier Model](https://www.mql5.com/en/articles/13451#model_2_14_2)

[2.14.3. ONNX Representation of the LogisticRegressionCV Classifier Model](https://www.mql5.com/en/articles/13451#model_2_14_3)

- [2.15. Passive-Aggressive (PA) Classifier](https://www.mql5.com/en/articles/13451#model_2_15)

[2.15.1. Code for Creating the Passive-Aggressive (PA) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_15_1)

[2.15.2. MQL5 Code for Working with the Passive-Aggressive (PA) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_15_2)

[2.15.3. ONNX Representation of the Passive-Aggressive (PA) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_15_3)

- [2.16. Perceptron Classifier](https://www.mql5.com/en/articles/13451#model_2_16)

[2.16.1. Code for Creating the Perceptron Classifier Model](https://www.mql5.com/en/articles/13451#model_2_16_1)

[2.16.2. MQL5 Code for Working with the Perceptron Classifier Model](https://www.mql5.com/en/articles/13451#model_2_16_2)

[2.16.3. ONNX Representation of the Perceptron Classifier Model](https://www.mql5.com/en/articles/13451#model_2_16_3)

- [2.17. Stochastic Gradient Descent Classifier](https://www.mql5.com/en/articles/13451#model_2_17)

[2.17.1. Code for Creating the Stochastic Gradient Descent Classifier Model](https://www.mql5.com/en/articles/13451#model_2_17_1)

[2.17.2. MQL5 Code for Working with the Stochastic Gradient Descent Classifier Model](https://www.mql5.com/en/articles/13451#model_2_17_2)

[2.17.3. ONNX Representation of the Stochastic Gradient Descent Classifier Model](https://www.mql5.com/en/articles/13451#model_2_17_3)

- [2.18. Gaussian Naive Bayes (GNB) Classifier](https://www.mql5.com/en/articles/13451#model_2_18)

[2.18.1. Code for Creating the Gaussian Naive Bayes (GNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_18_1)

[2.18.2. MQL5 Code for Working with the Gaussian Naive Bayes (GNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_18_2)

[2.18.3. ONNX Representation of the Gaussian Naive Bayes (GNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_18_3)

- [2.19. Multinomial Naive Bayes (MNB) Classifier](https://www.mql5.com/en/articles/13451#model_2_19)

[2.19.1. Code for Creating the Multinomial Naive Bayes (MNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_19_1)

[2.19.2. MQL5 Code for Working with the Multinomial Naive Bayes (MNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_19_2)

[2.19.3. ONNX Representation of the Multinomial Naive Bayes (MNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_19_3)

- [2.20. Complement Naive Bayes (CNB) Classifier](https://www.mql5.com/en/articles/13451#model_2_20)

[2.20.1. Code for Creating the Complement Naive Bayes (CNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_20_1)

[2.20.2. MQL5 Code for Working with the Complement Naive Bayes (CNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_20_2)

[2.20.3. ONNX Representation of the Complement Naive Bayes (CNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_20_3)

- [2.21. Bernoulli Naive Bayes (BNB) Classifier](https://www.mql5.com/en/articles/13451#model_2_21)

[2.21.1. Code for Creating the Bernoulli Naive Bayes (BNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_21_1)

[2.21.2. MQL5 Code for Working with the Bernoulli Naive Bayes (BNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_21_2)

[2.21.3. ONNX Representation of the Bernoulli Naive Bayes (BNB) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_21_3)

- [2.22. Multilayer Perceptron Classifier](https://www.mql5.com/en/articles/13451#model_2_22)

[2.22.1. Code for Creating the Multilayer Perceptron Classifier Model](https://www.mql5.com/en/articles/13451#model_2_22_1)

[2.22.2. MQL5 Code for Working with the Multilayer Perceptron Classifier Model](https://www.mql5.com/en/articles/13451#model_2_22_2)

[2.22.3. ONNX Representation of the Multilayer Perceptron Classifier Model](https://www.mql5.com/en/articles/13451#model_2_22_3)

- [2.23. Linear Discriminant Analysis (LDA) Classifier](https://www.mql5.com/en/articles/13451#model_2_23)

[2.23.1. Code for Creating the Linear Discriminant Analysis (LDA) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_23_1)

[2.23.2. MQL5 Code for Working with the Linear Discriminant Analysis (LDA) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_23_2)

[2.23.3. ONNX Representation of the Linear Discriminant Analysis (LDA) Classifier Model](https://www.mql5.com/en/articles/13451#model_2_23_3)

- [2.24. Hist Gradient Boosting](https://www.mql5.com/en/articles/13451#model_2_24)

[2.24.1. Code for Creating the Histogram-Based Gradient Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_24_1)

[2.24.2. MQL5 Code for Working with the Histogram-Based Gradient Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_24_2) [2.24.3. ONNX Representation of the Histogram-Based Gradient Boosting Classifier Model](https://www.mql5.com/en/articles/13451#model_2_24_3)
- [2.25. CategoricalNB Classifier](https://www.mql5.com/en/articles/13451#model_2_25)

[2.25.1. Code for Creating the CategoricalNB Classifier Model](https://www.mql5.com/en/articles/13451#model_2_25_1)

[2.25.2. MQL5 Code for Working with the CategoricalNB Classifier Model](https://www.mql5.com/en/articles/13451#model_2_25_2)

[2.25.3. ONNX Representation of the CategoricalNB Classifier Model](https://www.mql5.com/en/articles/13451#model_2_25_3)

- [2.26. ExtraTreeClassifier](https://www.mql5.com/en/articles/13451#model_2_26)

[2.26.1. Code for Creating the ExtraTreeClassifier Model](https://www.mql5.com/en/articles/13451#model_2_26_1)

[2.26.2. MQL5 Code for Working with the ExtraTreeClassifier Model](https://www.mql5.com/en/articles/13451#model_2_26_2)

[2.26.3. ONNX Representation of the ExtraTreeClassifier Model](https://www.mql5.com/en/articles/13451#model_2_26_3)

- [2.27. ExtraTreesClassifier](https://www.mql5.com/en/articles/13451#model_2_27)

[2.27.1. Code for Creating the ExtraTreesClassifier Model](https://www.mql5.com/en/articles/13451#model_2_27_1)

[2.27.2. MQL5 Code for Working with the ExtraTreesClassifier Model](https://www.mql5.com/en/articles/13451#model_2_27_2)

[2.27.3. ONNX Representation of the ExtraTreesClassifier Model](https://www.mql5.com/en/articles/13451#model_2_27_3)

- [2.28. Comparing the Accuracy of All Models](https://www.mql5.com/en/articles/13451#model_2_28)

[2.28.1. Code for Calculating All Models and Building an Accuracy Comparison Chart](https://www.mql5.com/en/articles/13451#model_2_28_1)

[2.28.2. MQL5 Code for Executing All ONNX Models](https://www.mql5.com/en/articles/13451#model_2_28_2)

- [2.29. Scikit-Learn Classification Models That Could Not Be Converted to ONNX](https://www.mql5.com/en/articles/13451#model_2_29)

- [2.29.1. DummyClassifier](https://www.mql5.com/en/articles/13451#model_2_29_1)

[2.29.1.1. Code for Creating the DummyClassifier Model](https://www.mql5.com/en/articles/13451#model_2_29_1_1)

- [2.29.2. GaussianProcessClassifier](https://www.mql5.com/en/articles/13451#model_2_29_2)

[2.29.2.1. Code for Creating the GaussianProcessClassifier Model](https://www.mql5.com/en/articles/13451#model_2_29_2_1)

- [2.29.3. LabelPropagation Classifier](https://www.mql5.com/en/articles/13451#model_2_29_3)

[2.29.3.1. Code for Creating the LabelPropagationClassifier Model](https://www.mql5.com/en/articles/13451#model_2_29_3_1)

- [2.29.4. LabelSpreading Classifier](https://www.mql5.com/en/articles/13451#model_2_29_4)

[2.29.4.1. Code for Creating the LabelSpreadingClassifier Model](https://www.mql5.com/en/articles/13451#model_2_29_4_1)

- [2.29.5. NearestCentroid Classifier](https://www.mql5.com/en/articles/13451#model_2_29_5)

[2.29.5.1. Code for Creating the NearestCentroid Model](https://www.mql5.com/en/articles/13451#model_2_29_5_1)

- [2.29.6. Quadratic Discriminant Analysis Classifier](https://www.mql5.com/en/articles/13451#model_2_29_6)

[2.29.6.1. Code for Creating the Quadratic Discriminant Analysis Model](https://www.mql5.com/en/articles/13451#model_2_29_6_1)

- [Conclusions](https://www.mql5.com/en/articles/13451#conclusions)

### 1\. Fisher's Irises

The Iris dataset is one of the most well-known and widely used datasets in the field of machine learning. It was first introduced in 1936 by the statistician and biologist R.A. Fisher and has since become a classic dataset for classification tasks.

The Iris dataset consists of measurements of sepals and petals of three species of irises - Iris setosa, Iris virginica, and Iris versicolor.

![Iris setosa](https://c.mql5.com/2/58/IRIS_setosa.png)

Figure 1. Iris setosa

![Figure 2. Iris virginica](https://c.mql5.com/2/58/IRIS_virginica.png)

Figure 2. Iris virginica

![Figure 3. Iris versicolor](https://c.mql5.com/2/58/IRIS_versicolor.png)

Figure 3. Iris versicolor

The Iris dataset comprises 150 instances of irises, with 50 instances of each of the three species. Each instance has four numerical features (measured in centimeters):

1. Sepal length
2. Sepal width
3. Petal length
4. Ppetal width

Each instance also has a corresponding class indicating the iris species (Iris setosa, Iris virginica, or Iris versicolor). This classification attribute makes the Iris dataset an ideal dataset for machine learning tasks such as classification and clustering.

MetaEditor allows working with Python scripts. To create a Python script, select "New" from the "File" menu in MetaEditor, and a dialog for choosing the object to be created will appear (see Figure 4).

![Figure 4. Creating a Python script in MQL5 Wizard - Step 1](https://c.mql5.com/2/58/Wizard1.png)

Figure 4. Creating a Python script in MQL5 Wizard - Step 1

Next, provide a name for the script, for example, "IRIS.py" (see Figure 5).

![Figure 5. Creating a Python script in MQL5 Wizard - Step 2 - Script Name](https://c.mql5.com/2/58/Wizard2.png)

Figure 5. Creating a Python script in MQL5 Wizard - Step 2 - Script Name

After that, you can specify which libraries will be used. In our case, we will leave these fields empty (see Figure 6).

![Figure 6: Creating a Python script in MQL5 Wizard - Step 3](https://c.mql5.com/2/58/Wizard3.png)

Figure 6: Creating a Python script in MQL5 Wizard - Step 3

One way to start analyzing the Iris dataset is by visualizing the data. A graphical representation allows us to better understand the data's structure and relationships between features.

For example, you can create a scatter plot to see how different species of irises are distributed in the feature space.

Python script code:

```
# The script shows the scatter plot of the Iris dataset features
# Copyright 2023, MetaQuotes Ltd.
# https://mql5.com

import matplotlib.pyplot as plt
from sklearn import datasets

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# extract sepal length and sepal width (the first two features)
sepal_length = X[:, 0]
sepal_width = X[:, 1]

# create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(sepal_length, sepal_width, c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatter Plot for Sepal Length and Sepal Width')
plt.colorbar(label='Iris Species', ticks=[0, 1, 2])
plt.show()

# save the scatter plot to a file (optional)
# plt.savefig('scatter_plot_sepal_length_width.png')

# Extract petal length and petal width (the third and fourth features)
petal_length = X[:, 2]
petal_width = X[:, 3]

# create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(petal_length, petal_width, c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Scatter Plot for Petal Length and Petal Width')
plt.colorbar(label='Iris Species', ticks=[0, 1, 2])
plt.show()

# save the scatter plot to a file (optional)
# plt.savefig('scatter_plot_petal_length_width.png')
```

To run this script, you need to copy it into MetaEditor (see Figure 7) and click "Compile."

![Figure 7: IRIS.py script in MetaEditor](https://c.mql5.com/2/58/Fig7_Iris_script_in_MetaEditor.png)

Figure 7: IRIS.py script in MetaEditor

After that, the plots will appear on the screen:

![Figure 8: IRIS.py script in MetaEditor with Sepal Length/Sepal Width plot](https://c.mql5.com/2/58/Fig8_Iris_script_in_MetaEditor_Scatter_plot_Sepal_Length_vs_Sepal_Width.png)

Figure 8: IRIS.py script in MetaEditor with Sepal Length/Sepal Width plot

![Figure 9: IRIS.py script in MetaEditor with Petal Length/Petal Width plot](https://c.mql5.com/2/58/Fig9_Iris_script_in_MetaEditor_Scatter_plot_Petal_Length_vs_Petal_Width.png)

Figure 9: IRIS.py script in MetaEditor with Petal Length/Petal Width plot

Let's take a closer look at them.

![Figure 10: Scatter Plot Sepal Length vs Sepal Width](https://c.mql5.com/2/58/IRIS_1.png)

Figure 10: Scatter Plot Sepal Length vs Sepal Width

In this plot, we can see how different iris species are distributed based on sepal length and sepal width. We can observe that Iris setosa typically has shorter and wider sepals compared to the other two species.

![Figure 11: Scatter Plot Petal Length vs Petal Width](https://c.mql5.com/2/58/IRIS_2.png)

Figure 11: Scatter Plot Petal Length vs Petal Width

In this plot, we can see how different iris species are distributed based on petal length and petal width. We can notice that Iris setosa has the shortest and narrowest petals, Iris virginica has the longest and widest petals, and Iris versicolor falls in between.

The Iris dataset is an ideal dataset for training and testing machine learning models. We will use it to analyze the effectiveness of machine learning models for a classification task.

### 2\. Models for Classification

Classification is one of the fundamental tasks in machine learning, and its goal is to categorize data into different categories or classes based on certain features.

Let's explore the main machine learning models in the scikit-learn package.

**List of Scikit-learn classifiers**

To display a list of available classifiers in scikit-learn, you can use the following script:

```
# ScikitLearnClassifiers.py
# The script lists all the classification algorithms available in scikit-learn
# Copyright 2023, MetaQuotes Ltd.
# https://mql5.com

# print Python version
from platform import python_version
print("The Python version is ", python_version())

# print scikit-learn version
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

# print scikit-learn classifiers
from sklearn.utils import all_estimators
classifiers = all_estimators(type_filter='classifier')
for index, (name, ClassifierClass) in enumerate(classifiers, start=1):
    print(f"Classifier {index}: {name}")
```

Output:

Python    The Python version is  3.10.0

Python    The scikit-learn version is 1.2.2.

Python    Classifier 1: AdaBoostClassifier

Python    Classifier 2: BaggingClassifier

Python    Classifier 3: BernoulliNB

Python    Classifier 4: CalibratedClassifierCV

Python    Classifier 5: CategoricalNB

Python    Classifier 6: ClassifierChain

Python    Classifier 7: ComplementNB

Python    Classifier 8: DecisionTreeClassifier

Python    Classifier 9: DummyClassifier

Python    Classifier 10: ExtraTreeClassifier

Python    Classifier 11: ExtraTreesClassifier

Python    Classifier 12: GaussianNB

Python    Classifier 13: GaussianProcessClassifier

Python    Classifier 14: GradientBoostingClassifier

Python    Classifier 15: HistGradientBoostingClassifier

Python    Classifier 16: KNeighborsClassifier

Python    Classifier 17: LabelPropagation

Python    Classifier 18: LabelSpreading

Python    Classifier 19: LinearDiscriminantAnalysis

Python    Classifier 20: LinearSVC

Python    Classifier 21: LogisticRegression

Python    Classifier 22: LogisticRegressionCV

Python    Classifier 23: MLPClassifier

Python    Classifier 24: MultiOutputClassifier

Python    Classifier 25: MultinomialNB

Python    Classifier 26: NearestCentroid

Python    Classifier 27: NuSVC

Python    Classifier 28: OneVsOneClassifier

Python    Classifier 29: OneVsRestClassifier

Python    Classifier 30: OutputCodeClassifier

Python    Classifier 31: PassiveAggressiveClassifier

Python    Classifier 32: Perceptron

Python    Classifier 33: QuadraticDiscriminantAnalysis

Python    Classifier 34: RadiusNeighborsClassifier

Python    Classifier 35: RandomForestClassifier

Python    Classifier 36: RidgeClassifier

Python    Classifier 37: RidgeClassifierCV

Python    Classifier 38: SGDClassifier

Python    Classifier 39: SVC

Python    Classifier 40: StackingClassifier

Python    Classifier 41: VotingClassifier

For convenience in this list of classifiers, they are highlighted with different colors. Models that require base classifiers are highlighted in yellow, while other models can be used independently.

Looking ahead, it's worth noting that green-colored models have been successfully exported to the ONNX format, while red-colored models encounter errors during conversion in the current version of scikit-learn 1.2.2.

**Different representation of output data in models**

It should be noted that different models represent output data differently, so when working with models converted to ONNX, one should be attentive.

For the Fisher's Iris classification task, the input tensors have the same format for all these models:

Information about input tensors in ONNX:

1\. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

The output tensors of ONNX models differ.

**1\. Models that do not require post-processing:**

1. SVC Classifier;
2. LinearSVC Classifier;
3. NuSVC Classifier;

4. Radius Neighbors Classifier;
5. Ridge Classifier;
6. Ridge Classifier CV.


Information about output tensors in ONNX:

1\. Name: label, Data Type: tensor(int64), Shape: \[None\]

2\. Name: probabilities, Data Type: tensor(float), Shape: \[None, 3\]

These models return the result (class number) explicitly in the first output integer tensor "label," without requiring post-processing.

**2\. Models whose results require post-processing:**

01. Random Forest Classifier;
02. Gradient Boosting Classifier;
03. AdaBoost Classifier;
04. Bagging Classifier;
05. K-NN\_Classifier;
06. Decision Tree Classifier;
07. Logistic Regression Classifier;
08. Logistic Regression CV Classifier;

09. Passive-Aggressive Classifier;
10. Perceptron Classifier;
11. SGD Classifier;
12. Gaussian Naive Bayes Classifier;
13. Multinomial Naive Bayes Classifier;
14. Complement Naive Bayes Classifier;
15. Bernoulli Naive Bayes Classifier;
16. Multilayer Perceptron Classifier;
17. Linear Discriminant Analysis Classifier;
18. Hist Gradient Boosting Classifier;
19. Categorical  Naive Bayes Classifier;
20. ExtraTree Classifier;
21. ExtraTrees Classifier.


Information about output tensors in ONNX:

1\. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

2\. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

These models return a list of classes and probabilities of belonging to each class.

To obtain the result in these cases, post-processing is required, such as seq(map(int64, tensor(float)) (finding the element with the highest probability).

Therefore, it is essential to be attentive and consider these aspects when working with ONNX models. An example of different result processing is presented in script in [2.28.2](https://www.mql5.com/en/articles/13451#model_2_28_2).

**iris.mqh**

To test models on the full Iris dataset in MQL5, data preparation is required. For this purpose, the function PrepareIrisDataset() will be used.

It's convenient to move these functions to the iris.mqh file.

```
//+------------------------------------------------------------------+
//|                                                         Iris.mqh |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"

//+------------------------------------------------------------------+
//| Structure for the IRIS Dataset sample                            |
//+------------------------------------------------------------------+
struct sIRISsample
  {
   int               sample_id;   // sample id (1-150)
   double            features[4]; // SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm
   string            class_name;  // class ("Iris-setosa","Iris-versicolor","Iris-virginica")
   int               class_id;    // class id (0,1,2), calculated by function IRISClassID
  };

//--- Iris dataset
sIRISsample ExtIRISDataset[];
int Exttotal=0;

//+------------------------------------------------------------------+
//| Returns class id by class name                                   |
//+------------------------------------------------------------------+
int IRISClassID(string class_name)
  {
//---
   if(class_name=="Iris-setosa")
      return(0);
   else
      if(class_name=="Iris-versicolor")
         return(1);
      else
         if(class_name=="Iris-virginica")
            return(2);
//---
   return(-1);
  }

//+------------------------------------------------------------------+
//| AddSample                                                        |
//+------------------------------------------------------------------+
bool AddSample(const int Id,const double SepalLengthCm,const double SepalWidthCm,const double PetalLengthCm,const double PetalWidthCm, const string Species)
  {
//---
   ExtIRISDataset[Exttotal].sample_id=Id;
//---
   ExtIRISDataset[Exttotal].features[0]=SepalLengthCm;
   ExtIRISDataset[Exttotal].features[1]=SepalWidthCm;
   ExtIRISDataset[Exttotal].features[2]=PetalLengthCm;
   ExtIRISDataset[Exttotal].features[3]=PetalWidthCm;
//---
   ExtIRISDataset[Exttotal].class_name=Species;
   ExtIRISDataset[Exttotal].class_id=IRISClassID(Species);
//---
   Exttotal++;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Prepare Iris Dataset                                             |
//+------------------------------------------------------------------+
bool PrepareIrisDataset(sIRISsample &iris_samples[])
  {
   ArrayResize(ExtIRISDataset,150);
   Exttotal=0;
//---
   AddSample(1,5.1,3.5,1.4,0.2,"Iris-setosa");
   AddSample(2,4.9,3.0,1.4,0.2,"Iris-setosa");
   AddSample(3,4.7,3.2,1.3,0.2,"Iris-setosa");
   AddSample(4,4.6,3.1,1.5,0.2,"Iris-setosa");
   AddSample(5,5.0,3.6,1.4,0.2,"Iris-setosa");
   AddSample(6,5.4,3.9,1.7,0.4,"Iris-setosa");
   AddSample(7,4.6,3.4,1.4,0.3,"Iris-setosa");
   AddSample(8,5.0,3.4,1.5,0.2,"Iris-setosa");
   AddSample(9,4.4,2.9,1.4,0.2,"Iris-setosa");
   AddSample(10,4.9,3.1,1.5,0.1,"Iris-setosa");
   AddSample(11,5.4,3.7,1.5,0.2,"Iris-setosa");
   AddSample(12,4.8,3.4,1.6,0.2,"Iris-setosa");
   AddSample(13,4.8,3.0,1.4,0.1,"Iris-setosa");
   AddSample(14,4.3,3.0,1.1,0.1,"Iris-setosa");
   AddSample(15,5.8,4.0,1.2,0.2,"Iris-setosa");
   AddSample(16,5.7,4.4,1.5,0.4,"Iris-setosa");
   AddSample(17,5.4,3.9,1.3,0.4,"Iris-setosa");
   AddSample(18,5.1,3.5,1.4,0.3,"Iris-setosa");
   AddSample(19,5.7,3.8,1.7,0.3,"Iris-setosa");
   AddSample(20,5.1,3.8,1.5,0.3,"Iris-setosa");
   AddSample(21,5.4,3.4,1.7,0.2,"Iris-setosa");
   AddSample(22,5.1,3.7,1.5,0.4,"Iris-setosa");
   AddSample(23,4.6,3.6,1.0,0.2,"Iris-setosa");
   AddSample(24,5.1,3.3,1.7,0.5,"Iris-setosa");
   AddSample(25,4.8,3.4,1.9,0.2,"Iris-setosa");
   AddSample(26,5.0,3.0,1.6,0.2,"Iris-setosa");
   AddSample(27,5.0,3.4,1.6,0.4,"Iris-setosa");
   AddSample(28,5.2,3.5,1.5,0.2,"Iris-setosa");
   AddSample(29,5.2,3.4,1.4,0.2,"Iris-setosa");
   AddSample(30,4.7,3.2,1.6,0.2,"Iris-setosa");
   AddSample(31,4.8,3.1,1.6,0.2,"Iris-setosa");
   AddSample(32,5.4,3.4,1.5,0.4,"Iris-setosa");
   AddSample(33,5.2,4.1,1.5,0.1,"Iris-setosa");
   AddSample(34,5.5,4.2,1.4,0.2,"Iris-setosa");
   AddSample(35,4.9,3.1,1.5,0.2,"Iris-setosa");
   AddSample(36,5.0,3.2,1.2,0.2,"Iris-setosa");
   AddSample(37,5.5,3.5,1.3,0.2,"Iris-setosa");
   AddSample(38,4.9,3.6,1.4,0.1,"Iris-setosa");
   AddSample(39,4.4,3.0,1.3,0.2,"Iris-setosa");
   AddSample(40,5.1,3.4,1.5,0.2,"Iris-setosa");
   AddSample(41,5.0,3.5,1.3,0.3,"Iris-setosa");
   AddSample(42,4.5,2.3,1.3,0.3,"Iris-setosa");
   AddSample(43,4.4,3.2,1.3,0.2,"Iris-setosa");
   AddSample(44,5.0,3.5,1.6,0.6,"Iris-setosa");
   AddSample(45,5.1,3.8,1.9,0.4,"Iris-setosa");
   AddSample(46,4.8,3.0,1.4,0.3,"Iris-setosa");
   AddSample(47,5.1,3.8,1.6,0.2,"Iris-setosa");
   AddSample(48,4.6,3.2,1.4,0.2,"Iris-setosa");
   AddSample(49,5.3,3.7,1.5,0.2,"Iris-setosa");
   AddSample(50,5.0,3.3,1.4,0.2,"Iris-setosa");
   AddSample(51,7.0,3.2,4.7,1.4,"Iris-versicolor");
   AddSample(52,6.4,3.2,4.5,1.5,"Iris-versicolor");
   AddSample(53,6.9,3.1,4.9,1.5,"Iris-versicolor");
   AddSample(54,5.5,2.3,4.0,1.3,"Iris-versicolor");
   AddSample(55,6.5,2.8,4.6,1.5,"Iris-versicolor");
   AddSample(56,5.7,2.8,4.5,1.3,"Iris-versicolor");
   AddSample(57,6.3,3.3,4.7,1.6,"Iris-versicolor");
   AddSample(58,4.9,2.4,3.3,1.0,"Iris-versicolor");
   AddSample(59,6.6,2.9,4.6,1.3,"Iris-versicolor");
   AddSample(60,5.2,2.7,3.9,1.4,"Iris-versicolor");
   AddSample(61,5.0,2.0,3.5,1.0,"Iris-versicolor");
   AddSample(62,5.9,3.0,4.2,1.5,"Iris-versicolor");
   AddSample(63,6.0,2.2,4.0,1.0,"Iris-versicolor");
   AddSample(64,6.1,2.9,4.7,1.4,"Iris-versicolor");
   AddSample(65,5.6,2.9,3.6,1.3,"Iris-versicolor");
   AddSample(66,6.7,3.1,4.4,1.4,"Iris-versicolor");
   AddSample(67,5.6,3.0,4.5,1.5,"Iris-versicolor");
   AddSample(68,5.8,2.7,4.1,1.0,"Iris-versicolor");
   AddSample(69,6.2,2.2,4.5,1.5,"Iris-versicolor");
   AddSample(70,5.6,2.5,3.9,1.1,"Iris-versicolor");
   AddSample(71,5.9,3.2,4.8,1.8,"Iris-versicolor");
   AddSample(72,6.1,2.8,4.0,1.3,"Iris-versicolor");
   AddSample(73,6.3,2.5,4.9,1.5,"Iris-versicolor");
   AddSample(74,6.1,2.8,4.7,1.2,"Iris-versicolor");
   AddSample(75,6.4,2.9,4.3,1.3,"Iris-versicolor");
   AddSample(76,6.6,3.0,4.4,1.4,"Iris-versicolor");
   AddSample(77,6.8,2.8,4.8,1.4,"Iris-versicolor");
   AddSample(78,6.7,3.0,5.0,1.7,"Iris-versicolor");
   AddSample(79,6.0,2.9,4.5,1.5,"Iris-versicolor");
   AddSample(80,5.7,2.6,3.5,1.0,"Iris-versicolor");
   AddSample(81,5.5,2.4,3.8,1.1,"Iris-versicolor");
   AddSample(82,5.5,2.4,3.7,1.0,"Iris-versicolor");
   AddSample(83,5.8,2.7,3.9,1.2,"Iris-versicolor");
   AddSample(84,6.0,2.7,5.1,1.6,"Iris-versicolor");
   AddSample(85,5.4,3.0,4.5,1.5,"Iris-versicolor");
   AddSample(86,6.0,3.4,4.5,1.6,"Iris-versicolor");
   AddSample(87,6.7,3.1,4.7,1.5,"Iris-versicolor");
   AddSample(88,6.3,2.3,4.4,1.3,"Iris-versicolor");
   AddSample(89,5.6,3.0,4.1,1.3,"Iris-versicolor");
   AddSample(90,5.5,2.5,4.0,1.3,"Iris-versicolor");
   AddSample(91,5.5,2.6,4.4,1.2,"Iris-versicolor");
   AddSample(92,6.1,3.0,4.6,1.4,"Iris-versicolor");
   AddSample(93,5.8,2.6,4.0,1.2,"Iris-versicolor");
   AddSample(94,5.0,2.3,3.3,1.0,"Iris-versicolor");
   AddSample(95,5.6,2.7,4.2,1.3,"Iris-versicolor");
   AddSample(96,5.7,3.0,4.2,1.2,"Iris-versicolor");
   AddSample(97,5.7,2.9,4.2,1.3,"Iris-versicolor");
   AddSample(98,6.2,2.9,4.3,1.3,"Iris-versicolor");
   AddSample(99,5.1,2.5,3.0,1.1,"Iris-versicolor");
   AddSample(100,5.7,2.8,4.1,1.3,"Iris-versicolor");
   AddSample(101,6.3,3.3,6.0,2.5,"Iris-virginica");
   AddSample(102,5.8,2.7,5.1,1.9,"Iris-virginica");
   AddSample(103,7.1,3.0,5.9,2.1,"Iris-virginica");
   AddSample(104,6.3,2.9,5.6,1.8,"Iris-virginica");
   AddSample(105,6.5,3.0,5.8,2.2,"Iris-virginica");
   AddSample(106,7.6,3.0,6.6,2.1,"Iris-virginica");
   AddSample(107,4.9,2.5,4.5,1.7,"Iris-virginica");
   AddSample(108,7.3,2.9,6.3,1.8,"Iris-virginica");
   AddSample(109,6.7,2.5,5.8,1.8,"Iris-virginica");
   AddSample(110,7.2,3.6,6.1,2.5,"Iris-virginica");
   AddSample(111,6.5,3.2,5.1,2.0,"Iris-virginica");
   AddSample(112,6.4,2.7,5.3,1.9,"Iris-virginica");
   AddSample(113,6.8,3.0,5.5,2.1,"Iris-virginica");
   AddSample(114,5.7,2.5,5.0,2.0,"Iris-virginica");
   AddSample(115,5.8,2.8,5.1,2.4,"Iris-virginica");
   AddSample(116,6.4,3.2,5.3,2.3,"Iris-virginica");
   AddSample(117,6.5,3.0,5.5,1.8,"Iris-virginica");
   AddSample(118,7.7,3.8,6.7,2.2,"Iris-virginica");
   AddSample(119,7.7,2.6,6.9,2.3,"Iris-virginica");
   AddSample(120,6.0,2.2,5.0,1.5,"Iris-virginica");
   AddSample(121,6.9,3.2,5.7,2.3,"Iris-virginica");
   AddSample(122,5.6,2.8,4.9,2.0,"Iris-virginica");
   AddSample(123,7.7,2.8,6.7,2.0,"Iris-virginica");
   AddSample(124,6.3,2.7,4.9,1.8,"Iris-virginica");
   AddSample(125,6.7,3.3,5.7,2.1,"Iris-virginica");
   AddSample(126,7.2,3.2,6.0,1.8,"Iris-virginica");
   AddSample(127,6.2,2.8,4.8,1.8,"Iris-virginica");
   AddSample(128,6.1,3.0,4.9,1.8,"Iris-virginica");
   AddSample(129,6.4,2.8,5.6,2.1,"Iris-virginica");
   AddSample(130,7.2,3.0,5.8,1.6,"Iris-virginica");
   AddSample(131,7.4,2.8,6.1,1.9,"Iris-virginica");
   AddSample(132,7.9,3.8,6.4,2.0,"Iris-virginica");
   AddSample(133,6.4,2.8,5.6,2.2,"Iris-virginica");
   AddSample(134,6.3,2.8,5.1,1.5,"Iris-virginica");
   AddSample(135,6.1,2.6,5.6,1.4,"Iris-virginica");
   AddSample(136,7.7,3.0,6.1,2.3,"Iris-virginica");
   AddSample(137,6.3,3.4,5.6,2.4,"Iris-virginica");
   AddSample(138,6.4,3.1,5.5,1.8,"Iris-virginica");
   AddSample(139,6.0,3.0,4.8,1.8,"Iris-virginica");
   AddSample(140,6.9,3.1,5.4,2.1,"Iris-virginica");
   AddSample(141,6.7,3.1,5.6,2.4,"Iris-virginica");
   AddSample(142,6.9,3.1,5.1,2.3,"Iris-virginica");
   AddSample(143,5.8,2.7,5.1,1.9,"Iris-virginica");
   AddSample(144,6.8,3.2,5.9,2.3,"Iris-virginica");
   AddSample(145,6.7,3.3,5.7,2.5,"Iris-virginica");
   AddSample(146,6.7,3.0,5.2,2.3,"Iris-virginica");
   AddSample(147,6.3,2.5,5.0,1.9,"Iris-virginica");
   AddSample(148,6.5,3.0,5.2,2.0,"Iris-virginica");
   AddSample(149,6.2,3.4,5.4,2.3,"Iris-virginica");
   AddSample(150,5.9,3.0,5.1,1.8,"Iris-virginica");
//---
   ArrayResize(iris_samples,150);
   for(int i=0; i<Exttotal; i++)
     {
      iris_samples[i]=ExtIRISDataset[i];
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
```

**Note on Classification Methods: SVC, LinearSVC, and NuSVC**

Let's compare three popular classification methods: SVC (Support Vector Classification), LinearSVC (Linear Support Vector Classification), and NuSVC (Nu Support Vector Classification).

Principles of Operation:

    SVC (Support Vector Classification)

        Working Principle: SVC is a classification method based on maximizing the margin between classes. It seeks an optimal separating hyperplane that maximally separates classes and supports support vectors - points closest to the hyperplane.

        Kernel Functions: SVC can use various kernel functions, such as linear, radial basis function (RBF), polynomial, and others. The kernel function determines how data is transformed to find the optimal hyperplane.

    LinearSVC (Linear Support Vector Classification)

        Working Principle: LinearSVC is a variant of SVC specializing in linear classification. It seeks an optimal linear separating hyperplane without using kernel functions. This makes it faster and more efficient when working with large volumes of data.

    NuSVC (Nu Support Vector Classification)

        Working Principle: NuSVC is also based on support vector methods but introduces a parameter Nu (nu), which controls the model's complexity and the fraction of support vectors. The Nu value falls in the range from 0 to 1 and determines how much of the data can be used for support vectors and errors.

Advantages:

    SVC

        Powerful Algorithm: SVC can handle complex classification tasks and work with non-linear data thanks to the use of kernel functions.

        Robustness to Outliers: SVC is robust to data outliers as it uses support vectors to build the separating hyperplane.

    LinearSVC

        High Efficiency: LinearSVC is faster and more efficient when dealing with large datasets, especially when the data is large and linear separation is suitable for the task.

        Linear Classification: If the problem is well-linearly separable, LinearSVC can yield good results without the need for complex kernel functions.

    NuSVC

        Model Complexity Control: The Nu parameter in NuSVC allows you to control the model's complexity and the trade-off between fitting the data and generalization.

        Robustness to Outliers: Similar to SVC, NuSVC is robust to outliers, making it useful for tasks with noisy data.

Limitations:

    SVC

        Computational Complexity: SVC can be slow on large datasets and/or when using complex kernel functions.

        Kernel Sensitivity: Choosing the right kernel function can be a challenging task and significantly impact model performance.

    LinearSVC

        Linearity Constraint: LinearSVC is constrained by linear data separation and can perform poorly in cases with non-linear dependencies between features and the target variable.

    NuSVC

        Nu Parameter Tuning: Tuning the Nu parameter may require time and experimentation to achieve optimal results.

Depending on the task characteristics and data volume, each of these methods can be the best choice. It's important to conduct experiments and select the method that best suits the specific classification task requirements.

**2.1. SVC Classifier**

The Support Vector Classification (SVC) classification method is a powerful machine learning algorithm widely used for solving classification tasks.

Principles of Operation:

1. Optimal Separating Hyperplane

Working Principle: The main idea behind SVC is to find the optimal separating hyperplane in the feature space. This hyperplane should maximize the separation between objects of different classes and support support vectors, which are data points closest to the hyperplane.

Maximizing Margin: SVC aims to maximize the margin between classes, which means the distance from support vectors to the hyperplane. This allows the method to be robust to outliers and generalize well to new data.

2. Utilization of Kernel Functions

Kernel Functions: SVC can use various kernel functions, such as linear, radial basis function (RBF), polynomial, and others. The kernel function allows data to be projected into a higher-dimensional space where the task becomes linear, even if there is no linear separability in the original data space.

Kernel Selection: Choosing the right kernel function can significantly impact the performance of the SVC model. A linear hyperplane is not always the optimal solution.

Advantages:

- Powerful Algorithm. Handling Complex Tasks: SVC can solve complex classification tasks, including those with non-linear dependencies between features and the target variable.
- Robustness to Outliers: The use of support vectors makes the method robust to data outliers. It depends on support vectors rather than the entire dataset.
- Kernel Flexibility. Adaptability to Data: The ability to use different kernel functions allows SVC to adapt to specific data and discover non-linear relationships.
- Good Generalization. Generalization to New Data: The SVC model can generalize well to new data, making it useful for prediction tasks.

Limitations:

- Computational Complexity. Training Time: SVC can be slow to train, especially when dealing with large volumes of data or complex kernel functions.
- Kernel Selection. Choosing the Right Kernel Function: Selecting the correct kernel function may require experimentation and depends on data characteristics.
- Sensitivity to Feature Scaling. Data Normalization: SVC is sensitive to feature scaling, so it is recommended to normalize or standardize data before training.
- Model Interpretability. Interpretation Complexity: SVC models can be complex to interpret due to the use of non-linear kernels and a multitude of support vectors.

Depending on the specific task and data volume, the SVC method can be a powerful tool for solving classification tasks. However, it's essential to consider its limitations and tune parameters to achieve optimal results.

**2.1.1. Code for Creating the SVC Classifier Model**

This code demonstrates the process of training an SVC Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_SVCClassifier.py
# The code demonstrates the process of training SVC model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create an SVC Classifier model with a linear kernel
svc_model = SVC(kernel='linear', C=1.0)

# train the model on the entire dataset
svc_model.fit(X, y)

# predict classes for the entire dataset
y_pred = svc_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of SVC Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(svc_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path +"svc_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of SVC Classifier model in ONNX format:", accuracy_onnx)
```

After running the script in MetaEditor by using the "Compile" button, you can view the results of its execution in the Journal tab.

![Figure 12. Results of the Iris_SVMClassifier.py script in MetaEditor](https://c.mql5.com/2/58/Iris_SVCClassifier__2.png)

Figure 12. Results of the Iris\_SVMClassifier.py script in MetaEditor

Output of the Iris\_SVCClassifier.py script:

Python    Accuracy of SVC Classifier model: 0.9933333333333333

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      0.98      0.99        50

Python               2       0.98      1.00      0.99        50

Python

Python        accuracy                           0.99       150

Python       macro avg       0.99      0.99      0.99       150

Python    weighted avg       0.99      0.99      0.99       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\svc\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: probabilities, Data Type: tensor(float), Shape: \[None, 3\]

Python

Python    Accuracy of SVC Classifier model in ONNX format: 0.9933333333333333

Here, you can find information about the path where the ONNX model was saved, the types of input and output parameters of the ONNX model, as well as the accuracy in describing the Iris dataset.

The accuracy of describing the dataset using the SVM Classifier is 99%, and the model exported to the ONNX format shows the same level of accuracy.

Now, we will verify these results in MQL5 by running the constructed model for each of the 150 data samples. Additionally, the script includes an example of batch data processing.

**2.1.2. MQL5 Code for Working with the SVC Classifier model**

```
//+------------------------------------------------------------------+
//|                                           Iris_SVCClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "svc_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   ulong input_shape[]= { batch_size, input_data.Range(1)};
   OnnxSetInputShape(model,0,input_shape);
//---
   int output1[];
   float output2[][3];
//---
   ArrayResize(output1,(int)batch_size);
   ArrayResize(output2,(int)batch_size);
//---
   ulong output_shape[]= {batch_size};
   OnnxSetOutputShape(model,0,output_shape);
//---
   ulong output_shape2[]= {batch_size,3};
   OnnxSetOutputShape(model,1,output_shape2);
//---
   bool res=OnnxRun(model,ONNX_DEBUG_LOGS,input_data,output1,output2);
//--- classes are ready in output1[k];
   if(res)
     {
      for(int k=0; k<(int)batch_size; k++)
         model_classes_id[k]=output1[k];
     }
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="SVCClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

The results of the script's execution are displayed in the "Experts" tab of the MetaTrader 5 terminal.

```
Iris_SVCClassifier (EURUSD,H1)  model:SVCClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_SVCClassifier (EURUSD,H1)  model:SVCClassifier   correct results: 99.33%
Iris_SVCClassifier (EURUSD,H1)  model=SVCClassifier all samples accuracy=0.993333
Iris_SVCClassifier (EURUSD,H1)  model=SVCClassifier batch test accuracy=1.000000
```

The SVC model correctly classified 149 out of 150 samples, which is an excellent result. The model made only one classification error in the Iris dataset, predicting class 2 (versicolor) instead of class 1 (virginica) for sample #84.

It's worth noting that the accuracy of the exported ONNX model on the full Iris dataset is 99.33%, which matches the accuracy of the original model.

**2.1.3. ONNX Representation of the SVC Classifier Model**

You can view the built ONNX model in MetaEditor.

![Figure 13. ONNX Model svc_iris.onnx in MetaEditor](https://c.mql5.com/2/58/Fig13_ONNX_SVC_MetaEditor.png)

Figure 13. ONNX Model svc\_iris.onnx in MetaEditor

For more detailed information about the model's architecture, you can use Netron. To do this, click the "Open in Netron" button in the model's description in MetaEditor.

![Figure 14. ONNX Model svc_iris.onnx in Netron](https://c.mql5.com/2/58/Fig14_ONNX_SVC_Iris_Netron.png)

Figure 14. ONNX Model svc\_iris.onnx in Netron

Moreover, by pointing the mouse at the ONNX operators present in the model, you can get information about the parameters of these operators (SVMClassifier in Figure 15).

![Figure 15. ONNX Model svc_iris.onnx in Netron (SVMClassifier ONNX Operator Parameters)](https://c.mql5.com/2/58/Fig15_ONNX_SVC_Iris_SVMClassifier_options.png)

Figure 15. ONNX Model svc\_iris.onnx in Netron (SVMClassifier ONNX Operator Parameters)

**2.2. LinearSVC Classifier**

LinearSVC (Linear Support Vector Classification) is a powerful machine learning algorithm used for binary and multiclass classification tasks. It is based on the idea of finding a hyperplane that best separates the data.

Principles of LinearSVC:

1. Finding the optimal hyperplane: The main idea of LinearSVC is to find the optimal hyperplane that maximally separates the two classes of data. A hyperplane is a multi-dimensional plane defined by a linear equation.
2. Margin minimization: LinearSVC aims to minimize the margins (the distances between data points and the hyperplane). The larger the margins, the more effectively the hyperplane separates the classes.
3. Handling linearly non-separable data: LinearSVC can work with data that cannot be linearly separated in the original feature space, thanks to the use of kernel functions (kernel trick) that project the data into a higher-dimensional space where they can be linearly separated.

Advantages of LinearSVC:

- Good generalization: LinearSVC has good generalization ability and can perform well on new, unseen data.
- Efficiency: LinearSVC works quickly on large datasets and requires relatively few computational resources.
- Handling linearly non-separable data: Using kernel functions, LinearSVC can address classification tasks with linearly non-separable data.
- Scalability: LinearSVC can be efficiently used in tasks with a large number of features and substantial data volumes.

Limitations of LinearSVC:

- Only linear separating hyperplanes: LinearSVC constructs only linear separating hyperplanes, which may be insufficient for complex classification tasks with non-linear dependencies.
- Parameter selection: Choosing the right parameters (e.g., regularization parameter) may require expert knowledge or cross-validation.
- Sensitivity to outliers: LinearSVC can be sensitive to outliers in the data, which can affect classification quality.
- Model interpretability: Models created using LinearSVC may be less interpretable compared to some other methods.

LinearSVC is a powerful classification algorithm that excels in generalization, efficiency, and handling linearly non-separable data. It finds applications in various classification tasks, especially when data can be separated by a linear hyperplane. However, for complex tasks that require modeling non-linear dependencies, LinearSVC may be less suitable, and in such cases, alternative methods with more complex decision boundaries should be considered.

**2.2.1. Code for Сreating LinearSVC Classifier Model**

This code demonstrates the process of training a LinearSVC Classifier model on the Iris dataset, exporting it to ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_LinearSVC.py
# The code demonstrates the process of training LinearSVC model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a LinearSVC model
linear_svc_model = LinearSVC(C=1.0, max_iter=10000)

# train the model on the entire dataset
linear_svc_model.fit(X, y)

# predict classes for the entire dataset
y_pred = linear_svc_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of LinearSVC model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(linear_svc_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "linear_svc_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of LinearSVC model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of LinearSVC model: 0.9666666666666667

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.96      0.94      0.95        50

Python               2       0.94      0.96      0.95        50

Python

Python        accuracy                           0.97       150

Python       macro avg       0.97      0.97      0.97       150

Python    weighted avg       0.97      0.97      0.97       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\linear\_svc\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: probabilities, Data Type: tensor(float), Shape: \[None, 3\]

Python

Python    Accuracy of LinearSVC model in ONNX format: 0.9666666666666667

**2.2.2. MQL5 Code for Working with the LinearSVC Classifier Model**

```
//+------------------------------------------------------------------+
//|                                               Iris_LinearSVC.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "linear_svc_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   ulong input_shape[]= { batch_size, input_data.Range(1)};
   OnnxSetInputShape(model,0,input_shape);
//---
   int output1[];
   float output2[][3];
//---
   ArrayResize(output1,(int)batch_size);
   ArrayResize(output2,(int)batch_size);
//---
   ulong output_shape[]= {batch_size};
   OnnxSetOutputShape(model,0,output_shape);
//---
   ulong output_shape2[]= {batch_size,3};
   OnnxSetOutputShape(model,1,output_shape2);
//---
   bool res=OnnxRun(model,ONNX_DEBUG_LOGS,input_data,output1,output2);
//--- classes are ready in output1[k];
   if(res)
     {
      for(int k=0; k<(int)batch_size; k++)
         model_classes_id[k]=output1[k];
     }
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="LinearSVC";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_LinearSVC (EURUSD,H1)      model:LinearSVC  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_LinearSVC (EURUSD,H1)      model:LinearSVC  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_LinearSVC (EURUSD,H1)      model:LinearSVC  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_LinearSVC (EURUSD,H1)      model:LinearSVC  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_LinearSVC (EURUSD,H1)      model:LinearSVC  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_LinearSVC (EURUSD,H1)      model:LinearSVC   correct results: 96.67%
Iris_LinearSVC (EURUSD,H1)      model=LinearSVC all samples accuracy=0.966667
Iris_LinearSVC (EURUSD,H1)      model=LinearSVC batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 96.67%, which corresponds to the accuracy of the original model.

**2.2.3. ONNX Representation of the LinearSVC Classifier Model**

![Figure 16. ONNX Representation of the LinearSVC Classifier Model in Netron](https://c.mql5.com/2/58/onnx_2_linear_svc.png)

Figure 16. ONNX Representation of the LinearSVC Classifier Model in Netron

**2.3. NuSVC Classifier**

The Nu-Support Vector Classification (NuSVC) method is a powerful machine learning algorithm based on the Support Vector Machine (SVM) approach.

Principles of NuSVC:

1. Support Vector Machine (SVM): NuSVC is a variant of SVM used for binary and multiclass classification tasks. The core principle of SVM is to find the optimal separating hyperplane that maximally separates classes while maintaining the maximum margin.
2. The Nu Parameter: A key parameter in NuSVC is the Nu parameter (nu), which controls the model's complexity and defines the proportion of the sample that can be used as support vectors and errors. The value of Nu ranges from 0 to 1, where 0.5 means roughly half of the sample will be used as support vectors and errors.
3. Parameter Tuning: Determining the optimal values for the Nu parameter and other hyperparameters may require cross-validation and a search for the best values on the training data.
4. Kernel Functions: NuSVC can use various kernel functions such as linear, radial basis function (RBF), polynomial, and others. The kernel function determines how the feature space is transformed to find the separating hyperplane.

Advantages of NuSVC:

- Efficiency in High-Dimensional Spaces: NuSVC can work efficiently in high-dimensional spaces, making it suitable for tasks with a high number of features.
- Robustness to Outliers: SVM, and NuSVC in particular, are robust to outliers in data due to the use of support vectors.
- Control of Model Complexity: The Nu parameter allows for controlling model complexity and balancing data fitting with generalization.
- Good Generalization: SVM and NuSVC, in particular, exhibit good generalization, resulting in excellent performance on new, previously unseen data.

Limitations of NuSVC:

- Inefficiency with Large Data Volumes: NuSVC can be inefficient when trained on large data volumes due to computational complexity.
- Parameter Tuning Required: Tuning the Nu parameter and kernel function may require time and computational resources.
- Kernel Function Linearity: The effectiveness of NuSVC can significantly depend on the choice of kernel function, and for some tasks, experimentation with different functions may be necessary.
- Model Interpretability: SVM and NuSVC provide excellent results, but their models can be complex to interpret, especially when non-linear kernels are used.

Nu-Support Vector Classification (NuSVC) is a powerful classification method based on SVM with several advantages, including robustness to outliers and good generalization. However, its effectiveness depends on parameter and kernel function selection, and it can be inefficient for large data volumes. It is essential to carefully select parameters and adapt the method to specific classification tasks.

**2.3.1. Code for Creating NuSVC Classifier Model**

This code demonstrates the process of training a NuSVC Classifier model on the Iris dataset, exporting it to ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_NuSVC.py
# The code demonstrates the process of training NuSVC model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a NuSVC model
nusvc_model = NuSVC(nu=0.5, kernel='linear')

# train the model on the entire dataset
nusvc_model.fit(X, y)

# predict classes for the entire dataset
y_pred = nusvc_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of NuSVC model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(nusvc_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "nusvc_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of NuSVC model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of NuSVC model: 0.9733333333333334

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.96      0.96      0.96        50

Python               2       0.96      0.96      0.96        50

Python

Python        accuracy                           0.97       150

Python       macro avg       0.97      0.97      0.97       150

Python    weighted avg       0.97      0.97      0.97       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\nusvc\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: probabilities, Data Type: tensor(float), Shape: \[None, 3\]

Python

Python    Accuracy of NuSVC model in ONNX format: 0.9733333333333334

**2.3.2. MQL5 Code for Working with the NuSVC Classifier Model**

```
//+------------------------------------------------------------------+
//|                                                   Iris_NuSVC.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "nusvc_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   ulong input_shape[]= { batch_size, input_data.Range(1)};
   OnnxSetInputShape(model,0,input_shape);
//---
   int output1[];
   float output2[][3];
//---
   ArrayResize(output1,(int)batch_size);
   ArrayResize(output2,(int)batch_size);
//---
   ulong output_shape[]= {batch_size};
   OnnxSetOutputShape(model,0,output_shape);
//---
   ulong output_shape2[]= {batch_size,3};
   OnnxSetOutputShape(model,1,output_shape2);
//---
   bool res=OnnxRun(model,ONNX_DEBUG_LOGS,input_data,output1,output2);
//--- classes are ready in output1[k];
   if(res)
     {
      for(int k=0; k<(int)batch_size; k++)
         model_classes_id[k]=output1[k];
     }
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="NuSVC";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_NuSVC (EURUSD,H1)  model:NuSVC  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_NuSVC (EURUSD,H1)  model:NuSVC  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_NuSVC (EURUSD,H1)  model:NuSVC  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_NuSVC (EURUSD,H1)  model:NuSVC  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_NuSVC (EURUSD,H1)  model:NuSVC   correct results: 97.33%
Iris_NuSVC (EURUSD,H1)  model=NuSVC all samples accuracy=0.973333
Iris_NuSVC (EURUSD,H1)  model=NuSVC batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full iris dataset is 97.33%, which corresponds to the accuracy of the original model.

**2.3.3. ONNX Representation of the NuSVC Classifier Model**

![Figure 17. ONNX Representation of the NuSVC Classifier Model in Netron](https://c.mql5.com/2/58/onnx_3_nu_svc.png)

Figure 17. ONNX Representation of the NuSVC Classifier Model in Netron

**2.4. Radius Neighbors Classifier**

The Radius Neighbors Classifier is a machine learning method used for classification tasks based on the principle of proximity between objects. Unlike the classical K-Nearest Neighbors (K-NN) Classifier, where a fixed number of nearest neighbors (K) is chosen, in the Radius Neighbors Classifier, objects are classified based on the distance to the nearest neighbors within a specified radius.

Principles of the Radius Neighbors Classifier:

1. Determining the radius: The main parameter of the Radius Neighbors Classifier is the radius, which defines the maximum distance between an object and its neighbors for it to be considered close to the neighbors' class.
2. Finding nearest neighbors: The distance to all other objects in the training dataset is calculated for each object. Those objects located within the specified radius are considered neighbors of the object.
3. Voting: The Radius Neighbors Classifier uses majority voting among the neighbors to determine the class of the object. For example, if the majority of neighbors belong to class A, the object will also be classified as class A.

Advantages of the Radius Neighbors Classifier:

- Adaptability to data density: The Radius Neighbors Classifier is suitable for tasks where the data density in different feature space regions may vary.
- Ability to work with different class shapes: This method performs well in tasks where classes have complex and nonlinear shapes.
- Suitable for data with outliers: The Radius Neighbors Classifier is more robust to outliers than K-NN because it disregards neighbors located beyond the specified radius.

Limitations of the Radius Neighbors Classifier:

- Sensitivity to the choice of radius: Selecting the optimal radius value can be a non-trivial task and requires tuning.
- Inefficiency on large datasets: For large datasets, computing distances to all objects can be computationally expensive.
- Dependence on data density: This method may be less efficient when data has non-uniform density in the feature space.

The Radius Neighbors Classifier is a valuable machine learning method in situations where object proximity is important, and class shapes can be complex. It can be applied in various domains, including image analysis, natural language processing, and others.

**2.4.1. Code for Creating a Radius Neighbors Classifier Model**

This code demonstrates the process of training a Radius Neighbors Classifier model on the Iris dataset, exporting it in the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_RadiusNeighborsClassifier.py
# The code demonstrates the process of training an Radius Neughbors model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023 MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Radius Neighbors Classifier model
radius_model = RadiusNeighborsClassifier(radius=1.0)

# train the model on the entire dataset
radius_model.fit(X, y)

# predict classes for the entire dataset
y_pred = radius_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Radius Neighbors Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(radius_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "radius_neighbors_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Radius Neighbors Classifier model in ONNX format:", accuracy_onnx)
```

Results of the script Iris\_RadiusNeighbors.py:

Python    Accuracy of Radius Neighbors Classifier model: 0.9733333333333334

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.94      0.98      0.96        50

Python               2       0.98      0.94      0.96        50

Python

Python        accuracy                           0.97       150

Python       macro avg       0.97      0.97      0.97       150

Python    weighted avg       0.97      0.97      0.97       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\radius\_neighbors\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: probabilities, Data Type: tensor(float), Shape: \[None, 3\]

Python

Python    Accuracy of Radius Neighbors Classifier model in ONNX format: 0.9733333333333334

The accuracy of the original model and the accuracy of the model exported in ONNX format are the same.

**2.4.2. MQL5 Code for Working with the Radius Neighbors Classifier Model**

```
//+------------------------------------------------------------------+
//|                               Iris_RadiusNeighborsClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "radius_neighbors_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   ulong input_shape[]= { batch_size, input_data.Range(1)};
   OnnxSetInputShape(model,0,input_shape);
//---
   int output1[];
   float output2[][3];
//---
   ArrayResize(output1,(int)batch_size);
   ArrayResize(output2,(int)batch_size);
//---
   ulong output_shape[]= {batch_size};
   OnnxSetOutputShape(model,0,output_shape);
//---
   ulong output_shape2[]= {batch_size,3};
   OnnxSetOutputShape(model,1,output_shape2);
//---
   bool res=OnnxRun(model,ONNX_DEBUG_LOGS,input_data,output1,output2);
//--- classes are ready in output1[k];
   if(res)
     {
      for(int k=0; k<(int)batch_size; k++)
         model_classes_id[k]=output1[k];
     }
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="RadiusNeighborsClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_RadiusNeighborsClassifier (EURUSD,H1)      model:RadiusNeighborsClassifier  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_RadiusNeighborsClassifier (EURUSD,H1)      model:RadiusNeighborsClassifier  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_RadiusNeighborsClassifier (EURUSD,H1)      model:RadiusNeighborsClassifier  sample=127 FAILED [class=1, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_RadiusNeighborsClassifier (EURUSD,H1)      model:RadiusNeighborsClassifier  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_RadiusNeighborsClassifier (EURUSD,H1)      model:RadiusNeighborsClassifier   correct results: 97.33%
Iris_RadiusNeighborsClassifier (EURUSD,H1)      model=RadiusNeighborsClassifier all samples accuracy=0.973333
Iris_RadiusNeighborsClassifier (EURUSD,H1)      model=RadiusNeighborsClassifier batch test accuracy=1.000000
```

The Radius Neighbors Classifier model showed an accuracy of 97.33% with 4 classification errors (samples 78, 107, 127, and 139).

The accuracy of the exported ONNX model on the full iris dataset is 97.33%, which matches the accuracy of the original model.

**2.4.3.ONNX Representation of the Radius Neighbors Classifier Model**

![Figure 18. ONNX Representation of the Radius Neighbors Classifier in Netron](https://c.mql5.com/2/58/onnx_4_radius_neighbors.png)

Figure 18. ONNX Representation of the Radius Neighbors Classifier in Netron

**Note on RidgeClassifier and RidgeClassifierCV Methods**

RidgeClassifier and RidgeClassifierCV are two classification methods based on Ridge Regression, but they differ in the way parameters are tuned and hyperparameters are automatically selected:

RidgeClassifier:

- RidgeClassifier is a classification method based on Ridge Regression, used for binary and multiclass classification tasks.
- In the case of multiclass classification, RidgeClassifier converts the task into multiple binary tasks (one vs. all) and builds a model for each.
- The regularization parameter alpha needs to be manually tuned by the user, meaning you have to choose the optimal alpha value through experiments or analysis of validation data.

RidgeClassifierCV:

- RidgeClassifierCV is an extension of RidgeClassifier that provides built-in support for cross-validation and automatic selection of the optimal regularization parameter alpha.
- Instead of manually setting alpha, you can provide RidgeClassifierCV with a list of alpha values to investigate and specify the cross-validation method (e.g., through the cv parameter).
- RidgeClassifierCV automatically selects the optimal alpha value that performs best during cross-validation.

So, the main difference between them lies in the level of automation in selecting the optimal value for the regularization parameter alpha. RidgeClassifier requires manual tuning of alpha, while RidgeClassifierCV allows for automatic selection of the optimal alpha value using cross-validation. The choice between them depends on your needs and your desire for automation in the model tuning process.

**2.5. Ridge Classifier**

Ridge Classifier is a variant of logistic regression that includes L2 regularization (Ridge Regression) in the model. L2 regularization adds a penalty to the large coefficients of the model, helping to reduce overfitting and improve the model's generalization ability.

Principles of Ridge Classifier:

1. Prediction of Probabilities: Like logistic regression, Ridge Classifier models the probability of an object belonging to a specific class using a logistic (sigmoid) function.
2. L2 Regularization: Ridge Classifier adds an L2 regularization term that penalizes large coefficients of the model. This is done to control the complexity of the model and reduce overfitting.
3. Parameter Training: The Ridge Classifier model is trained on the training dataset to adjust the weights (coefficients) for features and the regularization parameter.

Advantages of Ridge Classifier:

- Overfitting Reduction: L2 regularization helps reduce the model's tendency to overfit, which is especially useful when there is limited data.
- Handling Multicollinearity: Ridge Classifier handles multicollinearity issues well, where features are highly correlated with each other.

Limitations of Ridge Classifier:

- Sensitivity to the Choice of Regularization Parameter: Like other regularization methods, choosing the right value for the regularization parameter (alpha) requires tuning and assessment.
- Multiclass Classification Constraint: Ridge Classifier is initially designed for binary classification but can be adapted to multiclass classification using approaches like One-vs-All.

Ridge Classifier is a powerful machine learning method that combines the benefits of logistic regression with regularization to combat overfitting and improve the model's generalization ability. It finds applications in various fields where probabilistic classification and model complexity control are important.

**2.5.1. Ridge Classifier Model Creation Code**

This code demonstrates the process of training the Ridge Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_RidgeClassifier.py
# The code demonstrates the process of training Ridge Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Ridge Classifier model
ridge_model = RidgeClassifier()

# train the model on the entire dataset
ridge_model.fit(X, y)

# predict classes for the entire dataset
y_pred = ridge_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Ridge Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(ridge_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "ridge_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Ridge Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Ridge Classifier model: 0.8533333333333334

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.87      0.66      0.75        50

Python               2       0.73      0.90      0.80        50

Python

Python        accuracy                           0.85       150

Python       macro avg       0.86      0.85      0.85       150

Python    weighted avg       0.86      0.85      0.85       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\ridge\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: probabilities, Data Type: tensor(float), Shape: \[None, 3\]

Python

Python    Accuracy of Ridge Classifier model in ONNX format: 0.8533333333333334

**2.5.2. MQL5 Code for Working with the Ridge Classifier Model**

```
//+------------------------------------------------------------------+
//|                                         Iris_RidgeClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "ridge_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   ulong input_shape[]= { batch_size, input_data.Range(1)};
   OnnxSetInputShape(model,0,input_shape);
//---
   int output1[];
   float output2[][3];
//---
   ArrayResize(output1,(int)batch_size);
   ArrayResize(output2,(int)batch_size);
//---
   ulong output_shape[]= {batch_size};
   OnnxSetOutputShape(model,0,output_shape);
//---
   ulong output_shape2[]= {batch_size,3};
   OnnxSetOutputShape(model,1,output_shape2);
//---
   bool res=OnnxRun(model,ONNX_DEBUG_LOGS,input_data,output1,output2);
//--- classes are ready in output1[k];
   if(res)
     {
      for(int k=0; k<(int)batch_size; k++)
         model_classes_id[k]=output1[k];
     }
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="RidgeClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=51 FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=52 FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=53 FAILED [class=2, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=57 FAILED [class=2, true class=1] features=(6.30,3.30,4.70,1.60]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=62 FAILED [class=2, true class=1] features=(5.90,3.00,4.20,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=65 FAILED [class=2, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=66 FAILED [class=2, true class=1] features=(6.70,3.10,4.40,1.40]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=67 FAILED [class=2, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=76 FAILED [class=2, true class=1] features=(6.60,3.00,4.40,1.40]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=79 FAILED [class=2, true class=1] features=(6.00,2.90,4.50,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=86 FAILED [class=2, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=87 FAILED [class=2, true class=1] features=(6.70,3.10,4.70,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=89 FAILED [class=2, true class=1] features=(5.60,3.00,4.10,1.30]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=92 FAILED [class=2, true class=1] features=(6.10,3.00,4.60,1.40]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=109 FAILED [class=1, true class=2] features=(6.70,2.50,5.80,1.80]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier   correct results: 85.33%
Iris_RidgeClassifier (EURUSD,H1)        model=RidgeClassifier all samples accuracy=0.853333
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40)
Iris_RidgeClassifier (EURUSD,H1)        model:RidgeClassifier  FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50)
Iris_RidgeClassifier (EURUSD,H1)        model=RidgeClassifier batch test accuracy=0.000000
```

On the full iris dataset, the model demonstrated an accuracy of 85.33%, which corresponds to the accuracy of the original.

**2.5.3. ONNX Representation of the Ridge Classifier Model**

![Figure 19. ONNX Representation of the Ridge Classifier Model in Netron](https://c.mql5.com/2/58/onnx_5_ridge_classifier.png)

Figure 19. ONNX Representation of the Ridge Classifier Model in Netron

**2.6. RidgeClassifierCV**

RidgeClassifierCV Classification Method is a powerful algorithm for binary and multiclass classification based on Ridge Regression.

Principles of RidgeClassifierCV:

1. Linear Ridge Regression: RidgeClassifierCV is based on linear Ridge Regression. This method is a modification of linear regression where L2 regularization is added. Regularization helps control overfitting by reducing the magnitude of feature weights.
2. Binary and Multiclass Classification: RidgeClassifierCV can be used for both binary classification (when there are only two classes) and multiclass classification (when there are more than two classes). For multiclass classification, it converts the task into multiple binary tasks (one-vs-all) and builds a model for each.
3. Automatic Selection of Regularization Parameter: One of the key advantages of RidgeClassifierCV is its built-in support for cross-validation and automatic selection of the optimal regularization parameter alpha. Instead of manually tuning alpha, the method iterates over different alpha values and selects the best one based on cross-validation.
4. Handling Multicollinearity: Ridge regression handles multicollinearity issues well, where features are highly correlated with each other. Regularization allows controlling the contribution of each feature, making the model robust to correlated data.

Advantages of RidgeClassifierCV:

- Automatic Hyperparameter Selection: One of the significant advantages of RidgeClassifierCV is its ability to automatically select the optimal alpha value using cross-validation. This eliminates the need for experimenting with different alpha values and increases the likelihood of achieving good results.
- Overfitting Control: L2 regularization provided by RidgeClassifierCV helps control model complexity and reduces the risk of overfitting. This is especially important for tasks with limited data.
- Transparency and Interpretability: RidgeClassifierCV provides interpretable feature weights, allowing analysis of each feature's contribution to predictions and making feature importance conclusions.
- Efficiency: The method is highly efficient and can be applied to large datasets.

Limitations of RidgeClassifierCV:

- Linearity: RidgeClassifierCV assumes linear relationships between features and the target variable. If the data exhibits strong nonlinear relationships, the method may not be sufficiently accurate.
- Feature Scaling Sensitivity: The method is sensitive to feature scaling. It is recommended to standardize or normalize features before applying RidgeClassifierCV.
- Optimal Feature Selection: RidgeClassifierCV does not perform automatic feature selection, so you need to manually decide which features to include in the model.

The RidgeClassifierCV classification method is a powerful tool for binary and multiclass classification with automatic selection of the optimal regularization parameter. Its overfitting control, interpretability, and efficiency make it a popular choice for various classification tasks. However, it's important to keep in mind its limitations, especially the assumption of linear relationships between features and the target variable.

**2.6.1. RidgeClassifierCV Model Creation Code**

This code demonstrates the process of training the RidgeClassifierCV model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_RidgeClassifierCV.py
# The code demonstrates the process of training RidgeClassifierCV model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a RidgeClassifierCV model
ridge_classifier_cv_model = RidgeClassifierCV()

# train the model on the entire dataset
ridge_classifier_cv_model.fit(X, y)

# predict classes for the entire dataset
y_pred = ridge_classifier_cv_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of RidgeClassifierCV model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(ridge_classifier_cv_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "ridge_classifier_cv_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of RidgeClassifierCV model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of RidgeClassifierCV model: 0.8533333333333334

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.87      0.66      0.75        50

Python               2       0.73      0.90      0.80        50

Python

Python        accuracy                           0.85       150

Python       macro avg       0.86      0.85      0.85       150

Python    weighted avg       0.86      0.85      0.85       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\ridge\_classifier\_cv\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: probabilities, Data Type: tensor(float), Shape: \[None, 3\]

Python

Python    Accuracy of RidgeClassifierCV model in ONNX format: 0.8533333333333334

**2.6.2. MQL5 Code for Working with the RidgeClassifierCV Model**

```
//+------------------------------------------------------------------+
//|                                       Iris_RidgeClassifierCV.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "ridge_classifier_cv_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   ulong input_shape[]= { batch_size, input_data.Range(1)};
   OnnxSetInputShape(model,0,input_shape);
//---
   int output1[];
   float output2[][3];
//---
   ArrayResize(output1,(int)batch_size);
   ArrayResize(output2,(int)batch_size);
//---
   ulong output_shape[]= {batch_size};
   OnnxSetOutputShape(model,0,output_shape);
//---
   ulong output_shape2[]= {batch_size,3};
   OnnxSetOutputShape(model,1,output_shape2);
//---
   bool res=OnnxRun(model,ONNX_DEBUG_LOGS,input_data,output1,output2);
//--- classes are ready in output1[k];
   if(res)
     {
      for(int k=0; k<(int)batch_size; k++)
         model_classes_id[k]=output1[k];
     }
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="RidgeClassifierCV";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=51 FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=52 FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=53 FAILED [class=2, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=57 FAILED [class=2, true class=1] features=(6.30,3.30,4.70,1.60]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=62 FAILED [class=2, true class=1] features=(5.90,3.00,4.20,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=65 FAILED [class=2, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=66 FAILED [class=2, true class=1] features=(6.70,3.10,4.40,1.40]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=67 FAILED [class=2, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=76 FAILED [class=2, true class=1] features=(6.60,3.00,4.40,1.40]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=79 FAILED [class=2, true class=1] features=(6.00,2.90,4.50,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=86 FAILED [class=2, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=87 FAILED [class=2, true class=1] features=(6.70,3.10,4.70,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=89 FAILED [class=2, true class=1] features=(5.60,3.00,4.10,1.30]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=92 FAILED [class=2, true class=1] features=(6.10,3.00,4.60,1.40]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=109 FAILED [class=1, true class=2] features=(6.70,2.50,5.80,1.80]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV   correct results: 85.33%
Iris_RidgeClassifierCV (EURUSD,H1)      model=RidgeClassifierCV all samples accuracy=0.853333
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40)
Iris_RidgeClassifierCV (EURUSD,H1)      model:RidgeClassifierCV  FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50)
Iris_RidgeClassifierCV (EURUSD,H1)      model=RidgeClassifierCV batch test accuracy=0.000000
```

The ONNX model's performance also perfectly matches the performance of the original scikit-learn model (85.33%).

**2.6.3. ONNX Representation of the RidgeClassifierCV Model**

![Figure 20. ONNX Representation of the RidgeClassifierCV in Netron](https://c.mql5.com/2/58/onnx_6_ridgecv_classifier.png)

Figure 20. ONNX Representation of the RidgeClassifierCV in Netron

**2.7. Random Forest Classifier**

The Random Forest Classifier is an ensemble machine learning method based on constructing multiple decision trees and combining their results to improve classification quality. This method is extremely popular due to its effectiveness and ability to work with diverse data.

Principles of the Random Forest Classifier:

1. Bagging (Bootstrap Aggregating): Random Forest uses the bagging method, which involves creating multiple subsamples (bootstrap samples) from the training data with replacement. For each subsample, a separate decision tree is constructed.
2. Random Feature Selection: When building each tree, a random subset of features is selected from the entire set of features. This promotes diversity among trees and reduces correlations between them.
3. Voting: When classifying an object, each tree provides its own prediction, and the class that receives the majority of votes among all trees is chosen as the final model prediction.

Advantages of the Random Forest Classifier:

- High Accuracy: Random Forest typically achieves high classification accuracy by averaging the results of multiple trees.
- Ability to Handle Diverse Data: It works well with numerical and categorical features, as well as data of varying structures.
- Overfitting Resistance: Random Forest has built-in regularization, making it resistant to overfitting.
- Feature Importance: Random Forest can assess feature importance, helping data scientists and feature engineers better understand the data.

Limitations of the Random Forest Classifier:

- Computational Complexity: Training a Random Forest can be time-consuming, especially with a large number of trees and features.
- Interpretability Challenges: Due to a large number of trees and random feature selection, model interpretation can be challenging.
- No Guaranteed Outlier Robustness: Random Forest doesn't always provide robustness to data outliers.

The Random Forest Classifier is a powerful machine learning algorithm widely used in various fields, including biomedicine, financial analysis, and text data analysis. It excels at solving classification and regression tasks and has high generalization capability.

**2.7.1. Random Forest Classifier Model Creation Code**

This code demonstrates the process of training the Random Forest Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_RandomForestClassifier.py
# The code demonstrates the process of training Random Forest Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023,2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# train the model on the entire dataset
rf_model.fit(X, y)

# predict classes for the entire dataset
y_pred = rf_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Random Forest Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(rf_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "rf_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Random Forest Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Random Forest Classifier model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\rf\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Random Forest Classifier model in ONNX format: 1.0

The Random Forest Classifier model (and its ONNX version) solves the Fisher's Iris classification problem with 100% accuracy.

**2.7.2. MQL5 Code for Working with the Random Forest Classifier Model**

```
//+------------------------------------------------------------------+
//|                                  Iris_RandomForestClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "rf_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="RandomForestClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_RandomForestClassifier (EURUSD,H1) model:RandomForestClassifier   correct results: 100.00%
Iris_RandomForestClassifier (EURUSD,H1) model=RandomForestClassifier all samples accuracy=1.000000
Iris_RandomForestClassifier (EURUSD,H1) model=RandomForestClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full iris dataset is 100%, which matches the accuracy of the original model.

**2.7.3. ONNX representation of the Random Forest Classifier model**

![Figure 21. ONNX representation of the Random Forest Classifier model in Netron](https://c.mql5.com/2/58/onnx_7_rf_classifier.png)

Figure 21. ONNX representation of the Random Forest Classifier model in Netron

**2.8. Gradient Boosting Classifier**

Gradient boosting is one of the most powerful machine learning methods and finds applications in various domains, including data analysis, computer vision, natural language processing, and financial analysis, thanks to its high accuracy and ability to work with diverse data. The Gradient Boosting Classifier is an ensemble machine learning method that builds a composition of decision trees to solve classification tasks. This method is popular for its ability to achieve high accuracy and resistance to overfitting.

Principles of the Gradient Boosting Classifier:

1. Ensemble of Decision Trees: The Gradient Boosting Classifier constructs an ensemble of decision trees, where each tree aims to improve the predictions of the previous tree.
2. Gradient Descent: Gradient boosting uses gradient descent to optimize the loss function. It minimizes the classification error by computing the gradient of the loss function and updating predictions based on this gradient.
3. Tree Weighting: Each tree in the composition has a weight, and in the end, predictions from all trees are combined, considering their weights.

Advantages of the Gradient Boosting Classifier:

- High Accuracy: The Gradient Boosting Classifier typically provides high classification accuracy and is one of the most powerful machine learning methods.
- Overfitting Resistance: Thanks to the use of regularization and gradient descent, this method is resistant to overfitting, especially when tuning hyperparameters.
- Ability to Work with Different Data Types: The Gradient Boosting Classifier can handle various data types, including numerical and categorical features.

Limitations of the Gradient Boosting Classifier:

- Computational Complexity: Training the Gradient Boosting Classifier can be computationally intensive, especially with a large number of trees or deep trees.
- Interpretability Challenges: Due to the complexity of the composition of multiple trees, interpreting the results can be challenging.
- Not Always Suitable for Small Datasets: Gradient boosting usually requires a significant amount of data for effective operation and can be prone to overfitting on small datasets.

The Gradient Boosting Classifier is a powerful machine learning method often used in data analysis competitions, effectively solving various classification tasks. It can discover complex nonlinear relationships in data and exhibits good generalization when hyperparameters are properly tuned.

**2.8.1. Gradient Boosting Classifier Model Creation Code**

This code demonstrates the process of training the Gradient Boosting Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_GradientBoostingClassifier.py
# The code demonstrates the process of training Gradient Boostring Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Gradient Boosting Classifier model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# train the model on the entire dataset
gb_model.fit(X, y)

# predict classes for the entire dataset
y_pred = gb_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Gradient Boosting Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(gb_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "gb_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Gradient Boosting Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Gradient Boosting Classifier model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\gb\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Gradient Boosting Classifier model in ONNX format: 1.0

The accuracy of the exported ONNX model on the full iris dataset is 100%, which matches the accuracy of the original model.

**2.8.2. MQL5 Code for Working with the Gradient Boosting Classifier Model**

```
//+------------------------------------------------------------------+
//|                              Iris_GradientBoostingClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "gb_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="GradientBoostingClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_GradientBoostingClassifier (EURUSD,H1)     model:GradientBoostingClassifier   correct results: 100.00%
Iris_GradientBoostingClassifier (EURUSD,H1)     model=GradientBoostingClassifier all samples accuracy=1.000000
Iris_GradientBoostingClassifier (EURUSD,H1)     model=GradientBoostingClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full iris dataset is 100%, which matches the accuracy of the original model.

**2.8.3. ONNX representation of the Gradient Boosting Classifier Model**

![Figure 22. ONNX representation of the Gradient Boosting Classifier Model in Netron](https://c.mql5.com/2/58/onnx_8_gb_classifier.png)

Figure 22. ONNX representation of the Gradient Boosting Classifier Model in Netron

**2.9. Adaptive Boosting Classifier**

AdaBoost (Adaptive Boosting) Classifier is an ensemble machine learning method used to enhance classification by combining the results of multiple weak (e.g., decision trees) classifiers to create a stronger algorithm.

Principles of AdaBoost Classifier:

1. Ensemble of Weak Classifiers: AdaBoost starts with initializing the weights of each sample in the training set, assigning them equal initial values.
2. Training Weak Classifiers: AdaBoost then trains a weak classifier (e.g., a decision tree) on the training set considering the sample weights. This classifier attempts to correctly classify the samples.
3. Weight Redistribution: AdaBoost adjusts the sample weights, increasing the weights of incorrectly classified samples and decreasing the weights of correctly classified samples.
4. Composition Creation: AdaBoost repeats the process of training weak classifiers and redistributing weights multiple times. The results of these weak classifiers are then combined into a composition, with each classifier contributing based on its accuracy.

Advantages of AdaBoost Classifier:

- High Accuracy: AdaBoost typically provides high classification accuracy by combining several weak classifiers.
- Overfitting Resistance: AdaBoost has built-in regularization, making it resistant to overfitting.
- Ability to Work with Various Classifiers: AdaBoost can use different base classifiers, allowing adaptation to specific tasks.

Limitations of AdaBoost Classifier:

- Sensitivity to Outliers: AdaBoost can be sensitive to outliers in the data, as they may have a significant weight.
- Not Always Suitable for Complex Tasks: In some complex tasks, AdaBoost may require a large number of base classifiers to achieve good results.
- Dependence on the Quality of Base Classifiers: AdaBoost performs better when the base classifiers are better than random guessing.

AdaBoost Classifier is a powerful machine learning algorithm commonly used in practice to solve classification tasks. It is well-suited for both binary and multiclass problems and can be adapted to various base classifiers.

**2.9.1. Adaptive Boosting Classifier Model Creation Code**

This code demonstrates the process of training the Adaptive Boosting Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_AdaBoostClassifier.py
# The code demonstrates the process of training AdaBoost Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create an AdaBoost Classifier model
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# train the model on the entire dataset
adaboost_model.fit(X, y)

# predict classes for the entire dataset
y_pred = adaboost_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of AdaBoost Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(adaboost_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "adaboost_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of AdaBoost Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of AdaBoost Classifier model: 0.96

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.92      0.96      0.94        50

Python               2       0.96      0.92      0.94        50

Python

Python        accuracy                           0.96       150

Python       macro avg       0.96      0.96      0.96       150

Python    weighted avg       0.96      0.96      0.96       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\adaboost\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of AdaBoost Classifier model in ONNX format: 0.96

**2.9.2. MQL5 Code for Working with the Adaptive Boosting Classifier Model**

```
//+------------------------------------------------------------------+
//|                                      Iris_AdaBoostClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "adaboost_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="AdaBoostClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_AdaBoostClassifier (EURUSD,H1)     model:AdaBoostClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AdaBoostClassifier (EURUSD,H1)     model:AdaBoostClassifier  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AdaBoostClassifier (EURUSD,H1)     model:AdaBoostClassifier  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AdaBoostClassifier (EURUSD,H1)     model:AdaBoostClassifier  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AdaBoostClassifier (EURUSD,H1)     model:AdaBoostClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AdaBoostClassifier (EURUSD,H1)     model:AdaBoostClassifier  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_AdaBoostClassifier (EURUSD,H1)     model:AdaBoostClassifier   correct results: 96.00%
Iris_AdaBoostClassifier (EURUSD,H1)     model=AdaBoostClassifier all samples accuracy=0.960000
Iris_AdaBoostClassifier (EURUSD,H1)     model=AdaBoostClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 96%, which corresponds to the accuracy of the original model.

**2.9.3. ONNX representation of the Adaptive Boosting Classifier Model**

![Figure 23. ONNX representation of the Adaptive Boosting Classifier in Netron](https://c.mql5.com/2/58/onnx_9_adaboost_classifier.png)

Figure 23. ONNX representation of the Adaptive Boosting Classifier in Netron

**2.10. Bootstrap Aggregating Classifier**

The Bagging (Bootstrap Aggregating) Classifier is an ensemble machine learning method that is based on creating multiple random subsamples (bootstrap samples) from the training data and building separate models on each of them. The results are then combined to improve the model's generalization ability.

Principles of Bagging Classifier:

1. Creating Subsamples: Bagging starts by creating several random subsamples (bootstrap samples) from the training data with replacement. This means that the same samples may appear in multiple subsamples, and some samples may be omitted.
2. Training Base Models: On each subsample, a separate base model (e.g., a decision tree) is trained. Each model is trained independently of the others.
3. Aggregation of Results: After training all base models, the results of their predictions are combined to obtain the final prediction. In binary classification, this can be done through majority voting.

Advantages of Bagging Classifier:

- Reduced Variance: Bagging reduces the model's variance by averaging the results of multiple base models, which can lead to more stable and reliable predictions.
- Overfitting Reduction: Since each base model is trained on different subsamples, Bagging can reduce the model's tendency to overfit.
- Versatility: Bagging can use various base models, allowing adaptation to different data types and tasks.\

Limitations of Bagging Classifier:

- Doesn't Improve Bias: Bagging tends to reduce variance but doesn't address the model's bias. If base models tend to be biased (e.g., underfit), Bagging won't correct this issue.
- Not Always Suitable for Complex Tasks: In some complex tasks, Bagging may require a large number of base models to achieve good results.

Bagging Classifier is an effective machine learning method that can enhance the model's generalization ability and reduce overfitting. It is often used in combination with different base models to address various classification and regression tasks.

**2.10.1. Bootstrap Aggregating Classifier Model Creation Code**

This code demonstrates the process of training the Bootstrap Aggregating Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_BootstrapAggregatingClassifier.py
# The code demonstrates the process of training Bagging Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Bagging Classifier model with a Decision Tree base estimator
bagging_model = BaggingClassifier(n_estimators=100, random_state=42)

# train the model on the entire dataset
bagging_model.fit(X, y)

# predict classes for the entire dataset
y_pred = bagging_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Bagging Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(bagging_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "bagging_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Bagging Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Bagging Classifier model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\bagging\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Bagging Classifier model in ONNX format: 1.0

The Bootstrap Aggregating Classifier model (and its ONNX version) achieved 100% accuracy in classifying the Iris dataset.

**2.10.2. MQL5 Code for Working with the Bootstrap Aggregating Classifier Model**

```
//+------------------------------------------------------------------+
//|                          Iris_BootstrapAggregatingClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "bagging_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="BootstrapAggregatingClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_BootstrapAggregatingClassifier (EURUSD,H1) model:BootstrapAggregatingClassifier   correct results: 100.00%
Iris_BootstrapAggregatingClassifier (EURUSD,H1) model=BootstrapAggregatingClassifier all samples accuracy=1.000000
Iris_BootstrapAggregatingClassifier (EURUSD,H1) model=BootstrapAggregatingClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 100%, which is consistent with the accuracy of the original model.

**2.10.3. ONNX Representation of the Bootstrap Aggregating Classifier**

![Figure 24. ONNX Representation of the Bootstrap Aggregating Classifier in Netron](https://c.mql5.com/2/58/onnx_10_bagging_classifier.png)

Figure 24. ONNX Representation of the Bootstrap Aggregating Classifier in Netron

**2.11. K-Nearest Neighbors (K-NN) Classifier**

K-Nearest Neighbors (K-NN) Classifier is a machine learning method used to solve classification and regression tasks based on the similarity between data points. It operates on the principle that objects close to each other in a multi-dimensional feature space have similar characteristics and, therefore, may have similar class labels.

Principles of the K-NN Classifier:

1. Determining proximity: The K-NN classifier calculates the proximity between the object to be classified and other objects in the training dataset. This is often done using a distance metric, such as Euclidean distance or Manhattan distance.
2. Choosing the number of neighbors: The parameter K determines the number of nearest neighbors to be used for classifying an object. Typically, K is chosen based on the task and data.
3. Voting: K-NN uses majority voting among the K nearest neighbors to determine the class of the object. For example, if the majority of the K neighbors belong to class A, the object will also be classified as class A.

Advantages of the K-NN Classifier:

- Simplicity and intuitiveness: K-NN is a simple and intuitive method that is easy to understand and apply.
- Ability to work with different data types: K-NN can be used for various data types, including numerical, categorical, and text data.
- Adaptability to changing data: K-NN can quickly adapt to changes in the data, making it suitable for tasks with dynamic data.

Limitations of the K-NN Classifier:

- Sensitivity to the choice of K: Selecting the optimal K value can be a non-trivial task. A small K can lead to overfitting, while a large K can lead to underfitting.
- Sensitivity to feature scaling: K-NN is sensitive to feature scaling, so data normalization can be important.
- Computational complexity: For large datasets and a high number of features, computing distances between all pairs of objects can be computationally expensive.
- Lack of interpretability: K-NN results can be challenging to interpret, especially when K is large, and there is a lot of data.

K-NN Classifier is a machine learning method that can be useful in tasks where object proximity is essential, such as recommendation systems, text classification, and pattern recognition. It is well-suited for initial data analysis and rapid model prototyping.

**2.11.1. K-Nearest Neighbors (K-NN) Classifier Model Creation Code**

This code demonstrates the process of training a K-Nearest Neighbors (K-NN) Classifier model on the Iris dataset, exporting it in ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_KNearestNeighborsClassifier.py
# The code uses the K-Nearest Neighbors (KNN) Classifier for the Iris dataset, converts the model to ONNX format, saves it, and evaluates its accuracy.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a K-Nearest Neighbors (KNN) Classifier model
knn_model = KNeighborsClassifier(n_neighbors=3)

# train the model on the entire dataset
knn_model.fit(X, y)

# predict classes for the entire dataset
y_pred = knn_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of KNN Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(knn_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "knn_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of KNN Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of KNN Classifier model: 0.96

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.94      0.94      0.94        50

Python               2       0.94      0.94      0.94        50

Python

Python        accuracy                           0.96       150

Python       macro avg       0.96      0.96      0.96       150

Python    weighted avg       0.96      0.96      0.96       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\knn\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of KNN Classifier model in ONNX format: 0.96

**2.11.2. MQL5 Code for Working with the K-Nearest Neighbors (K-NN) Classifier Model**

```
//+------------------------------------------------------------------+
//|                             Iris_KNearestNeighborsClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "knn_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="KNearestNeighborsClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model:KNearestNeighborsClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model:KNearestNeighborsClassifier  sample=73 FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model:KNearestNeighborsClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model:KNearestNeighborsClassifier  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model:KNearestNeighborsClassifier  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model:KNearestNeighborsClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model:KNearestNeighborsClassifier   correct results: 96.00%
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model=KNearestNeighborsClassifier all samples accuracy=0.960000
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model:KNearestNeighborsClassifier  FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50)
Iris_KNearestNeighborsClassifier (EURUSD,H1)    model=KNearestNeighborsClassifier batch test accuracy=0.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 96%, which is consistent with the accuracy of the original model.

**2.11.3. ONNX Representation of the K-Nearest Neighbors (K-NN) Classifier**

![Figure 25. ONNX Representation of the K-Nearest Neighbors in Netron](https://c.mql5.com/2/58/onnx_11_knn_classifier.png)

Figure 25. ONNX Representation of the K-Nearest Neighbors in Netron

**2.12. Decision Tree Classifier**

Decision Tree Classifier is a machine learning method used for classification tasks based on the construction of a decision tree. This method divides the dataset into smaller subgroups by performing a series of conditional tests on features and determines the class of an object based on the path it follows in the tree.

Principles of the Decision Tree Classifier:

1. Building the decision tree: Initially, all data is represented at the root of the tree. For each node of the tree, data is split into two or more subgroups based on the values of one feature, aiming to minimize uncertainty (e.g., entropy or Gini index) in each subgroup.
2. Recursive construction: The process of splitting data is performed recursively until the tree reaches its leaves. The leaves represent the final classes of objects.
3. Decision-making: When an object enters the tree, it follows a path from the root to one of the leaves, where its class is determined based on the majority of objects in that leaf.

Advantages of Decision Tree Classifier:

- Interpretability: Decision trees are easy to interpret and visualize. The decision rules used for classification are understandable.
- Handling different data types: Decision Tree Classifier can work with both numerical and categorical features.
- Feature importance: Decision trees can assess feature importance, helping data analysts and feature engineers understand the data.

Limitations of Decision Tree Classifier:

- Overfitting: Large and deep trees can be prone to overfitting, making them less generalizable to new data.
- Sensitivity to noise: Decision trees can be sensitive to noise and outliers in the data.
- Greedy construction: Decision trees are built using a greedy algorithm, which can lead to suboptimal global solutions.
- Instability to data changes: Minor changes in the data can lead to significant changes in the tree structure.

Decision Tree Classifier is a useful machine learning method for classification tasks, especially in situations where model interpretability is essential, and you need to understand which features influence the decision. This method can also be used in ensemble methods like Random Forest and Gradient Boosting.

**2.12.1. Code for Creating the Decision Tree Classifier Model**

This code demonstrates the process of training a Decision Tree Classifier model on the Iris dataset, exporting it in ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_DecisionTreeClassifier.py
# The code uses the Decision Tree Classifier for the Iris dataset, converts the model to ONNX format, saves it, and evaluates its accuracy.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Decision Tree Classifier model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# train the model on the entire dataset
decision_tree_model.fit(X, y)

# predict classes for the entire dataset
y_pred = decision_tree_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Decision Tree Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(decision_tree_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "decision_tree_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Decision Tree Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Decision Tree Classifier model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\decision\_tree\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Decision Tree Classifier model in ONNX format: 1.0

The Decision Tree Classifier model (and its ONNX version) demonstrated 100% accuracy in classifying the entire Fisher's iris dataset.

**2.12.2. MQL5 Code for Working with the Decision Tree Classifier Model**

```
//+------------------------------------------------------------------+
//|                                  Iris_DecisionTreeClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "decision_tree_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="DecisionTreeClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_DecisionTreeClassifier (EURUSD,H1) model:DecisionTreeClassifier   correct results: 100.00%
Iris_DecisionTreeClassifier (EURUSD,H1) model=DecisionTreeClassifier all samples accuracy=1.000000
Iris_DecisionTreeClassifier (EURUSD,H1) model=DecisionTreeClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 100%, which matches the accuracy of the original model.

**2.12.3. ONNX representation of the Decision Tree Classifier**

![Figure 26. ONNX representation of the Decision Tree Classifier in Netron](https://c.mql5.com/2/58/onnx_12_decision_tree_classifier.png)

Figure 26. ONNX representation of the Decision Tree Classifier in Netron

**Note about LogisticRegression and LogisticRegressionCV:**

LogisticRegression and LogisticRegressionCV are two classifiers used for binary classification using logistic regression, but they differ in how the model parameters are tuned:

    LogisticRegression:

- LogisticRegression is a classifier that uses the logistic function to model the probability of belonging to one of two classes (binary classification).
- It provides basic parameters for customization, such as C (inverse regularization strength), penalty (type of regularization, e.g., L1 or L2), solver (optimization algorithm), and others.
- When using LogisticRegression, you typically manually choose parameter values and their combinations and then train the model on the data.

    LogisticRegressionCV:

- LogisticRegressionCV is an extension of LogisticRegression that provides built-in support for cross-validation and selecting the optimal value of the regularization parameter C.
- Instead of manually selecting C, you can pass LogisticRegressionCV a list of C values to explore and specify the cross-validation method (e.g., through the cv parameter).
- LogisticRegressionCV automatically selects the optimal C value that performs best in cross-validation.
- This is convenient when you need to automatically tune the regularization, especially if you have a lot of data or are unsure about which C value to choose.

So, the main difference between them lies in the level of automation in parameter tuning. LogisticRegression requires manual tuning of C, while LogisticRegressionCV allows for the automatic selection of the optimal C value through cross-validation. The choice between them depends on your needs and the desire for automation in the model tuning process.

**2.13. Logistic Regression Classifier**

Logistic Regression Classifier is a machine learning method used for binary and multi-class classification tasks. Despite its name, "regression," logistic regression actually predicts the probability of an object belonging to one of the classes. Based on these probabilities, the final classification decision is made.

Principles of Logistic Regression Classifier:

1. Probability Prediction: Logistic regression models the probability of an object belonging to a specific class using the logistic (sigmoid) function.
2. Decision Boundary: Based on the predicted probabilities, logistic regression determines the decision boundary that separates the classes. If the probability exceeds a certain threshold (typically 0.5), the object is classified into one class; otherwise, it's classified into another class.
3. Parameter Learning: The logistic regression model is trained on a training dataset by adjusting the weights (coefficients) associated with the features to minimize the loss function.

Advantages of Logistic Regression Classifier:

- Simplicity and Interpretability: Logistic regression is a straightforward model with easily interpretable results regarding the influence of features on class predictions.
- Efficiency with Large Datasets: Logistic regression can efficiently handle large datasets and train on them quickly.
- Usage in Ensemble Methods: Logistic regression can serve as a base classifier in ensemble methods such as stacking.

Limitations of Logistic Regression Classifier:

- Linearity: Logistic regression assumes a linear relationship between features and the logarithm of odds, which may be inadequate for complex tasks.
- Multi-Class Constraint: In its original form, logistic regression is designed for binary classification, but there are methods like One-vs-All (One-vs-Rest) for extending it to multi-class classification.
- Sensitivity to Outliers: Logistic regression can be sensitive to outliers in the data.

Logistic regression is a classic machine learning method widely used in practice for classification tasks, especially when interpretability of the model is important, and data exhibit a linear or near-linear structure. It is also used in statistics and medical data analysis to assess the impact of factors on the likelihood of events.

**2.13.1. Code for Creating a Logistic Regression Classifier Model**

This code demonstrates the process of training a Logistic Regression Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_LogisticRegressionClassifier.py
# The code uses the Logistic Regression Classifier for the Iris dataset, converts the model to ONNX format, saves it, and evaluates its accuracy.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Logistic Regression Classifier model
logistic_regression_model = LogisticRegression(max_iter=1000, random_state=42)

# train the model on the entire dataset
logistic_regression_model.fit(X, y)

# predict classes for the entire dataset
y_pred = logistic_regression_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Logistic Regression Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(logistic_regression_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "logistic_regression_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Logistic Regression Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Logistic Regression Classifier model: 0.9733333333333334

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.98      0.94      0.96        50

Python               2       0.94      0.98      0.96        50

Python

Python        accuracy                           0.97       150

Python       macro avg       0.97      0.97      0.97       150

Python    weighted avg       0.97      0.97      0.97       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\logistic\_regression\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Logistic Regression Classifier model in ONNX format: 0.9733333333333334

**2.13.2. MQL5 Code for Working with the Regression Classifier Model**

```
//+------------------------------------------------------------------+
//|                            Iris_LogisticRegressionClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "logistic_regression_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="LogisticRegressionClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_LogisticRegressionClassifier (EURUSD,H1)   model:LogisticRegressionClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_LogisticRegressionClassifier (EURUSD,H1)   model:LogisticRegressionClassifier  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_LogisticRegressionClassifier (EURUSD,H1)   model:LogisticRegressionClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_LogisticRegressionClassifier (EURUSD,H1)   model:LogisticRegressionClassifier  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_LogisticRegressionClassifier (EURUSD,H1)   model:LogisticRegressionClassifier   correct results: 97.33%
Iris_LogisticRegressionClassifier (EURUSD,H1)   model=LogisticRegressionClassifier all samples accuracy=0.973333
Iris_LogisticRegressionClassifier (EURUSD,H1)   model=LogisticRegressionClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 97.33%, which corresponds to the accuracy of the original model.

**2.13.3. ONNX Representation of the Logistic Regression Classifier**

![Figure 27. ONNX Representation of the Logistic Regression Classifier in Netron](https://c.mql5.com/2/58/onnx_13_logistic_regression_classifier.png)

Figure 27. ONNX Representation of the Logistic Regression Classifier in Netron

**2.14. LogisticRegressionCV Classifier**

LogisticRegressionCV (Logistic Regression with Cross-Validation) is a powerful and flexible method for binary classification. This method not only allows you to create classification models based on logistic regression but also automatically tunes parameters to achieve the best performance.

Working Principles of LogisticRegressionCV:

1. Logistic Regression: LogisticRegressionCV is fundamentally based on logistic regression. Logistic regression is a statistical method used to model the probability of an object belonging to one of two classes. This model is applied when the dependent variable is binary (two classes) or can be transformed into binary.
2. Cross-Validation: The key advantage of LogisticRegressionCV is its integrated cross-validation. This means that instead of manually selecting the optimal value for the regularization parameter C, the method automatically tries different C values and selects the one that performs best in cross-validation.
3. Choosing the Optimal C: LogisticRegressionCV employs a cross-validation strategy to assess the model's performance at different C values. C is the regularization parameter controlling the extent of model regularization. A small C value indicates strong regularization, while a large C value indicates weak regularization. Cross-validation helps select the optimal C value to balance underfitting and overfitting.
4. Regularization: LogisticRegressionCV also supports various types of regularization, including L1 (lasso) and L2 (ridge) regularization. These regularization types help improve the model's generalization and prevent overfitting.

Advantages of LogisticRegressionCV:

    Automatic Parameter Tuning: One of the primary advantages of LogisticRegressionCV is its ability to automatically choose the optimal C value using cross-validation. This eliminates the need for manual model tuning and allows you to focus on the data and the task.

    Overfitting Robustness: The regularization supported by LogisticRegressionCV helps control the model's complexity and reduces the risk of overfitting, especially with limited data.

    Transparency: Logistic regression is an interpretable method. You can analyze the contribution of each feature to the prediction, which is useful for understanding feature importance.

    High Performance: Logistic regression can work quickly and efficiently, particularly with a large volume of data.

Limitations of LogisticRegressionCV:

    Linear Dependencies: LogisticRegressionCV is suitable for solving linear and near-linear classification problems. If the relationship between features and the target variable is highly nonlinear, the model may not perform well.

    Handling a Large Number of Features: With a large number of features, logistic regression may require substantial data or dimensionality reduction techniques to prevent overfitting.

    Data Representation Dependence: The effectiveness of logistic regression can depend on how the data is represented and which features are chosen.

LogisticRegressionCV is a powerful tool for binary classification with automatic parameter tuning and overfitting robustness. It is especially useful when you need to quickly build an interpretable classification model. However, it is important to remember that it performs best in cases where data exhibits linear or near-linear dependencies.

**2.14.1. Code for Creating the LogisticRegressionCV Classifier Model**

This code demonstrates the process of training a LogisticRegressionCV Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_LogisticRegressionCVClassifier.py
# The code demonstrates the process of training LogisticRegressionCV model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a LogisticRegressionCV model
logistic_regression_model = LogisticRegressionCV(cv=5, max_iter=1000)

# train the model on the entire dataset
logistic_regression_model.fit(X, y)

# predict classes for the entire dataset
y_pred = logistic_regression_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of LogisticRegressionCV model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(logistic_regression_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "logistic_regressioncv_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of LogisticRegressionCV model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of LogisticRegressionCV model: 0.98

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.98      0.96      0.97        50

Python               2       0.96      0.98      0.97        50

Python

Python        accuracy                           0.98       150

Python       macro avg       0.98      0.98      0.98       150

Python    weighted avg       0.98      0.98      0.98       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\logistic\_regression\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of LogisticRegressionCV model in ONNX format: 0.98

**2.14.2. MQL5 Code for Working with the LogisticRegressionCV Classifier Model**

```
//+------------------------------------------------------------------+
//|                          Iris_LogisticRegressionCVClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "logistic_regressioncv_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="LogisticRegressionCVClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_LogisticRegressionCVClassifier (EURUSD,H1) model:LogisticRegressionCVClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_LogisticRegressionCVClassifier (EURUSD,H1) model:LogisticRegressionCVClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_LogisticRegressionCVClassifier (EURUSD,H1) model:LogisticRegressionCVClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_LogisticRegressionCVClassifier (EURUSD,H1) model:LogisticRegressionCVClassifier   correct results: 98.00%
Iris_LogisticRegressionCVClassifier (EURUSD,H1) model=LogisticRegressionCVClassifier all samples accuracy=0.980000
Iris_LogisticRegressionCVClassifier (EURUSD,H1) model=LogisticRegressionCVClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 98%, which corresponds to the accuracy of the original model.

**2.14.3. ONNX Representation of the LogisticRegressionCV Classifier**

![Figure 28. ONNX Representation of the LogisticRegressionCV Classifier in Netron](https://c.mql5.com/2/58/onnx_14_logistic_regression_cv_classifier.png)

Figure 28. ONNX Representation of the LogisticRegressionCV Classifier in Netron

**2.15. Passive-Aggressive (PA) Classifier**

The Passive-Aggressive (PA) Classifier is a machine learning method used for classification tasks. The core idea of this method is to adapt the model's weights (coefficients) during training to minimize classification errors. The Passive-Aggressive Classifier can be useful in online learning scenarios and situations where data changes over time.

Working Principles of the Passive-Aggressive Classifier:

1. Weight Adaptation: Instead of updating the model's weights in the direction of minimizing the loss function, as done in stochastic gradient descent, the Passive-Aggressive Classifier adapts the weights in the direction that minimizes the classification error for the current example.
2. Maintaining Aggressiveness: The method has a parameter called aggressiveness (C), which determines how strongly the model's weights should be adapted. Larger C values make the method more aggressive in adaptation, while smaller values make it less aggressive.

Advantages of the Passive-Aggressive Classifier:

- Suitable for Online Learning: The Passive-Aggressive Classifier can be updated as new data arrives, making it suitable for online learning tasks where data arrives in a stream.
- Adaptability to Data Changes: The method performs well with changing data since it adapts the model to new circumstances.

Limitations of the Passive-Aggressive Classifier:

- Sensitivity to Aggressiveness Parameter Choice: Selecting the optimal value for the aggressiveness parameter C may require tuning and depends on data characteristics.
- Not Always Suitable for Complex Tasks: The Passive-Aggressive Classifier may not provide high accuracy in complex tasks where intricate feature dependencies need to be considered.
- Interpretation of Weights: Model weights obtained using this method may be less interpretable compared to weights obtained using linear or logistic regression.

The Passive-Aggressive Classifier is a machine learning method suitable for classification tasks with evolving data and situations where rapid adaptation of the model to new circumstances is crucial. It finds applications in various domains, including text data analysis, image classification, and other tasks.

**2.15.1. Code for Creating the Passive-Aggressive (PA) Classifier Model**

This code demonstrates the process of training a Passive-Aggressive (PA) Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_PassiveAgressiveClassifier.py
# The code uses the Passive-Aggressive (PA) Classifier for the Iris dataset, converts the model to ONNX format, saves it, and evaluates its accuracy.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Passive-Aggressive (PA) Classifier model
pa_classifier_model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)

# train the model on the entire dataset
pa_classifier_model.fit(X, y)

# predict classes for the entire dataset
y_pred = pa_classifier_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Passive-Aggressive (PA) Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(pa_classifier_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "pa_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Passive-Aggressive (PA) Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Passive-Aggressive (PA) Classifier model: 0.96

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.96      0.92      0.94        50

Python               2       0.92      0.96      0.94        50

Python

Python        accuracy                           0.96       150

Python       macro avg       0.96      0.96      0.96       150

Python    weighted avg       0.96      0.96      0.96       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\pa\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Passive-Aggressive (PA) Classifier model in ONNX format: 0.96

**2.15.2. MQL5 Code for Working with the Passive-Aggressive (PA) Classifier Model**

```
//+------------------------------------------------------------------+
//|                              Iris_PassiveAgressiveClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "pa_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="PassiveAgressiveClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model:PassiveAgressiveClassifier  sample=67 FAILED [class=2, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model:PassiveAgressiveClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model:PassiveAgressiveClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model:PassiveAgressiveClassifier  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model:PassiveAgressiveClassifier  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model:PassiveAgressiveClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model:PassiveAgressiveClassifier   correct results: 96.00%
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model=PassiveAgressiveClassifier all samples accuracy=0.960000
Iris_PassiveAgressiveClassifier (EURUSD,H1)     model=PassiveAgressiveClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 96%, which corresponds to the accuracy of the original model.

**2.15.3. ONNX Representation of the Passive-Aggressive (PA) Classifier**

![Figure 29. ONNX Representation of the Passive-Aggressive (PA) Classifier in Netron](https://c.mql5.com/2/58/onnx_15_passive-agressive_classifier.png)

Figure 29. ONNX Representation of the Passive-Aggressive (PA) Classifier in Netron

**2.16. Perceptron Classifier**

The Perceptron Classifier is a linear binary classifier used to separate two classes based on a linear separating hyperplane. It is one of the simplest and oldest machine learning methods, and its core principle is to train the model's weights (coefficients) to maximize the classification accuracy on the training dataset.

Working Principles of the Perceptron Classifier:

1. Linear Hyperplane: The Perceptron constructs a linear hyperplane in the feature space that separates two classes. This hyperplane is determined by the model's weights (coefficients).
2. Weight Training: Initially, the weights are initialized randomly or to zeros. Then, for each object in the training dataset, the model predicts the class based on the current weights and adjusts them in case of an error. Training continues until all objects are classified correctly or until a maximum number of iterations is reached.

Advantages of the Perceptron Classifier:

- Simplicity: The Perceptron is a very simple algorithm, easy to understand and implement.
- High Training Speed: The Perceptron can be trained quickly, especially on large datasets, and can be used in online learning tasks.

Limitations of the Perceptron Classifier:

- Linear Separability Constraint: The Perceptron works well only in cases where data is linearly separable. If data cannot be separated linearly, the Perceptron may not achieve high accuracy.
- Sensitivity to Initial Weights: The initial choice of weights can impact the convergence of the algorithm. Poor initial weight choices may lead to slow convergence or a neuron that cannot correctly separate the classes.
- Inability to Determine Probabilities: The Perceptron does not provide probability estimates of class membership, which can be important for certain tasks.

The Perceptron Classifier is a basic algorithm for binary classification that can be useful in simple cases where data is linearly separable. It can also serve as a foundation for more complex methods, such as multilayer neural networks. It's important to remember that in more complex tasks where data has intricate structures, other methods like logistic regression or Support Vector Machines (SVM) may provide higher classification accuracy.

**2.16.1. Code for Creating the Perceptron Classifier Model**

This code demonstrates the process of training a Perceptron Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_PerceptronClassifier.py
# The code demonstrates the process of training Perceptron Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Perceptron Classifier model
perceptron_model = Perceptron(max_iter=1000, random_state=42)

# train the model on the entire dataset
perceptron_model.fit(X, y)

# predict classes for the entire dataset
y_pred = perceptron_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Perceptron Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(perceptron_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "perceptron_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Perceptron Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Perceptron Classifier model: 0.6133333333333333

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      0.80      0.89        50

Python               1       0.46      1.00      0.63        50

Python               2       1.00      0.04      0.08        50

Python

Python        accuracy                           0.61       150

Python       macro avg       0.82      0.61      0.53       150

Python    weighted avg       0.82      0.61      0.53       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\perceptron\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Perceptron Classifier model in ONNX format: 0.6133333333333333

**2.16.2. MQL5 Code for Working with the Perceptron Classifier Model**

```
//+------------------------------------------------------------------+
//|                                    Iris_PerceptronClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include "iris.mqh"
#resource "perceptron_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="PerceptronClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=2 FAILED [class=1, true class=0] features=(4.90,3.00,1.40,0.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=9 FAILED [class=1, true class=0] features=(4.40,2.90,1.40,0.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=10 FAILED [class=1, true class=0] features=(4.90,3.10,1.50,0.10]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=13 FAILED [class=1, true class=0] features=(4.80,3.00,1.40,0.10]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=21 FAILED [class=1, true class=0] features=(5.40,3.40,1.70,0.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=26 FAILED [class=1, true class=0] features=(5.00,3.00,1.60,0.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=31 FAILED [class=1, true class=0] features=(4.80,3.10,1.60,0.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=35 FAILED [class=1, true class=0] features=(4.90,3.10,1.50,0.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=42 FAILED [class=1, true class=0] features=(4.50,2.30,1.30,0.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=46 FAILED [class=1, true class=0] features=(4.80,3.00,1.40,0.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=102 FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=103 FAILED [class=1, true class=2] features=(7.10,3.00,5.90,2.10]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=104 FAILED [class=1, true class=2] features=(6.30,2.90,5.60,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=105 FAILED [class=1, true class=2] features=(6.50,3.00,5.80,2.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=106 FAILED [class=1, true class=2] features=(7.60,3.00,6.60,2.10]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=108 FAILED [class=1, true class=2] features=(7.30,2.90,6.30,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=109 FAILED [class=1, true class=2] features=(6.70,2.50,5.80,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=110 FAILED [class=1, true class=2] features=(7.20,3.60,6.10,2.50]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=111 FAILED [class=1, true class=2] features=(6.50,3.20,5.10,2.00]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=112 FAILED [class=1, true class=2] features=(6.40,2.70,5.30,1.90]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=113 FAILED [class=1, true class=2] features=(6.80,3.00,5.50,2.10]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=114 FAILED [class=1, true class=2] features=(5.70,2.50,5.00,2.00]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=116 FAILED [class=1, true class=2] features=(6.40,3.20,5.30,2.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=117 FAILED [class=1, true class=2] features=(6.50,3.00,5.50,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=118 FAILED [class=1, true class=2] features=(7.70,3.80,6.70,2.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=119 FAILED [class=1, true class=2] features=(7.70,2.60,6.90,2.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=121 FAILED [class=1, true class=2] features=(6.90,3.20,5.70,2.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=122 FAILED [class=1, true class=2] features=(5.60,2.80,4.90,2.00]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=123 FAILED [class=1, true class=2] features=(7.70,2.80,6.70,2.00]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=124 FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=125 FAILED [class=1, true class=2] features=(6.70,3.30,5.70,2.10]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=126 FAILED [class=1, true class=2] features=(7.20,3.20,6.00,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=127 FAILED [class=1, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=128 FAILED [class=1, true class=2] features=(6.10,3.00,4.90,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=129 FAILED [class=1, true class=2] features=(6.40,2.80,5.60,2.10]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=131 FAILED [class=1, true class=2] features=(7.40,2.80,6.10,1.90]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=132 FAILED [class=1, true class=2] features=(7.90,3.80,6.40,2.00]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=133 FAILED [class=1, true class=2] features=(6.40,2.80,5.60,2.20]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=136 FAILED [class=1, true class=2] features=(7.70,3.00,6.10,2.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=137 FAILED [class=1, true class=2] features=(6.30,3.40,5.60,2.40]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=138 FAILED [class=1, true class=2] features=(6.40,3.10,5.50,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=140 FAILED [class=1, true class=2] features=(6.90,3.10,5.40,2.10]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=141 FAILED [class=1, true class=2] features=(6.70,3.10,5.60,2.40]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=142 FAILED [class=1, true class=2] features=(6.90,3.10,5.10,2.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=143 FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=144 FAILED [class=1, true class=2] features=(6.80,3.20,5.90,2.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=145 FAILED [class=1, true class=2] features=(6.70,3.30,5.70,2.50]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=146 FAILED [class=1, true class=2] features=(6.70,3.00,5.20,2.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=147 FAILED [class=1, true class=2] features=(6.30,2.50,5.00,1.90]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=148 FAILED [class=1, true class=2] features=(6.50,3.00,5.20,2.00]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=149 FAILED [class=1, true class=2] features=(6.20,3.40,5.40,2.30]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  sample=150 FAILED [class=1, true class=2] features=(5.90,3.00,5.10,1.80]
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier   correct results: 61.33%
Iris_PerceptronClassifier (EURUSD,H1)   model=PerceptronClassifier all samples accuracy=0.613333
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80)
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  FAILED [class=1, true class=0] features=(4.90,3.10,1.50,0.10)
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90)
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  FAILED [class=1, true class=2] features=(7.10,3.00,5.90,2.10)
Iris_PerceptronClassifier (EURUSD,H1)   model:PerceptronClassifier  FAILED [class=1, true class=2] features=(6.30,2.90,5.60,1.80)
Iris_PerceptronClassifier (EURUSD,H1)   model=PerceptronClassifier batch test accuracy=0.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 61.33%, which corresponds to the accuracy of the original model.

**2.16.3. ONNX Representation of the Perceptron Classifier**

![Figure 30. ONNX Representation of the Perceptron Classifier in Netron](https://c.mql5.com/2/58/onnx_16_perceptron_classifier.png)

Figure 30. ONNX Representation of the Perceptron Classifier in Netron

**2.17. Stochastic Gradient Descent Classifier**

The SGD Classifier (Stochastic Gradient Descent Classifier) is a machine learning method used for classification tasks. It is a specific case of linear models and is a linear classifier that is trained using stochastic gradient descent.

Principles of the SGD Classifier:

1. Linear Hyperplane: The SGD Classifier constructs a linear hyperplane in the multi-dimensional feature space that separates two classes. This hyperplane is determined by the model's weights (coefficients).
2. Stochastic Gradient Descent: The method is trained using stochastic gradient descent, which means that weight updates are performed on each object in the training dataset (or a randomly selected subset), rather than the entire dataset. This makes the SGD Classifier suitable for large volumes of data and online learning.
3. Loss Function: SGD Classifier optimizes a loss function, such as the logistic loss function for binary classification or the softmax loss function for multi-class classification.

Advantages of the SGD Classifier:

- Training Speed: The SGD Classifier trains quickly, especially on large volumes of data, thanks to stochastic gradient descent.
- Suitable for Online Learning: The method is well-suited for online learning tasks where data arrives in a streaming fashion, and the model needs to be updated as new data comes in.

Limitations of the SGD Classifier:

- Sensitivity to Parameters: The SGD Classifier has many hyperparameters, such as the learning rate and regularization parameter, which require careful tuning.
- Weight Initialization: The initial choice of weights can influence convergence and model quality.
- Convergence to Local Minima: Due to the stochastic nature of the SGD method, it can converge to local minima of the loss function, which can affect model quality.

The SGD Classifier is a versatile machine learning method that can be used for binary and multi-class classification tasks, especially when dealing with large volumes of data that require fast processing. It's important to properly tune its hyperparameters and monitor convergence to achieve high classification accuracy.

**2.17.1. Code for Creating the Stochastic Gradient Descent Classifier Model**

This code demonstrates the process of training an SGD Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_SGDClassifier.py
# The code demonstrates the process of training Stochastic Gradient Descent Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create an SGD Classifier model
sgd_model = SGDClassifier(max_iter=1000, random_state=42)

# train the model on the entire dataset
sgd_model.fit(X, y)

# predict classes for the entire dataset
y_pred = sgd_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of SGD Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(sgd_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "sgd_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of SGD Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of SGD Classifier model: 0.9333333333333333

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       0.96      1.00      0.98        50

Python               1       0.88      0.92      0.90        50

Python               2       0.96      0.88      0.92        50

Python

Python        accuracy                           0.93       150

Python       macro avg       0.93      0.93      0.93       150

Python    weighted avg       0.93      0.93      0.93       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\perceptron\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of SGD Classifier model in ONNX format: 0.9333333333333333

**2.17.2. MQL5 Code for Working with the Stochastic Gradient Descent Classifier Model**

```
//+------------------------------------------------------------------+
//|                                           Iris_SGDClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "sgd_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="SGDClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=65 FAILED [class=0, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=86 FAILED [class=0, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=124 FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=127 FAILED [class=1, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier   correct results: 93.33%
Iris_SGDClassifier (EURUSD,H1)  model=SGDClassifier all samples accuracy=0.933333
Iris_SGDClassifier (EURUSD,H1)  model:SGDClassifier  FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80)
Iris_SGDClassifier (EURUSD,H1)  model=SGDClassifier batch test accuracy=0.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 93.33%, which corresponds to the accuracy of the original model.

**2.17.3. ONNX Representation of the Stochastic Gradient Descent Classifier**

![Figure 31. ONNX Representation of the Stochastic Gradient Descent Classifier in Netron](https://c.mql5.com/2/58/onnx_17_sgd_classifier.png)

Figure 31. ONNX Representation of the Stochastic Gradient Descent Classifier in Netron

**2.18. Gaussian Naive Bayes (GNB) Classifier**

The Gaussian Naive Bayes (GNB) Classifier is a machine learning method based on a Bayesian probabilistic model used for classification tasks. It is part of the family of naive Bayes classifiers and assumes that all features are independent and have a normal distribution.

Principles of the Gaussian Naive Bayes Classifier:

1. Bayesian Approach: GNB is based on the Bayesian approach to classification, which uses Bayes' theorem to compute the probability of an object belonging to each class.
2. Naive Assumption: The key assumption in GNB is that all features are independent and follow a normal (Gaussian) distribution. This assumption is considered naive because, in real-world data, features often correlate with each other.
3. Parameter Estimation: The GNB model is trained on the training dataset by computing the parameters of the distribution (mean and standard deviation) for each feature within each class.

Advantages of the Gaussian Naive Bayes Classifier:

- Simplicity and Training Speed: GNB is a very simple algorithm and trains quickly, even on large datasets.
- Effectiveness for Small and Medium-sized Data: GNB can be effective for classification tasks with a small or medium number of features, especially when the assumption of normal feature distributions holds.

Limitations of the Gaussian Naive Bayes Classifier:

- Naive Assumption: The assumption of feature independence and normal distribution can be overly simplistic and incorrect for real-world data, leading to reduced classification accuracy.
- Sensitivity to Outliers: GNB can be sensitive to outliers in the data as they can significantly skew the parameters of the normal distribution.
- Inability to Capture Feature Dependencies: Due to the independence assumption, GNB does not account for dependencies between features.

The Gaussian Naive Bayes Classifier is a good choice for simple classification tasks, particularly when the assumption of normal feature distributions is approximately met. However, in more complex tasks where features are correlated or do not follow a normal distribution, other methods like Support Vector Machines (SVM) or gradient boosting may provide more accurate results.

**2.18.1. Code for Creating the Gaussian Naive Bayes (GNB) Classifier Model**

This code demonstrates the process of training a Gaussian Naive Bayes (GNB) Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_GaussianNaiveBayesClassifier.py
# The code demonstrates the process of training Gaussian Naive Bayes Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Gaussian Naive Bayes (GNB) Classifier model
gnb_model = GaussianNB()

# train the model on the entire dataset
gnb_model.fit(X, y)

# predict classes for the entire dataset
y_pred = gnb_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Gaussian Naive Bayes (GNB) Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(gnb_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "gnb_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Gaussian Naive Bayes (GNB) Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Gaussian Naive Bayes (GNB) Classifier model: 0.96

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.94      0.94      0.94        50

Python               2       0.94      0.94      0.94        50

Python

Python        accuracy                           0.96       150

Python       macro avg       0.96      0.96      0.96       150

Python    weighted avg       0.96      0.96      0.96       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\gnb\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Gaussian Naive Bayes (GNB) Classifier model in ONNX format: 0.96

**2.18.2. MQL5 Code for Working with the Gaussian Naive Bayes (GNB) Classifier Model**

```
//+------------------------------------------------------------------+
//|                            Iris_GaussianNaiveBayesClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "gnb_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="GaussianNaiveBayesClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model:GaussianNaiveBayesClassifier  sample=53 FAILED [class=2, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model:GaussianNaiveBayesClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model:GaussianNaiveBayesClassifier  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model:GaussianNaiveBayesClassifier  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model:GaussianNaiveBayesClassifier  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model:GaussianNaiveBayesClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model:GaussianNaiveBayesClassifier   correct results: 96.00%
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model=GaussianNaiveBayesClassifier all samples accuracy=0.960000
Iris_GaussianNaiveBayesClassifier (EURUSD,H1)   model=GaussianNaiveBayesClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 96%, which corresponds to the accuracy of the original model.

**2.18.3. ONNX Representation of the Gaussian Naive Bayes (GNB) Classifier**

![Figure 32. ONNX Representation of the Gaussian Naive Bayes (GNB) Classifier in Netron](https://c.mql5.com/2/58/onnx_18_gnb_classifier.png)

Figure 32. ONNX Representation of the Gaussian Naive Bayes (GNB) Classifier in Netron

**2.19. Multinomial Naive Bayes (MNB) Classifier**

The Multinomial Naive Bayes (MNB) Classifier is a machine learning method based on a Bayesian probabilistic model and is used for classification tasks, especially in text processing. It is one of the variants of naive Bayes classifiers and assumes that features represent counts, such as the number of word occurrences in text.

Principles of the Multinomial Naive Bayes Classifier:

1. Bayesian Approach: MNB also follows the Bayesian approach to classification, using Bayes' theorem to compute the probability of an object belonging to each class.
2. Assumption of Multinomial Distribution: The primary assumption in MNB is that features represent counts, such as the number of word occurrences in text, and follow a multinomial distribution. This assumption is often valid for textual data.
3. Parameter Estimation: The MNB model is trained on the training dataset by computing the parameters of the distribution for each feature within each class.

Advantages of the Multinomial Naive Bayes Classifier:

- Effectiveness in Text Processing: MNB performs well in tasks related to the analysis of textual data, such as text classification or spam filtering, thanks to the assumption of feature counts.
- Simplicity and Training Speed: Like other naive Bayes classifiers, MNB is a straightforward algorithm that trains quickly, even on large volumes of textual data.

Limitations of the Multinomial Naive Bayes Classifier:

- Naive Assumption: The assumption of a multinomial distribution of features can be overly simplistic and inaccurate for real-world data, especially when features have complex structures.
- Inability to Account for Word Order: MNB does not consider the order of words in text, which can be important in certain text analysis tasks.
- Sensitivity to Rare Words: MNB can be sensitive to rare words, and an insufficient number of occurrences can reduce classification accuracy.

The Multinomial Naive Bayes Classifier is a useful method for text analysis tasks, particularly when features are related to counts, such as the number of words in text. It is widely used in natural language processing (NLP) for text classification, document categorization, and other text analyses.

**2.19.1. Code for Creating the Multinomial Naive Bayes (MNB) Classifier Model**

This code demonstrates the process of training a Multinomial Naive Bayes (MNB) Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_MultinomialNaiveBayesClassifier.py
# The code demonstrates the process of training Multinomial Naive Bayes (MNB) Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Multinomial Naive Bayes (MNB) Classifier model
mnb_model = MultinomialNB()

# train the model on the entire dataset
mnb_model.fit(X, y)

# predict classes for the entire dataset
y_pred = mnb_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Multinomial Naive Bayes (MNB) Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(mnb_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "mnb_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Multinomial Naive Bayes (MNB) Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Multinomial Naive Bayes (MNB) Classifier model: 0.9533333333333334

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.94      0.92      0.93        50

Python               2       0.92      0.94      0.93        50

Python

Python        accuracy                           0.95       150

Python       macro avg       0.95      0.95      0.95       150

Python    weighted avg       0.95      0.95      0.95       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\mnb\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Multinomial Naive Bayes (MNB) Classifier model in ONNX format: 0.9533333333333334

**2.19.2. MQL5 Code for Working with the Multinomial Naive Bayes (MNB) Classifier Model**

```
//+------------------------------------------------------------------+
//|                         Iris_MultinomialNaiveBayesClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "mnb_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="MultinomialNaiveBayesClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier  sample=69 FAILED [class=2, true class=1] features=(6.20,2.20,4.50,1.50]
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier  sample=73 FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier  sample=132 FAILED [class=1, true class=2] features=(7.90,3.80,6.40,2.00]
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier   correct results: 95.33%
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model=MultinomialNaiveBayesClassifier all samples accuracy=0.953333
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model:MultinomialNaiveBayesClassifier  FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50)
Iris_MultinomialNaiveBayesClassifier (EURUSD,H1)        model=MultinomialNaiveBayesClassifier batch test accuracy=0.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 95.33%, which corresponds to the accuracy of the original model.

**2.19.3. ONNX Representation of the Multinomial Naive Bayes (MNB) Classifier**

![Figure 33. ONNX Representation of the Multinomial Naive Bayes (MNB) Classifier in Netron](https://c.mql5.com/2/58/onnx_19_mnb_classifier.png)

Figure 33. ONNX Representation of the Multinomial Naive Bayes (MNB) Classifier in Netron

**2.20. Complement Naive Bayes (CNB) Classifier**

The Complement Naive Bayes (CNB) Classifier is a variant of the naive Bayes classifier that has been specifically designed to work with unbalanced data, where one class may be significantly more prevalent than the other. This classifier adapts the classic naive Bayes method to address class imbalance.

Principles of the Complement Naive Bayes Classifier:

1. Bayesian Approach: Like other Bayesian classifiers, CNB follows the Bayesian approach to classification and uses Bayes' theorem to compute the probability of an object belonging to each class.
2. Addressing Class Imbalance: The primary purpose of CNB is to correct class imbalance. Instead of considering the probability of features in the class, as the standard naive Bayes method does, CNB attempts to consider the probability of features outside the class. This is especially useful when one class is significantly less represented than the other.
3. Parameter Estimation: The CNB model is trained on the training dataset by computing the parameters of the distribution for each feature outside the class.

Advantages of the Complement Naive Bayes Classifier:

- Suitability for Unbalanced Data: CNB performs well in classification tasks with unbalanced data, where classes have different frequencies.
- Simplicity and Training Speed: Like other naive Bayes classifiers, CNB is a simple algorithm that trains quickly, even on large volumes of data.

Limitations of the Complement Naive Bayes Classifier:

- Sensitivity to the Choice of Regularization Parameter: As in other Bayesian methods, selecting the right value for the regularization parameter may require tuning and evaluation.
- Naive Assumption: Like other naive Bayes classifiers, CNB makes the assumption of feature independence, which can be overly simplistic for some tasks.

The Complement Naive Bayes Classifier is a good choice for classification tasks with unbalanced data, especially when one class is significantly less represented than the other. It can be particularly useful in text classification tasks where words may be heavily imbalanced across classes, such as sentiment analysis or spam filtering.

**2.20.1. Code for Creating the Complement Naive Bayes (CNB) Classifier Model**

This code demonstrates the process of training a Complement Naive Bayes (CNB) Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_CNBClassifier.py
# The code demonstrates the process of training Complement Naive Bayes (CNB) Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Complement Naive Bayes (CNB) Classifier model
cnb_model = ComplementNB()

# train the model on the entire dataset
cnb_model.fit(X, y)

# predict classes for the entire dataset
y_pred = cnb_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Complement Naive Bayes (CNB) Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(cnb_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "cnb_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Complement Naive Bayes (CNB) Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Complement Naive Bayes (CNB) Classifier model: 0.6666666666666666

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       0.96      1.00      0.98        50

Python               1       0.00      0.00      0.00        50

Python               2       0.51      1.00      0.68        50

Python

Python        accuracy                           0.67       150

Python       macro avg       0.49      0.67      0.55       150

Python    weighted avg       0.49      0.67      0.55       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\cnb\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Complement Naive Bayes (CNB) Classifier model in ONNX format: 0.6666666666666666

**2.20.2. MQL5 Code for Working with the Complement Naive Bayes (CNB) Classifier Model**

```
//+------------------------------------------------------------------+
//|                                           Iris_CNBClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "cnb_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="CNBClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=51 FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=52 FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=53 FAILED [class=2, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=54 FAILED [class=2, true class=1] features=(5.50,2.30,4.00,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=55 FAILED [class=2, true class=1] features=(6.50,2.80,4.60,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=56 FAILED [class=2, true class=1] features=(5.70,2.80,4.50,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=57 FAILED [class=2, true class=1] features=(6.30,3.30,4.70,1.60]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=58 FAILED [class=2, true class=1] features=(4.90,2.40,3.30,1.00]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=59 FAILED [class=2, true class=1] features=(6.60,2.90,4.60,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=60 FAILED [class=2, true class=1] features=(5.20,2.70,3.90,1.40]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=61 FAILED [class=2, true class=1] features=(5.00,2.00,3.50,1.00]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=62 FAILED [class=2, true class=1] features=(5.90,3.00,4.20,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=63 FAILED [class=2, true class=1] features=(6.00,2.20,4.00,1.00]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=64 FAILED [class=2, true class=1] features=(6.10,2.90,4.70,1.40]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=65 FAILED [class=2, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=66 FAILED [class=2, true class=1] features=(6.70,3.10,4.40,1.40]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=67 FAILED [class=2, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=68 FAILED [class=2, true class=1] features=(5.80,2.70,4.10,1.00]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=69 FAILED [class=2, true class=1] features=(6.20,2.20,4.50,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=70 FAILED [class=2, true class=1] features=(5.60,2.50,3.90,1.10]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=72 FAILED [class=2, true class=1] features=(6.10,2.80,4.00,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=73 FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=74 FAILED [class=2, true class=1] features=(6.10,2.80,4.70,1.20]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=75 FAILED [class=2, true class=1] features=(6.40,2.90,4.30,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=76 FAILED [class=2, true class=1] features=(6.60,3.00,4.40,1.40]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=77 FAILED [class=2, true class=1] features=(6.80,2.80,4.80,1.40]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=79 FAILED [class=2, true class=1] features=(6.00,2.90,4.50,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=80 FAILED [class=0, true class=1] features=(5.70,2.60,3.50,1.00]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=81 FAILED [class=2, true class=1] features=(5.50,2.40,3.80,1.10]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=82 FAILED [class=2, true class=1] features=(5.50,2.40,3.70,1.00]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=83 FAILED [class=2, true class=1] features=(5.80,2.70,3.90,1.20]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=86 FAILED [class=2, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=87 FAILED [class=2, true class=1] features=(6.70,3.10,4.70,1.50]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=88 FAILED [class=2, true class=1] features=(6.30,2.30,4.40,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=89 FAILED [class=2, true class=1] features=(5.60,3.00,4.10,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=90 FAILED [class=2, true class=1] features=(5.50,2.50,4.00,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=91 FAILED [class=2, true class=1] features=(5.50,2.60,4.40,1.20]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=92 FAILED [class=2, true class=1] features=(6.10,3.00,4.60,1.40]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=93 FAILED [class=2, true class=1] features=(5.80,2.60,4.00,1.20]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=94 FAILED [class=2, true class=1] features=(5.00,2.30,3.30,1.00]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=95 FAILED [class=2, true class=1] features=(5.60,2.70,4.20,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=96 FAILED [class=2, true class=1] features=(5.70,3.00,4.20,1.20]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=97 FAILED [class=2, true class=1] features=(5.70,2.90,4.20,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=98 FAILED [class=2, true class=1] features=(6.20,2.90,4.30,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=99 FAILED [class=0, true class=1] features=(5.10,2.50,3.00,1.10]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  sample=100 FAILED [class=2, true class=1] features=(5.70,2.80,4.10,1.30]
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier   correct results: 66.67%
Iris_CNBClassifier (EURUSD,H1)  model=CNBClassifier all samples accuracy=0.666667
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50)
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40)
Iris_CNBClassifier (EURUSD,H1)  model:CNBClassifier  FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50)
Iris_CNBClassifier (EURUSD,H1)  model=CNBClassifier batch test accuracy=0.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 66.67%, which corresponds to the accuracy of the original model.

**2.20.3. ONNX Representation of the Complement Naive Bayes (CNB) Classifier**

![Figure 34. ONNX Representation of the Complement Naive Bayes (CNB) Classifier in Netron](https://c.mql5.com/2/58/onnx_20_cnb_classifier.png)

Figure 34. ONNX Representation of the Complement Naive Bayes (CNB) Classifier in Netron

**2.21. Bernoulli Naive Bayes (BNB) Classifier**

The Bernoulli Naive Bayes (BNB) Classifier is another variant of the naive Bayes classifier used for binary classification tasks. This classifier is particularly useful in situations where features are represented as binary data, such as in text analysis where features may be the presence or absence of words in the text.

Principles of the Bernoulli Naive Bayes Classifier:

1. Bayesian Approach: Like other Bayesian classifiers, BNB follows the Bayesian approach to classification and uses Bayes' theorem to compute the probability of an object belonging to each class.
2. Assumption of Binary Features: The primary assumption of BNB is that features are represented as binary data, meaning they can only have two values, such as 1 and 0, where 1 represents the presence of the feature, and 0 represents its absence.
3. Parameter Estimation: The BNB model is trained on the training dataset by computing the parameters of the distribution for each feature in each class.

Advantages of the Bernoulli Naive Bayes Classifier:

- Effectiveness for Binary Data: BNB works well in tasks where features are represented as binary data, and it can be particularly useful in text analysis or event classification.
- Simplicity and Training Speed: Like other naive Bayes classifiers, BNB is a simple algorithm that trains quickly.

Limitations of the Bernoulli Naive Bayes Classifier:

- Restriction to Binary Features: BNB is not suitable for tasks where features are not binary. If features have more than two values, BNB does not take that information into account.
- Naive Assumption: Like other naive Bayes classifiers, BNB makes the assumption of feature independence, which can be overly simplistic for some tasks.

The Bernoulli Naive Bayes Classifier is a good choice for binary classification tasks with binary features, such as sentiment analysis of text or spam classification. It is easy to use and performs well with this type of data.

**2.21.1. Code for Creating the Bernoulli Naive Bayes (BNB) Classifier Model**

This code demonstrates the process of training a Bernoulli Naive Bayes (BNB) Classifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_BNBClassifier.py
# The code demonstrates the process of training Bernoulli Naive Bayes (BNB) Classifier on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Bernoulli Naive Bayes (BNB) Classifier model
bnb_model = BernoulliNB()

# train the model on the entire dataset
bnb_model.fit(X, y)

# predict classes for the entire dataset
y_pred = bnb_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Bernoulli Naive Bayes (BNB) Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(bnb_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "bnb_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Bernoulli Naive Bayes (BNB) Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Bernoulli Naive Bayes (BNB) Classifier model: 0.3333333333333333

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       0.33      1.00      0.50        50

Python               1       0.00      0.00      0.00        50

Python               2       0.00      0.00      0.00        50

Python

Python        accuracy                           0.33       150

Python       macro avg       0.11      0.33      0.17       150

Python    weighted avg       0.11      0.33      0.17       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\bnb\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Bernoulli Naive Bayes (BNB) Classifier model in ONNX format: 0.3333333333333333

**2.21.2. MQL5 Code for Working with the Bernoulli Naive Bayes (BNB) Classifier Model**

```
//+------------------------------------------------------------------+
//|                                           Iris_BNBClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "bnb_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="BNBClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=51 FAILED [class=0, true class=1] features=(7.00,3.20,4.70,1.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=52 FAILED [class=0, true class=1] features=(6.40,3.20,4.50,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=53 FAILED [class=0, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=54 FAILED [class=0, true class=1] features=(5.50,2.30,4.00,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=55 FAILED [class=0, true class=1] features=(6.50,2.80,4.60,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=56 FAILED [class=0, true class=1] features=(5.70,2.80,4.50,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=57 FAILED [class=0, true class=1] features=(6.30,3.30,4.70,1.60]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=58 FAILED [class=0, true class=1] features=(4.90,2.40,3.30,1.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=59 FAILED [class=0, true class=1] features=(6.60,2.90,4.60,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=60 FAILED [class=0, true class=1] features=(5.20,2.70,3.90,1.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=61 FAILED [class=0, true class=1] features=(5.00,2.00,3.50,1.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=62 FAILED [class=0, true class=1] features=(5.90,3.00,4.20,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=63 FAILED [class=0, true class=1] features=(6.00,2.20,4.00,1.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=64 FAILED [class=0, true class=1] features=(6.10,2.90,4.70,1.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=65 FAILED [class=0, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=66 FAILED [class=0, true class=1] features=(6.70,3.10,4.40,1.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=67 FAILED [class=0, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=68 FAILED [class=0, true class=1] features=(5.80,2.70,4.10,1.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=69 FAILED [class=0, true class=1] features=(6.20,2.20,4.50,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=70 FAILED [class=0, true class=1] features=(5.60,2.50,3.90,1.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=71 FAILED [class=0, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=72 FAILED [class=0, true class=1] features=(6.10,2.80,4.00,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=73 FAILED [class=0, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=74 FAILED [class=0, true class=1] features=(6.10,2.80,4.70,1.20]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=75 FAILED [class=0, true class=1] features=(6.40,2.90,4.30,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=76 FAILED [class=0, true class=1] features=(6.60,3.00,4.40,1.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=77 FAILED [class=0, true class=1] features=(6.80,2.80,4.80,1.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=78 FAILED [class=0, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=79 FAILED [class=0, true class=1] features=(6.00,2.90,4.50,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=80 FAILED [class=0, true class=1] features=(5.70,2.60,3.50,1.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=81 FAILED [class=0, true class=1] features=(5.50,2.40,3.80,1.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=82 FAILED [class=0, true class=1] features=(5.50,2.40,3.70,1.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=83 FAILED [class=0, true class=1] features=(5.80,2.70,3.90,1.20]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=84 FAILED [class=0, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=85 FAILED [class=0, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=86 FAILED [class=0, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=87 FAILED [class=0, true class=1] features=(6.70,3.10,4.70,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=88 FAILED [class=0, true class=1] features=(6.30,2.30,4.40,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=89 FAILED [class=0, true class=1] features=(5.60,3.00,4.10,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=90 FAILED [class=0, true class=1] features=(5.50,2.50,4.00,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=91 FAILED [class=0, true class=1] features=(5.50,2.60,4.40,1.20]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=92 FAILED [class=0, true class=1] features=(6.10,3.00,4.60,1.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=93 FAILED [class=0, true class=1] features=(5.80,2.60,4.00,1.20]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=94 FAILED [class=0, true class=1] features=(5.00,2.30,3.30,1.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=95 FAILED [class=0, true class=1] features=(5.60,2.70,4.20,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=96 FAILED [class=0, true class=1] features=(5.70,3.00,4.20,1.20]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=97 FAILED [class=0, true class=1] features=(5.70,2.90,4.20,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=98 FAILED [class=0, true class=1] features=(6.20,2.90,4.30,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=99 FAILED [class=0, true class=1] features=(5.10,2.50,3.00,1.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=100 FAILED [class=0, true class=1] features=(5.70,2.80,4.10,1.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=101 FAILED [class=0, true class=2] features=(6.30,3.30,6.00,2.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=102 FAILED [class=0, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=103 FAILED [class=0, true class=2] features=(7.10,3.00,5.90,2.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=104 FAILED [class=0, true class=2] features=(6.30,2.90,5.60,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=105 FAILED [class=0, true class=2] features=(6.50,3.00,5.80,2.20]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=106 FAILED [class=0, true class=2] features=(7.60,3.00,6.60,2.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=107 FAILED [class=0, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=108 FAILED [class=0, true class=2] features=(7.30,2.90,6.30,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=109 FAILED [class=0, true class=2] features=(6.70,2.50,5.80,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=110 FAILED [class=0, true class=2] features=(7.20,3.60,6.10,2.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=111 FAILED [class=0, true class=2] features=(6.50,3.20,5.10,2.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=112 FAILED [class=0, true class=2] features=(6.40,2.70,5.30,1.90]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=113 FAILED [class=0, true class=2] features=(6.80,3.00,5.50,2.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=114 FAILED [class=0, true class=2] features=(5.70,2.50,5.00,2.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=115 FAILED [class=0, true class=2] features=(5.80,2.80,5.10,2.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=116 FAILED [class=0, true class=2] features=(6.40,3.20,5.30,2.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=117 FAILED [class=0, true class=2] features=(6.50,3.00,5.50,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=118 FAILED [class=0, true class=2] features=(7.70,3.80,6.70,2.20]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=119 FAILED [class=0, true class=2] features=(7.70,2.60,6.90,2.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=120 FAILED [class=0, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=121 FAILED [class=0, true class=2] features=(6.90,3.20,5.70,2.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=122 FAILED [class=0, true class=2] features=(5.60,2.80,4.90,2.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=123 FAILED [class=0, true class=2] features=(7.70,2.80,6.70,2.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=124 FAILED [class=0, true class=2] features=(6.30,2.70,4.90,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=125 FAILED [class=0, true class=2] features=(6.70,3.30,5.70,2.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=126 FAILED [class=0, true class=2] features=(7.20,3.20,6.00,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=127 FAILED [class=0, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=128 FAILED [class=0, true class=2] features=(6.10,3.00,4.90,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=129 FAILED [class=0, true class=2] features=(6.40,2.80,5.60,2.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=130 FAILED [class=0, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=131 FAILED [class=0, true class=2] features=(7.40,2.80,6.10,1.90]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=132 FAILED [class=0, true class=2] features=(7.90,3.80,6.40,2.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=133 FAILED [class=0, true class=2] features=(6.40,2.80,5.60,2.20]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=134 FAILED [class=0, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=135 FAILED [class=0, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=136 FAILED [class=0, true class=2] features=(7.70,3.00,6.10,2.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=137 FAILED [class=0, true class=2] features=(6.30,3.40,5.60,2.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=138 FAILED [class=0, true class=2] features=(6.40,3.10,5.50,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=139 FAILED [class=0, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=140 FAILED [class=0, true class=2] features=(6.90,3.10,5.40,2.10]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=141 FAILED [class=0, true class=2] features=(6.70,3.10,5.60,2.40]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=142 FAILED [class=0, true class=2] features=(6.90,3.10,5.10,2.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=143 FAILED [class=0, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=144 FAILED [class=0, true class=2] features=(6.80,3.20,5.90,2.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=145 FAILED [class=0, true class=2] features=(6.70,3.30,5.70,2.50]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=146 FAILED [class=0, true class=2] features=(6.70,3.00,5.20,2.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=147 FAILED [class=0, true class=2] features=(6.30,2.50,5.00,1.90]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=148 FAILED [class=0, true class=2] features=(6.50,3.00,5.20,2.00]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=149 FAILED [class=0, true class=2] features=(6.20,3.40,5.40,2.30]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  sample=150 FAILED [class=0, true class=2] features=(5.90,3.00,5.10,1.80]
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier   correct results: 33.33%
Iris_BNBClassifier (EURUSD,H1)  model=BNBClassifier all samples accuracy=0.333333
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  FAILED [class=0, true class=1] features=(6.30,2.50,4.90,1.50)
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  FAILED [class=0, true class=2] features=(6.30,2.70,4.90,1.80)
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  FAILED [class=0, true class=1] features=(7.00,3.20,4.70,1.40)
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  FAILED [class=0, true class=1] features=(6.40,3.20,4.50,1.50)
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  FAILED [class=0, true class=2] features=(6.30,3.30,6.00,2.50)
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  FAILED [class=0, true class=2] features=(5.80,2.70,5.10,1.90)
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  FAILED [class=0, true class=2] features=(7.10,3.00,5.90,2.10)
Iris_BNBClassifier (EURUSD,H1)  model:BNBClassifier  FAILED [class=0, true class=2] features=(6.30,2.90,5.60,1.80)
Iris_BNBClassifier (EURUSD,H1)  model=BNBClassifier batch test accuracy=0.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 33.33%, which corresponds to the accuracy of the original model.

**2.21.3. ONNX Representation of the Bernoulli Naive Bayes (BNB) Classifier**

![Figure 35. ONNX Representation of the Bernoulli Naive Bayes (BNB) Classifier in Netron](https://c.mql5.com/2/58/onnx_21_bnb_classifier.png)

Figure 35. ONNX Representation of the Bernoulli Naive Bayes (BNB) Classifier in Netron

**2.22. Multilayer Perceptron Classifier**

The Multilayer Perceptron (MLP) Classifier is a multi-layer neural network used for classification tasks. It consists of multiple layers of neurons, including an input layer, hidden layers, and an output layer. The MLP Classifier has the ability to learn complex nonlinear dependencies in data.

Principles of the MLP Classifier:

1. Multi-Layer Architecture: The MLP Classifier has a multi-layer architecture that includes an input layer, one or more hidden layers, and an output layer. Each neuron in the layers is connected to neurons in neighboring layers with weights that are learned.
2. Activation Functions: Inside each neuron, an activation function is applied, introducing nonlinearity into the model and allowing the MLP Classifier to model complex data dependencies.
3. Training with Backpropagation: The MLP Classifier is trained using the backpropagation method, which minimizes the error between the model's predictions and the true class labels.

Advantages of the MLP Classifier:

- Ability to Model Complex Dependencies: The MLP Classifier can learn complex nonlinear dependencies in data and, as a result, can perform well in tasks where simple linear models are insufficient.
- Versatility: The MLP Classifier can be used for a wide range of classification tasks, including multi-class classification and multi-task problems.

Limitations of the MLP Classifier:

- Sensitivity to Hyperparameters: The MLP Classifier has many hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and others. Tuning these parameters can be time-consuming and resource-intensive.
- Requirement for Large Amounts of Data: The MLP Classifier requires a substantial amount of training data to avoid overfitting, especially when the model has many parameters.
- Overfitting: If the model has too many parameters or insufficient data, it can overfit and perform poorly on new data.

The MLP Classifier is a powerful tool for classification tasks, especially when data exhibits complex dependencies. It is commonly used in the fields of machine learning and deep learning to solve various classification problems. However, proper tuning of hyperparameters and ensuring an adequate amount of training data are essential for successful application of this model.

**2.22.1. Code for Creating the Multilayer Perceptron Classifier Model**

This code demonstrates the process of training a Multilayer Perceptron Classifier on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_MLPClassifier.py
# The code demonstrates the process of training Multilayer Perceptron Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Multilayer Perceptron (MLP) Classifier model
mlp_model = MLPClassifier(max_iter=1000, random_state=42)

# train the model on the entire dataset
mlp_model.fit(X, y)

# predict classes for the entire dataset
y_pred = mlp_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Multilayer Perceptron (MLP) Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(mlp_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path +"mlp_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Multilayer Perceptron (MLP) Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Multilayer Perceptron (MLP) Classifier model: 0.98

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      0.94      0.97        50

Python               2       0.94      1.00      0.97        50

Python

Python        accuracy                           0.98       150

Python       macro avg       0.98      0.98      0.98       150

Python    weighted avg       0.98      0.98      0.98       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\mlp\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Multilayer Perceptron (MLP) Classifier model in ONNX format: 0.98

**2.22.2. MQL5 code for Working with the Multilayer Perceptron Classifier Model**

```
//+------------------------------------------------------------------+
//|                                           Iris_MLPClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "mlp_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="MLPClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_MLPClassifier (EURUSD,H1)  model:MLPClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_MLPClassifier (EURUSD,H1)  model:MLPClassifier  sample=73 FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_MLPClassifier (EURUSD,H1)  model:MLPClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_MLPClassifier (EURUSD,H1)  model:MLPClassifier   correct results: 98.00%
Iris_MLPClassifier (EURUSD,H1)  model=MLPClassifier all samples accuracy=0.980000
Iris_MLPClassifier (EURUSD,H1)  model:MLPClassifier  FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50)
Iris_MLPClassifier (EURUSD,H1)  model=MLPClassifier batch test accuracy=0.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 98%, which corresponds to the accuracy of the original model.

**2.22.3. ONNX Representation of the Multilayer Perceptron Classifier**

![Figure 36. ONNX Representation of the Multilayer Perceptron Classifier in Netron](https://c.mql5.com/2/58/onnx_22_mlp_classifier.png)

Figure 36. ONNX Representation of the Multilayer Perceptron Classifier in Netron

**2.23. Linear Discriminant Analysis (LDA) Classifier**

The Linear Discriminant Analysis (LDA) Classifier is a machine learning method used for classification tasks. It belongs to the family of dimensionality reduction methods and classification in lower-dimensional space. LDA constructs hyperplanes to maximize the separation between classes.

Principles of the LDA Classifier:

1. Dimensionality Reduction: The core idea of LDA is dimensionality reduction. It aims to find a new feature space where the data classes are maximally separated.
2. Maximizing Separation: LDA constructs hyperplanes (linear combinations of features) that maximize the difference between the mean values of features in different classes and minimize the variance within each class.
3. Training Parameters: The LDA model is trained on the training dataset, calculating the parameters of hyperplanes and data projections into the new feature space.

Advantages of LDA Classifier:

- Improved Class Separation: LDA can significantly improve class separation in data, especially in cases where classes heavily overlap in the original feature space.
- Dimensionality Reduction: LDA can also be used for data dimensionality reduction, which can be useful for visualization and reducing computational complexity.

Limitations of LDA Classifier:

- Normality Assumption: LDA assumes that features follow a normal distribution, and classes have equal covariance matrices. If these assumptions are not met, LDA may provide less accurate results.
- Sensitivity to Outliers: LDA can be sensitive to outliers in the data as they can affect the model parameter calculations.
- Challenges in Multiclass Classification: LDA was originally developed for binary classification, and its extension to multiclass tasks requires adaptation.

The LDA Classifier is a valuable method for classification and dimensionality reduction tasks, especially when there is a need to improve class separation. It is frequently used in statistics, biology, medical analysis, and other fields for data analysis and classification.

**2.23.1. Code for Creating the Linear Discriminant Analysis (LDA) Classifier Model**

This code demonstrates the process of training a Linear Discriminant Analysis (LDA) Classifier on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_LDAClassifier.py
# The code demonstrates the process of training Linear Discriminant Analysis (LDA) Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Linear Discriminant Analysis (LDA) Classifier model
lda_model = LinearDiscriminantAnalysis()

# train the model on the entire dataset
lda_model.fit(X, y)

# predict classes for the entire dataset
y_pred = lda_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Linear Discriminant Analysis (LDA) Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(lda_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path +"lda_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Linear Discriminant Analysis (LDA) Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Linear Discriminant Analysis (LDA) Classifier model: 0.98

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.98      0.96      0.97        50

Python               2       0.96      0.98      0.97        50

Python

Python        accuracy                           0.98       150

Python       macro avg       0.98      0.98      0.98       150

Python    weighted avg       0.98      0.98      0.98       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\lda\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Linear Discriminant Analysis (LDA) Classifier model in ONNX format: 0.98

**2.23.2. MQL5 code for Working with the Linear Discriminant Analysis (LDA) Classifier Model**

```
//+------------------------------------------------------------------+
//|                                           Iris_LDAClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "lda_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="LDAClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_LDAClassifier (EURUSD,H1)  model:LDAClassifier  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_LDAClassifier (EURUSD,H1)  model:LDAClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_LDAClassifier (EURUSD,H1)  model:LDAClassifier  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_LDAClassifier (EURUSD,H1)  model:LDAClassifier   correct results: 98.00%
Iris_LDAClassifier (EURUSD,H1)  model=LDAClassifier all samples accuracy=0.980000
Iris_LDAClassifier (EURUSD,H1)  model=LDAClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 98%, which corresponds to the accuracy of the original model.

**2.23.3. ONNX Representation of the Linear Discriminant Analysis (LDA) Classifier**

![Figure 37. ONNX Representation of the Linear Discriminant Analysis (LDA) Classifier in Netron](https://c.mql5.com/2/58/onnx_23_lda_classifier.png)

Figure 37. ONNX Representation of the Linear Discriminant Analysis (LDA) Classifier in Netron

**2.24. Hist Gradient Boosting**

The Hist Gradient Boosting Classifier is a machine learning algorithm that belongs to the gradient boosting family and is designed for classification tasks. It is an efficient and powerful method widely used in data analysis and machine learning.

Principles of the Hist Gradient Boosting Classifier:

1. Gradient Boosting: The Hist Gradient Boosting Classifier is based on the gradient boosting method, which builds an ensemble of decision trees to improve classification. It does this by sequentially training weak models and correcting the errors of previous models.
2. Histogram Usage: The "Hist" in the name indicates that this algorithm uses histograms for efficient data processing. Instead of exhaustive feature enumeration, Hist Gradient Boosting constructs histograms of features, enabling quick decision tree construction.
3. Training on Residuals: Like other gradient boosting methods, Hist Gradient Boosting trains each new tree on the residuals of the previous model to refine predictions.

Advantages of Hist Gradient Boosting Classifier:

- High Accuracy: The Hist Gradient Boosting Classifier typically provides high classification accuracy, especially when using a large number of trees.
- Efficiency: Using histograms allows the algorithm to efficiently process large datasets and quickly build an ensemble.
- Ability to Handle Heterogeneous Data: The algorithm can handle heterogeneous data, including categorical and numerical features.

Limitations of Hist Gradient Boosting Classifier:

- Sensitivity to Overfitting: When parameters are not properly tuned or when a large number of trees are used, the Hist Gradient Boosting Classifier may be prone to overfitting.
- Parameter Tuning: Like other gradient boosting algorithms, Hist Gradient Boosting requires careful parameter tuning for optimal performance.

The Hist Gradient Boosting Classifier is a powerful algorithm for classification and regression tasks that offers high accuracy and efficiency in data processing. It finds applications in various fields, such as data analysis, bioinformatics, finance, and more.

**2.24.1. Code for Creating the Histogram-Based Gradient Boosting Classifier Model**

This code demonstrates the process of training a Histogram-Based Gradient Boosting Classifier on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_HistGradientBoostingClassifier.py
# The code demonstrates the process of training Histogram-Based Gradient Boosting Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a Histogram-Based Gradient Boosting Classifier model
hist_gradient_boosting_model = HistGradientBoostingClassifier(random_state=42)

# train the model on the entire dataset
hist_gradient_boosting_model.fit(X, y)

# predict classes for the entire dataset
y_pred = hist_gradient_boosting_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of Hist Gradient Boosting Classifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(hist_gradient_boosting_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path +"hist_gradient_boosting_classifier_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of Hist Gradient Boosting Classifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of Hist Gradient Boosting Classifier model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\hist\_gradient\_boosting\_classifier\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of Hist Gradient Boosting Classifier model in ONNX format: 1.0

**2.24.2. MQL5 code for Working with the Histogram-Based Gradient Boosting Classifier Model**

```
//+------------------------------------------------------------------+
//|                          Iris_HistGradientBoostingClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "hist_gradient_boosting_classifier_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="HistGradientBoostingClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_HistGradientBoostingClassifier (EURUSD,H1) model:HistGradientBoostingClassifier   correct results: 100.00%
Iris_HistGradientBoostingClassifier (EURUSD,H1) model=HistGradientBoostingClassifier all samples accuracy=1.000000
Iris_HistGradientBoostingClassifier (EURUSD,H1) model=HistGradientBoostingClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 100%, which corresponds to the accuracy of the original model.

**2.24.3. ONNX Representation of the Histogram-Based Gradient Boosting Classifier**

![Figure 38. ONNX Representation of the Histogram-Based Gradient Boosting Classifier in Netron](https://c.mql5.com/2/58/onnx_24_histgb_classifier.png)

Figure 38. ONNX Representation of the Histogram-Based Gradient Boosting Classifier in Netron

**2.25. CategoricalNB** **Classifier**

CategoricalNB is a classification algorithm based on Bayes' theorem. It is specifically designed for datasets with categorical features and is widely used in text classification, spam detection, and other applications involving discrete data.

Principles of CategoricalNB:

1. Naive Bayes Classifier: CategoricalNB is one type of naive Bayes classifier based on Bayes' theorem. It calculates the probability of belonging to a particular class for a set of features using conditional probabilities of each feature given the class.
2. Categorical Features: Unlike the Gaussian Naive Bayes classifier, which assumes continuous features with a normal distribution, CategoricalNB is suitable for datasets with categorical features. It models the probability distribution of each feature for each class.
3. Independence Assumption: The "naive" in the naive Bayes classifier comes from the assumption of feature independence. CategoricalNB assumes that features are conditionally independent given the class. Despite the fact that this assumption is not always met in practice, naive Bayes methods can perform well on many real-world datasets.

Advantages of CategoricalNB:

- Efficiency: CategoricalNB is computationally efficient and scalable to large datasets. It requires minimal memory and can provide fast predictions.
- Interpretability: Its probabilistic nature makes CategoricalNB interpretable. It can provide insights into which features influence the prediction.
- Handling Categorical Data: CategoricalNB is specifically designed for datasets with categorical features. It can efficiently handle textual data and other discrete feature types.
- Baseline Performance: It often serves as a strong baseline model for text classification tasks and can outperform more complex algorithms on small datasets.

Limitations of CategoricalNB:

- Independence Assumption: The assumption of feature independence may not hold for all datasets. If features are highly dependent, the performance of CategoricalNB may deteriorate.
- Sensitivity to Feature Scaling: CategoricalNB does not require feature scaling since it works with categorical data. However, in some cases, normalizing or encoding categorical features in different ways may affect its performance.
- Limited Expressiveness: CategoricalNB may not capture complex data dependencies as well as more complex algorithms, such as deep learning models.
- Handling Missing Data: It assumes that there are no missing values in the dataset, and missing values need to be preprocessed.

CategoricalNB is a valuable classification algorithm, especially suitable for datasets with categorical features. Its simplicity, efficiency, and interpretability make it a useful tool for various classification tasks. Despite limitations such as the independence assumption, it remains a popular choice for text classification and other tasks where discrete data dominates. When working with categorical data, considering CategoricalNB as a baseline model is often a reasonable choice. However, it's important to assess its performance compared to more complex models, especially if there are feature dependencies in the data.

**2.25.1. Code for Creating the CategoricalNB Classifier Model**

This code demonstrates the process of training a CategoricalNB Classifier on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_CategoricalNBClassifier.py
# The code demonstrates the process of training CategoricalNB Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create a CategoricalNB model
categorical_nb_model = CategoricalNB()

# train the model on the entire dataset
categorical_nb_model.fit(X, y)

# predict classes for the entire dataset
y_pred = categorical_nb_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of CategoricalNB model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(categorical_nb_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "categorical_nb_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of CategoricalNB model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of CategoricalNB model: 0.9333333333333333

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.86      0.96      0.91        50

Python               2       0.95      0.84      0.89        50

Python

Python        accuracy                           0.93       150

Python       macro avg       0.94      0.93      0.93       150

Python    weighted avg       0.94      0.93      0.93       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\categorical\_nb\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of CategoricalNB model in ONNX format: 0.9333333333333333

**2.25.2. MQL5 code for Working with the CategoricalNB Classifier Model**

```
//+------------------------------------------------------------------+
//|                                 Iris_CategoricalNBClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "categorical_nb_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="CategoricalNBClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=102 FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=122 FAILED [class=1, true class=2] features=(5.60,2.80,4.90,2.00]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=124 FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=127 FAILED [class=1, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=128 FAILED [class=1, true class=2] features=(6.10,3.00,4.90,1.80]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  sample=143 FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier   correct results: 93.33%
Iris_CategoricalNBClassifier (EURUSD,H1)        model=CategoricalNBClassifier all samples accuracy=0.933333
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80)
Iris_CategoricalNBClassifier (EURUSD,H1)        model:CategoricalNBClassifier  FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90)
Iris_CategoricalNBClassifier (EURUSD,H1)        model=CategoricalNBClassifier batch test accuracy=0.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 93.33%, which corresponds to the accuracy of the original model.

**2.25.3. ONNX Representation of the CategoricalNB Classifier**

![Figure 39. ONNX Representation of the CategoricalNB Classifier in Netron](https://c.mql5.com/2/58/onnx_25_categorial_nb_classifier.png)

Figure 39. ONNX Representation of the CategoricalNB Classifier in Netron

**Note on ExtraTreeClassifier and ExtraTreesClassifier Models**

ExtraTreeClassifier and ExtraTreesClassifier are two different classifiers, and their main difference lies in how they work:

ExtraTreeClassifier (Extremely Randomized Trees Classifier):

- This classifier is also known as Extremely Randomized Trees or Extra-Trees.
- It is based on the idea of random decision trees.
- In ExtraTreeClassifier, the choice of split for each tree node occurs randomly without any prior search for the best split.
- This makes the classifier less computationally intensive than classical Random Forest because it doesn't require the computation of optimal splits for each node.
- ExtraTreeClassifier often uses random thresholds for features and random splitting, resulting in more random trees.
- The absence of a search for the best splits makes ExtraTreeClassifier faster but less accurate compared to Random Forest.

ExtraTreesClassifier (Extremely Randomized Trees Classifier):

- ExtraTreesClassifier is also a classifier based on the Extremely Randomized Trees method.
- The main difference between ExtraTreesClassifier and ExtraTreeClassifier is that ExtraTreesClassifier performs random splits to choose the best splits at each tree node.
- This means that ExtraTreesClassifier applies a random forest with an additional level of randomness when selecting optimal splits.
- ExtraTreesClassifier is usually more accurate than ExtraTreeClassifier because it performs random splits to find the best features for splitting.
- However, ExtraTreesClassifier can be more computationally intensive due to the need to perform a broader search for optimal splits.

In summary, the main difference between these two classifiers lies in the level of randomness in split selection. ExtraTreeClassifier makes a random choice for each node without searching for the best splits, while ExtraTreesClassifier performs random splits while looking for optimal splits at each node.

**2.26. ExtraTreeClassifier**

ExtraTreeClassifier, or Extremely Randomized Trees, is a powerful machine learning algorithm used in classification and regression tasks. This algorithm is based on the idea of decision trees and offers improvements compared to traditional random forests and decision trees.

Principles of ExtraTreeClassifier:

1. Random Node Splits: The main principle of ExtraTreeClassifier is that it randomly selects the split for each tree node. This differs from traditional decision trees, which choose the best feature for splitting. ExtraTreeClassifier performs splits without considering the best split, making it more random and resistant to overfitting.
2. Aggregation of Results: During the ensemble construction, ExtraTreeClassifier creates multiple random trees and aggregates their results. This is done to improve the model's generalization and reduce variance. An ensemble of trees helps combat overfitting and increases prediction stability.
3. Random Thresholds: When splitting nodes, ExtraTreeClassifier selects random thresholds for each feature rather than specific optimal values. This introduces more randomness and model stability.

Advantages of ExtraTreeClassifier:

- Resistance to Overfitting: Thanks to random splits and the absence of the best split selection, ExtraTreeClassifier is usually less prone to overfitting compared to regular decision trees.
- High Training Speed: ExtraTreeClassifier requires fewer computational resources for training than many other algorithms, such as random forests. This makes it fast and efficient for large datasets.
- Versatility: ExtraTreeClassifier can be used for both classification and regression tasks, making it a versatile algorithm for various types of problems.

Limitations of ExtraTreeClassifier:

- Randomness: The use of random splits can lead to less accurate models in some cases. Careful parameter tuning is important.
- Sensitivity to Outliers: ExtraTreeClassifier can be sensitive to outliers in the data as it builds random splits. This can result in unstable predictions in some cases.
- Lower Interpretability: Compared to regular decision trees, ExtraTreeClassifier is less interpretable and harder to explain.

ExtraTreeClassifier is a powerful machine learning algorithm that combines resistance to overfitting and high training speed. It can be useful in various classification and regression tasks, especially when computational resources are limited. However, it's important to consider the random nature of this algorithm and its limitations, such as sensitivity to outliers and reduced interpretability. When using ExtraTreeClassifier, careful parameter tuning and consideration of data characteristics are essential.

**2.26.1. Code for Creating the ExtraTreeClassifier Model**

This code demonstrates the process of training an ExtraTreeClassifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_ExtraTreeClassifier.py
# The code demonstrates the process of training ExtraTree Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create an ExtraTreeClassifier model
extra_tree_model = ExtraTreeClassifier()

# train the model on the entire dataset
extra_tree_model.fit(X, y)

# predict classes for the entire dataset
y_pred = extra_tree_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of ExtraTreeClassifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(extra_tree_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "extra_tree_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of ExtraTreeClassifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of ExtraTreeClassifier model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\extra\_tree\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of ExtraTreeClassifier model in ONNX format: 1.0

**2.26.2. MQL5 Code for Working with the ExtraTreeClassifier Model**

```
//+------------------------------------------------------------------+
//|                                     Iris_ExtraTreeClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "extra_tree_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="ExtraTreeClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_ExtraTreeClassifier (EURUSD,H1)    model:ExtraTreeClassifier   correct results: 100.00%
Iris_ExtraTreeClassifier (EURUSD,H1)    model=ExtraTreeClassifier all samples accuracy=1.000000
Iris_ExtraTreeClassifier (EURUSD,H1)    model=ExtraTreeClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 100%, which corresponds to the accuracy of the original model.

**2.26.3. ONNX Representation of the ExtraTreeClassifier**

![Figure 40. ONNX Representation of the ExtraTreeClassifier in Netron](https://c.mql5.com/2/58/onnx_26_extra_tree_classifier.png)

Figure 40. ONNX Representation of the ExtraTreeClassifier in Netron

**2.27. ExtraTreesClassifier**

ExtraTreesClassifier is a powerful machine learning algorithm used for classification tasks. This algorithm is an extension and improvement over Random Forest, offering several advantages and disadvantages.

Principles of ExtraTreesClassifier:

1. Bootstrap Sampling: Similar to Random Forest, ExtraTreesClassifier uses the bootstrap method to create multiple sub-samples from the training dataset. This means that for each tree, a random sub-sample with replacement is created from the original data.
2. Random Splits: Unlike Random Forest, where the best feature for splitting is chosen for each tree node, ExtraTreesClassifier uses random features and random thresholds to split nodes. This makes the trees more random and reduces overfitting.
3. Voting: After constructing a set of trees, each tree votes for the class of the object. Ultimately, the class with the most votes becomes the predicted class.

Advantages of ExtraTreesClassifier:

- Reduced Overfitting: The use of random splits and random features makes ExtraTreesClassifier less prone to overfitting compared to traditional decision trees.
- High Training Speed: ExtraTreesClassifier requires fewer computational resources for training compared to some other algorithms, such as gradient boosting. This makes it fast and efficient, especially for large datasets.
- Outlier Robustness: Thanks to the ensemble of trees and random splits, ExtraTreesClassifier is generally more robust to outliers in the data.

Limitations of ExtraTreesClassifier:

- Complex Interpretability: Analyzing and interpreting an ExtraTreesClassifier model can be challenging due to the large number of random splits and features.
- Parameter Tuning: Despite its efficiency, ExtraTreesClassifier may require careful hyperparameter tuning to achieve optimal performance.
- Not Always the Best Performer: In some tasks, ExtraTreesClassifier may be less accurate than other algorithms, such as gradient boosting.

ExtraTreesClassifier is a powerful classification algorithm known for its resistance to overfitting, high training speed, and robustness to outliers. It can be a valuable tool in data analysis and classification tasks, especially when dealing with large datasets that require efficient solutions. However, it's essential to note that the algorithm is not always the best choice, and its effectiveness may depend on the specific task and data.

**2.27.1. Code for Creating the ExtraTreesClassifier Model**

This code demonstrates the process of training an ExtraTreesClassifier model on the Iris dataset, exporting it to the ONNX format, and performing classification using the ONNX model. It also evaluates the accuracy of both the original model and the ONNX model.

```
# Iris_ExtraTreesClassifier.py
# The code demonstrates the process of training ExtraTrees Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original model and the ONNX model.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create an ExtraTreesClassifier model
extra_trees_model = ExtraTreesClassifier()

# train the model on the entire dataset
extra_trees_model.fit(X, y)

# predict classes for the entire dataset
y_pred = extra_trees_model.predict(X)

# evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy of ExtraTreesClassifier model:", accuracy)

# display the classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# define the input data type
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# export the model to ONNX format with float data type
onnx_model = convert_sklearn(extra_trees_model, initial_types=initial_type, target_opset=12)

# save the model to a file
onnx_filename = data_path + "extra_trees_iris.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# print model path
print(f"Model saved to {onnx_filename}")

# load the ONNX model and make predictions
onnx_session = ort.InferenceSession(onnx_filename)
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# display information about input tensors in ONNX
print("\nInformation about input tensors in ONNX:")
for i, input_tensor in enumerate(onnx_session.get_inputs()):
    print(f"{i + 1}. Name: {input_tensor.name}, Data Type: {input_tensor.type}, Shape: {input_tensor.shape}")

# display information about output tensors in ONNX
print("\nInformation about output tensors in ONNX:")
for i, output_tensor in enumerate(onnx_session.get_outputs()):
    print(f"{i + 1}. Name: {output_tensor.name}, Data Type: {output_tensor.type}, Shape: {output_tensor.shape}")

# convert data to floating-point format (float32)
X_float32 = X.astype(np.float32)

# predict classes for the entire dataset using ONNX
y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

# evaluate the accuracy of the ONNX model
accuracy_onnx = accuracy_score(y, y_pred_onnx)
print("\nAccuracy of ExtraTreesClassifier model in ONNX format:", accuracy_onnx)
```

Output:

Python    Accuracy of ExtraTreesClassifier model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\extra\_trees\_iris.onnx

Python

Python    Information about input tensors in ONNX:

Python    1. Name: float\_input, Data Type: tensor(float), Shape: \[None, 4\]

Python

Python    Information about output tensors in ONNX:

Python    1. Name: output\_label, Data Type: tensor(int64), Shape: \[None\]

Python    2. Name: output\_probability, Data Type: seq(map(int64,tensor(float))), Shape: \[\]

Python

Python    Accuracy of ExtraTreesClassifier model in ONNX format: 1.

**2.27.2. MQL5 Code for Working with the ExtraTreesClassifier Model**

```
//+------------------------------------------------------------------+
//|                                    Iris_ExtraTreesClassifier.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"
#resource "extra_trees_iris.onnx" as const uchar ExtModel[];

//+------------------------------------------------------------------+
//| Test IRIS dataset samples                                        |
//+------------------------------------------------------------------+
bool TestSamples(long model,float &input_data[][4], int &model_classes_id[])
  {
//--- check number of input samples
   ulong batch_size=input_data.Range(0);
   if(batch_size==0)
      return(false);
//--- prepare output array
   ArrayResize(model_classes_id,(int)batch_size);
//---
   float output_data[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } output_data_map[];
//--- check consistency
   bool res=ArrayResize(output_data,(int)batch_size)==batch_size;
//---
   if(res)
     {
      //--- set input shape
      ulong input_shape[]= {batch_size,input_data.Range(1)};
      OnnxSetInputShape(model,0,input_shape);
      //--- set output shapeы
      ulong output_shape1[]= {batch_size};
      ulong output_shape2[]= {batch_size};
      OnnxSetOutputShape(model,0,output_shape1);
      OnnxSetOutputShape(model,1,output_shape2);
      //--- run the model
      res=OnnxRun(model,0,input_data,output_data,output_data_map);
      //--- postprocessing
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         for(uint n=0; n<output_data_map.Size(); n++)
           {
            int model_class_id=-1;
            int max_idx=-1;
            float max_value=-1;
            //--- copy to arrays
            ArrayCopy(output_keys,output_data_map[n].key);
            ArrayCopy(output_values,output_data_map[n].value);
            //ArrayPrint(output_keys);
            //ArrayPrint(output_values);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
            //--- store the result to the output array
            model_classes_id[n]=model_class_id;
            //Print("model_class_id=",model_class_id);
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Test all samples from IRIS dataset (150)                         |
//| Here we test all samples with batch=1, sample by sample          |
//+------------------------------------------------------------------+
bool TestAllIrisDataset(const long model,const string model_name,double &model_accuracy)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("iris dataset not prepared");
      return(false);
     }
//--- show dataset
   for(int k=0; k<total_samples; k++)
     {
      //PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }
//--- array for output classes
   int model_output_classes_id[];
//--- check all Iris dataset samples
   int correct_results=0;
   for(int k=0; k<total_samples; k++)
     {
      //--- input array
      float iris_sample_input_data[1][4];
      //--- prepare input data from kth iris sample dataset
      iris_sample_input_data[0][0]=(float)iris_samples[k].features[0];
      iris_sample_input_data[0][1]=(float)iris_samples[k].features[1];
      iris_sample_input_data[0][2]=(float)iris_samples[k].features[2];
      iris_sample_input_data[0][3]=(float)iris_samples[k].features[3];
      //--- run model
      bool res=TestSamples(model,iris_sample_input_data,model_output_classes_id);
      //--- check result
      if(res)
        {
         if(model_output_classes_id[0]==iris_samples[k].class_id)
           {
            correct_results++;
           }
         else
           {
            PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_output_classes_id[0],iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
           }
        }
     }
   model_accuracy=1.0*correct_results/total_samples;
//---
   PrintFormat("model:%s   correct results: %.2f%%",model_name,100*model_accuracy);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Here we test batch execution of the model                        |
//+------------------------------------------------------------------+
bool TestBatchExecution(const long model,const string model_name,double &model_accuracy)
  {
   model_accuracy=0;
//--- array for output classes
   int model_output_classes_id[];
   int correct_results=0;
   int total_results=0;
   bool res=false;

//--- run batch with 3 samples
   float input_data_batch3[3][4]=
     {
        {5.1f,3.5f,1.4f,0.2f}, // iris dataset sample id=1, Iris-setosa
        {6.3f,2.5f,4.9f,1.5f}, // iris dataset sample id=73, Iris-versicolor
        {6.3f,2.7f,4.9f,1.8f}  // iris dataset sample id=124, Iris-virginica
     };
   int correct_classes_batch3[3]= {0,1,2};
//--- run model
   res=TestSamples(model,input_data_batch3,model_output_classes_id);
   if(res)
     {
      //--- check result
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         //--- check result
         if(model_output_classes_id[j]==correct_classes_batch3[j])
            correct_results++;
         else
           {
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch3[j],input_data_batch3[j][0],input_data_batch3[j][1],input_data_batch3[j][2],input_data_batch3[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- run batch with 10 samples
   float input_data_batch10[10][4]=
     {
        {5.5f,3.5f,1.3f,0.2f}, // iris dataset sample id=37 (Iris-setosa)
        {4.9f,3.1f,1.5f,0.1f}, // iris dataset sample id=38 (Iris-setosa)
        {4.4f,3.0f,1.3f,0.2f}, // iris dataset sample id=39 (Iris-setosa)
        {5.0f,3.3f,1.4f,0.2f}, // iris dataset sample id=50 (Iris-setosa)
        {7.0f,3.2f,4.7f,1.4f}, // iris dataset sample id=51 (Iris-versicolor)
        {6.4f,3.2f,4.5f,1.5f}, // iris dataset sample id=52 (Iris-versicolor)
        {6.3f,3.3f,6.0f,2.5f}, // iris dataset sample id=101 (Iris-virginica)
        {5.8f,2.7f,5.1f,1.9f}, // iris dataset sample id=102 (Iris-virginica)
        {7.1f,3.0f,5.9f,2.1f}, // iris dataset sample id=103 (Iris-virginica)
        {6.3f,2.9f,5.6f,1.8f}  // iris dataset sample id=104 (Iris-virginica)
     };
//--- correct classes for all 10 samples in the batch
   int correct_classes_batch10[10]= {0,0,0,0,1,1,2,2,2,2};

//--- run model
   res=TestSamples(model,input_data_batch10,model_output_classes_id);
//--- check result
   if(res)
     {
      for(int j=0; j<ArraySize(model_output_classes_id); j++)
        {
         if(model_output_classes_id[j]==correct_classes_batch10[j])
            correct_results++;
         else
           {
            double f1=input_data_batch10[j][0];
            double f2=input_data_batch10[j][1];
            double f3=input_data_batch10[j][2];
            double f4=input_data_batch10[j][3];
            PrintFormat("model:%s  FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f)",model_name,model_output_classes_id[j],correct_classes_batch10[j],input_data_batch10[j][0],input_data_batch10[j][1],input_data_batch10[j][2],input_data_batch10[j][3]);
           }
         total_results++;
        }
     }
   else
      return(false);

//--- calculate accuracy
   model_accuracy=correct_results/total_results;
//---
   return(res);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   string model_name="ExtraTreesClassifier";
//---
   long model=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model==INVALID_HANDLE)
     {
      PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
     }
   else
     {
      //--- test all dataset
      double model_accuracy=0;
      //-- test sample by sample execution for all Iris dataset
      if(TestAllIrisDataset(model,model_name,model_accuracy))
         PrintFormat("model=%s all samples accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- test batch execution for several samples
      if(TestBatchExecution(model,model_name,model_accuracy))
         PrintFormat("model=%s batch test accuracy=%f",model_name,model_accuracy);
      else
         PrintFormat("error in testing model=%s ",model_name);
      //--- release model
      OnnxRelease(model);
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_ExtraTreesClassifier (EURUSD,H1)   model:ExtraTreesClassifier   correct results: 100.00%
Iris_ExtraTreesClassifier (EURUSD,H1)   model=ExtraTreesClassifier all samples accuracy=1.000000
Iris_ExtraTreesClassifier (EURUSD,H1)   model=ExtraTreesClassifier batch test accuracy=1.000000
```

The accuracy of the exported ONNX model on the full Iris dataset is 100%, which corresponds to the accuracy of the original model.

**2.27.3. ONNX Representation of the ExtraTreesClassifier**

![Figure 41. ONNX Representation of the ExtraTrees Classifier in Netron](https://c.mql5.com/2/58/onnx_27_extra_trees_classifier.png)

Figure 41. ONNX Representation of the ExtraTrees Classifier in Netron

**2.28. Comparing the Accuracy of All Models**

Now, let's consider all the models together and compare their performance. First, we will perform the comparison using Python, and then we will load and execute the saved ONNX models in MetaTrader 5.

**2.28.1. Code for Calculating All Models and Building an Accuracy Comparison Chart**

This script calculates 27 classification models from the Scikit-learn package on the full Fisher's Iris dataset, exports the models to ONNX format, executes them, and compares the accuracy of the original and ONNX models.

```
# Iris_AllClassifiers.py
# The code demonstrates the process of training 27 Classifier models on the Iris dataset, exports them to ONNX format, and making predictions using the ONNX model.
# It also evaluates the accuracy of both the original and the ONNX models.
# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# import necessary libraries
from sklearn import datasets
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

# define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create and train each classifier model
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X, y)

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X, y)

from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X, y)

from sklearn.ensemble import AdaBoostClassifier
adaboost_model = AdaBoostClassifier(random_state=42)
adaboost_model.fit(X, y)

from sklearn.ensemble import BaggingClassifier
bagging_model = BaggingClassifier(random_state=42)
bagging_model.fit(X, y)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X, y)

from sklearn.neighbors import RadiusNeighborsClassifier
radius_neighbors_model = RadiusNeighborsClassifier(radius=1.0)
radius_neighbors_model.fit(X, y)

from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X, y)

from sklearn.linear_model import LogisticRegression
logistic_regression_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_regression_model.fit(X, y)

from sklearn.linear_model import RidgeClassifier
ridge_classifier_model = RidgeClassifier(random_state=42)
ridge_classifier_model.fit(X, y)

from sklearn.linear_model import PassiveAggressiveClassifier
passive_aggressive_model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
passive_aggressive_model.fit(X, y)

from sklearn.linear_model import Perceptron
perceptron_model = Perceptron(max_iter=1000, random_state=42)
perceptron_model.fit(X, y)

from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier(max_iter=1000, random_state=42)
sgd_model.fit(X, y)

from sklearn.naive_bayes import GaussianNB
gaussian_nb_model = GaussianNB()
gaussian_nb_model.fit(X, y)

from sklearn.naive_bayes import MultinomialNB
multinomial_nb_model = MultinomialNB()
multinomial_nb_model.fit(X, y)

from sklearn.naive_bayes import ComplementNB
complement_nb_model = ComplementNB()
complement_nb_model.fit(X, y)

from sklearn.naive_bayes import BernoulliNB
bernoulli_nb_model = BernoulliNB()
bernoulli_nb_model.fit(X, y)

from sklearn.naive_bayes import CategoricalNB
categorical_nb_model = CategoricalNB()
categorical_nb_model.fit(X, y)

from sklearn.tree import ExtraTreeClassifier
extra_tree_model = ExtraTreeClassifier(random_state=42)
extra_tree_model.fit(X, y)

from sklearn.ensemble import ExtraTreesClassifier
extra_trees_model = ExtraTreesClassifier(random_state=42)
extra_trees_model.fit(X, y)

from sklearn.svm import LinearSVC  # Import LinearSVC
linear_svc_model = LinearSVC(random_state=42)
linear_svc_model.fit(X, y)

from sklearn.svm import NuSVC
nu_svc_model = NuSVC()
nu_svc_model.fit(X, y)

from sklearn.linear_model import LogisticRegressionCV
logistic_regression_cv_model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
logistic_regression_cv_model.fit(X, y)

from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(max_iter=1000, random_state=42)
mlp_model.fit(X, y)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X, y)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hist_gradient_boosting_model = HistGradientBoostingClassifier(random_state=42)
hist_gradient_boosting_model.fit(X, y)

from sklearn.linear_model import RidgeClassifierCV
ridge_classifier_cv_model = RidgeClassifierCV()
ridge_classifier_cv_model.fit(X, y)

# define a dictionary to store results
results = {}

# loop through the models
for model_name, classifier_model in [\
    ('SVC Classifier', svc_model),\
    ('Random Forest Classifier', random_forest_model),\
    ('Gradient Boosting Classifier', gradient_boosting_model),\
    ('AdaBoost Classifier', adaboost_model),\
    ('Bagging Classifier', bagging_model),\
    ('K-NN Classifier', knn_model),\
    ('Radius Neighbors Classifier', radius_neighbors_model),\
    ('Decision Tree Classifier', decision_tree_model),\
    ('Logistic Regression Classifier', logistic_regression_model),\
    ('Ridge Classifier', ridge_classifier_model),\
    ('Ridge ClassifierCV', ridge_classifier_cv_model),\
    ('Passive-Aggressive Classifier', passive_aggressive_model),\
    ('Perceptron Classifier', perceptron_model),\
    ('SGD Classifier', sgd_model),\
    ('Gaussian Naive Bayes Classifier', gaussian_nb_model),\
    ('Multinomial Naive Bayes Classifier', multinomial_nb_model),\
    ('Complement Naive Bayes Classifier', complement_nb_model),\
    ('Bernoulli Naive Bayes Classifier', bernoulli_nb_model),\
    ('Categorical Naive Bayes Classifier', categorical_nb_model),\
    ('Extra Tree Classifier', extra_tree_model),\
    ('Extra Trees Classifier', extra_trees_model),\
    ('LinearSVC Classifier', linear_svc_model),\
    ('NuSVC Classifier', nu_svc_model),\
    ('Logistic RegressionCV Classifier', logistic_regression_cv_model),\
    ('MLP Classifier', mlp_model),\
    ('Linear Discriminant Analysis Classifier', lda_model),\
    ('Hist Gradient Boosting Classifier', hist_gradient_boosting_model)\
]:
    # predict classes for the entire dataset
    y_pred = classifier_model.predict(X)

    # evaluate the model's accuracy
    accuracy = accuracy_score(y, y_pred)

    # define the input data type
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

    # export the model to ONNX format with float data type
    onnx_model = convert_sklearn(classifier_model, initial_types=initial_type, target_opset=12)

    # save the model to a file
    onnx_filename = data_path + f"{model_name.lower().replace(' ', '_')}_iris.onnx"
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # load the ONNX model and make predictions
    onnx_session = ort.InferenceSession(onnx_filename)
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # convert data to floating-point format (float32)
    X_float32 = X.astype(np.float32)

    # predict classes for the entire dataset using ONNX
    y_pred_onnx = onnx_session.run([output_name], {input_name: X_float32})[0]

    # evaluate the accuracy of the ONNX model
    accuracy_onnx = accuracy_score(y, y_pred_onnx)

    # store results
    results[model_name] = {
        'accuracy': accuracy,
        'accuracy_onnx': accuracy_onnx
    }

    # print the accuracy of the original model and the ONNX model
    #print(f"{model_name} - Original Accuracy: {accuracy}, ONNX Accuracy: {accuracy_onnx}")

# sort the models based on accuracy
sorted_results = dict(sorted(results.items(), key=lambda item: item[1]['accuracy'], reverse=True))

# print the sorted results
print("Sorted Results:")
for model_name, metrics in sorted_results.items():
    print(f"{model_name} - Original Accuracy: {metrics['accuracy']:.4f}, ONNX Accuracy: {metrics['accuracy_onnx']:.4f}")

# create comparison plots for sorted results
fig, ax = plt.subplots(figsize=(12, 8))

model_names = list(sorted_results.keys())
accuracies = [sorted_results[model_name]['accuracy'] for model_name in model_names]
accuracies_onnx = [sorted_results[model_name]['accuracy_onnx'] for model_name in model_names]

bar_width = 0.35
index = range(len(model_names))

bar1 = plt.bar(index, accuracies, bar_width, label='Model Accuracy')
bar2 = plt.bar([i + bar_width for i in index], accuracies_onnx, bar_width, label='ONNX Accuracy')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model and ONNX Accuracy (Sorted)')
plt.xticks([i + bar_width / 2 for i in index], model_names, rotation=90, ha='center')
plt.legend()

plt.tight_layout()
plt.show()
```

Output:

```
Python  Sorted Results:
Python  Random Forest Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
Python  Gradient Boosting Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
Python  Bagging Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
Python  Decision Tree Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
Python  Extra Tree Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
Python  Extra Trees Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
Python  Hist Gradient Boosting Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
Python  Logistic RegressionCV Classifier - Original Accuracy: 0.9800, ONNX Accuracy: 0.9800
Python  MLP Classifier - Original Accuracy: 0.9800, ONNX Accuracy: 0.9800
Python  Linear Discriminant Analysis Classifier - Original Accuracy: 0.9800, ONNX Accuracy: 0.9800
Python  SVC Classifier - Original Accuracy: 0.9733, ONNX Accuracy: 0.9733
Python  Radius Neighbors Classifier - Original Accuracy: 0.9733, ONNX Accuracy: 0.9733
Python  Logistic Regression Classifier - Original Accuracy: 0.9733, ONNX Accuracy: 0.9733
Python  NuSVC Classifier - Original Accuracy: 0.9733, ONNX Accuracy: 0.9733
Python  K-NN Classifier - Original Accuracy: 0.9667, ONNX Accuracy: 0.9667
Python  LinearSVC Classifier - Original Accuracy: 0.9667, ONNX Accuracy: 0.9667
Python  AdaBoost Classifier - Original Accuracy: 0.9600, ONNX Accuracy: 0.9600
Python  Passive-Aggressive Classifier - Original Accuracy: 0.9600, ONNX Accuracy: 0.9600
Python  Gaussian Naive Bayes Classifier - Original Accuracy: 0.9600, ONNX Accuracy: 0.9600
Python  Multinomial Naive Bayes Classifier - Original Accuracy: 0.9533, ONNX Accuracy: 0.9533
Python  SGD Classifier - Original Accuracy: 0.9333, ONNX Accuracy: 0.9333
Python  Categorical Naive Bayes Classifier - Original Accuracy: 0.9333, ONNX Accuracy: 0.9333
Python  Ridge Classifier - Original Accuracy: 0.8533, ONNX Accuracy: 0.8533
Python  Ridge ClassifierCV - Original Accuracy: 0.8533, ONNX Accuracy: 0.8533
Python  Complement Naive Bayes Classifier - Original Accuracy: 0.6667, ONNX Accuracy: 0.6667
Python  Perceptron Classifier - Original Accuracy: 0.6133, ONNX Accuracy: 0.6133
Python  Bernoulli Naive Bayes Classifier - Original Accuracy: 0.3333, ONNX Accuracy: 0.3333
```

The script will also generate an image with summary results for all 27 models.

![Figure 42. Comparison of Accuracy for 27 Classification Models and Their ONNX Versions on the Iris Dataset](https://c.mql5.com/2/58/ALL_27_models_sorted__2.png)

Figure 42. Comparison of Accuracy for 27 Classification Models and Their ONNX Versions on the Iris Dataset

Based on the accuracy evaluation results of the original models and their ONNX versions, the following conclusions can be drawn:

Seven models showed perfect accuracy (1.0000) in both the original and ONNX versions. These models include:

1. Random Forest Classifier
2. Gradient Boosting Classifier
3. Bagging Classifier
4. Decision Tree Classifier
5. Extra Tree Classifier
6. Extra Trees Classifier
7. Hist Gradient Boosting Classifier

The ONNX representations of these models also maintain high accuracy.

Three models - Logistic RegressionCV Classifier, MLP Classifier, and Linear Discriminant Analysis Classifier - achieved high accuracy in both the original and ONNX versions, with an accuracy of 0.9800. These models perform well in both representations.

Several models, including SVC Classifier, Radius Neighbors Classifier, NuSVC Classifier, K-NN Classifier, LinearSVC Classifier, AdaBoost Classifier, Passive-Aggressive Classifier, Gaussian Naive Bayes Classifier, and Multinomial Naive Bayes Classifier, showed good accuracy in both the original and ONNX versions, with accuracy scores of 0.9733, 0.9667, or 0.9600. These models also maintain their accuracy in the ONNX representation.

Models such as SGD Classifier, Categorical Naive Bayes Classifier, Ridge Classifier, Complement Naive Bayes Classifier, Perceptron Classifier, and Bernoulli Naive Bayes Classifier have lower accuracy but still perform well in maintaining accuracy in ONNX.

All the models considered maintain their accuracy when exported to ONNX format, indicating that ONNX provides an efficient way to save and restore machine learning models. However, it's important to remember that the quality of the exported model can depend on the specific model algorithm and parameters.

**2.28.2. MQL5 Code for Executing All ONNX Models**

This script executes all the ONNX models saved by the script in [2.28.1](https://www.mql5.com/en/articles/13451#model_2_28_1) on the full Fisher's Iris dataset.

```
//+------------------------------------------------------------------+
//|                                          Iris_AllClassifiers.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include "iris.mqh"

//+------------------------------------------------------------------+
//| TestSampleSequenceMapOutput                                      |
//+------------------------------------------------------------------+
bool TestSampleSequenceMapOutput(long model,sIRISsample &iris_sample, int &model_class_id)
  {
//---
   model_class_id=-1;
   float input_data[1][4];
   for(int k=0; k<4; k++)
     {
      input_data[0][k]=(float)iris_sample.features[k];
     }
//---
   float out1[];
//---
   struct Map
     {
      ulong          key[];
      float          value[];
     } out2[];
//---
   bool res=ArrayResize(out1,input_data.Range(0))==input_data.Range(0);
//---
   if(res)
     {
      ulong input_shape[]= { input_data.Range(0), input_data.Range(1) };
      ulong output_shape[]= { input_data.Range(0) };
      //---
      OnnxSetInputShape(model,0,input_shape);
      OnnxSetOutputShape(model,0,output_shape);
      //---
      res=OnnxRun(model,0,input_data,out1,out2);
      //---
      if(res)
        {
         //--- postprocessing of sequence map data
         //--- find class with maximum probability
         ulong output_keys[];
         float output_values[];
         //---
         model_class_id=-1;
         int max_idx=-1;
         float max_value=-1;
         //---
         for(uint n=0; n<out2.Size(); n++)
           {
            //--- copy to arrays
            ArrayCopy(output_keys,out2[n].key);
            ArrayCopy(output_values,out2[n].value);
            //--- find the key with maximum probability
            for(int k=0; k<ArraySize(output_values); k++)
              {
               if(k==0)
                 {
                  max_idx=0;
                  max_value=output_values[max_idx];
                  model_class_id=(int)output_keys[max_idx];
                 }
               else
                 {
                  if(output_values[k]>max_value)
                    {
                     max_idx=k;
                     max_value=output_values[max_idx];
                     model_class_id=(int)output_keys[max_idx];
                    }
                 }
              }
           }
        }
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| TestSampleTensorOutput                                           |
//+------------------------------------------------------------------+
bool TestSampleTensorOutput(long model,sIRISsample &iris_sample, int &model_class_id)
  {
//---
   model_class_id=-1;
   float input_data[1][4];
   for(int k=0; k<4; k++)
     {
      input_data[0][k]=(float)iris_sample.features[k];
     }
//---
   ulong input_shape[]= { 1, 4};
   OnnxSetInputShape(model,0,input_shape);
//---
   int output1[1];
   float output2[1,3];
//---
   ulong output_shape[]= {1};
   OnnxSetOutputShape(model,0,output_shape);
//---
   ulong output_shape2[]= {1,3};
   OnnxSetOutputShape(model,1,output_shape2);
//---
   bool res=OnnxRun(model,0,input_data,output1,output2);
//--- class for these models in output1[0];
   if(res)
      model_class_id=output1[0];
//---
   return(res);
  }

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   sIRISsample iris_samples[];
//--- load dataset from file
   PrepareIrisDataset(iris_samples);
//--- test
   int total_samples=ArraySize(iris_samples);
   if(total_samples==0)
     {
      Print("error in loading iris dataset from iris.csv");
      return(false);
     }
   /*for(int k=0; k<total_samples; k++)
     {
      PrintFormat("%d (%.2f,%.2f,%.2f,%.2f) class %d (%s)",iris_samples[k].sample_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3],iris_samples[k].class_id,iris_samples[k].class_name);
     }*/
//----

   string iris_models[]=
     {
      "random_forest_classifier_iris.onnx",
      "gradient_boosting_classifier_iris.onnx",
      "bagging_classifier_iris.onnx",
      "decision_tree_classifier_iris.onnx",
      "extra_tree_classifier_iris.onnx",
      "extra_trees_classifier_iris.onnx",
      "hist_gradient_boosting_classifier_iris.onnx",
      "logistic_regressioncv_classifier_iris.onnx",
      "mlp_classifier_iris.onnx",
      "linear_discriminant_analysis_classifier_iris.onnx",
      "svc_classifier_iris.onnx",
      "radius_neighbors_classifier_iris.onnx",
      "logistic_regression_classifier_iris.onnx",
      "nusvc_classifier_iris.onnx",
      "k-nn_classifier_iris.onnx",
      "linearsvc_classifier_iris.onnx",
      "adaboost_classifier_iris.onnx",
      "passive-aggressive_classifier_iris.onnx",
      "gaussian_naive_bayes_classifier_iris.onnx",
      "multinomial_naive_bayes_classifier_iris.onnx",
      "sgd_classifier_iris.onnx",
      "categorical_naive_bayes_classifier_iris.onnx",
      "ridge_classifier_iris.onnx",
      "ridge_classifiercv_iris.onnx",
      "complement_naive_bayes_classifier_iris.onnx",
      "perceptron_classifier_iris.onnx",
      "bernoulli_naive_bayes_classifier_iris.onnx"
     };

//--- test all iris dataset sample by sample
   for(int i=0; i<ArraySize(iris_models); i++)
     {
      //--- load ONNX-model
      string model_name="IRIS_models\\"+iris_models[i];
      //---
      long model=OnnxCreate(model_name,0);
      if(model==INVALID_HANDLE)
        {
         PrintFormat("model_name=%s OnnxCreate error %d for",model_name,GetLastError());
        }
      else
        {
         //--- check all samples
         int correct_results=0;
         for(int k=0; k<total_samples; k++)
           {
            int model_class_id=-1;
            //--- select data output processor
            string current_model=iris_models[i];
            if(current_model=="svc_classifier_iris.onnx" || current_model=="linearsvc_classifier_iris.onnx" || current_model=="nusvc_classifier_iris.onnx" || current_model=="ridge_classifier_iris.onnx" || current_model=="ridge_classifiercv_iris.onnx" || current_model=="radius_neighbors_classifier_iris.onnx")
              {
               TestSampleTensorOutput(model,iris_samples[k],model_class_id);
              }
            else
              {
               TestSampleSequenceMapOutput(model,iris_samples[k],model_class_id);
              }
            //---
            if(model_class_id==iris_samples[k].class_id)
              {
               correct_results++;
               //PrintFormat("sample=%d OK [class=%d]",iris_samples[k].sample_id,model_class_id);
              }
            else
              {
               //PrintFormat("model:%s  sample=%d FAILED [class=%d, true class=%d] features=(%.2f,%.2f,%.2f,%.2f]",model_name,iris_samples[k].sample_id,model_class_id,iris_samples[k].class_id,iris_samples[k].features[0],iris_samples[k].features[1],iris_samples[k].features[2],iris_samples[k].features[3]);
              }
           }
         PrintFormat("%d model:%s   accuracy: %.4f",i+1,model_name,1.0*correct_results/total_samples);
         //--- release model
         OnnxRelease(model);
        }
      //---
     }
   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

```
Iris_AllClassifiers (EURUSD,H1) 1 model:IRIS_models\random_forest_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 2 model:IRIS_models\gradient_boosting_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 3 model:IRIS_models\bagging_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 4 model:IRIS_models\decision_tree_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 5 model:IRIS_models\extra_tree_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 6 model:IRIS_models\extra_trees_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 7 model:IRIS_models\hist_gradient_boosting_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 8 model:IRIS_models\logistic_regressioncv_classifier_iris.onnx   accuracy: 0.9800
Iris_AllClassifiers (EURUSD,H1) 9 model:IRIS_models\mlp_classifier_iris.onnx   accuracy: 0.9800
Iris_AllClassifiers (EURUSD,H1) 10 model:IRIS_models\linear_discriminant_analysis_classifier_iris.onnx   accuracy: 0.9800
Iris_AllClassifiers (EURUSD,H1) 11 model:IRIS_models\svc_classifier_iris.onnx   accuracy: 0.9733
Iris_AllClassifiers (EURUSD,H1) 12 model:IRIS_models\radius_neighbors_classifier_iris.onnx   accuracy: 0.9733
Iris_AllClassifiers (EURUSD,H1) 13 model:IRIS_models\logistic_regression_classifier_iris.onnx   accuracy: 0.9733
Iris_AllClassifiers (EURUSD,H1) 14 model:IRIS_models\nusvc_classifier_iris.onnx   accuracy: 0.9733
Iris_AllClassifiers (EURUSD,H1) 15 model:IRIS_models\k-nn_classifier_iris.onnx   accuracy: 0.9667
Iris_AllClassifiers (EURUSD,H1) 16 model:IRIS_models\linearsvc_classifier_iris.onnx   accuracy: 0.9667
Iris_AllClassifiers (EURUSD,H1) 17 model:IRIS_models\adaboost_classifier_iris.onnx   accuracy: 0.9600
Iris_AllClassifiers (EURUSD,H1) 18 model:IRIS_models\passive-aggressive_classifier_iris.onnx   accuracy: 0.9600
Iris_AllClassifiers (EURUSD,H1) 19 model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx   accuracy: 0.9600
Iris_AllClassifiers (EURUSD,H1) 20 model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx   accuracy: 0.9533
Iris_AllClassifiers (EURUSD,H1) 21 model:IRIS_models\sgd_classifier_iris.onnx   accuracy: 0.9333
Iris_AllClassifiers (EURUSD,H1) 22 model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx   accuracy: 0.9333
Iris_AllClassifiers (EURUSD,H1) 23 model:IRIS_models\ridge_classifier_iris.onnx   accuracy: 0.8533
Iris_AllClassifiers (EURUSD,H1) 24 model:IRIS_models\ridge_classifiercv_iris.onnx   accuracy: 0.8533
Iris_AllClassifiers (EURUSD,H1) ONNX: Removing initializer 'class_log_prior'. It is not used by any node and should be removed from the model.
Iris_AllClassifiers (EURUSD,H1) 25 model:IRIS_models\complement_naive_bayes_classifier_iris.onnx   accuracy: 0.6667
Iris_AllClassifiers (EURUSD,H1) 26 model:IRIS_models\perceptron_classifier_iris.onnx   accuracy: 0.6133
Iris_AllClassifiers (EURUSD,H1) 27 model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx   accuracy: 0.3333
```

Comparing with the results of script 2.28.1.1:

```
Python  Random Forest Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
1 model:IRIS_models\random_forest_classifier_iris.onnx   accuracy: 1.0000

Python  Gradient Boosting Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
2 model:IRIS_models\gradient_boosting_classifier_iris.onnx   accuracy: 1.0000

Python  Bagging Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
3 model:IRIS_models\bagging_classifier_iris.onnx   accuracy: 1.0000

Python  Decision Tree Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
4 model:IRIS_models\decision_tree_classifier_iris.onnx   accuracy: 1.0000

Python  Extra Tree Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
5 model:IRIS_models\extra_tree_classifier_iris.onnx   accuracy: 1.0000

Python  Extra Trees Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
6 model:IRIS_models\extra_trees_classifier_iris.onnx   accuracy: 1.0000

Python  Hist Gradient Boosting Classifier - Original Accuracy: 1.0000, ONNX Accuracy: 1.0000
7 model:IRIS_models\hist_gradient_boosting_classifier_iris.onnx   accuracy: 1.0000

Python  Logistic RegressionCV Classifier - Original Accuracy: 0.9800, ONNX Accuracy: 0.9800
8 model:IRIS_models\logistic_regressioncv_classifier_iris.onnx   accuracy: 0.9800

Python  MLP Classifier - Original Accuracy: 0.9800, ONNX Accuracy: 0.9800
9 model:IRIS_models\mlp_classifier_iris.onnx   accuracy: 0.9800

Python  Linear Discriminant Analysis Classifier - Original Accuracy: 0.9800, ONNX Accuracy: 0.9800
10 model:IRIS_models\linear_discriminant_analysis_classifier_iris.onnx   accuracy: 0.9800

Python  SVC Classifier - Original Accuracy: 0.9733, ONNX Accuracy: 0.9733
11 model:IRIS_models\svc_classifier_iris.onnx   accuracy: 0.9733

Python  Radius Neighbors Classifier - Original Accuracy: 0.9733, ONNX Accuracy: 0.9733
12 model:IRIS_models\radius_neighbors_classifier_iris.onnx   accuracy: 0.9733

Python  Logistic Regression Classifier - Original Accuracy: 0.9733, ONNX Accuracy: 0.9733
13 model:IRIS_models\logistic_regression_classifier_iris.onnx   accuracy: 0.9733

Python  NuSVC Classifier - Original Accuracy: 0.9733, ONNX Accuracy: 0.9733
14 model:IRIS_models\nusvc_classifier_iris.onnx   accuracy: 0.9733

Python  K-NN Classifier - Original Accuracy: 0.9667, ONNX Accuracy: 0.9667
15 model:IRIS_models\k-nn_classifier_iris.onnx   accuracy: 0.9667

Python  LinearSVC Classifier - Original Accuracy: 0.9667, ONNX Accuracy: 0.9667
16 model:IRIS_models\linearsvc_classifier_iris.onnx   accuracy: 0.9667

Python  AdaBoost Classifier - Original Accuracy: 0.9600, ONNX Accuracy: 0.9600
17 model:IRIS_models\adaboost_classifier_iris.onnx   accuracy: 0.9600

Python  Passive-Aggressive Classifier - Original Accuracy: 0.9600, ONNX Accuracy: 0.9600
18 model:IRIS_models\passive-aggressive_classifier_iris.onnx   accuracy: 0.9600

Python  Gaussian Naive Bayes Classifier - Original Accuracy: 0.9600, ONNX Accuracy: 0.9600
19 model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx   accuracy: 0.9600

Python  Multinomial Naive Bayes Classifier - Original Accuracy: 0.9533, ONNX Accuracy: 0.9533
20 model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx   accuracy: 0.9533

Python  SGD Classifier - Original Accuracy: 0.9333, ONNX Accuracy: 0.9333
21 model:IRIS_models\sgd_classifier_iris.onnx   accuracy: 0.9333

Python  Categorical Naive Bayes Classifier - Original Accuracy: 0.9333, ONNX Accuracy: 0.9333
22 model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx   accuracy: 0.9333

Python  Ridge Classifier - Original Accuracy: 0.8533, ONNX Accuracy: 0.8533
23 model:IRIS_models\ridge_classifier_iris.onnx   accuracy: 0.8533

Python  Ridge ClassifierCV - Original Accuracy: 0.8533, ONNX Accuracy: 0.8533
24 model:IRIS_models\ridge_classifiercv_iris.onnx   accuracy: 0.8533

Python  Complement Naive Bayes Classifier - Original Accuracy: 0.6667, ONNX Accuracy: 0.6667
25 model:IRIS_models\complement_naive_bayes_classifier_iris.onnx   accuracy: 0.6667

Python  Perceptron Classifier - Original Accuracy: 0.6133, ONNX Accuracy: 0.6133
26 model:IRIS_models\perceptron_classifier_iris.onnx   accuracy: 0.6133

Python  Bernoulli Naive Bayes Classifier - Original Accuracy: 0.3333, ONNX Accuracy: 0.3333
27 model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx   accuracy: 0.3333
```

it's worth noting that executing all the saved ONNX models in MQL5 corresponds completely to the results of [2.28.1](https://www.mql5.com/en/articles/13451#model_2_28_1).

Thus, the models we examined, converted to the ONNX format, preserved their classification accuracy.

It is worth mentioning that seven models achieved perfect classification accuracy (accuracy=1.0) for the iris dataset:

1. Random Forest Classifier;
2. Gradient Boosting Classifier;
3. Bagging Classifier;
4. Decision Tree Classifier;
5. Extra Tree Classifier;
6. Extra Trees Classifier;
7. Histogram Gradient Boosting Classifier.

The remaining 20 models made classification errors.

If you uncomment line 208, the script will also display the iris dataset samples that were misclassified by each of the models:

```
Iris_AllClassifiers (EURUSD,H1) 1 model:IRIS_models\random_forest_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 2 model:IRIS_models\gradient_boosting_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 3 model:IRIS_models\bagging_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 4 model:IRIS_models\decision_tree_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 5 model:IRIS_models\extra_tree_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 6 model:IRIS_models\extra_trees_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) 7 model:IRIS_models\hist_gradient_boosting_classifier_iris.onnx   accuracy: 1.0000
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\logistic_regressioncv_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\logistic_regressioncv_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\logistic_regressioncv_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) 8 model:IRIS_models\logistic_regressioncv_classifier_iris.onnx   accuracy: 0.9800
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\mlp_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\mlp_classifier_iris.onnx  sample=73 FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\mlp_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) 9 model:IRIS_models\mlp_classifier_iris.onnx   accuracy: 0.9800
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\linear_discriminant_analysis_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\linear_discriminant_analysis_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\linear_discriminant_analysis_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) 10 model:IRIS_models\linear_discriminant_analysis_classifier_iris.onnx   accuracy: 0.9800
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\svc_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\svc_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\svc_classifier_iris.onnx  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\svc_classifier_iris.onnx  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) 11 model:IRIS_models\svc_classifier_iris.onnx   accuracy: 0.9733
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\radius_neighbors_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\radius_neighbors_classifier_iris.onnx  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\radius_neighbors_classifier_iris.onnx  sample=127 FAILED [class=1, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\radius_neighbors_classifier_iris.onnx  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) 12 model:IRIS_models\radius_neighbors_classifier_iris.onnx   accuracy: 0.9733
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\logistic_regression_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\logistic_regression_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\logistic_regression_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\logistic_regression_classifier_iris.onnx  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) 13 model:IRIS_models\logistic_regression_classifier_iris.onnx   accuracy: 0.9733
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\nusvc_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\nusvc_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\nusvc_classifier_iris.onnx  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\nusvc_classifier_iris.onnx  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) 14 model:IRIS_models\nusvc_classifier_iris.onnx   accuracy: 0.9733
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\k-nn_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\k-nn_classifier_iris.onnx  sample=73 FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\k-nn_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\k-nn_classifier_iris.onnx  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\k-nn_classifier_iris.onnx  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AllClassifiers (EURUSD,H1) 15 model:IRIS_models\k-nn_classifier_iris.onnx   accuracy: 0.9667
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\linearsvc_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\linearsvc_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\linearsvc_classifier_iris.onnx  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\linearsvc_classifier_iris.onnx  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\linearsvc_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) 16 model:IRIS_models\linearsvc_classifier_iris.onnx   accuracy: 0.9667
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\adaboost_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\adaboost_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\adaboost_classifier_iris.onnx  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\adaboost_classifier_iris.onnx  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\adaboost_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\adaboost_classifier_iris.onnx  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_AllClassifiers (EURUSD,H1) 17 model:IRIS_models\adaboost_classifier_iris.onnx   accuracy: 0.9600
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\passive-aggressive_classifier_iris.onnx  sample=67 FAILED [class=2, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\passive-aggressive_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\passive-aggressive_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\passive-aggressive_classifier_iris.onnx  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\passive-aggressive_classifier_iris.onnx  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\passive-aggressive_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) 18 model:IRIS_models\passive-aggressive_classifier_iris.onnx   accuracy: 0.9600
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx  sample=53 FAILED [class=2, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) 19 model:IRIS_models\gaussian_naive_bayes_classifier_iris.onnx   accuracy: 0.9600
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx  sample=69 FAILED [class=2, true class=1] features=(6.20,2.20,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx  sample=73 FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx  sample=132 FAILED [class=1, true class=2] features=(7.90,3.80,6.40,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) 20 model:IRIS_models\multinomial_naive_bayes_classifier_iris.onnx   accuracy: 0.9533
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=65 FAILED [class=0, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=86 FAILED [class=0, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=124 FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=127 FAILED [class=1, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\sgd_classifier_iris.onnx  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_AllClassifiers (EURUSD,H1) 21 model:IRIS_models\sgd_classifier_iris.onnx   accuracy: 0.9333
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=102 FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=122 FAILED [class=1, true class=2] features=(5.60,2.80,4.90,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=124 FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=127 FAILED [class=1, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=128 FAILED [class=1, true class=2] features=(6.10,3.00,4.90,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx  sample=143 FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_AllClassifiers (EURUSD,H1) 22 model:IRIS_models\categorical_naive_bayes_classifier_iris.onnx   accuracy: 0.9333
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=51 FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=52 FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=53 FAILED [class=2, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=57 FAILED [class=2, true class=1] features=(6.30,3.30,4.70,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=62 FAILED [class=2, true class=1] features=(5.90,3.00,4.20,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=65 FAILED [class=2, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=66 FAILED [class=2, true class=1] features=(6.70,3.10,4.40,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=67 FAILED [class=2, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=76 FAILED [class=2, true class=1] features=(6.60,3.00,4.40,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=79 FAILED [class=2, true class=1] features=(6.00,2.90,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=86 FAILED [class=2, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=87 FAILED [class=2, true class=1] features=(6.70,3.10,4.70,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=89 FAILED [class=2, true class=1] features=(5.60,3.00,4.10,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=92 FAILED [class=2, true class=1] features=(6.10,3.00,4.60,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=109 FAILED [class=1, true class=2] features=(6.70,2.50,5.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifier_iris.onnx  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_AllClassifiers (EURUSD,H1) 23 model:IRIS_models\ridge_classifier_iris.onnx   accuracy: 0.8533
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=51 FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=52 FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=53 FAILED [class=2, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=57 FAILED [class=2, true class=1] features=(6.30,3.30,4.70,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=62 FAILED [class=2, true class=1] features=(5.90,3.00,4.20,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=65 FAILED [class=2, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=66 FAILED [class=2, true class=1] features=(6.70,3.10,4.40,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=67 FAILED [class=2, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=76 FAILED [class=2, true class=1] features=(6.60,3.00,4.40,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=79 FAILED [class=2, true class=1] features=(6.00,2.90,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=86 FAILED [class=2, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=87 FAILED [class=2, true class=1] features=(6.70,3.10,4.70,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=89 FAILED [class=2, true class=1] features=(5.60,3.00,4.10,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=92 FAILED [class=2, true class=1] features=(6.10,3.00,4.60,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=109 FAILED [class=1, true class=2] features=(6.70,2.50,5.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\ridge_classifiercv_iris.onnx  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_AllClassifiers (EURUSD,H1) 24 model:IRIS_models\ridge_classifiercv_iris.onnx   accuracy: 0.8533
Iris_AllClassifiers (EURUSD,H1) ONNX: Removing initializer 'class_log_prior'. It is not used by any node and should be removed from the model.
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=51 FAILED [class=2, true class=1] features=(7.00,3.20,4.70,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=52 FAILED [class=2, true class=1] features=(6.40,3.20,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=53 FAILED [class=2, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=54 FAILED [class=2, true class=1] features=(5.50,2.30,4.00,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=55 FAILED [class=2, true class=1] features=(6.50,2.80,4.60,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=56 FAILED [class=2, true class=1] features=(5.70,2.80,4.50,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=57 FAILED [class=2, true class=1] features=(6.30,3.30,4.70,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=58 FAILED [class=2, true class=1] features=(4.90,2.40,3.30,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=59 FAILED [class=2, true class=1] features=(6.60,2.90,4.60,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=60 FAILED [class=2, true class=1] features=(5.20,2.70,3.90,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=61 FAILED [class=2, true class=1] features=(5.00,2.00,3.50,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=62 FAILED [class=2, true class=1] features=(5.90,3.00,4.20,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=63 FAILED [class=2, true class=1] features=(6.00,2.20,4.00,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=64 FAILED [class=2, true class=1] features=(6.10,2.90,4.70,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=65 FAILED [class=2, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=66 FAILED [class=2, true class=1] features=(6.70,3.10,4.40,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=67 FAILED [class=2, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=68 FAILED [class=2, true class=1] features=(5.80,2.70,4.10,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=69 FAILED [class=2, true class=1] features=(6.20,2.20,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=70 FAILED [class=2, true class=1] features=(5.60,2.50,3.90,1.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=71 FAILED [class=2, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=72 FAILED [class=2, true class=1] features=(6.10,2.80,4.00,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=73 FAILED [class=2, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=74 FAILED [class=2, true class=1] features=(6.10,2.80,4.70,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=75 FAILED [class=2, true class=1] features=(6.40,2.90,4.30,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=76 FAILED [class=2, true class=1] features=(6.60,3.00,4.40,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=77 FAILED [class=2, true class=1] features=(6.80,2.80,4.80,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=78 FAILED [class=2, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=79 FAILED [class=2, true class=1] features=(6.00,2.90,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=80 FAILED [class=0, true class=1] features=(5.70,2.60,3.50,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=81 FAILED [class=2, true class=1] features=(5.50,2.40,3.80,1.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=82 FAILED [class=2, true class=1] features=(5.50,2.40,3.70,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=83 FAILED [class=2, true class=1] features=(5.80,2.70,3.90,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=84 FAILED [class=2, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=85 FAILED [class=2, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=86 FAILED [class=2, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=87 FAILED [class=2, true class=1] features=(6.70,3.10,4.70,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=88 FAILED [class=2, true class=1] features=(6.30,2.30,4.40,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=89 FAILED [class=2, true class=1] features=(5.60,3.00,4.10,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=90 FAILED [class=2, true class=1] features=(5.50,2.50,4.00,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=91 FAILED [class=2, true class=1] features=(5.50,2.60,4.40,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=92 FAILED [class=2, true class=1] features=(6.10,3.00,4.60,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=93 FAILED [class=2, true class=1] features=(5.80,2.60,4.00,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=94 FAILED [class=2, true class=1] features=(5.00,2.30,3.30,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=95 FAILED [class=2, true class=1] features=(5.60,2.70,4.20,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=96 FAILED [class=2, true class=1] features=(5.70,3.00,4.20,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=97 FAILED [class=2, true class=1] features=(5.70,2.90,4.20,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=98 FAILED [class=2, true class=1] features=(6.20,2.90,4.30,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=99 FAILED [class=0, true class=1] features=(5.10,2.50,3.00,1.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\complement_naive_bayes_classifier_iris.onnx  sample=100 FAILED [class=2, true class=1] features=(5.70,2.80,4.10,1.30]
Iris_AllClassifiers (EURUSD,H1) 25 model:IRIS_models\complement_naive_bayes_classifier_iris.onnx   accuracy: 0.6667
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=2 FAILED [class=1, true class=0] features=(4.90,3.00,1.40,0.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=9 FAILED [class=1, true class=0] features=(4.40,2.90,1.40,0.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=10 FAILED [class=1, true class=0] features=(4.90,3.10,1.50,0.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=13 FAILED [class=1, true class=0] features=(4.80,3.00,1.40,0.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=21 FAILED [class=1, true class=0] features=(5.40,3.40,1.70,0.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=26 FAILED [class=1, true class=0] features=(5.00,3.00,1.60,0.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=31 FAILED [class=1, true class=0] features=(4.80,3.10,1.60,0.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=35 FAILED [class=1, true class=0] features=(4.90,3.10,1.50,0.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=42 FAILED [class=1, true class=0] features=(4.50,2.30,1.30,0.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=46 FAILED [class=1, true class=0] features=(4.80,3.00,1.40,0.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=102 FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=103 FAILED [class=1, true class=2] features=(7.10,3.00,5.90,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=104 FAILED [class=1, true class=2] features=(6.30,2.90,5.60,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=105 FAILED [class=1, true class=2] features=(6.50,3.00,5.80,2.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=106 FAILED [class=1, true class=2] features=(7.60,3.00,6.60,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=107 FAILED [class=1, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=108 FAILED [class=1, true class=2] features=(7.30,2.90,6.30,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=109 FAILED [class=1, true class=2] features=(6.70,2.50,5.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=110 FAILED [class=1, true class=2] features=(7.20,3.60,6.10,2.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=111 FAILED [class=1, true class=2] features=(6.50,3.20,5.10,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=112 FAILED [class=1, true class=2] features=(6.40,2.70,5.30,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=113 FAILED [class=1, true class=2] features=(6.80,3.00,5.50,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=114 FAILED [class=1, true class=2] features=(5.70,2.50,5.00,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=116 FAILED [class=1, true class=2] features=(6.40,3.20,5.30,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=117 FAILED [class=1, true class=2] features=(6.50,3.00,5.50,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=118 FAILED [class=1, true class=2] features=(7.70,3.80,6.70,2.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=119 FAILED [class=1, true class=2] features=(7.70,2.60,6.90,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=120 FAILED [class=1, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=121 FAILED [class=1, true class=2] features=(6.90,3.20,5.70,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=122 FAILED [class=1, true class=2] features=(5.60,2.80,4.90,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=123 FAILED [class=1, true class=2] features=(7.70,2.80,6.70,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=124 FAILED [class=1, true class=2] features=(6.30,2.70,4.90,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=125 FAILED [class=1, true class=2] features=(6.70,3.30,5.70,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=126 FAILED [class=1, true class=2] features=(7.20,3.20,6.00,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=127 FAILED [class=1, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=128 FAILED [class=1, true class=2] features=(6.10,3.00,4.90,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=129 FAILED [class=1, true class=2] features=(6.40,2.80,5.60,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=130 FAILED [class=1, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=131 FAILED [class=1, true class=2] features=(7.40,2.80,6.10,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=132 FAILED [class=1, true class=2] features=(7.90,3.80,6.40,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=133 FAILED [class=1, true class=2] features=(6.40,2.80,5.60,2.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=134 FAILED [class=1, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=135 FAILED [class=1, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=136 FAILED [class=1, true class=2] features=(7.70,3.00,6.10,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=137 FAILED [class=1, true class=2] features=(6.30,3.40,5.60,2.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=138 FAILED [class=1, true class=2] features=(6.40,3.10,5.50,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=139 FAILED [class=1, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=140 FAILED [class=1, true class=2] features=(6.90,3.10,5.40,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=141 FAILED [class=1, true class=2] features=(6.70,3.10,5.60,2.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=142 FAILED [class=1, true class=2] features=(6.90,3.10,5.10,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=143 FAILED [class=1, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=144 FAILED [class=1, true class=2] features=(6.80,3.20,5.90,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=145 FAILED [class=1, true class=2] features=(6.70,3.30,5.70,2.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=146 FAILED [class=1, true class=2] features=(6.70,3.00,5.20,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=147 FAILED [class=1, true class=2] features=(6.30,2.50,5.00,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=148 FAILED [class=1, true class=2] features=(6.50,3.00,5.20,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=149 FAILED [class=1, true class=2] features=(6.20,3.40,5.40,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\perceptron_classifier_iris.onnx  sample=150 FAILED [class=1, true class=2] features=(5.90,3.00,5.10,1.80]
Iris_AllClassifiers (EURUSD,H1) 26 model:IRIS_models\perceptron_classifier_iris.onnx   accuracy: 0.6133
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=51 FAILED [class=0, true class=1] features=(7.00,3.20,4.70,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=52 FAILED [class=0, true class=1] features=(6.40,3.20,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=53 FAILED [class=0, true class=1] features=(6.90,3.10,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=54 FAILED [class=0, true class=1] features=(5.50,2.30,4.00,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=55 FAILED [class=0, true class=1] features=(6.50,2.80,4.60,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=56 FAILED [class=0, true class=1] features=(5.70,2.80,4.50,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=57 FAILED [class=0, true class=1] features=(6.30,3.30,4.70,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=58 FAILED [class=0, true class=1] features=(4.90,2.40,3.30,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=59 FAILED [class=0, true class=1] features=(6.60,2.90,4.60,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=60 FAILED [class=0, true class=1] features=(5.20,2.70,3.90,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=61 FAILED [class=0, true class=1] features=(5.00,2.00,3.50,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=62 FAILED [class=0, true class=1] features=(5.90,3.00,4.20,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=63 FAILED [class=0, true class=1] features=(6.00,2.20,4.00,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=64 FAILED [class=0, true class=1] features=(6.10,2.90,4.70,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=65 FAILED [class=0, true class=1] features=(5.60,2.90,3.60,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=66 FAILED [class=0, true class=1] features=(6.70,3.10,4.40,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=67 FAILED [class=0, true class=1] features=(5.60,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=68 FAILED [class=0, true class=1] features=(5.80,2.70,4.10,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=69 FAILED [class=0, true class=1] features=(6.20,2.20,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=70 FAILED [class=0, true class=1] features=(5.60,2.50,3.90,1.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=71 FAILED [class=0, true class=1] features=(5.90,3.20,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=72 FAILED [class=0, true class=1] features=(6.10,2.80,4.00,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=73 FAILED [class=0, true class=1] features=(6.30,2.50,4.90,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=74 FAILED [class=0, true class=1] features=(6.10,2.80,4.70,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=75 FAILED [class=0, true class=1] features=(6.40,2.90,4.30,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=76 FAILED [class=0, true class=1] features=(6.60,3.00,4.40,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=77 FAILED [class=0, true class=1] features=(6.80,2.80,4.80,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=78 FAILED [class=0, true class=1] features=(6.70,3.00,5.00,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=79 FAILED [class=0, true class=1] features=(6.00,2.90,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=80 FAILED [class=0, true class=1] features=(5.70,2.60,3.50,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=81 FAILED [class=0, true class=1] features=(5.50,2.40,3.80,1.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=82 FAILED [class=0, true class=1] features=(5.50,2.40,3.70,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=83 FAILED [class=0, true class=1] features=(5.80,2.70,3.90,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=84 FAILED [class=0, true class=1] features=(6.00,2.70,5.10,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=85 FAILED [class=0, true class=1] features=(5.40,3.00,4.50,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=86 FAILED [class=0, true class=1] features=(6.00,3.40,4.50,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=87 FAILED [class=0, true class=1] features=(6.70,3.10,4.70,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=88 FAILED [class=0, true class=1] features=(6.30,2.30,4.40,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=89 FAILED [class=0, true class=1] features=(5.60,3.00,4.10,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=90 FAILED [class=0, true class=1] features=(5.50,2.50,4.00,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=91 FAILED [class=0, true class=1] features=(5.50,2.60,4.40,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=92 FAILED [class=0, true class=1] features=(6.10,3.00,4.60,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=93 FAILED [class=0, true class=1] features=(5.80,2.60,4.00,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=94 FAILED [class=0, true class=1] features=(5.00,2.30,3.30,1.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=95 FAILED [class=0, true class=1] features=(5.60,2.70,4.20,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=96 FAILED [class=0, true class=1] features=(5.70,3.00,4.20,1.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=97 FAILED [class=0, true class=1] features=(5.70,2.90,4.20,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=98 FAILED [class=0, true class=1] features=(6.20,2.90,4.30,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=99 FAILED [class=0, true class=1] features=(5.10,2.50,3.00,1.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=100 FAILED [class=0, true class=1] features=(5.70,2.80,4.10,1.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=101 FAILED [class=0, true class=2] features=(6.30,3.30,6.00,2.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=102 FAILED [class=0, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=103 FAILED [class=0, true class=2] features=(7.10,3.00,5.90,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=104 FAILED [class=0, true class=2] features=(6.30,2.90,5.60,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=105 FAILED [class=0, true class=2] features=(6.50,3.00,5.80,2.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=106 FAILED [class=0, true class=2] features=(7.60,3.00,6.60,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=107 FAILED [class=0, true class=2] features=(4.90,2.50,4.50,1.70]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=108 FAILED [class=0, true class=2] features=(7.30,2.90,6.30,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=109 FAILED [class=0, true class=2] features=(6.70,2.50,5.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=110 FAILED [class=0, true class=2] features=(7.20,3.60,6.10,2.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=111 FAILED [class=0, true class=2] features=(6.50,3.20,5.10,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=112 FAILED [class=0, true class=2] features=(6.40,2.70,5.30,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=113 FAILED [class=0, true class=2] features=(6.80,3.00,5.50,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=114 FAILED [class=0, true class=2] features=(5.70,2.50,5.00,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=115 FAILED [class=0, true class=2] features=(5.80,2.80,5.10,2.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=116 FAILED [class=0, true class=2] features=(6.40,3.20,5.30,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=117 FAILED [class=0, true class=2] features=(6.50,3.00,5.50,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=118 FAILED [class=0, true class=2] features=(7.70,3.80,6.70,2.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=119 FAILED [class=0, true class=2] features=(7.70,2.60,6.90,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=120 FAILED [class=0, true class=2] features=(6.00,2.20,5.00,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=121 FAILED [class=0, true class=2] features=(6.90,3.20,5.70,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=122 FAILED [class=0, true class=2] features=(5.60,2.80,4.90,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=123 FAILED [class=0, true class=2] features=(7.70,2.80,6.70,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=124 FAILED [class=0, true class=2] features=(6.30,2.70,4.90,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=125 FAILED [class=0, true class=2] features=(6.70,3.30,5.70,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=126 FAILED [class=0, true class=2] features=(7.20,3.20,6.00,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=127 FAILED [class=0, true class=2] features=(6.20,2.80,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=128 FAILED [class=0, true class=2] features=(6.10,3.00,4.90,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=129 FAILED [class=0, true class=2] features=(6.40,2.80,5.60,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=130 FAILED [class=0, true class=2] features=(7.20,3.00,5.80,1.60]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=131 FAILED [class=0, true class=2] features=(7.40,2.80,6.10,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=132 FAILED [class=0, true class=2] features=(7.90,3.80,6.40,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=133 FAILED [class=0, true class=2] features=(6.40,2.80,5.60,2.20]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=134 FAILED [class=0, true class=2] features=(6.30,2.80,5.10,1.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=135 FAILED [class=0, true class=2] features=(6.10,2.60,5.60,1.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=136 FAILED [class=0, true class=2] features=(7.70,3.00,6.10,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=137 FAILED [class=0, true class=2] features=(6.30,3.40,5.60,2.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=138 FAILED [class=0, true class=2] features=(6.40,3.10,5.50,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=139 FAILED [class=0, true class=2] features=(6.00,3.00,4.80,1.80]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=140 FAILED [class=0, true class=2] features=(6.90,3.10,5.40,2.10]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=141 FAILED [class=0, true class=2] features=(6.70,3.10,5.60,2.40]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=142 FAILED [class=0, true class=2] features=(6.90,3.10,5.10,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=143 FAILED [class=0, true class=2] features=(5.80,2.70,5.10,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=144 FAILED [class=0, true class=2] features=(6.80,3.20,5.90,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=145 FAILED [class=0, true class=2] features=(6.70,3.30,5.70,2.50]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=146 FAILED [class=0, true class=2] features=(6.70,3.00,5.20,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=147 FAILED [class=0, true class=2] features=(6.30,2.50,5.00,1.90]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=148 FAILED [class=0, true class=2] features=(6.50,3.00,5.20,2.00]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=149 FAILED [class=0, true class=2] features=(6.20,3.40,5.40,2.30]
Iris_AllClassifiers (EURUSD,H1) model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx  sample=150 FAILED [class=0, true class=2] features=(5.90,3.00,5.10,1.80]
Iris_AllClassifiers (EURUSD,H1) 27 model:IRIS_models\bernoulli_naive_bayes_classifier_iris.onnx   accuracy: 0.3333
```

**2.29. Classification Models in Scikit-Learn That Couldn't Be Converted to ONNX**

Some classification models couldn't be converted to the ONNX format due to errors in the convert\_sklearn process.

**2.29.1. DummyClassifier**

DummyClassifier is a classifier in the Scikit-learn library used as a simple baseline model for classification tasks. It is designed for testing and evaluating the performance of more complex classification models.

Working Principle:

The DummyClassifier works very simply; it makes random or naive predictions without considering the input data. It offers different strategies (selected through the "strategy" parameter):

1. "most\_frequent" (Most Frequent Class): This strategy always predicts the class that appears most frequently in the training dataset. It can be useful in situations where classes are imbalanced, and you need to predict the dominant class.
2. "stratified" (Stratified Choice): This strategy attempts to make predictions that match the class distribution in the training dataset. It uses random guessing but takes class proportions into account.
3. "uniform" (Uniform Distribution): This strategy makes random predictions with equal probability for each class. It is useful when the classes are balanced, and you want to test how your model performs on average.

Capabilities:

- Simplicity: DummyClassifier is useful for testing how quickly you can train a baseline model and what results it produces. It can be helpful for a quick performance assessment of other classifiers.
- Pipeline Usage: You can use DummyClassifier as a baseline model within a pipeline, combined with other transformations and models for comparison and testing.

Limitations:

- Doesn't Utilize Data: DummyClassifier makes random or naive predictions without considering actual data. It cannot learn from data or discover patterns.
- Unsuitable for Complex Tasks: This classifier is not designed for solving complex classification tasks and generally does not yield good results for tasks with large datasets and intricate patterns.
- Lack of Informativeness: Results obtained using DummyClassifier may lack informativeness and not provide useful information about model performance. They are more useful for code testing and evaluation.

DummyClassifier is a useful tool for initial testing and evaluating classification models, but its usage is limited in complex tasks, and it cannot replace more advanced classification algorithms.

**2.29.1.1. Code to Create a DummyClassifier Model**

\# Iris\_DummyClassifier.py

\# The code demonstrates the process of training DummyClassifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.

\# It also evaluates the accuracy of both the original model and the ONNX model.

\# Copyright 2023, MetaQuotes Ltd.

\# https://www.mql5.com

\# import necessary libraries

**from** sklearn **import** datasets

**from** sklearn.dummy **import** DummyClassifier

**from** sklearn.metrics **import** accuracy\_score, classification\_report

**from** skl2onnx **import** convert\_sklearn

**from** skl2onnx.common.data\_types **import** FloatTensorType

**import** onnxruntime **as** ort

**import** numpy **as** np

from sys import argv

\# define the path for saving the model

data\_path = argv\[0\]

last\_index = data\_path.rfind("\\\") \+ 1

data\_path = data\_path\[0:last\_index\]

\# load the Iris dataset

iris = datasets.load\_iris()

X = iris.data

y = iris.target

\# create a DummyClassifier model with the strategy "most\_frequent"

dummy\_classifier = DummyClassifier(strategy="most\_frequent")

\# train the model on the entire dataset

dummy\_classifier.fit(X, y)

\# predict classes for the entire dataset

y\_pred = dummy\_classifier.predict(X)

\# evaluate the model's accuracy

accuracy = accuracy\_score(y, y\_pred)

**print**("Accuracy of DummyClassifier model:", accuracy)

\# display the classification report

**print**("\\nClassification Report:\\n", classification\_report(y, y\_pred))

\# define the input data type

initial\_type = \[('float\_input', FloatTensorType(\[None, X.shape\[1\]\]))\]

\# export the model to ONNX format with float data type

onnx\_model = convert\_sklearn(dummy\_classifier, initial\_types=initial\_type, target\_opset=12)

\# save the model to a file

onnx\_filename = data\_path + "dummy\_classifier\_iris.onnx"

**with** open(onnx\_filename, "wb") **as** f:

    f.write(onnx\_model.SerializeToString())

\# print model path

**print**(f"Model saved to {onnx\_filename}")

\# load the ONNX model and make predictions

onnx\_session = ort.InferenceSession(onnx\_filename)

input\_name = onnx\_session.get\_inputs()\[0\].name

output\_name = onnx\_session.get\_outputs()\[0\].name

\# display information about input tensors in ONNX

**print**("\\nInformation about input tensors in ONNX:")

**for** i, input\_tensor in enumerate(onnx\_session.get\_inputs()):

**print**(f"{i + 1}. Name: {input\_tensor.name}, Data Type: {input\_tensor.type}, Shape: {input\_tensor.shape}")

\# display information about output tensors in ONNX

**print**("\\nInformation about output tensors in ONNX:")

**for** i, output\_tensor in enumerate(onnx\_session.get\_outputs()):

**print**(f"{i + 1}. Name: {output\_tensor.name}, Data Type: {output\_tensor.type}, Shape: {output\_tensor.shape}")

\# convert data to floating-point format (float32)

X\_float32 = X.astype(np.float32)

\# predict classes for the entire dataset using ONNX

y\_pred\_onnx = onnx\_session.run(\[output\_name\], {input\_name: X\_float32})\[0\]

\# evaluate the accuracy of the ONNX model

accuracy\_onnx = accuracy\_score(y, y\_pred\_onnx)

**print**("\\nAccuracy of DummyClassifier model in ONNX format:", accuracy\_onnx)

Output:

Python    Accuracy of DummyClassifier model: 0.3333333333333333

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       0.33      1.00      0.50        50

Python               1       0.00      0.00      0.00        50

Python               2       0.00      0.00      0.00        50

Python

Python        accuracy                           0.33       150

Python       macro avg       0.11      0.33      0.17       150

Python    weighted avg       0.11      0.33      0.17       150

Python

The model was built and ran successfully in Scikit-learn, but errors occurred during the conversion to ONNX.

In the "Errors" tab, messages regarding conversion errors to the ONNX format are displayed:

     onnx\_model = convert\_sklearn(dummy\_classifier, initial\_types=initial\_type, target\_opset=12)    Iris\_DummyClassifier.py    44    1

    onnx\_model = convert\_topology(    convert.py    208    1

    topology.convert\_operators(container=container, verbose=verbose)    \_topology.py    1532    1

    self.call\_shape\_calculator(operator)    \_topology.py    1348    1

    operator.infer\_types()    \_topology.py    1163    1

    raise MissingShapeCalculator(    \_topology.py    629    1

skl2onnx.common.exceptions.MissingShapeCalculator: Unable to find a shape calculator for type '<class 'sklearn.dummy.DummyClassifier'>'.    \_topology.py    629    1

It usually means the pipeline being converted contains a    \_topology.py    629    1

transformer or a predictor with no corresponding converter    \_topology.py    629    1

implemented in sklearn-onnx. If the converted is implemented    \_topology.py    629    1

in another library, you need to register    \_topology.py    629    1

the converted so that it can be used by sklearn-onnx (function    \_topology.py    629    1

update\_registered\_converter). If the model is not yet covered    \_topology.py    629    1

by sklearn-onnx, you may raise an issue to    \_topology.py    629    1

https://github.com/onnx/sklearn-onnx/issues    \_topology.py    629    1

to get the converter implemented or even contribute to the    \_topology.py    629    1

project. If the model is a custom model, a new converter must    \_topology.py    629    1

be implemented. Examples can be found in the gallery.    \_topology.py    629    1

Iris\_DummyClassifier.py finished in 2071 ms        19    1

Therefore, the DummyClassifier model could not be converted to ONNX.

**2.29.2. GaussianProcessClassifier**

GaussianProcessClassifier is a classifier that uses Gaussian processes for classification tasks. It belongs to the family of models that use Gaussian processes and can be useful in tasks where probabilistic class estimates are needed.

Working Principle:

1. GaussianProcessClassifier uses a Gaussian process to model the mapping from feature space to the space of class probability estimates.
2. It builds a probabilistic model for each class by evaluating the probability of a point belonging to each class.
3. During classification, it selects the class with the highest probability for a given point.


Capabilities:

- Probabilistic Classification: GaussianProcessClassifier provides probabilistic class estimates, which can be useful for assessing model uncertainty.
- Adaptiveness: This classifier can adapt to data and update its predictions based on new observations.
- Calibration: The model can be calibrated using the "calibrate" method to improve probability estimates.


Limitations:

- Computational Complexity: GaussianProcessClassifier can be computationally expensive for large datasets and/or high-dimensional feature spaces.
- Not Suitable for Large Samples: Due to its computational complexity, this classifier may not be efficient for training on large datasets.
- Interpretation Complexity: Gaussian processes can be challenging to interpret and understand, especially for users without experience in Bayesian statistics.


GaussianProcessClassifier is valuable in tasks where probabilistic class estimates are important and where one can handle computational costs. Otherwise, for classification tasks on large datasets or with simple data structures, other classification algorithms may be more suitable.

**2.29.2.1.  Code for Creating a GaussianProcessClassifier Model**

\# Iris\_GaussianProcessClassifier.py

\# The code demonstrates the process of training Iris\_GaussianProcess Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.

\# It also evaluates the accuracy of both the original model and the ONNX model.

\# Copyright 2023, MetaQuotes Ltd.

\# https://www.mql5.com

\# import necessary libraries

**from** sklearn **import** datasets

**from** sklearn.gaussian\_process **import** GaussianProcessClassifier

**from** sklearn.gaussian\_process.kernels **import** RBF

**from** sklearn.metrics **import** accuracy\_score, classification\_report

**from** skl2onnx **import** convert\_sklearn

**from** skl2onnx.common.data\_types **import** FloatTensorType

**import** onnxruntime **as** ort

**import** numpy **as** np

from sys **import** argv

\# define the path for saving the model

data\_path = argv\[0\]

last\_index = data\_path.rfind("\\\") \+ 1

data\_path = data\_path\[0:last\_index\]

\# load the Iris dataset

iris = datasets.load\_iris()

X = iris.data

y = iris.target

\# create a GaussianProcessClassifier model with an RBF kernel

kernel = 1.0 \* RBF(1.0)

gpc\_model = GaussianProcessClassifier(kernel=kernel)

\# train the model on the entire dataset

gpc\_model.fit(X, y)

\# predict classes for the entire dataset

y\_pred = gpc\_model.predict(X)

\# evaluate the model's accuracy

accuracy = accuracy\_score(y, y\_pred)

**print**("Accuracy of GaussianProcessClassifier model:", accuracy)

\# display the classification report

**print**("\\nClassification Report:\\n", classification\_report(y, y\_pred))

\# define the input data type

initial\_type = \[('float\_input', FloatTensorType(\[None, X.shape\[1\]\]))\]

\# export the model to ONNX format with float data type

onnx\_model = convert\_sklearn(gpc\_model, initial\_types=initial\_type, target\_opset=12)

\# save the model to a file

onnx\_filename = data\_path + "gpc\_iris.onnx"

**with** open(onnx\_filename, "wb") **as** f:

    f.write(onnx\_model.SerializeToString())

\# print the path to the model

**print**(f"Model saved to {onnx\_filename}")

\# load the ONNX model and make predictions

onnx\_session = ort.InferenceSession(onnx\_filename)

input\_name = onnx\_session.get\_inputs()\[0\].name

output\_name = onnx\_session.get\_outputs()\[0\].name

\# display information about input tensors in ONNX

**print**("\\nInformation about input tensors in ONNX:")

**for** i, input\_tensor in enumerate(onnx\_session.get\_inputs()):

**print**(f"{i + 1}. Name: {input\_tensor.name}, Data Type: {input\_tensor.type}, Shape: {input\_tensor.shape}")

\# display information about output tensors in ONNX

**print**("\\nInformation about output tensors in ONNX:")

**for** i, output\_tensor in enumerate(onnx\_session.get\_outputs()):

**print**(f"{i + 1}. Name: {output\_tensor.name}, Data Type: {output\_tensor.type}, Shape: {output\_tensor.shape}")

\# convert data to floating-point format (float32)

X\_float32 = X.astype(np.float32)

\# predict classes for the entire dataset using ONNX

y\_pred\_onnx = onnx\_session.run(\[output\_name\], {input\_name: X\_float32})\[0\]

\# evaluate the accuracy of the ONNX model

accuracy\_onnx = accuracy\_score(y, y\_pred\_onnx)

**print**("\\nAccuracy of GaussianProcessClassifier model in ONNX format:", accuracy\_onnx)

Output:

Python    Accuracy of GaussianProcessClassifier model: 0.9866666666666667

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.98      0.98      0.98        50

Python               2       0.98      0.98      0.98        50

Python

Python        accuracy                           0.99       150

Python       macro avg       0.99      0.99      0.99       150

Python    weighted avg       0.99      0.99      0.99       150

Python

In the "Errors" tab, messages regarding conversion errors to the ONNX format are displayed:

     onnx\_model = convert\_sklearn(gpc\_model, initial\_types=initial\_type, target\_opset=12)    Iris\_GaussianProcessClassifier.py    46    1

    onnx\_model = convert\_topology(    convert.py    208    1

    topology.convert\_operators(container=container, verbose=verbose)    \_topology.py    1532    1

    self.call\_converter(operator, container, verbose=verbose)    \_topology.py    1349    1

    conv(self.scopes\[0\], operator, container)    \_topology.py    1132    1

    return self.\_fct(\*args)    \_registration.py    27    1

    raise NotImplementedError("Only binary classification is iplemented.")    gaussian\_process.py    247    1

NotImplementedError: Only binary classification is iplemented.    gaussian\_process.py    247    1

Iris\_GaussianProcessClassifier.py finished in 4004 ms        9    1

Thus, the GaussianProcessClassifier model could not be converted to ONNX.

**2.29.3. LabelPropagation Classifier**

LabelPropagation is a semi-supervised learning method used for classification tasks. The primary idea behind this method is to propagate labels (classes) from labeled instances to unlabeled instances in a graph-based data structure.

The LabelPropagation Process:

1. It starts with the construction of a graph where nodes represent data instances, and edges between nodes reflect similarity or proximity between instances.
2. Initial label assignment: Labeled instances are given their labels, and unlabeled instances start with some undefined label.
3. Label propagation on the graph: Labels from labeled instances are propagated to unlabeled instances based on the similarity between instances. This similarity can be determined in various ways, e.g., using nearest neighbors in the graph.
4. Iterative process: Labels can change over several iterations, where each iteration updates labels on unlabeled instances based on the current labels and instance similarity.
5. Stabilization: The process continues until labels stabilize or a certain stopping criterion is met.

Advantages of LabelPropagation:

- Utilizes information from unlabeled data: LabelPropagation allows using information from among unlabeled instances to improve classification quality. This is especially useful when there is a scarcity of labeled data.
- Robustness to noise: The method handles data with noise effectively because it takes instance similarity into account and doesn't rely solely on labels.

Limitations of LabelPropagation:

- Dependence on graph choice: The quality of LabelPropagation classification can heavily depend on the choice of the graph and the method of determining instance similarity. Incorrect parameter choices can lead to poor results.
- Computational complexity: Depending on the data's size and complexity, as well as method parameters, LabelPropagation can require significant computational resources.
- Overfitting potential: If the graph contains too many noisy edges or incorrect labels, the method can overfit.
- Convergence not guaranteed: In rare cases, LabelPropagation may not converge to stable labels, requiring limiting the number of iterations or adjusting other settings.

LabelPropagation is a powerful method, but it requires careful parameter tuning and analysis of the data's graph structure to achieve good results.

**2.29.3.1. Code for Creating a LabelPropagationClassifier Model**

\# Iris\_LabelPropagationClassifier.py

\# The code demonstrates the process of training LabelPropagation Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.

\# It also evaluates the accuracy of both the original model and the ONNX model.

\# Copyright 2023, MetaQuotes Ltd.

\# https://www.mql5.com

# import necessary libraries

**from** sklearn **import** datasets

**from** sklearn.semi\_supervised **import** LabelPropagation

**from** sklearn.metrics **import** accuracy\_score, classification\_report

**from** skl2onnx **import** convert\_sklearn

**from** skl2onnx.common.data\_types **import** FloatTensorType

**import** onnxruntime **as** ort

**import** numpy **as** np

**from** sys import argv

\# define the path for saving the model

data\_path = argv\[0\]

last\_index = data\_path.rfind("\\\") \+ 1

data\_path = data\_path\[0:last\_index\]

\# load the Iris dataset

iris = datasets.load\_iris()

X = iris.data

y = iris.target

\# create a LabelPropagation model

lp\_model = LabelPropagation()

\# train the model on the entire dataset

lp\_model.fit(X, y)

\# predict classes for the entire dataset

y\_pred = lp\_model.predict(X)

\# evaluate the model's accuracy

accuracy = accuracy\_score(y, y\_pred)

**print**("Accuracy of LabelPropagation model:", accuracy)

\# display the classification report

**print**("\\nClassification Report:\\n", classification\_report(y, y\_pred))

\# define the input data type

initial\_type = \[('float\_input', FloatTensorType(\[None, X.shape\[1\]\]))\]

\# export the model to ONNX format with float data type

onnx\_model = convert\_sklearn(lp\_model, initial\_types=initial\_type, target\_opset=12)

\# save the model to a file

onnx\_filename = data\_path + "lp\_iris.onnx"

**with** open(onnx\_filename, "wb") **as** f:

    f.write(onnx\_model.SerializeToString())

\# print the path to the model

**print**(f"Model saved to {onnx\_filename}")

\# load the ONNX model and make predictions

onnx\_session = ort.InferenceSession(onnx\_filename)

input\_name = onnx\_session.get\_inputs()\[0\].name

output\_name = onnx\_session.get\_outputs()\[0\].name

\# display information about input tensors in ONNX

**print**("\\nInformation about input tensors in ONNX:")

**for** i, input\_tensor in enumerate(onnx\_session.get\_inputs()):

**print**(f"{i + 1}. Name: {input\_tensor.name}, Data Type: {input\_tensor.type}, Shape: {input\_tensor.shape}")

\# display information about output tensors in ONNX

**print**("\\nInformation about output tensors in ONNX:")

**for** i, output\_tensor in enumerate(onnx\_session.get\_outputs()):

**print**(f"{i + 1}. Name: {output\_tensor.name}, Data Type: {output\_tensor.type}, Shape: {output\_tensor.shape}")

\# convert data to floating-point format (float32)

X\_float32 = X.astype(np.float32)

\# predict classes for the entire dataset using ONNX

y\_pred\_onnx = onnx\_session.run(\[output\_name\], {input\_name: X\_float32})\[0\]

\# evaluate the accuracy of the ONNX model

accuracy\_onnx = accuracy\_score(y, y\_pred\_onnx)

**print**("\\nAccuracy of LabelPropagation model in ONNX format:", accuracy\_onnx)

Output:

Python    Accuracy of LabelPropagation model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

The model was constructed, but errors occurred during the conversion to ONNX format.

In the "Errors" tab, messages regarding conversion errors to the ONNX format are displayed:

     onnx\_model = convert\_sklearn(lp\_model, initial\_types=initial\_type, target\_opset=12)    Iris\_LabelPropagation.py    44    1

    onnx\_model = convert\_topology(    convert.py    208    1

    topology.convert\_operators(container=container, verbose=verbose)    \_topology.py    1532    1

    self.call\_shape\_calculator(operator)    \_topology.py    1348    1

    operator.infer\_types()    \_topology.py    1163    1

    raise MissingShapeCalculator(    \_topology.py    629    1

skl2onnx.common.exceptions.MissingShapeCalculator: Unable to find a shape calculator for type '<class 'sklearn.semi\_supervised.\_label\_propagation.LabelPropagation'>'.    \_topology.py    629    1

It usually means the pipeline being converted contains a    \_topology.py    629    1

transformer or a predictor with no corresponding converter    \_topology.py    629    1

implemented in sklearn-onnx. If the converted is implemented    \_topology.py    629    1

in another library, you need to register    \_topology.py    629    1

the converted so that it can be used by sklearn-onnx (function    \_topology.py    629    1

update\_registered\_converter). If the model is not yet covered    \_topology.py    629    1

by sklearn-onnx, you may raise an issue to    \_topology.py    629    1

https://github.com/onnx/sklearn-onnx/issues    \_topology.py    629    1

to get the converter implemented or even contribute to the    \_topology.py    629    1

project. If the model is a custom model, a new converter must    \_topology.py    629    1

be implemented. Examples can be found in the gallery.    \_topology.py    629    1

Iris\_LabelPropagation.py finished in 2064 ms        19    1

Thus, the LabelSpreading Classifier model could not be converted to ONNX.

**2.29.4. LabelSpreading Classifier**

LabelSpreading is a semi-supervised learning method used for classification tasks. It is based on the idea of propagating labels (classes) from labeled instances to unlabeled instances in a graph-based data structure, similar to LabelPropagation. However, LabelSpreading includes additional stabilization and regularization of the label propagation process.

The LabelSpreading Process:

1. It starts with the construction of a graph where nodes represent data instances, and edges between nodes reflect similarity or proximity between instances.
2. Initial label assignment: Labeled instances are given their labels, and unlabeled instances start with some undefined label.
3. Label propagation on the graph: Labels from labeled instances are propagated to unlabeled instances based on the similarity between instances.
4. Regularization and stabilization: LabelSpreading includes regularization that helps stabilize the label propagation process and reduce overfitting. This is achieved by considering not only the similarity between instances but also the differences between labels of neighboring instances.
5. Iterative process: Labels can change over several iterations, where each iteration updates labels on unlabeled instances based on the current labels and regularization.
6. Stabilization: The process continues until labels stabilize or a certain stopping criterion is met.

Advantages of LabelSpreading:

- Utilizes information from unlabeled data: LabelSpreading allows using information from among unlabeled instances to improve classification quality.
- Regularization: The presence of regularization in LabelSpreading helps reduce overfitting and makes the label propagation process more stable.

Limitations of LabelSpreading:

- Dependence on graph choice: Similar to LabelPropagation, the quality of LabelSpreading classification can heavily depend on the choice of the graph and method parameters.
- Computational complexity: Depending on the data's size and complexity, as well as method parameters, LabelSpreading can require significant computational resources.
- Not always convergent: In rare cases, LabelSpreading may not converge to stable labels, requiring limiting the number of iterations or adjusting other settings.

LabelSpreading is a method that also requires careful tuning and can be a powerful tool for using unlabeled data in classification tasks.

**2.29.4.1. Code for Creating a LabelSpreadingClassifier Model**

\# Iris\_LabelSpreadingClassifier.py

\# The code demonstrates the process of training LabelSpreading Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.

\# It also evaluates the accuracy of both the original model and the ONNX model.

\# Copyright 2023, MetaQuotes Ltd.

\# import necessary libraries

**from** sklearn **import** datasets

**from** sklearn.semi\_supervised import LabelSpreading

**from** sklearn.metrics **import** accuracy\_score, classification\_report

**from** skl2onnx **import** convert\_sklearn

**from** skl2onnx.common.data\_types **import** FloatTensorType

**import** onnxruntime **as** ort

**import** numpy **as** np

**import** sys

\# get the script path

script\_path = sys.argv\[0\]

last\_index = script\_path.rfind("\\\") \+ 1

data\_path = script\_path\[0:last\_index\]

\# load the Iris dataset

iris = datasets.load\_iris()

X = iris.data

y = iris.target

\# create a LabelSpreading model

ls\_model = LabelSpreading()

\# train the model on the entire dataset

ls\_model.fit(X, y)

\# predict classes for the entire dataset

y\_pred = ls\_model.predict(X)

\# evaluate the model's accuracy

accuracy = accuracy\_score(y, y\_pred)

**print**("Accuracy of LabelSpreading model:", accuracy)

\# display the classification report

**print**("\\nClassification Report:\\n", classification\_report(y, y\_pred))

\# define the input data type

initial\_type = \[('float\_input', FloatTensorType(\[None, X.shape\[1\]\]))\]

\# export the model to ONNX format with float data type

onnx\_model = convert\_sklearn(ls\_model, initial\_types=initial\_type, target\_opset=12)

\# save the model to a file

onnx\_filename = data\_path + "ls\_iris.onnx"

**with** open(onnx\_filename, "wb") as f:

    f.write(onnx\_model.SerializeToString())

\# print the path to the model

**print**(f"Model saved to {onnx\_filename}")

\# load the ONNX model and make predictions

onnx\_session = ort.InferenceSession(onnx\_filename)

input\_name = onnx\_session.get\_inputs()\[0\].name

output\_name = onnx\_session.get\_outputs()\[0\].name

\# display information about input tensors in ONNX

**print**("\\nInformation about input tensors in ONNX:")

**for** i, input\_tensor in enumerate(onnx\_session.get\_inputs()):

**print**(f"{i + 1}. Name: {input\_tensor.name}, Data Type: {input\_tensor.type}, Shape: {input\_tensor.shape}")

\# display information about output tensors in ONNX

**print**("\\nInformation about output tensors in ONNX:")

**for** i, output\_tensor in enumerate(onnx\_session.get\_outputs()):

**print**(f"{i + 1}. Name: {output\_tensor.name}, Data Type: {output\_tensor.type}, Shape: {output\_tensor.shape}")

\# convert data to floating-point format (float32)

X\_float32 = X.astype(np.float32)

\# predict classes for the entire dataset using ONNX

y\_pred\_onnx = onnx\_session.run(\[output\_name\], {input\_name: X\_float32})\[0\]

\# evaluate the accuracy of the ONNX model

accuracy\_onnx = accuracy\_score(y, y\_pred\_onnx)

**print**("\\nAccuracy of LabelSpreading model in ONNX format:", accuracy\_onnx)

Output:

Python    Accuracy of LabelSpreading model: 1.0

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       1.00      1.00      1.00        50

Python               2       1.00      1.00      1.00        50

Python

Python        accuracy                           1.00       150

Python       macro avg       1.00      1.00      1.00       150

Python    weighted avg       1.00      1.00      1.00       150

Python

In the "Errors" tab, messages regarding conversion errors to the ONNX format are displayed:

     onnx\_model = convert\_sklearn(ls\_model, initial\_types=initial\_type, target\_opset=12)    Iris\_LabelSpreading.py    45    1

    onnx\_model = convert\_topology(    convert.py    208    1

    topology.convert\_operators(container=container, verbose=verbose)    \_topology.py    1532    1

    self.call\_shape\_calculator(operator)    \_topology.py    1348    1

    operator.infer\_types()    \_topology.py    1163    1

    raise MissingShapeCalculator(    \_topology.py    629    1

skl2onnx.common.exceptions.MissingShapeCalculator: Unable to find a shape calculator for type '<class 'sklearn.semi\_supervised.\_label\_propagation.LabelSpreading'>'.    \_topology.py    629    1

It usually means the pipeline being converted contains a    \_topology.py    629    1

transformer or a predictor with no corresponding converter    \_topology.py    629    1

implemented in sklearn-onnx. If the converted is implemented    \_topology.py    629    1

in another library, you need to register    \_topology.py    629    1

the converted so that it can be used by sklearn-onnx (function    \_topology.py    629    1

update\_registered\_converter). If the model is not yet covered    \_topology.py    629    1

by sklearn-onnx, you may raise an issue to    \_topology.py    629    1

https://github.com/onnx/sklearn-onnx/issues    \_topology.py    629    1

to get the converter implemented or even contribute to the    \_topology.py    629    1

project. If the model is a custom model, a new converter must    \_topology.py    629    1

be implemented. Examples can be found in the gallery.    \_topology.py    629    1

Iris\_LabelSpreading.py finished in 2032 ms        19    1

The LabelPropagation Classifier model could not be converted to ONNX.

**2.29.5. NearestCentroid Classifier**

NearestCentroid is a classification method based on the idea of determining the centroid for each class and classifying objects based on the nearest centroid. This method is suitable for multi-class problems and works well on datasets with linearly separable classes.

The NearestCentroid Process:

1. For each class, a centroid is calculated, which represents the average value of features for all objects belonging to that class. This can be done by computing the mean value of each feature for the objects in that class.
2. When classifying a new object, its nearest centroid is calculated among the centroids of all classes.
3. The new object is assigned to the class whose centroid is closest to it in the metric space.

Advantages of NearestCentroid:

- Simplicity and speed: NearestCentroid is a computationally simple method and operates quickly on large datasets.
- Suitable for linearly separable classes: The method performs well on tasks where classes are linearly separable or close to being linearly separable.
- Effective for multi-class problems: NearestCentroid is suitable for multi-class problems, and it can be used as a base classifier in ensembles.

Limitations of NearestCentroid:

- Sensitivity to outliers: The NearestCentroid method is sensitive to outliers in the data, as the centroid can be significantly distorted by the presence of outliers.
- Spatial bias: If classes in the data have different variances and shapes, the NearestCentroid method may work less efficiently.
- Assumes equal means: The method assumes that classes have roughly equal means of features, which may not always hold in real-world data.
- Not suitable for nonlinear tasks: NearestCentroid is not suitable for tasks with nonlinear boundaries between classes.

NearestCentroid is a simple and interpretable classification method that can be useful in specific scenarios, especially when classes are linearly separable, and there are no outliers in the data.

**2.29.5.1. Code for Creating a NearestCentroid Model**

\# Iris\_NearestCentroidClassifier.py

\# The code demonstrates the process of training NearestCentroid Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.

\# It also evaluates the accuracy of both the original model and the ONNX model.

\# Copyright 2023, MetaQuotes Ltd.

\# https://www.mql5.com

# import necessary libraries

**from** sklearn **import** datasets

**from** sklearn.neighbors **import** NearestCentroid

**from** sklearn.metrics **import** accuracy\_score, classification\_report

**from** skl2onnx **import** convert\_sklearn

**from** skl2onnx.common.data\_types **import** FloatTensorType

**import** onnxruntime **as** ort

**import** numpy **as** np

**import** sys

\# get the script path

script\_path = sys.argv\[0\]

last\_index = script\_path.rfind("\\\") \+ 1

data\_path = script\_path\[0:last\_index\]

\# load the Iris dataset

iris = datasets.load\_iris()

X = iris.data

y = iris.target

\# create a NearestCentroid model

nc\_model = NearestCentroid()

\# train the model on the entire dataset

nc\_model.fit(X, y)

\# predict classes for the entire dataset

y\_pred = nc\_model.predict(X)

\# evaluate the model's accuracy

accuracy = accuracy\_score(y, y\_pred)

**print**("Accuracy of NearestCentroid model:", accuracy)

\# display the classification report

**print**("\\nClassification Report:\\n", classification\_report(y, y\_pred))

\# define the input data type

initial\_type = \[('float\_input', FloatTensorType(\[None, X.shape\[1\]\]))\]

\# export the model to ONNX format with float data type

onnx\_model = convert\_sklearn(nc\_model, initial\_types=initial\_type, target\_opset=12)

\# save the model to a file

onnx\_filename = data\_path + "nc\_iris.onnx"

**with** open(onnx\_filename, "wb") **as** f:

    f.write(onnx\_model.SerializeToString())

\# print the path to the model

**print**(f"Model saved to {onnx\_filename}")

\# load the ONNX model and make predictions

onnx\_session = ort.InferenceSession(onnx\_filename)

input\_name = onnx\_session.get\_inputs()\[0\].name

output\_name = onnx\_session.get\_outputs()\[0\].name

\# display information about input tensors in ONNX

**print**("\\nInformation about input tensors in ONNX:")

**for** i, input\_tensor in enumerate(onnx\_session.get\_inputs()):

**print**(f"{i + 1}. Name: {input\_tensor.name}, Data Type: {input\_tensor.type}, Shape: {input\_tensor.shape}")

\# display information about output tensors in ONNX

**print**("\\nInformation about output tensors in ONNX:")

**for** i, output\_tensor in enumerate(onnx\_session.get\_outputs()):

**print**(f"{i + 1}. Name: {output\_tensor.name}, Data Type: {output\_tensor.type}, Shape: {output\_tensor.shape}")

\# convert data to floating-point format (float32)

X\_float32 = X.astype(np.float32)

\# predict classes for the entire dataset using ONNX

y\_pred\_onnx = onnx\_session.run(\[output\_name\], {input\_name: X\_float32})\[0\]

\# evaluate the accuracy of the ONNX model

accuracy\_onnx = accuracy\_score(y, y\_pred\_onnx)

print("\\nAccuracy of NearestCentroid model in ONNX format:", accuracy\_onnx)

Output:

Python    Accuracy of NearestCentroid model: 0.9266666666666666

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.87      0.92      0.89        50

Python               2       0.91      0.86      0.89        50

Python

Python        accuracy                           0.93       150

Python       macro avg       0.93      0.93      0.93       150

Python    weighted avg       0.93      0.93      0.93       150

Python

In the "Errors" tab, messages regarding conversion errors to the ONNX format are displayed:

     onnx\_model = convert\_sklearn(nc\_model, initial\_types=initial\_type, target\_opset=12)    Iris\_NearestCentroid.py    45    1

    onnx\_model = convert\_topology(    convert.py    208    1

    topology.convert\_operators(container=container, verbose=verbose)    \_topology.py    1532    1

    self.call\_shape\_calculator(operator)    \_topology.py    1348    1

    operator.infer\_types()    \_topology.py    1163    1

    raise MissingShapeCalculator(    \_topology.py    629    1

skl2onnx.common.exceptions.MissingShapeCalculator: Unable to find a shape calculator for type '<class 'sklearn.neighbors.\_nearest\_centroid.NearestCentroid'>'.    \_topology.py    629    1

It usually means the pipeline being converted contains a    \_topology.py    629    1

transformer or a predictor with no corresponding converter    \_topology.py    629    1

implemented in sklearn-onnx. If the converted is implemented    \_topology.py    629    1

in another library, you need to register    \_topology.py    629    1

the converted so that it can be used by sklearn-onnx (function    \_topology.py    629    1

update\_registered\_converter). If the model is not yet covered    \_topology.py    629    1

by sklearn-onnx, you may raise an issue to    \_topology.py    629    1

https://github.com/onnx/sklearn-onnx/issues    \_topology.py    629    1

to get the converter implemented or even contribute to the    \_topology.py    629    1

project. If the model is a custom model, a new converter must    \_topology.py    629    1

be implemented. Examples can be found in the gallery.    \_topology.py    629    1

Iris\_NearestCentroid.py finished in 2131 ms        19    1

The NearestCentroid Classifier model could not be converted to ONNX.

**2.29.6. Quadratic Discriminant Analysis Classifier**

Quadratic Discriminant Analysis (QDA) is a classification method that uses a probabilistic model to separate data into classes. It is a generalization of Linear Discriminant Analysis (LDA) and allows for the consideration of feature covariances within each class. The main idea of QDA is to model the feature distribution for each class and then use this distribution for classifying new objects.

The QDA Process:

1. Parameters of the distribution are calculated for each class, such as the mean and the covariance matrix of features. These parameters are estimated based on the training data for each class.
2. Using the obtained parameters, probability densities for each class can be computed using a multivariate normal distribution (or a quadratic distribution function).
3. When classifying a new object, probability density values are calculated for each class, and the object is assigned to the class with the highest probability.

Advantages of Quadratic Discriminant Analysis (QDA):

- Considers feature covariances: QDA is more flexible than LDA because it allows different covariance matrices for different classes, making it more adaptable to different data structures.
- Suitable for nonlinear boundaries: QDA is capable of modeling complex and nonlinear boundaries between classes.
- Robust to imbalanced data: QDA can perform well on tasks with imbalanced classes.

Limitations of Quadratic Discriminant Analysis (QDA):

- Computational complexity: QDA requires the estimation of parameters for each class, including covariance matrices, which can be computationally expensive on large datasets.
- Limited data: QDA may work less effectively when data is limited, and parameter estimation becomes less precise.
- Assumption of normal distribution: QDA assumes that data follows a normal distribution, which may not hold for some types of data.
- Risk of overfitting: With insufficient training data or strong feature covariance, QDA may face overfitting issues.

Quadratic Discriminant Analysis (QDA) is a powerful classification method suitable for various data types and capable of considering feature covariances within classes. However, it also has limitations that should be considered when using it.

**2.29.6.1. Code for Creating a Quadratic Discriminant Analysis Model**

\# Iris\_QuadraticDiscriminantAnalysisClassifier.py

\# The code demonstrates the process of training Quadratic Discriminant Analysis Classifier model on the Iris dataset, exporting it to ONNX format, and making predictions using the ONNX model.

\# It also evaluates the accuracy of both the original model and the ONNX model.

\# Copyright 2023, MetaQuotes Ltd.

\# https://www.mql5.com

\# import necessary libraries

**from** sklearn **import** datasets

**from** sklearn.discriminant\_analysis **import** QuadraticDiscriminantAnalysis

**from** sklearn.metrics **import** accuracy\_score, classification\_report

**from** skl2onnx **import** convert\_sklearn

**from** skl2onnx.common.data\_types **import** FloatTensorType

**import** onnxruntime **as** ort

**import** numpy **as** np

**import** sys

\# get the script path

script\_path = sys.argv\[0\]

last\_index = script\_path.rfind("\\\") \+ 1

data\_path = script\_path\[0:last\_index\]

\# load the Iris dataset

iris = datasets.load\_iris()

X = iris.data

y = iris.target

\# create a QuadraticDiscriminantAnalysis model

qda\_model = QuadraticDiscriminantAnalysis()

\# train the model on the entire dataset

qda\_model.fit(X, y)

\# predict classes for the entire dataset

y\_pred = qda\_model.predict(X)

\# evaluate the model's accuracy

accuracy = accuracy\_score(y, y\_pred)

**print**("Accuracy of Quadratic Discriminant Analysis model:", accuracy)

\# display the classification report

**print**("\\nClassification Report:\\n", classification\_report(y, y\_pred))

\# define the input data type

initial\_type = \[('float\_input', FloatTensorType(\[None, X.shape\[1\]\]))\]

\# export the model to ONNX format with float data type

onnx\_model = convert\_sklearn(qda\_model, initial\_types=initial\_type, target\_opset=12)

\# save the model to a file

onnx\_filename = data\_path + "qda\_iris.onnx"

**with** open(onnx\_filename, "wb") **as** f:

    f.write(onnx\_model.SerializeToString())

\# print the path to the model

**print**(f"Model saved to {onnx\_filename}")

\# load the ONNX model and make predictions

onnx\_session = ort.InferenceSession(onnx\_filename)

input\_name = onnx\_session.get\_inputs()\[0\].name

output\_name = onnx\_session.get\_outputs()\[0\].name

\# display information about input tensors in ONNX

**print**("\\nInformation about input tensors in ONNX:")

**for** i, input\_tensor in enumerate(onnx\_session.get\_inputs()):

**print**(f"{i + 1}. Name: {input\_tensor.name}, Data Type: {input\_tensor.type}, Shape: {input\_tensor.shape}")

\# display information about output tensors in ONNX

**print**("\\nInformation about output tensors in ONNX:")

**for** i, output\_tensor in enumerate(onnx\_session.get\_outputs()):

**print**(f"{i + 1}. Name: {output\_tensor.name}, Data Type: {output\_tensor.type}, Shape: {output\_tensor.shape}")

\# convert data to floating-point format (float32)

X\_float32 = X.astype(np.float32)

\# predict classes for the entire dataset using ONNX

y\_pred\_onnx = onnx\_session.run(\[output\_name\], {input\_name: X\_float32})\[0\]

\# evaluate the accuracy of the ONNX model

accuracy\_onnx = accuracy\_score(y, y\_pred\_onnx)

**print**("\\nAccuracy of Quadratic Discriminant Analysis model in ONNX format:", accuracy\_onnx)

Output:

Python    Accuracy of Quadratic Discriminant Analysis model: 0.98

Python

Python    Classification Report:

Python                   precision    recall  f1-score   support

Python

Python               0       1.00      1.00      1.00        50

Python               1       0.98      0.96      0.97        50

Python               2       0.96      0.98      0.97        50

Python

Python        accuracy                           0.98       150

Python       macro avg       0.98      0.98      0.98       150

Python    weighted avg       0.98      0.98      0.98       150

Python

Python    Model saved to C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\qda\_iris.onnx

This time, the model was successfully saved in ONNX format. However, when executing it, errors are displayed in the Errors tab:

     onnx\_session = ort.InferenceSession(onnx\_filename)    Iris\_QuadraticDiscriminantAnalysisClassifier.py    55    1

    self.\_create\_inference\_session(providers, provider\_options, disabled\_optimizers)    onnxruntime\_inference\_collection.py    383    1

    sess = C.InferenceSession(session\_options, self.\_model\_path, True, self.\_read\_config\_from\_model)    onnxruntime\_inference\_collection.py    424    1

onnxruntime.capi.onnxruntime\_pybind11\_state.InvalidGraph: \[ONNXRuntimeError\] : 10 : INVALID\_GRAPH : Load model from C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\qda\_iris.onnx failed:This is an invalid mode    onnxruntime\_inference\_collection.py    424    1

Iris\_QuadraticDiscriminantAnalysisClassifier.py finished in 2063 ms        5    1

Conversion of the Quadratic Discriminant Analysis Classifier model to ONNX encountered an error.

### **Conclusions**

In this study, we conducted research on 33 classification models using the Iris dataset, leveraging the Scikit-learn library version 1.2.2.

1\. From this set, six models encountered difficulties when converting them to the ONNX format:

1. **DummyClassifier**: Dummy Classifier;
2. **GaussianProcessClassifier**:  Gaussian Process Classifier;
3. **LabelPropagation**: Label Propagation Classifier ;
4. **LabelSpreading**: Label Spreading Classifier;
5. **NearestCentroid**: Nearest Centroid Classifier;
6. **QuadraticDiscriminantAnalysis**: Quadratic Discriminant Analysis Classifier.

It seems that these models are more complex in terms of their structure and/or logic, and their adaptation for ONNX format might require additional efforts. It's also possible that they utilize specific data structures or algorithms that are not fully supported or suitable for the ONNX format.

2\. The remaining 27 models were successfully converted to the ONNX format and demonstrated the preservation of their accuracy. This highlights the effectiveness of ONNX as a tool for saving and restoring machine learning models, enabling easy model transfer between different environments and applications while maintaining their performance.

The full list of models successfully converted to the ONNX format includes:

01. **SVC**: Support Vector Classifier;
02. **LinearSVC**: Linear Support Vector Classifier;
03. **NuSVC**: Nu Support Vector Classifier;
04. **AdaBoostClassifier**: Adaptive Boosting Classifier;
05. **BaggingClassifier**: Bootstrap Aggregating Classifier;
06. **BernoulliNB**: Bernoulli Naive Bayes Classifier;
07. **CategoricalNB**: Categorical Naive Bayes Classifier;
08. **ComplementNB**: Complement Naive Bayes Classifier;
09. **DecisionTreeClassifier**: Decision Tree Classifier;
10. **ExtraTreeClassifier**: Extra Tree Classifier;
11. **ExtraTreesClassifier**: Extra Trees Classifier;
12. **GaussianNB**: Gaussian Naive Bayes Classifier;
13. **GradientBoostingClassifier**: Gradient Boosting Classifier;
14. **HistGradientBoostingClassifier**: Histogram-Based Gradient Boosting Classifier;
15. **KNeighborsClassifier**: k-Nearest Neighbors Classifier;
16. **LinearDiscriminantAnalysis**: Linear Discriminant Analysis Classifier;
17. **LogisticRegression**: Logistic Regression Classifier;
18. **LogisticRegressionCV**: Logistic Regression Classifier with Cross-Validation;
19. **MLPClassifier**: Multi-Layer Perceptron Classifier;
20. **MultinomialNB**: Multinomial Naive Bayes Classifier;
21. **PassiveAggressiveClassifier**: Passive-Aggressive Classifier;
22. **Perceptron**: Perceptron Classifier;
23. **RadiusNeighborsClassifier**: Radius Neighbors Classifier;
24. **RandomForestClassifier**: Random Forest Classifier;
25. **RidgeClassifier**: Ridge Classifier;
26. **RidgeClassifierCV**: Ridge Classifier with Cross-Validation;
27. **SGDClassifier**: Stochastic Gradient Descent Classifier.

3\. Additionally, during the research, models that exhibited outstanding classification performance on the Iris dataset were identified. Classification models such as Random Forest Classifier, Gradient Boosting Classifier, Bagging Classifier, Decision Tree Classifier, Extra Tree Classifier, Extra Trees Classifier, and Hist Gradient Boosting Classifier achieved perfect accuracy in predictions. This implies that they can accurately determine the class to which each iris sample belongs.

These results can be especially valuable when selecting the best model for specific classification tasks. Models that achieved perfect accuracy on the Iris data can be an excellent choice for tasks involving the analysis or classification of similar data.

Thus, this research emphasizes the importance of choosing the right model for specific tasks and highlights the advantages of using ONNX for preserving and applying machine learning models for classification tasks.

### Conclusion

In this article, we analyzed 33 classification models using the Iris dataset with Scikit-learn version 1.2.2.

Of all the models we examined, six proved challenging to convert to the ONNX format. These models include the Dummy Classifier, Gaussian Process Classifier, Label Propagation Classifier, Label Spreading Classifier, Nearest Centroid Classifier, and Quadratic Discriminant Analysis Classifier. Their complex structure or logic likely requires additional adaptation for successful conversion to the ONNX format.

The remaining 27 models were successfully converted to the ONNX format and demonstrated the preservation of their accuracy. This reaffirms ONNX's efficiency in preserving and restoring machine learning models, ensuring portability while maintaining model performance.

Notably, some models, such as the Random Forest Classifier, Gradient Boosting Classifier, Bagging Classifier, Decision Tree Classifier, Extra Tree Classifier, Extra Trees Classifier, and Hist Gradient Boosting Classifier, achieved perfect accuracy in classifying Iris data. These models can be particularly attractive for tasks where high accuracy is critical.

This research underscores the importance of selecting the right model for specific tasks and demonstrates the benefits of using ONNX for preserving and applying machine learning models in classification tasks.

All the scripts from the article are also available in the [public project](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#public "https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#public") "MQL5\\Shared Projects\\Scikit.Classification.ONNX."

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/13451](https://www.mql5.com/ru/articles/13451)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13451.zip "Download all attachments in the single ZIP archive")

[Scikit-learn-ClassificationModels-ONNX.zip](https://www.mql5.com/en/articles/download/13451/scikit-learn-classificationmodels-onnx.zip "Download Scikit-learn-ClassificationModels-ONNX.zip")(220.05 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/455642)**
(2)


![Xiaoyu Huang](https://c.mql5.com/avatar/2016/3/56ECB06F-85EA.png)

**[Xiaoyu Huang](https://www.mql5.com/en/users/maxlake)**
\|
13 Oct 2023 at 18:10

**MetaQuotes:**

Check out the new article: [Classification models in the Scikit-Learn library and their export to ONNX](https://www.mql5.com/en/articles/13451).

Author: [MetaQuotes](https://www.mql5.com/en/users/MetaQuotes "MetaQuotes")

Very useful article,thank you!

![Rorschach](https://c.mql5.com/avatar/2022/3/6244E941-EF6F.jpg)

**[Rorschach](https://www.mql5.com/en/users/rorschach)**
\|
13 Oct 2023 at 20:32

Thanks, that's interesting. I'd love to see the speeds.


![Alternative risk return metrics in MQL5](https://c.mql5.com/2/58/alternative_risk_return_metrics___avatar_3.png)[Alternative risk return metrics in MQL5](https://www.mql5.com/en/articles/13514)

In this article we present the implementation of several risk return metrics billed as alternatives to the Sharpe ratio and examine hypothetical equity curves to analyze their characteristics.

![Studying PrintFormat() and applying ready-made examples](https://c.mql5.com/2/56/printformat-avatar.png)[Studying PrintFormat() and applying ready-made examples](https://www.mql5.com/en/articles/12905)

The article will be useful for both beginners and experienced developers. We will look at the PrintFormat() function, analyze examples of string formatting and write templates for displaying various information in the terminal log.

![Category Theory in MQL5 (Part 23): A different look at the Double Exponential Moving Average](https://c.mql5.com/2/58/category-theory-p18-avatar.png)[Category Theory in MQL5 (Part 23): A different look at the Double Exponential Moving Average](https://www.mql5.com/en/articles/13456)

In this article we continue with our theme in the last of tackling everyday trading indicators viewed in a ‘new’ light. We are handling horizontal composition of natural transformations for this piece and the best indicator for this, that expands on what we just covered, is the double exponential moving average (DEMA).

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 2): Indicator Signals: Multi Timeframe Parabolic SAR Indicator](https://c.mql5.com/2/58/FXSAR_MTF_MCEA_icon.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 2): Indicator Signals: Multi Timeframe Parabolic SAR Indicator](https://www.mql5.com/en/articles/13470)

The Multi-Currency Expert Advisor in this article is Expert Advisor or trading robot that can trade (open orders, close orders and manage orders for example: Trailing Stop Loss and Trailing Profit) for more than 1 symbol pair only from one symbol chart. This time we will use only 1 indicator, namely Parabolic SAR or iSAR in multi-timeframes starting from PERIOD\_M15 to PERIOD\_D1.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mithyvlmxibrqdqkuiofeogduzxbwrtd&ssn=1769252673801514317&ssn_dr=0&ssn_sr=0&fv_date=1769252673&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13451&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Classification%20models%20in%20the%20Scikit-Learn%20library%20and%20their%20export%20to%20ONNX%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925267332281186&fz_uniq=5083317953803327808&sv=2552)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).

![close](https://c.mql5.com/i/close.png)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)