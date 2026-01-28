---
title: Data Science and Machine Learning (Part 19): Supercharge Your AI models with AdaBoost
url: https://www.mql5.com/en/articles/14034
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:03:30.548398
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/14034&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068995144109195200)

MetaTrader 5 / Trading


### What is Adaboost?

Adaboost, short for adaptive boosting is an ensemble machine learning model that attempts to build a strong classifier out of weak classifiers.

### How does it work?

- The algorithm assigns weights to instances based on their correct or incorrect classification.
- It combines weak learners using a weighted sum.
- The final strong learner is a linear combination of weak learners with weights determined during the training process.

![adaboost in mql5](https://c.mql5.com/2/64/article_image.png)

### Why should anyone care about using Adaboost?

Adaboost provides several benefits including:

- **Improved Accuracy** – Boosting can improve the overall accuracy of the model by combining several weak model predictions. Averaging the predictions made by all the models for regression or voting over them for classification to increase the accuracy of the final model.
- **Robustness to Overfitting** – Boosting can reduce the risk of overfitting by assigning the weights to the misclassified inputs.
- **Better handling of imbalanced data** – Boosting can handle the imbalanced data by focusing more on the data points that are misclassified.
- **Better Interpretability** – Boosting can increase the interpretability of the model by breaking the model decision process into multiple processes.

### What is a Decision Stump?

A Decision Stump is a simple machine learning model used as a weak learner in ensemble methods like Adaboost. It is essentially a machine learning model simplified to make decisions based on a single feature and a threshold. The goal of using a decision stump as a weak learner is to capture a basic pattern in the data that can contribute to improving the overall ensemble model.

Below is a brief explanation of the theory behind a decision stump, using the [Decision tree](https://www.mql5.com/en/articles/13765) classifier as an example:

1\. **Structure:**

\- A decision stump makes a binary decision based on a single feature and a threshold.

\- It splits the data into two groups: those with feature values below the threshold and those with values above.

\- This is commonly deployed in the constructor of most of our classifiers in this [Library:](https://www.mql5.com/go?link=https://github.com/MegaJoctan/MALE5/releases/tag/v2.0.0 "https://github.com/MegaJoctan/MALE5/releases/tag/v2.0.0")

```
 CDecisionTreeClassifier(uint min_samples_split=2, uint max_depth=2, mode mode_=MODE_GINI);
~CDecisionTreeClassifier(void);
```

2\. **Training:**

\- During training, the decision stump identifies the feature and threshold that minimizes a certain criterion, often the weighted misclassification error.

\- The misclassification error is calculated for each possible split, and the one with the lowest error is chosen.

\- The fit function(s) is where all the training is done:

```
void fit(matrix &x, vector &y);
```

3\. **Prediction:**

\- For prediction, a data point is classified based on whether its feature value is above or below the chosen threshold.

```
double predict(vector &x);
vector predict(matrix &x);
```

Commonly used weak learners in ensemble methods like AdaBoost include:

1. **Decision Stumps/Decision Trees:**

\- As described above, decision stumps or shallow decision trees are commonly used due to their simplicity.

2\. **Linear Models:**

\- Linear models like logistic regression or linear SVMs can be used as weak learners.

3\. **Polynomial Models:**

\- Higher-degree polynomial models can capture more complex relationships, and using low-degree polynomials can act as weak learners.

4\. **Neural Networks (Shallow):**

\- Shallow neural networks with a small number of layers and neurons are sometimes used.

5\. **Gaussian Models:**

\- Gaussian models, such as Gaussian Naive Bayes, can be employed as weak learners.

The choice of a weak learner depends on the specific characteristics of the data and the problem at hand. In practice, decision stumps are popular due to their simplicity and efficiency in boosting algorithms. The ensemble approach of AdaBoost enhances the performance by combining the predictions of these weak learners.

Since a decision stump is a model we need to give the Adaboost class constructor the arguments of our weak model parameters.

```
class AdaBoost
  {

protected:
                     vector m_alphas;
                     vector classes_in_data;
                     uint m_min_split, m_max_depth;

                     CDecisionTreeClassifier *weak_learners[]; //store weak_learner pointers for memory allocation tracking
                     CDecisionTreeClassifier *weak_learner;

                     uint m_estimators;

public:
                     AdaBoost(uint min_split, uint max_depth, uint n_estimators=50);
                    ~AdaBoost(void);

                    void fit(matrix &x, vector &y);
                    int predict(vector &x);
                    vector predict(matrix &x);
  };
```

This example uses the decision tree but any classification machine learning model can be deployed.

### Building the AdaBoost class:

The term **number of estimators** refers to the number of weak learners (base models or decision stumps) that are combined to create the final strong learner in an ensemble learning algorithm. In the context of algorithms like AdaBoost or gradient boosting, this parameter is often denoted as n\_estimators.

```
AdaBoost::AdaBoost(uint min_split, uint max_depth, uint n_estimators=50)
:m_estimators(n_estimators),
 m_min_split(min_split),
 m_max_depth(max_depth)
 {
   ArrayResize(weak_learners, n_estimators);   //Resizing the array to retain the number of base weak_learners
 }
```

From the protected section of the class you have seen the vector **m\_alphas**, while also hearing the term **weights** several times in this article, Below is the clarification:

### Alpha Values:

The alpha values represent the contribution or weight assigned to each weak learner in the ensemble, these values are calculated based on the performance of each weak learner during training.

Higher alpha values are assigned to weak learners that perform well in reducing training errors.

The formula to calculate alpha is often given by:

![alpha formula adaboost](https://c.mql5.com/2/64/alpha_formula__1.gif)

where:

![](https://c.mql5.com/2/64/error_i.gif) is the weighted error of the _t_-th weak learner.

Implementation:

```
double alpha = 0.5 * log((1-error) / (error + 1e-10));
```

### Weights:

The weights vectors represent the importance of each training instance at each iteration.

During training, the weights of misclassified instances are increased to focus on the difficult-to-classify examples in the next iteration.

The formula for updating instance weights is often given by:

![weights adaboost](https://c.mql5.com/2/64/weights_formula.gif)

![](https://c.mql5.com/2/64/wi5t.gif) is the weight of instance _i_ at iteration

![](https://c.mql5.com/2/64/alpha_t.gif) ​is the weight assigned to the _t_-th weak learner.

![](https://c.mql5.com/2/64/yi.gif) ​is the true label of instance

![](https://c.mql5.com/2/64/hixi.gif) is the prediction of the _t_-th weak learner on instance

![](https://c.mql5.com/2/64/zt.gif) is a normalization factor to ensure that the weights sum to _1_.

Implementation:

```
 for (ulong j=0; j<m; j++)
    misclassified[j] = (preds[j] != y[j]);

 error = (misclassified * weights).Sum() / (double)weights.Sum();

//--- Calculate the weight of a weak learner in the final model

double alpha = 0.5 * log((1-error) / (error + 1e-10));

//--- Update instance weights

weights *= exp(-alpha * y * preds);
weights /= weights.Sum();
```

### Training the Adaboost model:

Just like other ensemble techniques, n number of models ( _referred to as m\_estimators in the below function)_ are used to make predictions on the same data, the majority vote or other techniques can be deployed to determine the best possible outcome.

```
void AdaBoost::fit(matrix &x,vector &y)
 {
   m_alphas.Resize(m_estimators);
   classes_in_data = MatrixExtend::Unique(y); //Find the target variables in the class

   ulong m = x.Rows(), n = x.Cols();
   vector weights(m); weights = weights.Fill(1.0) / m; //Initialize instance weights
   vector preds(m);
   vector misclassified(m);


   double error = 0;

   for (uint i=0; i<m_estimators; i++)
    {

//---

      weak_learner = new CDecisionTreeClassifier(this.m_min_split, m_max_depth);

      weak_learner.fit(x, y); //fitting the randomized data to the i-th weak_learner
      preds = weak_learner.predict(x); //making predictions for the i-th weak_learner


       for (ulong j=0; j<m; j++)
          misclassified[j] = (preds[j] != y[j]);

       error = (misclassified * weights).Sum() / (double)weights.Sum();

      //--- Calculate the weight of a weak learner in the final weak_learner

      double alpha = 0.5 * log((1-error) / (error + 1e-10));

      //--- Update instance weights

      weights *= exp(-alpha * y* preds);
      weights /= weights.Sum();

      //--- save a weak learner and its weight

      this.m_alphas[i] = alpha;
      this.weak_learners[i] = weak_learner;
    }
 }
```

Just like any other ensemble technique, [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics) "https://en.wikipedia.org/wiki/Bootstrapping_(statistics)") is a crucial thing too, without it all the models and the data are simply the same, and this could cause the performance of all the models to become indistinguishable from one another, Bootstrapping needs to be deployed:

```
void AdaBoost::fit(matrix &x,vector &y)
 {
   m_alphas.Resize(m_estimators);
   classes_in_data = MatrixExtend::Unique(y); //Find the target variables in the class

   ulong m = x.Rows(), n = x.Cols();
   vector weights(m); weights = weights.Fill(1.0) / m; //Initialize instance weights
   vector preds(m);
   vector misclassified(m);

//---

   matrix data = MatrixExtend::concatenate(x, y);
   matrix temp_data;

   matrix x_subset;
   vector y_subset;

   double error = 0;

   for (uint i=0; i<m_estimators; i++)
    {

      temp_data = data;
      MatrixExtend::Randomize(temp_data, this.m_random_state, this.m_boostrapping);

       if (!MatrixExtend::XandYSplitMatrices(temp_data, x_subset, y_subset)) //Get randomized subsets
         {
            ArrayRemove(weak_learners,i,1); //Delete the invalid weak_learner
            printf("%s %d Failed to split data",__FUNCTION__,__LINE__);
            continue;
         }

//---

      weak_learner = new CDecisionTreeClassifier(this.m_min_split, m_max_depth);

      weak_learner.fit(x_subset, y_subset); //fitting the randomized data to the i-th weak_learner
      preds = weak_learner.predict(x_subset); //making predictions for the i-th weak_learner

      //printf("[%d] Accuracy %.3f ",i,Metrics::accuracy_score(y_subset, preds));

       for (ulong j=0; j<m; j++)
          misclassified[j] = (preds[j] != y_subset[j]);

       error = (misclassified * weights).Sum() / (double)weights.Sum();

      //--- Calculate the weight of a weak learner in the final weak_learner

      double alpha = 0.5 * log((1-error) / (error + 1e-10));

      //--- Update instance weights

      weights *= exp(-alpha * y_subset * preds);
      weights /= weights.Sum();

      //--- save a weak learner and its weight

      this.m_alphas[i] = alpha;
      this.weak_learners[i] = weak_learner;
    }
 }
```

The class constructor also had to be changed:

```
class AdaBoost
  {

protected:
                     vector m_alphas;
                     vector classes_in_data;
                     int m_random_state;
                     bool m_boostrapping;
                     uint m_min_split, m_max_depth;

                     CDecisionTreeClassifier *weak_learners[]; //store weak_learner pointers for memory allocation tracking
                     CDecisionTreeClassifier *weak_learner;

                     uint m_estimators;

public:
                     AdaBoost(uint min_split, uint max_depth, uint n_estimators=50, int random_state=42, bool bootstrapping=true);
                    ~AdaBoost(void);

                    void fit(matrix &x, vector &y);
                    int predict(vector &x);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AdaBoost::AdaBoost(uint min_split, uint max_depth, uint n_estimators=50, int random_state=42, bool bootstrapping=true)
:m_estimators(n_estimators),
 m_random_state(random_state),
 m_boostrapping(bootstrapping),
 m_min_split(min_split),
 m_max_depth(max_depth)
 {
   ArrayResize(weak_learners, n_estimators);   //Resizing the array to retain the number of base weak_learners
 }
```

### Getting the Majority Predictions

Using the trained weights the Adaboost classifier determines the class with the maximum votes, meaning it is more likely to appear than the others.

```
int AdaBoost::predict(vector &x)
 {
   // Combine weak learners using weighted sum

   vector weak_preds(m_estimators),
          final_preds(m_estimators);

   for (uint i=0; i<this.m_estimators; i++)
     weak_preds[i] = this.weak_learners[i].predict(x);

  return (int)weak_preds[(this.m_alphas*weak_preds).ArgMax()]; //Majority decision class
 }
```

Now let us see how we can use the model inside an Expert Advisor:

```
#include <MALE5\Ensemble\AdaBoost.mqh>

DecisionTree::AdaBoost *ada_boost_tree;
LogisticRegression::AdaBoost *ada_boost_logit;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   string headers;
   matrix data = MatrixExtend::ReadCsv("iris.csv",headers);

   matrix x; vector y;
   MatrixExtend::XandYSplitMatrices(data,x,y);

   ada_boost_tree = new DecisionTree::AdaBoost(2,1,10,42);
   ada_boost_tree.fit(x,y);

   vector predictions = ada_boost_tree.predict(x);

   printf("Adaboost acc = %.3f",Metrics::accuracy_score(y, predictions));

//---
   return(INIT_SUCCEEDED);
  }
```

Using the [iris.csv](https://www.mql5.com/go?link=https://www.kaggle.com/datasets/omegajoctan/iris-csv "https://www.kaggle.com/datasets/omegajoctan/iris-csv") dataset just for the sake of building the model and debugging purposes. This resulted into:

```
2024.01.17 17:52:27.914 AdaBoost Test (EURUSD,H1)       Adaboost acc = 0.960
```

It seems our model is doing well so far, with accuracy values around 90 percent(s), After I set the **random\_state** to -1 which will cause the [GetTickCount](https://www.mql5.com/en/docs/common/gettickcount) to be used as a random seed each time the EA is run so that I can assess the model in a much more random environment.

```
QK      0       17:52:27.914    AdaBoost Test (EURUSD,H1)       Adaboost acc = 0.960
LL      0       17:52:35.436    AdaBoost Test (EURUSD,H1)       Adaboost acc = 0.947
JD      0       17:52:42.806    AdaBoost Test (EURUSD,H1)       Adaboost acc = 0.960
IL      0       17:52:50.071    AdaBoost Test (EURUSD,H1)       Adaboost acc = 0.933
MD      0       17:52:57.822    AdaBoost Test (EURUSD,H1)       Adaboost acc = 0.967
```

The same coding patterns and structure can be followed for other weak learners present in our library, See when logistic regression was used as a weak learner:

The only difference in the entire class is found under the **fit** function the type of model deployed as a weak learner is **Logistic Regression**:

```
void AdaBoost::fit(matrix &x,vector &y)
 {
   m_alphas.Resize(m_estimators);
   classes_in_data = MatrixExtend::Unique(y); //Find the target variables in the class

   ulong m = x.Rows(), n = x.Cols();
   vector weights(m); weights = weights.Fill(1.0) / m; //Initialize instance weights
   vector preds(m);
   vector misclassified(m);

//---

   matrix data = MatrixExtend::concatenate(x, y);
   matrix temp_data;

   matrix x_subset;
   vector y_subset;

   double error = 0;

   for (uint i=0; i<m_estimators; i++)
    {

      temp_data = data;
      MatrixExtend::Randomize(temp_data, this.m_random_state, this.m_boostrapping);

       if (!MatrixExtend::XandYSplitMatrices(temp_data, x_subset, y_subset)) //Get randomized subsets
         {
            ArrayRemove(weak_learners,i,1); //Delete the invalid weak_learner
            printf("%s %d Failed to split data",__FUNCTION__,__LINE__);
            continue;
         }

//---

      weak_learner = new CLogisticRegression();

      weak_learner.fit(x_subset, y_subset); //fitting the randomized data to the i-th weak_learner
      preds = weak_learner.predict(x_subset); //making predictions for the i-th weak_learner

       for (ulong j=0; j<m; j++)
          misclassified[j] = (preds[j] != y_subset[j]);

       error = (misclassified * weights).Sum() / (double)weights.Sum();

      //--- Calculate the weight of a weak learner in the final weak_learner

      double alpha = 0.5 * log((1-error) / (error + 1e-10));

      //--- Update instance weights

      weights *= exp(-alpha * y_subset * preds);
      weights /= weights.Sum();

      //--- save a weak learner and its weight

      this.m_alphas[i] = alpha;
      this.weak_learners[i] = weak_learner;
    }
 }
```

### Adaboosted AI-Models in the Strategy Tester.

By using the Bollinger band indicator applied to the open price, we are trying to train our models to learn to predict the next close candle whether it is **bullish**, **bearish** or none of the two( **HOLD**).

Let us build an Expert advisor to test our models in the trading environment,  starting with the code to collect the training data:

**Collecting the Training data:**

```
//--- Data Collection for training the model

//--- x variables Bollinger band only

   matrix dataset;

   indicator_v.CopyIndicatorBuffer(bb_handle,0,0,train_bars); //Main LINE
   dataset = MatrixExtend::concatenate(dataset, indicator_v);

   indicator_v.CopyIndicatorBuffer(bb_handle,1,0,train_bars); //UPPER BB
   dataset = MatrixExtend::concatenate(dataset, indicator_v);

   indicator_v.CopyIndicatorBuffer(bb_handle,2,0,train_bars); //LOWER BB
   dataset = MatrixExtend::concatenate(dataset, indicator_v);

//--- Target Variable

   int size = CopyRates(Symbol(),PERIOD_CURRENT,0,train_bars,rates);
   vector y(size);

   switch(model)
     {
      case  DECISION_TREE:
         {
            for (ulong i=0; i<y.Size(); i++)
              {
                 if (rates[i].close > rates[i].open)
                   y[i] = 1; //buy signal
                 else if (rates[i].close < rates[i].open)
                   y[i] = 2; //sell signal
                 else
                   y[i] = 0; //Hold signal
              }
         }
        break;
      case LOGISTIC_REGRESSION:
            for (ulong i=0; i<indicator_v.Size(); i++)
              {
                 y[i] = (rates[i].close > rates[i].open); //if close > open buy else sell
              }
        break;
     }

   dataset = MatrixExtend::concatenate(dataset, y); //Add the target variable to the dataset

   if (MQLInfoInteger(MQL_DEBUG))
    {
       Print("Data Head");
       MatrixExtend::PrintShort(dataset);
    }
```

Notice the change in the making of the target variable? Since the decision tree is versatile and can handle multiple classes well in the target variable, it has the ability to classify three patterns in the market, **BUY**, **SELL**, and **HOLD**. Unlike the Logistic regression model which has the sigmoid function at its core which classifies well two classes 0 and 1, the best thing is to prepare the target variable condition to be either **BUY** or **SELL**.

**Training and Testing the Trained Model:**

An Elephant in the room:

```
MatrixExtend::TrainTestSplitMatrices(dataset,train_x,train_y,test_x,test_y,0.7,_random_state);
```

This function splits the data into training and testing samples given a random state for shuffling, 70% of data to test sample while the rest 30% for testing.

But before we can use the newly collected data, **Normalization is still crucial for model performance**.

```
//--- Training and testing the trained model

   matrix train_x, test_x;
   vector train_y, test_y;

   MatrixExtend::TrainTestSplitMatrices(dataset,train_x,train_y,test_x,test_y,0.7,_random_state); //Train test split data | This function splits the data into training and testing sample given a random state and 70% of data to test while the rest 30% for testing

   train_x = scaler.fit_transform(train_x); //Standardize the training data
   test_x = scaler.transform(test_x); //Do the same for the test data

   Print("-----> ",EnumToString(model));

   vector preds;
   switch(model)
     {
      case  DECISION_TREE:
         ada_boost_tree = new DecisionTree::AdaBoost(tree_min_split, tree_max_depth, _n_estimators, -1, _bootstrapping);  //Building the

        //--- Training

         ada_boost_tree.fit(train_x, train_y);

         preds = ada_boost_tree.predict(train_x);
         printf("Train Accuracy %.3f",Metrics::accuracy_score(train_y, preds));

        //--- Testing

         preds = ada_boost_tree.predict(test_x);
         printf("Test Accuracy %.3f",Metrics::accuracy_score(test_y, preds));

        break;
      case LOGISTIC_REGRESSION:
         ada_boost_logit = new LogisticRegression::AdaBoost(_n_estimators,-1, _bootstrapping);

        //--- Training

         ada_boost_logit.fit(train_x, train_y);

         preds = ada_boost_logit.predict(train_x);
         printf("Train Accuracy %.3f",Metrics::accuracy_score(train_y, preds));

        //--- Testing

         preds = ada_boost_logit.predict(test_x);
         printf("Test Accuracy %.3f",Metrics::accuracy_score(test_y, preds));

        break;
     }
```

**Outputs:**

```
PO      0       22:59:11.807    AdaBoost Test (EURUSD,H1)       -----> DECISION_TREE
CI      0       22:59:20.204    AdaBoost Test (EURUSD,H1)       Building Estimator [1/10] Accuracy Score 0.561
OD      0       22:59:27.883    AdaBoost Test (EURUSD,H1)       Building Estimator [2/10] Accuracy Score 0.601
NP      0       22:59:38.316    AdaBoost Test (EURUSD,H1)       Building Estimator [3/10] Accuracy Score 0.541
LO      0       22:59:48.327    AdaBoost Test (EURUSD,H1)       Building Estimator [4/10] Accuracy Score 0.549
LK      0       22:59:56.813    AdaBoost Test (EURUSD,H1)       Building Estimator [5/10] Accuracy Score 0.570
OF      0       23:00:09.552    AdaBoost Test (EURUSD,H1)       Building Estimator [6/10] Accuracy Score 0.517
GR      0       23:00:18.322    AdaBoost Test (EURUSD,H1)       Building Estimator [7/10] Accuracy Score 0.571
GI      0       23:00:29.254    AdaBoost Test (EURUSD,H1)       Building Estimator [8/10] Accuracy Score 0.556
HE      0       23:00:37.632    AdaBoost Test (EURUSD,H1)       Building Estimator [9/10] Accuracy Score 0.599
DS      0       23:00:47.522    AdaBoost Test (EURUSD,H1)       Building Estimator [10/10] Accuracy Score 0.567
OP      0       23:00:47.524    AdaBoost Test (EURUSD,H1)       Train Accuracy 0.590
OG      0       23:00:47.525    AdaBoost Test (EURUSD,H1)       Test Accuracy 0.513
MK      0       23:24:06.573    AdaBoost Test (EURUSD,H1)       Building Estimator [1/10] Accuracy Score 0.491
HK      0       23:24:06.575    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.43700
QO      0       23:24:06.575    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.43432
KP      0       23:24:06.576    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.43168
MD      0       23:24:06.577    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.42909
FI      0       23:24:06.578    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.42652
QJ      0       23:24:06.579    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.42400
IN      0       23:24:06.580    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.42151
NS      0       23:24:06.581    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.41906
GD      0       23:24:06.582    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.41664
DK      0       23:24:06.582    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.41425
IQ      0       23:24:06.585    AdaBoost Test (EURUSD,H1)       Building Estimator [2/10] Accuracy Score 0.477
JP      0       23:24:06.586    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.43700
DE      0       23:24:06.587    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.43432
RF      0       23:24:06.588    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.43168
KJ      0       23:24:06.588    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.42909
FO      0       23:24:06.589    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.42652
NP      0       23:24:06.590    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.42400
CD      0       23:24:06.591    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.42151
KI      0       23:24:06.591    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.41906
NM      0       23:24:06.592    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.41664
EM      0       23:24:06.592    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.41425
EO      0       23:24:06.593    AdaBoost Test (EURUSD,H1)       Building Estimator [3/10] Accuracy Score 0.477
KF      0       23:24:06.594    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.41931
HK      0       23:24:06.594    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.41690
RL      0       23:24:06.595    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.41452
IP      0       23:24:06.596    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.41217
KE      0       23:24:06.596    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.40985
DI      0       23:24:06.597    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.40757
IJ      0       23:24:06.597    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.40533
MO      0       23:24:06.598    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.40311
PS      0       23:24:06.599    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.40093
CG      0       23:24:06.600    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.39877
NE      0       23:24:06.601    AdaBoost Test (EURUSD,H1)       Building Estimator [4/10] Accuracy Score 0.499
EL      0       23:24:06.602    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.41931
MQ      0       23:24:06.603    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.41690
PE      0       23:24:06.603    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.41452
OF      0       23:24:06.604    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.41217
JK      0       23:24:06.605    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.40985
KO      0       23:24:06.606    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.40757
FP      0       23:24:06.606    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.40533
PE      0       23:24:06.607    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.40311
CI      0       23:24:06.608    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.40093
NI      0       23:24:06.609    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.39877
KS      0       23:24:06.610    AdaBoost Test (EURUSD,H1)       Building Estimator [5/10] Accuracy Score 0.499
QR      0       23:24:06.611    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.42037
MG      0       23:24:06.611    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.41794
LK      0       23:24:06.612    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.41555
ML      0       23:24:06.613    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.41318
PQ      0       23:24:06.614    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.41085
FE      0       23:24:06.614    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.40856
FF      0       23:24:06.615    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.40630
FJ      0       23:24:06.616    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.40407
KO      0       23:24:06.617    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.40187
NS      0       23:24:06.618    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.39970
EH      0       23:24:06.619    AdaBoost Test (EURUSD,H1)       Building Estimator [6/10] Accuracy Score 0.497
FH      0       23:24:06.620    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.41565
LM      0       23:24:06.621    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.41329
IQ      0       23:24:06.622    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.41096
KR      0       23:24:06.622    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.40867
LF      0       23:24:06.623    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.40640
NK      0       23:24:06.624    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.40417
OL      0       23:24:06.625    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.40197
RP      0       23:24:06.627    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.39980
OE      0       23:24:06.628    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.39767
EE      0       23:24:06.628    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.39556
QF      0       23:24:06.629    AdaBoost Test (EURUSD,H1)       Building Estimator [7/10] Accuracy Score 0.503
CN      0       23:24:06.630    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.41565
IR      0       23:24:06.631    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.41329
HG      0       23:24:06.632    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.41096
RH      0       23:24:06.632    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.40867
ML      0       23:24:06.633    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.40640
FQ      0       23:24:06.633    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.40417
QR      0       23:24:06.634    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.40197
NF      0       23:24:06.634    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.39980
EK      0       23:24:06.635    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.39767
CL      0       23:24:06.635    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.39556
LL      0       23:24:06.636    AdaBoost Test (EURUSD,H1)       Building Estimator [8/10] Accuracy Score 0.503
HD      0       23:24:06.637    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.44403
IH      0       23:24:06.638    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.44125
CM      0       23:24:06.638    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.43851
DN      0       23:24:06.639    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.43580
DR      0       23:24:06.639    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.43314
CG      0       23:24:06.640    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.43051
EK      0       23:24:06.640    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.42792
JL      0       23:24:06.641    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.42537
EQ      0       23:24:06.641    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.42285
OF      0       23:24:06.642    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.42037
GJ      0       23:24:06.642    AdaBoost Test (EURUSD,H1)       Building Estimator [9/10] Accuracy Score 0.469
GJ      0       23:24:06.643    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [1/10] mse 0.44403
ON      0       23:24:06.643    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [2/10] mse 0.44125
LS      0       23:24:06.644    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [3/10] mse 0.43851
HG      0       23:24:06.644    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [4/10] mse 0.43580
KH      0       23:24:06.645    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [5/10] mse 0.43314
FM      0       23:24:06.645    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [6/10] mse 0.43051
IQ      0       23:24:06.646    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [7/10] mse 0.42792
QR      0       23:24:06.646    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [8/10] mse 0.42537
IG      0       23:24:06.647    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [9/10] mse 0.42285
RH      0       23:24:06.647    AdaBoost Test (EURUSD,H1)       ---> Logistic regression build epoch [10/10] mse 0.42037
KS      0       23:24:06.648    AdaBoost Test (EURUSD,H1)       Building Estimator [10/10] Accuracy Score 0.469
NP      0       23:24:06.652    AdaBoost Test (EURUSD,H1)       Train Accuracy 0.491
GG      0       23:24:06.654    AdaBoost Test (EURUSD,H1)       Test Accuracy 0.447
```

So far, so good! Let's finish our Expert Advisor with the lines of code able to execute trades:

**Obtaining the Trading signals:**

```
void OnTick()
  {
//--- x variables Bollinger band only | The current buffer only this time

   indicator_v.CopyIndicatorBuffer(bb_handle,0,0,1); //Main LINE
   x_inputs[0] = indicator_v[0];
   indicator_v.CopyIndicatorBuffer(bb_handle,1,0,1); //UPPER BB
   x_inputs[1] = indicator_v[0];
   indicator_v.CopyIndicatorBuffer(bb_handle,2,0,1); //LOWER BB
   x_inputs[2] = indicator_v[0];

//---

   int signal = INT_MAX;
   switch(model)
     {
      case  DECISION_TREE:
        x_inputs = scaler.transform(x_inputs); //New inputs data must be normalized the same way
        signal = ada_boost_tree.predict(x_inputs);
        break;
      case LOGISTIC_REGRESSION:
         x_inputs = scaler.transform(x_inputs); //New inputs data must be normalized the same way
         signal = ada_boost_logit.predict(x_inputs);
        break;
     }
  }
```

Remember, **1** _represents the_ **buy signal** _for the_ **decision tree** _,_ **0** _represents the_ **buy signal** _for the_ **logistic regression;** **2** _represents the_ **sell signal** _for the decision tree;_ **1** _represents the_ **sell signal** _for the_ **logistic regression** _. We don't care about_ **0** _signal which represents_ **hold** _for the_ **_decision tree._** Let's unify these signals to be identified as 0 and 1 for buy and sell signals respectively.

```
case  DECISION_TREE:

  x_inputs = scaler.transform(x_inputs); //New inputs data must be normalized the same way
  signal = ada_boost_tree.predict(x_inputs);

   if (signal == 1) //buy signal for decision tree
     signal = 0;
   else if (signal == 2)
     signal = 1;
   else
     signal = INT_MAX; //UNKNOWN NUMBER FOR HOLD
  break;
```

**A simple strategy made out of our models:**

```
 SymbolInfoTick(Symbol(), ticks);

 if (isnewBar(PERIOD_CURRENT))
  {
   if (signal == 0) //buy signal
     {
        if (!PosExists(MAGIC_NUMBER, POSITION_TYPE_BUY))
          m_trade.Buy(lot_size, Symbol(), ticks.ask, ticks.ask-stop_loss*Point(), ticks.ask+take_profit*Point());
     }

   if (signal == 1)
     {
       if (!PosExists(MAGIC_NUMBER, POSITION_TYPE_SELL))
         m_trade.Sell(lot_size, Symbol(), ticks.bid, ticks.bid+stop_loss*Point(), ticks.bid-take_profit*Point());
     }
  }
```

Results in the strategy tester \| from January 2020 - February 2023 \| Timeframe 1 HOUR:

Decision Tree Adaboost:

![decision tree adaboost](https://c.mql5.com/2/64/decision_tree_one_hour_tf.png)

Logistic Regression Adaboost:

![logisitic regression adaboost 1hr tf ](https://c.mql5.com/2/64/logistic_regression_one_hour_tf.png)

Both performances on **ONE-HOUR TIMEFRAME** are very far from good. A major reason for this is that our strategy is based on the current bars, In my experience, these kinds of strategies are mostly suited well on higher timeframes as a single bar represents a significant move unlike the small bars that occur 24 in a day, let us try **12-HOUR TIMEFRAME.**

All parameters were left the same except for the **train\_bars** which were reduced to **100** from **1000**, as higher timeframes don't have many bars to request price history in the past.

Decision Tree Adaboost:

![decision tree adaboost](https://c.mql5.com/2/64/decision_tree_12_hour_tf.png)

Logistic Regression Adaboost:

![logistic regression adaboost 12 hr tf](https://c.mql5.com/2/64/logistic_regression_12_hour_tf.png)

### In conclusion:

AdaBoost emerges as a potent algorithm capable of significantly enhancing the performance of the AI models discussed throughout this article series. While it does come with a computational cost, the investment proves worthwhile when implemented judiciously. AdaBoost has found applications across diverse sectors and industries, including finance, entertainment, education, and beyond.

As we wrap up our exploration, it's essential to acknowledge the algorithm's versatility and its ability to address complex classification challenges by leveraging the collective strength of weak learners. The below Frequently Asked Questions (FAQs) provided aim to offer clarity and insight, addressing common queries that may arise during your exploration of AdaBoost.

### Frequently Asked Questions(FAQs) on Adaboost:

**Question**: How does AdaBoost work?

**Answer:** AdaBoost works by iteratively training weak learners (usually simple models like decision stumps) on the dataset, adjusting the weights of misclassified instances at each iteration. The final model is a weighted sum of the weak learners, with higher weights given to more accurate ones.

**Question:** What are weak learners in AdaBoost?

**Answer:** Weak learners are simple models that perform slightly better than random chance. Decision stumps (shallow decision trees) are commonly used as weak learners in AdaBoost, but other algorithms can also serve this purpose.

**Question:** What is the role of instance weights in AdaBoost?

**Answer:** Instance weights in AdaBoost control the importance of each training instance during the learning process. Initially, all weights are set equally, and they are adjusted at each iteration to focus more on misclassified instances, improving the model's ability to generalize.

**Question:** How does AdaBoost handle errors made by weak learners?

**Answer:** AdaBoost assigns higher weights to misclassified instances, forcing subsequent weak learners to focus more on correcting these errors. The final model gives more weight to weak learners with lower error rates.

**Question:** Is AdaBoost sensitive to noise and outliers?

**Answer:** AdaBoost can be sensitive to noise and outliers since it tries to correct misclassifications. Outliers might receive higher weights, influencing the final model. Robust weak learners or data preprocessing techniques may be applied to mitigate this sensitivity.

**Question:** Does AdaBoost suffer from overfitting?

**Answer:** AdaBoost can be prone to overfitting if the weak learners are too complex or if the dataset contains noise. Using simpler weak learners and applying techniques like cross-validation can help prevent overfitting.

**Question:** Can AdaBoost be used for regression problems?

**Answer:** AdaBoost is primarily designed for classification tasks, but it can be adapted for regression by modifying the algorithm to predict continuous values. Techniques like AdaBoost.R2 exist for regression problems.

**Question:** Are there alternatives to AdaBoost?

**Answer:** Yes, there are other ensemble learning methods, such as [Random Forest](https://www.mql5.com/en/articles/13765), [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting "https://en.wikipedia.org/wiki/Gradient_boosting"), and [XGBoost](https://en.wikipedia.org/wiki/XGBoost "https://en.wikipedia.org/wiki/XGBoost"). Each method has its strengths and weaknesses, and the choice depends on the specific characteristics of the data and the problem at hand.

**Question:** In which situations is AdaBoost particularly effective?

**Answer:** AdaBoost is effective when dealing with a variety of weak learners and in scenarios where there is a need to combine multiple classifiers to create a robust model. It is often used in face detection, text classification, and other real-world applications.

To stay updated on the development progress and bug fixes for this algorithm, as well as many others, please visit the GitHub repository at [https://github.com/MegaJoctan/MALE5](https://www.mql5.com/go?link=https://github.com/MegaJoctan/MALE5 "https://github.com/MegaJoctan/MALE5").

Peace out.

Attachments:

| File | Description\|Usage |
| --- | --- |
| Adaboost.mqh | Contains adaboost namespace classes for both logistic regression and the decision tree. |
| Logistic Regression .mqh | Contains the main logistic regression class |
| [MatrixExtend.mqh](https://www.mql5.com/en/articles/11858) | Contains additional matrix manipulation function |
| metrics.mqh | A library containing code for measuring performance of machine learning models |
| preprocessing.mqh | A class containing functions for preprocessing data to make it suitable for machine learning |
| tree.mqh | [Decision tree](https://www.mql5.com/en/articles/13862) library can be found on this file |
| AdaBoost Test.mq5(EA) | The main test Expert Advisor, all the code explained here, is executed inside this file |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14034.zip "Download all attachments in the single ZIP archive")

[AdaBoost\_attachments.zip](https://www.mql5.com/en/articles/download/14034/adaboost_attachments.zip "Download AdaBoost_attachments.zip")(24.41 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 04): Tester 101](https://www.mql5.com/en/articles/20917)
- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)

**[Go to discussion](https://www.mql5.com/en/forum/461323)**

![Ready-made templates for including indicators to Expert Advisors (Part 2): Volume and Bill Williams indicators](https://c.mql5.com/2/58/Volume_Bill_Williams_indicators_avatar.png)[Ready-made templates for including indicators to Expert Advisors (Part 2): Volume and Bill Williams indicators](https://www.mql5.com/en/articles/13277)

In this article, we will look at standard indicators of the Volume and Bill Williams' indicators category. We will create ready-to-use templates for indicator use in EAs - declaring and setting parameters, indicator initialization and deinitialization, as well as receiving data and signals from indicator buffers in EAs.

![Introduction to MQL5 (Part 3): Mastering the Core Elements of MQL5](https://c.mql5.com/2/65/Introduction_to_MQL5_rPart_38_Mastering_the_Core_Elements_of_MQL5____LOGO___small-transformed.png)[Introduction to MQL5 (Part 3): Mastering the Core Elements of MQL5](https://www.mql5.com/en/articles/14099)

Explore the fundamentals of MQL5 programming in this beginner-friendly article, where we demystify arrays, custom functions, preprocessors, and event handling, all explained with clarity making every line of code accessible. Join us in unlocking the power of MQL5 with a unique approach that ensures understanding at every step. This article sets the foundation for mastering MQL5, emphasizing the explanation of each line of code, and providing a distinct and enriching learning experience.

![Pair trading](https://c.mql5.com/2/58/pair_trading_avatar.png)[Pair trading](https://www.mql5.com/en/articles/13338)

In this article, we will consider pair trading, namely what its principles are and if there are any prospects for its practical application. We will also try to create a pair trading strategy.

![ALGLIB numerical analysis library in MQL5](https://c.mql5.com/2/58/ALGLIB_in_MQL5_avatar.png)[ALGLIB numerical analysis library in MQL5](https://www.mql5.com/en/articles/13289)

The article takes a quick look at the ALGLIB 3.19 numerical analysis library, its applications and new algorithms that can improve the efficiency of financial data analysis.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kmzeqrfhjepesplujytxebebdznbciul&ssn=1769180609223952558&ssn_dr=0&ssn_sr=0&fv_date=1769180609&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14034&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Data%20Science%20and%20Machine%20Learning%20(Part%2019)%3A%20Supercharge%20Your%20AI%20models%20with%20AdaBoost%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918060912543647&fz_uniq=5068995144109195200&sv=2552)

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