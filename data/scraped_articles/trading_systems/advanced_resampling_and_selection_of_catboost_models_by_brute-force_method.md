---
title: Advanced resampling and selection of CatBoost models by brute-force method
url: https://www.mql5.com/en/articles/8662
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:51:35.397550
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=fssjmintkutxmccncvsnxzpuktnrzxak&ssn=1769251894530573288&ssn_dr=0&ssn_sr=0&fv_date=1769251894&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8662&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Advanced%20resampling%20and%20selection%20of%20CatBoost%20models%20by%20brute-force%20method%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925189445060648&fz_uniq=5083163257671259702&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

In the previous [article](https://www.mql5.com/en/articles/8642), I tried to provide a general idea about the main machine learning model creation steps and its further practical use. In this part, I want to switch from naive models to statistically significant ones. Since the creation of a machine learning-based trading system is not a trivial task, we will start with some data preparation improvements which will assist in achieving optimal results. Various resampling techniques can be used to improve the presentation of the source data (training examples). One of such techniques will be discussed in this article.

A simple random sampling of labels used in the previous article has some disadvantages:

- Classes can be imbalanced. Suppose that the market was mainly growing during the training period, while the general population (the entire history of quotes) implies both ups and downs. In this case, naive sampling will create more buy labels and less sell labels. Accordingly, labels of one class will prevail over another one, due to which the model will learn to predict buy deals more often than sell deals, which however can be invalid for new data.

- Autocorrelation of features and labels. If random sampling is used, the labels of the same class follow one another, while the features themselves (such as for example, increments) change insignificantly. This process can be shown using an example of a regression model training - in this case it will turn out that autocorrelation will be observed in the model residuals, which will lead to a possible model overestimation and overtraining. This situation is shown below:

![](https://c.mql5.com/2/41/04-autocorrelation-1__1.png)

_Model 1_ has autocorrelation of residuals, which can be compared to model overfitting on certain market properties (for example, related to the volatility of training data), while other patterns are not taken into account. _Model 2_ has residuals with the same variance (on average), which indicates that the model covered more information or other dependencies were found (in addition to the correlation of neighboring samples).

The same effect is also observed for classification, though it is less intuitive because it has only a few classes, in contrast to a continuous variable used in regression models. However, the effect can still be measured, for example, by using Pearson residuals and similar metrics. These dependencies (as in _Model 1_) should be eliminated.

- Classes can overlap significantly. Imagine a hypothetical 2D feature space (multidimensional spaces is more complex), each point of which is assigned to class 0 or 1.

![](https://c.mql5.com/2/41/123__1.png)

When using random sampling, sets of examples can intersect. This may lead to a decrease in the distance (say, Euclidean distance) between points of different classes and to an increase in the distance between points of the same class, which leads to the creation of an overly complex model at the training stage, having many boundaries separating the classes. Small deviations in features cause jumps in model predictions from class to class. This effect ruins the model stability on new data and must be eliminated.

Ideally, class labels should not intersect in the feature space and should be separated either linearly (as is shown below) or by any other simple method. This solution would provide greater model stability on new data.

![](https://c.mql5.com/2/41/123__2.png)

### Analysis of the original GIGO dataset

Modified and improved functions from the previous part are used in this article. Load the data:

```
LOOK_BACK = 5
MA_PERIODS = [15, 55, 150, 250]

SYMBOL = 'EURUSD'
MARKUP = 0.00010
TIMEFRAME = mt5.TIMEFRAME_H1
START_DATE = datetime(2020, 1, 1)
TSTART_DATE = datetime(2015, 1, 1)
STOP_DATE = datetime(2021, 1, 1)

# make dataset
pr = get_prices(START_DATE, STOP_DATE)
pr = add_labels(pr, min=10, max=25, add_noize=0)
res = tester(pr, plot=True)
pca_plot(pr)
```

Since dimension of the original dataset is 20 features (loock\_back \* len(ma\_periods)) or any other large one, it is not very convenient to display it on a plane. Let us use the PCA method and display only 5 main components, which will allow to compact the feature space with the least information loss:

If you are not familiar with PCA (Principal Component Analysis), please search in [Google](https://www.mql5.com/go?link=https://www.google.com/search?newwindow=1%26sxsrf=ALeKk03u4Cdr9CkOPTW4JGHOYQVr9qV19g%253A1605539942496%26ei=ZpiyX5vkHdCprgTumpbgAg%26q=principal+component+analysis%26oq=principal+component+analysis%26gs_lcp=CgZwc3ktYWIQDDIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQR1AAWABg6PABaABwAngAgAEAiAEAkgEAmAEAqgEHZ3dzLXdpesgBCMABAQ%26sclient=psy-ab%26ved=0ahUKEwibr8evroftAhXQlIsKHW6NBSwQ4dUDCA0 "https://www.google.com/search?newwindow=1&sxsrf=ALeKk03u4Cdr9CkOPTW4JGHOYQVr9qV19g%3A1605539942496&ei=ZpiyX5vkHdCprgTumpbgAg&q=principal+component+analysis&oq=principal+component+analysis&gs_lcp=CgZwc3ktYWIQDDIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQR1AAWABg6PABaABwAngAgAEAiAEAkgEAmAEAqgEHZ3dzLXdpesgBCMABAQ&sclient=psy-ab&ved=0ahUKEwibr8evroftAhXQlIsKHW6NBSwQ4dUDCA0").

```
def pca_plot(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 5)
    components = pd.DataFrame(pca.fit_transform(data[data.columns[1:-1]]))
    components['labels'] = data['labels'].reset_index(drop = True)
    import seaborn as sns
    g = sns.PairGrid(components, hue="labels", height=1.2)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    plt.show()
```

![](https://c.mql5.com/2/41/Figure_1__12.png)

Now you can see the dependence of each component on the other: this is the 2D feature space, labeled into classes 0 and 1. Component pairs form loops, which are not similar to the usual point cloud. This is caused by the autocorrelation of points. The rings will disappear if you thin out the row. Another fact is that the classes overlap strongly. I order to classify the labels with the least error, the classifier will have to create a very complex model, with a lot of dividing boundaries. We can say that the original dataset is just garbage, and the rule states _Garbage in — Garbage out (GIGO)_. To avoid the GIGO philosophy and to make the research more meaningful, I suggest improving the representation of original data for a machine learning model (for example, CatBoost)

### Ideal feature space

In order to effectively divide the feature space into two classes, we can implement clustering for example, using the K-means method. This will give an idea of how the feature space could be ideally divided.

The source dataset is clustered into two clusters; five main components are displayed:

```
# perform K-means clustering over dataset
from sklearn.cluster import KMeans
pr = get_prices(look_back=LOOK_BACK)
X = pr[pr.columns[1:]]
kmeans = KMeans(n_clusters=2).fit(X)
y_kmeans = kmeans.predict(X)
pr['labels'] = y_kmeans
pca_plot(pr)
```

![](https://c.mql5.com/2/41/Figure_1__13.png)

The feature space looks ideal, but class labels (0, 1) obviously do not correspond to profitable trading. This example only illustrates a more preferred feature space than the GIGO dataset. That is why we need to create a compromise between ideal and garbage data. This is what we are going to do next.

### Generative model for resampling training examples

“What I cannot create, I do not understand.”

—Richard Feynman

In this section, we will consider a model that learns to "understand" data and to recreate new ones.

The k-means clustering method is relatively simple and easy to understand. However, it has a number of disadvantages and is not suitable for our case. In particular, it has poor performance in many real-world cases because it is not probabilistic. Imagine that this method places circles (or hyperspheres) around a given number of centroids with a radius which is determined by the outermost point of the cluster. This radius strictly limits the set of points for each cluster. Thus, all clusters can only be described by circles and hyperspheres, while real clusters do not always satisfy this criterion (as they can be oblong or in the form of ellipses). This will cause overlapping of different cluster values.

A more advanced algorithm is the Gaussian Mixture Model. This model searches for a mixture of multivariate Gaussian probability distributions that best models the dataset. Since the model is probabilistic, this outputs the probabilities of an example being categorized as a particular cluster. In addition, each cluster is associated not with a strictly defined sphere, but with a smooth Gaussian model, which can be represented not only as circles, but also as ellipses which are arbitrarily oriented in space.

![](https://c.mql5.com/2/41/vbh_lzlwd79i.png)

Different kinds of probabilistic models, depending on covaiance\_type

Below is a comparison of clusters obtained by k-means and GMM ( [source](https://www.mql5.com/go?link=https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html "https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html")):

![](https://c.mql5.com/2/41/7su_f52gpx13_v2o.png)

K-means clustering

![](https://c.mql5.com/2/41/t3s_po9fiqf6_11k.png)

GMM clustering

In fact, the Gaussian Mixture Model (GMM) algorithm is not really a clusterizer, because its main task is to estimate the probability density. Clusters in this model are represented as data generated from probability distributions describing this data. Thus, after estimating the probability density of each cluster, new datasets can be generated from these distributions. These sets will be similar to the original data, but they have more or less variability and will have less outliers. Moreover, datasets in many cases will be less correlated. We can obtain random examples and then train the CatBoost classifier using these examples.

### Pipeline for iterative resampling of the original dataset and CatBoost model training

Firstly, it is necessary to cluster the source data, including class labels:

```
# perform GMM clustering over dataset
from sklearn import mixture
pr_c = pr.copy()
X = pr_c[pr_c.columns[1:]]
gmm = mixture.GaussianMixture(n_components=75, covariance_type='full').fit(X)
```

The main parameter which can be selected is _n\_components_. It was empirically set to 75 (clusters). Other parameters are not so important and are not considered here. After the model is trained, we can generate some artificial samples from the multivariate distribution of the GMM model and visualize several main components:

```
# plot resampled components
generated = gmm.sample(5000)
gen = pd.DataFrame(generated[0])
gen.rename(columns={ gen.columns[-1]: "labels" }, inplace = True)
gen.loc[gen['labels'] >= 0.5, 'labels'] = 1
gen.loc[gen['labels'] < 0.5, 'labels'] = 0
pca_plot(gen)
```

Please note that labels have also been clustered, and thus they no longer represent a binary series. The labels are again converted to values (0;1) in the above code. Now, the resulting feature space can be displayed, using the _pca\_plot()_ function:

![](https://c.mql5.com/2/41/Figure_1__19.png)

If you compare this diagram with the earlier presented GIGO dataset diagram, you can see that it does not have data loops. Features and labels have become less correlated, which should have a positive effect on the learning result. At the same time, labels sometimes tend to form denser clusters and the model may turn out to be simpler, with fewer dividing boundaries. We have partly achieved the desired effect in eliminating problems with garbage data. Nevertheless, the data is essentially the same. We have simply resampled the original data.

Provided that GMM generates samples randomly, this leads to data pluralism. The best model can be selected using brute force. A special brute force function has been written for this purpose:

```
# brute force loop
def brute_force(samples = 5000):
    # sample new dataset
    generated = gmm.sample(samples)
    # make labels
    gen = pd.DataFrame(generated[0])
    gen.rename(columns={ gen.columns[-1]: "labels" }, inplace = True)
    gen.loc[gen['labels'] >= 0.5, 'labels'] = 1
    gen.loc[gen['labels'] < 0.5, 'labels'] = 0
    X = gen[gen.columns[:-1]]
    y = gen[gen.columns[-1]]
    # train\test split
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size = 0.5, test_size = 0.5, shuffle=True)
    #learn with train and validation subsets
    model = CatBoostClassifier(iterations=500,
                            depth=6,
                            learning_rate=0.1,
                            custom_loss=['Accuracy'],
                            eval_metric='Accuracy',
                            verbose=False,
                            use_best_model=True,
                            task_type='CPU')
    model.fit(train_X, train_y, eval_set = (test_X, test_y), early_stopping_rounds=25, plot=False)
    # test on new data
    pr_tst = get_prices(TSTART_DATE, START_DATE)
    X = pr_tst[pr_tst.columns[1:]]
    X.columns = [''] * len(X.columns)

    #test the learned model
    p = model.predict_proba(X)
    p2 = [x[0]<0.5 for x in p]
    pr2 = pr_tst.iloc[:len(p2)].copy()
    pr2['labels'] = p2
    R2 = tester(pr2, MARKUP, plot=False)

    return [R2, samples, model]
```

I have highlighted the main points in the code. First, it generated n random examples from the GMM distribution. Then the CatBoost model is trained using this data. The function returns the R^2 score calculated in the tester. Pay attention that the model is tested not only using the training period data, but it also uses earlier data. For example, the model was trained on data since early 2020, and it was tested using data since early 2015. You can change the date ranges as you like.

Let us write a loop that will call the specified function several times and will save each pass results to a list:

```
res = []
for i in range(50):
    res.append(brute_force(10000))
    print('Iteration: ', i, 'R^2: ', res[-1][0])

res.sort()
test_model(res[-1])
```

Then the list is sorted and the model at the end of the list has the best R^2 score. Let us display the best result:

![](https://c.mql5.com/2/41/Figure_1__20.png)

The last (right) part of the graph (about 1000 deals) is a training dataset, from the beginning of 2020, while the rest uses new data that was not used in model training. Since the models are sorted in ascending order, according to the R^2 metric, we can test previous models with a lower score:

```
test_model(res[-2])
```

![](https://c.mql5.com/2/41/Figure_1__21.png)

You can also look at the R^2 score itself:

```
>>> res[-2][0]
0.9576444017048906
```

As you can see, now the model is tested on a long five-year period, although it was trained on a one-year period. Then, the model can be exported to MQH format. The CatBoost model object is located in the nested list, with index 2 - the first dimension contains model numbers. Here we export the model with the index \[-2\] (the second from the end of the sorted list):

```
# export best model to mql
export_model_to_MQL_code(res[-2][2])
```

After export, the model can be tested in the standard MetaTrader 5 Strategy Tester. Since the spread in the custom tester was less than the real one, the curves are slightly different. Nevertheless, their general shape is the same.

![](https://c.mql5.com/2/41/0ig9ig_ipd6i0_2020-11-23_172200.png)

### How can the models be improved?

Model training implies many random components which are every time different. For example, random sampling of deals, GMM training (which also has an element of randomness), random sampling from the posterior GMM distribution and CatBoost training which also contains an element of randomness. Therefore, the entire program can be restarted several times to get the best result. If a stable model cannot be obtained, you should adjust the LOOK\_BACK parameter and the number of moving averages and their periods. You can also change the number of samples received from GMM, as well as the training and testing intervals.

### Change log and code refactoring

Some changes have been made to the Python code of the program. They require some clarification.

Now, a list of moving averages with different averaging periods can be set. A combination of several MAs usually has a positive effect on training results.

```
MA_PERIODS = [15, 55, 150, 250]
```

Added configurable start date for testing process, model evaluation and selection.

```
TSTART_DATE = datetime(2015, 1, 1)
```

The random sampling function has undergone a number of changes. Added the add\_noize parameter, which allows you to add noise to the original dataset. This will make trading less ideal by adding drawdowns and mixing deals. Sometimes, a model can be improved on new data by introducing an error at the level of 0.1 - 02.

Now the spread is taken into account. The trades which do not cover the spread are marked with a label of 2.0, and are then deleted from the dataset due to being uninformative.

```
def add_labels(dataset, min, max, add_noize = 0.1):
    labels = []
    for i in range(dataset.shape[0]-max):
        rand = random.randint(min, max)
        curr_pr = dataset['close'][i]
        future_pr = dataset['close'][i + rand]

        if future_pr + MARKUP < curr_pr:
            labels.append(1.0)
        elif future_pr - MARKUP > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2].index).reset_index(drop=True)

    if add_noize==0:
        return dataset

    # add noize to samples
    noize_b = dataset[dataset.labels == 0]['labels'].sample(frac = add_noize)
    noize_s = dataset[dataset.labels == 1]['labels'].sample(frac = add_noize)
    noize_b = noize_b+1
    noize_s = noize_s-1
    dataset.update(noize_b)
    dataset.update(noize_s)
    return dataset
```

The test function now returns the R^2 score:

```
def tester(dataset, markup = 0.0, plot = False):
    last_deal = int(2)
    last_price = 0.0
    report = [0.0]
    for i in range(dataset.shape[0]):
        pred = dataset['labels'][i]
        if last_deal == 2:
            last_price = dataset['close'][i]
            last_deal = 0 if pred <= 0.5 else 1
            continue
        if last_deal == 0 and pred > 0.5:
            last_deal = 1
            report.append(report[-1] - markup + (dataset['close'][i] - last_price))
            last_price = dataset['close'][i]
            continue
        if last_deal == 1 and pred < 0.5:
            last_deal = 0
            report.append(report[-1] - markup + (last_price - dataset['close'][i]))
            last_price = dataset['close'][i]

    y = np.array(report).reshape(-1,1)
    X = np.arange(len(report)).reshape(-1,1)
    lr = LinearRegression()
    lr.fit(X,y)

    l = lr.coef_
    if l >= 0:
        l = 1
    else:
        l = -1

    if(plot):
        plt.plot(report)
        plt.show()

    return lr.score(X,y) * l
```

Added a helper function for data visualization through the main component method. This may assist in better understanding your data.

```
def pca_plot(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 5)
    components = pd.DataFrame(pca.fit_transform(data[data.columns[1:-1]]))
    components['labels'] = data['labels'].reset_index(drop = True)
    import seaborn as sns
    g = sns.PairGrid(components, hue="labels", height=1.2)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    plt.show()
```

The code parser has been extended. Now it takes into account all periods of moving averages, which are added to the MQL program, after which the fill\_arrays function forms a feature vector.

```
def export_model_to_MQL_code(model):
    model.save_model('catmodel.h',
           format="cpp",
           export_parameters=None,
           pool=None)

    # add variables
    code = 'int ' + 'loock_back = ' + str(LOOK_BACK) + ';\n'
    code += 'int hnd[];\n'
    code += 'int OnInit() {\n'
    code +=     'ArrayResize(hnd,' + str(len(MA_PERIODS)) + ');\n'

    count = len(MA_PERIODS) - 1
    for i in MA_PERIODS:
        code +=     'hnd[' + str(count) + ']' + ' =' + ' iMA(NULL,PERIOD_CURRENT,' + str(i) + ',0,MODE_SMA,PRICE_CLOSE);\n'
        count -= 1

    code += 'return(INIT_SUCCEEDED);\n'
    code += '}\n\n'

    # get features
    code += 'void fill_arays(int look_back, double &features[]) {\n'
    code += '   double ma[], pr[], ret[];\n'
    code += '   ArrayResize(ret,' + str(LOOK_BACK) +');\n'
    code += '   CopyClose(NULL,PERIOD_CURRENT,1,look_back,pr);\n'
    code += '   for(int i=0;i<' + str(len(MA_PERIODS)) +';i++) {\n'
    code += '       CopyBuffer(hnd[' + 'i' + '], 0, 1, look_back, ma);\n'
    code += '       for(int f=0;f<' + str(LOOK_BACK) +';f++)\n'
    code += '           ret[f] = pr[f] - ma[f];\n'
    code += '       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }\n'
    code += '   ArraySetAsSeries(features, true);\n'
    code += '}\n\n'
```

### Conclusion

This article demonstrated an example of how to use a simple generative model - GMM (Gaussian Mixture Model) for resampling the original dataset. This model allows improving the performance of the CatBoost classifier on new data, by improving the characteristics of the feature space. For selecting the best model, we have implemented an iterative data resampling, with the possibility to select the desired result.

This was a kind of breakthrough from naive models to meaningful ones. By spending a minimum of effort for developing a logical component of a trading strategy, you can get interesting machine learning-based trading robots.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8662](https://www.mql5.com/ru/articles/8662)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8662.zip "Download all attachments in the single ZIP archive")

[cat\_trader.mq5](https://www.mql5.com/en/articles/download/8662/cat_trader.mq5 "Download cat_trader.mq5")(4.38 KB)

[cat\_model.mqh](https://www.mql5.com/en/articles/download/8662/cat_model.mqh "Download cat_model.mqh")(157.06 KB)

[clustering\_catboost.py](https://www.mql5.com/en/articles/download/8662/clustering_catboost.py "Download clustering_catboost.py")(9.45 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Creating a mean-reversion strategy based on machine learning](https://www.mql5.com/en/articles/16457)
- [Fast trading strategy tester in Python using Numba](https://www.mql5.com/en/articles/14895)
- [Time series clustering in causal inference](https://www.mql5.com/en/articles/14548)
- [Propensity score in causal inference](https://www.mql5.com/en/articles/14360)
- [Causal inference in time series classification problems](https://www.mql5.com/en/articles/13957)
- [Cross-validation and basics of causal inference in CatBoost models, export to ONNX format](https://www.mql5.com/en/articles/11147)
- [Metamodels in machine learning and trading: Original timing of trading orders](https://www.mql5.com/en/articles/9138)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/357471)**
(150)


![Aleksei Kuznetsov](https://c.mql5.com/avatar/2013/10/52601B64-7C6E.jpg)

**[Aleksei Kuznetsov](https://www.mql5.com/en/users/elibrarius)**
\|
12 Jan 2021 at 15:41

Maxim, it would be nice to do a signal on the article, seems like good results.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
12 Jan 2021 at 15:43

**elibrarius:**

Maxim, it would be good to make a signal on the article, it seems to have good results.

There are more advanced methods already, in terms of data preparation, I am working with them.

Monitoring every article is not an option.

It's more for scientific and cognitive purposes.

![chengxiaoyu](https://c.mql5.com/avatar/avatar_na2.png)

**[chengxiaoyu](https://www.mql5.com/en/users/chengxiaoyu)**
\|
25 Jun 2021 at 07:32

when I change the train start and stop datetime, the model [backtests](https://www.mql5.com/en/articles/2612 "Article: Testing trading strategies on real ticks ") result is bad, what can I do to improve model performance?


![Yang Fan](https://c.mql5.com/avatar/2021/12/61C339B7-3305.jpg)

**[Yang Fan](https://www.mql5.com/en/users/yfclark)**
\|
22 Dec 2021 at 14:47

I spend a lot time,finally I get what you are doing.Because the label of the ml algorithm is  imbalance,you use a GaussMixtureModel to simulate the price born,then sampled from the model,then you can train a better ml algorithm


![Xiong Feng Shi](https://c.mql5.com/avatar/avatar_na2.png)

**[Xiong Feng Shi](https://www.mql5.com/en/users/zhen550218)**
\|
2 May 2023 at 22:34

In regards to the article, although I didn't read it, I thought it was very powerful. So I decided to take a moment to give some advice. Firstly, the data source in this market is only part of the chips in the market, or a small portion of the chips, it is the majority of the chips that traders have on hand that can determine the direction of the market, so it is difficult to achieve the results we expect from the data collection, relying on what methodology and approach to optimise what may be just a fit to the past market. Secondly, this market for a short period of time it is not random, as an example, when there are only 2 multi-side traders and 2 short-side traders, one short-side N price listed for sale, the other short-side N-1 price for sale. A multi party N-1 buy, the [current price of](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_double "MQL5 documentation: Position Properties") N, assuming that another multi party N price buy, theoretically on the price should be N, in fact, the empty N no single, the aggregation mechanism will go to N-1 to find a deal, so the current price of N-1, is probably so mean. So N and N-1, N+1, etc. are all related and not completely random, so data optimisation can be better from momentum. Finally, whether it is EA or manual trading, it is difficult to make money from the market steadily, because if it is stable then the wealth will inevitably be transferred to a certain market participant, and this market will cease to exist. So investment is to invest in risk, harvest risk process, too concerned about the stability of the possible losses, I do not object to some people in the market transactions found a certain law, the equivalent of the market BUG realised wealth only, in fact, the market itself is also in the process of self-improvement because of the complexity of the participants, but the gold is not red, no one is perfect. The direction of intelligent trading is theoretically the process of constantly looking for market bugs, this BUG is only a small number of people to use, with more people on the failure. I hope my comment can be a reference for you. vx tiger54088 pass by


![Basic math behind Forex trading](https://c.mql5.com/2/40/56.png)[Basic math behind Forex trading](https://www.mql5.com/en/articles/8274)

The article aims to describe the main features of Forex trading as simply and quickly as possible, as well as share some basic ideas with beginners. It also attempts to answer the most tantalizing questions in the trading community along with showcasing the development of a simple indicator.

![Timeseries in DoEasy library (part 51): Composite multi-period multi-symbol standard indicators](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library__3.png)[Timeseries in DoEasy library (part 51): Composite multi-period multi-symbol standard indicators](https://www.mql5.com/en/articles/8354)

In the article, complete development of objects of multi-period multi-symbol standard indicators. Using Ichimoku Kinko Hyo standard indicator example, analyze creation of compound custom indicators which have auxiliary drawn buffers for displaying data on the chart.

![Continuous walk-forward optimization (Part 8): Program improvements and fixes](https://c.mql5.com/2/40/MQL5-avatar-continuous_optimization__4.png)[Continuous walk-forward optimization (Part 8): Program improvements and fixes](https://www.mql5.com/en/articles/7891)

The program has been modified based on comments and requests from users and readers of this article series. This article contains a new version of the auto optimizer. This version implements requested features and provides other improvements, which I found when working with the program.

![A scientific approach to the development of trading algorithms](https://c.mql5.com/2/40/algorithm_2.png)[A scientific approach to the development of trading algorithms](https://www.mql5.com/en/articles/8231)

The article considers the methodology for developing trading algorithms, in which a consistent scientific approach is used to analyze possible price patterns and to build trading algorithms based on these patterns. Development ideals are demonstrated using examples.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lebaxcwpgjlpncpqoyuzvwuntbzmrrxy&ssn=1769251894530573288&ssn_dr=0&ssn_sr=0&fv_date=1769251894&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8662&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Advanced%20resampling%20and%20selection%20of%20CatBoost%20models%20by%20brute-force%20method%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925189445086740&fz_uniq=5083163257671259702&sv=2552)

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