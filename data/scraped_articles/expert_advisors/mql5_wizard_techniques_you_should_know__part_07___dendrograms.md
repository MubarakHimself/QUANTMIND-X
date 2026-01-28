---
title: MQL5 Wizard Techniques you should know (Part 07): Dendrograms
url: https://www.mql5.com/en/articles/13630
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:25:23.491592
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/13630&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071821245474877201)

MetaTrader 5 / Expert Advisors


### **Introduction**

This article which is part of a series on using the MQL5 wizard looks at [dendrograms](https://en.wikipedia.org/wiki/Dendrogram "https://en.wikipedia.org/wiki/Dendrogram"). We have considered already a few ideas that can be useful to traders via the MQL5 wizard like: [Linear discriminant analysis](https://www.mql5.com/en/articles/11687), [Markov chains](https://www.mql5.com/en/articles/11930), [Fourier transform](https://www.mql5.com/en/articles/12599), and a few others, and this article aims to take this endeavor further of looking at ways of capitalizing on the extensive ALGLIB code as translated by MetaQuotes together with the use of the inbuilt MQL5 wizard, to proficiently test and develop new ideas.

Agglomerative [Hierarchical Classification](https://en.wikipedia.org/wiki/Hierarchical_clustering "https://en.wikipedia.org/wiki/Hierarchical_clustering") sounds like a mouthful but it’s actually quite simple. Put plainly it is a means of relating different parts of a dataset by first considering the basic individual [clusters](https://en.wikipedia.org/wiki/Cluster_analysis "https://en.wikipedia.org/wiki/Cluster_analysis") and then systematically grouping them a step at a time until the entire dataset can be viewed as a single sorted unit. The output of this process is a hierarchical diagram more commonly referred to as a Dendrogram.

![](https://c.mql5.com/2/59/1798127682777.png)

This article will focus on how these constituent clusters can be used in assessing and thus forecasting price bar range but unlike in the past where we did this to help in trailing stop adjustment, we will consider it here for money management or position sizing purposes. The style to be adopted for this article will assume the reader is relatively new to the MetaTrader platform and MQL5 programming language and as such we may dwell on some topics and areas that are jejune for more experienced traders.

The importance of accurate price range forecasts is largely subjective. This is because its significance mostly depends on the trader’s strategy and overall trade approach. When would it not matter much? Well this could be, in situations for example; if in your trade-setups firstly you engage minimal to no leverage, you have a definitive stop loss, and you tend to hold positions over extended periods which could run into say months, and you have fixed margin position sizing (or even a fixed lot approach). In this case price bar volatility can be put on a backburner while you focus on screening for entry and exit signals. If on the other hand you are an intra-day trader, or someone who employs a significant amount of leverage, or somebody who does not hold trade positions over weekends, or anybody with a mid to short horizon when it comes to exposure to the markets then price bar range is certainly something you should pay attention to. We look at how this can be of use in money management by creating a custom instance of the ‘ExpertMoney’ class but applications of this could spread beyond money management and even encompass risk, if you consider that the ability to understand and reasonably forecast price bar range could help in deciding when to add on to open positions and conversely when to reduce.

### **Volatility**

Price bar ranges (which is how we quantify volatility for this article) in the context of trading are the difference between the high and low of a traded symbol’s price within a set timeframe. So, if we take say the daily time frame, and in a day the price for a traded symbol rises as high as H and not above H and falls as low as L and again not below L then our range, for the purposes of this article, is:

H – L;

Paying attention to volatility is arguably significant because of what is often called [volatility clustering](https://en.wikipedia.org/wiki/Volatility_clustering "https://en.wikipedia.org/wiki/Volatility_clustering"). This is the phenomenon of high volatility periods tending to be followed by more volatility and conversely low volatility periods also being followed by lower volatility. The significance if this is subjective as highlighted above however for most traders (including all beginners in my opinion) knowing how to trade with leverage can be a plus in the long run, since as most traders are familiar that high bouts of volatility can stop-out accounts not because the entry signal was wrong but because there was too much volatility. And most would also appreciate that even if you had a decent stop loss on your position there are times when the stop loss price may not be available, take the swiss franc debacle of January 2015, in which case your position would get closed by the broker at the next best available price which is often worse than your stop loss. This is because only limit orders guarantee price, stop orders and stop losses do not.

So, price bar ranges besides providing an overall sense of the market environment, can help in guiding on entry and even exit price levels as well. Again, depending on your strategy, if for instance you’re long a particular symbol, then the extents of your price bar range outlook (what you’re forecasting) can easily determine or at least guide where you place your entry price and even take profit.

At the risk of sounding mundane, it may be helpful to highlight some of the very basic price candle types and also illustrating their respective ranges. The more prominent types are the [bearish](https://www.mql5.com/go?link=https://www.investopedia.com/terms/b/bearishbelthold.asp "https://www.investopedia.com/terms/b/bearishbelthold.asp"), [bullish](https://www.mql5.com/go?link=https://www.investopedia.com/articles/active-trading/062315/using-bullish-candlestick-patterns-buy-stocks.asp "https://www.investopedia.com/articles/active-trading/062315/using-bullish-candlestick-patterns-buy-stocks.asp"), [hammer](https://www.mql5.com/go?link=https://www.investopedia.com/terms/h/hammer.asp "https://www.investopedia.com/terms/h/hammer.asp"), [gravestone](https://www.mql5.com/go?link=https://www.investopedia.com/terms/g/gravestone-doji.asp "https://www.investopedia.com/terms/g/gravestone-doji.asp"), [long-legged](https://www.mql5.com/go?link=https://www.investopedia.com/terms/l/long-legged-doji.asp "https://www.investopedia.com/terms/l/long-legged-doji.asp"), and [dragonfly](https://www.mql5.com/go?link=https://www.investopedia.com/terms/d/dragonfly-doji.asp "https://www.investopedia.com/terms/d/dragonfly-doji.asp"). There are certainly more types but arguably these do cover what one would most likely encounter when faced with a price chart. In all these instances as shown in the diagrams below, the price bar range is simply the high price less the low price.

### **Agglomerative Hierarchical Classification**

Agglomerative Hierarchical Classification (AHC) is a method of classifying data into a preset number of clusters and then relating these clusters in a systematic hierarchical manner through what is called a [dendrogram](https://en.wikipedia.org/wiki/Dendrogram "https://en.wikipedia.org/wiki/Dendrogram"). The benefits of this mostly stem from the fact that the data being classified is often multi-dimensional and therefore the need to consider the many variables within a single data point is something which may not be too easy for one to grapple with when making comparisons. For example, a company looking to grade its customers based on the information they have from them could utilize this since this information is bound to be covering different aspects of the customers’ lives such as past spending habits, their age, gender, address, etc. AHC, by quantifying all these variables for each customer creates clusters from the apparent centroids of each data point. But more than that these clusters get grouped into a hierarchy for systematic relations so if a classification called for say 5 clusters then AHC would provide in a sorted format those 5 clusters meaning you can infer which clusters are more alike and which are more different. This cluster comparison though secondary can come in handy if you need to compare more than one data point and it turns out they are in separate clusters. The cluster ranking would inform how far apart the two points differ by using the separation magnitude between their respective clusters.

The classification by AHC is [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning "https://en.wikipedia.org/wiki/Unsupervised_learning") meaning it can be used in forecasting under different classifiers. In our case we are forecasting price bar ranges, someone else with the same trained clusters could use them to forecast close price changes, or another aspect pertinent to his trading. This allows more flexibility than if it was [supervised](https://en.wikipedia.org/wiki/Supervised_learning "https://en.wikipedia.org/wiki/Supervised_learning") and trained on a specific classifier because in that instance the model would only be used to make forecasts in what it was classified, implying forecasting for another purpose would require retraining a model with that new data set.

### **Tools and Libraries**

The MQL5 platform, with the help of its IDE, does allow as one would expect developing custom expert advisors from scratch and in order to showcase what is being shared here we could take that route hypothetically. However, that option would involve making a lot of decisions regarding the trading system that could have been taken differently by another trader when faced with implementing the same concept. And also, the code of such an implementation could be too customized and may be error- prone to be easily modified for different situations. That is why assembling our idea as part of other ‘standard’ Expert advisor classes provided by the MQL5 wizard is a compelling case. Not only do we have to do less debugging (there are occasional bugs even in MQL5’s in built classes but not a lot), but by saving it as an instance of one the standard classes, it can be used and combined with the wide variety of other classes in the MQL5 wizard to come up with different expert advisors which provides a more comprehensive test-bed.

MQL5 library code provides classes of AlgLib, which have been referenced in earlier articles in these series, and will be used again for this article. Specifically, within the ‘DATAANALYSIS.MQH’ file we will rely on ‘CClustering’ class and a couple of other related classes in coming up with an AHC classification for our price series data. Since our primary interest is price bar range it follows that our training data will consist of such ranges from previous periods. When using data training classes from the data analysis include file typically this data is placed in an ‘XY’ matrix with the X standing for the independent variables, and the Y representing the classifiers or the ‘labels’ to which the model is trained. Both are usually furnished in the same matrix.

### **Preparing Training Data**

For this article though, since we are having unsupervised training, our input data consists only of the X independent variables. These will be historical price bar ranges. At the same time though we would like to make forecasts by considering another stream of related data that is the eventual price bar range. This would be equivalent to the Y mentioned just above. In order to marry these two data sets while maintaining the unsupervised learning flexibility we can adopt the data struct below:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CMoneyAHC                  : public CExpertMoney
  {
protected:

   double                        m_decrease_factor;

   int                           m_clusters;                            // clusters
   int                           m_training_points;                     // training points
   int                           m_point_featues;                       // point featues

...

public:
                                 CMoneyAHC(void);
                                 ~CMoneyAHC(void);

   virtual bool                  ValidationSettings(void);
   //---
   virtual double                CheckOpenLong(double price,double sl);
   virtual double                CheckOpenShort(double price,double sl);
   //---
   void                          DecreaseFactor(double decrease_factor) { m_decrease_factor=decrease_factor;            }

   void                          Clusters(int value)                    { m_clusters=value;                             }
   void                          TrainingPoints(int value)              { m_training_points=value;                      }
   void                          PointFeatures(int value)               { m_point_featues=value;                        }

protected:

   double                        Optimize(double lots);


   double                        GetOutput();

   CClusterizerState             m_state;

   CAHCReport                    m_report;

   struct                        Sdata
                                 {
                                    CMatrixDouble  x;
                                    CRowDouble     y;

                                                   Sdata(){};
                                                   ~Sdata(){};
                                 };

   Sdata                         m_data;

   CClustering                   m_clustering;

   CRowInt                       m_clustering_index;
   CRowInt                       m_clustering_z;

  };
```

So, the historical price bar ranges will be collected as a fresh batch on each new bar. Experts generated by the MQL5 wizard tend to execute trade decisions on each new bar and for our testing purposes this is sufficient. There are indeed alternative approaches such as getting a large batch spanning many months or even years and then from testing examining how well the clusters of the model can separate eventual low volatility price bars from high volatility bars. Also keep in mind that we are using only 3 clusters where one extreme cluster would be for very volatile bars, one for very low volatility, and one for mid-range volatility. Again, one could investigate with 5 clusters for instance but the principle, for our purposes, would be the same. Place the clusters in order of most eventual volatility to least eventual volatility order and identify in which cluster our current data point lies.

### **Filling Data**

The code for getting the latest bar ranges on each new bar and populating our custom struct will be as follows:

```
      m_data.x.Resize(m_training_points,m_point_featues);
      m_data.y.Resize(m_training_points-1);

      m_high.Refresh(-1);
      m_low.Refresh(-1);

      for(int i=0;i<m_training_points;i++)
      {
         for(int ii=0;ii<m_point_featues;ii++)
         {
            m_data.x.Set(i,ii,m_high.GetData(StartIndex()+ii+i)-m_low.GetData(StartIndex()+ii+i));
         }
      }
```

The number of training points define how large our training data set is. This is a customizable input parameter and so the data point features parameter. This parameter though defines the number of ‘dimensions’ each data point has. So, in our case we have 4 by default but this simply means we are using the last 4 price bar ranges to define any given data point. It is akin to a vector.

### **Generating Clusters**

So, once we have data in our custom struct, the next step is to model it with the AlgLib AHC model generator. This model is referred to as a ‘state’ in the code listing and to that end our model is named ‘m\_state’. This is a two-step process. First, we have to generate model points based on the provided training data and then we run the AHC generator. Setting the points can be thought of as initializing the model and ensuring all key parameters are well defined. This is called as follows in our code:

```
m_clustering.ClusterizerSetPoints(m_state, m_data.x, m_training_points, m_point_featues, 20);
```

The second important step is running the model to define the clusters of each of the provided data points within the training data set. This is done through calling the ‘ClusterizerRunAHC’ function as listed below:

```
m_clustering.ClusterizerRunAHC(m_state, m_report);
```

From the AlgLib end this is the meat and potatoes of generating the clusters we need. This function does some brief pre-processing and then calls the protected (private) function ‘ClusterizerRunAHCInternal’ which does the heavy lifting. All this source is in your ‘include\\math\\AlgLib\\dataanalysis.mqh’ file visible from line: 22463. What could be noteworthy here is the generation of the dendrogram in the ‘cidx’ output array. This array cleverly condenses a lot of cluster information in a single array. Just prior to this a distance matrix will need to be generated for all the training data points by utilizing their centroids. The mapping of distance matrix values to cluster indices is captured by this array with the first values up to the total number of training points represent the cluster of each point with subsequent indices representing the merging of these clusters to form the dendrogram.

Equally noteworthy perhaps is the distance type used in generating the distance matrix. There are nine options available ranging from the Chebyshev distance, Euclidean distance and up to the spearman rank correlation. Each of these alternatives is assigned an index that we establish when we call the set points function mentioned above. The choice of distance type is bound to be very sensitive to the nature and type of clusters generated so attention should be paid to this. Using the Euclidean distance (whose index is 2) allows more flexibility on implementation when setting the AHC algorithm as Ward’s method can used unlike for other distance types.

### **Retrieving Clusters**

Retrieving the clusters is also as straightforward as generating them. We simply call one function ‘ClusterizerGetKClusters’ and it retrieves two arrays from the output report of the cluster generating function we called earlier (run AHC). The arrays are the cluster index array and cluster z arrays and they guide not just how the clusters are defined but also how the dendrogram can be formed from them. Calling this function is simply done as indicated below:

```
m_clustering.ClusterizerGetKClusters(m_report, m_clusters, m_clustering_index, m_clustering_z);
```

The structure of the resulting clusters is very simple since on our case we only classified our training data set to 3 clusters. This means we have no more than three levels of merging within the dendrogram. Had we used more clusters then our dendrogram would certainly have been more complex potentially having n-1 merging levels where n is the number of clusters used by the model.

### **Labeling Data Points**

The post-labelling of training data points to aid in forecasting is what then follows. We are not interested in simply classifying data sets but we want to put them to use and therefore our ‘labels’ will be the eventual price bar range after each training data point. We are fetching a new data set on each new bar that would include the current data point for which its eventual volatility is unknown. That is why when labelling, we skip the data point indexed 0 as shown on our code below:

```
      for(int i=0;i<m_training_points;i++)
      {
         if(i>0)//assign classifier only for data points for which eventual bar range is known
         {
            m_data.y.Set(i-1,m_high.GetData(StartIndex()+i-1)-m_low.GetData(StartIndex()+i-1));
         }
      }
```

Other implementations of this labelling process could of course be used. For example, rather than focusing on the price bar range of only the next bar we could have taken a macro-view by looking at the range of say the next 5, or 10 bars by having the overall range of these bars as our y value. This approach could lead to more ‘accurate’ and less erratic values and in fact the same outlook could be used if our labels are for price direction (change in close price) whereby we would try to forecast many more bars ahead rather than just one. In either way, as we skipped the first index because we did not have its eventual value, we would skip n bars (where n is the bars ahead we are looking to project). This long view approach would lead to considerable lag as n gets larger on the flip side though the large lags would allow a safe comparison with the projection because remember the lag is only one bar shy of the target y value.

### **Forecasting Volatility**

Once we have finished ‘labelling’ the trained data set, we can proceed to establish which cluster our current data point belongs among the clusters defined in the model. This is done by easily iterating through the output arrays of the modelling report and comparing the cluster index of the current data point to that of other training data points. If they match then they belong to the same cluster. Here is the simple listing of this:

```
      if(m_report.m_terminationtype==1)
      {
         int _clusters_by_index[];
         if(m_clustering_index.ToArray(_clusters_by_index))
         {
            int _output_count=0;
            for(int i=1;i<m_training_points;i++)
            {
               //get mean target bar range of matching cluster
               if(_clusters_by_index[0]==_clusters_by_index[i])
               {
                  _output+=(m_data.y[i-1]);
                  _output_count++;
               }
            }
            //
            if(_output_count>0){ _output/=_output_count; }
         }
      }
```

Once a match is found, we would then concurrently proceed to compute the average Y value of all training data points within that cluster. Getting the average could be considered crude but it is one way. Another could be finding the median, or may be the mode, whichever option is chosen the same principle of getting the Y value of our current point only from other data points within its cluster, applies.

### **Using Dendrograms**

What we have shown so far with source code shared is how the created **_individual_** clusters can be used to classify and make projections. What then is the role of a dendrogram? Why would quantifying by how much each cluster is different from each other be important? To answer this question, we could consider comparing two training data points as opposed to classifying just one as we’ve done. In this scenario, we could get a data point from history at a key inflection point as far as volatility is concerned (this could be a key fractal in price swings if you are forecasting price direction, but we are looking at volatility for this article). Since we’d have the clusters of both points then the distance between them would tell us how close our current data point is to the past inflection point.

### **Case Studies**

A few tests were run with a wizard assembled expert advisor that utilized a customized instance of the money management class. Our signal class was based on the library provided awesome oscillator, we run for the symbol EURUSD on the 4-hour time frame from 2022.10.01 to 2023.10.01 and this yielded a report as follows:

![r1](https://c.mql5.com/2/59/WZ_7_report.png)

As a control we also run tests with the same conditions as above except the money management used was the library provided fixed margin option and this gave us the following report:

![r2](https://c.mql5.com/2/59/WZ_7_report_ctrl.png)

Implications of our brief testing from these two reports is there is potential in adjusting our volume in accordance with the prevailing volatility of a symbol. The settings used for our expert advisor and the control are also shown below respectively.

![s1](https://c.mql5.com/2/59/WZ_7_settings.png)

And

![s2](https://c.mql5.com/2/59/WZ_7_settings_ctrl.png)

As is evident simillar settings were used mostly with only exception being in our expert advisor where we had to use more for the custom money management.

### **Conclusion**

To sum up, we have explored how agglomerative hierarchical classification with the help of its dendrogram can help identify and size up different sets of data, and how this classification can be used in making projections. More on this subject can be found here and as always, the ideas and source code shared are for testing ideas especially in settings when they are paired with different approaches. This is why the code format for MQL5 wizard classes is adopted.

### Notes on Attachments

The attached code is meant to be assembled with the MQL5 wizard as part of an assembly that includes a signal class file and a trailing class file. For this article the signal file was the awesome oscillator (SignalAO.mqh). More information on how to use the wizard can be found [here](https://www.mql5.com/en/articles/275).

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13630.zip "Download all attachments in the single ZIP archive")

[MoneyWZ\_7.mqh](https://www.mql5.com/en/articles/download/13630/moneywz_7.mqh "Download MoneyWZ_7.mqh")(11.68 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)
- [Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)
- [MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/456653)**
(1)


![Aleksey Vyazmikin](https://c.mql5.com/avatar/2024/6/6678986f-2caa.png)

**[Aleksey Vyazmikin](https://www.mql5.com/en/users/-aleks-)**
\|
23 Mar 2024 at 23:24

It is very good that you cover the functionality of the AlgLib library - it can be useful!

The code for visualising dendrograms is very lacking in the article.

![Developing an MQTT client for Metatrader 5: a TDD approach — Part 4](https://c.mql5.com/2/59/mechanism_in_MQTT_logo.png)[Developing an MQTT client for Metatrader 5: a TDD approach — Part 4](https://www.mql5.com/en/articles/13651)

This article is the fourth part of a series describing our development steps of a native MQL5 client for the MQTT protocol. In this part, we describe what MQTT v5.0 Properties are, their semantics, how we are reading some of them, and provide a brief example of how Properties can be used to extend the protocol.

![Neural networks made easy (Part 46): Goal-conditioned reinforcement learning (GCRL)](https://c.mql5.com/2/55/Neural_Networks_Part_46_avatar.png)[Neural networks made easy (Part 46): Goal-conditioned reinforcement learning (GCRL)](https://www.mql5.com/en/articles/12816)

In this article, we will have a look at yet another reinforcement learning approach. It is called goal-conditioned reinforcement learning (GCRL). In this approach, an agent is trained to achieve different goals in specific scenarios.

![Neural networks made easy (Part 47): Continuous action space](https://c.mql5.com/2/55/Neural_Networks_Part_47_avatar.png)[Neural networks made easy (Part 47): Continuous action space](https://www.mql5.com/en/articles/12853)

In this article, we expand the range of tasks of our agent. The training process will include some aspects of money and risk management, which are an integral part of any trading strategy.

![Neural networks made easy (Part 45): Training state exploration skills](https://c.mql5.com/2/55/Neural_Networks_Part_45_avatar.png)[Neural networks made easy (Part 45): Training state exploration skills](https://www.mql5.com/en/articles/12783)

Training useful skills without an explicit reward function is one of the main challenges in hierarchical reinforcement learning. Previously, we already got acquainted with two algorithms for solving this problem. But the question of the completeness of environmental research remains open. This article demonstrates a different approach to skill training, the use of which directly depends on the current state of the system.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/13630&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071821245474877201)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).