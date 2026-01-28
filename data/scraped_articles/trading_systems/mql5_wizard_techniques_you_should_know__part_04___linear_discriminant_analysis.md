---
title: MQL5 Wizard techniques you should know (Part 04): Linear Discriminant Analysis
url: https://www.mql5.com/en/articles/11687
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T19:28:29.377560
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/11687&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070314562357498795)

MetaTrader 5 / Trading systems


[Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis "https://en.wikipedia.org/wiki/Linear_discriminant_analysis") (LDA) is a very common dimensionality reduction technique for classification problems. Like kohonen maps in prior article if you have high-dimensional data (i.e. with a large number of attributes or _variables_) from which you wish to _classify_ observations, LDA will help you transform your data so as to make the classes as distinct as possible. More rigorously, LDA will find the linear projection of your data into a lower-dimensional subspace that optimizes some measure of class separation. The dimension of this subspace is never more than the number of classes. For this article we will look at how LDA can be used as a signal, trailing indicator and money management tool. But first let’s look at an intrepid definition then work our way to its applications.

LDA is very much like the techniques PCA, QDA, & ANOVA; and the fact that they are all usually abbreviated is not very helpful. This article isn’t going to introduce or explain these various techniques, but simply highlight their differences.

_1) Principal components analysis (PCA):_

LDA is very similar toPCA: in fact, some have asked whether or not it would make sense to perform PCA followed by LDA regularisation ( to avoid curve fitting). That is a lengthy topic which perhaps should be an article for another day.

For this article though, crucial difference between the two dimensionality reduction methods is PCA tries to find the axes with _maximum variance_ for the whole data set with the assumption being the more dispersed the data the more the separability, whereas LDA tries to find the axes that actually set the data apart based on  classification.

![lda](https://c.mql5.com/2/50/lda_separation.png)

So from the illustration above, it’s not hard to see that PCA would give us LD2, whereas LDA would give us LD1. This makes the main difference (and therefore LDA preference) between PCA and LDA painfully obvious: just because a feature has a high variance (dispersion), doesn’t mean it will be useful in making predictions for the classes.

_2) Quadratic discriminant analysis (QDA):_

QDA is a generalization of LDA as a classifer. LDA assumes that the class conditional distributions are Gaussian with the same covariance matrix, if we want it to do any classification for us.

QDA doesn’t make this [homoskedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity "https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity") assumption, and attempts to estimate the covariance of all classes. While this might seem like a more robust algorithm (given fewer assumptions), it also implies there is a much larger number of parameters to estimate. It is well established that the number of parameters grows quadratically with the number of classes! So unless you can guarantee that your covariance estimates are reliable, you might not want to use QDA.

After all of this, there might be some confusion about the relationship between LDA, QDA, such as what’s better suited for dimensionality reduction, and what is better at classification, etc. This [CrossValidated](https://www.mql5.com/go?link=https://stats.stackexchange.com/questions/71489/three-versions-of-discriminant-analysis-differences-and-how-to-use-them/71571%2371571 "https://stats.stackexchange.com/questions/71489/three-versions-of-discriminant-analysis-differences-and-how-to-use-them/71571#71571") postand everything that it links to, could help.

_3) Analysis of variance (ANOVA):_

LDA andANOVAseem to have similar aims: both try to break-down an observed variable into several independent/dependent variables. However, the instrument used by ANOVA as per Wikipedia, is the   mirrored version of what LDA uses:

> "LDA is closely related to analysis of variance (ANOVA) and regression analysis, which also attempt to express one dependent variable as a linear combination of other features or measurements. However, ANOVA uses **categorical** independent variables and a **continuous** dependent variable, whereas discriminant analysis has **continuous** independent variables and a **categorical** dependent variable (i.e. the class label)."

LDA is typically defined as follows.

Let:

- n be the number of classes
- μ  be the mean of all observations
- N i be the number of observations in the i th class
- μ i be the mean of the i th class
- Σ i be thescatter matrixof the i th class

Now, define SW to be the _within-class scatter matrix_, given by

> SW = ∑ i = 1 n Σ i

and define SB to be the _between-class scatter matrix_, given by

> SB = ∑ i = 1 n N i ( μ i − μ ) ( μ i − μ ) T

Diagonalize SW − 1 SB to get its eigenvalues and eigenvectors.

Pick the k largest eigenvalues, and their associated eigenvectors. We will project our observations onto the subspace spanned by these vectors.

Concretely, what this means is that we form the matrix A , whose columns are the k eigenvectors chosen above.The CLDA class in the alglib library does exactly this and sorts the vectors based on their eigen values in descending order meaning we only need to pick the best predictor vector to make a forecast.

Like in previous articles we will use the MQL code library in implementing LDA for our expert advisor. Specifically, we will rely on the ‘CLDA’ class in the ‘dataanalysis.mqh’ file.

We will explore LDA for the forex pair USDJPY over the this year 2022 on the daily timeframe. The choice of input data for our expert is largely up to the user. In our case for this LDA the input data has a variable and class component. We need to prepare this data before running tests on it. Since we’ll be dealing with close prices, it will be ‘continuized’ by default (in its raw state). We’ll apply normalization and discretization to the variable and class components of our data. Normalization means all data is between a set minimum and maximum while discretization implies data is converted to Boolean (true or false). Below are the preparations we’ll have for 5 sets of data for our signal: -

1. Discretized variables data tracking close price changes to match class categories.
2. Normalized variables data of raw close price changes to the range -1.0 to +1.0.
3. Continuized variables data in raw close price changes.
4. Raw close prices.

Normalization will provide the change in close price as a proportion of the last 2 bar range in decimal (from -1.0 to +1.0), while Discretization will state whether the price rose (giving an index of 2) or remained in a neutral range (meaning index is 1) or declined (implying index of 0). We will test all data types to examine performance. This preparation is done by the 'Data' method shown below. All 4 data types are regularised with the 'm\_signal\_regulizer' input to define a neutral zone for our data and thus reduce white noise.

```
//+------------------------------------------------------------------+
//| Data Set method                                                  |
//| INPUT PARAMETERS                                                 |
//|     Index   -   int, read index within price buffer.             |
//|                                                                  |
//|     Variables                                                    |
//|             -   whether data component is variables or .         |
//|                  classifier.                                     |
//| OUTPUT                                                           |
//|     double  -   Data depending on data set type                  |
//|                                                                  |
//| DATA SET TYPES                                                   |
//| 1. Discretized variables. - 0                                    |
//|                                                                  |
//| 2. Normalized variables. - 1                                     |
//|                                                                  |
//| 3. Continuized variables. - 2                                    |
//|                                                                  |
//| 4. Raw data variables. - 3                                       |
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalDA::Data(int Index,bool Variables=true)
   {
      m_close.Refresh(-1);

      m_low.Refresh(-1);
      m_high.Refresh(-1);

      if(Variables)
      {
         if(m_signal_type==0)
         {
            return(fabs(Close(StartIndex()+Index)-Close(StartIndex()+Index+1))<m_signal_regulizer*Range(Index)?1.0:((Close(StartIndex()+Index)>Close(StartIndex()+Index+1))?2.0:0.0));
         }
         else if(m_signal_type==1)
         {
            if(fabs(Close(StartIndex()+Index)-Close(StartIndex()+Index+1))<m_signal_regulizer*Range(Index))
            {
               return(0.0);
            }
            return((Close(StartIndex()+Index)-Close(StartIndex()+Index+1))/fmax(m_symbol.Point(),fmax(High(StartIndex()+Index),High(StartIndex()+Index+1))-fmin(Low(StartIndex()+Index),Low(StartIndex()+Index+1))));
         }
         else if(m_signal_type==2)
         {
            if(fabs(Close(StartIndex()+Index)-Close(StartIndex()+Index+1))<m_signal_regulizer*Range(Index))
            {
               return(0.0);
            }
            return(Close(StartIndex()+Index)-Close(StartIndex()+Index+1));
         }
         else if(m_signal_type==3)
         {
            if(fabs(Close(StartIndex()+Index)-Close(StartIndex()+Index+1))<m_signal_regulizer*Range(Index))
            {
               return(Close(StartIndex()+Index+1));
            }
            return(Close(StartIndex()+Index));
         }
      }

      return(fabs(Close(StartIndex()+Index)-Close(StartIndex()+Index+1))<m_signal_regulizer*Range(Index)?1.0:((Close(StartIndex()+Index)>Close(StartIndex()+Index+1))?2.0:0.0));
   }
```

We are using a dimensionality of four meaning each indicator value will provide 4 variables. So, for brevity in our case we will look at the last four indicator values for each data set in training. Our classification will also be basic taking on just the two classes (the minimum) in the class component of each data set. We will also need to set the number of data points in our training sample. This value is stored by input parameter ‘m\_signal\_points’.

LDA’s output is typically a matrix of coefficients. These coefficients are sorted in vectors and a dot product of any one of those vectors with the current indicator data point should yield a value that is then compared to similar values yielded by products of the training data set in order to classify this new/ current data point. So, for simplicity, if our training set had only 2 data points with LDA projections 0 and 1 and our new value yields a dot product of 0.9, we would conclude it is in the same category with the data point whose LDA projection was 1 since it closer to it. If on the other hand it yielded a value of say 0.1 then we would be of the opinion this new data point must belong to the same category as the data point whose LDA projection was 0.

Training datasets are seldom only two data points therefore, in practice we would take the ‘centroid’ of each class as a comparison to the output of the new data point’s dot product to LDA’s output vector. This ‘centroid’ would be the LDA projection mean of each class.

To classify each data point as bullish or bearish, we’ll simply look at the close price change _after_ the indicator value. If it is positive that data point is bullish, if negative it is bearish. Note it could be flat. For simplicity we’ll take flats or no price changes as also bullish as well.

The ‘ExpertSignal’ class typically relies on normalized integer values (0-100) to weight long and short decisions. Since LDA projections are bound to be double type we will normalize them as shown below to fall in the range of -1.0 to +1.0 (negative for bearish and positive for bullish).

```
         // best eigen vector is the first
         for(int v=0;v<__S_VARS;v++){ _unknown_centroid+= (_w[0][v]*_z[0][v]); }

         //


         if(fabs(_centroids[__S_BULLISH]-_unknown_centroid)<fabs(_centroids[__S_BEARISH]-_unknown_centroid) && fabs(_centroids[__S_BULLISH]-_unknown_centroid)<fabs(_centroids[__S_WHIPSAW]-_unknown_centroid))
         {
            _da=(1.0-(fabs(_centroids[__S_BULLISH]-_unknown_centroid)/(fabs(_centroids[__S_BULLISH]-_unknown_centroid)+fabs(_centroids[__S_WHIPSAW]-_unknown_centroid)+fabs(_centroids[__S_BEARISH]-_unknown_centroid))));
         }
         else if(fabs(_centroids[__S_BEARISH]-_unknown_centroid)<fabs(_centroids[__S_BULLISH]-_unknown_centroid) && fabs(_centroids[__S_BEARISH]-_unknown_centroid)<fabs(_centroids[__S_WHIPSAW]-_unknown_centroid))
         {
            _da=-1.0*(1.0-(fabs(_centroids[__S_BEARISH]-_unknown_centroid)/(fabs(_centroids[__S_BULLISH]-_unknown_centroid)+fabs(_centroids[__S_WHIPSAW]-_unknown_centroid)+fabs(_centroids[__S_BEARISH]-_unknown_centroid))));
         }
```

This value then is easily normalized to the typical integer (0-100) that is expected by the signal class.

```
   if(_da>0.0)
     {
      result=int(round(100.0*_da));
     }
```

for the Check long function and,

```
   if(_da<0.0)
     {
      result=int(round(-100.0*_da));
     }
```

for the check short.

So, a test run for each of the input data types gives the strategy tester reports below.

**Data set 1 report**

[![sr1](https://c.mql5.com/2/50/sr1.png)](https://c.mql5.com/2/50/sr1.png "https://c.mql5.com/2/50/sr1.png")

![cs1](https://c.mql5.com/2/50/curve_signal_1__1.png)

**Data set 2 report**

[![sr2](https://c.mql5.com/2/50/sr2.png)](https://c.mql5.com/2/50/sr2.png "https://c.mql5.com/2/50/sr2.png")

![cs2](https://c.mql5.com/2/50/curve_signal_2__1.png)

**Data set 3 report**

[![sr3](https://c.mql5.com/2/50/sr3.png)](https://c.mql5.com/2/50/sr3.png "https://c.mql5.com/2/50/sr3.png")

![cs3](https://c.mql5.com/2/50/curve_signal_3__1.png)

**Data set 4 report**

[![sr4](https://c.mql5.com/2/50/sr4.png)](https://c.mql5.com/2/50/sr4.png "https://c.mql5.com/2/50/sr4.png")

![cs4](https://c.mql5.com/2/50/curve_signal_4__1.png)

These reports exhibit the potential of LDA as tool for a trader.

The ‘ExpertTrailing’ class adjusts or sets a stop loss for an open position. The key output here is a double for the new stop loss. So, depending on the open position we’ll consider High prices and Low prices as our primary data sets. These will be prepared as follows for both High prices and Low prices with the choice depending on the type of open position: -

1. Discretized variables data tracking (high or low) price changes to match class categories.
2. Normalized variables data ofraw (high or low) price changes to the range -1.0 to +1.0.
3. Continuized variables data in raw (high or low) price changes.
4. Raw (high or low) prices.

The output from the LDA will be a normalized double as with the signal class. Since this is not helpful in defining a stop loss it will be adjusted as shown below depending on the type of open position to come up with a stop loss price.

```
      int _index   =StartIndex();
      double _min_l=Low(_index),_max_l=Low(_index),_min_h=High(_index),_max_h=High(_index);

      for(int d=_index;d<m_trailing_points+_index;d++)
      {
         _min_l=fmin(_min_l,Low(d));
         _max_l=fmax(_max_l,Low(d));
         _min_h=fmin(_min_h,High(d));
         _max_h=fmax(_max_h,High(d));
      }

      if(Type==POSITION_TYPE_BUY)
      {
         _da*=(_max_l-_min_l);
         _da+=_min_l;
      }
      else if(Type==POSITION_TYPE_SELL)
      {
         _da*=(_max_h-_min_h);
         _da+=_max_h;
      }
```

Also here is how we adjust and set our new stop loss levels. For the long positions:

```
   m_long_sl=ProcessDA(StartIndex(),POSITION_TYPE_BUY);

   double level =NormalizeDouble(m_symbol.Bid()-m_symbol.StopsLevel()*m_symbol.Point(),m_symbol.Digits());
   double new_sl=NormalizeDouble(m_long_sl,m_symbol.Digits());
   double pos_sl=position.StopLoss();
   double base  =(pos_sl==0.0) ? position.PriceOpen() : pos_sl;
//---
   sl=EMPTY_VALUE;
   tp=EMPTY_VALUE;
   if(new_sl>base && new_sl<level)
      sl=new_sl;
```

What we're doing here is determining the likely low price point, until the next bar, for a long open position ('m\_long\_sl') and then setting it as our new stop loss if it is more than either the position's open price or its current stop loss while being below the bid price minus stops level. The data type used in calculating this is low prices.

The setting of stop loss for short positions is a mirrored version of this.

So, a test run for each of the input data type while using the … data type for signal gives the strategy tester reports below.

**Data set 1 report**

[![tr1](https://c.mql5.com/2/50/tr1.png)](https://c.mql5.com/2/50/tr1.png "https://c.mql5.com/2/50/tr1.png")

![ct1](https://c.mql5.com/2/50/curve_trailing_1__1.png)

**Data set 2 report**

[![tr2](https://c.mql5.com/2/50/tr2.png)](https://c.mql5.com/2/50/tr2.png "https://c.mql5.com/2/50/tr2.png")

![ct2](https://c.mql5.com/2/50/curve_trailing_2__1.png)

**Data set 3 report**

[![tr3](https://c.mql5.com/2/50/tr3.png)](https://c.mql5.com/2/50/tr3.png "https://c.mql5.com/2/50/tr3.png")

![ct3](https://c.mql5.com/2/50/curve_trailing_3__1.png)

Data set 4 report.

[![tr4](https://c.mql5.com/2/50/tr4.png)](https://c.mql5.com/2/50/tr4.png "https://c.mql5.com/2/50/tr4.png")

![ct4](https://c.mql5.com/2/50/curve_trailing_4__1.png)

These reports perhaps point to .data set of continued raw changes as best suited given its recovery factor of 6.82.

The ‘ExpertMoney’ class sets our position lot size. This can be a function of past performance which is why we’re building on the ‘OptimizedVolume’ class. However, LDA can help with initial sizing if we consider volatility or the range between High and Low prices. Our primary data set therefore will be price bar range. We’ll look to see if price bar range is increasing or decreasing. With that let’s have the following data preparations: -

1. Discretized variables data tracking range value changes to match class categories.
2. Normalized variables data ofrawrangevalue changes to the range -1.0 to +1.0.
3. Continuized variables data in rawrangevalue changes.
4. Rawrangevalues.

The output from the LDA will be a normalized double as with the signal and trailing class. Since once again this is not immediately helpful we'll make adjustments shown below to better project a new bar range.

```
      int _index   =StartIndex();
      double _min_l=Low(_index),_max_h=High(_index);

      for(int d=_index;d<m_money_points+_index;d++)
      {
         _min_l=fmin(_min_l,Low(d));
         _max_h=fmax(_max_h,High(d));
      }

      _da*=(_max_h-_min_l);
      _da+=(_max_h-_min_l);
```

The setting of open volume is handled by 2 mirrored functions depending on whether the expert is opening a long or short position. Below are highlights for a long position.

```
   double _da=ProcessDA(StartIndex());

   if(m_symbol==NULL)
      return(0.0);

   sl=m_symbol.Bid()-_da;

//--- select lot size
   double _da_1_lot_loss=(_da/m_symbol.TickSize())*m_symbol.TickValue();
   double lot=((m_percent/100.0)*m_account.FreeMargin())/_da_1_lot_loss;

//--- calculate margin requirements for 1 lot
   if(m_account.FreeMarginCheck(m_symbol.Name(),ORDER_TYPE_BUY,lot,m_symbol.Ask())<0.0)
     {
      printf(__FUNCSIG__" insufficient margin for sl lot! ");
      lot=m_account.MaxLotCheck(m_symbol.Name(),ORDER_TYPE_BUY,m_symbol.Ask(),m_percent);
     }

//--- return trading volume
   return(Optimize(lot));
```

What is noteworthy here is we determine projected change in range price and subtract this projection from our bid price (Should have subtracted stops level as well). This will give us a 'risk adjusted' stop loss from which if we use the percent input parameter as a maximum risk loss parameter, we can compute a lot size that will cap our drawdown percentage at the percent input parameter value should we experience a drawdown below the bid price which is as projected.

So, a test run for each of the input data type while using the raw close prices data type for signal and … for trailing gives the strategy tester reports below.

**Data set 1 report**

[![mr1](https://c.mql5.com/2/50/mr1.png)](https://c.mql5.com/2/50/mr1.png "https://c.mql5.com/2/50/mr1.png")

![cm1](https://c.mql5.com/2/50/curve_money_1__1.png)

**Data set 2 report**

[![mr2](https://c.mql5.com/2/50/mr2.png)](https://c.mql5.com/2/50/mr2.png "https://c.mql5.com/2/50/mr2.png")

![cm2](https://c.mql5.com/2/50/curve_money_2__1.png)

**Data set 3 report**

[![mr3](https://c.mql5.com/2/50/mr3.png)](https://c.mql5.com/2/50/mr3.png "https://c.mql5.com/2/50/mr3.png")

![cm3](https://c.mql5.com/2/50/curve_money_3__1.png)

**Data set 4 report**

[![mr4](https://c.mql5.com/2/50/mr4.png)](https://c.mql5.com/2/50/mr4.png "https://c.mql5.com/2/50/mr4.png")

![cm4](https://c.mql5.com/2/50/curve_money_4__1.png)

It appears data set of discrete range value changes is most promising for money management. Noteworthy as well is the huge variance in results for the data sets at money management considering they are all using the same signal and trailing settings.

This article has highlighted potential of discriminant analysis’ use as a trading tool in an expert advisor. It was not exhaustive. Further analysis could be undertaken with more diverse data sets that span longer periods.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11687.zip "Download all attachments in the single ZIP archive")

[TrailingDA.mqh](https://www.mql5.com/en/articles/download/11687/trailingda.mqh "Download TrailingDA.mqh")(16.66 KB)

[SignalDA.mqh](https://www.mql5.com/en/articles/download/11687/signalda.mqh "Download SignalDA.mqh")(13.48 KB)

[MoneyDA.mqh](https://www.mql5.com/en/articles/download/11687/moneyda.mqh "Download MoneyDA.mqh")(15.19 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/437725)**
(2)


![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
15 Dec 2022 at 11:40

Hi Stephan,

Great article and great content!. I have enjoyed studying your previous articles as well.

I am currently converting my mq4 EA to mq5 and would like to include this content into the conversion to enhance the signals,stoploss and [money management](https://www.mql5.com/en/articles/4162 "Article: Money Management by Vince. Implementation as a MQL5 Wizard Module").  As you did not include an EA, i it possibleto post one that could be used as a learning example for studying the application of the DA techniques?

I am looking forward to your next articles.

Cheers, CapeCoddah

![Livio Alves](https://c.mql5.com/avatar/2024/9/66f6e96c-8d25.png)

**[Livio Alves](https://www.mql5.com/en/users/livioalves)**
\|
10 Oct 2024 at 13:53

Nice one!!!

I add it to an Expert and got this message when compiling.

'operator\[\]' - constant variable cannot be passed as reference

1.

for(int v=0;v<\_\_S\_VARS;v++)

         {

            \_xy\[p\].Set(v, Data(Index+p+v+1));

2.

\_xy\[p\].Set(\_\_S\_VARS,Data(Index+p,false));

3.

for(int v=0;v<\_\_S\_VARS;v++)

         {

            \_z\[0\].Set(v,Data(Index));

![Adaptive indicators](https://c.mql5.com/2/50/adaptive_indicators_avatar.png)[Adaptive indicators](https://www.mql5.com/en/articles/11627)

In this article, I will consider several possible approaches to creating adaptive indicators. Adaptive indicators are distinguished by the presence of feedback between the values of the input and output signals. This feedback allows the indicator to independently adjust to the optimal processing of financial time series values.

![DoEasy. Controls (Part 23): Improving TabControl and SplitContainer WinForms objects](https://c.mql5.com/2/50/MQL5-avatar-doeasy-library-2.png)[DoEasy. Controls (Part 23): Improving TabControl and SplitContainer WinForms objects](https://www.mql5.com/en/articles/11634)

In this article, I will add new mouse events relative to the boundaries of the working areas of WinForms objects and fix some shortcomings in the functioning of the TabControl and SplitContainer controls.

![DoEasy. Controls (Part 24): Hint auxiliary WinForms object](https://c.mql5.com/2/50/MQL5-avatar-doeasy-library-2__1.png)[DoEasy. Controls (Part 24): Hint auxiliary WinForms object](https://www.mql5.com/en/articles/11661)

In this article, I will revise the logic of specifying the base and main objects for all WinForms library objects, develop a new Hint base object and several of its derived classes to indicate the possible direction of moving the separator.

![DoEasy. Controls (Part 22): SplitContainer. Changing the properties of the created object](https://c.mql5.com/2/49/MQL5-avatar-doeasy-library-2__4.png)[DoEasy. Controls (Part 22): SplitContainer. Changing the properties of the created object](https://www.mql5.com/en/articles/11601)

In the current article, I will implement the ability to change the properties and appearance of the newly created SplitContainer control.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/11687&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070314562357498795)

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