---
title: The case for using a Composite Data Set this Q4 in weighing SPDR XLY's next performance
url: https://www.mql5.com/en/articles/13775
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:49:56.579944
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=tjlkmsjjpxdvekjicngkolfkyojubarx&ssn=1769251795328039113&ssn_dr=0&ssn_sr=0&fv_date=1769251795&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13775&back_ref=https%3A%2F%2Fwww.google.com%2F&title=The%20case%20for%20using%20a%20Composite%20Data%20Set%20this%20Q4%20in%20weighing%20SPDR%20XLY%27s%20next%20performance%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925179565542948&fz_uniq=5083144127886923219&sv=2552)

MetaTrader 5 / Trading systems


**Introduction**

SPDR’s consumer discretionary ETF [XLY](https://www.mql5.com/go?link=https://www.ssga.com/us/en/intermediary/etfs/funds/the-consumer-discretionary-select-sector-spdr-fund-xly "https://www.ssga.com/us/en/intermediary/etfs/funds/the-consumer-discretionary-select-sector-spdr-fund-xly") was inducted on 22nd December 1998 and has grown to an AUM of slightly over U$16 billion as of November 2023. The bling ETF among SPDR’s 11, it provides investor’s exposure to specialty retail, hotels, luxury goods & apparel, automobiles and companies providing other non-essential expenditures that consumers may indulge in. The definition though could present some ambiguity such as whether [AMZN](https://www.mql5.com/go?link=https://www.marketwatch.com/investing/stock/amzn "https://www.marketwatch.com/investing/stock/amzn"), for instance, is really a discretionary goods seller or a staple goods seller or a tech company. The latter two are all covered by separate ETFs so investors seeking AMZN exposure can only get it from this sector ETF. In principle though the ETF is for companies that sell non-essential goods, that are purchased when consumers have some disposable income, and as a result this ETF is traditionally exposed to a lot of economic cycles.

MQL5 language with MetaTrader’s IDE are a hotbed for coding a developing not just indicators, and scripts to assist in manual trading, but also a place for developing autonomous trading systems as most readers would be well aware. Trading ETFs on the platform of course depends if your Broker offers them. Time series analysis and forecasting is a useful part of a trade system that we will focus on for this article. In doing so we will consider various time series data sets as candidates for a model to make projections for XLY’s performance. Taking this a step further, we’ll see if there is any gains/ benefit in using a composite data set that brings together the features (data columns) with the highest feature importance weightings into a single data set for running with our model.

**Data Collection**

The data set sources for the XLY ETF could include its own past performance metrics like price and volume but since in the [last article](https://www.mql5.com/en/articles/13715) on the XLV ETF these data sets had the least dispersion and thus least significant feature importance weightings we are going to consider alternative data sets. To that end we will consider data sets on economy, interest rates, employment rates, the consumer, retail, and demographics. Each of these has a different number of data columns (what we’ve referred to as features) and the assessment of each will be done by computing the [feature importance](https://en.wikipedia.org/wiki/Feature_selection "https://en.wikipedia.org/wiki/Feature_selection") weightings for each of their respective columns.

Obtaining and organizing time series data used to be a serious hassle since most of the data pertinent to any given ETF or security was in many places. Today though, with the dawn of AI, most of it can be sourced in one place and easily exported as an excel sheet. We have used bard to source all the data sets featured in this article and will look to use openAI in the future.

In the last article the data sets used did not have a uniform time frame which arguably could have biased the feature importance weightings as longer time frames tended to have more sparse weightings than smaller time frames. For this article therefore, we will try to keep all data sets in the same time frame of a quarter. This allows for better consistency not just among the data sets but also a seamless data sets review every quarter.

**Time Series Data Sets**

As mentioned above we are considering 6 possible data sets for modelling XLY performance. They will all have the same time frame which is quarterly. Let’s dive into each one.

Economic Indicators

The economic indicators data set has 4 features, all centered on US economic data, namely: GDP, personal income, international trade in goods, and international trade in services. There is a lot to choose from in this window for instance inflation would be another indicator, however these values have been bundled with employment data set that way no one data set gets too wide at the possible expense of the others.

![](https://c.mql5.com/2/61/4230613094550.png)

The significance of economic indicators to XLY stems from the analogy that disposable income tends to correlate positively with personal income and GDP. The United States, which is the primary source of statistics data, also tends to import more of its goods and some services which is why these are also important data sets to consider when assessing the outlook for non-essential expenditure. Possible key data sets that should have been added as economic indicators are as mentioned inflation plus interest rates. The former as stated is part of the employment data set while the latter is perhaps too important that it forms its own data set which we will look at next.

The dynamics in the recent past of economic indicators have painted an uncertain picture for US discretionary spending, going forward. Firstly, even though personal income is expected to continue growing recent rises in inflation are putting headwinds on this prognosis. In addition, US GDP is also expected to grow by 2%, but this is slower than the prior 3.5% print and similarly growth in imports of goods and services is expected to continue albeit at a tapered pace from recent quarters as questions linger on the resurgence of China’s economy. Consumer spending is expected to be the driver of US economic growth and since discretionary spending is a key part of it, the modeling of this relationship could be insightful in forecasting the outlook of XLY.

Interest Rates

The features for the interest rates data set are three namely the federal reserve’s effective benchmark rate, the yield of the 2yr treasury bills to maturity, and the 10yr yield to maturity. Within this data window there were other data sets to consider such as interest rates of other prominent central banks like China’s PBoC, or Germany’s Bundesbank, or Japan’s BOJ, or [YTMs](https://www.mql5.com/go?link=https://www.investopedia.com/terms/y/yieldtomaturity.asp "https://www.investopedia.com/terms/y/yieldtomaturity.asp") to alternative bonds not just those issued by the US but also those issued by these and other developed economy countries.

![](https://c.mql5.com/2/61/1234334396936.png)

We however focused on just these three data sets because firstly because the federal reserve’s interest rate determines not just the cost of borrowing which can drive discretionary spending, it also sets the incentive to save since higher rates could induce more saving and less expenditure. The inclusion of YTMs for 2yr [note](https://www.mql5.com/go?link=https://www.investopedia.com/terms/t/treasurynote.asp "https://www.investopedia.com/terms/t/treasurynote.asp") YTMs and 10yr [bond](https://www.mql5.com/go?link=https://www.investopedia.com/terms/b/bond.asp "https://www.investopedia.com/terms/b/bond.asp") is because they are the most liquid notes and bonds by trade volume annually posting figures of 10 trillion and 36 trillion respectively. And again, their yields could be used as an indicator as to whether consumers are more inclined to save by purchasing the instruments when yields are high or spend when the yields low.

Recent actions by the federal reserve paint a very hawkish outlook with Mr. Jerome Powell keen on battling inflation. Increases in interest rates this far have not yet had material impact on the US economy or XLY performance but the yield curve remains inverted and this is statistically significant in pointing to a recession. So, the monitoring of this data set and modelling it towards XLY performance could provide helpful harbingers to any changes in the outlook.

Unemployment

The unemployment data set has four features that include [CPI](https://www.mql5.com/go?link=https://www.investopedia.com/terms/c/consumerpriceindex.asp "https://www.investopedia.com/terms/c/consumerpriceindex.asp"), unemployment rate, productivity changes, and average hourly earnings. The US Bureau of Labor and Statistics is our primary data source here and as stated for consistency we are sticking to quarterly readings across all data sets. The productivity numbers are percentage changes quarter on quarter and including CPI within this data set helps put hourly earnings in a proper perspective.

![](https://c.mql5.com/2/61/4974263041636.png)

Unemployment data set is vital to XLY because the working class and middle class contribute 30% and 20% of discretionary spending respectively, per Pew Research study in 2022. These groups are a key demographic that need to have a job so the state of employment could be a window to what is to unfold in the XLY space. Add to that the wages earned when paired to CPI could provide a measure of purchasing power which again plays into discretionary spending. The final feature in this data set of productivity captures the sense of how efficient businesses are and the extent to which they depend on personnel versus say capital. Increasing productivity in theory could be a warning sign for pending unemployment.

In practice though if trends from the past 10yrs are studied the relation between the two has been very complex with a number of possible explanations being floated. One such is that increasing productivity could be a metric for re-employability of the workforce which should be good for spending. This stems from the fact that rising productivity usually comes with work force re-training. To that end despite the hawkish federal reserve the US employment has remained quite resilient and so including this data set could also be key in accurately modelling XLY performance.

Consumer Surveys

This data set has four features namely the consumer confidence index as published by the conference board, consumer credit from Federal Reserve Economic Data, Consumer Spending and Consumer Disposable income both sourced from the Bureau of Economic Analysis website. Possible extra data sets that could have been considered are Michigan consumer sentiment (it over laps with consumer confidence), retail sales (is considered in a separate data set), and employment data (has already been looked at above).

![](https://c.mql5.com/2/61/3388488691002.png)

Consumer surveys are key in weighing XLY because its lead indicators tend to auto-correlate positively with XLY performance. The more confident consumers are the more we should expect consumer spending and therefore discretionary spending to improve. The same can be said for disposable income, consumer credit and certainly aggregate consumer spending.

Recent events have seen consumer spending continue to rise in tandem with credit despite inflation headwinds. This could explain XLY’s relative outperformance to other SPRD sector ETFs this year and this clearly makes the case for this data set to be considered in modelling future XLY performance.

Retail Sales

This data set is used with 4 features namely total retail sales; motor vehicle sales; furniture, building & electronic sales; and clothing & general merchandise sales. The data is sourced from US Census Bureau and like all the other 5 data sets is logged quarterly.

![](https://c.mql5.com/2/61/6240944335120.png)

The relevance to XLY is obvious. Ancillary data sets, from the US Census Bureau, that could have been considered within this category are the ecommerce sales totals for each of the four sets we’ve picked. They were not chosen because there is a lot of overlap with what we have.

Current trends in retail sales as indicated in the data sets and highlighted above in consumer data, paint a bullish picture which as mentioned could explain XLY relative sector out performance. This data set therefore could be a good corollary to XLY that accounts for a lot of XLY’s performance and is thus suitable as a candidate data set for our model.

Demographics

Our final data set has 5 features namely: total population, population number below 25 years, population number over 65 years, number of immigrants, and net migration rate. All data is sourced from the US Census Bureau. An extra data set that was considered within this category but later abandoned was the percentage of males within the population. This is because its variability through the quarters was limited.

![](https://c.mql5.com/2/61/2867730768017.png)

Demographics matter to XLY because an increasing population and a more young and vibrant population tend to imply more discretionary spending especially if the population is young and working. Similarly, more immigrants in the long run can lead to increases in discretionary spending once they settle down and can find employment.

Recent trends point to increases not just in population, perhaps from a more health aware populace, but huge inflows in immigrants most of which are poorly documented. These are major shifts that do not necessarily bode positively for discretionary spending in the short run, but should be very significant in the long run as argued above. This data set therefore may help model long term XLY attributes that other data sets may miss.

**Feature Importance Calculation**

Feature importance is a weighting of data columns in a data set to establish which of them is more important in data classification or forecasting as is the case for XLY in this article. To compute the relative weights of each feature in each data set we’ll use the Gini impurity as computed by AlgLib’s decision forest classes. We used something similar in the [last article](https://www.mql5.com/en/articles/13715). The code listing for this has not changed only difference is we’re now checking for the weight of each column, as indicated below:

```
//+------------------------------------------------------------------+
//| Set the feature importance weights                               |
//+------------------------------------------------------------------+
void Cgini::Set(string FileName)
{
      f.DFBuilderCreate(b);
      f.DFBuilderSetDataset(b,xy,xy.Rows(),xy.Cols()-1,1);
      f.DFBuilderSetImportanceTrnGini(b);
      f.DFBuilderSetSubsampleRatio(b,1.0);
      f.DFBuilderBuildRandomForest(b,50,m,r);
      //
      for(int i=0;i<int(r.m_varimportances.Size());i++)
      {
         printf(__FUNCSIG__+" "+FileName+" weight at: "+IntegerToString(i)+" is: "+DoubleToString(r.m_varimportances[i]));
      }
}
```

On running the script, that calls the class implementing the above code, across all the data sets we get the following importance weights for each column. Complete csv files are attached at end of article.

For economic data set the logs print is:

```
2023.11.21 20:40:57.519 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) economic weight at: 0 is: 0.26282632

2023.11.21 20:40:57.519 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) economic weight at: 1 is: 0.30208064

2023.11.21 20:40:57.519 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) economic weight at: 2 is: 0.38399929

2023.11.21 20:40:57.519 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) economic weight at: 3 is: 0.05109375
```

The weightings are for the features GDP, personal income, international trade in goods and international trade in services respectively. Clearly the trade in international goods, which is mostly with China, is a key driver for XLV performance.

For the Interest rate data set the print is:

```
2023.11.21 20:57:52.792 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) interest_rate weight at: 0 is: 0.29987368

2023.11.21 20:57:52.792 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) interest_rate weight at: 1 is: 0.34667516

2023.11.21 20:57:52.792 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) interest_rate weight at: 2 is: 0.35091898
```

With the weightings being for fed benchmark rate, 2yr yield and 10yr yield; The 10yr yields are the most important drivers of XLY performance.

For the Employment data set we have:

```
2023.11.21 21:04:19.979 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) employment weight at: 0 is: 0.17145942

2023.11.21 21:04:19.979 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) employment weight at: 1 is: 0.29297243

2023.11.21 21:04:19.979 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) employment weight at: 2 is: 0.25804000

2023.11.21 21:04:19.979 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) employment weight at: 3 is: 0.27752815
```

The CPI Index and perhaps average hourly earnings are the most important drivers for XLY performance since the 4 features of unemployment rate, CPI, productivity, and average hourly earnings all have their weights printed above respectively.

With the Consumer data set we have:

```
2023.11.21 21:14:34.451 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) consumer weight at: 0 is: 0.15715087

2023.11.21 21:14:34.451 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) consumer weight at: 1 is: 0.27208766

2023.11.21 21:14:34.451 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) consumer weight at: 2 is: 0.29780771

2023.11.21 21:14:34.451 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) consumer weight at: 3 is: 0.27295376
```

This is for the features: consumer confidence, credit, spending, and disposable income respectively; and our winners are clearly the final 3 with spending the clear overall winner.

The Retail data set prints:

```
2023.11.21 21:28:17.112 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) retail weight at: 0 is: 0.24864306

2023.11.21 21:28:17.112 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) retail weight at: 1 is: 0.24721743

2023.11.21 21:28:17.112 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) retail weight at: 2 is: 0.27211211

2023.11.21 21:28:17.113 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) retail weight at: 3 is: 0.23202741
```

Which represents the features: total sales, vehicle sales, furniture & building materials sales and clothing and general merchandise sales respectively. Furniture & building materials is the key driver here for XLY performance with the highest Gini impurity of 0.27.

And finally, with the Demographic data set we print:

```
2023.11.21 21:40:19.042 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) demographic weight at: 0 is: 0.17343077

2023.11.21 21:40:19.042 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) demographic weight at: 1 is: 0.17324359

2023.11.21 21:40:19.043 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) demographic weight at: 2 is: 0.20091084

2023.11.21 21:40:19.043 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) demographic weight at: 3 is: 0.25756628

2023.11.21 21:40:19.043 xly_1 (GBPJPY.ln,M12) void Cgini::Set(string) demographic weight at: 4 is: 0.19484851
```

That is for the features: total population, population below 25yrs, population over 65yrs, no. of immigrants, and net migration rate. Paradoxically here the number of immigrants is the largest driver of XLY performance. One would have thought most immigrants cannot afford discretionary spending but the Gini impurity accorded here, given the large number of columns, is very telling.

**The composite Data Set**

With these Gini impurity values across all 6 data sets, comparing them to each other directly would be an err because the number of features (or data columns) in each data set is different. Remember the Gini impurity weights in each data set add up to one, so the more features a data set has the more likely it is for the individual Gini impurities to be small. The need to therefore normalize this across all data sets is important before any meaningful comparison can be made.

A tactful approach at this would be to multiply each Gini impurity of each data set feature with the number of features in the data set. So, if a data set has 3 columns all its Gini impurities get multiplied by 3. If we run this across all features in all the data sets we are presented with the table below:

<table: Gini and weighted values/>

In coming up with a composite data set we would need to select some features, and omit others thus we need a cut off threshold. I keep things simple here and omit all features that score less than one out of the composite data set. What we are left with is the values highlighted in green in the table image above, 12 in total.

Interestingly, as a side note, if we run the feature importance weights for this new 12 feature data set, we get trade in international goods as a clear winner by some margin, followed by productivity at some distance! All csv files and scripts for this are attached at the end of the article.

**The Multilayer Perceptron (MLP)**

Our composite data set will now be tested with our model, and our model of choice is a multilayer perceptron. The MLP to be used, is class instance provided by AlgLib and it allows no more than 2 hidden layers. Some work typically needs to be put in, in determining the ideal number to use as well as the size of each of the hidden layers. But off the bat before we get into that the broker I am testing with does not provide XLY ETF as a tradeable security, and whereas XLY one minute and tick price information can be imported into MQL5 as a custom symbol, we will not hustle that much for this article and instead we’ll use a symbol that is the largest holding of the ETF, Amazon (AMZN), as a proxy to our XLY ETF.

So, to establish the ideal settings of our model (i.e. number of hidden layers and the size of each) we perform a cross validation by first optimizing for the ideal values from a period from 2013.01.01 up to 2022.01.01, and then do a forward pass from that date up to 2023.10.01. In doing so, these are our test run settings:

![s](https://c.mql5.com/2/61/settings.png)

Optimizations were run, in open price mode with questionable price history quality as can be seen from the reports below. For our best forward pass, we got the following reports for back testing and forward testing respectively:

![r_back](https://c.mql5.com/2/61/back_test.png)

![r_forward](https://c.mql5.com/2/61/forward_test.png)

The forward test and back testing were not performed on XLY and the data quality used needed improvement, but the thesis of using our composite data set may have merit based on indications in these reports.

**Aspects not covered in this article**

Before concluding it may be a good idea to highlight a few extra aspects that would be important to consider when setting up to trade XLY within MetaTrader that have not been the focus for this article. As always, a lot can be said on this topic so we’ll stick to just a few crucial ideas.

Consider alternative Models

Before embarking on cross validation, it would be good idea to have alternative models for testing with the ideal data set such that you compare not just walk forward results of one model, but you compare these results across more than one model. We have considered the MLP as our model here but alternatives easily include not just different types of neural networks like the Boltzmann Machine for instance but even a Linear Discriminant Model could serve as an alternative. The more models you explore the more you are aware of not just what works, but what works with minimal drawdowns and risk.

Use Broker’s Real Tick Data

Testing and validating a model with the price data of your Broker is just as important as having a good well coded model. Price gaps and bar prices that are inconsistent with real tick data are just some of the pitfalls that can easily blind side you when testing, to the real under performance of your system so some diligence here is always warranted.

**Conclusion**

In conclusion we have considered alternative data sets that can be used to forecast XLY performance from publicly available data. While ensuring this data is gathered at the same interval (quarterly time frame), we have computed the feature importance weights of each respective data column (feature) and selected those with the most bearing to XLY’s performance. After selecting the features, we brought them together in a composite data set and modelled it in a perceptron on AMZN, a major holding of XLY ETF, over the past decade with a cross validation period from 2022.01.01 to October of this year. Though the forward results were promising, more testing on alternative models and with real tick broker data would be ideal before this system can be considered for use.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13775.zip "Download all attachments in the single ZIP archive")

[attach.zip](https://www.mql5.com/en/articles/download/13775/attach.zip "Download attach.zip")(22.04 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/458156)**

![Neural networks made easy (Part 52): Research with optimism and distribution correction](https://c.mql5.com/2/57/optimistic-actor-critic-avatar.png)[Neural networks made easy (Part 52): Research with optimism and distribution correction](https://www.mql5.com/en/articles/13055)

As the model is trained based on the experience reproduction buffer, the current Actor policy moves further and further away from the stored examples, which reduces the efficiency of training the model as a whole. In this article, we will look at the algorithm of improving the efficiency of using samples in reinforcement learning algorithms.

![Developing a Replay System — Market simulation (Part 15): Birth of the SIMULATOR (V) - RANDOM WALK](https://c.mql5.com/2/55/Desenvolvendo_um_sistema_de_Replay_Parte_15_AVATAR.png)[Developing a Replay System — Market simulation (Part 15): Birth of the SIMULATOR (V) - RANDOM WALK](https://www.mql5.com/en/articles/11071)

In this article we will complete the development of a simulator for our system. The main goal here will be to configure the algorithm discussed in the previous article. This algorithm aims to create a RANDOM WALK movement. Therefore, to understand today's material, it is necessary to understand the content of previous articles. If you have not followed the development of the simulator, I advise you to read this sequence from the very beginning. Otherwise, you may get confused about what will be explained here.

![Developing a Replay System — Market simulation (Part 16): New class system](https://c.mql5.com/2/55/replay-p16-avatar.png)[Developing a Replay System — Market simulation (Part 16): New class system](https://www.mql5.com/en/articles/11095)

We need to organize our work better. The code is growing, and if this is not done now, then it will become impossible. Let's divide and conquer. MQL5 allows the use of classes which will assist in implementing this task, but for this we need to have some knowledge about classes. Probably the thing that confuses beginners the most is inheritance. In this article, we will look at how to use these mechanisms in a practical and simple way.

![Trade transactions. Request and response structures, description and logging](https://c.mql5.com/2/57/printformat_trading_transactions_avatar.png)[Trade transactions. Request and response structures, description and logging](https://www.mql5.com/en/articles/13052)

The article considers handling trade request structures, namely creating a request, its preliminary verification before sending it to the server, the server's response to a trade request and the structure of trade transactions. We will create simple and convenient functions for sending trading orders to the server and, based on everything discussed, create an EA informing of trade transactions.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=znxwpdreittxqsrdszuheglxsnqcowgm&ssn=1769251795328039113&ssn_dr=0&ssn_sr=0&fv_date=1769251795&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13775&back_ref=https%3A%2F%2Fwww.google.com%2F&title=The%20case%20for%20using%20a%20Composite%20Data%20Set%20this%20Q4%20in%20weighing%20SPDR%20XLY%27s%20next%20performance%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925179565582941&fz_uniq=5083144127886923219&sv=2552)

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