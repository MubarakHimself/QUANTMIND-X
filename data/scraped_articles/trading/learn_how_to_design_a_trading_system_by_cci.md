---
title: Learn how to design a trading system by CCI
url: https://www.mql5.com/en/articles/10592
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:14:27.817917
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=uxtdexykidorrrqruxbemoauquksymiq&ssn=1769181266858174490&ssn_dr=0&ssn_sr=0&fv_date=1769181266&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10592&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Learn%20how%20to%20design%20a%20trading%20system%20by%20CCI%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918126617352433&fz_uniq=5069269038468629073&sv=2552)

MetaTrader 5 / Trading


### Introduction

This is a new article from our series in which we learn how to design trading systems based on simple strategies using the most commonly used technical indicators. This time we will discuss the Commodity Channel Index (CCI) indicator. As always, I will try to explain the fundamentals of the topic to help you understand the idea and usage. Ultimately, this way may give you insights and new ideas which you can use in your trading.

In this article, we will design a trading system for CCI. Topics to be covered:

- [CCI definition](https://www.mql5.com/en/articles/10592#definition)
- [CCI strategy](https://www.mql5.com/en/articles/10592#strategy)
- [CCI trading system blueprint](https://www.mql5.com/en/articles/10592#blueprint)
- [CCI trading system](https://www.mql5.com/en/articles/10592#system)
- [Conclusion](https://www.mql5.com/en/articles/10592#conclusion)

We will start with detailed information about what the Commodity Channel Index (CCI), what it measures, how we can calculate it. When we understand the fundamentals and the roots of what we are doing, we will be able to use the tools more efficiently and find more ideas and insights about it. This is what we are going to discuss in the "CCI Definition" topic. Then, we will work on a simple strategy that can be used with CCI during different market trends or conditions — this is what we will learn in the "CCI Strategy" part. Then, we will learn how we can design a trading system based on this strategy by planning what we need to design and what we want the computer to do — part "CCI trading system blueprint". Then, we will learn how to design what we planned by the trading system blueprint — part "CCI trading system".

We will use the MetaTrader 5 trading platform and MetaQuotes Language Editor in this article. You can download MetaTrader 5 from this link: [https://www.metatrader5.com/en/download](https://www.metatrader5.com/en/download "https://www.metatrader5.com/en/download")

Once you download and install MetaTrader 5, you will see the terminal as in the following picture:

![MT5 terminal](https://c.mql5.com/2/45/MT5_terminal.png)

Then, you can open MetaQuotes Language editor by pressing F4 while the trading terminal is open or by selecting Tools menu in the terminal and then clicking on MetaQuotes Language Editor:

![ MQL opening](https://c.mql5.com/2/45/MQL_opening.png)

Or press on IDE button in the MetaTrader 5 toolbar:

![MT5 IDE button](https://c.mql5.com/2/45/MT5_IDE_button.png)

Here I need to mention the benefit of programming in trading, as it can help us to be disciplined. The discipline is an important factor to our trading success, because emotions can be harmful to trading: the right decision should be taken according to winning trading plan but it happens to most of traders that they cannot take the right action or decision because of fear, greed, or other emotions.

The programming features come as we can program our trading plan according to our conditions and execute it. The program will do what we need according to our rules without involving emotions that can affect our trading, which means that programming will help us to be disciplined. This is where MQL5 comes in handy as it help us to program our trading conditions and plans in a way which can be executed automatically in the trading terminal. These program can be from very simple to complex ones, depending on our trading plan.

Disclaimer: All content of this article is made for the purpose of education only not for anything else. So, you will be responsible for any action you take based on the content of this article, as the content of this article does not guarantee any kind of results.

Now, we are ready to start our article to build a new block in our learning journey. So, let us do that...

### CCI definition

In this part, we will learn about the Commodity Channel Index (CCI) indicator in more detail: what it measures, how it is calculated and how to use it.

Commodity Channel index (CCI) indicator created by Donald Lambert is a momentum indicator which measures the current price relative to an average of price of a given period of time. When Lambert created this indicator, the initial objective was to measure cyclical movements in commodities but it can measure also other financial instruments. CCI can be used to spot the trend strength and reversals and this is a normal result according to its nature as a momentum indicator.

According to the calculation of CCI and what it measures, when CCI is high this means that the prices are far above its average and vice versa, when CCI is low this means that the prices are far below its average.

The CCI is calculated in the following steps:

- Get Typical Price (TP):

> **TP = (High + Low + Close)/3**

- Get Simple Moving Average of TP:

> **SMA of TP = Sum of TP/n**

- Get Mean deviation:

> **Mean Deviation = Sum of Abs(TP-SMA)/n**

- Get CCI:

> **CCI =  (TP-SMA of TP)/(0.015\*Mean Deviation)**

Now, let us take an example to apply this calculation:

Suppose, we have the following data for an instrument for a period of 14 days:

| Days | High | Low | Close |
| --- | --- | --- | --- |
| 1 | 100 | 90 | 110 |
| 2 | 130 | 100 | 120 |
| 3 | 140 | 110 | 130 |
| 4 | 130 | 100 | 120 |
| 5 | 120 | 95 | 110 |
| 6 | 140 | 120 | 130 |
| 7 | 160 | 130 | 150 |
| 8 | 170 | 150 | 160 |
| 9 | 155 | 130 | 140 |
| 10 | 140 | 120 | 130 |
| 11 | 160 | 140 | 150 |
| 12 | 180 | 160 | 170 |
| 13 | 190 | 170 | 190 |
| 14 | 200 | 180 | 210 |

Here is how we can calculate CCI using this data:

- First we will calculate Typical price (TP):

> **TP = (High + Low + Close)/3**

So, after the calculation of Typical price, it will be as follows:

![Example CCI1](https://c.mql5.com/2/45/Example_CCI1.png)

- Then, we need to calculate the Simple Moving Average (SMA) of the calculated TP:

> **SMA of TP = Sum of TP/n**

The result is as follows:

![Example CCI2](https://c.mql5.com/2/45/Example_CCI2.png)

- Then, we will calculate the difference between TP and SMA in absolute values to calculate the mean deviation:

> **Abs (TP-SMA)**

Here it is:

![Example CCI3](https://c.mql5.com/2/45/Example_CCI3.png)

- Now, we need to calculate Mean deviation:

> **Mean Deviation = Sum of Abs(TP-SMA)/n**

**![Example CCI4](https://c.mql5.com/2/45/Example_CCI4.png)**

- Now, it is time to calculate CCI:

> **CCI = (TP-SMA of TP)/(0.015\*Mean Deviation)**

> CCI = (196.67-141.19)/(0.015\*22.86)
>
> CCI = 55.48/0.3429
>
> > ****CCI = 161.80****

Nowadays, we do not need to calculate indicators manually because MetaTrader 5 offers ready-to-use built-in indicators. You can use it immediately by selecting the CCI indicator in the platform as is shown in the following picture:

![CCI insert](https://c.mql5.com/2/45/CCI_insert.png)

After selecting CCI the following window will be appeared:

![ CCI window](https://c.mql5.com/2/45/CCI_window.png)

The previous window of the indicator shows its parameters:

- 1: to determine the period of the indicator.
- 2: to set price type that will be used in indicator calculation and here we will choose Typical price.
- 3: to set the indicator style: CCI color, CCI line type, and CCI line thickness.

After specifying all parameters press Ok, and the indicator will appear on the chart:

![CCI indicator on the chart](https://c.mql5.com/2/45/CCI_indicator_on_the_chart.png)

CCI is an oscillator which oscillates around and between 100 and -100

### CCI strategy

In this part, we will talk about two simple strategies which can be used with CCI. One of the strategies can be used according to market trend or direction (uptrend, downtrend and sideways). The other one is a simple crossover between CCI value with both of zero and 100 or -100. Before we talk about these strategies, I need to mention that there are many strategies which can be used with CCI. You should test every strategy by yourself on a demo account to see if it is useful or not in your trading before using it on a real account with real money.

So, the strategies are as follows.

**First strategy: Using CCI according to market trend**

- During uptrend:

Uptrend is the market direction during which prices create higher lows and higher highs which means that there is a control from buyers as prices move up. In this market condition we can use CCI as a signal provider to generate a buying signal when CCI line breaks above 100 level. So, CCI > 100. Then we can use another effective tool like price action for example to take profit.

> > **CCI > 100 = buy** **Take profit signal can be used from another tool which can be more useful, such as price action by breaking below previous low**

- During downtrend:

Downtrend is the trend that it is the opposite of uptrend: prices create lower highs and lower lows which means that there is a control from sellers and prices move down. In such a market, CCI can also be used to generate signals: a shorting signal appears when CCI line breaks below - 100 level. Price action can be used to take profits.

> > **CCI < -100 = short** **Take profit signal can be taken from another tool which can be more useful, such as price action by breaking above previous high**

- During sideways:

During sideways movements, we can find that there is a balance between buyers and sellers so no one visually prevails on the market. It is any situation except uptrend and downtrend.

During sideways we can use CCI to generate signals. When CCI line breaks below -100 level, this will be a buy signal and when CCI line breaks above 100, this will be a take profit. Or when CCI line breaks above 100, this will be a short signal and when CCI line breaks below -100, this will be a take profit.

> **CCI < -100 = buy**
>
> **CCI > 100 = take profit **CCI > 100 = short****
>
> **CCI < -100 = take profit**

**Second strategy: Zero crossover signals:**

In this article, we can use CCI in a different way as we can take the action of entering the market according to the crossover with zero level and take profits according to the crossover with 100 and -100 values as per our position type (buy or short).

- For buy signal:

Watch CCI value ralive to the zero level. When CCI value breaks above zero level, this will be a buy signal and we can take profit when CCI value breaks above 100.

**CCI > 0 = Buy**

**CCI > 100 = take profit**

- For short signal:

Watch CCI value ralive to the zero level. When CCI value breaks below zero level, this will be a short signal and we can take profit when CCI value break below -100.

**CCI < 0 = Buy**

**CCI < 100 = take profit**

### CCI trading system blueprint

Now, we come to the most interesting part in this article as we need to program these mentioned strategies to automate signals when observing CCI manually.

So, we need now to code these mentioned strategies and inform the computer what to do according to what we need exactly. We will start step by step by designing a blueprint for that.

**First strategy: Using CCI according to market trend:**

- During uptrend:

We need the program to check CCI value every tick and do something according to the CCI value.

If CCI above 100, give buy and if not do nothing.

![CCI uptrend blueprint](https://c.mql5.com/2/45/CCI_uptrend_blueprint.png)

- During downtrend:

Check CCI value every tick and take an action according to this value.

If CCI value below -100, give a short signal and if not do nothing.

![CCI downtrend blueprint](https://c.mql5.com/2/45/CCI_downtrend_blueprint.png)

- During sideways:

During sideways, we need the program to check CCI value then do something according to that.

If CCI value below -100, give a buy signal, give a take profit. If CCI is not below -100, check if CCI above -100 and below 100 do nothing (hold) and if not check if CCI above 100, give short then check if CCI is below -100, give a take profit and if not check if CCI value below 100 and above -100 do nothing (hold).

![CCI sideway blueprint](https://c.mql5.com/2/45/CCI_sideway_blueprint.png)

**Second strategy: Zero crossover signals:**

- For buy signal:

Check every tick CCI value and Zero level. When CCI value becomes greater than zero, generate a buy signal on the chart, and when CCI value become above 100 show a take profit signal on the chart.

![Zero crossover CCI strategy - buy blueprint](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_buy_blueprint.png)

- For short signal:

Check every tick CCI value and Zero level and when CCI value becomes below zero level show a short signal on the chart, and when CCI value becomes below -100 give a take profit signal.

![Zero crossover CCI strategy - short blueprint](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_short_blueprint.png)

### CCI trading system

Now, we need to write the mentioned strategy as a code for the computer to do what we need.

First, we need to code a simple program which shows CCI values on the chart. The programming consists of the following steps:

1. Creating an array of prices.
2. Sorting price array from current data.
3. Defining CCI properties.
4. Sorting results.
5. Getting value of current data.
6. Charting CCI values and showing these values on the chart.

The following is how to code the previous steps to create a program which can let the computer display CCI values on the chart automatically:

```
//+------------------------------------------------------------------+
//|                                            Simple CCI System.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array of prices
   double ArrayOfPrices[];

   //Sorting price array from current data
   ArraySetAsSeries(ArrayOfPrices,true);

   //CCI properties Definition
   int CCIDef = iCCI(_Symbol,_Period,14,PRICE_CLOSE);

   //Storing results
   CopyBuffer(CCIDef,0,0,3,ArrayOfPrices);

   //Getting value of current data
   double CCIValue = (ArrayOfPrices[0]);

   //Charting CCI values
   Comment ("CCI Value = ",CCIValue);
  }
//+------------------------------------------------------------------+
```

After that we can find this program (Simple CCI System) in the Navigator, select it and execute in the trading platform:

![Updated CCI navigator](https://c.mql5.com/2/46/Updated_CCI_navigator.png)

The program (Simple CCI System) can be opened by double click or drag and drop on the chart. After that the following window will be opened:

![Simple CCI System window](https://c.mql5.com/2/45/Simple_CCI_System_window.png)

After enabling the "Allow Algo Trading" option and pressing ok, the program (EA) Simple CCI System will be attached to the chart and CCI value will be visualized on the chart:

![Simple CCI System attach](https://c.mql5.com/2/45/Simple_CCI_System_attach.png)

Now, we will learn how to code the two mentioned strategies:

- First strategy: Using CCI according to market trend:

  - During uptrend:

> CCI > 100 = buy

We will do the following steps:

1. Creating an array for prices.
2. Sorting price array from current data.
3. defining CCI properties.
4. Sorting results.
5. Getting value of current data.
6. Setting condition CCI signals during uptrend.

Here is how to code these steps:

```
//+------------------------------------------------------------------+
//|                                         Uptrend CCI Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array of prices
   double ArrayOfPrices[];

   //Sorting price array from current data
   ArraySetAsSeries(ArrayOfPrices,true);

   //CCI properties Definition
   int CCIDef = iCCI(_Symbol,_Period,14,PRICE_CLOSE);

   //Storing results
   CopyBuffer(CCIDef,0,0,3,ArrayOfPrices);

   //Getting value of current data
   double CCIValue = (ArrayOfPrices[0]);

   //CCI signals for uptrend
   if(CCIValue> 100)
   Comment ("UPTREND CCI - BUY SIGNAL ");

  }
//+------------------------------------------------------------------+
```

After that we can find this program Uptrend CCI Strategy in the Navigator:

![Uptrend strategy CCI navigator](https://c.mql5.com/2/46/Uptrrend_strategy_CCI_navigator.png)

Again open this program CCI Uptrend Strategy by double click or drag and drop on the chart. The following window will open:

![Uptrend CCI strategy window](https://c.mql5.com/2/45/Uptrend_CCI_strategy_window.png)

After enabling "Allow Algo Trading" and pressing ok, the program (EA) - Uptrend CCI Strategy - will be attached to the chart:

![Uptrend CCI strategy attach](https://c.mql5.com/2/45/Uptrend_CCI_strategy_attach.png)

Signals of Uptrend CCI Strategy will be displayed according to this strategy:

![Uptrend CCI Strategy - Buy Signal](https://c.mql5.com/2/45/Uptrend_CCI_Strategy_-_Buy_Signal.png)

- During downtrend:

> CCI < -100 = short

We will code the following steps:

1. Creating an array for prices.
2. Sorting price array from current data.
3. defining CCI properties.
4. Sorting results.
5. Getting value of current data.
6. Setting condition CCI signals during downtrend.

Below is the code of the program which can be launched in the trading platform and execute what we need automatically:

```
//+------------------------------------------------------------------+
//|                                       Downtrend CCI Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array of prices
   double ArrayOfPrices[];

   //Sorting price array from current data
   ArraySetAsSeries(ArrayOfPrices,true);

   //CCI properties Definition
   int CCIDef = iCCI(_Symbol,_Period,14,PRICE_CLOSE);

   //Storing results
   CopyBuffer(CCIDef,0,0,3,ArrayOfPrices);

   //Getting value of current data
   double CCIValue = (ArrayOfPrices[0]);

   //CCI signals
   if(CCIValue< -100)
   Comment ("DOWNTREND CCI - SHORT SIGNAL ");
  }
//+------------------------------------------------------------------+
```

Find this program Downtrend CCI Strategy in the Navigator:

![Downtrend strategy CCI navigator](https://c.mql5.com/2/46/Downtrend_strategy_CCI_navigator.png)

Open this program Downtrend CCI Strategy by double click or drag and drop on the chart. The following window will be opened:

![Downtrend CCI strategy window](https://c.mql5.com/2/45/Downtrend_CCI_strategy_window.png)

Enable "Allow Algo Trading" and press ok, the program (EA) -  Downtrend CCI Strategy - will be attached to the chart:

![Downtrend CCI strategy attach](https://c.mql5.com/2/45/Downtrend_CCI_strategy_attach.png)

The ignals of Downtrend CCI Strategy program will appear according to this strategy the same like the following picture:

![Downtrend CCI Strategy - Short Signal](https://c.mql5.com/2/45/Downtrend_CCI_Strategy_-_Short_Signal.png)

- During sideways:

  - For buy signals:

> CCI < -100 = buy
>
> CCI > 100 = take profit

Steps:

1. Creating an array for prices.
2. Sorting price array from current data.
3. Defining CCI properties.
4. Sorting results.
5. Getting value of current data.
6. Setting condition CCI buy signal during sideways.

Code these steps to create a program which can do what we need:

```
//+------------------------------------------------------------------+
//|                                  Sideways CCI Strategy - Buy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array of prices
   double ArrayOfPrices[];

   //Sorting price array from current data
   ArraySetAsSeries(ArrayOfPrices,true);

   //CCI properties Definition
   int CCIDef = iCCI(_Symbol,_Period,14,PRICE_CLOSE);

   //Storing results
   CopyBuffer(CCIDef,0,0,3,ArrayOfPrices);

   //Getting value of current data
   double CCIValue = (ArrayOfPrices[0]);

   //CCI signals
   if(CCIValue< -100)
   Comment ("SIDEWAYS CCI - BUY SIGNAL ");

   if(CCIValue> 100)
   Comment ("SIDEWAYS CCI - TAKE PROFIT");
  }
//+------------------------------------------------------------------+
```

Find this program Sideways CCI Strategy - Buy in the Navigator:

![ Sideways buy strategy CCI navigator](https://c.mql5.com/2/46/Sideways_buy_strategy_CCI_navigator.png)

Open it by double click or drag and drop. The following window will be opened:

![Sideways CCI buy strategy window](https://c.mql5.com/2/46/Sideways_CCI_buy_strategy_window.png)

Enable "Allow Algo Trading" and press ok, the program (EA) -  Sideways CCI Strategy - Buy - will be attached to the chart:

![Sideways CCI buy strategy attach](https://c.mql5.com/2/46/Sideways_CCI_buy_strategy_attach.png)

Then, we can find signals appear according to this strategy Sideways CCI Strategy - Buy:

![Sideways CCI - Buy](https://c.mql5.com/2/46/Sideways_CCI_-_Buy.png)

Take profit signal is at CCI > 100 = take profit. Example:

![Sideways CCI - Buy - TP](https://c.mql5.com/2/46/Sideways_CCI_-_Buy_-_TP.png)

> - For short signals:

> > CCI > 100 = short
> >
> > CCI < -100 = take profit

We will code that through the following steps:

1. Creating an array for prices.
2. Sorting price array from current data.
3. Defining CCI properties.
4. Sorting results.
5. Getting value of current data.
6. Setting condition CCI short signal during sideways.

And the following is how to code these steps to create the program Sideways CCI Strategy - Short:

```
//+------------------------------------------------------------------+
//|                                Sideways CCI Strategy - Short.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array of prices
   double ArrayOfPrices[];

   //Sorting price array from current data
   ArraySetAsSeries(ArrayOfPrices,true);

   //CCI properties Definition
   int CCIDef = iCCI(_Symbol,_Period,14,PRICE_CLOSE);

   //Storing results
   CopyBuffer(CCIDef,0,0,3,ArrayOfPrices);

   //Getting value of current data
   double CCIValue = (ArrayOfPrices[0]);

   //CCI signals
   if(CCIValue> 100)
   Comment ("SIDEWAYS CCI - SHORT SIGNAL ");

   if(CCIValue< -100)
   Comment ("SIDEWAYS CCI - TAKE PROFIT");
  }
//+------------------------------------------------------------------+
```

After that we the program Sideways CCI Strategy - Short can be found in the Navigator:

![Sideways short strategy CCI navigator](https://c.mql5.com/2/46/Sideways_short_strategy_CCI_navigator.png)

Open Sideways CCI Strategy - Short by double click or drag and drop on the chart. The following window will be opened:

![Sideways CCI short strategy window](https://c.mql5.com/2/46/Sideways_CCI_short_strategy_window.png)

After enabling "Allow Algo Trading" and pressing ok, the program (EA) - Sideways CCI Strategy - Short - will be attached to the chart and it will be the same like the following:

![Sideways CCI short strategy attach](https://c.mql5.com/2/46/Sideways_CCI_short_strategy_attach.png)

Signals will appear according to this strategy Sideways CCI Strategy - Short:

![Sideways CCI - Short](https://c.mql5.com/2/46/Sideways_CCI_-_Short.png)

We can take profit signal when CCI < -100 = take profit:

![Sideways CCI - Short - TP](https://c.mql5.com/2/46/Sideways_CCI_-_Short_-_TP.png)

- Second strategy: Zero crossover signals:

  - For buy signal:

> CCI > 0 = Buy
>
> CCI > 100 = take profit

We will code that through the following steps:

1. Creating an array for prices.
2. Sorting price array from current data.
3. Defining CCI properties.
4. Sorting results.
5. Getting value of current data.
6. Setting condition for CCI value and zero level crossover.

Steps to create the program Zero crossover CCI Strategy - Buy:

```
//+------------------------------------------------------------------+
//|                            Zero crossover CCI Strategy - Buy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array of prices
   double ArrayOfPrices[];

   //Sorting price array from current data
   ArraySetAsSeries(ArrayOfPrices,true);

   //CCI properties Definition
   int CCIDef = iCCI(_Symbol,_Period,14,PRICE_CLOSE);

   //Storing results
   CopyBuffer(CCIDef,0,0,3,ArrayOfPrices);

   //Getting value of current data
   double CCIValue = (ArrayOfPrices[0]);

   //CCI signals
   if(CCIValue > 0)
   Comment ("Zero crossover CCI - BUY SIGNAL ");

   if(CCIValue > 100)
   Comment ("Zero crossover CCI - TAKE PROFIT");
  }
//+------------------------------------------------------------------+
```

Find this program Zero crossover CCI Strategy - Buy in the Navigator:

![Zero crossover buy strategy CCI navigator](https://c.mql5.com/2/46/Zero_crossover_buy_strategy_CCI_navigator.png)

Open this program by double click or drag and drop on the chart. The following window will be opened:

![Zero crossover CCI strategy - Buy window](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_Buy_window.png)

After enabling "Allow Algo Trading" and pressing ok, the program (EA) - Zero crossover CCI Strategy - Buy - will be attached to the chart:

![Zero crossover CCI strategy - Buy attach](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_Buy_attach.png)

Signals will appear according to this strategy Zero crossover CCI Strategy - Buy:

![Zero crossover CCI strategy - Buy](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_Buy.png)

We can take profit signal when CCI > 100 = take profitand the following picture is an example for it:

![Zero crossover CCI strategy - Buy - TP](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_Buy_-_TP.png)

- For short signal:

> CCI < 0 = Buy
>
> CCI < -100 = take profit

We will code that through the following steps:

1. Creating an array for prices.
2. Sorting price array from current data.
3. Defining CCI properties.
4. Sorting results.
5. Getting value of current data.
6. Setting condition for CCI value and zero level crossover.

Coding steps to create the program Zero crossover CCI Strategy - Short:

```
//+------------------------------------------------------------------+
//|                          Zero crossover CCI Strategy - Short.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array of prices
   double ArrayOfPrices[];

   //Sorting price array from current data
   ArraySetAsSeries(ArrayOfPrices,true);

   //CCI properties Definition
   int CCIDef = iCCI(_Symbol,_Period,14,PRICE_CLOSE);

   //Storing results
   CopyBuffer(CCIDef,0,0,3,ArrayOfPrices);

   //Getting value of current data
   double CCIValue = (ArrayOfPrices[0]);

   //CCI signals
   if(CCIValue < 0)
   Comment ("Zero crossover CCI - SHORT SIGNAL ");

   if(CCIValue < -100)
   Comment ("Zero crossover CCI - TAKE PROFIT");
  }
//+------------------------------------------------------------------+
```

After that find this program Zero crossover CCI Strategy - Short in the Navigator:

![Zero crossover short strategy CCI navigator](https://c.mql5.com/2/46/Zero_crossover_short_strategy_CCI_navigator.png)

Open this program Zero crossover CCI Strategy - Short by double click on it or drag it and drop on the chart. The following window will be opened:

![Zero crossover CCI strategy - Short window](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_Short_window.png)

After enabling "Allow Algo Trading" and pressing ok, the program (EA)-  Zero crossover CCI Strategy - Short - will be attached to the chart:

![Zero crossover CCI strategy - Short attach](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_Short_attach.png)

Signals will appear according to this strategy -  Zero crossover CCI Strategy - Short:

![Zero crossover CCI strategy - Short](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_Short.png)

We can take profit signal when CCI < -100 = take profitand the following picture is an example for it:

![Zero crossover CCI strategy - Short - TP](https://c.mql5.com/2/46/Zero_crossover_CCI_strategy_-_Short_-_TP.png)

### Conclusion

The Commodities Channel Index indicator - CCI - is another tool which can be used in our favor to be beneficial to our trading and get better results after testing every strategy of CCI and optimize it in case of being profitable after testing. and you can consider this article is an introduction to a new tool that can be used in your trading if it is suitable for your trading plan and style.

As we knew at this article what is CCI indicator and what does it mean and what does it measure and how can we calculate it and saw an example for that to deepen our knowledge and awareness toward this tool which can a reason to open our eyes to more insights and new ideas.

Then, we knew simple strategies which can be used by CCI during different market conditions (uptrend, downtrend, and sideways) and Zero crossover strategy and like what I mentioned that there many strategies can be used by CCI and you can learn more to know which of them can be useful for your trading but I believe that when you know basics of something, your usage will be more effective.

Then, we knew how to design a blueprint for this strategy which can be helpful to write our program to design a trading system by CCI. Then, we knew how to code this strategy by MQL5 and how to attach and execute this trading system to our Meta Trader 5 trading platform to generate signals automatically and then we knew how signals can be generated on our charts.

What I need to confirm now, is you must test any new strategy before you use it live on your real account. And I hope that you find this article useful for you and your trading by included information or by opening your eyes to a new ideas or insights which can be used in your trading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10592.zip "Download all attachments in the single ZIP archive")

[Simple\_CCI\_System.mq5](https://www.mql5.com/en/articles/download/10592/simple_cci_system.mq5 "Download Simple_CCI_System.mq5")(1.1 KB)

[Uptrend\_CCI\_Strategy.mq5](https://www.mql5.com/en/articles/download/10592/uptrend_cci_strategy.mq5 "Download Uptrend_CCI_Strategy.mq5")(1.13 KB)

[Downtrend\_CCI\_Strategy.mq5](https://www.mql5.com/en/articles/download/10592/downtrend_cci_strategy.mq5 "Download Downtrend_CCI_Strategy.mq5")(1.13 KB)

[Sideways\_CCI\_Strategy\_-\_Buy.mq5](https://www.mql5.com/en/articles/download/10592/sideways_cci_strategy_-_buy.mq5 "Download Sideways_CCI_Strategy_-_Buy.mq5")(1.19 KB)

[Sideways\_CCI\_Strategy\_-\_Short.mq5](https://www.mql5.com/en/articles/download/10592/sideways_cci_strategy_-_short.mq5 "Download Sideways_CCI_Strategy_-_Short.mq5")(1.19 KB)

[Zero\_crossover\_CCI\_Strategy\_-\_Buy.mq5](https://www.mql5.com/en/articles/download/10592/zero_crossover_cci_strategy_-_buy.mq5 "Download Zero_crossover_CCI_Strategy_-_Buy.mq5")(1.2 KB)

[Zero\_crossover\_CCI\_Strategy\_-\_Short.mq5](https://www.mql5.com/en/articles/download/10592/zero_crossover_cci_strategy_-_short.mq5 "Download Zero_crossover_CCI_Strategy_-_Short.mq5")(1.2 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/392856)**
(7)


![Toto Sarono](https://c.mql5.com/avatar/2022/4/6255B670-31DE.jpg)

**[Toto Sarono](https://www.mql5.com/en/users/totosarono)**
\|
12 Apr 2022 at 22:15

**Mohamed Abdelmaaboud [#](https://www.mql5.com/en/forum/392856#comment_29214767):**

Hello Toto,

Thanks for your comment and nice to meet you again.

First thing, I appreciate that you try applying and testing everything. About your question, the answer is that I wrote the code using PRICE\_CLOSE only and this is one of options of the default CCI on Metatrader 5 and if you changed parameters of the default CCI to be Apply to Close, you will find that it will be the same value. If you preferred to use Typical for this code you can update it to PRICE\_TYPICAL in the code line of CCIDef.

Thanks,

Mohamed

Hi...

Thanks a lot for your explanation, already corrected and worked well. the question is which price better to use? typical or close?

I found it had better additional reference to decide opening position. In this case, RSI band is used whether on upper band for sell and lower band for buy.

It is just to increase to succeed the probability thus of course, increasing our confidence. In my exercised showed that one indicator is not enough to anticipate trend reversal.

Do you have any suggestion to this matter? please let me know. Especially the reference to define uptrend, sideway and downtrend.

Cheers,

Toto S

![Toto Sarono](https://c.mql5.com/avatar/2022/4/6255B670-31DE.jpg)

**[Toto Sarono](https://www.mql5.com/en/users/totosarono)**
\|
12 Apr 2022 at 22:34

**Toto Sarono [#](https://www.mql5.com/en/forum/392856#comment_29271424):**

Hi...

Thanks a lot for your explanation, already corrected and worked well. the question is which price better to use? typical or close?

I found it had better additional reference to decide opening position. In this case, RSI band is used whether on upper band for sell and lower band for buy.

It is just to increase to succeed the probability thus of course, increasing our confidence. In my exercised showed that one indicator is not enough to anticipate trend reversal.

Do you have any suggestion to this matter? please let me know. Especially the reference to define uptrend, sideway and downtrend.

Cheers,

Toto S

For your illustration about the trend questioning and additional indicator, please refer to the attached file.

Thank you.

![John Winsome Munar](https://c.mql5.com/avatar/2022/6/62A73E8B-F64C.jpg)

**[John Winsome Munar](https://www.mql5.com/en/users/trozovka)**
\|
21 May 2022 at 16:28

Thanks for sharing


![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
21 May 2022 at 23:46

**John Winsome Munar [#](https://www.mql5.com/en/forum/392856#comment_39719244):**

Thanks for sharing

You are welcome and thanks for your comment.

![Andres Ngi](https://c.mql5.com/avatar/2021/9/613820C3-FA3E.jpg)

**[Andres Ngi](https://www.mql5.com/en/users/andres_valenzue)**
\|
22 Jul 2022 at 04:34

it's great!


![Data Science and Machine Learning (Part 02): Logistic Regression](https://c.mql5.com/2/48/logistic_regression__1.png)[Data Science and Machine Learning (Part 02): Logistic Regression](https://www.mql5.com/en/articles/10626)

Data Classification is a crucial thing for an algo trader and a programmer. In this article, we are going to focus on one of classification logistic algorithms that can probability help us identify the Yes's or No's, the Ups and Downs, Buys and Sells.

![Mathematics in trading: Sharpe and Sortino ratios](https://c.mql5.com/2/45/math_trading.png)[Mathematics in trading: Sharpe and Sortino ratios](https://www.mql5.com/en/articles/9171)

Return on investments is the most obvious indicator which investors and novice traders use for the analysis of trading efficiency. Professional traders use more reliable tools to analyze strategies, such as Sharpe and Sortino ratios, among others.

![Multiple indicators on one chart (Part 01): Understanding the concepts](https://c.mql5.com/2/44/variety_of_indicators.png)[Multiple indicators on one chart (Part 01): Understanding the concepts](https://www.mql5.com/en/articles/10229)

Today we will learn how to add multiple indicators running simultaneously on one chart, but without occupying a separate area on it. Many traders feel more confident if they monitor multiple indicators at a time (for example, RSI, STOCASTIC, MACD, ADX and some others), or in some cases even at different assets which an index is made of.

![Graphics in DoEasy library (Part 96): Graphics in form objects and handling mouse events](https://c.mql5.com/2/45/MQL5-avatar-doeasy-library3-2.png)[Graphics in DoEasy library (Part 96): Graphics in form objects and handling mouse events](https://www.mql5.com/en/articles/10417)

In this article, I will start creating the functionality for handling mouse events in form objects, as well as add new properties and their tracking to a symbol object. Besides, I will improve the symbol object class since the chart symbols now have new properties to be considered and tracked.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/10592&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069269038468629073)

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