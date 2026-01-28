---
title: Testing and optimization of binary options strategies in MetaTrader 5
url: https://www.mql5.com/en/articles/12103
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:44:57.350387
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=awvffkwltedlobhodqrtvxpqfvgwptvq&ssn=1769157895849305705&ssn_dr=0&ssn_sr=0&fv_date=1769157895&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12103&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Testing%20and%20optimization%20of%20binary%20options%20strategies%20in%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915789575138026&fz_uniq=5062687984634996487&sv=2552)

MetaTrader 5 / Tester


### Introduction

Recently, I became interested in binary options. After surfing the Internet and looking at brokers, it turned out that almost everyone uses their platforms for trading. These platforms do not have the ability to test strategies and have, at best, a meager set of standard indicators. After reviewing a large number of different strategies for binary options, I started wondering how to check these strategies and their optimization. As always, our beloved MetaTrader 5 came to my rescue. As always, I will try to share my experiments with you as simply as possible presenting the material without complex equations and code. But first, a little theory and arguments on whether it is worth delving into binary options or not.

### Theory

Binary option is a digital contract, the subject of which is the forecast of an asset price direction in a selected period of time. The task is to correctly determine one of the two possible scenarios for the development of the market situation: whether the asset will rise or fall. Today, we have a wide selection of online options for currencies, securities and commodities. Practically anything (including a weather forecast) can be an asset. Trading does not require huge investments and deep financial and economic knowledge in the field of market exchange processes.

### Types of binary options

High/Low is the simplest type of binary optionsas traders only have to determine, in which direction the price will go. In a bullish trend, it is advisable to buy High (Call). While Down (Put) is purchased if a downward movement of the asset is expected. The profit of High/Low options varies from 10% to 80% of the bet.

One touch represents an agreement, in which it is required to determine the achievement of the desired price level. The location of quotes during a closure does not matter. The touching of a given level is sufficient. The profitability of "One touch" is higher than usual and may reach 95%, since in most cases it is very difficult to predict them, but you can lose all 100%.

Range determines a price corridor, inside which the value of the asset will be located at the time of expiration. It is also difficult to understand and predict. Profitability is up to 95%.

Below I will test and optimize strategies for High/Low options as the most popular in my opinion.

### Advantages and disadvantages

_The advantages are as follows:_

- Simplicity - you fix the amount of potential losses or profits avoiding the need to make complex calculations of stop loss and take profit levels, which is especially useful if you are a beginner.
- Simple registration on the website of the options broker - no need for a package of documents.
- The widest variety of resources - company stocks, stock indices, oil, gold, cryptocurrencies with the initial capital starting from $100 or even less. Trading on stock or futures markets requires greater amount of the initial deposit.

_Disadvantages:_

- High risks and negative expectation. It takes two wins to make up for one loss. Profitable efficiency up to 80%. A loss-making operation will cost more than a winning one.
- The "Casino" mode brings losses in the long run. Some try to apply tricks from the casino when trading or averaging methods such as Martingale, which violates the rules of proper trading and ultimately leads to a deposit destruction.
- Large commissions.

### Currency pair. Optimization and forward test range. Settings

Below are all optimization and test parameters:

- Forex;
- EURUSD;
- M5, M15, M30, H1;
- Expiration time 5, 15, 30 minutes and 1 hour.
- Optimization range 1 year. 2021.01.28 \- 2022.01.28.
- Forward test range is 1 year. 2022.01.28 - 2023.01.28;
- Initial deposit 10,000;
- Rate 10;
- Interest rate 80%.

### Technical aspects

We need inputs for testing and optimization:

1. StartDepo\- starting deposit;
2. OptionRate - rate;
3. ExpirationTime\- expiration time;
4. ProfitPercent - profitability percentage;
5. TimeFrame -indicator timeframe;
6. Optimization - optimization switch;
7. OptimizationFileName– file name for storing optimization results.

```
input string N0 = "------------Open settings----------------";
input double StartDepo = 10000;
input int OptionRate = 10;
input string N1 = "------------Close settings---------------";
input int ExpirationTime = 1; //ExpirationTime 1=5 min, 2=15 min, 3=30 min, 4=60 min
input double ProfitPercent = 80;
input string N2 = "------------Optimization settings--------";
input int TimeFrame = 1; //TimeFrame 1=5 min, 2=15 min, 3=30 min, 4=60 min
input bool Optimization = false;
input string OptimizationFileName = "Optimization.csv";
input string N3 = "------------Other settings---------------";
input int Slippage = 10;
input int Magic = 111111;
input string EAComment = "2Ma+RSI+Stochastic Oscillator";
```

We will also need variables to store and track changes in the deposit and in the number of profitable and unprofitable option purchases:

1. XStartDepo - for storing the current deposit;
2. Profit - for storing the number of profitableoption purchases;
3. Loss - for storing the number of unprofitable options;

```
double XStartDepo = StartDepo;
int Profit=0;
int Loss=0;
```

To track the position opening time and close the position after the expiration time in MetaTrader 5, we will use the function that returns the position opening time.

```
//+------------------------------------------------------------------+
//| Get open time in positions                                       |
//+------------------------------------------------------------------+
datetime GetTime(string symb="0", int type=-1, int mg=-1,int index=0) {
 datetime p[];
 int c=-1, pr=0;
  if(symb=="0") { symb=Symbol();}
   for(int i=PositionsTotal()-1;i>=0;i--){
      if(position.SelectByIndex(i)) {
     if(position.PositionType()==POSITION_TYPE_BUY || position.PositionType()==POSITION_TYPE_SELL) {
      if((position.Symbol()==symb||symb=="")&&(type<0||position.PositionType()==type)&&(mg<0||position.Magic()==mg)) {
       c++;
       ArrayResize(p, c+1);
       p[c]=position.Time();
       pr=c>=index?index:c;

 }}}}
  return(c==-1?0:p[pr]);
 }
```

To track the profitability of a position and to make a decision whether we won a bet or not, we will use a function that returns the profit of an open position at the current time. Commissions and swaps are not considered.

```
//+------------------------------------------------------------------+
//| Get profit in positions                                          |
//+------------------------------------------------------------------+
double GetProfit(string symb="0", int type=-1, int mg=-1,int index=0) {
 double p[];
 int c=-1, pr=0;
  if(symb=="0") { symb=Symbol();}
   for(int i=PositionsTotal()-1;i>=0;i--){
      if(position.SelectByIndex(i)) {
     if(position.PositionType()==POSITION_TYPE_BUY || position.PositionType()==POSITION_TYPE_SELL) {
      if((position.Symbol()==symb||symb=="")&&(type<0||position.PositionType()==type)&&(mg<0||position.Magic()==mg)) {
       c++;
       ArrayResize(p, c+1);
       p[c]=position.Profit();
       pr=c>=index?index:c;

 }}}}
  return(c==-1?0:p[pr]);
 }
```

Apply several triggers in order to handle only a new incoming signal. In other words, permission to enter occurs only with a new signal. Repeated entries on the same signal are ignored:

1. TrigerSell - buy Put option;
2. TrigerBuy - buy Call option;

```
int TrigerSell=0;
int TrigerBuy=0;
```

Exiting a position, adding and subtracting funds from the initial deposit, as well as calculating the profitable and unprofitable purchases of options is carried out by tracking the current time relative to the opening of the position and the expiration time.

```
//Sell (Put)

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment);
   }

//Buy (Call)

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment);
   }
```

For optimization, use variables to determine the desired timeframe and expiration time. The optimization mode is enabled by the Optimization variable. The essence of optimization comes down to the use of files like CSV. After testing is completed in OnDeinit, write all the necessary variables to the file.When optimizing, Expert Advisors create a CSV file with the result at "C:\\Users\\Your username\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files".

1. Final deposit;
2. Number of profitable trades;
3. Number of losing trades;
4. Timeframe;
5. Expiration time.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {

if (Optimization==true){
 if (FileIsExist(OptimizationFileName)==false){
   filehandle = FileOpen(OptimizationFileName,FILE_WRITE|FILE_READ|FILE_CSV|FILE_COMMON|FILE_ANSI, ";");
    if(filehandle!=INVALID_HANDLE)
     {
      FileSeek(filehandle, 0, SEEK_END);
      FileWrite(filehandle,DoubleToString(XStartDepo),IntegerToString(Profit),IntegerToString(Loss),IntegerToString(XTimeFrame),IntegerToString(XExpirationTime));
      FileClose(filehandle);
     }
 }
}

  }
```

You can watch testing in visualization mode. Let's display the current result using Comment() for simplicity.

![Current result](https://c.mql5.com/2/51/Res.png)

### Strategies

**2Ma+RSI+Stochastic Oscillator strategy**

The strategy is offered by one well-known broker as a scalping strategy for binary options. Suggested timeframe is 5 minutes. Expiration time is 5 minutes.

Indicators:

1. A pair of exponential moving averages with periods of 5 and 10;
2. RSI, default settings;
3. Stochastic Oscillator with the setting values of 14, 3, 3.

The "up" signal (or buying a Call option) when a number of conditions are met:

1. The red moving average has crossed the blue moving average upwards;
2. RSI is above 50;
3. The fast Stochastic line crossed the slow (dotted) line upwards.

![Upward strategy 1](https://c.mql5.com/2/51/Buy.png)

The "down" signal (buying a Put option) is formed if certain factors are present:

1. The red MA crossed the blue one downwards;
2. RSI is located below 50;
3. The fast line of the Stochastic oscillator crosses the slow one downwards.

![Downward strategy 1](https://c.mql5.com/2/51/Sell.png)

I have selected the intersection of MA in the opposite direction as a trigger in this new signal strategy. The strategy code is displayed below.

```
//Sell (Put)

if((ind_In1S1[1]>ind_In2S1[1]) && (ind_In1S1[2]>ind_In2S1[2])){TrigerSell=1;}

   if ((TrigerSell==1) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (ind_In1S1[1]<ind_In2S1[1]) && (ind_In1S1[0]<ind_In2S1[0]) && (ind_In4S1[1]<50) && (ind_In4S1[0]<50) && (ind_In3S1_1[1]<ind_In3S1_2[1])){
   OpenSell(symbolS1.Name(), 0.01, 0, 0, EAComment);
   TrigerSell=0;
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment);
   }

//Buy (Call)

if((ind_In1S1[1]<ind_In2S1[1]) && (ind_In1S1[2]<ind_In2S1[2])){TrigerBuy=1;}

   if ((TrigerBuy==1) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (ind_In1S1[1]>ind_In2S1[1]) && (ind_In1S1[0]>ind_In2S1[0]) && (ind_In4S1[1]>50) && (ind_In4S1[0]>50) && (ind_In3S1_1[1]>ind_In3S1_2[1])){
   OpenBuy(symbolS1.Name(), 0.01, 0, 0, EAComment);
   TrigerBuy=0;
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment);
   }
```

The test yields the following results:

- Final deposit 1964;
- Profitable trades 1273;
- Losing trades 1822.

![Result](https://c.mql5.com/2/51/Res__1.png)

After optimization, we get the result. As we can see, we did not get an increase to our deposit on any timeframe and expiration time.

![Optimization result](https://c.mql5.com/2/51/Opt_res.png)

**Maverick strategy**

This is an interesting and simple option strategy named Maverick. It is based on two technical analysis indicators. It is said to determine the trade entry and exit points quite accurately. Trading is offered on a timeframe of 1-5 minutes. In order to receive more signals, we can open several charts with different assets at the same time.

Indicators:

1. Bollinger Bands 20 and StDev (standard deviation) 2;
2. RSI indicator. RSI parameters – 4 with the borders of 80 and 20.

Growth forecast. Buying a Call option:

As soon as RSI indicator line enters the oversold area below 20, while the price line touches or goes beyond the Bollinger Bands, wait for the first bullish candle and enter an UPWARD trade.

![Buying upward](https://c.mql5.com/2/51/Buy__1.png)

Fall forecast. Buying a Put option:

After the RSI indicator line has entered the overbought area above 80, and the price line has gone beyond the upper limits of the Bollinger, wait for the first bearish candle and open a DOWNWARD trade.

![Buying downward](https://c.mql5.com/2/51/Sell__1.png)

I chose the closing of the previous candle inside the Bollinger indicator channel as a new signal trigger. The strategy code is displayed below.

```
//Sell (Put)

if(iClose(symbolS1.Name(),XTimeFrame,1)<ind_In1S1_1[1]){TrigerSell=1;}

   if ((TrigerSell==1) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (iClose(symbolS1.Name(),XTimeFrame,2)>ind_In1S1_1[2]) && (ind_In2S1[2]>80) && (iClose(symbolS1.Name(),XTimeFrame,1)<iOpen(symbolS1.Name(),XTimeFrame,1))){
   OpenSell(symbolS1.Name(), 0.01, 0, 0, EAComment);
   TrigerSell=0;
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment);
   }

//Buy (Call)

if(iClose(symbolS1.Name(),XTimeFrame,1)>ind_In1S1_2[1]){TrigerBuy=1;}

   if ((TrigerBuy==1) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (iClose(symbolS1.Name(),XTimeFrame,2)<ind_In1S1_2[2]) && (ind_In2S1[2]<20) && (iClose(symbolS1.Name(),XTimeFrame,1)>iOpen(symbolS1.Name(),XTimeFrame,1))){
   OpenBuy(symbolS1.Name(), 0.01, 0, 0, EAComment);
   TrigerBuy=0;
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment);
   }
```

Let's start the check. Let's conduct a test on M5. Get the result:

- Final deposit 3312;
- Profitable trades 1589;
- Losing trades 1940.

![Result](https://c.mql5.com/2/51/Res__2.png)

We have suffered a loss. Although the result is slightly better than in the previous strategy. Let's perform optimization in the hope of positive deposit values. Alas, the strategy does not work:

![Optimization result](https://c.mql5.com/2/51/Opt_res__1.png)

**Vortex+TSI strategy**

The strategy for binary options is called Vortex due to the use of the indicator of the same name. In fact, the strategy involves two indicators - the main one and the second one as a filter. The suggested timeframe is 1-5 minutes.

Indicators:

1. Vortex 14;
2. True Strength Indicator (TSI) 25, 13, 5, Exponential.

For buying a Call option:

1. Wait for the simultaneous intersection of the lines of both indicators, when the blue lines are at the top, and the red ones go down;
2. Vortex indicator lines should diverge.

![Growth](https://c.mql5.com/2/51/Buy__2.png)

For buying a Put option:

1. Wait for the simultaneous intersection of the lines of both indicators, when the red lines are at the top, and the blue ones go down;
2. Vortex indicator lines should diverge.

![Fall](https://c.mql5.com/2/51/Sell__2.png)

A reverse crossing of the True Strength Indicator serves as a new signal trigger. The strategy code is displayed below.

```
//Sell (Put)

if(ind_In1S1_1[1]>ind_In1S1_2[1]){TrigerSell=1;}

   if ((TrigerSell==1) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (ind_In1S1_1[0]<ind_In1S1_2[0]) && (ind_In1S1_1[1]<ind_In1S1_2[1]) && (ind_In2S1_1[0]<ind_In2S1_2[0]) && (ind_In2S1_1[1]<ind_In2S1_2[1]) && (ind_In2S1_1[0]<ind_In2S1_1[1]) && (ind_In2S1_2[0]>ind_In2S1_2[1]) && (ind_In2S1_1[1]<ind_In2S1_1[2]) && (ind_In2S1_2[1]>ind_In2S1_2[2])){
   OpenSell(symbolS1.Name(), 0.01, 0, 0, EAComment);
   TrigerSell=0;
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment);
   }

//Buy (Call)

if(ind_In1S1_1[1]<ind_In1S1_2[1]){TrigerBuy=1;}

   if ((TrigerBuy==1) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (ind_In1S1_1[0]>ind_In1S1_2[0]) && (ind_In1S1_1[1]>ind_In1S1_2[1]) && (ind_In2S1_1[0]>ind_In2S1_2[0]) && (ind_In2S1_1[1]>ind_In2S1_2[1]) && (ind_In2S1_1[0]>ind_In2S1_1[1]) && (ind_In2S1_2[0]<ind_In2S1_2[1]) && (ind_In2S1_1[1]>ind_In2S1_1[2]) && (ind_In2S1_2[1]<ind_In2S1_2[2])){
   OpenBuy(symbolS1.Name(), 0.01, 0, 0, EAComment);
   TrigerBuy=0;
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment);
   }
```

It is time to start testing our indicator strategy. Let's conduct a test on M5. Get the result:

- Final deposit -3118;
- Profitable trades 1914;
- Losing trades 2843.

The number of losing trades is greater. The final deposit went negative. Perhaps optimization will help.

![Result](https://c.mql5.com/2/51/Res__3.png)

The optimization results speak for themselves:

![Optimization result](https://c.mql5.com/2/51/Opt_res__2.png)

After testing about 10 more different strategies without the desired result, I decided to develop my own strategy. Based on the experience gained, I decided to try a different approach. The strategy is named New.

**New strategy**

I will use oversold and overbought situations as the central part of the strategy. I have also decided to add averaging. Although I am against such a decision, it is still interesting to check out the results.

Indicators:

Envelopes14, 0, 0,1 Simplemethod is applied to Close.

"Upward" signal or buying a Call option:

1. The Ask price is less than the lower line of the Envelopes indicator;
2. The distance in points from the bottom line of the Envelopes indicator to the Ask price is greater than the Distance value.

![Upward](https://c.mql5.com/2/51/Buy__3.png)

"Downward" signal or buying a Put option:

1. The Bid price is higher than the upper line of the Envelopes indicator;
2. The distance in points from the Bid price to the Envelopes indicator line is greater than the Distance value.

![Downward](https://c.mql5.com/2/51/Sell__3.png)

No trigger is used in the strategy. A new entry signal is constantly monitored if the conditions are met and there is no open trade. Below is the strategy code without using averaging:

```
//Sell (Put)

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (BidS1>(ind_In1S1_1[0]+(Distance*PointS1)))){
   OpenSell(symbolS1.Name(), 0.01, 0, 0, EAComment);
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment);
   }

//Buy (Call)

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (AskS1<(ind_In1S1_2[0]-(Distance*PointS1)))){
   OpenBuy(symbolS1.Name(), 0.01, 0, 0, EAComment);
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)>0){
     XStartDepo=XStartDepo+((OptionRate*ProfitPercent)/100);
     Profit++;
    }
    else{
     XStartDepo=XStartDepo-OptionRate;
     Loss++;
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment);
   }
```

Let's test on M5. Distance=150. The test result without using averaging is in the screenshot below:

![Result 1](https://c.mql5.com/2/51/Res_1.png)

Final deposit 8952; Profitable trades 874;

Losing trades 804.

There are more profitable entries than unprofitable ones, but we have lost part of the deposit. This is the negative expectation. With this approach, the maximum series of losing entries in a row was two, which I think will play into our hands when averaging.

The result of optimization without using averaging is in the screenshot below. It was decided to include only profitable situations in the report for convenience and clarity:

![Optimization result 1](https://c.mql5.com/2/51/Opt_res_1.png)

Let's add averaging. I decided to limit the number of averaging entries. To do this, I introduced a new parameter Averaging, which is 4 by default. The averaging strategy code is provided below:

```
//Sell (Put)

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (BidS1>(ind_In1S1_1[0]+(Distance*PointS1)))){
   OpenSell(symbolS1.Name(), 0.01, 0, 0, EAComment);
   Print(((XOptionRate*LossIn)*LossIn));
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)>0){
     XStartDepo=XStartDepo+(((XOptionRate*LossIn)*ProfitPercent)/100);
     Profit++;
     LossIn=1;
    }
    else{
     XStartDepo=XStartDepo-(XOptionRate*LossIn);
     Loss++;
     LossIn++;
     if(LossIn>Averaging){LossIn=1;}
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment);
   }

//Buy (Call)

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (AskS1<(ind_In1S1_2[0]-(Distance*PointS1)))){
   OpenBuy(symbolS1.Name(), 0.01, 0, 0, EAComment);
   Print(((XOptionRate*LossIn)*LossIn));
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)>0){
     XStartDepo=XStartDepo+(((XOptionRate*LossIn)*ProfitPercent)/100);
     Profit++;
     LossIn=1;
    }
    else{
     XStartDepo=XStartDepo-(XOptionRate*LossIn);
     Loss++;
     LossIn++;
     if(LossIn>Averaging){LossIn=1;}
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment);
   }
```

Test results with averaging. Finally, success:

![Averaging 1](https://c.mql5.com/2/51/Profit_1.png)

Optimization results with averaging:

![Optimization result 1](https://c.mql5.com/2/51/Opt_res_1__1.png)

The results of a more interesting option with multiplying the rate after losing trades. The code is provided below:

```
//Sell (Put)

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (BidS1>(ind_In1S1_1[0]+(Distance*PointS1)))){
   OpenSell(symbolS1.Name(), 0.01, 0, 0, EAComment);
   Print(((XOptionRate*LossIn)*LossIn));
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_SELL, Magic, 0)>0){
     XStartDepo=XStartDepo+((((XOptionRate*LossIn)*LossIn)*ProfitPercent)/100);
     Profit++;
     LossIn=1;
    }
    else{
     XStartDepo=XStartDepo-((XOptionRate*LossIn)*LossIn);
     Loss++;
     LossIn++;
     if(LossIn>Averaging){LossIn=1;}
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment);
   }

//Buy (Call)

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (AskS1<(ind_In1S1_2[0]-(Distance*PointS1)))){
   OpenBuy(symbolS1.Name(), 0.01, 0, 0, EAComment);
   Print(((XOptionRate*LossIn)*LossIn));
   }

   if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)>0) && (TimeCurrent()>=(GetTime(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)+(XExpirationTime*60)))){
    if(GetProfit(symbolS1.Name(), POSITION_TYPE_BUY, Magic, 0)>0){
     XStartDepo=XStartDepo+((((XOptionRate*LossIn)*LossIn)*ProfitPercent)/100);
     Profit++;
     LossIn=1;
    }
    else{
     XStartDepo=XStartDepo-((XOptionRate*LossIn)*LossIn);
     Loss++;
     LossIn++;
     if(LossIn>Averaging){LossIn=1;}
    }
   Comment("Depo = ",XStartDepo," Profit = ",Profit," Loss = ",Loss);
   ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment);
   }
```

Results of optimizing an interesting variant:

![Optimization result 2](https://c.mql5.com/2/51/Opt_res_2.png)

### Conclusion

What conclusions can be drawn from all the work done? 99% of all indicator strategies practically do not work. By changing the approach a little, it is possible to develop a good profitable strategy. It is necessary to check everything before entering the market. As always, MetaTrader 5 will help us with this.By the way, there are already options with 100% profitability, which gives additional food for thought. You can download all considered strategies in the form of Expert Advisors in the attached archive.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12103](https://www.mql5.com/ru/articles/12103)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12103.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/12103/mql5.zip "Download MQL5.zip")(24.76 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Experiments with neural networks (Part 7): Passing indicators](https://www.mql5.com/en/articles/13598)
- [Experiments with neural networks (Part 6): Perceptron as a self-sufficient tool for price forecast](https://www.mql5.com/en/articles/12515)
- [Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://www.mql5.com/en/articles/12459)
- [Experiments with neural networks (Part 4): Templates](https://www.mql5.com/en/articles/12202)
- [Experiments with neural networks (Part 3): Practical application](https://www.mql5.com/en/articles/11949)
- [Experiments with neural networks (Part 2): Smart neural network optimization](https://www.mql5.com/en/articles/11186)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/444366)**
(14)


![sydiya63](https://c.mql5.com/avatar/avatar_na2.png)

**[sydiya63](https://www.mql5.com/en/users/sydiya63)**
\|
20 Sep 2023 at 11:02

Very interesting. I've been sitting on binary for about 10 years. What only strategies I have not invented. Any standard indicators do not work IMHO. I think you need to dig into candlestick [analysis](https://www.mql5.com/en/articles/5630 "Article: Research of candlestick analysis methods (Part II): Auto search for new patterns "), candle body, shadows. So far in the minus.


![HeAic](https://c.mql5.com/avatar/2017/4/5901CB2B-E197.jpg)

**[HeAic](https://www.mql5.com/en/users/heaic)**
\|
3 Feb 2024 at 19:29

For binary options (3min. expiry time) once successfully !!! used standard [OsMA indicator](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/mao "MetaTrader 5 Help: Moving Average of Oscillator indicator").

The only problem - the accuracy of entry was not enough due to the lack of 30sec timeframe in MT4 :)

Look for market imbalance - search for the beginning of movement towards balance, this signal is always fulfilled ;)

![maxvoronin74](https://c.mql5.com/avatar/avatar_na2.png)

**[maxvoronin74](https://www.mql5.com/en/users/maxvoronin74)**
\|
15 Apr 2024 at 09:07

Tried your [concatenation](https://www.mql5.com/en/docs/matrix/matrix_manipulations/matrix_concat " MQL5 Documentation: function Concat")

```
   int         total            = PositionsTotal();
   for(int i=total-1; i>=0; i--)
   {
      int    position_magic        = position.Magic();
      string position_symbol       = position.Symbol(); /*PositionGetString(POSITION_SYMBOL)*/;
      ulong  position_ticket       = position.SelectByIndex(i);
```

MetaEditor does not pass: 'position' - undeclared identifier

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
15 Apr 2024 at 09:50

Do you have this part of the code?

```
#include <Trade\PositionInfo.mqh>
CPositionInfo  position;
```

If not, insert it right after

```
#property version   "1.000"
```

Regards, Vladimir.

![maxvoronin74](https://c.mql5.com/avatar/avatar_na2.png)

**[maxvoronin74](https://www.mql5.com/en/users/maxvoronin74)**
\|
15 Apr 2024 at 15:33

**MrBrooklin [#](https://www.mql5.com/ru/forum/441494#comment_53055147):**

Do you have that part of the code?

If not, insert it right after

Regards, Vladimir.

Thank you. It worked.


![Neural networks made easy (Part 34): Fully Parameterized Quantile Function](https://c.mql5.com/2/50/Neural_Networks_Made_Easy_quantile-parameterized_avatar.png)[Neural networks made easy (Part 34): Fully Parameterized Quantile Function](https://www.mql5.com/en/articles/11804)

We continue studying distributed Q-learning algorithms. In previous articles, we have considered distributed and quantile Q-learning algorithms. In the first algorithm, we trained the probabilities of given ranges of values. In the second algorithm, we trained ranges with a given probability. In both of them, we used a priori knowledge of one distribution and trained another one. In this article, we will consider an algorithm which allows the model to train for both distributions.

![Learn how to design a trading system by Fibonacci](https://c.mql5.com/2/52/learnhow_trading_system_fibonacci_avatar.png)[Learn how to design a trading system by Fibonacci](https://www.mql5.com/en/articles/12301)

In this article, we will continue our series of creating a trading system based on the most popular technical indicator. Here is a new technical tool which is the Fibonacci and we will learn how to design a trading system based on this technical indicator.

![Data Science and Machine Learning(Part 14): Finding Your Way in the Markets with Kohonen Maps](https://c.mql5.com/2/52/data_science_ml_kohonen_maps_avatar.png)[Data Science and Machine Learning(Part 14): Finding Your Way in the Markets with Kohonen Maps](https://www.mql5.com/en/articles/12261)

Are you looking for a cutting-edge approach to trading that can help you navigate complex and ever-changing markets? Look no further than Kohonen maps, an innovative form of artificial neural networks that can help you uncover hidden patterns and trends in market data. In this article, we'll explore how Kohonen maps work, and how they can be used to develop smarter, more effective trading strategies. Whether you're a seasoned trader or just starting out, you won't want to miss this exciting new approach to trading.

![Data Science and Machine Learning (Part 13): Improve your financial market analysis with Principal Component Analysis (PCA)](https://c.mql5.com/2/52/pca_avatar.png)[Data Science and Machine Learning (Part 13): Improve your financial market analysis with Principal Component Analysis (PCA)](https://www.mql5.com/en/articles/12229)

Revolutionize your financial market analysis with Principal Component Analysis (PCA)! Discover how this powerful technique can unlock hidden patterns in your data, uncover latent market trends, and optimize your investment strategies. In this article, we explore how PCA can provide a new lens for analyzing complex financial data, revealing insights that would be missed by traditional approaches. Find out how applying PCA to financial market data can give you a competitive edge and help you stay ahead of the curve

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/12103&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062687984634996487)

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