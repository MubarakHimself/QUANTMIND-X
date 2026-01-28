---
title: Trademinator 3: Rise of the Trading Machines
url: https://www.mql5.com/en/articles/350
categories: Trading, Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:39:46.379465
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/350&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083022472938263408)

MetaTrader 5 / Trading


**Prologue**

Once upon a time on a faraway forum ( [MQL5](https://www.mql5.com/en/forum)) two articles: " [Genetic Algorithms - It's Easy!](https://www.mql5.com/en/articles/55)" by [joo](https://www.mql5.com/en/users/joo) and " [Dr. Tradelove...](https://www.mql5.com/en/articles/334)" by [me](https://www.mql5.com/en/users/Rich) were published. In the first article the author equipped us with a powerful tool for optimizing whatever you need, including trading strategies - a genetic algorithm implemented by means of the [MQL5](https://www.mql5.com/en/docs) language.

Using this algorithm, in the second article I tried to develop a self-optimizing Expert Advisor based on it. The article ended with the formulation of the following task: to create an Expert Advisor (self-optimizing, of course), which can not only select the best parameters for a particular trading system, but also choose the best strategy of all developed strategies. Let's see whether it is possible, and if it is, then how.

**Tales of Trading Robots**

First, we formulate the general requirements for a self-optimizing Expert Advisor.

It should be able to (based on historical data):

- select the best strategy of described ones
- choose the best financial instrument
- choose the best deposit size for trading with leverage correction
- choose the best parameters of indicators in the selected strategy

Further, in real life, it should be able to:

- open and close positions
- choose the size of the position
- decide on whether a new optimization is needed

The below figure shows a schematic diagram of the proposed Expert Advisor.

![](https://c.mql5.com/2/3/image_en.png)

A detailed scheme with bounds is in the attached file Scheme\_en.

Keeping in mind that it is impossible to grasp the immensity, we introduce restrictions in the Expert Advisor logic. We agree that (IMPORTANT):

1. The Expert Advisor will make trade decisions upon the occurrence of a [new bar](https://www.mql5.com/en/articles/159 "Article: \"New Bar\" Event Handler") (on any timeframe that we select).
2. On the basis of p.1, but not limited to, the Expert Advisor will close trades only on the indicator signals not using Take Profit and Stop Loss and, accordingly, not using Trailing Stop.
3. The condition to start a new optimization: a drawdown of the balance is higher than the preset value during initialization of the level. Please note that this is my personal condition, and each of you can select your specific condition.
4. A fitness function models trading on the history and maximizes the modeled balance, provided that the relative drawdown of the balance of the simulated trades is below a certain preset level. Also note that this is my personal fitness function, and you can select your specific one.
5. We limit the number of parameters to be optimized, except for the three general ones (strategy, instrument and deposit share), to five for the parameters of indicator buffers. This limitation follows logically from [the maximum number of indicator buffers](https://www.mql5.com/en/docs/indicators) for built-in technical indicators. If you are going to describe the strategies that use custom indicators with a large number of indicator buffers, simply change the OptParamCount variable in the main.mq5 file to the desired amount.

Now that the requirements are specified and limitations selected, you can look at the code that implements all this.

Let's start with the function, where everything runs.

```
void OnTick()
{
  if(isNewBars()==true)
  {
    trig=false;
    switch(strat)
    {
      case  0: {trig=NeedCloseMA()   ; break;};                      //The number of case strings must be equal to the number of strategies
      case  1: {trig=NeedCloseSAR()  ; break;};
      case  2: {trig=NeedCloseStoch(); break;};
      default: {trig=NeedCloseMA()   ; break;};
    }
    if(trig==true)
    {
      if(GetRelDD()>maxDD)                                           //If a balance drawdown is above the max allowed value:
      {
        GA();                                                        //Call the genetic optimization function
        GetTrainResults();                                           //Get the optimized parameters
        maxBalance=AccountInfoDouble(ACCOUNT_BALANCE);               //Now count the drawdown not from the balance maximum...
                                                                     //...but from the current balance
      }
    }
    switch(strat)
    {
      case  0: {trig=NeedOpenMA()   ; break;};                       //The number of case strings must be equal to the number of strategies
      case  1: {trig=NeedOpenSAR()  ; break;};
      case  2: {trig=NeedOpenStoch(); break;};
      default: {trig=NeedOpenMA()   ; break;};
    }
    Print(TimeToString(TimeCurrent()),";","Main:OnTick:isNewBars(true)",
          ";","strat=",strat);
  }
}
```

What is here? As drawn in the diagram, we look at each tick, whether there is a new bar. If there is, then, knowing what strategy is now chosen, we call its specific function for checking if there is an open position and close it, if necessary. Suppose now the best breakthrough strategy is SAR, respectively, the NeedCloseSAR function will be called:

```
bool NeedCloseSAR()
{
  CopyBuffer(SAR,0,0,count,SARBuffer);
  CopyOpen(s,tf,0,count,o);
  Print(TimeToString(TimeCurrent()),";","StrategySAR:NeedCloseSAR",
        ";","SAR[0]=",SARBuffer[0],";","SAR[1]=",SARBuffer[1],";","Open[0]=",o[0],";","Open[1]=",o[1]);
  if((SARBuffer[0]>o[0]&&SARBuffer[1]<o[1])||
     (SARBuffer[0]<o[0]&&SARBuffer[1]>o[1]))
  {
    if(PositionsTotal()>0)
    {
      ClosePosition();
      return(true);
    }
  }
  return(false);
}
```

Any position closing function must be boolean and return true when closing a position. This allows the next code block of the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) function to decide on whether a new optimization is needed:

```
    if(trig==true)
    {
      if(GetRelDD()>maxDD)                                           //If the balance drawdown is above the max allowed one:
      {
        GA();                                                        //Call the genetic optimization function
        GetTrainResults();                                           //Get optimized parameters
        maxBalance=AccountInfoDouble(ACCOUNT_BALANCE);                   //Now count the drawdown not from the balance maximum...
                                                                     //...but from the current balance
      }
    }
```

Get the current balance drawdown and compare it with the maximum allowed one. If it has exceeded the max value, run a new optimization (GA()). The GA() function, in turn, calls the heart of the Expert Advisor - the fitness function FitnessFunction(int chromos) of the GAModule.mqh module:

```
void FitnessFunction(int chromos)                                    //A fitness function for the genetic optimizer:...
                                                                     //...selects a strategy, symbol, deposit share,...
                                                                     //...parameters of indicator buffers;...
                                                                     //...you can optimize whatever you need, but...
                                                                     //...watch carefully the number of genes
{
  double ff=0.0;                                                     //The fitness function
  strat=(int)MathRound(Colony[GeneCount-2][chromos]*StratCount);     //GA selects a strategy
 //For EA testing mode use the following code...
  z=(int)MathRound(Colony[GeneCount-1][chromos]*3);                  //GA selects a symbol
  switch(z)
  {
    case  0: {s="EURUSD"; break;};
    case  1: {s="GBPUSD"; break;};
    case  2: {s="USDCHF"; break;};
    case  3: {s="USDJPY"; break;};
    default: {s="EURUSD"; break;};
  }
//..for real mode, comment the previous code and uncomment the following one (symbols are selected in the MarketWatch window)
/*
  z=(int)MathRound(Colony[GeneCount-1][chromos]*(SymbolsTotal(true)-1));//GA selects a symbol
  s=SymbolName(z,true);
*/
  optF=Colony[GeneCount][chromos];                                   //GA selects a deposit share
  switch(strat)
  {
    case  0: {ff=FFMA(   Colony[1][chromos],                         //The number of case strings must be equal to the number of strategies
                         Colony[2][chromos],
                         Colony[3][chromos],
                         Colony[4][chromos],
                         Colony[5][chromos]); break;};
    case  1: {ff=FFSAR(  Colony[1][chromos],
                         Colony[2][chromos],
                         Colony[3][chromos],
                         Colony[4][chromos],
                         Colony[5][chromos]); break;};
    case  2: {ff=FFStoch(Colony[1][chromos],
                         Colony[2][chromos],
                         Colony[3][chromos],
                         Colony[4][chromos],
                         Colony[5][chromos]); break;};
    default: {ff=FFMA(   Colony[1][chromos],
                         Colony[2][chromos],
                         Colony[3][chromos],
                         Colony[4][chromos],
                         Colony[5][chromos]); break;};
  }
  AmountStartsFF++;
  Colony[0][chromos]=ff;
  Print(TimeToString(TimeCurrent()),";","GAModule:FitnessFunction",
        ";","strat=",strat,";","s=",s,";","optF=",optF,
        ";",Colony[1][chromos],";",Colony[2][chromos],";",Colony[3][chromos],";",Colony[4][chromos],";",Colony[5][chromos]);
}
```

Depending on the currently selected strategy, the fitness function calculation module, that is specific to a particular strategy, is called. For example, the GA has chosen a stochastic, FFStoch () will be called, and optimizing parameters of indicator buffers will be transfered to it:

```
double FFStoch(double par1,double par2,double par3,double par4,double par5)
{
  int    b;
  bool   FFtrig=false;                                               //Is there an open position?
  string dir="";                                                     //Direction of the open position
  double OpenPrice;                                                  //Position Open price
  double t=cap;                                                      //Current balance
  double maxt=t;                                                     //Maximum balance
  double aDD=0.0;                                                    //Absolute drawdown
  double rDD=0.000001;                                               //Relative drawdown
  Stoch=iStochastic(s,tf,(int)MathRound(par1*MaxStochPeriod)+1,
                         (int)MathRound(par2*MaxStochPeriod)+1,
                         (int)MathRound(par3*MaxStochPeriod)+1,MODE_SMA,STO_CLOSECLOSE);
  StochTopLimit   =par4*100.0;
  StochBottomLimit=par5*100.0;
  dig=MathPow(10.0,(double)SymbolInfoInteger(s,SYMBOL_DIGITS));
  leverage=AccountInfoInteger(ACCOUNT_LEVERAGE);
  contractSize=SymbolInfoDouble(s,SYMBOL_TRADE_CONTRACT_SIZE);
  b=MathMin(Bars(s,tf)-1-count-MaxMAPeriod,depth);
  for(from=b;from>=1;from--)                                         //Where to start copying of history
  {
    CopyBuffer(Stoch,0,from,count,StochBufferMain);
    CopyBuffer(Stoch,1,from,count,StochBufferSignal);
    if((StochBufferMain[0]>StochBufferSignal[0]&&StochBufferMain[1]<StochBufferSignal[1])||
       (StochBufferMain[0]<StochBufferSignal[0]&&StochBufferMain[1]>StochBufferSignal[1]))
    {
      if(FFtrig==true)
      {
        if(dir=="BUY")
        {
          CopyOpen(s,tf,from,count,o);
          if(t>0) t=t+t*optF*leverage*(o[1]-OpenPrice)*dig/contractSize; else t=0;
          if(t>maxt) {maxt=t; aDD=0;} else if((maxt-t)>aDD) aDD=maxt-t;
          if((maxt>0)&&(aDD/maxt>rDD)) rDD=aDD/maxt;
        }
        if(dir=="SELL")
        {
          CopyOpen(s,tf,from,count,o);
          if(t>0) t=t+t*optF*leverage*(OpenPrice-o[1])*dig/contractSize; else t=0;
          if(t>maxt) {maxt=t; aDD=0;} else if((maxt-t)>aDD) aDD=maxt-t;
          if((maxt>0)&&(aDD/maxt>rDD)) rDD=aDD/maxt;
        }
        FFtrig=false;
      }
   }
    if(StochBufferMain[0]>StochBufferSignal[0]&&StochBufferMain[1]<StochBufferSignal[1]&&StochBufferMain[1]>StochTopLimit)
    {
      CopyOpen(s,tf,from,count,o);
      OpenPrice=o[1];
      dir="SELL";
      FFtrig=true;
    }
    if(StochBufferMain[0]<StochBufferSignal[0]&&StochBufferMain[1]>StochBufferSignal[1]&&StochBufferMain[1]<StochBottomLimit)
    {
      CopyOpen(s,tf,from,count,o);
      OpenPrice=o[1];
      dir="BUY";
      FFtrig=true;
    }
  }
  Print(TimeToString(TimeCurrent()),";","StrategyStoch:FFStoch",
        ";","K=",(int)MathRound(par1*MaxStochPeriod)+1,";","D=",(int)MathRound(par2*MaxStochPeriod)+1,
        ";","Slow=",(int)MathRound(par3*MaxStochPeriod)+1,";","TopLimit=",StochTopLimit,";","BottomLimit=",StochBottomLimit,
        ";","rDD=",rDD,";","Cap=",t);
  if(rDD<=trainDD) return(t); else return(0.0);
}
```

The fitness function of the stochastic returns a simulated balance to the main function, which will pass it to the genetic algorithm. At some point in time the GA decides to end the optimization, and using the GetTrainResults() function, we return the best current values of the strategy (for example - moving averages), symbol, the deposit share and parameters of the indicator buffers to the basic program, as well as create indicators for further real trading:

```
void GetTrainResults()                                               //Get the best parameters
{
  strat=(int)MathRound(Chromosome[GeneCount-2]*StratCount);          //Remember the best strategy
//For EA testing mode use the following code...
  z=(int)MathRound(Chromosome[GeneCount-1]*3);                       //Remember the best symbol
  switch(z)
  {
    case  0: {s="EURUSD"; break;};
    case  1: {s="GBPUSD"; break;};
    case  2: {s="USDCHF"; break;};
    case  3: {s="USDJPY"; break;};
    default: {s="EURUSD"; break;};
  }
//...for real mode, comment the previous code and uncomment the following one (symbols are selected in the MarketWatch window)
/*
  z=(int)MathRound(Chromosome[GeneCount-1]*(SymbolsTotal(true)-1));  //Remember the best symbol
  s=SymbolName(z,true);
*/
  optF=Chromosome[GeneCount];                                        //Remember the best deposit share
  switch(strat)
  {
    case  0: {GTRMA(   Chromosome[1],                                //The number of case strings must be equal to the number of strategies
                       Chromosome[2],
                       Chromosome[3],
                       Chromosome[4],
                       Chromosome[5]) ; break;};
    case  1: {GTRSAR(  Chromosome[1],
                       Chromosome[2],
                       Chromosome[3],
                       Chromosome[4],
                       Chromosome[5]) ; break;};
    case  2: {GTRStoch(Chromosome[1],
                       Chromosome[2],
                       Chromosome[3],
                       Chromosome[4],
                       Chromosome[5]) ; break;};
    default: {GTRMA(   Chromosome[1],
                       Chromosome[2],
                       Chromosome[3],
                       Chromosome[4],
                       Chromosome[5]) ; break;};
  }
  Print(TimeToString(TimeCurrent()),";","GAModule:GetTrainResults",
        ";","strat=",strat,";","s=",s,";","optF=",optF,
        ";",Chromosome[1],";",Chromosome[2],";",Chromosome[3],";",Chromosome[4],";",Chromosome[5]);
}

void GTRMA(double par1,double par2,double par3,double par4,double par5)
{
  MAshort=iMA(s,tf,(int)MathRound(par1*MaxMAPeriod)+1,0,MODE_SMA,PRICE_OPEN);
  MAlong =iMA(s,tf,(int)MathRound(par2*MaxMAPeriod)+1,0,MODE_SMA,PRICE_OPEN);
  CopyBuffer(MAshort,0,from,count,ShortBuffer);
  CopyBuffer(MAlong, 0,from,count,LongBuffer );
  Print(TimeToString(TimeCurrent()),";","StrategyMA:GTRMA",
        ";","MAL=",(int)MathRound(par2*MaxMAPeriod)+1,";","MAS=",(int)MathRound(par1*MaxMAPeriod)+1);
}
```

Now it all is back to the place where everything is running (OnTick()): knowing what strategy is now the best one, it is checked whether it is time to go to the market:

```
bool NeedOpenMA()
{
  CopyBuffer(MAshort,0,0,count,ShortBuffer);
  CopyBuffer(MAlong, 0,0,count,LongBuffer );
  Print(TimeToString(TimeCurrent()),";","StrategyMA:NeedOpenMA",
        ";","LB[0]=",LongBuffer[0],";","LB[1]=",LongBuffer[1],";","SB[0]=",ShortBuffer[0],";","SB[1]=",ShortBuffer[1]);
  if(LongBuffer[0]>LongBuffer[1]&&ShortBuffer[0]>LongBuffer[0]&&ShortBuffer[1]<LongBuffer[1])
  {
    request.type=ORDER_TYPE_SELL;
    OpenPosition();
    return(false);
  }
  if(LongBuffer[0]<LongBuffer[1]&&ShortBuffer[0]<LongBuffer[0]&&ShortBuffer[1]>LongBuffer[1])
  {
    request.type=ORDER_TYPE_BUY;
    OpenPosition();
    return(false);
  }
  return(true);
}
```

The circle closed up.

Let's check how it works. Here is a 2011 report on the 1-hour timeframe with four major pairs: EURUSD, GBPUSD, USDCHF, USDJPY:

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| **Strategy Tester Report** |
| **InstaForex-Server (Build 567)** |
|  |
| **Settings** |
| Expert: | **Main** |
| Symbol: | **EURUSD** |
| Period: | **H1 (2011.01.01 - 2011.12.31)** |
| Input Parameters: | **trainDD=0.50000000** |
|  | **maxDD=0.20000000** |
| Broker: | **InstaForex Companies Group** |
| Currency: | **USD** |
| Initial Deposit: | **10 000.00** |
| Leverage: | **1:100** |
|  |
| **Results** |
| History Quality: | **100%** |
| Bars: | **6197** | Ticks: | **1321631** |
| Total Net Profit: | **-538.74** | Gross Profit: | **3 535.51** | Gross Loss: | **-4 074.25** |
| Profit Factor: | **0.87** | Expected Payoff: | **-89.79** | Margin Level: | **85.71%** |
| Recovery Factor: | **-0.08** | Sharpe Ratio: | **0.07** | OnTester Result: | **0** |
|  |
| Balance Drawdown: |
| Balance Drawdown Absolute: | **4 074.25** | Balance Drawdown Maximal: | **4 074.25 (40.74%)** | Balance Drawdown Relative: | **40.74% (4 074.25)** |
| Equity Drawdown: |
| Equity Drawdown Absolute: | **4 889.56** | Equity Drawdown Maximal: | **6 690.90 (50.53%)** | Equity Drawdown Relative: | **50.53% (6 690.90)** |
|  |
| Total Trades: | **6** | Short Trades (won %): | **6 (16.67%)** | Long Trades (won %): | **0 (0.00%)** |
| Total Trades: | **12** | Profit Trades (% of total): | **1 (16.67%)** | Loss Trades (% of total): | **5 (83.33%)** |
|  | Largest Profit Trade: | **3 535.51** | Largest Loss Trade: | **-1 325.40** |
|  | Average Profit Trade: | **3 535.51** | Average Loss Trade: | **-814.85** |
|  | Maximum consecutive wins: | **1 (3 535.51)** | Maximum consecutive losses: | **5 (-4 074.25)** |
|  | Maximum consecutive profit (count): | **3 535.51 (1)** | Maximum consecutive loss (count): | **-4 074.25 (5)** |
|  | Average consecutive wins: | **1** | Average consecutive losses: | **5** |
|  |

|     |
| --- |
|  |

![](https://c.mql5.com/2/3/grafic_T3.png)

|  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Orders** |
| **Open Time** | **Order** | **Symbol** | **Type** | **Volume** | **Price** | **S / L** | **T / P** | **Time** | **State** | **Comment** |
| 2011.01.03 01:00 | 2 | USDCHF | sell | 28.21 / 28.21 | 0.9321 |  |  | 2011.01.03 01:00 | filled |  |
| 2011.01.03 03:00 | 3 | USDCHF | buy | 28.21 / 28.21 | 0.9365 |  |  | 2011.01.03 03:00 | filled |  |
| 2011.01.03 06:00 | 4 | USDCHF | sell | 24.47 / 24.47 | 0.9352 |  |  | 2011.01.03 06:00 | filled |  |
| 2011.01.03 09:00 | 5 | USDCHF | buy | 24.47 / 24.47 | 0.9372 |  |  | 2011.01.03 09:00 | filled |  |
| 2011.01.03 13:00 | 6 | USDCHF | sell | 22.99 / 22.99 | 0.9352 |  |  | 2011.01.03 13:00 | filled |  |
| 2011.01.03 16:00 | 7 | USDCHF | buy | 22.99 / 22.99 | 0.9375 |  |  | 2011.01.03 16:00 | filled |  |
| 2011.01.03 18:00 | 8 | USDJPY | sell | 72.09 / 72.09 | 81.57 |  |  | 2011.01.03 18:00 | filled |  |
| 2011.01.03 21:00 | 9 | USDJPY | buy | 72.09 / 72.09 | 81.66 |  |  | 2011.01.03 21:00 | filled |  |
| 2011.01.04 01:00 | 10 | USDJPY | sell | 64.54 / 64.54 | 81.67 |  |  | 2011.01.04 01:00 | filled |  |
| 2011.01.04 02:00 | 11 | USDJPY | buy | 64.54 / 64.54 | 81.78 |  |  | 2011.01.04 02:00 | filled |  |
| 2011.10.20 21:00 | 12 | USDCHF | sell | 56.30 / 56.30 | 0.8964 |  |  | 2011.10.20 21:00 | filled |  |
| 2011.10.21 12:00 | 13 | USDCHF | buy | 56.30 / 56.30 | 0.8908 |  |  | 2011.10.21 12:00 | filled |  |
|  |
| **Deals** |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** | **Comment** |
| 2011.01.01 00:00 | 1 |  | balance |  |  |  |  | 0.00 | 0.00 | 10 000.00 | 10 000.00 |  |
| 2011.01.03 01:00 | 2 | USDCHF | sell | in | 28.21 | 0.9321 | 2 | 0.00 | 0.00 | 0.00 | 10 000.00 |  |
| 2011.01.03 03:00 | 3 | USDCHF | buy | out | 28.21 | 0.9365 | 3 | 0.00 | 0.00 | -1 325.40 | 8 674.60 |  |
| 2011.01.03 06:00 | 4 | USDCHF | sell | in | 24.47 | 0.9352 | 4 | 0.00 | 0.00 | 0.00 | 8 674.60 |  |
| 2011.01.03 09:00 | 5 | USDCHF | buy | out | 24.47 | 0.9372 | 5 | 0.00 | 0.00 | -522.19 | 8 152.41 |  |
| 2011.01.03 13:00 | 6 | USDCHF | sell | in | 22.99 | 0.9352 | 6 | 0.00 | 0.00 | 0.00 | 8 152.41 |  |
| 2011.01.03 16:00 | 7 | USDCHF | buy | out | 22.99 | 0.9375 | 7 | 0.00 | 0.00 | -564.02 | 7 588.39 |  |
| 2011.01.03 18:00 | 8 | USDJPY | sell | in | 72.09 | 81.57 | 8 | 0.00 | 0.00 | 0.00 | 7 588.39 |  |
| 2011.01.03 21:00 | 9 | USDJPY | buy | out | 72.09 | 81.66 | 9 | 0.00 | 0.00 | -794.53 | 6 793.86 |  |
| 2011.01.04 01:00 | 10 | USDJPY | sell | in | 64.54 | 81.67 | 10 | 0.00 | 0.00 | 0.00 | 6 793.86 |  |
| 2011.01.04 02:00 | 11 | USDJPY | buy | out | 64.54 | 81.78 | 11 | 0.00 | 0.00 | -868.11 | 5 925.75 |  |
| 2011.10.20 21:00 | 12 | USDCHF | sell | in | 56.30 | 0.8964 | 12 | 0.00 | 0.00 | 0.00 | 5 925.75 |  |
| 2011.10.21 12:00 | 13 | USDCHF | buy | out | 56.30 | 0.8908 | 13 | 0.00 | -3.78 | 3 539.29 | 9 461.26 |  |
|  | **0.00** | **-3.78** | **-534.96** | **9 461.26** |  |
|  |
| **[Copyright 2001-2011, MetaQuotes Software Corp.](https://www.metaquotes.net/)** |

Let me explain the zone that are marked on the chart (explanations are taken from the log analysis):

1. After the start of the Expert Advisor, the genetic algorithm selected the breakthrough strategy SAR on USDCHF with a share of the deposit in trade equal to 28%, then traded till the evening of January 3rd, lost more than 20% of the balance and began to re-optimize.
2. Then the Expert Advisor decided to trade SAR breakthrough on USDJPY, but with the whole deposit (98%). Naturally, it could not trade long, and therefore started its third optimization in the morning of January 4th.
3. This time it decided to trade golden and dead cross of moving averages on USDCHF once again for the entire deposit. And it waited for the first dead cross up to the 20th of October, and sold it to the maximum, and won back everything that it has lost. After that till the end of the year the Expert Advisor didn't see favorable conditions to enter the market.

**To be continued?**

Can it be continued? What would be the next generation of Expert Advisors? The Expert Advisor who invents strategies and selects the best one among them. And further, it can manage money, buying more powerful hardware, channel and so on...

**Risk Warning:**

This brief statement does not disclose completely all of the risks and other significant aspects of forex currency trading on margin. You should understand the nature of trading and the extent of your exposure to risk. You should carefully consider whether trading is appropriate for you in light of your experience, objectives, financial resources and other relevant circumstances.

Forex is not only a profitable, but also a highly risky market. In terms of margin trading, relatively small exchange fluctuations can have a significant impact on the trader's account, resulting in a loss equal to the initial deposit and any funds additionally deposited to the account to maintain open positions. You should not invest money that you cannot afford to lose. Before deciding to trade, please ensure that you understand all the risks and take into account your level of experience. If necessary, seek independent advice.

**Licenses:**

Module UGAlib.mqh is developed and distributed under the BSD license by Andrey Dik aka [joo](https://www.mql5.com/en/users/joo).

The Expert Advisor and auxiliary modules attached to this article are developed and distributed under the BSD license by the author [Roman Rich](https://www.mql5.com/en/users/Rich). The license text is available in file Lic.txt.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/350](https://www.mql5.com/ru/articles/350)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/350.zip "Download all attachments in the single ZIP archive")

[lic.txt](https://www.mql5.com/en/articles/download/350/lic.txt "Download lic.txt")(1.43 KB)

[sheme\_en.gif](https://www.mql5.com/en/articles/download/350/sheme_en.gif "Download sheme_en.gif")(147.11 KB)

[gamodule.mqh](https://www.mql5.com/en/articles/download/350/gamodule.mqh "Download gamodule.mqh")(7.24 KB)

[main.mq5](https://www.mql5.com/en/articles/download/350/main.mq5 "Download main.mq5")(4.99 KB)

[musthave.mqh](https://www.mql5.com/en/articles/download/350/musthave.mqh "Download musthave.mqh")(7.79 KB)

[strategyma.mqh](https://www.mql5.com/en/articles/download/350/strategyma.mqh "Download strategyma.mqh")(5.13 KB)

[strategysar.mqh](https://www.mql5.com/en/articles/download/350/strategysar.mqh "Download strategysar.mqh")(4.09 KB)

[strategystoch.mqh](https://www.mql5.com/en/articles/download/350/strategystoch.mqh "Download strategystoch.mqh")(6.04 KB)

[ugalib.mqh](https://www.mql5.com/en/articles/download/350/ugalib.mqh "Download ugalib.mqh")(33.36 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tales of Trading Robots: Is Less More?](https://www.mql5.com/en/articles/910)
- [The Last Crusade](https://www.mql5.com/en/articles/368)
- [Dr. Tradelove or How I Stopped Worrying and Created a Self-Training Expert Advisor](https://www.mql5.com/en/articles/334)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6260)**
(15)


![Ivan Kornilov](https://c.mql5.com/avatar/2014/5/5376E42C-F1C6.jpg)

**[Ivan Kornilov](https://www.mql5.com/en/users/excelf)**
\|
21 Mar 2012 at 14:58

The idea is interesting, but there are problems - [genetic algorithms](https://www.mql5.com/en/articles/55 "Genetic algorithms are easy!") are not well suited for optimising trading systems, you may miss many successful parameters. Nothing prevents you from doing a full run if you trade only on the bar opening. I don't really like the trading signals you are trying to optimise. I think there are much better signal systems for optimisation nowadays.


![Gilles Maine](https://c.mql5.com/avatar/avatar_na2.png)

**[Gilles Maine](https://www.mql5.com/en/users/smok)**
\|
17 Apr 2012 at 11:50

Good job , can you give us MQLArticle.pdf in  MQLArticle.txt and whe can translate it to any languages ;)

Sorry, your attached archive MQL5Article.zip is not here :(

![Gilles Maine](https://c.mql5.com/avatar/avatar_na2.png)

**[Gilles Maine](https://www.mql5.com/en/users/smok)**
\|
7 Jun 2012 at 17:29

**Rich:**

Sorry, but I don't speak English...

Римскаяхорошомы знаем, чтовы неговорите по-английски.

Ноесли вы дадите намтекстpouronsформатемы переводимего сGOOGLE, а не вформатеPDF

![supercoder2006](https://c.mql5.com/avatar/2012/7/5017070D-710E.jpg)

**[supercoder2006](https://www.mql5.com/en/users/supercoder2006)**
\|
31 Jul 2012 at 21:08

Я неговорю на русском,ноя люблюшутки.

![40565239](https://c.mql5.com/avatar/avatar_na2.png)

**[40565239](https://www.mql5.com/en/users/40565239)**
\|
27 May 2025 at 08:43

**MetaQuotes:**

A new article [Trademinator 3: Rise of the Trading Machines](https://www.mql5.com/en/articles/350) has been published:

Author: [Roman Zamozhnyy](https://www.mql5.com/en/users/Rich "Rich")

This is really fascinating


![Time Series Forecasting Using Exponential Smoothing (continued)](https://c.mql5.com/2/0/Exponent_Smoothing2.png)[Time Series Forecasting Using Exponential Smoothing (continued)](https://www.mql5.com/en/articles/346)

This article seeks to upgrade the indicator created earlier on and briefly deals with a method for estimating forecast confidence intervals using bootstrapping and quantiles. As a result, we will get the forecast indicator and scripts to be used for estimation of the forecast accuracy.

![Promote Your Development Projects Using EX5 Libraries](https://c.mql5.com/2/0/Use_ex5_libraries.png)[Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)

Hiding of the implementation details of classes/functions in an .ex5 file will enable you to share your know-how algorithms with other developers, set up common projects and promote them in the Web. And while the MetaQuotes team spares no effort to bring about the possibility of direct inheritance of ex5 library classes, we are going to implement it right now.

![Multiple Regression Analysis. Strategy Generator and Tester in One](https://c.mql5.com/2/0/Multiple_Regression_Analysis_MQL5.png)[Multiple Regression Analysis. Strategy Generator and Tester in One](https://www.mql5.com/en/articles/349)

The article gives a description of ways of use of the multiple regression analysis for development of trading systems. It demonstrates the use of the regression analysis for strategy search automation. A regression equation generated and integrated in an EA without requiring high proficiency in programming is given as an example.

![Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://c.mql5.com/2/0/MQL5_protection_methods.png)[Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)

Most developers need to have their code secured. This article will present a few different ways to protect MQL5 software - it presents methods to provide licensing capabilities to MQL5 Scripts, Expert Advisors and Indicators. It covers password protection, key generators, account license, time-limit evaluation and remote protection using MQL5-RPC calls.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/350&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083022472938263408)

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