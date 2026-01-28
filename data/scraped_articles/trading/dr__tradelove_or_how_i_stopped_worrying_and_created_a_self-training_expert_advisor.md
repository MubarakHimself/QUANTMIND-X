---
title: Dr. Tradelove or How I Stopped Worrying and Created a Self-Training Expert Advisor
url: https://www.mql5.com/en/articles/334
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:22:18.085612
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/334&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069383044080534567)

MetaTrader 5 / Trading


### Concept

After creating the Expert Advisor we all resort to using the built-in [Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "MetaTrader 5 Trading Strategy Tester ") to select optimal parameters. Upon selection of those, we run the Expert Advisor and once any significant change in it occurs, the Expert Advisor is then stopped and optimised over and over again using the Strategy Tester, and so on.

Can we assign the reoptimisation decision-making and reoptimisation as a process to the Expert Advisor without naturally interrupting its work?

One of the solutions to this problem was proposed by Quantum in his article ["Adaptive Trading Systems and Their Use in MetaTrader5 Terminal"](https://www.mql5.com/en/articles/143 "Adaptive Trading Systems and Their Use in MetaTrader5 Terminal."), dedicated to the use of a real trading system alongside a few (unlimited in number) virtual trading strategies out of which a strategy was selected that had until now brought the highest profit. Decision to change the trading strategy is adopted after a certain fixed bar value has been surpassed.

I propose to use a genetic algorithm (GA) code set out by [joo](https://www.mql5.com/en/users/joo "Andrey Dik") in the article ["Genetic Algorithms - It's Easy!"](https://www.mql5.com/en/articles/55 "Genetic Algorithms - It's Easy! "). Let us have a look at the implementation of such Expert Advisor (one of the below examples is an EA proposed for participation in [Automated Trading Championship 2011](https://championship.mql5.com/2011/en)).

### Work in progress

So we need to define what the Expert Advisor should be able to do. Firstly, and it goes without saying, to trade using the selected strategy. Secondly, to make a decision: whether it is time to reoptimise (to perform a new optimisation of the input parameters). And thirdly, to reoptimise utilising GA. To begin with, we will review the simplest reoptimisation - there is a strategy and we just select the new parameters. We will then see if we can, utilising GA, select another strategy in a changed market environment and if so - how this can be done.

Further, to facilitate simulation in the fitness function we make a decision to trade only completed bars in one instrument. There will be no adding positions and partial closes. Those who prefer to use fixed stops and takes as well as trailing stops, please refer to the article ["Tick Generation Algorithm in MetaTrader5 Strategy Tester"](https://www.mql5.com/en/articles/75 "Tick Generation Algorithm in the MetaTrader 5 Strategy Tester") in order to implement Stop Loss and Take Profit order checks in the fitness function. I shall expand on the below clever phrase:

In the fitness function I simulate a test mode known in the Tester as "Open Prices Only". BUT! It does not mean that this is the only possible test process simulation in the fitness function. More scrupulous people might want to implement a fitness function test using the "Every Tick" mode. In order not to reinvent the wheel or make up "every tick" I would like to draw their attention to an existing algorithm developed by MetaQuotes. In other words, having read this article one will be able to simulate the "Every Tick" mode in the fitness function which is a necessary condition for correct simulation of stops and takes in FF.

Before proceeding to the main point - strategy implementation - let us briefly review the technicalities and implement auxiliary functions defining the opening of a new bar as well as opening and closing of positions:

```
//+------------------------------------------------------------------+
//| Define whether a new bar has opened                             |
//+------------------------------------------------------------------+
bool isNewBars()
  {
   CopyTime(s,tf,0,1,curBT);
   TimeToStruct(curBT[0],curT);
   if(tf==PERIOD_M1||
      tf==PERIOD_M2||
      tf==PERIOD_M3||
      tf==PERIOD_M4||
      tf==PERIOD_M5||
      tf==PERIOD_M6||
      tf==PERIOD_M10||
      tf==PERIOD_M12||
      tf==PERIOD_M15||
      tf==PERIOD_M20||
      tf==PERIOD_M30)
      if(curT.min!=prevT.min)
        {
         prevBT[0]=curBT[0];
         TimeToStruct(prevBT[0],prevT);
         return(true);
        };
   if(tf==PERIOD_H1||
      tf==PERIOD_H2||
      tf==PERIOD_H3||
      tf==PERIOD_H4||
      tf==PERIOD_H6||
      tf==PERIOD_H8||
      tf==PERIOD_M12)
      if(curT.hour!=prevT.hour)
        {
         prevBT[0]=curBT[0];
         TimeToStruct(prevBT[0],prevT);
         return(true);
        };
   if(tf==PERIOD_D1||
      tf==PERIOD_W1)
      if(curT.day!=prevT.day)
        {
         prevBT[0]=curBT[0];
         TimeToStruct(prevBT[0],prevT);
         return(true);
        };
   if(tf==PERIOD_MN1)
      if(curT.mon!=prevT.mon)
        {
         prevBT[0]=curBT[0];
         TimeToStruct(prevBT[0],prevT);
         return(true);
        };
   return(false);
  }
//+------------------------------------------------------------------+
//|  ClosePosition                                                   |
//+------------------------------------------------------------------+
void ClosePosition()
  {
   request.action=TRADE_ACTION_DEAL;
   request.symbol=PositionGetSymbol(0);
   if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY) request.type=ORDER_TYPE_SELL;
   else request.type=ORDER_TYPE_BUY;
   request.type_filling=ORDER_FILLING_FOK;
   if(SymbolInfoInteger(PositionGetSymbol(0),SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_REQUEST||
      SymbolInfoInteger(PositionGetSymbol(0),SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_INSTANT)
     {
      request.sl=NULL;
      request.tp=NULL;
      request.deviation=100;
     }
   while(PositionsTotal()>0)
     {
      request.volume=NormalizeDouble(MathMin(PositionGetDouble(POSITION_VOLUME),SymbolInfoDouble(PositionGetSymbol(0),SYMBOL_VOLUME_MAX)),2);
      if(SymbolInfoInteger(PositionGetSymbol(0),SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_REQUEST||
         SymbolInfoInteger(PositionGetSymbol(0),SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_INSTANT)
        {
         if(request.type==ORDER_TYPE_SELL) request.price=SymbolInfoDouble(s,SYMBOL_BID);
         else request.price=SymbolInfoDouble(s,SYMBOL_ASK);
        }
      OrderSend(request,result);
      Sleep(10000);
     }
  }
//+------------------------------------------------------------------+
//|  OpenPosition                                                    |
//+------------------------------------------------------------------+
void OpenPosition()
  {
   double vol;
   request.action=TRADE_ACTION_DEAL;
   request.symbol=s;
   request.type_filling=ORDER_FILLING_FOK;
   if(SymbolInfoInteger(s,SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_REQUEST||
      SymbolInfoInteger(s,SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_INSTANT)
     {
      request.sl=NULL;
      request.tp=NULL;
      request.deviation=100;
     }
   vol=MathFloor(AccountInfoDouble(ACCOUNT_FREEMARGIN)*optF*AccountInfoInteger(ACCOUNT_LEVERAGE)
       /(SymbolInfoDouble(s,SYMBOL_TRADE_CONTRACT_SIZE)*SymbolInfoDouble(s,SYMBOL_VOLUME_STEP)))*SymbolInfoDouble(s,SYMBOL_VOLUME_STEP);
   vol=MathMax(vol,SymbolInfoDouble(s,SYMBOL_VOLUME_MIN));
   vol=MathMin(vol,GetPossibleLots()*0.95);
   if(SymbolInfoDouble(s,SYMBOL_VOLUME_LIMIT)!=0) vol=NormalizeDouble(MathMin(vol,SymbolInfoDouble(s,SYMBOL_VOLUME_LIMIT)),2);
   request.volume=NormalizeDouble(MathMin(vol,SymbolInfoDouble(s,SYMBOL_VOLUME_MAX)),2);
   while(PositionSelect(s)==false)
     {
      if(SymbolInfoInteger(s,SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_REQUEST||
         SymbolInfoInteger(s,SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_INSTANT)
        {
         if(request.type==ORDER_TYPE_SELL) request.price=SymbolInfoDouble(s,SYMBOL_BID);
         else request.price=SymbolInfoDouble(s,SYMBOL_ASK);
        }
      OrderSend(request,result);
      Sleep(10000);
      PositionSelect(s);
     }
   while(PositionGetDouble(POSITION_VOLUME)<vol)
     {
      request.volume=NormalizeDouble(MathMin(vol-PositionGetDouble(POSITION_VOLUME),SymbolInfoDouble(s,SYMBOL_VOLUME_MAX)),2);
      if(SymbolInfoInteger(PositionGetSymbol(0),SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_REQUEST||
         SymbolInfoInteger(PositionGetSymbol(0),SYMBOL_TRADE_EXEMODE)==SYMBOL_TRADE_EXECUTION_INSTANT)
        {
         if(request.type==ORDER_TYPE_SELL) request.price=SymbolInfoDouble(s,SYMBOL_BID);
         else request.price=SymbolInfoDouble(s,SYMBOL_ASK);
        }
      OrderSend(request,result);
      Sleep(10000);
      PositionSelect(s);
     }
  }
//+------------------------------------------------------------------+
```

Upon careful consideration, you can notice three significant parameters in the position opening function: **s** and **optF** variables and GetPossibleLots() function call:

- s - trading instrument, one of the variables optimised by GA,

- optF - part of the deposit to be used for trading (another variable optimised by GA),
- GetPossibleLots() function - returns the part of the deposit to be used for trading:

```
//+------------------------------------------------------------------+
//|  GetPossibleLots                                                 |
//+------------------------------------------------------------------+
double GetPossibleLots()
  {
   request.volume=1.0;
   if(request.type==ORDER_TYPE_SELL) request.price=SymbolInfoDouble(s,SYMBOL_BID);
   else request.price=SymbolInfoDouble(s,SYMBOL_ASK);
   OrderCheck(request,check);
   return(NormalizeDouble(AccountInfoDouble(ACCOUNT_FREEMARGIN)/check.margin,2));
  }
```

Slightly breaking the order of the narrative, we introduce two more functions common to all Expert Advisors and essential at the stage Two:

```
//+------------------------------------------------------------------+
//|  InitRelDD                                                       |
//+------------------------------------------------------------------+
void InitRelDD()
  {
   ulong DealTicket;
   double curBalance;
   prevBT[0]=D'2000.01.01 00:00:00';
   TimeToStruct(prevBT[0],prevT);
   curBalance=AccountInfoDouble(ACCOUNT_BALANCE);
   maxBalance=curBalance;
   HistorySelect(D'2000.01.01 00:00:00',TimeCurrent());
   for(int i=HistoryDealsTotal();i>0;i--)
     {
      DealTicket=HistoryDealGetTicket(i);
      curBalance=curBalance+HistoryDealGetDouble(DealTicket,DEAL_PROFIT);
      if(curBalance>maxBalance) maxBalance=curBalance;
     }
  }
//+------------------------------------------------------------------+
//|  GetRelDD                                                        |
//+------------------------------------------------------------------+
double GetRelDD()
  {
   if(AccountInfoDouble(ACCOUNT_BALANCE)>maxBalance) maxBalance=AccountInfoDouble(ACCOUNT_BALANCE);
   return((maxBalance-AccountInfoDouble(ACCOUNT_BALANCE))/maxBalance);
  }
```

What can we see here? The first function determines the maximum account balance value, the second function calculates the relative current drawdown of the account. Their peculiarities will be set out in detail in the description of stage Two.

Moving on to the Expert Advisors as such. Since we are only beginners, we will not get an Expert Advisor to select a strategy but will strictly implement two Expert Advisors with the following strategies:

- one trades using intersections of moving averages (Golden Cross - we buy an instrument, Death Cross - we sell);

- the other one is a simple neural network that receives price changes in the range of \[0..1\] over the five last trading sessions.

Algorithmically the work of a self-optimising Expert Advisor can be exemplified as follows:

1. Initialisation of variables used by the Expert Advisor: define and initialize indicator buffers or set up the neural network topology (number of layers/neurons in a layer; a simple neural network where the number of neurons is the same in all layers is given as an example), set the working timeframe. Further, probably the most important step - we call the Genetic Optimisation function which in its turn addresses the foremost function - fitness function (hereinafter - FF).



**IMPORTANT!** There is a new FF for every trading strategy, i.e. it is created every time anew, e.g. FF for a single moving average is completely different from FF for two moving averages and it significantly differs from the neural network FF.



    FF performance result in my Expert Advisors is a maximum balance provided that the relative drawdown has not surpassed the critical value set as an external variable (in our examples - 0,5). In other words, if the next GA run gives the balance of 100,000, while the relative balance drawdown is -0,6, then FF=0,0. In your case, my dear Reader, the FF result may bring up completely different criteria.

    Collect the Genetic Algorithm performance results: for intersection of moving averages these will obviously be moving average periods, in case of a neural network there will be synapse weights, and the result common for both of them (and for my other Expert Advisors) is an instrument to be traded until the next reoptimisation and already familiar to us optF, i.e. a part of the deposit to be used for trading. You are free to add optimised parameters to your FF at your own discretion, for instance you can also select timeframe or other parameters...



    The last step in the initialisation is to find out the maximum account balance value. Why is it important? Because this is a starting point for the reoptimisation decision making.





**IMPORTANT!** How the decision on reoptimisation is taken: once the relative BALANCE drawdown reaches a certain critical value set as an external variable (in our examples - 0,2), we need to reoptimise. In order not to get an Expert Advisor to implement reoptimisation at every bar upon reaching the critical drawdown, the maximum balance value is replaced with a current value.



    You, my dear Reader, may have a totally different criterion for the implementation of reoptimisation.

2. Trade in progress.

3. Upon every closed position we check whether the balance drawdown has reached the critical value. If the critical value has been reached, we run GA and collect its performance results (that's reoptimisation!)

4. And we are waiting for either a call from the forex director asking not to bankrupt the world or (which is more often) for Stop Out, Margin Call, emergency ambulance...


Please find below a programmed implementation of the above for the Expert Advisor using moving averages strategy (source code is also available) and using neural network - all as a source code.

The code is provided under GPL license terms and conditions.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   tf=Period();
//---for bar-to-bar test...
   prevBT[0]=D'2001.01.01';
//---... long ago
   TimeToStruct(prevBT[0],prevT);
//--- historical depth (should be set since the optimisation is based on historical data)
   depth=10000;
//--- copies at a time (should be set since the optimisation is based on historical data)
   count=2;
   ArrayResize(LongBuffer,count);
   ArrayResize(ShortBuffer,count);
   ArrayInitialize(LongBuffer,0);
   ArrayInitialize(ShortBuffer,0);
//--- calling the neural network genetic optimisation function
   GA();
//--- getting the optimised neural network parameters and other variables
   GetTrainResults();
//--- getting the account drawdown
   InitRelDD();
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(isNewBars()==true)
     {
      bool trig=false;
      CopyBuffer(MAshort,0,0,count,ShortBuffer);
      CopyBuffer(MAlong,0,0,count,LongBuffer);
      if(LongBuffer[0]>LongBuffer[1] && ShortBuffer[0]>LongBuffer[0] && ShortBuffer[1]<LongBuffer[1])
        {
         if(PositionsTotal()>0)
           {
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
              {
               ClosePosition();
               trig=true;
              }
           }
        }
      if(LongBuffer[0]<LongBuffer[1] && ShortBuffer[0]<LongBuffer[0] && ShortBuffer[1]>LongBuffer[1])
        {
         if(PositionsTotal()>0)
           {
            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
              {
               ClosePosition();
               trig=true;
              }
           }
        }
      if(trig==true)
        {
         //--- if the account drawdown has exceeded the allowable value:
         if(GetRelDD()>maxDD)
           {
            //--- calling the neural network genetic optimisation function
            GA();
            //--- getting the optimised neural network parameters and other variables
            GetTrainResults();
            //--- readings of the drawdown will from now on be based on the current balance instead of the maximum balance
            maxBalance=AccountInfoDouble(ACCOUNT_BALANCE);
           }
        }
      CopyBuffer(MAshort,0,0,count,ShortBuffer);
      CopyBuffer(MAlong,0,0,count,LongBuffer);
      if(LongBuffer[0]>LongBuffer[1] && ShortBuffer[0]>LongBuffer[0] && ShortBuffer[1]<LongBuffer[1])
        {
         request.type=ORDER_TYPE_SELL;
         OpenPosition();
        }
      if(LongBuffer[0]<LongBuffer[1] && ShortBuffer[0]<LongBuffer[0] && ShortBuffer[1]>LongBuffer[1])
        {
         request.type=ORDER_TYPE_BUY;
         OpenPosition();
        }
     };
  }
//+------------------------------------------------------------------+
//| Preparing and calling the genetic optimizer                      |
//+------------------------------------------------------------------+
void GA()
  {
//--- number of genes (equal to the number of optimised variables),
//--- all of them should be specified in the FitnessFunction())
   GeneCount      =OptParamCount+2;
//--- number of chromosomes in a colony
   ChromosomeCount=GeneCount*11;
//--- minimum search range
   RangeMinimum   =0.0;
//--- maximum search range
   RangeMaximum   =1.0;
//--- search pitch
   Precision      =0.0001;
//--- 1 is a minimum, anything else is a maximum
   OptimizeMethod =2;
   ArrayResize(Chromosome,GeneCount+1);
   ArrayInitialize(Chromosome,0);
//--- number of epochs without any improvement
   Epoch          =100;
//--- ratio of replication, natural mutation, artificial mutation, gene borrowing,
//--- crossingover, interval boundary displacement ratio, every gene mutation probabilty, %
   UGA(100.0,1.0,1.0,1.0,1.0,0.5,1.0);
  }
//+------------------------------------------------------------------+
//| Fitness function for neural network genetic optimizer:           |
//| selecting a pair, optF, synapse weights;                         |
//| anything can be optimised but it is necessary                    |
//| to carefully monitor the number of genes                         |
//+------------------------------------------------------------------+
void FitnessFunction(int chromos)
  {
   int    b;
//--- is there an open position?
   bool   trig=false;
//--- direction of an open position
   string dir="";
//--- opening price
   double OpenPrice=0;
//--- intermediary between a gene colony and optimised parameters
   int    z;
//--- current balance
   double t=cap;
//--- maximum balance
   double maxt=t;
//--- absolute drawdown
   double aDD=0;
//--- relative drawdown
   double rDD=0.000001;
//--- fitness function proper
   double ff=0;
//--- GA is selecting a pair
   z=(int)MathRound(Colony[GeneCount-1][chromos]*12);
   switch(z)
     {
      case  0: {s="AUDUSD"; break;};
      case  1: {s="AUDUSD"; break;};
      case  2: {s="EURAUD"; break;};
      case  3: {s="EURCHF"; break;};
      case  4: {s="EURGBP"; break;};
      case  5: {s="EURJPY"; break;};
      case  6: {s="EURUSD"; break;};
      case  7: {s="GBPCHF"; break;};
      case  8: {s="GBPJPY"; break;};
      case  9: {s="GBPUSD"; break;};
      case 10: {s="USDCAD"; break;};
      case 11: {s="USDCHF"; break;};
      case 12: {s="USDJPY"; break;};
      default: {s="EURUSD"; break;};
     }
   MAshort=iMA(s,tf,(int)MathRound(Colony[1][chromos]*MaxMAPeriod)+1,0,MODE_SMA,PRICE_OPEN);
   MAlong =iMA(s,tf,(int)MathRound(Colony[2][chromos]*MaxMAPeriod)+1,0,MODE_SMA,PRICE_OPEN);
   dig=MathPow(10.0,(double)SymbolInfoInteger(s,SYMBOL_DIGITS));

//--- GA is selecting the optimal F
   optF=Colony[GeneCount][chromos];

   leverage=AccountInfoInteger(ACCOUNT_LEVERAGE);
   contractSize=SymbolInfoDouble(s,SYMBOL_TRADE_CONTRACT_SIZE);
   b=MathMin(Bars(s,tf)-1-count-MaxMAPeriod,depth);

//--- for a neural network using historical data - where the data is copied from
   for(from=b;from>=1;from--)
     {
      CopyBuffer(MAshort,0,from,count,ShortBuffer);
      CopyBuffer(MAlong,0,from,count,LongBuffer);
      if(LongBuffer[0]>LongBuffer[1] && ShortBuffer[0]>LongBuffer[0] && ShortBuffer[1]<LongBuffer[1])
        {
         if(trig==false)
           {
            CopyOpen(s,tf,from,count,o);
            OpenPrice=o[1];
            dir="SELL";
            trig=true;
           }
         else
           {
            if(dir=="BUY")
              {
               CopyOpen(s,tf,from,count,o);
               if(t>0) t=t+t*optF*leverage*(o[1]-OpenPrice)*dig/contractSize; else t=0;
               if(t>maxt) {maxt=t; aDD=0;} else if((maxt-t)>aDD) aDD=maxt-t;
               if((maxt>0) && (aDD/maxt>rDD)) rDD=aDD/maxt;
               OpenPrice=o[1];
               dir="SELL";
               trig=true;
              }
           }
        }
      if(LongBuffer[0]<LongBuffer[1] && ShortBuffer[0]<LongBuffer[0] && ShortBuffer[1]>LongBuffer[1])
        {
         if(trig==false)
           {
            CopyOpen(s,tf,from,count,o);
            OpenPrice=o[1];
            dir="BUY";
            trig=true;
           }
         else
           {
            if(dir=="SELL")
              {
               CopyOpen(s,tf,from,count,o);
               if(t>0) t=t+t*optF*leverage*(OpenPrice-o[1])*dig/contractSize; else t=0;
               if(t>maxt) {maxt=t; aDD=0;} else if((maxt-t)>aDD) aDD=maxt-t;
               if((maxt>0) && (aDD/maxt>rDD)) rDD=aDD/maxt;
               OpenPrice=o[1];
               dir="BUY";
               trig=true;
              }
           }
        }
     }
   if(rDD<=trainDD) ff=t; else ff=0.0;
   AmountStartsFF++;
   Colony[0][chromos]=ff;
  }

//+---------------------------------------------------------------------+
//| getting the optimized neural network parameters and other variables |
//| should always be equal to the number of genes                       |
//+---------------------------------------------------------------------+
void GetTrainResults()
  {
//---  intermediary between a gene colony and optimised parameters
   int z;
   MAshort=iMA(s,tf,(int)MathRound(Chromosome[1]*MaxMAPeriod)+1,0,MODE_SMA,PRICE_OPEN);
   MAlong =iMA(s,tf,(int)MathRound(Chromosome[2]*MaxMAPeriod)+1,0,MODE_SMA,PRICE_OPEN);
   CopyBuffer(MAshort,0,from,count,ShortBuffer);
   CopyBuffer(MAlong,0,from,count,LongBuffer);
//--- save the best pair
   z=(int)MathRound(Chromosome[GeneCount-1]*12);
   switch(z)
     {
      case  0: {s="AUDUSD"; break;};
      case  1: {s="AUDUSD"; break;};
      case  2: {s="EURAUD"; break;};
      case  3: {s="EURCHF"; break;};
      case  4: {s="EURGBP"; break;};
      case  5: {s="EURJPY"; break;};
      case  6: {s="EURUSD"; break;};
      case  7: {s="GBPCHF"; break;};
      case  8: {s="GBPJPY"; break;};
      case  9: {s="GBPUSD"; break;};
      case 10: {s="USDCAD"; break;};
      case 11: {s="USDCHF"; break;};
      case 12: {s="USDJPY"; break;};
      default: {s="EURUSD"; break;};
     }
//--- saving the best optimal F
   optF=Chromosome[GeneCount];
  }
//+------------------------------------------------------------------+
```

Let us look into the main function of the algorithm - the fitness function.

The whole idea behind a self-optimising Expert Advisor is based on simulation of the trading process (as in the standard [Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "MetaTrader 5 Strategy Tester") by MetaQuotes) within a period of time (say, a history of 10000 bars) in the fitness function, which receives an input of optimised variables from the Genetic Algorithm (GA function()). In case of an algorithm based on the intersection of moving averages, the optimised variables include:

- instrument (in forex - a currency pair); yes, it is a typical multicurrency Expert Advisor and the Genetic Algorithm selects an instrument (since the code has been taken from the Expert Advisor proposed for participation in the Championship, the pairs it has correspond with the Championship currency pairs; in general there can be any instrument quoted by a broker).


Note: unfortunately, the Expert Advisor cannot get the pair list from the MarketWatch window in the test mode (we have clarified it here thanks to MetaQuotes users - [No way!](https://www.mql5.com/ru/forum/4739 "Availability of the Market Watch symbols in the test mode")). Therefore if you want to run the Expert Advisor in the Tester separately for forex and shares, just specify your own instruments in FF and GetTrainResults() function.


- part of the deposit to be used for trading;
- periods of two moving averages.

The below examples show a variant of an Expert Advisor FOR TESTING!

REAL TRADE code can be significantly simplified by using the list of instruments from the Market Watch window.

In order to do this in FF and GetTrainResults() function with comments "//--- GA is selecting a pair" and "//--- saving the best pair" just write:

```
//--- GA is selecting a pair
  z=(int)MathRound(Colony[GeneCount-1][chromos]*(SymbolsTotal(true)-1));
  s=SymbolName(z,true);
```

So, at the beginning of FF we specify and initialise, where necessary, the variables for simulation of the history-based trading. At the next stage we collect different values of optimised variables from the Genetic Algorithm, for example from this line "optF=Colony\[GeneCount\]\[chromos\]; the deposit part value is transferred to FF from GA.

We further check the available number of bars in the history and starting either with 10000th bar or the first available bar we simulate the process of receiving quotes in the "Open prices only" mode and making trading decisions:

- Copy the values of moving averages to buffers;
- Check, whether it is a Death Cross.
- If there is a Death Cross and no open positions (if(trig==false)) - open a virtual SELL position (just remember the opening price and direction);
- If there is a Death Cross and an open BUY position (if(dir=="BUY")) - take the bar opening price and pay attention to the three very important lines as follows:

> 1. Simulate closing of a position and change in the balance: the current balance is increased by a current balance value, which is multiplied by the part of the deposit to be traded, multiplied by the difference between the open and close prices, and multiplied by the pip price (rough);
> 2. Check whether the current balance has reached the maximum over the history of trade simulation; if not, calculate the maximal balance drawdown in money;
> 3. Convert the previously calculated drawdown in money in relative balance drawdown;

- Open a virtual SELL position (just remember the open price and direction);
- Do similar checks and calculations for a Golden Cross.

After going through the whole available history and simulation of the virtual trade, calculate the final FF value: if the calculated relative balance drawdown is less than the one set for testing, then FF=balance, otherwise FF=0. Genetic Algorithm aims at maximisation of the fitness function!

After all, giving various values of the instruments, deposit parts and periods of moving averages the Genetic Algorithm will find the values that maximise the balance at the minimum (minimum is set by the user) relative drawdown.

### Conclusion

Here is a brief conclusion: it is easy to create a self-training Expert Advisor, the difficult part is to find what to input (the important thing is an idea, implementation is just a technical issue).

In anticipation of the question from pessimists - "Does it work?", I have an answer - it does; my word to optimists - this is not the Holy Grail.

What is the fundamental difference between the proposed method and the one by Quantum? It can be best exemplified by comparing the Expert Advisors using MA's:

1. The decision on periods of MA's in Adaptive Trading System should be taken before compilation and strictly coded and selection is only possible out of this limited number of variants; we do not take any decision on periods before compilation in Genetically Optimised Expert Advisor, this decision will be taken by GA and the number of variants is limited only by common sense.
2. Virtual trade in Adaptive Trading System is bar-to-bar; it is rarely so in Genetically Optimised Expert Advisor - and then only upon the occurrence of conditions for reoptimisation. Computer performance upon the increasing number of strategies, parameters, instruments may be a limiting factor for Adaptive Trading System.

### Annex

Here's what we get if the neural network is run in the Tester without any optimisation and based on daily charts as from 01.01.2010:

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| **Strategy Tester Report** |
| **MetaQuotes-Demo (Build 523)** |
|  |
| Expert Advisor: | **ANNExample** |
| Symbol: | **EURUSD** |
| Period: | **Daily (2010.01.01 - 2011.09.30)** |
| Input parameters: | **trainDD=0.9** |
|  | **maxDD=0.1** |
| Broker: | **Alpari NZ Limited** |
| Currency: | **USD** |
| Initial deposit: | **10 000.00** |
| Leverage: | **1:100** |
| **Results** |
| History quality: | **100%** |
| Bars: | **454** | Ticks: | **2554879** |
| Total net profit: | **-9 094.49** | Gross profit: | **29 401.09** | Gross loss: | **-38 495.58** |
| Profit factor: | **0.76** | Expected payoff: | **-20.53** | Margin level: | **732.30%** |
| Recovery factor: | **-0.76** | Sharpe ratio: | **-0.06** | OnTester result: | **0** |
|  |
| Balance drawdown: |
| Abs. balance drawdown: | **9 102.56** | Maximal balance drawdown: | **11 464.70 (92.74%)** | Relative balance drawdown: | **92.74% (11 464.70)** |
| Equity drawdown: |
| Abs. equity drawdown: | **9 176.99** | Maximal equity drawdown: | **11 904.00 (93.53%)** | Relative equity drawdown: | **93.53% (11 904.00)** |
|  |
| Total trades: | **443** | Short trades (won, %): | **7 (14.29%)** | Long trades (won, %): | **436 (53.44%)** |
| Total deals: | **886** | Profit trades (% of total): | **234 (52.82%)** | Loss trades (% of total): | **209 (47.18%)** |
|  | Largest profit trade: | **1 095.57** | Largest loss trade: | **-1 438.85** |
|  | Average profit trade: | **125.65** | Average loss trade: | **-184.19** |
|  | Max. consecutive wins (profit in money): | **8 (397.45)** | Max. consecutive losses (loss in money): | **8 (-1 431.44)** |
|  | Max. consecutive profit (count of wins): | **1 095.57 (1)** | Max. consecutive loss (count of losses): | **-3 433.21 (6)** |
|  | Average consecutive wins: | **2** | Average consecutive losses: | **2** |

and herebelow are three variants of reoptimisation to choose from:

first...

|     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Swap** | **Profit** | **Balance** |
| 2010.01.01 00:00 | 1 |  | balance |  |  |  |  | 0.00 | 10 000.00 | 10 000.00 |
| 2010.01.04 00:00 | 2 | AUDUSD | buy | in | 0.90 | 0.89977 | 2 | 0.00 | 0.00 | 10 000.00 |
| 2010.01.05 00:00 | 3 | AUDUSD | sell | out | 0.90 | 0.91188 | 3 | 5.67 | 1 089.90 | 11 095.57 |
| 2010.01.05 00:00 | 4 | AUDUSD | buy | in | 0.99 | 0.91220 | 4 | 0.00 | 0.00 | 11 095.57 |
| 2010.01.06 00:00 | 5 | AUDUSD | sell | out | 0.99 | 0.91157 | 5 | 6.24 | -62.37 | 11 039.44 |
| 2010.01.06 00:00 | 6 | AUDUSD | buy | in | 0.99 | 0.91190 | 6 | 0.00 | 0.00 | 11 039.44 |
| 2010.01.07 00:00 | 7 | AUDUSD | sell | out | 0.99 | 0.91924 | 7 | 18.71 | 726.66 | 11 784.81 |

second...

|     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Commission** | **Swap** | **Profit** | **Balance** |
| 2010.05.19 00:00 | 189 | AUDUSD | sell | out | 0.36 | 0.86110 | 189 | 0.00 | 2.27 | -595.44 | 4 221.30 |
| 2010.05.19 00:00 | 190 | EURAUD | sell | in | 0.30 | 1.41280 | 190 | 0.00 | 0.00 | 0.00 | 4 221.30 |
| 2010.05.20 00:00 | 191 | EURAUD | buy | out | 0.30 | 1.46207 | 191 | 0.00 | 7.43 | -1 273.26 | 2 955.47 |
| 2010.05.20 00:00 | 192 | AUDUSD | buy | in | 0.21 | 0.84983 | 192 | 0.00 | 0.00 | 0.00 | 2 955.47 |

third

|     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Time** | **Deal** | **Symbol** | **Type** | **Direction** | **Volume** | **Price** | **Order** | **Swap** | **Profit** | **Balance** |
| 2010.06.16 00:00 | 230 | GBPCHF | buy | in | 0.06 | 1.67872 | 230 | 0.00 | 0.00 | 2 128.80 |
| 2010.06.17 00:00 | 231 | GBPCHF | sell | out | 0.06 | 1.66547 | 231 | 0.13 | -70.25 | 2 058.68 |
| 2010.06.17 00:00 | 232 | GBPCHF | buy | in | 0.06 | 1.66635 | 232 | 0.00 | 0.00 | 2 058.68 |
| 2010.06.18 00:00 | 233 | GBPCHF | sell | out | 0.06 | 1.64705 | 233 | 0.04 | -104.14 | 1 954.58 |
| 2010.06.18 00:00 | 234 | AUDUSD | buy | in | 0.09 | 0.86741 | 234 | 0.00 | 0.00 | 1 954.58 |
| 2010.06.21 00:00 | 235 | AUDUSD | sell | out | 0.09 | 0.87184 | 235 | 0.57 | 39.87 | 1 995.02 |
| 2010.06.21 00:00 | 236 | AUDUSD | buy | in | 0.09 | 0.88105 | 236 | 0.00 | 0.00 | 1 995.02 |
| 2010.06.22 00:00 | 237 | AUDUSD | sell | out | 0.09 | 0.87606 | 237 | 0.57 | -44.91 | 1 950.68 |
| 2010.06.22 00:00 | 238 | AUDUSD | buy | in | 0.09 | 0.87637 | 238 | 0.00 | 0.00 | 1 950.68 |
| 2010.06.23 00:00 | 239 | AUDUSD | sell | out | 0.09 | 0.87140 | 239 | 0.57 | -44.73 | 1 906.52 |
| 2010.06.23 00:00 | 240 | AUDUSD | buy | in | 0.08 | 0.87197 | 240 | 0.00 | 0.00 | 1 906.52 |
| 2010.06.24 00:00 | 241 | AUDUSD | sell | out | 0.08 | 0.87385 | 241 | 1.51 | 15.04 | 1 923.07 |
| 2010.06.24 00:00 | 242 | AUDUSD | buy | in | 0.08 | 0.87413 | 242 | 0.00 | 0.00 | 1 923.07 |
| 2010.06.25 00:00 | 243 | AUDUSD | sell | out | 0.08 | 0.86632 | 243 | 0.50 | -62.48 | 1 861.09 |
| 2010.06.25 00:00 | 244 | AUDUSD | buy | in | 0.08 | 0.86663 | 244 | 0.00 | 0.00 | 1 861.09 |
| 2010.06.28 00:00 | 245 | AUDUSD | sell | out | 0.08 | 0.87375 | 245 | 0.50 | 56.96 | 1 918.55 |
| 2010.06.28 00:00 | 246 | AUDUSD | buy | in | 0.08 | 0.87415 | 246 | 0.00 | 0.00 | 1 918.55 |
| 2010.06.29 00:00 | 247 | AUDUSD | sell | out | 0.08 | 0.87140 | 247 | 0.50 | -22.00 | 1 897.05 |
| 2010.06.29 00:00 | 248 | AUDUSD | buy | in | 0.08 | 0.87173 | 248 | 0.00 | 0.00 | 1 897.05 |
| 2010.07.01 00:00 | 249 | AUDUSD | sell | out | 0.08 | 0.84053 | 249 | 2.01 | -249.60 | 1 649.46 |
| 2010.07.01 00:00 | 250 | EURGBP | sell | in | 0.07 | 0.81841 | 250 | 0.00 | 0.00 | 1 649.46 |
| 2010.07.02 00:00 | 251 | EURGBP | buy | out | 0.07 | 0.82535 | 251 | -0.04 | -73.69 | 1 575.73 |
| 2010.07.02 00:00 | 252 | EURGBP | sell | in | 0.07 | 0.82498 | 252 | 0.00 | 0.00 | 1 575.73 |
| 2010.07.05 00:00 | 253 | EURGBP | buy | out | 0.07 | 0.82676 | 253 | -0.04 | -18.93 | 1 556.76 |
| 2010.07.05 00:00 | 254 | EURGBP | sell | in | 0.06 | 0.82604 | 254 | 0.00 | 0.00 | 1 556.76 |
| 2010.07.06 00:00 | 255 | EURGBP | buy | out | 0.06 | 0.82862 | 255 | -0.04 | -23.43 | 1 533.29 |

**P.S.** As a homework: to not only select the parameters of a certain system but also to select the system that best fits the market at a given moment (hint - from the bank of systems).

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/334](https://www.mql5.com/ru/articles/334)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/334.zip "Download all attachments in the single ZIP archive")

[anntrainlib.mqh](https://www.mql5.com/en/articles/download/334/anntrainlib.mqh "Download anntrainlib.mqh")(9.76 KB)

[matrainlib.mqh](https://www.mql5.com/en/articles/download/334/matrainlib.mqh "Download matrainlib.mqh")(8.94 KB)

[ugalib.mqh](https://www.mql5.com/en/articles/download/334/ugalib.mqh "Download ugalib.mqh")(33.26 KB)

[annexample.mq5](https://www.mql5.com/en/articles/download/334/annexample.mq5 "Download annexample.mq5")(4.32 KB)

[maexample.mq5](https://www.mql5.com/en/articles/download/334/maexample.mq5 "Download maexample.mq5")(4.22 KB)

[musthavelib.mqh](https://www.mql5.com/en/articles/download/334/musthavelib.mqh "Download musthavelib.mqh")(8.14 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tales of Trading Robots: Is Less More?](https://www.mql5.com/en/articles/910)
- [The Last Crusade](https://www.mql5.com/en/articles/368)
- [Trademinator 3: Rise of the Trading Machines](https://www.mql5.com/en/articles/350)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/5324)**
(15)


![_anatoly](https://c.mql5.com/avatar/avatar_na2.png)

**[\_anatoly](https://www.mql5.com/en/users/_anatoly)**
\|
18 Nov 2013 at 17:33

Gentlemen, professionals!

Help a dummy to put all this together to make it work.

Sorry for the stupid question. :-)

![JD4](https://c.mql5.com/avatar/2015/6/55873126-C675.jpg)

**[JD4](https://www.mql5.com/en/users/jd4)**
\|
4 Jul 2015 at 04:42

**Brett Luedtke:**

This is literally the most creatively written title in all of trade-related publications. The Doomsday Device of trade automation!

Thanks for your efforts integrating the powerful concepts provided at this site. Probably the biggest hurdle to readers is finding the proper application of their ideas. It is important to continue authoring articles of this kind, wherein we build off the ideas already presented at the community. I have no doubt this article is capable of inspiring great minds to see the possibilities in the field of trade automation.

Kudos

I was wondering if anyone else would even get the not so thinly veiled reference in the article title.  Apparently at least 1 person did.


![Airton Raposo](https://c.mql5.com/avatar/2017/3/58D1A067-DC42.jpg)

**[Airton Raposo](https://www.mql5.com/en/users/airtonraposo)**
\|
21 Apr 2016 at 18:49

**1**

**MetaQuotes Software Corp:**

New article [Dr Tradelove or how I stopped worrying and created an Expert Advisor for auto-training](https://www.mql5.com/en/articles/334) has been published:

Author: [Roman Zamozhnyy](https://www.mql5.com/en/users/Rich "Rich")

Good afternoon!

Great article!

Reading your post blew my mind!

After spending a lot of time optimising the results of the robot I realised what was obvious, a strategy will only be valid for a certain time, as long as the market holds, but over time the market changes and the strategy that was good a month ago is no longer so good, forcing me to carry out new tests with parameters to find the periods and other settings most adaptable to the current market...

My idea is for the robot to be able to realise on its own, over a period of time, new configurations that are more adaptable to the market, and for it to choose the best one, which I define beforehand.

My question is whether it would be possible to have the robot do this "self-training" every xxx weeks or xx months and update itself with the best value chosen using the external variables chosen in the parameter settings.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
12 Mar 2017 at 20:24

Good work, interesting article! I just wonder if it possible to replace the existing optF, MaxMaPeriod values with optimised ones (to rewrite them in EA\`s file after every optimisation)?

![gardee005](https://c.mql5.com/avatar/avatar_na2.png)

**[gardee005](https://www.mql5.com/en/users/gardee005)**
\|
24 Oct 2024 at 17:13

will not compile, many errors


![ATC Champions League: Interview with Boris Odintsov (ATC 2011)](https://c.mql5.com/2/0/bobsley_ava__1.png)[ATC Champions League: Interview with Boris Odintsov (ATC 2011)](https://www.mql5.com/en/articles/550)

Interview with Boris Odintsov (bobsley) is the last one within the ATC Champions League project. Boris won the Automated Trading Championship 2010 - the first Championship held for the Expert Advisors in the new MQL5 language. Having appeared in the top ten already in the first week of the ATC 2010, his EA brought it to the finish and earned $77,000. This year, Boris participates in the competition with the same Expert Advisor with modified settings. Perhaps the robot would still be able to repeat its success.

![Interview with Ge Senlin (ATC 2011)](https://c.mql5.com/2/0/yyy999_avatar.png)[Interview with Ge Senlin (ATC 2011)](https://www.mql5.com/en/articles/549)

The Expert Advisor by Ge Senlin (yyy999) from China got featured in the top ten of the Automated Trading Championship 2011 in late October and hasn't left it since then. Not often participants from the PRC show good results in the Championship - Forex trading is not allowed in this country. After the poor results in the previous year ATC, Senlin has prepared a new multicurrency Expert Advisor that never closes loss positions and uses position increase instead. Let's see whether this EA will be able to rise even higher with such a risky strategy.

![Testing (Optimization) Technique and Some Criteria for Selection of the Expert Advisor Parameters](https://c.mql5.com/2/17/884_29.gif)[Testing (Optimization) Technique and Some Criteria for Selection of the Expert Advisor Parameters](https://www.mql5.com/en/articles/1347)

There is no trouble finding the Holy Grail of testing, it is however much more difficult to get rid of it. This article addresses the selection of the Expert Advisor operating parameters with automated group processing of optimisation and testing results upon maximum utilisation of the Terminal performance capabilities and minimum end user load.

![Using MetaTrader 5 as a Signal Provider for MetaTrader 4](https://c.mql5.com/2/0/MetaTrader_5_Signal_Provider_MetaTrader_4.png)[Using MetaTrader 5 as a Signal Provider for MetaTrader 4](https://www.mql5.com/en/articles/344)

Analyse and examples of techniques how trading analysis can be performed on MetaTrader 5 platform, but executed by MetaTrader 4. Article will show you how to create simple signal provider in your MetaTrader 5, and connect to it with multiple clients, even running MetaTrader 4. Also you will find out how you can follow participants of Automated Trading Championship in your real MetaTrader 4 account.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ajoxzzdqbxhgrmzsbnotmyqrlotzytmc&ssn=1769181735767583160&ssn_dr=1&ssn_sr=0&fv_date=1769181735&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F334&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Dr.%20Tradelove%20or%20How%20I%20Stopped%20Worrying%20and%20Created%20a%20Self-Training%20Expert%20Advisor%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918173604585637&fz_uniq=5069383044080534567&sv=2552)

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