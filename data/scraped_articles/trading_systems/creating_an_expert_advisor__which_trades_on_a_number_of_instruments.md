---
title: Creating an Expert Advisor, which Trades on a Number of Instruments
url: https://www.mql5.com/en/articles/105
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:52:42.764022
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/105&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062784930636802265)

MetaTrader 5 / Examples


### Introduction

The technical side of implementing the program code in order for a single Expert Advisor, launched on a single chart, to be able to trade with different financial assets at the same time. In general, this was not a problems even in MQL4. But only with the advent of the MetaTrader 5 client terminal, traders finally got the opportunity to perform a full analysis of the work of such automates, using strategy testers.

So now multi-currency automates will become more popular than ever, and we can forecast a surge of interest in the construction of such trading systems. But the main problem of implementation of such robots is in the fact that their dimensions in the program code expand, at best, in an arithmetic progression, and this is not easy to embrace for a typical programmer.

In this article we will write a simple multi-currency Expert Advisor, in which the structure flaws are, if not absent, then at least minimized.

### 1\. Implementing of a simple trend-following system

In fact, we could start with a maximally simple trading system, following the trend on the basis of a built-in terminal of a technical indicator Triple Exponential Moving Average. This is a very simple algorithm, which does not require special commentaries, and which we will now embody in the program code.

But first and foremost, I would like to make the most general conclusions about the Expert Advisor. It makes sense to begin with the block of incoming Expert Advisor parameters, declared on a global level.

So, first of all we must choose the financial assets that we will be working with. This can be done using line input variables, in which the asset symbols can be stored. Now it would be nice to have a trade ban switch for each financial asset, which would allow to disable trading operations by the asset.

Naturally, each asset should be associated with their individual trading parameters of Stop Loss, Take Profit, the volume of the open position, and slippage. And for obvious reasons, the input parameters of the indicator Triple Exponential Moving Average for each trading chip should be individual.

Here is one final block of input variables for just one chip, performed in accordance with these arguments. The remaining blocks differ only by the numbers in the names of input parameters of the Expert Advisor. For this example I limited myself to only twelve financial assets, although ideally there is no software limitations for the number of such blocks.

We only need something to trade on! And most importantly - our PC must have enough resources for solving this problem.

```
input string                Symb0 = "EURUSD";
input  bool                Trade0 = true;
input int                    Per0 = 15;
input ENUM_APPLIED_PRICE ApPrice0 = PRICE_CLOSE;
input int                 StLoss0 = 1000;
input int               TkProfit0 = 2000;
input double                Lots0 = 0.1;
input int               Slippage0 = 30;
```

Now that we figured out the variables at the global level, we can proceed to the construction of the code within the function OnTick(). The most rational option here would be the division of the algorithm for receiving trading signals and the actual trading part of the Expert Advisor into two custom functions.

And since the Expert Advisor works with twelve financial assets at the same time, there must also be twelve calls of these functions within the OnTick() block.

Naturally, the first input parameter of these functions should be a unique number, under which these trading assets will be listed. The second input parameter, for obvious reasons, will be the line name of the trading financial asset.

For the role of the third parameter, we will set a logical variable to resolve the trade. Next, for the algorithm of determining trading signals, follow the input indicator signals, and for a trading function - the distance to the pending orders, the volume of position and slippage (allowable slippage of the price of open position).

For transferring the trading signals from one function to another, static arrays should be set as the parameters of the function, which derive their values through a reference. This is the final version of the proposed code for the OnTick() function.

```
void OnTick()
  {
//--- declare variables arrays for trade signals
   static bool UpSignal[12], DnSignal[12], UpStop[12], DnStop[12];

//--- get trade signals
   TradeSignalCounter( 0, Symb0,  Trade0,  Per0,  ApPrice0,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 1, Symb1,  Trade1,  Per1,  ApPrice1,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 2, Symb2,  Trade2,  Per2,  ApPrice2,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 3, Symb3,  Trade3,  Per3,  ApPrice3,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 4, Symb4,  Trade4,  Per4,  ApPrice4,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 5, Symb5,  Trade5,  Per5,  ApPrice5,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 6, Symb6,  Trade6,  Per6,  ApPrice6,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 7, Symb7,  Trade7,  Per7,  ApPrice7,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 8, Symb8,  Trade8,  Per8,  ApPrice8,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter( 9, Symb9,  Trade9,  Per9,  ApPrice9,  UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter(10, Symb10, Trade10, Per10, ApPrice10, UpSignal, DnSignal, UpStop, DnStop);
   TradeSignalCounter(11, Symb11, Trade11, Per11, ApPrice11, UpSignal, DnSignal, UpStop, DnStop);

//--- perform trade operations
   TradePerformer( 0, Symb0,  Trade0,  StLoss0,  TkProfit0,  Lots0,  Slippage0,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 1, Symb1,  Trade1,  StLoss1,  TkProfit1,  Lots1,  Slippage1,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 2, Symb2,  Trade2,  StLoss2,  TkProfit2,  Lots2,  Slippage2,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 3, Symb3,  Trade3,  StLoss3,  TkProfit3,  Lots3,  Slippage3,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 4, Symb4,  Trade4,  StLoss4,  TkProfit4,  Lots4,  Slippage4,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 5, Symb5,  Trade5,  StLoss5,  TkProfit5,  Lots5,  Slippage5,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 6, Symb6,  Trade6,  StLoss6,  TkProfit6,  Lots6,  Slippage6,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 7, Symb7,  Trade7,  StLoss7,  TkProfit7,  Lots7,  Slippage7,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 8, Symb8,  Trade8,  StLoss8,  TkProfit8,  Lots8,  Slippage8,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 9, Symb9,  Trade9,  StLoss9,  TkProfit9,  Lots9,  Slippage9,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer(10, Symb10, Trade10, StLoss10, TkProfit10, Lots10, Slippage10, UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer(11, Symb11, Trade11, StLoss11, TkProfit11, Lots11, Slippage11, UpSignal, DnSignal, UpStop, DnStop);
//---
  }
```

Inside the TradeSignalCounter() function, it is only needed to obtain the handle of the technical indicator Triple Exponential Moving Average once at the start of each chip, and then at each change of the bar to calculate the trading signals.

This relatively simple scheme with the implementation in the code is beginning to overflow with minor details.

```
bool TradeSignalCounter(int Number,
                        string Symbol_,
                        bool Trade,
                        int period,
                        ENUM_APPLIED_PRICE ApPrice,
                        bool &UpSignal[],
                        bool &DnSignal[],
                        bool &UpStop[],
                        bool &DnStop[])
  {
//--- check if trade is prohibited
   if(!Trade)return(true);

//--- declare variable to store final size of variables arrays
   static int Size_=0;

//--- declare array to store handles of indicators as static variable
   static int Handle[];

   static int Recount[],MinBars[];
   double TEMA[4],dtema1,dtema2;

//--- initialization
   if(Number+1>Size_) // Entering the initialization block only on first start
     {
      Size_=Number+1; // For this number entering the block is prohibited

      //--- change size of variables arrays
      ArrayResize(Handle,Size_);
      ArrayResize(Recount,Size_);
      ArrayResize(MinBars,Size_);

      //--- determine minimum number of bars, sufficient for calculation
      MinBars[Number]=3*period;

      //--- setting array elements to 0
      DnSignal[Number] = false;
      UpSignal[Number] = false;
      DnStop  [Number] = false;
      UpStop  [Number] = false;

      //--- use array as timeseries
      ArraySetAsSeries(TEMA,true);

      //--- get indicator's handle
      Handle[Number]=iTEMA(Symbol_,0,period,0,ApPrice);
     }

//--- check if number of bars is sufficient for calculation
   if(Bars(Symbol_,0)<MinBars[Number])return(true);
//--- get trade signals
   if(IsNewBar(Number,Symbol_,0) || Recount[Number]) // Entering the block on bar change or on failed copying of data
     {
      DnSignal[Number] = false;
      UpSignal[Number] = false;
      DnStop  [Number] = false;
      UpStop  [Number] = false;

      //--- using indicator's handles, copy values of indicator's
      //--- buffers into static array, specially prepared for this purpose
      if(CopyBuffer(Handle[Number],0,0,4,TEMA)<0)
        {
         Recount[Number]=true; // As data were not received, we should return
                               // into this block (where trade signals are received) on next tick!
         return(false);        // Exiting the TradeSignalCounter() function without receiving trade signals
        }

      //--- all copy operations from indicator buffer are successfully completed
      Recount[Number]=false; // We may not return to this block until next change of bar

      int Digits_ = int(SymbolInfoInteger(Symbol_,SYMBOL_DIGITS)+4);
      dtema2 = NormalizeDouble(TEMA[2] - TEMA[3], Digits_);
      dtema1 = NormalizeDouble(TEMA[1] - TEMA[2], Digits_);

      //---- determining the input signals
      if(dtema2 > 0 && dtema1 < 0) DnSignal[Number] = true;
      if(dtema2 < 0 && dtema1 > 0) UpSignal[Number] = true;

      //---- determining the output signals
      if(dtema1 > 0) DnStop[Number] = true;
      if(dtema1 < 0) UpStop[Number] = true;
     }
//----+
   return(true);
  }
```

In this aspect, the code of the TradePerformer() function turns out to be quiet simple:

```
bool TradePerformer(int    Number,
                    string Symbol_,
                    bool   Trade,
                    int    StLoss,
                    int    TkProfit,
                    double Lots,
                    int    Slippage,
                    bool  &UpSignal[],
                    bool  &DnSignal[],
                    bool  &UpStop[],
                    bool  &DnStop[])
  {
//--- check if trade is prohibited
   if(!Trade)return(true);

//--- close opened positions
   if(UpStop[Number])BuyPositionClose(Symbol_,Slippage);
   if(DnStop[Number])SellPositionClose(Symbol_,Slippage);

//--- open new positions
   if(UpSignal[Number])
      if(BuyPositionOpen(Symbol_,Slippage,Lots,StLoss,TkProfit))
         UpSignal[Number]=false; //This trade signal will be no more on this bar!
//---
   if(DnSignal[Number])
      if(SellPositionOpen(Symbol_,Slippage,Lots,StLoss,TkProfit))
         DnSignal[Number]=false; //This trade signal will be no more on this bar!
//---
   return(true);
  }
```

But this is only because the actual commands for the performance of trading operations are packed into four additional functions:

```
BuyPositionClose();
SellPositionClose();
BuyPositionOpen();
SellPositionOpen();
```

All four functions work completely analogously, so we can limit ourselves to the examination of just one of them:

```
bool BuyPositionClose(const string symbol,ulong deviation)
  {
//--- declare structures of trade request and result of trade request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

//--- check if there is BUY position
   if(PositionSelect(symbol))
     {
      if(PositionGetInteger(POSITION_TYPE)!=POSITION_TYPE_BUY) return(false);
     }
   else  return(false);

//--- initializing structure of the MqlTradeRequest to close BUY position
   request.type   = ORDER_TYPE_SELL;
   request.price  = SymbolInfoDouble(symbol, SYMBOL_BID);
   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = PositionGetDouble(POSITION_VOLUME);
   request.sl = 0.0;
   request.tp = 0.0;
   request.deviation=(deviation==ULONG_MAX) ? deviation : deviation;
   request.type_filling=ORDER_FILLING_FOK;
//---
   string word="";
   StringConcatenate(word,
                     "<<< ============ BuyPositionClose():   Close Buy position at ",
                     symbol," ============ >>>");
   Print(word);

//--- send order to close position to trade server
   if(!OrderSend(request,result))
     {
      Print(ResultRetcodeDescription(result.retcode));
      return(false);
     }
//----+
   return(true);
  }
```

Basically, that's pretty much the whole multi-currency Expert Advisor (Exp\_TEMA.mq5)!

Apart from the considered functions, it contains two additional user functions:

```
bool IsNewBar(int Number, string symbol, ENUM_TIMEFRAMES timeframe);
string ResultRetcodeDescription(int retcode);
```

The first of these functions returns the true value at the moment of the bar change, based on the selected symbol and timeframe, and the second one, returns the line by the result code of the trading transaction, derived from the field retcode of the trade request structure [MqlTradeResult](https://www.mql5.com/en/docs/constants/structures/mqltraderesult).

The Expert Advisor is ready, it's time to begin with testing! There are no visible serious differences in the testing of the multi-currency Expert Advisor from his fellow single-currency Expert Advisor.

Determine the configurations on the tab "Parameters" of the Strategy Tester:

![Figure 1. "Settings" tab of Strategy tester](https://c.mql5.com/2/1/fig1_en.png)

Figure 1. "Settings" tab of Strategy tester

If necessary, adjust the values of the input parameters on the tab "Input parameters:

![Figure 2. "Parameters" tab of Strategy tester](https://c.mql5.com/2/1/fig2_en.png)

Figure 2. "Parameters" tab of Strategy tester

and then click the "Start" button in the Strategy Tester on the tab "Settings":

![Figure 3. Running the Expert Advisor test](https://c.mql5.com/2/1/fig3_en.png)

Figure 3. Running the Expert Advisor test

The passing time of the first test of the Expert Advisor may turn out to be very significant, due to the loading of the history for all twelve symbols. After completing the test in the strategy tester, open the tab "Results":

![Figure 4. Testing results](https://c.mql5.com/2/1/fig4_en.png)

Figure 4. Testing results

and make an analysis of the data, using the contents of the "Chart" tab:

![Figure 5. Chart of balance dynamics and equity](https://c.mql5.com/2/1/fig5_en.png)

Figure 5. Chart of balance dynamics and equity

and the "Journal":

![Figure 6. Strategy tester Journal](https://c.mql5.com/2/1/fig6_en.png)

Figure 6. Strategy tester Journal

Quite naturally, the very essence of the algorithm's entrances and exits of the market of this Expert Advisor's is too simple, and it would be naïve to expect very significant results, when using the first, random parameters. But our goal here is to demonstrates the fundamental idea of constructing a multi-currency Expert Advisor in the simplest way possible.

With the optimization of this Expert Advisor may arise some inconveniences due to too many input parameters. The genetic algorithm of optimization requires a much smaller amount of these parameters, so the Expert Advisor should be optimized on each chip individually, disabling the remaining chips of inputs parameters TradeN.

Now, when the very essence of the approach has been outlined, you can start working with the more interesting algorithm of decision making for multi-currency robot.

### 2\. Resonances on the financial markets and their application in trading systems

The idea of taking into account the correlations between the different financial assets, in general, is not new, and it would be interesting to implement the algorithm, which would be based precisely on the analysis of such trends. In this article I will implement a multi-currency automate, based on the article of Vasily Yakimkin "Resonances - a New Class of Technical Indicators" published in the journal "Currency Speculator" (in Russian) 04, 05, 2001.

The very essence of this approach in a nutshell looks as follows. For example, for researching the on EUR / USD, we use not only the results of some indicators on the financial asset, but also the results of the same indicator on related to the EUR/USD assets - EUR/JPY and USD/JPY. It is best to use the indicator, the values of which are normalized in the same range of changes for the simplicity and ease of measurements and calculations.

Considering these requirements, a well suited for this classic is the stochastic indicator. Although, in actuality, there is no difference in the use of other indicators. As the trend direction we will consider the difference sign between the value of the stochastic **Stoh** and its signal line **Sign**.

![Figure 7. Determining the trend direction](https://c.mql5.com/2/1/Stoh__1__1.png)

Figure 7. Determining the trend direction

For the variable symbol **dStoh** there is an entire table of possible combinations and their interpretations for the direction of the current trend:

![Figure 8. Combinations of the variable symbol dStoh and the trend direction](https://c.mql5.com/2/1/TrendTable__4.png)

Figure 8. Combinations of the variable symbol dStoh and the trend direction

In a case where two signals of the assets EUR / JPY and USD / JPY have the opposite values, we should determine their sum, and if this sum is greater than zero, consider both signals as positive, otherwise - as negative.

Thus, for the opening up Longs, use the situation when the trend is growing, and to exit, use a downward trend,or a trend when the signals of the indicator of the main asset EUR / USD are negative. Also, exit the long if the main asset has no signals, and if the sum of the variable dStoh for the remaining assets is less than zero. For the shorts, everything is absolutely analogous, only the situation is opposite.

The most rational solution would be to place the entire analytical part of the Expert Advisor in the multicurrency indicator, and for the Expert Advisor from the indicator buffers, take only the ready signals for trade control. The version of this indicator type is presented by the indicator MultiStochastic.mq5, providing a visual analyses of market conditions.

![Figure 9. MultiStochastic Indicator](https://c.mql5.com/2/1/MultiStoh__1.png)

Figure 9. MultiStochastic Indicator

The green bar signal the opening and retaining of longs, and the red bars - of shorts, respectively. Pink and light green points on the upper edge of the chart represent the signals of exiting the long and short positions.

This indicator can be directly used for receiving signals in the Expert Advisor, but still it would be better to ease its work and remove all of the unnecessary buffers and elements of visualization, leaving only what is directly involved in the supply of trading signals. This is precisely what was done in the MultiStochastic\_Exp.mq5 indicator.

In this Expert Advisor, I traded with only three chips, so the code of the OnTick() function became exceedingly simple:

```
void OnTick()
  {
//--- declare variables arrays for trade signals
  static bool UpSignal[], DnSignal[], UpStop[], DnStop[];

//--- get trade signals
  TradeSignalCounter(0, Trade0, Kperiod0, Dperiod0, slowing0, ma_method0, price_0, SymbolA0, SymbolB0, SymbolC0, UpSignal, DnSignal, UpStop, DnStop);
  TradeSignalCounter(1, Trade1, Kperiod1, Dperiod1, slowing1, ma_method1, price_1, SymbolA1, SymbolB1, SymbolC1, UpSignal, DnSignal, UpStop, DnStop);
  TradeSignalCounter(2, Trade2, Kperiod2, Dperiod2, slowing2, ma_method2, price_2, SymbolA2, SymbolB2, SymbolC2, UpSignal, DnSignal, UpStop, DnStop);

//--- perform trade operations
   TradePerformer( 0, SymbolA0,  Trade0,  StopLoss0,  0,  Lots0,  Slippage0,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 1, SymbolA1,  Trade1,  StopLoss1,  0,  Lots1,  Slippage1,  UpSignal, DnSignal, UpStop, DnStop);
   TradePerformer( 2, SymbolA2,  Trade2,  StopLoss2,  0,  Lots2,  Slippage2,  UpSignal, DnSignal, UpStop, DnStop);
//---
  }
```

However, the code of the TradeSignalCounter() function is a little more complex: The fact is that a multi-currency indicator works directly with three timeseries of different financial assets, and therefore, we implement a more subtle verification of bars for the adequacy of the minimum number of them in one of the three timeseries, using the Rates\_Total() function.

In addition, an additional verification on the synchronization of timeseries is made, using the SynchroCheck() function, to guarantee the accuracy of determining the moment when a bar change occurs in all timeseries simultaneously.

```
bool TradeSignalCounter(int Number,
                        bool Trade,
                        int Kperiod,
                        int Dperiod,
                        int slowing,
                        ENUM_MA_METHOD ma_method,
                        ENUM_STO_PRICE price_,
                        string SymbolA,
                        string SymbolB,
                        string SymbolC,
                        bool &UpSignal[],
                        bool &DnSignal[],
                        bool &UpStop[],
                        bool &DnStop[])
  {
//--- check if trade is prohibited
   if(!Trade)return(true);
//--- declare variable to store sizes of variables arrays
   static int Size_=0;
//--- declare arrays to store handles of indicators as static variables
   static int Handle[];
   static int Recount[],MinBars[];
//---
   double dUpSignal_[1],dDnSignal_[1],dUpStop_[1],dDnStop_[1];
//--- change size of variables arrays
   if(Number+1>Size_)
     {
      uint size=Number+1;
      //----
      if(ArrayResize(Handle,size)==-1
         || ArrayResize(Recount,size)==-1
         || ArrayResize(UpSignal, size) == -1
         || ArrayResize(DnSignal, size) == -1
         || ArrayResize(UpStop, size) == -1
         || ArrayResize(DnStop, size) == -1
         || ArrayResize(MinBars,size) == -1)
        {
         string word="";
         StringConcatenate(word,"TradeSignalCounter( ",Number,
                           " ): Error!!! Unable to change sizes of variables arrays!!!");
         int error=GetLastError();
         ResetLastError();
         //---
         if(error>4000)
           {
            StringConcatenate(word,"TradeSignalCounter( ",Number," ): Error code ",error);
            Print(word);
           }
         Size_=-2;
         return(false);
        }

      Size_=int(size);
      Recount[Number] = false;
      MinBars[Number] = Kperiod + Dperiod + slowing;

      //--- get indicator's handle
      Handle[Number]=iCustom(SymbolA,0,"MultiStochastic_Exp",
                             Kperiod,Dperiod,slowing,ma_method,price_,
                             SymbolA,SymbolB,SymbolC);
     }
//--- check if number of bars is sufficient for calculation
   if(Rates_Total(SymbolA,SymbolB,SymbolC)<MinBars[Number])return(true);
//--- check timeseries synchronization
   if(!SynchroCheck(SymbolA,SymbolB,SymbolC))return(true);
//--- get trade signals
   if(IsNewBar(Number,SymbolA,0) || Recount[Number])
     {
      DnSignal[Number] = false;
      UpSignal[Number] = false;
      DnStop  [Number] = false;
      UpStop  [Number] = false;

      //--- using indicators' handles, copy values of indicator's
      //--- buffers into static arrays, specially prepared for this purpose
      if(CopyBuffer(Handle[Number], 1, 1, 1, dDnSignal_) < 0){Recount[Number] = true; return(false);}
      if(CopyBuffer(Handle[Number], 2, 1, 1, dUpSignal_) < 0){Recount[Number] = true; return(false);}
      if(CopyBuffer(Handle[Number], 3, 1, 1, dDnStop_  ) < 0){Recount[Number] = true; return(false);}
      if(CopyBuffer(Handle[Number], 4, 1, 1, dUpStop_  ) < 0){Recount[Number] = true; return(false);}

      //--- convert obtained values into values of logic variables of trade commands
      if(dDnSignal_[0] == 300)DnSignal[Number] = true;
      if(dUpSignal_[0] == 300)UpSignal[Number] = true;
      if(dDnStop_  [0] == 300)DnStop  [Number] = true;
      if(dUpStop_  [0] == 300)UpStop  [Number] = true;

      //--- all copy operations from indicator's buffers completed successfully
      //--- unnecessary to return into this block until next bar change
      Recount[Number]=false;
     }
//----+
   return(true);
  }
```

There are no other radical ideological differences of the code of this Expert Advisor (Exp\_ResonanceHunter.mq5) due to the fact that it is compiled on the basis of the same functional components. Therefore I don't think that it will be necessary to spend any more time on its internal structure.

### Conclusion

In my opinion, the code of the multi-currency Expert Advisor in MQL5 is absolutely analogous to the code of a regular Expert Advisor.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/105](https://www.mql5.com/ru/articles/105)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/105.zip "Download all attachments in the single ZIP archive")

[multistochastic.mq5](https://www.mql5.com/en/articles/download/105/multistochastic.mq5 "Download multistochastic.mq5")(17.47 KB)

[multistochastic\_exp.mq5](https://www.mql5.com/en/articles/download/105/multistochastic_exp.mq5 "Download multistochastic_exp.mq5")(8.53 KB)

[exp\_resonancehunter.mq5](https://www.mql5.com/en/articles/download/105/exp_resonancehunter.mq5 "Download exp_resonancehunter.mq5")(23.01 KB)

[exp\_tema.mq5](https://www.mql5.com/en/articles/download/105/exp_tema.mq5 "Download exp_tema.mq5")(25.17 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Simple Trading Systems Using Semaphore Indicators](https://www.mql5.com/en/articles/358)
- [Averaging Price Series for Intermediate Calculations Without Using Additional Buffers](https://www.mql5.com/en/articles/180)
- [Creating an Indicator with Multiple Indicator Buffers for Newbies](https://www.mql5.com/en/articles/48)
- [The Principles of Economic Calculation of Indicators](https://www.mql5.com/en/articles/109)
- [Practical Implementation of Digital Filters in MQL5 for Beginners](https://www.mql5.com/en/articles/32)
- [Custom Indicators in MQL5 for Newbies](https://www.mql5.com/en/articles/37)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1330)**
(59)


![Andrey Kornishkin](https://c.mql5.com/avatar/2010/9/4C876740-C5E0.jpg)

**[Andrey Kornishkin](https://www.mql5.com/en/users/am2)**
\|
9 Sep 2010 at 21:33

**papaklass:**

I reworked the algorithm a bit and on 12 currencies the test runs for 727 seconds. Optimise the algorithm.

Even this probably won't be necessary because I have a 2-currency 4 minutes test on my computer, but when I uploaded it to the test in 1 minute!

4\. Start

finished in 1 min 4 sec

**GODZILLA:**

You can also just use the [OnTimer() function](https://www.mql5.com/ru/docs/basis/function/events#oninit "MQL5 Documentation: Event Handling Functions"). There are inexhaustible possibilities in terms of optimising program code.

By the way, why do you need to meet the five-minute deadline and why not, for example, fifteen minutes?

They have increased the testing time now.

Now testing is going on i7 950 and the maximum time has been increased up to 15 minutes.

.

![Nauris Zukas](https://c.mql5.com/avatar/2016/7/5790E613-D487.jpg)

**[Nauris Zukas](https://www.mql5.com/en/users/abeiks)**
\|
30 Sep 2010 at 18:34

Good afternoon!

Maybesomeone can help to understand [global variables](https://www.mql5.com/en/docs/basis/variables/global "MQL5 Documentation: Global Variables") on the example of the discussed  Expert Advisor (Creating an Expert Advisor that trades on different instruments). What  would you add in the Expert Advisor toperform such a function :

```
     if (dtema2 > 0 && dtema1 < 0)
     {
     DnSignal[Number] = true;
     volume = 0.1;
     } // If the if function is true, volume for Buy and Sell will be 0.1
```

![Anatoliy Ivanov](https://c.mql5.com/avatar/avatar_na2.png)

**[Anatoliy Ivanov](https://www.mql5.com/en/users/ias)**
\|
25 Aug 2011 at 10:03

6.Why is the int Recount\[\] [data type](https://www.mql5.com/ru/docs/basis/types "MQL5 Documentation: Data Types") chosen in exp\_tema.mq5 in:

```
 static int Recount[], MinBars[];
```

Given that Recount\[\] then takes the value of the bool data type:

```
Recount[Number] = true;
...
Recount[Number] = false;
```

7.Does it affect the results?

![Nikolay Kositsin](https://c.mql5.com/avatar/2014/6/538DE448-E682.jpeg)

**[Nikolay Kositsin](https://www.mql5.com/en/users/godzilla)**
\|
25 Aug 2011 at 15:13

**ias:**

6.Why is the int Recount\[\] [data type](https://www.mql5.com/ru/docs/basis/types "MQL5 Documentation: Data Types") selected in exp\_tema.mq5 in:

Given that Recount\[\] then takes the value of the bool data type:

7.Does it affect the results?

It will not affect the results in any way, but, actually, this variable should have been made a logical, static variable!


![Achmad Hidayat](https://c.mql5.com/avatar/avatar_na2.png)

**[Achmad Hidayat](https://www.mql5.com/en/users/achidayat)**
\|
9 Jul 2012 at 06:36

I have trouble when attach this EA. In Expert tab appear this message :

2012.07.09 11:31:16    exp\_tema (multicurrency)-new (EURUSD,M1)    cannot load indicator 'Triple [Exponential](https://www.mql5.com/en/docs/math/mathexp "MQL5 documentation: MathExp function") Moving Average' \[4302\]

What wrong? Thank you

![Functions for Money Management in an Expert Advisor](https://c.mql5.com/2/0/money_management_MQL5__1.png)[Functions for Money Management in an Expert Advisor](https://www.mql5.com/en/articles/113)

The development of trading strategies primarily focuses on searching for patterns for entering and exiting the market, as well as maintaining positions. If we are able to formalize some patterns into rules for automated trading, then the trader faces the question of calculating the volume of positions, the size of the margins, as well as maintaining a safe level of mortgage funds for assuring open positions in an automated mode. In this article we will use the MQL5 language to construct simple examples of conducting these calculations.

![Creating Information Boards Using Standard Library Classes and Google Chart API](https://c.mql5.com/2/0/info_panel_MQL5.png)[Creating Information Boards Using Standard Library Classes and Google Chart API](https://www.mql5.com/en/articles/102)

The MQL5 programming language primarily targets the creation of automated trading systems and complex instruments of technical analyses. But aside from this, it allows us to create interesting information systems for tracking market situations, and provides a return connection with the trader. The article describes the MQL5 Standard Library components, and shows examples of their use in practice for reaching these objectives. It also demonstrates an example of using Google Chart API for the creation of charts.

![Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://c.mql5.com/2/0/Expert_Advisor_classes_MQL5.png)[Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://www.mql5.com/en/articles/116)

This article focuses on the object oriented approach to doing what we did in the article "Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners" - creating a simple Expert Advisor. Most people think this is difficult, but I want to assure you that by the time you finish reading this article, you will be able to write your own Expert Advisor which is object oriented based.

![Genetic Algorithms - It's Easy!](https://c.mql5.com/2/0/genetic_algorithms_MQL5.png)[Genetic Algorithms - It's Easy!](https://www.mql5.com/en/articles/55)

In this article the author talks about evolutionary calculations with the use of a personally developed genetic algorithm. He demonstrates the functioning of the algorithm, using examples, and provides practical recommendations for its usage.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iobekvkmscodtgfvcnjlktpeczbwvndj&ssn=1769158361322383094&ssn_dr=0&ssn_sr=0&fv_date=1769158361&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F105&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20an%20Expert%20Advisor%2C%20which%20Trades%20on%20a%20Number%20of%20Instruments%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915836139079001&fz_uniq=5062784930636802265&sv=2552)

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