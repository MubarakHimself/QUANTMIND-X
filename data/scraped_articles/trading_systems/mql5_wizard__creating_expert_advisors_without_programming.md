---
title: MQL5 Wizard: Creating Expert Advisors without Programming
url: https://www.mql5.com/en/articles/171
categories: Trading Systems, Expert Advisors
relevance_score: 12
scraped_at: 2026-01-22T17:15:24.679600
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/171&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048999335452058362)

MetaTrader 5 / Trading systems


### Introduction

When you create automated trading systems it is necessary to write algorithms of analyzing market situation and generating trading signals, algorithms of trailing your open positions, as well as systems of money management and risk management.

Once the modules' code is written the most difficult task is to assemble all parts and to debug the source code of trading robot. Here the key role is played by the architecture of modules interaction: if it is built poorly, the majority of time will be spent on finding and correcting errors, and if you replace the algorithm of any module it will lead to rewriting the entire source code.

In MQL5 using the [object-oriented approach](https://www.mql5.com/en/docs/basis/oop) significantly eases writing and testing of automated trading systems.

[MetaQuotes Software Corp.](https://www.metaquotes.net/ "https://www.metaquotes.net/") has developed classes to implement trading strategies. Now you can generate Expert Advisors code [automatically](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") directly in MetaEditor by selecting the required [Trade Signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses) (currently there are 20 of them), [Trailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses) (4) and [Money Management](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses) (5) modules. By combining these modules you can get many variants of ready-to-use trading systems.

You can also use your own classes with implementation of any of these modules. Create them by your own or order them via the [Jobs](https://www.mql5.com/en/job) service.

In this article we will consider automatic generation of Expert Advisor's source code using [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate"). And there will be nothing to program!

### 1\. Creating Trading Robot Using MQL5 Wizard

Expert Advisor's source code is generated using [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") in MetaEditor.

Basic classes of trading strategies are located in the '\\<client\_terminal\_directory>\\MQL5\\Include\\Expert\\' folder. Ready-to-use algorithms of [trade signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses) classes, classes of [trailing open positions](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses) and classes of [money and risk management](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses) are located in the Signal, Trailing and Money sub-folders. MQL5 Wizard parses files in these folders and uses them to generate Expert Advisor's code.

To launch MQL5 Wizard you need to click "New" button on the toolbar or select "New" from the "File" menu (or simply press Ctrl+N):

![Figure 1. Launching MQL5 Wizard](https://c.mql5.com/2/2/001_MQL5_new_document_EN.png)

Fig. 1. Launching MQL5 Wizard

Then select the type of the program you want to create. In our case select the "Expert Advisor (generate)" option:

![Figure 2. Selecting the Type of Program](https://c.mql5.com/2/2/MQL5_Wizard_Expert_Advisor_generate__1__1.png)

Fig. 2. Selecting the Type of Program

**Step 1. General Properties of Expert Advisor**

Next opens the dialog box, where you can set the general properties of Expert Advisor:

![Figure 3. General Properties of Expert Advisor](https://c.mql5.com/2/2/MQL5_Wizard_Expert_Advisor_parameters__1__1.png)

Fig. 3. General Properties of Expert Advisor

Specify the name of your Expert Advisor, the author's name and the link to your web-site in the "Name", "Author" and "Link" fields (respectively).

Expert Advisor also has the following input parameters:

- Symbol (the string type) - Work symbol for Expert Advisor.
- Timeframe (the timeframe type) - Work timeframe for Expert Advisor..


On the next step select the type of trade signals, on which the expert will trade.

**Step 2. Select the Module of Trade Signals**

Algorithm of opening and closing positions is determined by the module of trade signals. Trade signals modules contain rules of opening/closing/reversing positions.

The Standard Library has ready-to-use [Modules of trade signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses):

01. [CSignalAC](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ac) \- The module of signals based on market models of the indicator [Accelerator Oscillator](https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/ao "https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/ao").
02. [CSignalAMA](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ama) \- The module of signals based on market models of the indicator [Adaptive Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ama "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ama").
03. [CSignalAO](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ao) \- The module of signals based on market models of the indicator [Awesome Oscillator](https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/awesome "https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/awesome").
04. [CSignalBearsPower](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_bears) \- The module of signals based on market models of the oscillator [Bears Power](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears").
05. [CSignalBullsPower](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_bulls) \- The module of signals based on market models of the oscillator [Bulls Power](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bulls "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bulls").
06. [CSignalCCI](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_cci) \- The module of signals based on market models of the oscillator [Commodity Channel Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/cci "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/cci").
07. [CSignalDeM](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_demarker) \- The module of signals based on market models of the oscillator [DeMarker](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/demarker "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/demarker").
08. [CSignalDEMA](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_dema) \- The module of signals based on market models of the indicator [Double Exponential Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/dema "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/dema").
09. [CSignalEnvelopes](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_envelopes) \- The module of signals based on market models of the indicator [Envelopes](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes").
10. [CSignalFrAMA](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_frama) \- The module of signals based on market models of the indicator [Fractal Adaptive Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/fama "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/fama").
11. [CSignalITF](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_time_filter) \- The module of filtration of signals by time.
12. [CSignalMACD](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_macd) \- The module of signals based on market models of the oscillator [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd").
13. [CSignalMA](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ma) \- The module of signals based on market models of the indicator [Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma").
14. [CSignalSAR](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_sar) \- The module of signals based on market models of the indicator [Parabolic SAR](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/psar "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/psar").
15. [CSignalRSI](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_rsi) \- The module of signals based on market models of the oscillator [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi").
16. [CSignalRVI](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_rvi) \- The module of signals based on market models of the oscillator [Relative Vigor Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rvi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rvi").
17. [CSignalStoch](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_stochastic) \- The module of signals based on market models of the oscillator [Stochastic](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so").
18. [CSignalTRIX](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_trix) \- The module of signals based on market models of the oscillator [Triple Exponential Average](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/tea "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/tea").
19. [CSignalTEMA](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_tema) \- The module of signals based on market models of the indicator [Triple Exponential Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/tema "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/tema").
20. [CSignalWPR](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_wpr) \- The module of signals based on market models of the oscillator [Williams Percent Range](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/wpr "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/wpr").


Type of trade signals is selected from the "Name" dropdown list.

After the pressing of Next button, you will see a window:

![Fig. 4. Selection of trade signals of Expert Advisor](https://c.mql5.com/2/2/MQL5_Wizard_Expert_Advisor_Signals_signal_options__1.png)

Fig. 4. Selection of trade signals of Expert Advisor

To add a module of trade signals, press "Add" button.

Let's add trade signals, based on [Moving Average indicator](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ma).

![Fig. 5. Select the Algorithm of Trade Signals](https://c.mql5.com/2/2/MQL5_Wizard_Expert_Advisor_Signals_MA_options__1.png)

Fig. 5. Select the Algorithm of Trade Signals

Each module of trade signals has it's own parameters. You can use the default values.

There are two modes of parameters creation. You can switch between them by double-clicking left mouse button on parameter's icon. If parameter has the highlighted icon ![](https://c.mql5.com/2/2/parameter_active_icon__1.png), then it will be available as the [input variable](https://www.mql5.com/en/docs/basis/variables/inputvariables) of Expert Advisor. Such parameters further can be used for expert optimization in [Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester"). If parameter has the gray icon ![](https://c.mql5.com/2/2/parameter_inactive_icon__1.png), then it will have the fixed value that you can't modify from Expert Advisor's properties.

The module of trade signals will appear in the list:

![Fig. 6. Module of trade signals has been added](https://c.mql5.com/2/2/MQL5_Wizard_Expert_Advisor_Signals_MA_options2__1.png)

Fig. 6. Module of trade signals has been added

**Step 3. Select the Module of Trailing Open Positions**

The next step is to select the [algorithm of trailing open positions](https://www.metatrader5.com/en/terminal/help/trading/general_concept#trailing_stop "https://www.metatrader5.com/en/terminal/help/trading/general_concept#trailing_stop") (Trailing Stop). Using the trailing allows you to save earned profit.

The Standard Library provides several [ways of trailing open positions](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses):

1. [CTrailingNone](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses/ctrailingnone) \- Trailing Stop is not used.
2. [CTrailingFixedPips](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses/ctrailingfixedpips) \- Trailing Stop based on fixed Stop Level.
3. [CTrailingMA](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses/ctrailingma) \- Trailing Stop based on MA.
4. [CTrailingPSAR](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses/ctrailingpsar) \- Trailing Stop based on [Parabolic SAR](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/psar "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/psar").

In our Expert Advisor select "Trailing Stop based on fixed Stop Level":

![Figure 6. Select the Algorithm of Trailing Open Positions](https://c.mql5.com/2/2/006_MQL5_Wizard_Expert_Advisor_Trailing_Stop_EN.png)

Fig. 7. Select the Algorithm of Trailing Open Positions

This type of trailing has two parameters: the StopLevel and ProfitLevel (in points for quotes with 2 and 4 digits after comma), which will be used to trail open positions:

**![Figure 7. Setting Parameters of Selected Algorithm of Trailing Open Positions](https://c.mql5.com/2/2/007_MQL5_Wizard_Expert_Advisor_Trailing_Stop_parameters_EN.png)**

Fig. 9. Setting Parameters of Selected Algorithm of Trailing Open Positions

**Step 4. Select the Module of Money and Risk Management**

On the last step you need to select system of money and risk management, which will be used in your Expert Advisor.

The purpose of this algorithm is to determine the trading volume (in lots) for trading operations, and also the risk management. When the loss value exceeds allowed limit (for example, 10% of equity), the money and risk managing module will forcibly close the unprofitable position.

The Standard Library provides several [ready-to-use implementations of money and risk management algorithms](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses):

1. [CMoneyFixedLot](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneyfixedlot) \- Trading with fixed trade volume.
2. [CMoneyFixedMargin](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneyfixedmargin) \- Trading with fixed margin.
3. [CMoneyFixedRisk](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneyfixedrisk) \- Trading with fixed risk.
4. [CMoneyNone](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneynone) \- Trading with minimal allowed trade volume.
5. [CMoneySizeOptimized](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneysizeoptimized) \- Trading with optimized trade volume.

**![Figure 8. Select the Algorithm of Money and Risk Management](https://c.mql5.com/2/2/008_MQL5_Wizard_Expert_Advisor_Money_Management_EN.png)**

Fig. 9. Select the Algorithm of Money and Risk Management

Select the 'Trading with fixed trade volume' algorithm.

The module we have selected has two parameters:

- Lots - trading volume in lots.

- Percent - maximal allowed percentage of risk.


![Figure 9. Setting Parameters of Selected Algorithm of Money and Risk Management](https://c.mql5.com/2/2/009_MQL5_Wizard_Expert_Advisor_Money_Management_parameters_EN.png)

Fig. 10. Setting Parameters of Selected Algorithm of Money and Risk Management

After clicking "Finish" the TestExpert.mq5 file will appear in the \\teminal\_data\_filder\\MQL5\\Experts\ folder. The filename corresponds to the specified name of Expert Advisor.

### 2\. The Structure of Expert Advisor Created Using MQL5 Wizard

The source code of Expert Advisor, generated by MQL5 wizard, looks as follows:

```
//+------------------------------------------------------------------+
//|                                                   TestExpert.mq5 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include                                                          |
//+------------------------------------------------------------------+
#include <Expert\Expert.mqh>
//--- available signals
#include <Expert\Signal\SignalMA.mqh>
//--- available trailing
#include <Expert\Trailing\TrailingFixedPips.mqh>
//--- available money management
#include <Expert\Money\MoneyFixedLot.mqh>
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
//--- inputs for expert
input string             Expert_Title                  ="TestExpert"; // Document name
ulong                    Expert_MagicNumber            =23689;        //
bool                     Expert_EveryTick              =false;        //
//--- inputs for main signal
input int                Signal_ThresholdOpen          =10;           // Signal threshold value to open [0...100]
input int                Signal_ThresholdClose         =10;           // Signal threshold value to close [0...100]
input double             Signal_PriceLevel             =0.0;          // Price level to execute a deal
input double             Signal_StopLevel              =50.0;         // Stop Loss level (in points)
input double             Signal_TakeLevel              =50.0;         // Take Profit level (in points)
input int                Signal_Expiration             =4;            // Expiration of pending orders (in bars)
input int                Signal_MA_PeriodMA            =85;           // Moving Average(85,0,...) Period of averaging
input int                Signal_MA_Shift               =0;            // Moving Average(85,0,...) Time shift
input ENUM_MA_METHOD      Signal_MA_Method              =MODE_SMA;      // Moving Average(85,0,...) Method of averaging
input ENUM_APPLIED_PRICE  Signal_MA_Applied             =PRICE_CLOSE;    // Moving Average(85,0,...) Prices series
input double             Signal_MA_Weight              =1.0;          // Moving Average(85,0,...) Weight [0...1.0]
//--- inputs for trailing
input int                Trailing_FixedPips_StopLevel  =30;           // Stop Loss trailing level (in points)
input int                Trailing_FixedPips_ProfitLevel=50;           // Take Profit trailing level (in points)
//--- inputs for money
input double             Money_FixLot_Percent          =10.0;         // Percent
input double             Money_FixLot_Lots             =0.1;          // Fixed volume
//+------------------------------------------------------------------+
//| Global expert object                                             |
//+------------------------------------------------------------------+
CExpert ExtExpert;
//+------------------------------------------------------------------+
//| Initialization function of the expert                            |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Initializing expert
   if(!ExtExpert.Init(Symbol(),Period(),Expert_EveryTick,Expert_MagicNumber))
     {
      //--- failed
      printf(__FUNCTION__+": error initializing expert");
      ExtExpert.Deinit();
      return(-1);
     }
//--- Creating signal
   CExpertSignal *signal=new CExpertSignal;
   if(signal==NULL)
     {
      //--- failed
      printf(__FUNCTION__+": error creating signal");
      ExtExpert.Deinit();
      return(-2);
     }
//---
   ExtExpert.InitSignal(signal);
   signal.ThresholdOpen(Signal_ThresholdOpen);
   signal.ThresholdClose(Signal_ThresholdClose);
   signal.PriceLevel(Signal_PriceLevel);
   signal.StopLevel(Signal_StopLevel);
   signal.TakeLevel(Signal_TakeLevel);
   signal.Expiration(Signal_Expiration);
//--- Creating filter CSignalMA
   CSignalMA *filter0=new CSignalMA;
   if(filter0==NULL)
     {
      //--- failed
      printf(__FUNCTION__+": error creating filter0");
      ExtExpert.Deinit();
      return(-3);
     }
   signal.AddFilter(filter0);
//--- Set filter parameters
   filter0.PeriodMA(Signal_MA_PeriodMA);
   filter0.Shift(Signal_MA_Shift);
   filter0.Method(Signal_MA_Method);
   filter0.Applied(Signal_MA_Applied);
   filter0.Weight(Signal_MA_Weight);
//--- Creation of trailing object
   CTrailingFixedPips *trailing=new CTrailingFixedPips;
   if(trailing==NULL)
     {
      //--- failed
      printf(__FUNCTION__+": error creating trailing");
      ExtExpert.Deinit();
      return(-4);
     }
//--- Add trailing to expert (will be deleted automatically))
   if(!ExtExpert.InitTrailing(trailing))
     {
      //--- failed
      printf(__FUNCTION__+": error initializing trailing");
      ExtExpert.Deinit();
      return(-5);
     }
//--- Set trailing parameters
   trailing.StopLevel(Trailing_FixedPips_StopLevel);
   trailing.ProfitLevel(Trailing_FixedPips_ProfitLevel);
//--- Creation of money object
   CMoneyFixedLot *money=new CMoneyFixedLot;
   if(money==NULL)
     {
      //--- failed
      printf(__FUNCTION__+": error creating money");
      ExtExpert.Deinit();
      return(-6);
     }
//--- Add money to expert (will be deleted automatically))
   if(!ExtExpert.InitMoney(money))
     {
      //--- failed
      printf(__FUNCTION__+": error initializing money");
      ExtExpert.Deinit();
      return(-7);
     }
//--- Set money parameters
   money.Percent(Money_FixLot_Percent);
   money.Lots(Money_FixLot_Lots);
//--- Check all trading objects parameters
   if(!ExtExpert.ValidationSettings())
     {
      //--- failed
      ExtExpert.Deinit();
      return(-8);
     }
//--- Tuning of all necessary indicators
   if(!ExtExpert.InitIndicators())
     {
      //--- failed
      printf(__FUNCTION__+": error initializing indicators");
      ExtExpert.Deinit();
      return(-9);
     }
//--- ok
   return(0);
  }
//+------------------------------------------------------------------+
//| Deinitialization function of the expert                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ExtExpert.Deinit();
  }
//+------------------------------------------------------------------+
//| "Tick" event handler function                                    |
//+------------------------------------------------------------------+
void OnTick()
  {
   ExtExpert.OnTick();
  }
//+------------------------------------------------------------------+
//| "Trade" event handler function                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
   ExtExpert.OnTrade();
  }
//+------------------------------------------------------------------+
//| "Timer" event handler function                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   ExtExpert.OnTimer();
  }
//+------------------------------------------------------------------+
```

Expert Advisor's code consists of several sections.

Section describing program properties:

```
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
```

Included files:

```
#include <Expert\Expert.mqh>
//--- available signals
#include <Expert\Signal\SignalMA.mqh>
//--- available trailing
#include <Expert\Trailing\TrailingFixedPips.mqh>
//--- available money management
#include <Expert\Money\MoneyFixedLot.mqh>
```

The code of the [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) class (its instance is used in Expert Advisor) is located in the Expert.mqh file.

The SignalMA.mqh file contains the source code of the selected trade signals class - [CSignalMA](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ma). The TrailingFixedPips.mqh file contains the source code of trailing open positions algorithm class - [CTrailingFixedPips](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses/ctrailingfixedpips). Money and risk management will be implemented by the [CMoneyFixedLot](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneyfixedlot) class contained in the MoneyFixedLot.mqh file.

Next come the input parameters of Expert Advisor:

```
//--- inputs for expert
input string             Expert_Title                  ="TestExpert"; // Document name
ulong                    Expert_MagicNumber            =23689;        //
bool                     Expert_EveryTick              =false;        //
//--- inputs for main signal
input int                Signal_ThresholdOpen          =10;           // Signal threshold value to open [0...100]
input int                Signal_ThresholdClose         =10;           // Signal threshold value to close [0...100]
input double             Signal_PriceLevel             =0.0;          // Price level to execute a deal
input double             Signal_StopLevel              =50.0;         // Stop Loss level (in points)
input double             Signal_TakeLevel              =50.0;         // Take Profit level (in points)
input int                Signal_Expiration             =4;            // Expiration of pending orders (in bars)
input int                Signal_MA_PeriodMA            =85;           // Moving Average(85,0,...) Period of averaging
input int                Signal_MA_Shift               =0;            // Moving Average(85,0,...) Time shift
input ENUM_MA_METHOD      Signal_MA_Method              =MODE_SMA;     // Moving Average(85,0,...) Method of averaging
input ENUM_APPLIED_PRICE   Signal_MA_Applied             =PRICE_CLOSE;  // Moving Average(85,0,...) Prices series
input double             Signal_MA_Weight              =1.0;          // Moving Average(85,0,...) Weight [0...1.0]
//--- inputs for trailing
input int                Trailing_FixedPips_StopLevel  =30;           // Stop Loss trailing level (in points)
input int                Trailing_FixedPips_ProfitLevel=50;           // Take Profit trailing level (in points)
//--- inputs for money
input double             Money_FixLot_Percent          =10.0;         // Percent
input double             Money_FixLot_Lots             =0.1;          // Fixed volume
```

The first three parameters (Expert\_Title, Expert\_MagicNumber and Expert\_EveryTick) are general. They are always present regardless of the selected [trading signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal), [trailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses), and [money and risk management](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses) algorithms.

The string Expert\_Title parameter specifies the name of Expert Advisor, Expert\_MagicNumber specifies its ID (this value will be used in trade requests' parameters), and the Expert\_EveryTick parameter is used to set EA's working mode. If Expert\_EveryTick is set to true, Expert Advisor will call handler functions (check for trade conditions, commit trade operations, trailing open position) each time a new tick is coming for the working symbol.

After the general parameters of Expert Advisor come the input parameters for selected trade signals algorithm (in our case it is the parameters used in the [CSignalMA](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ma) class).

We have selected the [CTrailingStopFixedPips](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses/ctrailingfixedpips) class of trailing open positions. It trails open position at fixed distance determined by Stop Loss and Take Profit levels, whose values are defined in "normal" 2/4 digit points. When price moves towards the open position by distance, that exceeds the number of points set by the Trailing\_FixedPips\_StopLevel level, Expert Advisor modifies the values of the Stop Loss and Take Profit levels (if Trailing\_FixedPips\_ProfitLevel > 0).

The Money\_FixLot\_Percent and Money\_FixLot\_Lots input parameters correspond to parameters of the algorithm with fixed trade lot, implemented in the [CMoneyFixedLot](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneyfixedlot) class. In our case the trade will be performed with fixed volume equal to the value of Money\_FixLot\_Lots.

The [CMoneyFixedLot](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneyfixedlot) class also implements the algorithm of risk management: if there is a loss (as a given percent of current equity) specified in the Inp\_Money\_FixLot\_Percent parameter, the [CMoneyFixedLot](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses/cmoneyfixedlot) class will recommend the Expert Advisor to forcibly close of unprofitable position, and so it will be done.

After input parameters of Expert Advisor the ExtExpert object of the [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) class is declared:

```
CExpert ExtExpert;
```

This is the instance of trading strategy class.

Being an instance of the [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) class, the ExtExpert object contains references to child objects of the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal) (the base class of trade signals), the [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) (the base class of money and risk management) and the [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) (the base class of trailing open positions) classes. In addition the [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) class contains instances of the CExpertTrade, [SSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo), [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo), [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo), [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) classes and the [CIndicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators/cindicators/cindicators2) container.

To set parameters of Expert Advisor you have to create instances of corresponding classes and to specify references to created objects in the ExtExpert class.

Let's consider the [OnInit](https://www.mql5.com/en/docs/basis/function/events#oninit) function of Expert Advisor initialization. Here we initialize and configure properties of the ExtExpert class.

**1\. Initialization of the ExtExpert class:**

```
//--- Initializing expert
   if(!ExtExpert.Init(Symbol(),Period(),Expert_EveryTick,Expert_MagicNumber))
     {
      //--- failed
      printf(__FUNCTION__+": error initializing expert");
      ExtExpert.Deinit();
      return(-1);
     }
```

The ExtExpert object is initialized using the [Init](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert/cexpertinit) method. Here we set the symbol, timeframe, flag of method calling on every tick, ID of Expert Advisor, and also create and initialize private objects of classes (on this stage the [CExpertSignal](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertsignal), [CExpertMoney](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertmoney) and [CExpertTrailing](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexperttrailing) classes are used as the signals, trailing and money management objects).

If the ExtExpert object doesn't initialize successfully, Expert Advisor will be terminated with return code -1.

**2\. Create and configure the Signal object properties**

```
//--- Creating signal
   CExpertSignal *signal=new CExpertSignal;
   if(signal==NULL)
     {
      //--- failed
      printf(__FUNCTION__+": error creating signal");
      ExtExpert.Deinit();
      return(-2);
     }
//---
   ExtExpert.InitSignal(signal);
   signal.ThresholdOpen(Signal_ThresholdOpen);
   signal.ThresholdClose(Signal_ThresholdClose);
   signal.PriceLevel(Signal_PriceLevel);
   signal.StopLevel(Signal_StopLevel);
   signal.TakeLevel(Signal_TakeLevel);
   signal.Expiration(Signal_Expiration);
//--- Creating filter CSignalMA
   CSignalMA *filter0=new CSignalMA;
   if(filter0==NULL)
     {
      //--- failed
      printf(__FUNCTION__+": error creating filter0");
      ExtExpert.Deinit();
      return(-3);
     }
   signal.AddFilter(filter0);
//--- Set filter parameters
   filter0.PeriodMA(Signal_MA_PeriodMA);
   filter0.Shift(Signal_MA_Shift);
   filter0.Method(Signal_MA_Method);
   filter0.Applied(Signal_MA_Applied);
   filter0.Weight(Signal_MA_Weight);
```

Configuration of the trade signals object consists of several steps:

- Creation of signal object and setting of its parameters;

- Creation of module of trade signals and its adding to CExpertSignal class instance.


If the ExtExpert object doesn't initialize successfully, Expert Advisor will be terminated with return code ( from -2 to -3), that depends on what step an error occurred.

Depending on how the parameters were specified in MQL5 Wizard, the appropriate code is generated.

```
//--- Set signal parameters
   filter0.PeriodMA(85);                        //--- Parameter was set as fixed in MQL5 Wizard
                                                   //--- (gray icon - fixed value equal to 85)
   filter0.SlowPeriod(Signal_MA_Shift);      //--- Parameter was set as input variable
                                                   //--- (blue icon - input parameter of Expert Advisor)
```

If parameter is fixed and its value does not differ from default value, it will not be written in the generated code. In such case the default value of parameter (specified in the corresponding class) will be used.

**3.** **Create and configure the Trailing object properties**

```
//--- Creation of trailing object
   CTrailingFixedPips *trailing=new CTrailingFixedPips;
   if(trailing==NULL)
     {
      //--- failed
      printf(__FUNCTION__+": error creating trailing");
      ExtExpert.Deinit();
      return(-4);
     }
//--- Add trailing to expert (will be deleted automatically))
   if(!ExtExpert.InitTrailing(trailing))
     {
      //--- failed
      printf(__FUNCTION__+": error initializing trailing");
      ExtExpert.Deinit();
      return(-5);
     }
//--- Set trailing parameters
   trailing.StopLevel(Trailing_FixedPips_StopLevel);
   trailing.ProfitLevel(Trailing_FixedPips_ProfitLevel);
```

Configuration of the trailing object also consists of several steps:

- Creation of trailing object;

- Adding trailing to expert;
- Setting trailing parameters.


If the trailing object doesn't initialize successfully, Expert Advisor will be terminated with return code ( from -4 to -5), that depends on what step an error occurred.

**4.** **Create and configure the money object properties**

```
//--- Creation of money object
   CMoneyFixedLot *money=new CMoneyFixedLot;
   if(money==NULL)
     {
      //--- failed
      printf(__FUNCTION__+": error creating money");
      ExtExpert.Deinit();
      return(-6);
     }
//--- Add money to expert (will be deleted automatically))
   if(!ExtExpert.InitMoney(money))
     {
      //--- failed
      printf(__FUNCTION__+": error initializing money");
      ExtExpert.Deinit();
      return(-7);
     }
//--- Set money parameters
   money.Percent(Money_FixLot_Percent);
   money.Lots(Money_FixLot_Lots);
```

Configuration of the money and risk management object also consists of 4 steps:

- Creation of money object;

- Adding money to expert;
- Setting money parameters.


If the money object doesn't initialize successfully, Expert Advisor will be terminated with return code ( from -6 to -7), that depends on what step an error occurred.

**5\. Initialize all indicators used in classes**

```
//--- Check all trading objects parameters
   if(!ExtExpert.ValidationSettings())
     {
      //--- failed
      ExtExpert.Deinit();
      return(-8);
     }
//--- Tuning of all necessary indicators
   if(!ExtExpert.InitIndicators())
     {
      //--- failed
      printf(__FUNCTION__+": error initializing indicators");
      ExtExpert.Deinit();
      return(-9);
     }
//--- ok
   return(0);
```

After you create and initialize objects of trade signals, trailing and money management, the ValidationSettings() method of ExtExpert is called. After that the  InitIndicators() method of ExtExpert object is called. It initializes indicators used in the signal, trailing and money objects.

The [OnDeinit](https://www.mql5.com/en/docs/basis/function/events#ondeinit), [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick), [OnTrade](https://www.mql5.com/en/docs/basis/function/events#ontrade) and [OnTimer](https://www.mql5.com/en/docs/basis/function/events#ontimer) events handling is performed by calling the appropriate methods of the ExtExpert class.

If you want to know the details of the [CExpert](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpert) methods implementation, you can view the indicator's source code, located in '\\<client\_terminal\_directory>\\MQL5\\Include\\Expert\\expert.mqh'.

### 3\. Checking Created Expert Advisor in MetaTrader 5 Strategy Tester

If all components of [Standard Library](https://www.mql5.com/en/docs/standardlibrary) are present, the code of generated Expert Advisor compiles successfully:

![Figure 10. Successful Compilation of Expert Advisor's Source Code Created in MQL5 Wizard](https://c.mql5.com/2/2/MQL5_Wizard_Expert_Advisor_compiled_EN.png)

Figure 10. Successful Compilation of Expert Advisor's Source Code Created in MQL5 Wizard

The resulting Expert Advisor will trade according to selected algorithms of [trade signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal), [trailing open positions](https://www.mql5.com/en/docs/standardlibrary/expertclasses/sampletrailingclasses) and [money and risk management](https://www.mql5.com/en/docs/standardlibrary/expertclasses/samplemmclasses).

You can check how your newly created trading system works using [Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester") from [MetaTrader 5](https://www.metatrader5.com/en/trading-platform "https://www.metatrader5.com/en/trading-platform") client terminal. On figure 11 you can see the results of testing on historical data with default settings (EURUSD, H1, 2010.01.01-2011.06.01):

![Figure 11. Results of Expert Advisor Testing on Historical Data (EURUSD, H1) ](https://c.mql5.com/2/2/MQL5_Wizard_Expert_Advisor_Test_Results__2.png)

Figure 11. Results of Expert Advisor Testing on Historical Data (EURUSD, H1)

The best set of Expert Advisor's parameters can be found after [optimization](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") in MetaTrader 5 Strategy Tester.

### Conclusion

Using the [classes of trading strategies](https://www.mql5.com/en/docs/standardlibrary/expertclasses) significantly eases creation and testing of your trading ideas. Now the entire source code of Expert Advisor can be constructed directly in MetaEditor using its [MQL5 Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate") on the basis of ready-to-use Standard Library [modules](https://www.mql5.com/en/docs/standardlibrary/expertclasses) or your own modules.

If you don't want to or can't write your own trade signals module, you can always benefit from the [Jobs](https://www.mql5.com/en/job) service and order either entire trading robot, or only required modules. This approach gives additional benefits:

- Development cost of separate module should be lower than the cost of entire Expert Advisor.

- The resulting module can be reused to create both a standalone Expert Advisor and a whole family of trading robots (based on this module) using the MQL5 Wizard.
- Ordered module must strictly comply to additional requirements of MQL5 Wizard, and this gives additional control over the quality of code.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/171](https://www.mql5.com/ru/articles/171)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/171.zip "Download all attachments in the single ZIP archive")

[testexpert.mq5](https://www.mql5.com/en/articles/download/171/testexpert.mq5 "Download testexpert.mq5")(7.49 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/2928)**
(33)


![Christian](https://c.mql5.com/avatar/2016/3/56F90C3B-A503.gif)

**[Christian](https://www.mql5.com/en/users/collider)**
\|
13 Apr 2020 at 14:14

**TS88:**

Is it possible to download non-listed indicators?

I think I once read an article where the wizard was expanded.

Have a look here.

![kurbads](https://c.mql5.com/avatar/2022/1/61D17632-1F57.jpg)

**[kurbads](https://www.mql5.com/en/users/kurbads)**
\|
5 Mar 2021 at 07:25

I apologise for a dumb question but I can't see where you specify whether it opens long or short position please?


![Thomas Schwabhaeuser](https://c.mql5.com/avatar/avatar_na2.png)

**[Thomas Schwabhaeuser](https://www.mql5.com/en/users/swtrades)**
\|
11 Apr 2022 at 19:10

**TS88 indicators?**

Even if this is no longer current, I'll write something useful in the thread.

1. Many custom indicators can **be  downloaded** from the CodeBase. We consider the corresponding CodeBase pages to be self-explanatory.
2. As a comment from articles on the MQL5 wizard, the only other question that really makes sense is how an indicator can be used by experts, even though no signal based on a given indicator appears in the list of trading signals when the wizard is executed.
3. The article [MQL5 WIZARD: New version](https://www.mql5.com/en/articles/275) of 2016 in combination with [this other article](https://www.mql5.com/en/articles/226) might have been more suitable.
4. To understand the context, the wizard should be run with a few different settings for the various components **and the generated code** including the CExpertSignal, CExpertMoney and CExpertTrailing classes should **be analysed**.
5. In fact, even the standard indicators are unknown to the wizard because they are also encapsulated by the signal classes stored in the right place.

Have fun!

![Sudiro Sudiro](https://c.mql5.com/avatar/2023/11/65665d40-7c3f.jpg)

**[Sudiro Sudiro](https://www.mql5.com/en/users/soediro)**
\|
2 Nov 2022 at 12:55

**kurbads [#](https://www.mql5.com/en/forum/2928/page2#comment_21131709):**

I apologise for a dumb question but I can't see where you specify whether it opens long or short position please?

I think it mentioned in step 2.


![farhadmax](https://c.mql5.com/avatar/avatar_na2.png)

**[farhadmax](https://www.mql5.com/en/users/farhadmax)**
\|
10 Jul 2023 at 12:25

**kurbads [#](https://www.mql5.com/en/forum/2928/page2#comment_21131709):**

I apologise for a dumb question but I can't see where you specify whether it opens long or short position please?

the signal module decides whether to open a short or long position. you should read this article  [https://www.mql5.com/en/articles/226](https://www.mql5.com/en/articles/226 "https://www.mql5.com/en/articles/226")

.

![MQL5 Wizard: How to Create a Module of Trading Signals](https://c.mql5.com/2/0/MQL5_CExpertSignal.png)[MQL5 Wizard: How to Create a Module of Trading Signals](https://www.mql5.com/en/articles/226)

The article discusses how to write your own class of trading signals with the implementation of signals on the crossing of the price and the moving average, and how to include it to the generator of trading strategies of the MQL5 Wizard, as well as describes the structure and format of the description of the generated class for the MQL5 Wizard.

![Create your own Market Watch using the Standard Library Classes](https://c.mql5.com/2/0/visual.png)[Create your own Market Watch using the Standard Library Classes](https://www.mql5.com/en/articles/179)

The new MetaTrader 5 client terminal and the MQL5 Language provides new opportunities for presenting visual information to the trader. In this article, we propose a universal and extensible set of classes, which handles all the work of organizing displaying of the arbitrary text information on the chart. The example of Market Watch indicator is presented.

![Create Your Own Expert Advisor in MQL5 Wizard](https://c.mql5.com/2/0/masterMQL5__2.png)[Create Your Own Expert Advisor in MQL5 Wizard](https://www.mql5.com/en/articles/240)

The knowledge of programming languages is no longer a prerequisite for creating trading robots. Earlier lack of programming skills was an impassable obstacle to the implementation of one's own trading strategies, but with the emergence of the MQL5 Wizard, the situation radically changed. Novice traders can stop worrying because of the lack of programming experience - with the new Wizard, which allows you to generate Expert Advisor code, it is not necessary.

![Andrey Voitenko: Programming errors cost me $15,000 (ATC 2010)](https://c.mql5.com/2/0/avoitenko_avatar.png)[Andrey Voitenko: Programming errors cost me $15,000 (ATC 2010)](https://www.mql5.com/en/articles/538)

Andrey Voitenko is participating in the Automated Trading Championship for the first time, but his Expert Advisor is showing mature trading. For already several weeks Andrey's Expert Advisors has been listed in the top ten and seems to be continuing his positive performance. In this interview Andrey is telling about his EA's features, errors and the price they cost him.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/171&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048999335452058362)

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