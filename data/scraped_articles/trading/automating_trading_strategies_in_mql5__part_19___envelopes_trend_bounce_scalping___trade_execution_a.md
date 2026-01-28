---
title: Automating Trading Strategies in MQL5 (Part 19): Envelopes Trend Bounce Scalping — Trade Execution and Risk Management (Part II)
url: https://www.mql5.com/en/articles/18298
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 8
scraped_at: 2026-01-22T17:44:11.478660
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/18298&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049340600668498361)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 18)](https://www.mql5.com/en/articles/18269), we laid the foundation for the [Envelopes](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes") Trend Bounce Scalping Strategy in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5), building the Expert Advisor’s core infrastructure and signal generation logic. Now, in Part 19, we advance by implementing trade execution and risk management to automate the strategy fully. We will cover the following topics:

1. [Strategy Roadmap and Architecture](https://www.mql5.com/en/articles/18298#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18298#para2)
3. [Backtesting and Optimization](https://www.mql5.com/en/articles/18298#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18298#para4)

By the end, you’ll have a complete MQL5 trading system for scalping trend bounces, optimized for performance—let’s get started!

### Strategy Roadmap and Architecture

In Part 18, we established the groundwork for the Envelopes Trend Bounce Scalping Strategy, setting up the system to detect trading signals using price interactions with the Envelopes indicator, confirmed by trend filters like [moving averages](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") and [RSI](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi"). We focused on building the infrastructure to monitor market conditions and identify potential trade opportunities, but we did not enable the system to execute trades or manage risks. Our roadmap for Part 19 shifts to activating trade execution and implementing risk management to ensure the strategy can act on signals effectively and safely.

Our architectural plan prioritizes a modular approach, creating a clear path to translate signals into trades while incorporating safeguards. We aim to develop a mechanism for placing buy and sell orders based on validated signals, alongside a risk management framework that sets stop-loss and take-profit levels, adjusts position sizes based on account balance, and limits overall losses to protect capital. This design will create a cohesive, automated scalping system. We will define some more classes to handle the trading and incorporate all the logic in the tick logic to bring everything to life. In a nutshell, this is what we aim to achieve.

![PLAN ARCHITECTURE](https://c.mql5.com/2/145/Screenshot_2025-05-28_011706.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Experts folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [interfaces](https://www.mql5.com/en/docs/basis/types/classes) and classes for enhanced signal generation, organization, and trade management. Let us start with the basic signal expression interface.

```
//--- Define interface for strategy signal expressions
interface IAdvisorStrategyExpression {
   bool Evaluate();                                       //--- Evaluate signal
   bool GetFireOnlyWhenReset();                           //--- Retrieve fire-only-when-reset flag
   void SetFireOnlyWhenReset(bool value);                 //--- Set fire-only-when-reset flag
   void ResetSignalValue();                               //--- Reset signal value
};

//--- Define base class for advisor signals
class ASSignal : public IAdvisorStrategyExpression {
private:
   bool _fireOnlyWhenReset;                               //--- Store fire-only-when-reset flag
   int _previousSignalValue;                              //--- Store previous signal value
   int _signalValue;                                      //--- Store current signal value

protected:
   //--- Declare pure virtual method for signal evaluation
   bool virtual EvaluateSignal() = 0;                      //--- Require derived classes to implement

public:
   //--- Initialize signal
   void ASSignal() {
      _fireOnlyWhenReset = false;                         //--- Set fire-only-when-reset to false
      _previousSignalValue = -1;                          //--- Set previous value to -1
      _signalValue = -1;                                  //--- Set current value to -1
   }

   //--- Evaluate signal
   bool Evaluate() {
      if (_signalValue == -1) {                           //--- Check if signal uncomputed
         _signalValue = EvaluateSignal();                 //--- Compute signal
      }
      if (_fireOnlyWhenReset) {                           //--- Check fire-only-when-reset
         return (_previousSignalValue <= 0 && _signalValue == 1); //--- Return true if signal transitions to 1
      } else {
         return _signalValue == 1;                        //--- Return true if signal is 1
      }
   }

   //--- Retrieve fire-only-when-reset flag
   bool GetFireOnlyWhenReset() {
      return _fireOnlyWhenReset;                          //--- Return flag
   }

   //--- Set fire-only-when-reset flag
   void SetFireOnlyWhenReset(bool value) {
      _fireOnlyWhenReset = value;                         //--- Set flag
   }

   //--- Reset signal value
   void ResetSignalValue() {
      _previousSignalValue = _signalValue;                //--- Store current as previous
      _signalValue = -1;                                  //--- Reset current value
   }
};
```

To establish a streamlined framework for handling trading signals, we focus on a modular and reliable signal evaluation system. We begin by defining the "IAdvisorStrategyExpression" interface, which we use to create a consistent blueprint for signal operations. This interface includes four key functions: the "Evaluate" function to determine if a signal is active, the "GetFireOnlyWhenReset" function to check a flag controlling signal triggers, the "SetFireOnlyWhenReset" function to modify this flag, and the "ResetSignalValue" function to clear the signal’s state for subsequent evaluations.

We then develop the "ASSignal" [class](https://www.mql5.com/en/docs/basis/types/classes), which we design to implement the "IAdvisorStrategyExpression" interface, serving as the foundation for specific signal types in our strategy. Within the "ASSignal" class, we define three private variables: "\_fireOnlyWhenReset" to control whether signals activate only after a reset, "\_previousSignalValue" to track the prior signal state, and "\_signalValue" to store the current state. We initialize these in the "ASSignal" [constructor function](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_ctors), setting "\_fireOnlyWhenReset" to false and both "\_previousSignalValue" and "\_signalValue" to -1, indicating an uncomputed state. Our "Evaluate" function checks if "\_signalValue" is -1, invoking the pure virtual "EvaluateSignal" function (to be defined in derived classes) to compute the signal, and returns true based on "\_fireOnlyWhenReset"—either when "\_signalValue" is 1 or when transitioning from a non-positive "\_previousSignalValue" to 1.

To manage signal behavior, we implement the "GetFireOnlyWhenReset" function to retrieve the "\_fireOnlyWhenReset" flag’s value and the "SetFireOnlyWhenReset" function to update it, allowing us to fine-tune when signals trigger. We also include the "ResetSignalValue" function, which we use to store "\_signalValue" in "\_previousSignalValue" and reset "\_signalValue" to -1, preparing for the next evaluation cycle. By marking "EvaluateSignal" as pure virtual, we require [derived classes](https://www.mql5.com/en/docs/basis/oop/inheritance) to provide specific signal logic, ensuring flexibility. We can now define some more classes for the signals management logic.

```
//--- Define class for managing trade signals
class TradeSignalCollection {
private:
   IAdvisorStrategyExpression* _tradeSignals[];           //--- Store array of signal pointers
   int _pointer;                                          //--- Track current iteration index
   int _size;                                             //--- Track number of signals

public:
   //--- Initialize empty signal collection
   void TradeSignalCollection() {
      _pointer = -1;                                      //--- Set initial pointer to -1
      _size = 0;                                          //--- Set initial size to 0
   }

   //--- Destructor to clean up signals
   void ~TradeSignalCollection() {
      for (int i = 0; i < ArraySize(_tradeSignals); i++) { //--- Iterate signals
         delete(_tradeSignals[i]);                        //--- Delete each signal
      }
   }

   //--- Add signal to collection
   void Add(IAdvisorStrategyExpression* item) {
      _size = _size + 1;                                  //--- Increment size
      ArrayResize(_tradeSignals, _size, 8);               //--- Resize array with reserve
      _tradeSignals[(_size - 1)] = item;                  //--- Store signal at last index
   }

   //--- Remove signal at index
   IAdvisorStrategyExpression* Remove(int index) {
      IAdvisorStrategyExpression* removed = NULL;         //--- Initialize removed signal as null
      if (index >= 0 && index < _size) {                  //--- Check valid index
         removed = _tradeSignals[index];                  //--- Store signal to remove
         for (int i = index; i < (_size - 1); i++) {      //--- Shift signals left
            _tradeSignals[i] = _tradeSignals[i + 1];      //--- Move signal
         }
         ArrayResize(_tradeSignals, ArraySize(_tradeSignals) - 1, 8); //--- Reduce array size
         _size = _size - 1;                               //--- Decrement size
      }
      return removed;                                     //--- Return removed signal or null
   }

   //--- Retrieve signal at index
   IAdvisorStrategyExpression* Get(int index) {
      if (index >= 0 && index < _size) {                  //--- Check valid index
         return _tradeSignals[index];                     //--- Return signal
      }
      return NULL;                                        //--- Return null for invalid index
   }

   //--- Retrieve number of signals
   int Count() {
      return _size;                                       //--- Return current size
   }

   //--- Reset iterator to start
   void Rewind() {
      _pointer = -1;                                      //--- Set pointer to -1
   }

   //--- Move to next signal
   IAdvisorStrategyExpression* Next() {
      _pointer++;                                         //--- Increment pointer
      if (_pointer == _size) {                            //--- Check if at end
         Rewind();                                        //--- Reset pointer
         return NULL;                                     //--- Return null
      }
      return Current();                                   //--- Return current signal
   }

   //--- Move to previous signal
   IAdvisorStrategyExpression* Prev() {
      _pointer--;                                         //--- Decrement pointer
      if (_pointer == -1) {                               //--- Check if before start
         return NULL;                                     //--- Return null
      }
      return Current();                                   //--- Return current signal
   }

   //--- Check if more signals exist
   bool HasNext() {
      return (_pointer < (_size - 1));                    //--- Return true if pointer is before end
   }

   //--- Retrieve current signal
   IAdvisorStrategyExpression* Current() {
      return _tradeSignals[_pointer];                     //--- Return signal at pointer
   }

   //--- Retrieve current iterator index
   int Key() {
      return _pointer;                                    //--- Return current pointer
   }
};

//--- Define class for managing trading signals
class AdvisorStrategy {
private:
   TradeSignalCollection* _openBuySignals;                //--- Store open Buy signals
   TradeSignalCollection* _openSellSignals;               //--- Store open Sell signals
   TradeSignalCollection* _closeBuySignals;               //--- Store close Buy signals
   TradeSignalCollection* _closeSellSignals;              //--- Store close Sell signals

   //--- Evaluate signal at specified level
   bool EvaluateASLevel(TradeSignalCollection* signals, int level) {
      if (level > 0 && level <= signals.Count()) {        //--- Check valid level
         return signals.Get(level - 1).Evaluate();        //--- Evaluate signal
      }
      return false;                                       //--- Return false for invalid level
   }

public:
   //--- Initialize strategy
   void AdvisorStrategy() {
      _openBuySignals = new TradeSignalCollection();      //--- Create open Buy signals collection
      _openSellSignals = new TradeSignalCollection();     //--- Create open Sell signals collection
      _closeBuySignals = new TradeSignalCollection();     //--- Create close Buy signals collection
      _closeSellSignals = new TradeSignalCollection();    //--- Create close Sell signals collection
   }

   //--- Destructor to clean up signals
   void ~AdvisorStrategy() {
      delete(_openBuySignals);                            //--- Delete open Buy signals
      delete(_openSellSignals);                           //--- Delete open Sell signals
      delete(_closeBuySignals);                           //--- Delete close Buy signals
      delete(_closeSellSignals);                          //--- Delete close Sell signals
   }

   //--- Retrieve trading advice
   bool GetAdvice(TradeAction tradeAction, int level) {
      if (tradeAction == OpenBuyAction) {                  //--- Check open Buy action
         return EvaluateASLevel(_openBuySignals, level);   //--- Evaluate Buy signal
      } else if (tradeAction == OpenSellAction) {          //--- Check open Sell action
         return EvaluateASLevel(_openSellSignals, level);  //--- Evaluate Sell signal
      } else if (tradeAction == CloseBuyAction) {          //--- Check close Buy action
         return EvaluateASLevel(_closeBuySignals, level);  //--- Evaluate close Buy signal
      } else if (tradeAction == CloseSellAction) {         //--- Check close Sell action
         return EvaluateASLevel(_closeSellSignals, level); //--- Evaluate close Sell signal
      } else {
         Alert("Unsupported TradeAction in Advisor Strategy. TradeAction: " + DoubleToStr(tradeAction)); //--- Log unsupported action
      }
      return false;                                       //--- Return false for invalid action
   }

   //--- Register open Buy signal
   void RegisterOpenBuy(IAdvisorStrategyExpression* openBuySignal, int level) {
      if (level <= _openBuySignals.Count()) {             //--- Check if level already set
         Alert("Register Open Buy failed: level already set."); //--- Log failure
         return;                                          //--- Exit
      }
      _openBuySignals.Add(openBuySignal);                 //--- Add signal
   }

   //--- Register open Sell signal
   void RegisterOpenSell(IAdvisorStrategyExpression* openSellSignal, int level) {
      if (level <= _openSellSignals.Count()) {            //--- Check if level already set
         Alert("Register Open Sell failed: level already set."); //--- Log failure
         return;                                          //--- Exit
      }
      _openSellSignals.Add(openSellSignal);               //--- Add signal
   }

   //--- Register close Buy signal
   void RegisterCloseBuy(IAdvisorStrategyExpression* closeBuySignal, int level) {
      if (level <= _closeBuySignals.Count()) {            //--- Check if level already set
         Alert("Register Close Buy failed: level already set."); //--- Log failure
         return;                                          //--- Exit
      }
      _closeBuySignals.Add(closeBuySignal);               //--- Add signal
   }

   //--- Register close Sell signal
   void RegisterCloseSell(IAdvisorStrategyExpression* closeSellSignal, int level) {
      if (level <= _closeSellSignals.Count()) {           //--- Check if level already set
         Alert("Register Close Sell failed: level already set."); //--- Log failure
         return;                                          //--- Exit
      }
      _closeSellSignals.Add(closeSellSignal);             //--- Add signal
   }

   //--- Retrieve number of signals for action
   int GetNumberOfExpressions(TradeAction tradeAction) {
      if (tradeAction == OpenBuyAction) {                 //--- Check open Buy action
         return _openBuySignals.Count();                  //--- Return Buy signal count
      } else if (tradeAction == OpenSellAction) {         //--- Check open Sell action
         return _openSellSignals.Count();                 //--- Return Sell signal count
      } else if (tradeAction == CloseBuyAction) {         //--- Check close Buy action
         return _closeBuySignals.Count();                 //--- Return close Buy signal count
      } else if (tradeAction == CloseSellAction) {        //--- Check close Sell action
         return _closeSellSignals.Count();                //--- Return close Sell signal count
      }
      return 0;                                           //--- Return 0 for invalid action
   }

   //--- Set fire-only-when-reset for all signals
   void SetFireOnlyWhenReset(bool value) {
      _openBuySignals.Rewind();                           //--- Reset Buy signals iterator
      while (_openBuySignals.Next() != NULL) {            //--- Iterate Buy signals
         _openBuySignals.Current().SetFireOnlyWhenReset(value); //--- Set flag
      }
      _openSellSignals.Rewind();                          //--- Reset Sell signals iterator
      while (_openSellSignals.Next() != NULL) {           //--- Iterate Sell signals
         _openSellSignals.Current().SetFireOnlyWhenReset(value); //--- Set flag
      }
      _closeBuySignals.Rewind();                          //--- Reset close Buy signals iterator
      while (_closeBuySignals.Next() != NULL) {           //--- Iterate close Buy signals
         _closeBuySignals.Current().SetFireOnlyWhenReset(value); //--- Set flag
      }
      _closeSellSignals.Rewind();                         //--- Reset close Sell signals iterator
      while (_closeSellSignals.Next() != NULL) {          //--- Iterate close Sell signals
         _closeSellSignals.Current().SetFireOnlyWhenReset(value); //--- Set flag
      }
   }

   //--- Handle tick event for signals
   void HandleTick() {
      _openBuySignals.Rewind();                           //--- Reset Buy signals iterator
      while (_openBuySignals.Next() != NULL) {            //--- Iterate Buy signals
         _openBuySignals.Current().ResetSignalValue();    //--- Reset signal value
         if (_openBuySignals.Current().GetFireOnlyWhenReset()) { //--- Check fire-only-when-reset
            _openBuySignals.Current().Evaluate();         //--- Evaluate signal
         }
      }
      _openSellSignals.Rewind();                          //--- Reset Sell signals iterator
      while (_openSellSignals.Next() != NULL) {           //--- Iterate Sell signals
         _openSellSignals.Current().ResetSignalValue();   //--- Reset signal value
         if (_openSellSignals.Current().GetFireOnlyWhenReset()) { //--- Check fire-only-when-reset
            _openSellSignals.Current().Evaluate();        //--- Evaluate signal
         }
      }
      _closeBuySignals.Rewind();                          //--- Reset close Buy signals iterator
      while (_closeBuySignals.Next() != NULL) {           //--- Iterate close Buy signals
         _closeBuySignals.Current().ResetSignalValue();   //--- Reset signal value
         if (_closeBuySignals.Current().GetFireOnlyWhenReset()) { //--- Check fire-only-when-reset
            _closeBuySignals.Current().Evaluate();        //--- Evaluate signal
         }
      }
      _closeSellSignals.Rewind();                         //--- Reset close Sell signals iterator
      while (_closeSellSignals.Next() != NULL) {          //--- Iterate close Sell signals
         _closeSellSignals.Current().ResetSignalValue();  //--- Reset signal value
         if (_closeSellSignals.Current().GetFireOnlyWhenReset()) { //--- Check fire-only-when-reset
            _closeSellSignals.Current().Evaluate();       //--- Evaluate signal
         }
      }
   }
};
```

Here, we implement more classes to manage the trading signals. Since we have already done some explanation on classes in the previous parts, we hope you already know the syntax of the classes, so we will just show the crucial parts. We create the "TradeSignalCollection" class, using "\_tradeSignals" to store "IAdvisorStrategyExpression" pointers, "\_pointer" for iteration, and "\_size" for signal count, initialized in the "TradeSignalCollection" [constructor](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_ctors). We add signals with the "Add" function, remove them with "Remove", and navigate using "Next", "Prev", "Current", "Rewind", and "HasNext" functions, with cleanup handled by the [destructor](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_dtors).

In the "AdvisorStrategy" [class](https://www.mql5.com/en/docs/basis/types/classes), we define "\_openBuySignals", "\_openSellSignals", "\_closeBuySignals", and "\_closeSellSignals" as "TradeSignalCollection" objects, set up in the "AdvisorStrategy" constructor. We evaluate signals with the "GetAdvice" and "EvaluateASLevel" functions, register signals via "RegisterOpenBuy", "RegisterOpenSell", "RegisterCloseBuy", and "RegisterCloseSell", and manage signal counts with "GetNumberOfExpressions". The "SetFireOnlyWhenReset" function applies reset flags, and "HandleTick" resets and evaluates signals using "ResetSignalValue" and "Evaluate", ensuring efficient signal handling for our strategy. We can now define classes for sell and buy signals at given levels. Let us start with the selling class.

```
//--- Define class for open Sell signal at level 1
class ASOpenSellLevel1 : public ASSignal {
protected:
   //--- Evaluate Sell signal
   bool EvaluateSignal() {
      Order* openOrder = _ea.GetWallet().GetMostRecentOpenOrder(); //--- Retrieve recent open order
      if (openOrder != NULL && openOrder.Type == ORDER_TYPE_BUY) { //--- Check if Buy order
         openOrder = NULL;                                //--- Clear if Buy to avoid conflict
      }
      if (((((openOrder != NULL ? TimeCurrent() - openOrder.OpenTime : EMPTY_VALUE) == EMPTY_VALUE) //--- Check no recent order
            && ((BidFunc.GetValue(0) < fn_iMA_SMA_4(Symbol(), 0)) //--- Check Bid below 4-period SMA
                && ((BidFunc.GetValue(0) > fn_iMA_SMA8(Symbol(), 0)) //--- Check Bid above 8-period SMA
                    && ((BidFunc.GetValue(0) < fn_iEnvelopes_ENV_UPPER(Symbol(), 0, 0)) //--- Check Bid below upper Envelope
                        && ((fn_iRSI_RSI(Symbol(), 1) < OpenSell_Const_0) //--- Check previous RSI below threshold
                            && (fn_iRSI_RSI(Symbol(), 0) >= OpenSell_Const_0) //--- Check current RSI above threshold
                           )
                       )
                   )
               )
           )
            || (((openOrder != NULL ? TimeCurrent() - openOrder.OpenTime : EMPTY_VALUE) != EMPTY_VALUE) //--- Check existing order
                && (BidFunc.GetValue(0) > ((openOrder != NULL ? openOrder.OpenPrice : EMPTY_VALUE) + (PipPoint * OpenSell_Const_1))) //--- Check Bid above open price plus offset
               )
          )) {
         return true;                                     //--- Return true for Sell signal
      }
      return false;                                       //--- Return false if no signal
   }

public:
   //--- Initialize Sell signal
   void ASOpenSellLevel1() {}                             //--- Empty constructor
};
```

We develop a sell signal logic using the "ASOpenSellLevel1" class, derived from the "ASSignal" class. In the protected "EvaluateSignal" function, we check the latest order via the "GetMostRecentOpenOrder" function from "\_ea"’s "Wallet", clearing it if it’s a buy order. We trigger a sell signal if no recent order exists and the bid price, from the "GetValue" function of "BidFunc", is below the 4-period SMA ("fn\_iMA\_SMA\_4"), above the 8-period SMA ("fn\_iMA\_SMA8"), below the upper Envelopes band ("fn\_iEnvelopes\_ENV\_UPPER"), with the prior RSI ("fn\_iRSI\_RSI") below "OpenSell\_Const\_0" and current RSI at or above it, or if an order exists and the bid exceeds the open price plus "PipPoint" times "OpenSell\_Const\_1".

We return true for a valid signal, else false. The "ASOpenSellLevel1" constructor function is empty, leveraging "ASSignal" initialization. We use the same logic for a buy signal.

```
//--- Define class for open Buy signal at level 1
class ASOpenBuyLevel1 : public ASSignal {
protected:
   //--- Evaluate Buy signal
   bool EvaluateSignal() {
      Order* openOrder = _ea.GetWallet().GetMostRecentOpenOrder();  //--- Retrieve most recent open order
      if (openOrder != NULL && openOrder.Type == ORDER_TYPE_SELL) { //--- Check if Sell order
         openOrder = NULL;                                          //--- Clear if Sell to avoid conflict
      }
      if (((((openOrder != NULL ? TimeCurrent() - openOrder.OpenTime : EMPTY_VALUE) == EMPTY_VALUE) //--- Check no recent order
            && ((AskFunc.GetValue(0) > fn_iMA_SMA_4(Symbol(), 0))                                   //--- Check Ask above 4-period SMA
                && ((AskFunc.GetValue(0) < fn_iMA_SMA8(Symbol(), 0))                                //--- Check Ask below 8-period SMA
                    && ((AskFunc.GetValue(0) > fn_iEnvelopes_ENV_LOW(Symbol(), 1, 0))               //--- Check Ask above lower Envelope
                        && ((fn_iRSI_RSI(Symbol(), 1) > OpenBuy_Const_0)                            //--- Check previous RSI above threshold
                            && (fn_iRSI_RSI(Symbol(), 0) <= OpenBuy_Const_0)                        //--- Check current RSI below threshold
                           )
                       )
                   )
               )
           )
            || (((openOrder != NULL ? TimeCurrent() - openOrder.OpenTime : EMPTY_VALUE) != EMPTY_VALUE) //--- Check existing order
                && (AskFunc.GetValue(0) < ((openOrder != NULL ? openOrder.OpenPrice : EMPTY_VALUE) - (PipPoint * OpenBuy_Const_1))) //--- Check Ask below open price minus offset
               )
          )) {
         return true;                                     //--- Return true for Buy signal
      }
      return false;                                       //--- Return false if no signal
   }

public:
   //--- Initialize Buy signal
   void ASOpenBuyLevel1() {}                              //--- Empty constructor
};
```

To define the buy logic, we simply use the same logic as we did for the sell. Then, we can add some inputs to control the signal trading logic.

```
//--- Define input group for trade and risk module settings
input string trademodule = "------TRADE/RISK MODULE------";//--- Label trade/risk module inputs
//--- Define input group for open Buy constants
input double LotSizePercentage = 1;                        //--- Set lot size as percentage of account balance (default: 1%)
input double OpenBuy_Const_0 = 11;                         //--- Set RSI threshold for Buy signal (default: 11)
input double OpenBuy_Const_1 = 10;                         //--- Set pip offset for additional Buy orders (default: 10 pips)
input double OpenSell_Const_0 = 89;                        //--- Set RSI threshold for Sell signal (default: 89)
input double OpenSell_Const_1 = 10;                        //--- Set pip offset for additional Sell orders (default: 10 pips)
```

After defining the [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) variables for signal generation, we can move on to defining logic for risk management and trade management after signal execution. To achieve that, we will need another [interface](https://www.mql5.com/en/docs/basis/types/classes) and class.

```
//--- Define interface for money management
interface IMoneyManager {
   double GetLotSize();                                   //--- Retrieve lot size
   int GetNextLevel(Wallet* wallet);                      //--- Retrieve next trading level
};

//--- Define class for money management
class MoneyManager : public IMoneyManager {
public:
   //--- Initialize money manager
   void MoneyManager(Wallet* wallet) {
      _minLot = MarketInfo_LibFunc(Symbol(), MODE_MINLOT);   //--- Set minimum lot size
      _maxLot = MarketInfo_LibFunc(Symbol(), MODE_MAXLOT);   //--- Set maximum lot size
      _lotStep = MarketInfo_LibFunc(Symbol(), MODE_LOTSTEP); //--- Set lot step
   }

   //--- Retrieve calculated lot size
   double GetLotSize() {
      double lotSize = NormalizeLots(NormalizeDouble(AccountInfoDouble(ACCOUNT_BALANCE) * 0.0001 * LotSizePercentage / 100.0, 2)); //--- Calculate lot size
      return lotSize;                                     //--- Return normalized lot size
   }

   //--- Retrieve next trading level
   int GetNextLevel(Wallet* wallet) {
      return wallet.GetOpenOrders().Count() + 1;          //--- Return next level based on open orders
   }

private:
   double _minLot;                                        //--- Store minimum lot size
   double _maxLot;                                        //--- Store maximum lot size
   double _lotStep;                                       //--- Store lot step

   //--- Normalize lot size to broker specifications
   double NormalizeLots(double lots) {
      lots = MathRound(lots / _lotStep) * _lotStep;       //--- Round to lot step
      if (lots < _minLot) lots = _minLot;                 //--- Enforce minimum lot
      else if (lots > _maxLot) lots = _maxLot;            //--- Enforce maximum lot
      return lots;                                        //--- Return normalized lot size
   }
};
```

To set up money management for the strategy, we define the "IMoneyManager" [interface](https://www.mql5.com/en/docs/basis/types/classes) with the "GetLotSize" and "GetNextLevel" functions to standardize trade sizing and level progression. In the "MoneyManager" class, implementing "IMoneyManager", we initialize "\_minLot", "\_maxLot", and "\_lotStep" variables in the "MoneyManager" constructor using the "MarketInfo\_LibFunc" function. Our "GetLotSize" function calculates lot size from account balance via [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble), adjusted by "LotSizePercentage", and normalized with "NormalizeLots" to fit broker rules.

The "GetNextLevel" function returns the count of open orders from "GetOpenOrders" plus one. The "NormalizeLots" function ensures valid lot sizes by rounding to "\_lotStep" and respecting "\_minLot" and "\_maxLot". This will help create a concise trade volume and level management system. We can now graduate to managing the trades and we will require another interface for that.

```
//--- Define enumeration for trading module demands
enum TradingModuleDemand {
   NoneDemand = 0,                                        //--- Represent no demand
   NoBuyDemand = 1,                                       //--- Prevent Buy orders
   NoSellDemand = 2,                                      //--- Prevent Sell orders
   NoOpenDemand = 4,                                      //--- Prevent all open orders
   OpenBuySellDemand = 8,                                 //--- Demand both Buy and Sell opens
   OpenBuyDemand = 16,                                    //--- Demand Buy open
   OpenSellDemand = 32,                                   //--- Demand Sell open
   CloseBuyDemand = 64,                                   //--- Demand Buy close
   CloseSellDemand = 128,                                 //--- Demand Sell close
   CloseBuySellDemand = 256                               //--- Demand both Buy and Sell closes
};

//--- Define interface for trading module signals
interface ITradingModuleSignal {
   string GetName();                                      //--- Retrieve signal name
   bool Evaluate(Order* openOrder = NULL);                //--- Evaluate signal
};

//--- Define interface for trading module values
interface ITradingModuleValue {
   string GetName();                                      //--- Retrieve value name
   double Evaluate(Order* openOrder = NULL);              //--- Evaluate value
};

//--- Define interface for trade strategy modules
interface ITradeStrategyModule {
   TradingModuleDemand Evaluate(Wallet* wallet, TradingModuleDemand demand, int level = 1); //--- Evaluate module
   void RegisterTradeSignal(ITradingModuleSignal* tradeSignal); //--- Register signal
};

//--- Define interface for open trade strategy modules
interface ITradeStrategyOpenModule : public ITradeStrategyModule {
   TradingModuleDemand EvaluateOpenSignals(Wallet* wallet, TradingModuleDemand demand, int requestedEvaluationLevel = 0); //--- Evaluate open signals
   TradingModuleDemand EvaluateCloseSignals(Wallet* wallet, TradingModuleDemand demand); //--- Evaluate close signals
};

//--- Define interface for close trade strategy modules
interface ITradeStrategyCloseModule : public ITradeStrategyModule {
   ORDER_GROUP_TYPE GetOrderGroupingType();                  //--- Retrieve grouping type
   void RegisterTradeValue(ITradingModuleValue* tradeValue); //--- Register value
};
```

Here, we lay the groundwork for managing trading decisions by defining a structured set of [enumerations](https://www.mql5.com/en/docs/basis/types/integer/enumeration) and interfaces. We start by creating the "TradingModuleDemand" enumeration, which we use to categorize trading actions and restrictions, such as "NoneDemand" for no action, "NoBuyDemand" and "NoSellDemand" to block specific order types, "OpenBuyDemand" and "OpenSellDemand" to initiate trades, and "CloseBuyDemand" and "CloseSellDemand" to close positions, among others. This enumeration will provide a clear way to signal the strategy’s intent for trade execution.

We then define the "ITradingModuleSignal" [interface](https://www.mql5.com/en/docs/basis/types/classes), which we design to standardize trading signal operations, including the "GetName" function to retrieve a signal’s identifier and the "Evaluate" function to assess whether a signal is active, optionally considering an open order. Similarly, we introduce the "ITradingModuleValue" interface with the "GetName" and "Evaluate" functions to manage numerical values associated with trading decisions, such as profit targets. For strategy modules, we create the "ITradeStrategyModule" interface, which includes the "Evaluate" function to process trading demands based on a "Wallet" object and the "RegisterTradeSignal" function to incorporate signals.

We extend this with the "ITradeStrategyOpenModule" interface, adding the "EvaluateOpenSignals" and "EvaluateCloseSignals" functions to handle signals for opening and closing trades, and the "ITradeStrategyCloseModule" interface, which includes the "GetOrderGroupingType" function for order grouping and the "RegisterTradeValue" function for value registration. Together, these components will form a flexible framework for coordinating trade actions in our strategy. We can now define management [classes](https://www.mql5.com/en/docs/basis/types/classes) based on these modules.

```
//--- Define class for managing trade strategy
class TradeStrategy {
public:
   ITradeStrategyCloseModule* CloseModules[];             //--- Store close modules

private:
   ITradeStrategyModule* _preventOpenModules[];           //--- Store prevent-open modules
   ITradeStrategyOpenModule* _openModule;                 //--- Store open module

   //--- Evaluate prevent-open modules
   TradingModuleDemand EvaluatePreventOpenModules(Wallet* wallet, TradingModuleDemand preventOpenDemand, int evaluationLevel = 1) {
      TradingModuleDemand preventOpenDemands[];                           //--- Declare prevent-open demands array
      ArrayResize(preventOpenDemands, ArraySize(_preventOpenModules), 8); //--- Resize array
      for (int i = 0; i < ArraySize(_preventOpenModules); i++) {          //--- Iterate modules
         preventOpenDemands[i] = _preventOpenModules[i].Evaluate(wallet, NoneDemand, evaluationLevel); //--- Evaluate module
      }
      return PreventOpenModuleBase::GetCombinedPreventOpenDemand(preventOpenDemands); //--- Return combined demand
   }

   //--- Evaluate close modules
   TradingModuleDemand EvaluateCloseModules(Wallet* wallet, TradingModuleDemand closeDemand, int evaluationLevel = 1) {
      TradingModuleDemand closeDemands[];                    //--- Declare close demands array
      ArrayResize(closeDemands, ArraySize(CloseModules), 8); //--- Resize array
      for (int i = 0; i < ArraySize(CloseModules); i++) {    //--- Iterate modules
         closeDemands[i] = CloseModules[i].Evaluate(wallet, NoneDemand, evaluationLevel); //--- Evaluate module
      }
      return CloseModuleBase::GetCombinedCloseDemand(closeDemands); //--- Return combined demand
   }

   //--- Evaluate close conditions for TP/SL
   void EvaluateCloseConditions(Wallet* wallet, TradingModuleDemand signalDemand) {
      OrderCollection* openOrders = wallet.GetOpenOrders(); //--- Retrieve open orders
      if (openOrders.Count() == 0) {                        //--- Check if no open orders
         return;                                            //--- Exit
      }
      double bid = Bid_LibFunc();                           //--- Retrieve Bid price
      double ask = Ask_LibFunc();                           //--- Retrieve Ask price
      for (int i = openOrders.Count() - 1; i >= 0; i--) {   //--- Iterate open orders
         Order* order = openOrders.Get(i);                  //--- Get order
         bool closeSignal = (order.Type == OP_BUY && signalDemand == CloseBuyDemand) ||   //--- Check Buy close signal
                            (order.Type == OP_SELL && signalDemand == CloseSellDemand) || //--- Check Sell close signal
                            signalDemand == CloseBuySellDemand;                           //--- Check Buy/Sell close signal
         bool closeManualSLTP = AllowManualTPSLChanges && ((order.StopLossManual != 0 && order.Type == OP_BUY && bid <= order.StopLossManual) ||    //--- Check manual Buy SL
                                                          (order.StopLossManual != 0 && order.Type == OP_SELL && ask >= order.StopLossManual) ||    //--- Check manual Sell SL
                                                          (order.TakeProfitManual != 0 && order.Type == OP_BUY && bid >= order.TakeProfitManual) || //--- Check manual Buy TP
                                                          (order.TakeProfitManual != 0 && order.Type == OP_SELL && ask <= order.TakeProfitManual)); //--- Check manual Sell TP
         bool fullOrderClose = closeSignal || closeManualSLTP; //--- Determine full close
         OrderCloseInfo* activePartialCloseCloseInfo = NULL;   //--- Initialize partial close info
         if (!fullOrderClose) {                                //--- Check if not full close
            if (!AllowManualTPSLChanges || order.StopLossManual == 0) { //--- Check manual SL
               for (int cli = 0; cli < ArraySize(order.CloseInfosSL); cli++) { //--- Iterate SL info
                  if (order.CloseInfosSL[cli].IsOld) continue; //--- Skip old info
                  if (order.CloseInfosSL[cli].IsClosePriceSLHit(order.Type, ask, bid)) { //--- Check SL hit
                     if (activePartialCloseCloseInfo == NULL || order.CloseInfosSL[cli].Percentage > activePartialCloseCloseInfo.Percentage) { //--- Check higher percentage
                        activePartialCloseCloseInfo = order.CloseInfosSL[cli]; //--- Set active info
                     }
                  }
               }
            }
            if (!AllowManualTPSLChanges || order.TakeProfitManual == 0) {      //--- Check manual TP
               for (int cli = 0; cli < ArraySize(order.CloseInfosTP); cli++) { //--- Iterate TP info
                  if (order.CloseInfosTP[cli].IsOld) continue;                 //--- Skip old info
                  if (order.CloseInfosTP[cli].IsClosePriceTPHit(order.Type, ask, bid)) { //--- Check TP hit
                     if (activePartialCloseCloseInfo == NULL || order.CloseInfosTP[cli].Percentage > activePartialCloseCloseInfo.Percentage) { //--- Check higher percentage
                        activePartialCloseCloseInfo = order.CloseInfosTP[cli]; //--- Set active info
                     }
                  }
               }
            }
            fullOrderClose = activePartialCloseCloseInfo != NULL && activePartialCloseCloseInfo.Percentage == 100; //--- Check if full close
         }
         if (fullOrderClose) {                            //--- Handle full close
            TradingModuleDemand finalPreventOpenAdvice = EvaluatePreventOpenModules(wallet, NoneDemand, 0);      //--- Evaluate prevent-open
            TradingModuleDemand openDemand = _openModule.EvaluateOpenSignals(wallet, finalPreventOpenAdvice, 1); //--- Evaluate open signals
            int orderTypeOfOpeningOrder = wallet.GetOpenOrders().Get(0).Type;                                    //--- Get first order type
            if ((orderTypeOfOpeningOrder == ORDER_TYPE_BUY && openDemand == OpenBuyDemand) ||                    //--- Check Buy re-entry
                (orderTypeOfOpeningOrder == ORDER_TYPE_SELL && openDemand == OpenSellDemand) ||                  //--- Check Sell re-entry
                (openDemand == OpenBuySellDemand)) {      //--- Check Buy/Sell re-entry
               return;                                    //--- Block close to prevent re-entry
            }
            wallet.SetOpenOrderToPendingClose(order);      //--- Move order to pending close
         } else if (activePartialCloseCloseInfo != NULL) { //--- Handle partial close
            Order* partialCloseOrder = order.SplitOrder(activePartialCloseCloseInfo.Percentage); //--- Split order
            if (partialCloseOrder.Lots < 1e-13) {          //--- Check if last piece
               delete(partialCloseOrder);                  //--- Delete split order
               wallet.SetOpenOrderToPendingClose(order);   //--- Move to pending close
            } else {
               partialCloseOrder.ParentOrder = order;      //--- Link to parent order
               if (wallet.AddPendingCloseOrder(partialCloseOrder)) { //--- Add to pending close
                  activePartialCloseCloseInfo.IsOld = true; //--- Mark info as old
               }
            }
         }
      }
   }

public:
   //--- Initialize trade strategy
   void TradeStrategy(ITradeStrategyOpenModule* openModule) {
      _openModule = openModule;                           //--- Set open module
   }

   //--- Destructor to clean up strategy
   void ~TradeStrategy() {
      for (int i = ArraySize(_preventOpenModules) - 1; i >= 0; i--) { //--- Iterate prevent-open modules
         delete(_preventOpenModules[i]);                  //--- Delete module
      }
      delete(_openModule);                                //--- Delete open module
      for (int i = ArraySize(CloseModules) - 1; i >= 0; i--) { //--- Iterate close modules
         delete(CloseModules[i]);                         //--- Delete module
      }
   }

   //--- Evaluate trading strategy
   void Evaluate(Wallet* wallet) {
      int orderCount = wallet.GetOpenOrders().Count();    //--- Retrieve open order count
      TradingModuleDemand finalPreventOpenAdvice = EvaluatePreventOpenModules(wallet, NoneDemand, orderCount + 1); //--- Evaluate prevent-open
      if (orderCount > 0) {                               //--- Check if orders exist
         EvaluateCloseModules(wallet, NoneDemand);        //--- Evaluate close modules
         TradingModuleDemand signalDemand = _openModule.EvaluateCloseSignals(wallet, finalPreventOpenAdvice); //--- Evaluate close signals
         EvaluateCloseConditions(wallet, signalDemand);   //--- Evaluate close conditions
      }
      _openModule.Evaluate(wallet, finalPreventOpenAdvice, 0); //--- Evaluate open module
   }

   //--- Register prevent-open module
   void RegisterPreventOpenModule(ITradeStrategyModule* preventOpenModule) {
      int size = ArraySize(_preventOpenModules);          //--- Get current array size
      ArrayResize(_preventOpenModules, size + 1, 8);      //--- Resize array
      _preventOpenModules[size] = preventOpenModule;      //--- Add module
   }

   //--- Register close module
   void RegisterCloseModule(ITradeStrategyCloseModule* closeModule) {
      int size = ArraySize(CloseModules);                 //--- Get current array size
      ArrayResize(CloseModules, size + 1, 8);             //--- Resize array
      CloseModules[size] = closeModule;                   //--- Add module
   }
};

//--- Define base class for module calculations
class ModuleCalculationsBase {
public:
   //--- Calculate profit for order collection
   static double CalculateOrderCollectionProfit(OrderCollection &orders, ORDER_PROFIT_CALCULATION_TYPE calculationType) {
      double collectionProfit = 0;                        //--- Initialize profit
      for (int i = 0; i < orders.Count(); i++) {          //--- Iterate orders
         Order* order = orders.Get(i);                    //--- Get order
         collectionProfit += CalculateOrderProfit(order, calculationType); //--- Add order profit
      }
      return collectionProfit;                            //--- Return total profit
   }

   //--- Calculate profit for single order
   static double CalculateOrderProfit(Order* order, ORDER_PROFIT_CALCULATION_TYPE calculationType) {
      if (calculationType == Pips) {                      //--- Check pips calculation
         return order.CalculateProfitPips();              //--- Return profit in pips
      } else if (calculationType == Money) {              //--- Check money calculation
         return order.CalculateProfitCurrency();          //--- Return profit in currency
      } else if (calculationType == EquityPercentage) {   //--- Check equity percentage
         return order.CalculateProfitEquityPercentage();  //--- Return profit as percentage
      } else {
         Alert("Can't execute CalculateOrderCollectionProfit. Unknown calculationType: " + IntegerToString(calculationType)); //--- Log error
         return 0;                                        //--- Return 0 for invalid type
      }
   }
};

//--- Define base class for open trade modules
class OpenModuleBase : public ITradeStrategyOpenModule {
protected:
   AdvisorStrategy* _advisorStrategy;                     //--- Store advisor strategy
   IMoneyManager* _moneyManager;                          //--- Store money manager

   //--- Create new order
   Order* OpenOrder(ENUM_ORDER_TYPE orderType, bool mustBeVisibleOnChart) {
      Order* order = new Order(mustBeVisibleOnChart);     //--- Create new order
      order.SymbolCode = Symbol();                        //--- Set symbol
      order.Type = orderType;                             //--- Set order type
      order.MagicNumber = MagicNumber;                    //--- Set magic number
      order.Lots = _moneyManager.GetLotSize();            //--- Set lot size
      if (order.Type == ORDER_TYPE_BUY) {                 //--- Check Buy order
         order.OpenPrice = Ask_LibFunc();                 //--- Set open price to Ask
         order.StopLoss = -DBL_MAX;                       //--- Set initial SL to minimum
         order.TakeProfit = DBL_MAX;                      //--- Set initial TP to maximum
      } else if (order.Type == ORDER_TYPE_SELL) {         //--- Check Sell order
         order.OpenPrice = Bid_LibFunc();                 //--- Set open price to Bid
         order.StopLoss = DBL_MAX;                        //--- Set initial SL to maximum
         order.TakeProfit = -DBL_MAX;                     //--- Set initial TP to minimum
      }
      order.LowestProfitPips = DBL_MAX;                   //--- Set initial lowest profit
      order.HighestProfitPips = -DBL_MAX;                 //--- Set initial highest profit
      order.Comment = OrderComment;                       //--- Set order comment
      OrderRepository::CalculateAndSetCommision(order);   //--- Calculate and set commission
      return order;                                       //--- Return order
   }

public:
   //--- Initialize open module
   void OpenModuleBase(AdvisorStrategy* advisorStrategy, IMoneyManager* moneyManager) {
      _advisorStrategy = advisorStrategy;                 //--- Set advisor strategy
      _moneyManager = moneyManager;                       //--- Set money manager
   }

   //--- Retrieve trade actions
   void GetTradeActions(Wallet* wallet, TradingModuleDemand preventOpenDemand, TradeAction& result[]) {
      TradeAction tempresult[];                             //--- Declare temporary actions array
      if (wallet.GetOpenOrders().Count() > 0) {             //--- Check if open orders exist
         Order* firstOrder = wallet.GetOpenOrders().Get(0); //--- Get first open order
         if (firstOrder.Type == ORDER_TYPE_BUY) {           //--- Check if Buy order
            ArrayResize(tempresult, ArraySize(tempresult) + 1, 8); //--- Resize array
            tempresult[0] = OpenBuyAction;                  //--- Add open Buy action
            ArrayResize(tempresult, ArraySize(tempresult) + 1, 8); //--- Resize array
            tempresult[1] = CloseBuyAction;                 //--- Add close Buy action
         } else if (firstOrder.Type == ORDER_TYPE_SELL) {   //--- Check if Sell order
            ArrayResize(tempresult, ArraySize(tempresult) + 1, 8); //--- Resize array
            tempresult[0] = OpenSellAction;                 //--- Add open Sell action
            ArrayResize(tempresult, ArraySize(tempresult) + 1, 8); //--- Resize array
            tempresult[1] = CloseSellAction;                //--- Add close Sell action
         } else {
            Alert("Unsupported ordertype. Ordertype: " + DoubleToStr(firstOrder.Type)); //--- Log error
         }
      } else {
         ArrayResize(tempresult, ArraySize(tempresult) + 1, 8); //--- Resize array
         tempresult[0] = OpenBuyAction;                   //--- Add open Buy action
         ArrayResize(tempresult, ArraySize(tempresult) + 1, 8); //--- Resize array
         tempresult[1] = OpenSellAction;                  //--- Add open Sell action
      }
      for (int i = 0; i < ArraySize(tempresult); i++) {   //--- Iterate actions
         if ((preventOpenDemand == NoOpenDemand && (tempresult[i] == OpenBuyAction || tempresult[i] == OpenSellAction)) || //--- Check no open demand
             (preventOpenDemand == NoBuyDemand && tempresult[i] == OpenBuyAction) || //--- Check no Buy demand
             (preventOpenDemand == NoSellDemand && tempresult[i] == OpenSellAction)) { //--- Check no Sell demand
            continue;                                     //--- Skip action
         }
         ArrayResize(result, ArraySize(result) + 1, 8);   //--- Resize result array
         result[ArraySize(result) - 1] = tempresult[i];   //--- Add action
      }
   }

   //--- Register trade signal (empty implementation)
   virtual void RegisterTradeSignal(ITradingModuleSignal* tradeSignal) {} //--- Do nothing

   //--- Combine open demands
   static TradingModuleDemand GetCombinedOpenDemand(TradingModuleDemand &openDemands[]) {
      TradingModuleDemand result = NoneDemand;            //--- Initialize result
      for (int i = 0; i < ArraySize(openDemands); i++) {  //--- Iterate demands
         if (result == OpenBuySellDemand) {               //--- Check if Buy/Sell demand
            return OpenBuySellDemand;                     //--- Return Buy/Sell demand
         }
         if (openDemands[i] == OpenBuySellDemand) {       //--- Check if demand is Buy/Sell
            result = OpenBuySellDemand;                   //--- Set Buy/Sell demand
         } else if (result == NoneDemand && openDemands[i] == OpenBuyDemand) { //--- Check Buy demand
            result = OpenBuyDemand;                       //--- Set Buy demand
         } else if (result == NoneDemand && openDemands[i] == OpenSellDemand) { //--- Check Sell demand
            result = OpenSellDemand;                      //--- Set Sell demand
         } else if (result == OpenBuyDemand && openDemands[i] == OpenSellDemand) { //--- Check mixed demands
            result = OpenBuySellDemand;                   //--- Set Buy/Sell demand
         } else if (result == OpenSellDemand && openDemands[i] == OpenBuyDemand) { //--- Check mixed demands
            result = OpenBuySellDemand;                   //--- Set Buy/Sell demand
         }
      }
      return result;                                      //--- Return combined demand
   }

   //--- Combine close demands
   static TradingModuleDemand GetCombinedCloseDemand(TradingModuleDemand &closeDemands[]) {
      TradingModuleDemand result = NoneDemand;            //--- Initialize result
      for (int i = 0; i < ArraySize(closeDemands); i++) { //--- Iterate demands
         if (result == CloseBuySellDemand) {              //--- Check if Buy/Sell demand
            return CloseBuySellDemand;                    //--- Return Buy/Sell demand
         }
         if (closeDemands[i] == CloseBuySellDemand) {     //--- Check if demand is Buy/Sell
            result = CloseBuySellDemand;                  //--- Set Buy/Sell demand
         } else if (result == NoneDemand && closeDemands[i] == CloseBuyDemand) { //--- Check Buy demand
            result = CloseBuyDemand;                      //--- Set Buy demand
         } else if (result == NoneDemand && closeDemands[i] == CloseSellDemand) { //--- Check Sell demand
            result = CloseSellDemand;                     //--- Set Sell demand
         } else if (result == CloseBuyDemand && closeDemands[i] == CloseSellDemand) { //--- Check mixed demands
            result = CloseBuySellDemand;                  //--- Set Buy/Sell demand
         } else if (result == CloseSellDemand && closeDemands[i] == CloseBuyDemand) { //--- Check mixed demands
            result = CloseBuySellDemand;                  //--- Set Buy/Sell demand
         }
      }
      return result;                                      //--- Return combined demand
   }

   //--- Retrieve number of open orders
   int GetNumberOfOpenOrders(Wallet* wallet) {
      return wallet.GetOpenOrders().Count();              //--- Return open order count
   }
};
```

We build the trading strategy framework using the "TradeStrategy" class, storing "ITradeStrategyCloseModule" objects in "CloseModules", "ITradeStrategyModule" objects in "\_preventOpenModules", and an "ITradeStrategyOpenModule" in "\_openModule". We initialize "\_openModule" in the "TradeStrategy" [constructor](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_ctors) and clean up modules in the [destructor](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_dtors). Our "Evaluate" function assesses open orders via "GetOpenOrders", evaluates "\_preventOpenModules" with "EvaluatePreventOpenModules" and "GetCombinedPreventOpenDemand", and processes "CloseModules" with "EvaluateCloseModules" and "GetCombinedCloseDemand".

The "EvaluateCloseConditions" function checks order types against "CloseBuyDemand" or "CloseSellDemand", verifies manual stop-loss/take-profit using "Ask\_LibFunc" and "Bid\_LibFunc", and handles closures with "SetOpenOrderToPendingClose" or "SplitOrder". We register modules with "RegisterPreventOpenModule" and "RegisterCloseModule". The "ModuleCalculationsBase" class computes profits via "CalculateOrderCollectionProfit" and "CalculateOrderProfit", while "OpenModuleBase" creates orders with "OpenOrder" and determines actions with "GetTradeActions", merging demands using "GetCombinedOpenDemand" and "GetCombinedCloseDemand" for a cohesive system.

We are now set to take trades, and to ensure we don't overtrade, we can have an external variable to control the maximum number of trades per instance.

```
//--- Define input for maximum open orders
input int MaxNumberOfOpenOrders1 = 1;                     //--- Set maximum number of open orders (default: 1)
```

After setting the trade count restriction variable, we can now define a [class](https://www.mql5.com/en/docs/basis/types/classes) to open multiple modules.

```
//--- Define class for multiple open module
class MultipleOpenModule_1 : public OpenModuleBase {
protected:
   TradingModuleDemand previousSignalDemand;              //--- Store previous signal demand

public:
   //--- Initialize multiple open module
   void MultipleOpenModule_1(AdvisorStrategy* advisorStrategy, MoneyManager* moneyManager)
      : OpenModuleBase(advisorStrategy, moneyManager) {
      _advisorStrategy.SetFireOnlyWhenReset(true);         //--- Configure signals to fire only when reset
   }

   //--- Evaluate and act on signals
   TradingModuleDemand Evaluate(Wallet* wallet, TradingModuleDemand preventOpenDemand, int level) {
      TradingModuleDemand newSignalsDemand = EvaluateSignals(wallet, preventOpenDemand, level); //--- Evaluate signals
      if (newSignalsDemand != NoneDemand) {                //--- Check if demand exists
         EvaluateOpenConditions(wallet, newSignalsDemand); //--- Evaluate open conditions
      }
      return newSignalsDemand;                             //--- Return new signal demand
   }

   //--- Evaluate open signals without acting
   TradingModuleDemand EvaluateOpenSignals(Wallet* wallet, TradingModuleDemand preventOpenDemand, int requestedEvaluationLevel) {
      TradingModuleDemand openDemands[];                   //--- Declare open demands array
      TradeAction tradeActionsToEvaluate[];                //--- Declare actions to evaluate
      GetTradeActions(wallet, preventOpenDemand, tradeActionsToEvaluate); //--- Retrieve actions
      AddPreviousDemandTradeActionIfMissing(tradeActionsToEvaluate); //--- Add previous demand actions
      int level;                                           //--- Declare level
      if (requestedEvaluationLevel == 0) {                 //--- Check if level unspecified
         level = _moneyManager.GetNextLevel(wallet);       //--- Set level based on orders
      } else {
         level = requestedEvaluationLevel;                 //--- Use specified level
      }
      for (int i = 0; i < ArraySize(tradeActionsToEvaluate); i++) { //--- Iterate actions
         if (tradeActionsToEvaluate[i] == CloseBuyAction || tradeActionsToEvaluate[i] == CloseSellAction) { //--- Skip close actions
            continue;                                      //--- Move to next
         }
         if (requestedEvaluationLevel == 0) {              //--- Check if level unspecified
            level = GetTopLevel(tradeActionsToEvaluate[i], level); //--- Cap level
            if (wallet.GetOpenOrders().Count() >= MaxNumberOfOpenOrders1) { //--- Check order limit
               level += 1;                                 //--- Increment level
            }
         }
         if (_advisorStrategy.GetAdvice(tradeActionsToEvaluate[i], level)) { //--- Check if action advised
            if (tradeActionsToEvaluate[i] == OpenBuyAction) { //--- Check Buy action
               int size = ArraySize(openDemands);             //--- Get current size
               int newSize = size + 1;                        //--- Calculate new size
               ArrayResize(openDemands, newSize, 8);          //--- Resize array
               openDemands[newSize - 1] = OpenBuyDemand;      //--- Add Buy demand
            } else if (tradeActionsToEvaluate[i] == OpenSellAction) { //--- Check Sell action
               int size = ArraySize(openDemands);             //--- Get current size
               int newSize = size + 1;                        //--- Calculate new size
               ArrayResize(openDemands, newSize, 8);          //--- Resize array
               openDemands[newSize - 1] = OpenSellDemand;     //--- Add Sell demand
            }
         }
      }
      TradingModuleDemand combinedOpenSignalDemand = OpenModuleBase::GetCombinedOpenDemand(openDemands); //--- Combine open demands
      TradingModuleDemand multiOrderOpenSignal = GetOpenDemandBasedOnPreviousOpenDemand(combinedOpenSignalDemand, level - 1); //--- Adjust for previous demand
      multiOrderOpenSignal = FilterPreventOpenDemand(multiOrderOpenSignal, preventOpenDemand); //--- Filter prevent-open demands
      return multiOrderOpenSignal;                        //--- Return final open signal
   }

   //--- Retrieve trade actions (custom for multiple orders)
   void GetTradeActions(Wallet* wallet, TradingModuleDemand preventOpenDemand, TradeAction& result[]) {
      if (wallet.GetOpenOrders().Count() > 0) {             //--- Check if open orders exist
         Order* firstOrder = wallet.GetOpenOrders().Get(0); //--- Get first open order
         if (firstOrder.Type == ORDER_TYPE_BUY) {           //--- Check if Buy order
            ArrayResize(result, ArraySize(result) + 1, 8);  //--- Resize array
            result[0] = OpenBuyAction;                      //--- Add open Buy action
            ArrayResize(result, ArraySize(result) + 1, 8);  //--- Resize array
            result[1] = CloseBuyAction;                     //--- Add close Buy action
         } else if (firstOrder.Type == ORDER_TYPE_SELL) {   //--- Check if Sell order
            ArrayResize(result, ArraySize(result) + 1, 8);  //--- Resize array
            result[0] = OpenSellAction;                     //--- Add open Sell action
            ArrayResize(result, ArraySize(result) + 1, 8);  //--- Resize array
            result[1] = CloseSellAction;                    //--- Add close Sell action
         } else {
            Alert("Unsupported ordertype");                 //--- Log error
         }
      } else {
         ArrayResize(result, ArraySize(result) + 1, 8);     //--- Resize array
         result[0] = OpenBuyAction;                         //--- Add open Buy action
         ArrayResize(result, ArraySize(result) + 1, 8);     //--- Resize array
         result[1] = OpenSellAction;                        //--- Add open Sell action
      }
   }

   //--- Evaluate close signals
   TradingModuleDemand EvaluateCloseSignals(Wallet* wallet, TradingModuleDemand preventOpenDemand) {
      TradingModuleDemand closeDemands[];                 //--- Declare close demands array
      TradeAction tradeActionsToEvaluate[];               //--- Declare actions to evaluate
      GetTradeActions(wallet, preventOpenDemand, tradeActionsToEvaluate); //--- Retrieve actions
      for (int i = 0; i < ArraySize(tradeActionsToEvaluate); i++) { //--- Iterate actions
         if (tradeActionsToEvaluate[i] != CloseBuyAction && tradeActionsToEvaluate[i] != CloseSellAction) { //--- Skip non-close actions
            continue;                                     //--- Move to next
         }
         if (_advisorStrategy.GetAdvice(tradeActionsToEvaluate[i], 1)) { //--- Check if action advised (level 1)
            if (tradeActionsToEvaluate[i] == CloseBuyAction) { //--- Check Buy close
               int size = ArraySize(closeDemands);          //--- Get current size
               int newSize = size + 1;                      //--- Calculate new size
               ArrayResize(closeDemands, newSize, 8);       //--- Resize array
               closeDemands[newSize - 1] = CloseBuyDemand;  //--- Add Buy close demand
            } else if (tradeActionsToEvaluate[i] == CloseSellAction) { //--- Check Sell close
               int size = ArraySize(closeDemands);          //--- Get current size
               int newSize = size + 1;                      //--- Calculate new size
               ArrayResize(closeDemands, newSize, 8);       //--- Resize array
               closeDemands[newSize - 1] = CloseSellDemand; //--- Add Sell close demand
            }
         }
      }
      TradingModuleDemand combinedCloseSignalDemand = OpenModuleBase::GetCombinedCloseDemand(closeDemands); //--- Combine close demands
      return combinedCloseSignalDemand;                   //--- Return combined demand
   }

private:
   //--- Evaluate open and close signals
   TradingModuleDemand EvaluateSignals(Wallet* wallet, TradingModuleDemand preventOpenDemand, int requestedEvaluationLevel) {
      TradingModuleDemand openDemands[];                  //--- Declare open demands array
      TradingModuleDemand closeDemands[];                 //--- Declare close demands array
      TradeAction tradeActionsToEvaluate[];               //--- Declare actions to evaluate
      GetTradeActions(wallet, preventOpenDemand, tradeActionsToEvaluate); //--- Retrieve actions
      AddPreviousDemandTradeActionIfMissing(tradeActionsToEvaluate); //--- Add previous demand actions
      int moneyManagementLevel;                           //--- Declare level
      if (requestedEvaluationLevel == 0) {                //--- Check if level unspecified
         moneyManagementLevel = _moneyManager.GetNextLevel(wallet); //--- Set level based on orders
      } else {
         moneyManagementLevel = requestedEvaluationLevel; //--- Use specified level
      }
      for (int i = 0; i < ArraySize(tradeActionsToEvaluate); i++) { //--- Iterate actions
         int tradeActionEvaluationLevel = GetTopLevel(tradeActionsToEvaluate[i], moneyManagementLevel); //--- Cap level
         if (wallet.GetOpenOrders().Count() >= MaxNumberOfOpenOrders1) { //--- Check order limit
            tradeActionEvaluationLevel += 1;              //--- Increment level
         }
         if (_advisorStrategy.GetAdvice(tradeActionsToEvaluate[i], tradeActionEvaluationLevel)) { //--- Check if action advised
            if (tradeActionsToEvaluate[i] == OpenBuyAction) { //--- Check Buy open
               int size = ArraySize(openDemands);         //--- Get current size
               int newSize = size + 1;                    //--- Calculate new size
               ArrayResize(openDemands, newSize, 8);      //--- Resize array
               openDemands[newSize - 1] = OpenBuyDemand;  //--- Add Buy demand
            } else if (tradeActionsToEvaluate[i] == OpenSellAction) { //--- Check Sell open
               int size = ArraySize(openDemands);         //--- Get current size
               int newSize = size + 1;                    //--- Calculate new size
               ArrayResize(openDemands, newSize, 8);      //--- Resize array
               openDemands[newSize - 1] = OpenSellDemand; //--- Add Sell demand
            } else if (tradeActionsToEvaluate[i] == CloseBuyAction) { //--- Check Buy close
               int size = ArraySize(closeDemands);        //--- Get current size
               int newSize = size + 1;                    //--- Calculate new size
               ArrayResize(closeDemands, newSize, 8);     //--- Resize array
               closeDemands[newSize - 1] = CloseBuyDemand; //--- Add Buy close demand
            } else if (tradeActionsToEvaluate[i] == CloseSellAction) { //--- Check Sell close
               int size = ArraySize(closeDemands);        //--- Get current size
               int newSize = size + 1;                    //--- Calculate new size
               ArrayResize(closeDemands, newSize, 8);     //--- Resize array
               closeDemands[newSize - 1] = CloseSellDemand; //--- Add Sell close demand
            }
         }
      }
      TradingModuleDemand combinedCloseSignalDemand = OpenModuleBase::GetCombinedCloseDemand(closeDemands); //--- Combine close demands
      if (combinedCloseSignalDemand != NoneDemand) {       //--- Check if close demand exists
         return combinedCloseSignalDemand;                //--- Return close demand
      }
      TradingModuleDemand combinedOpenSignalDemand = OpenModuleBase::GetCombinedOpenDemand(openDemands); //--- Combine open demands
      TradingModuleDemand multiOrderOpenSignal = GetOpenDemandBasedOnPreviousOpenDemand(combinedOpenSignalDemand, GetNumberOfOpenOrders(wallet)); //--- Adjust for previous demand
      previousSignalDemand = combinedOpenSignalDemand;    //--- Update previous demand
      multiOrderOpenSignal = FilterPreventOpenDemand(multiOrderOpenSignal, preventOpenDemand); //--- Filter prevent-open demands
      return multiOrderOpenSignal;                        //--- Return final signal
   }

   //--- Evaluate open conditions and add orders
   void EvaluateOpenConditions(Wallet* wallet, TradingModuleDemand signalDemand) {
      if (signalDemand == OpenBuySellDemand) {            //--- Check Buy/Sell demand
         return;                                          //--- Exit (hedging not supported)
      } else {
         double currentFreeMargin = AccountFreeMargin_LibFunc(); //--- Retrieve free margin
         double requiredMargin;                           //--- Declare required margin
         if (signalDemand == OpenBuyDemand) {             //--- Check Buy demand
            if (!MarginRequired(ORDER_TYPE_BUY, _moneyManager.GetLotSize(), requiredMargin)) { //--- Check margin
               return;                                    //--- Exit if margin check fails
            }
            if (currentFreeMargin < requiredMargin) {     //--- Check sufficient margin
               HandleErrors("Not enough free margin to open buy order with requested volume."); //--- Log error
               return;                                    //--- Exit
            }
            wallet.GetPendingOpenOrders().Add(OpenOrder(ORDER_TYPE_BUY, false)); //--- Add Buy order
         } else if (signalDemand == OpenSellDemand) {     //--- Check Sell demand
            if (!MarginRequired(ORDER_TYPE_SELL, _moneyManager.GetLotSize(), requiredMargin)) { //--- Check margin
               return;                                    //--- Exit if margin check fails
            }
            if (currentFreeMargin < requiredMargin) {     //--- Check sufficient margin
               HandleErrors("Not enough free margin to open sell order with requested volume."); //--- Log error
               return;                                    //--- Exit
            }
            wallet.GetPendingOpenOrders().Add(OpenOrder(ORDER_TYPE_SELL, false)); //--- Add Sell order
         }
      }
   }

   //--- Adjust open demand based on previous demand
   TradingModuleDemand GetOpenDemandBasedOnPreviousOpenDemand(TradingModuleDemand openDemand, int numberOfOpenOrders) {
      if (numberOfOpenOrders == 0 || previousSignalDemand == NoneDemand) { //--- Check no orders or no previous demand
         return openDemand;                               //--- Return current demand
      }
      if (previousSignalDemand == OpenBuyDemand && openDemand == OpenSellDemand) { //--- Check Buy to Sell switch
         return openDemand;                               //--- Allow Sell demand
      } else if (previousSignalDemand == OpenSellDemand && openDemand == OpenBuyDemand) { //--- Check Sell to Buy switch
         return openDemand;                               //--- Allow Buy demand
      }
      return NoneDemand;                                  //--- Block same-direction or mixed demands
   }

private:
   //--- Cap evaluation level
   int GetTopLevel(TradeAction tradeAction, int level) {
      int numberOfExpressions = _advisorStrategy.GetNumberOfExpressions(tradeAction); //--- Retrieve expression count
      if (level > numberOfExpressions) {                  //--- Check if level exceeds expressions
         level = numberOfExpressions;                     //--- Cap level
      }
      return level;                                       //--- Return capped level
   }

   //--- Add previous demand action if missing
   void AddPreviousDemandTradeActionIfMissing(TradeAction& result[]) {
      if (previousSignalDemand == NoneDemand) {           //--- Check if no previous demand
         return;                                          //--- Exit
      }
      bool foundPreviousDemand = false;                   //--- Initialize found flag
      if (previousSignalDemand == OpenBuyDemand) {        //--- Check Buy demand
         AddPreviousDemandTradeAction(result, OpenBuyDemand, OpenBuyAction); //--- Add Buy action
      } else if (previousSignalDemand == OpenSellDemand) { //--- Check Sell demand
         AddPreviousDemandTradeAction(result, OpenSellDemand, OpenSellAction); //--- Add Sell action
      } else if (previousSignalDemand == OpenBuySellDemand) { //--- Check Buy/Sell demand
         AddPreviousDemandTradeAction(result, OpenBuyDemand, OpenBuyAction); //--- Add Buy action
         AddPreviousDemandTradeAction(result, OpenSellDemand, OpenSellAction); //--- Add Sell action
      }
   }

   //--- Add specific previous demand action
   void AddPreviousDemandTradeAction(TradeAction& result[], TradingModuleDemand demand, TradeAction action) {
      bool foundPreviousDemand = false;                   //--- Initialize found flag
      if (previousSignalDemand == demand) {               //--- Check matching demand
         for (int i = 0; i < ArraySize(result); i++) {    //--- Iterate actions
            if (action == result[i]) {                    //--- Check if action exists
               foundPreviousDemand = true;                //--- Set found flag
            }
         }
         if (!foundPreviousDemand) {                      //--- Check if action missing
            ArrayResize(result, ArraySize(result) + 1, 8); //--- Resize array
            result[ArraySize(result) - 1] = action;       //--- Add action
         }
      }
   }

   //--- Filter prevent-open demands
   TradingModuleDemand FilterPreventOpenDemand(TradingModuleDemand multiOrderOpendDemand, TradingModuleDemand preventOpenDemand) {
      if (multiOrderOpendDemand == NoneDemand) {          //--- Check no demand
         return multiOrderOpendDemand;                    //--- Return no demand
      } else if (multiOrderOpendDemand == OpenBuyDemand && (preventOpenDemand == NoBuyDemand || preventOpenDemand == NoOpenDemand)) { //--- Check blocked Buy
         return NoneDemand;                               //--- Block Buy demand
      } else if (multiOrderOpendDemand == OpenSellDemand && (preventOpenDemand == NoSellDemand || preventOpenDemand == NoOpenDemand)) { //--- Check blocked Sell
         return NoneDemand;                               //--- Block Sell demand
      } else if (multiOrderOpendDemand == OpenBuySellDemand) { //--- Check Buy/Sell demand
         if (preventOpenDemand == NoBuyDemand) {          //--- Check no Buy
            return OpenSellDemand;                        //--- Allow Sell demand
         } else if (preventOpenDemand == NoSellDemand) {  //--- Check no Sell
            return OpenBuyDemand;                         //--- Allow Buy demand
         } else if (preventOpenDemand == NoOpenDemand) {  //--- Check no open
            return NoneDemand;                            //--- Block all demands
         }
      }
      return multiOrderOpendDemand;                       //--- Return unfiltered demand
   }
};
```

Here, we develop a module for managing multiple trade openings using the "MultipleOpenModule\_1" class, derived from "OpenModuleBase". We initialize it with the "MultipleOpenModule\_1" [constructor](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_ctors), setting "\_advisorStrategy" to trigger signals only after reset via "SetFireOnlyWhenReset" and storing previous demands in "previousSignalDemand". Our "Evaluate" function assesses signals with "EvaluateSignals" and acts on valid demands using "EvaluateOpenConditions", returning the demand.

In "EvaluateOpenSignals", we retrieve actions with "GetTradeActions", add prior demands via "AddPreviousDemandTradeActionIfMissing", and set the level using "\_moneyManager"’s "GetNextLevel" or a specified value. We iterate actions, skipping "CloseBuyAction" and "CloseSellAction", cap levels with "GetTopLevel", and check order limits with "GetOpenOrders" and "MaxNumberOfOpenOrders1". Valid signals from "\_advisorStrategy"’s "GetAdvice" add "OpenBuyDemand" or "OpenSellDemand" to an array, combined with "GetCombinedOpenDemand", adjusted by "GetOpenDemandBasedOnPreviousOpenDemand", and filtered by "FilterPreventOpenDemand".

The "EvaluateCloseSignals" function similarly evaluates "CloseBuyAction" and "CloseSellAction", adding "CloseBuyDemand" or "CloseSellDemand" and combining with "GetCombinedCloseDemand".

In "EvaluateOpenConditions", we verify the margin with "AccountFreeMargin\_LibFunc" and "MarginRequired", adding buy or sell orders via "OpenOrder" if sufficient, or logging errors with "HandleErrors". We use a similar logic to close the open orders, set take profit, and stop loss, as well as have the corresponding inputs. Upon execution, we now have the following inputs.

![CONTROL INPUTS](https://c.mql5.com/2/145/Screenshot_2025-05-27_180547.png)

We can now create a final class to handle all execution and management already implemented. To achieve that, we use the following logic.

```
//--- Define interface for trader
interface ITrader {
   void HandleTick();                                     //--- Handle tick event
   void Init();                                           //--- Initialize trader
   Wallet* GetWallet();                                   //--- Retrieve wallet
};

//--- Declare global trader pointer
ITrader *_ea;                                             //--- Store EA instance

//--- Define main Expert Advisor class
class EA : public ITrader {
private:
   bool _firstTick;                                       //--- Track first tick
   TradeStrategy* _tradeStrategy;                         //--- Store trade strategy
   AdvisorStrategy* _advisorStrategy;                     //--- Store advisor strategy
   IMoneyManager* _moneyManager;                          //--- Store money manager
   Wallet* _wallet;                                       //--- Store wallet

public:
   //--- Initialize EA
   void EA() {
      _firstTick = true;                                  //--- Set first tick flag
      _wallet = new Wallet();                             //--- Create wallet
      _wallet.SetLastClosedOrdersByTimeframe(DisplayOrderDuringTimeframe); //--- Set closed orders timeframe
      _advisorStrategy = new AdvisorStrategy();           //--- Create advisor strategy
      _advisorStrategy.RegisterOpenBuy(new ASOpenBuyLevel1(), 1); //--- Register Buy signal
      _advisorStrategy.RegisterOpenSell(new ASOpenSellLevel1(), 1); //--- Register Sell signal
      _moneyManager = new MoneyManager(_wallet);          //--- Create money manager
      _tradeStrategy = new TradeStrategy(new MultipleOpenModule_1(_advisorStrategy, _moneyManager)); //--- Create trade strategy
      _tradeStrategy.RegisterCloseModule(new TakeProfitCloseModule_1()); //--- Register TP module
      _tradeStrategy.RegisterCloseModule(new StopLossCloseModule_1()); //--- Register SL module
   }

   //--- Destructor to clean up EA
   void ~EA() {
      delete(_tradeStrategy);                             //--- Delete trade strategy
      delete(_moneyManager);                              //--- Delete money manager
      delete(_advisorStrategy);                           //--- Delete advisor strategy
      delete(_wallet);                                    //--- Delete wallet
   }

   //--- Initialize EA components
   void Init() {
      IsDemoLiveOrVisualMode = !MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE); //--- Set mode flag
      UnitsOneLot = MarketInfo_LibFunc(Symbol(), MODE_LOTSIZE); //--- Set lot size
      SetOrderGrouping();                                 //--- Configure order grouping
      _wallet.LoadOrdersFromBroker();                     //--- Load orders from broker
   }

   //--- Handle tick event
   void HandleTick() {
      if (MQLInfoInteger(MQL_TESTER) == 0) {              //--- Check if not in tester
         SyncOrders();                                    //--- Synchronize orders
      }
      if (AllowManualTPSLChanges) {                       //--- Check if manual TP/SL allowed
         SyncManualTPSLChanges();                         //--- Synchronize manual TP/SL
      }
      AskFunc.Evaluate();                                 //--- Update Ask price
      BidFunc.Evaluate();                                 //--- Update Bid price
      UpdateOrders();                                     //--- Update order profits
      if (!StopEA) {                                      //--- Check if EA not stopped
         _wallet.HandleTick();                            //--- Handle wallet tick
         _advisorStrategy.HandleTick();                   //--- Handle strategy tick
         if (_wallet.GetPendingOpenOrders().Count() == 0 && _wallet.GetPendingCloseOrders().Count() == 0) { //--- Check no pending orders
            _tradeStrategy.Evaluate(_wallet);             //--- Evaluate strategy
         }
         if (ExecutePendingCloseOrders()) {               //--- Execute close orders
            if (!ExecutePendingOpenOrders()) {            //--- Execute open orders
               HandleErrors(StringFormat("Open (all) order(s) failed. Please check EA %d and look at the Journal and Expert tab.", MagicNumber)); //--- Log error
            }
         } else {
            HandleErrors(StringFormat("Close (all) order(s) failed! Please check EA %d and look at the Journal and Expert tab.", MagicNumber)); //--- Log error
         }
      } else {
         if (ExecutePendingCloseOrders()) {               //--- Execute close orders
            _wallet.SetAllOpenOrdersToPendingClose();     //--- Move open orders to pending close
         } else {
            HandleErrors(StringFormat("Close (all) order(s) failed! Please check EA %d and look at the Journal and Expert tab.", MagicNumber)); //--- Log error
         }
      }
      if (_firstTick) {                                   //--- Check if first tick
         _firstTick = false;                              //--- Clear first tick flag
      }
   }

   //--- Retrieve wallet
   Wallet* GetWallet() {
      return _wallet;                                     //--- Return wallet
   }

private:
   //--- Configure order grouping
   void SetOrderGrouping() {
      int size = ArraySize(_tradeStrategy.CloseModules);  //--- Get close modules size
      ORDER_GROUP_TYPE groups[];                          //--- Declare groups array
      ArrayResize(groups, size);                          //--- Resize array
      for (int i = 0; i < ArraySize(_tradeStrategy.CloseModules); i++) { //--- Iterate modules
         groups[i] = _tradeStrategy.CloseModules[i].GetOrderGroupingType(); //--- Set grouping type
      }
      _wallet.ActivateOrderGroups(groups);                //--- Activate groups
   }

   //--- Synchronize orders with broker
   void SyncOrders() {
      OrderCollection* currentOpenOrders = OrderRepository::GetOpenOrders(MagicNumber, NULL, Symbol()); //--- Retrieve open orders
      if (currentOpenOrders.Count() != (_wallet.GetOpenOrders().Count() + _wallet.GetPendingCloseOrders().Count())) { //--- Check order mismatch
         Print("(Manual) orderchanges detected" + " (found in MT: " + IntegerToString(currentOpenOrders.Count()) + " and in wallet: " + IntegerToString(_wallet.GetOpenOrders().Count()) + "), resetting EA, loading open orders."); //--- Log mismatch
         _wallet.ResetOpenOrders();                       //--- Reset open orders
         _wallet.ResetPendingOrders();                    //--- Reset pending orders
         _wallet.LoadOrdersFromBroker();                  //--- Reload orders
      }
      delete(currentOpenOrders);                          //--- Delete orders collection
   }

   //--- Synchronize manual TP/SL changes
   void SyncManualTPSLChanges() {
      _wallet.GetOpenOrders().Rewind();                   //--- Reset orders iterator
      while (_wallet.GetOpenOrders().HasNext()) {         //--- Iterate orders
         Order* order = _wallet.GetOpenOrders().Next();   //--- Get order
         uint lineFindResult = ObjectFind(ChartID(), IntegerToString(order.Ticket) + "_SL"); //--- Find SL line
         if (lineFindResult != UINT_MAX) {                //--- Check if SL line exists
            double currentPosition = ObjectGetDouble(ChartID(), IntegerToString(order.Ticket) + "_SL", OBJPROP_PRICE); //--- Get SL position
            if ((order.StopLossManual == 0 && currentPosition != order.GetClosestSL()) || //--- Check manual SL change
                (order.StopLossManual != 0 && currentPosition != order.StopLossManual)) { //--- Check manual SL mismatch
               order.StopLossManual = currentPosition;       //--- Update manual SL
            }
         }
         lineFindResult = ObjectFind(ChartID(), IntegerToString(order.Ticket) + "_TP"); //--- Find TP line
         if (lineFindResult != UINT_MAX) {                //--- Check if TP line exists
            double currentPosition = ObjectGetDouble(ChartID(), IntegerToString(order.Ticket) + "_TP", OBJPROP_PRICE); //--- Get TP position
            if ((order.TakeProfitManual == 0 && currentPosition != order.GetClosestTP()) || //--- Check manual TP change
                (order.TakeProfitManual != 0 && currentPosition != order.TakeProfitManual)) { //--- Check manual TP mismatch
               order.TakeProfitManual = currentPosition;     //--- Update manual TP
            }
         }
      }
   }

   //--- Update order profits
   void UpdateOrders() {
      _wallet.GetOpenOrders().Rewind();                   //--- Reset orders iterator
      while (_wallet.GetOpenOrders().HasNext()) {         //--- Iterate orders
         Order* order = _wallet.GetOpenOrders().Next();   //--- Get order
         double pipsProfit = order.CalculateProfitPips(); //--- Calculate profit
         order.CurrentProfitPips = pipsProfit;            //--- Update current profit
         if (pipsProfit < order.LowestProfitPips) {       //--- Check if lowest profit
            order.LowestProfitPips = pipsProfit;          //--- Update lowest profit
         } else if (pipsProfit > order.HighestProfitPips) { //--- Check if highest profit
            order.HighestProfitPips = pipsProfit;         //--- Update highest profit
         }
      }
   }

   //--- Execute pending close orders
   bool ExecutePendingCloseOrders() {
      OrderCollection* pendingCloseOrders = _wallet.GetPendingCloseOrders(); //--- Retrieve pending close orders
      int ordersToCloseCount = pendingCloseOrders.Count(); //--- Get count
      if (ordersToCloseCount == 0) {                      //--- Check if no orders
         return true;                                     //--- Return true
      }
      if (_wallet.AreOrdersBeingOpened()) {               //--- Check if orders being opened
         return true;                                     //--- Return true
      }
      int ordersCloseSuccessCount = 0;                    //--- Initialize success count
      for (int i = ordersToCloseCount - 1; i >= 0; i--) { //--- Iterate orders
         Order* pendingCloseOrder = pendingCloseOrders.Get(i); //--- Get order
         if (pendingCloseOrder.IsAwaitingDealExecution) { //--- Check if awaiting execution
            ordersCloseSuccessCount++;                    //--- Increment success count
            continue;                                     //--- Move to next
         }
         bool success;                                    //--- Declare success flag
         if (AccountMarginMode == ACCOUNT_MARGIN_MODE_RETAIL_NETTING) { //--- Check netting mode
            Order* reversedOrder = new Order(pendingCloseOrder, false); //--- Create reversed order
            reversedOrder.Type = pendingCloseOrder.Type == ORDER_TYPE_BUY ? ORDER_TYPE_SELL : ORDER_TYPE_BUY; //--- Set opposite type
            success = OrderRepository::OpenOrder(reversedOrder); //--- Open reversed order
            if (success) {                                //--- Check if successful
               pendingCloseOrder.Ticket = reversedOrder.Ticket; //--- Update ticket
            }
            delete(reversedOrder);                        //--- Delete reversed order
         } else {
            success = OrderRepository::ClosePosition(pendingCloseOrder); //--- Close position
         }
         if (success) {                                   //--- Check if successful
            ordersCloseSuccessCount++;                    //--- Increment success count
         }
      }
      return ordersCloseSuccessCount == ordersToCloseCount; //--- Return true if all successful
   }

   //--- Execute pending open orders
   bool ExecutePendingOpenOrders() {
      OrderCollection* pendingOpenOrders = _wallet.GetPendingOpenOrders(); //--- Retrieve pending open orders
      int ordersToOpenCount = pendingOpenOrders.Count();  //--- Get count
      if (ordersToOpenCount == 0) {                       //--- Check if no orders
         return true;                                     //--- Return true
      }
      int ordersOpenSuccessCount = 0;                     //--- Initialize success count
      for (int i = ordersToOpenCount - 1; i >= 0; i--) {  //--- Iterate orders
         Order* order = pendingOpenOrders.Get(i);         //--- Get order
         if (order.IsAwaitingDealExecution) {             //--- Check if awaiting execution
            ordersOpenSuccessCount++;                     //--- Increment success count
            continue;                                     //--- Move to next
         }
         bool isTradeContextFree = false;                 //--- Initialize trade context flag
         double StartWaitingTime = GetTickCount();        //--- Start timer
         while (true) {                                   //--- Wait for trade context
            if (MQL5InfoInteger(MQL5_TRADE_ALLOWED)) {    //--- Check if trade allowed
               isTradeContextFree = true;                 //--- Set trade context free
               break;                                     //--- Exit loop
            }
            int MaxWaiting_sec = 10;                      //--- Set max wait time
            if (IsStopped()) {                            //--- Check if EA stopped
               HandleErrors("The expert was stopped by a user action."); //--- Log error
               break;                                     //--- Exit loop
            }
            if (GetTickCount() - StartWaitingTime > MaxWaiting_sec * 1000) { //--- Check if timeout
               HandleErrors(StringFormat("The (%d seconds) waiting time exceeded. Trade not allowed: EA disabled, market closed or trade context still not free.", MaxWaiting_sec)); //--- Log error
               break;                                     //--- Exit loop
            }
            Sleep(100);                                   //--- Wait briefly
         }
         if (!isTradeContextFree) {                       //--- Check if trade context not free
            if (!_wallet.CancelPendingOpenOrder(order)) { //--- Attempt to cancel order
               HandleErrors("Failed to cancel an order (because it couldn't open). Please see the Journal and Expert tab in Metatrader for more information."); //--- Log error
            }
            continue;                                     //--- Move to next
         }
         bool success = OrderRepository::OpenOrder(order); //--- Open order
         if (success) {                                    //--- Check if successful
            ordersOpenSuccessCount++;                      //--- Increment success count
         } else {
            if (!_wallet.CancelPendingOpenOrder(order)) {  //--- Attempt to cancel order
               HandleErrors("Failed to cancel an order (because it couldn't open). Please see the Journal and Expert tab in Metatrader for more information."); //--- Log error
            }
         }
      }
      return ordersOpenSuccessCount == ordersToOpenCount;  //--- Return true if all successful
   }
};
```

Finally, we establish the core logic control using the "EA" [class](https://www.mql5.com/en/docs/basis/types/classes), which implements the "ITrader" interface with "HandleTick", "Init", and "GetWallet" functions. You should now be fully aware of these functions that we already did. We define private variables "\_firstTick", "\_tradeStrategy", "\_advisorStrategy", "\_moneyManager", and "\_wallet" to manage state and components. In the "EA" [constructor](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_ctors), we set "\_firstTick" to true, initialize "\_wallet" with a new "Wallet" object, configure its timeframe via "SetLastClosedOrdersByTimeframe", and create "\_advisorStrategy" with a new "AdvisorStrategy" object, registering "ASOpenBuyLevel1" and "ASOpenSellLevel1" signals using "RegisterOpenBuy" and "RegisterOpenSell".

We also instantiate "\_moneyManager" with a "MoneyManager" object and "\_tradeStrategy" with a "TradeStrategy" object containing a "MultipleOpenModule\_1", registering "TakeProfitCloseModule\_1" and "StopLossCloseModule\_1" via "RegisterCloseModule". The [destructor](https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_dtors) cleans up all components.

Our "Init" function sets mode flags with [MQLInfoInteger](https://www.mql5.com/en/docs/check/mqlinfointeger), configures lot size via "MarketInfo\_LibFunc", calls "SetOrderGrouping" to activate order groups using "GetOrderGroupingType", and loads orders with "LoadOrdersFromBroker". The "HandleTick" function manages ticks, syncing orders with "SyncOrders" if not in tester mode, updating manual TP/SL with "SyncManualTPSLChanges" if allowed, and refreshing prices via "AskFunc" and "BidFunc"’s "Evaluate" functions.

It updates profits with "UpdateOrders", processes wallet and strategy ticks with "HandleTick", evaluates "\_tradeStrategy" with "Evaluate" if no pending orders, and executes pending closes and opens with "ExecutePendingCloseOrders" and "ExecutePendingOpenOrders", logging errors with "HandleErrors" if failures occur. If the EA is stopped, it closes orders with "SetAllOpenOrdersToPendingClose". We can now call this on the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to manage the trades.

```
//--- Handle tick event
datetime LastActionTime = 0;                          //--- Track last action time
void OnTick() {
   string AccountServer = AccountInfoString(ACCOUNT_SERVER);                //--- Retrieve account server
   string AccountCurrency = AccountInfoString(ACCOUNT_CURRENCY);            //--- Retrieve account currency
   string AccountName = AccountInfoString(ACCOUNT_NAME);                    //--- Retrieve account name
   long AccountTradeMode = AccountInfoInteger(ACCOUNT_TRADE_MODE);          //--- Retrieve trade mode
   string ReadableAccountTrademode = "";                                    //--- Initialize readable trade mode
   if (AccountTradeMode == 0) ReadableAccountTrademode = "DEMO ACCOUNT";    //--- Set demo mode
   if (AccountTradeMode == 1) ReadableAccountTrademode = "CONTEST ACCOUNT"; //--- Set contest mode
   if (AccountTradeMode == 2) ReadableAccountTrademode = "REAL ACCOUNT";    //--- Set real mode
   long AccountLogin = AccountInfoInteger(ACCOUNT_LOGIN);                   //--- Retrieve account login
   string AccountCompany = AccountInfoString(ACCOUNT_COMPANY);              //--- Retrieve account company
   long AccountLeverage = AccountInfoInteger(ACCOUNT_LEVERAGE);             //--- Retrieve account leverage
   long AccountLimitOrders = AccountInfoInteger(ACCOUNT_LIMIT_ORDERS);      //--- Retrieve order limit
   double AccountMarginFree = AccountInfoDouble(ACCOUNT_MARGIN_FREE);       //--- Retrieve free margin
   bool AccountTradeAllowed = AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);    //--- Retrieve trade allowed
   bool AccountTradeExpert = AccountInfoInteger(ACCOUNT_TRADE_EXPERT);      //--- Retrieve expert allowed
   string ReadableAccountMarginMode = "";                                   //--- Initialize readable margin mode
   if (AccountMarginMode == 0) ReadableAccountMarginMode = "NETTING MODE";  //--- Set netting mode
   if (AccountMarginMode == 1) ReadableAccountMarginMode = "EXCHANGE MODE"; //--- Set exchange mode
   if (AccountMarginMode == 2) ReadableAccountMarginMode = "HEDGING MODE";  //--- Set hedging mode
   if (OneQuotePerBar) {                                 //--- Check one quote per bar
      datetime currentTime = iTime(_Symbol, _Period, 0); //--- Get current bar time
      if (LastActionTime == currentTime) {               //--- Check if same bar
         return;                                         //--- Exit
      } else {
         LastActionTime = currentTime;                   //--- Update last action time
      }
   }
   Error = NULL;                                         //--- Clear current error
   _ea.HandleTick();                                     //--- Handle tick
   if (IsDemoLiveOrVisualMode) {                         //--- Check visual mode
      MqlDateTime mql_datetime;                          //--- Declare datetime
      TimeCurrent(mql_datetime);                         //--- Get current time
      string comment = "\n" + (string)mql_datetime.year + "." + (string)mql_datetime.mon + "." + (string)mql_datetime.day + "   " + TimeToString(TimeCurrent(), TIME_SECONDS) + OrderInfoComment; //--- Build comment
      if (DisplayOnChartError) {                         //--- Check if error display enabled
         if (Error != NULL) comment += "\n      :: Current error : " + Error; //--- Add current error
         if (ErrorPreviousQuote != NULL) comment += "\n      :: Last error : " + ErrorPreviousQuote; //--- Add previous error
      }
      comment += "";                                     //--- Append empty line
      Comment("ACCOUNT SERVER:  ", AccountServer, "\n",  //--- Display account info
              "ACCOUNT CURRENCY:  ", AccountCurrency, "\n",
              "ACCOUNT NAME:  ", AccountName, "\n",
              "ACCOUNT TRADEMODE:  ", ReadableAccountTrademode, "\n",
              "ACCOUNT LOGIN:  ", AccountLogin, "\n",
              "ACCOUNT COMPANY:  ", AccountCompany, "\n",
              "ACCOUNT LEVERAGE:  ", AccountLeverage, "\n",
              "ACCOUNT LIMIT ORDERS:  ", AccountLimitOrders, "\n",
              "ACCOUNT MARGIN FREE:  ", AccountMarginFree, "\n",
              "ACCOUNT TRADING ALLOWED:  ", AccountTradeAllowed, "\n",
              "ACCOUNT EXPERT ALLOWED:  ", AccountTradeExpert, "\n",
              "ACCOUNT MARGIN ALLOWED:  ", ReadableAccountMarginMode);
   }
}
```

Here, we manage real-time market updates for the program through the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, which processes each new price tick. We begin by collecting account details using functions like "AccountInfoString" for the server, currency, name, and company, [AccountInfoInteger](https://www.mql5.com/en/docs/account/accountinfointeger) for trade mode, login, leverage, and order limits, and [AccountInfoDouble](https://www.mql5.com/en/docs/account/accountinfodouble) for free margin. We convert the trade mode into a readable string ("DEMO ACCOUNT", "CONTEST ACCOUNT", or "REAL ACCOUNT") and the margin mode, stored in "AccountMarginMode", into "NETTING MODE", "EXCHANGE MODE", or "HEDGING MODE" for clarity.

To control tick processing, we check the "OneQuotePerBar" flag, and if enabled, we use the [iTime](https://www.mql5.com/en/docs/series/itime) function to get the current bar’s start time, comparing it with "LastActionTime". If they match, we exit to avoid duplicate processing; otherwise, we update "LastActionTime". We clear any existing error with "Error" set to null and call the "HandleTick" function on "\_ea" to execute the strategy’s logic.

For display, we check "IsDemoLiveOrVisualMode", then use [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) and [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) to format the timestamp, building a comment string with "OrderInfoComment". If "DisplayOnChartError" is enabled, we append errors from "Error" and "ErrorPreviousQuote". Finally, we use the [Comment](https://www.mql5.com/en/docs/common/comment) function to display account details on the chart, ensuring real-time monitoring and feedback for the trading system. When we run the program, we have the following output.

![SIGNAL GENERATION](https://c.mql5.com/2/145/Screenshot_2025-05-27_182313.png)

From the image, we can see that we can generate and execute a trade, in this case, a buy trade. What we now need to do is manage the trade, and to achieve that, we need to keep track of the transactions, so we will the [OnTradeTransaction](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) event handler to listen to successfully opened trades and manage them as follows.

```
//--- Handle trade transactions
void OnTradeTransaction(const MqlTradeTransaction& trans, const MqlTradeRequest& request, const MqlTradeResult& result) {
   switch (trans.type) {                               //--- Handle transaction type
   case TRADE_TRANSACTION_DEAL_ADD: {                  //--- Handle deal addition
      datetime end = TimeCurrent();                    //--- Get current server time
      datetime start = end - PeriodSeconds(PERIOD_D1); //--- Set start time (1 day ago)
      HistorySelect(start, end + PeriodSeconds(PERIOD_D1)); //--- Select history
      int dealsTotal = HistoryDealsTotal();            //--- Get total deals
      if (dealsTotal == 0) {                           //--- Check if no deals
         Print("No deals found");                      //--- Log message
         return;                                       //--- Exit
      }
      ulong orderTicketId = HistoryDealGetInteger(trans.deal, DEAL_ORDER); //--- Get order ticket
      CDealInfo dealInfo;                              //--- Declare deal info
      dealInfo.Ticket(trans.deal);                     //--- Set deal ticket
      ENUM_DEAL_ENTRY deal_entry = dealInfo.Entry();   //--- Get deal entry type
      bool found = false;                              //--- Initialize found flag
      if (deal_entry == DEAL_ENTRY_IN) {               //--- Handle deal entry
         OrderCollection* pendingOpenOrders = _ea.GetWallet().GetPendingOpenOrders(); //--- Retrieve pending open orders
         for (int i = 0; i < pendingOpenOrders.Count(); i++) { //--- Iterate orders
            Order* order = pendingOpenOrders.Get(i);   //--- Get order
            if (order.Ticket == orderTicketId) {       //--- Check matching ticket
               found = true;                           //--- Set found flag
               order.OpenTime = dealInfo.Time();       //--- Set open time
               order.OpenPrice = trans.price;          //--- Set open price
               order.TradePrice = order.OpenPrice;     //--- Set trade price
               if (OrderFillingType == ORDER_FILLING_FOK) {                 //--- Check FOK filling
                  order.OrderFilledLots += trans.volume;                    //--- Add volume
                  if (MathAbs(order.Lots - order.OrderFilledLots) < 1e-5) { //--- Check if fully filled
                     order.IsAwaitingDealExecution = false;                 //--- Clear execution flag
                     order.Lots = order.OrderFilledLots;                    //--- Update lots
                     order.TradeVolume = order.Lots;                        //--- Update trade volume
                     _ea.GetWallet().SetPendingOpenOrderToOpen(order);      //--- Move to open
                     Print(StringFormat("Execution done for order (%d) by EA (%d)", orderTicketId, MagicNumber)); //--- Log success
                  }
               } else {
                  order.IsAwaitingDealExecution = false; //--- Clear execution flag
                  bool actualVolumeDiffers = MathAbs(order.Lots - trans.volume) > 1e-5; //--- Check volume difference
                  order.OrderFilledLots += trans.volume; //--- Add volume
                  order.Lots = order.OrderFilledLots;    //--- Update lots
                  order.TradeVolume = order.Lots;        //--- Update trade volume
                  if (actualVolumeDiffers) {             //--- Check if volume differs
                     Print("Broker executed volume differs from requested volume. Executed volume: " + DoubleToStr(trans.volume)); //--- Log difference
                     OrderRepository::CalculateAndSetCommision(order); //--- Recalculate commission
                  }
                  _ea.GetWallet().SetPendingOpenOrderToOpen(order); //--- Move to open
                  Print(StringFormat("Execution done for order (%d) by EA (%d)", orderTicketId, MagicNumber)); //--- Log success
               }
            }
         }
      } else if (deal_entry == DEAL_ENTRY_OUT) {      //--- Handle deal exit
         OrderCollection* pendingCloseOrders = _ea.GetWallet().GetPendingCloseOrders(); //--- Retrieve pending close orders
         for (int i = 0; i < pendingCloseOrders.Count(); i++) { //--- Iterate orders
            Order* order = pendingCloseOrders.Get(i); //--- Get order
            if (order.Ticket == orderTicketId) {      //--- Check matching ticket
               found = true;                          //--- Set found flag
               if (OrderFillingType == ORDER_FILLING_FOK) {   //--- Check FOK filling
                  order.OrderFilledLots += trans.volume;      //--- Add volume
                  if (MathAbs(order.Lots - order.OrderFilledLots) < 1e-5) { //--- Check if fully filled
                     order.IsAwaitingDealExecution = false;   //--- Clear execution flag
                     order.CloseTime = dealInfo.Time();       //--- Set close time
                     order.ClosePrice = trans.price;          //--- Set close price
                     if (order.MagicNumber == MagicNumber) {  //--- Check EA order
                        TotalCommission += order.Commission;  //--- Add commission
                     }
                     if (IsDemoLiveOrVisualMode) {      //--- Check visual mode
                        AnyChartObjectDelete(ChartID(), IntegerToString(order.Ticket) + "_TP"); //--- Delete TP line
                        AnyChartObjectDelete(ChartID(), IntegerToString(order.Ticket) + "_SL"); //--- Delete SL line
                     }
                     if (order.ParentOrder != NULL) {   //--- Check if parent order
                        order.ParentOrder.Paint();      //--- Redraw parent
                     }
                     _ea.GetWallet().SetPendingCloseOrderToClosed(order); //--- Move to closed
                     Print(StringFormat("Execution done for order (%d) by EA (%d)", orderTicketId, MagicNumber)); //--- Log success
                  }
               } else {
                  bool actualVolumeDiffers = MathAbs(order.Lots - trans.volume) > 1e-5; //--- Check volume difference
                  if (actualVolumeDiffers) {                           //--- Handle partial close
                     Print("Broker executed volume differs from requested volume.Requested volume: " + DoubleToStr(order.Lots) + ".Executed volume: " + DoubleToStr(trans.volume)); //--- Log difference
                     Order* remainderOrder = new Order(order, false);  //--- Create remainder order
                     remainderOrder.Ticket = 0;                        //--- Clear ticket
                     remainderOrder.Lots = order.Lots - trans.volume;  //--- Set remaining volume
                     remainderOrder.TradeVolume = remainderOrder.Lots; //--- Update trade volume
                     OrderRepository::CalculateAndSetCommision(remainderOrder); //--- Recalculate commission
                     _ea.GetWallet().GetPendingCloseOrders().Add(remainderOrder); //--- Add remainder
                     order.Lots = trans.volume;                        //--- Update original volume
                     order.TradeVolume = order.Lots;                   //--- Update trade volume
                     OrderRepository::CalculateAndSetCommision(order); //--- Recalculate commission
                  } else {
                     Print("Broker executed volume: " + DoubleToStr(trans.volume)); //--- Log volume
                  }
                  order.IsAwaitingDealExecution = false;   //--- Clear execution flag
                  order.CloseTime = dealInfo.Time();       //--- Set close time
                  order.ClosePrice = trans.price;          //--- Set close price
                  if (order.MagicNumber == MagicNumber) {  //--- Check EA order
                     TotalCommission += order.Commission;  //--- Add commission
                  }
                  if (IsDemoLiveOrVisualMode) {            //--- Check visual mode
                     AnyChartObjectDelete(ChartID(), IntegerToString(order.Ticket) + "_TP"); //--- Delete TP line
                     AnyChartObjectDelete(ChartID(), IntegerToString(order.Ticket) + "_SL"); //--- Delete SL line
                  }
                  _ea.GetWallet().SetPendingCloseOrderToClosed(order); //--- Move to closed
                  Print(StringFormat("Execution done for order (%d) by EA (%d)", orderTicketId, MagicNumber)); //--- Log success
               }
            }
         }
      }
      if (found) {                                         //--- Check if deal found
         Print("Updated order with deal info.");           //--- Log update
      } else if (trans.symbol == Symbol() && dealInfo.Magic() == MagicNumber) { //--- Check EA deal
         Print("Couldn't find deal info for place/done order"); //--- Log missing deal
      }
      break;
   }
   }
}
```

On the [OnTradeTransaction](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) event handler where we process [MqlTradeTransaction](https://www.mql5.com/en/docs/constants/structures/mqltradetransaction) events, we focus on [TRADE\_TRANSACTION\_DEAL\_ADD](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_transaction_type) transactions, where we retrieve a one-day history window with [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) and "PeriodSeconds" via [HistorySelect](https://www.mql5.com/en/docs/trading/historyselect). If no deals are found with "HistoryDealsTotal", we log a message with "Print" and exit. For each deal, we extract the order ticket using "HistoryDealGetInteger" and create a "CDealInfo" object with "Ticket" to check the entry type via "Entry".

For [DEAL\_ENTRY\_IN](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_type) (new trades), we iterate "GetPendingOpenOrders" from "\_ea"’s "Wallet", matching tickets to update orders with "Time" and "price" from "trans". If "OrderFillingType" is "ORDER\_FILLING\_FOK", we update "OrderFilledLots" and, if filled, clear "IsAwaitingDealExecution" and move the order to open with "SetPendingOpenOrderToOpen". Otherwise, we handle partial fills, logging volume differences with "Print" and updating via "CalculateAndSetCommision".

For "DEAL\_ENTRY\_OUT" (closures), we process "GetPendingCloseOrders", updating matching orders similarly, removing chart lines with "AnyChartObjectDelete" if "IsDemoLiveOrVisualMode", adding commissions to "TotalCommission", and moving orders to close with "SetPendingCloseOrderToClosed". Partial closes create a new order with "CalculateAndSetCommision" for the remaining volume. We log updates or errors with "Print", ensuring accurate trade tracking. Now upon compilation, we have the following output.

![DEAL EXECUTIONS](https://c.mql5.com/2/145/Screenshot_2025-05-27_183735.png)

From the image, we can see that we manage the trade executions and display the metadata on the window. The thing that remains is backtesting the program to ensure it works correctly, and that is handled in the next section.

### Backtesting and Optimization

Upon testing the program, we noticed some oversight that we had missed and that was the release of the program and its components when we removed the program from the chart. The issue was as follows.

![BACKTEST ISSUES](https://c.mql5.com/2/145/Screenshot_2025-05-27_185825.png)

From the image, we can see the illustration of the issues. To take care of the issues, we implemented the following logic in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler.

```
//--- Deinitialize Expert Advisor
void OnDeinit(const int reason) {
   Comment("");
   delete(_ea);                                        //--- Delete EA instance
   delete(AskFunc);                                    //--- Delete Ask function
   delete(BidFunc);                                    //--- Delete Bid function
}
```

We managed the cleanup process for the program using the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, which is triggered when the Expert Advisor is removed or the terminal is shut down. We began by clearing any chart comments with the [Comment](https://www.mql5.com/en/docs/common/comment) function, ensuring a clean chart display. Next, we released memory by deleting the "\_ea" object, which represents the main "EA" instance, using the [delete](https://www.mql5.com/en/docs/basis/operators/deleteoperator) operator. We also deleted the "AskFunc" and "BidFunc" objects, which handle price data retrieval, to free up resources.

This concise process ensured proper deinitialization, preventing memory leaks and maintaining system efficiency when the program was stopped. That solved everything and upon thorough optimization and backtesting, using the default settings and changing the trade or risk module settings to using 1%, 5, 30, 10, 60 inputs respectively, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/145/Screenshot_2025-05-28_011031.png)

Backtest report:

![REPORT](https://c.mql5.com/2/145/Screenshot_2025-05-28_010953.png)

### Conclusion

In conclusion, we have developed an [MQL5](https://www.mql5.com/) implementation of the Envelopes Trend Bounce Scalping Strategy, enabling automated trade execution driven by precise signals from Envelopes, Moving Averages, and [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") indicators, with integrated risk management for controlled position sizing and loss protection. Our modular framework, featuring dynamic signal evaluation, trade execution, and real-time order monitoring, provides a scalable foundation for scalping. You can further refine this program by adjusting signal thresholds, optimizing risk parameters, or incorporating additional indicators to align with your trading goals.

Disclaimer: This implementation is for educational purposes only. Trading involves substantial financial risks, and market volatility can lead to significant losses. Thorough backtesting and prudent risk management are essential before using this program in live trading environments.

By mastering these techniques, we can enhance the program’s adaptability and robustness or leverage its structure to develop other trading strategies, advancing the program's [trading algorithm](https://en.wikipedia.org/wiki/Algorithm "https://en.wikipedia.org/wiki/Algorithm").

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18298.zip "Download all attachments in the single ZIP archive")

[Envelopes\_Trend\_Bounce\_Scalping\_Part\_2.mq5](https://www.mql5.com/en/articles/download/18298/envelopes_trend_bounce_scalping_part_2.mq5 "Download Envelopes_Trend_Bounce_Scalping_Part_2.mq5")(501.39 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)
- [Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

**[Go to discussion](https://www.mql5.com/en/forum/488390)**

![Price Action Analysis Toolkit Development (Part 26): Pin Bar, Engulfing Patterns and RSI Divergence (Multi-Pattern) Tool](https://c.mql5.com/2/147/17962-price-action-analysis-toolkit-logo__1.png)[Price Action Analysis Toolkit Development (Part 26): Pin Bar, Engulfing Patterns and RSI Divergence (Multi-Pattern) Tool](https://www.mql5.com/en/articles/17962)

Aligned with our goal of developing practical price-action tools, this article explores the creation of an EA that detects pin bar and engulfing patterns, using RSI divergence as a confirmation trigger before generating any trading signals.

![MQL5 Wizard Techniques you should know (Part 68):  Using Patterns of TRIX and the Williams Percent Range with a Cosine Kernel Network](https://c.mql5.com/2/147/18305-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 68): Using Patterns of TRIX and the Williams Percent Range with a Cosine Kernel Network](https://www.mql5.com/en/articles/18305)

We follow up our last article, where we introduced the indicator pair of TRIX and Williams Percent Range, by considering how this indicator pairing could be extended with Machine Learning. TRIX and William’s Percent are a trend and support/ resistance complimentary pairing. Our machine learning approach uses a convolution neural network that engages the cosine kernel in its architecture when fine-tuning the forecasts of this indicator pairing. As always, this is done in a custom signal class file that works with the MQL5 wizard to assemble an Expert Advisor.

![ALGLIB library optimization methods (Part II)](https://c.mql5.com/2/99/Alglib_Library_Optimization_Techniques_PartI___LOGO__3.png)[ALGLIB library optimization methods (Part II)](https://www.mql5.com/en/articles/16164)

In this article, we will continue to study the remaining optimization methods from the ALGLIB library, paying special attention to their testing on complex multidimensional functions. This will allow us not only to evaluate the efficiency of each algorithm, but also to identify their strengths and weaknesses in different conditions.

![Developing a Replay System (Part 71): Getting the Time Right (IV)](https://c.mql5.com/2/99/Desenvolvendo_um_sistema_de_Replay_Parte_71___LOGO.png)[Developing a Replay System (Part 71): Getting the Time Right (IV)](https://www.mql5.com/en/articles/12335)

In this article, we will look at how to implement what was shown in the previous article related to our replay/simulation service. As in many other things in life, problems are bound to arise. And this case was no exception. In this article, we continue to improve things. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/18298&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049340600668498361)

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