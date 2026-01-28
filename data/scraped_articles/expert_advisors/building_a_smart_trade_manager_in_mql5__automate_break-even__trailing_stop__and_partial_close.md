---
title: Building a Smart Trade Manager in MQL5: Automate Break-Even, Trailing Stop, and Partial Close
url: https://www.mql5.com/en/articles/19911
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:27:59.019207
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/19911&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068267860117092205)

MetaTrader 5 / Expert Advisors


### Introduction

Trade management is one of the most overlooked aspects of trading. Many traders focus on finding great entries but forget that profits may be made or lost in how trades are managed after entry. Moving stop-losses, taking partial profits, or trailing a stop-loss often requires keen monitoring of trades and quick reactions that are challenging to maintain in a live market environment.

Manual trade management quickly becomes stressful, especially when monitoring several positions across different symbols. A trader may hesitate to move a stop-loss, adjust it too late, or forget it altogether. These small mistakes can turn winning trades into losses, leading to emotional decisions and inconsistent results over time.

Even the disciplined traders who follow their rules can find manual execution to be tiring. Constantly watching charts, calculating levels, and reacting to price movements creates fatigue and distraction. This reduces efficiency and takes away the calm focus that successful trading requires.

This article presents an automated solution to these problems. It will guide you through the creation of a fully functional Trade Manager Expert Advisor. The EA handles three essential aspects of trade management: automatically moving the stop-loss to break-even to prevent losses, trailing the stop-loss levels as the price moves in favor of a position, and closing partial profits to secure gains progressively.

### How the Smart Trade Manager EA is designed

The Smart Trade Manager, named AutoProtect, is designed as an Expert Advisor (EA) within the MQL5 environment. This is important to mention because MQL5 supports different program types, including scripts, indicators, services, and expert advisors. Since the purpose of AutoProtect is to monitor and manage open positions automatically, it must operate continuously in real time.

AutoProtect has three main functions that work together to protect trades and automate the trader's management process. These include the break-even function, the trailing-stop function, and the partial profits close function. Each of these functions can be enabled or disabled by the user, giving full flexibility to configure the EA according to personal trading preferences. For example, a trader who only wants to use the break-even function can simply disable the other two features.

The break-even function automatically moves the stop loss to the entry price (plus an optional buffer distance) when the trade reaches a defined profit level. This helps eliminate risk and secure the position once it moves favorably. The trailing stop function then takes over, adjusting the stop loss as the market continues in the profitable direction. It ensures that gains are protected while allowing trades to grow naturally. Finally, the partial close function locks in profits progressively by closing a portion of the position when a certain profit distance (in points) is reached.

AutoProtect manages only the trades of the chart where it is attached. This design decision is intentional. Each financial security has a unique point value, so managing trades universally across multiple instruments with a single configuration could create inconsistencies. Restricting trade management to one symbol ensures precise and consistent behavior. Additionally, users can configure the EA to manage trades based on a specific magic number. This is useful when a trader wants AutoProtect to manage only trades opened by a specific Expert Advisor.

It is also important to note that AutoProtect monitors and manages only the trades opened after it is launched on the chart. It does not react to older or already open positions at the time of initialization. This approach ensures that only trades created under its supervision are affected, avoiding unintended behavior.

The EA comes with several input parameters organized into logical groups for clarity and ease of configuration:

```
//+------------------------------------------------------------------+
//| User Input Variables                                             |
//+------------------------------------------------------------------+
input group "Information"
input bool  manageTradesByMagicNumber   = true;
input ulong tradeSelectionMagicNumber   = 254700680002;

input group "Break-Even"
input bool  enableBreakEven            = false;
input int   breakEvenTriggerPoints     = 50;
input int   breakEvenLockPoints        = 0;

input group "Trailing Stop"
input bool  enableTrailingStop         = false;
input int   trailingStartPoints        = 50;
input int   trailingStepPoints         = 20;
input int   trailingDistancePoints     = 100;

input group "Partial Close"
input bool   enablePartialClose        = false;
input int    partialCloseTriggerPoints = 100;
input double partialClosePercent       = 50.0;
```

Each parameter serves a specific role. For example, the breakEvenTriggerPoints defines when to activate the break-even move, and the breakEvenLockPoints determines how many extra points beyond the entry to lock as profit. In the trailing stop section, trailingStartPoints marks the profit level where trailing begins, trailingStepPoints defines how frequently the stop moves, and trailingDistancePoints represents how far the stop should stay behind the current price. The partialCloseTriggerPoints and partialClosePercent define how much of a trade to close when price moves a certain distance (specified in points) into profit.

One of the design rules of AutoProtect ensures that when both the break-even and trailing stop functions are enabled, the trailing stop starts immediately after the break-even condition has been met. This is enforced in the initialization logic as shown below:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   ...

   //--- If both BreakEven and Trailing are enabled,trailing should only start after break-even has triggered.
   if(enableBreakEven && enableTrailingStop){
      usefulTrailingStartPoints = breakEvenTriggerPoints;
   }

   ...

   return(INIT_SUCCEEDED);
}
```

Internally, the EA follows a modular programming approach with clean, well-commented code. Each function is written as a self-contained module. This design enhances readability, simplifies debugging, and allows future extensions without affecting existing features.

### Setting Up the Project

To begin developing the AutoProtect Expert Advisor, open MetaEditor, create a new Expert Advisor project, and name it AutoProtect.mq5. Once the project file is created, replace its contents with the following boilerplate code.

```
//+------------------------------------------------------------------+
//|                                                  AutoProtect.mq5 |
//|          Copyright 2025, MetaQuotes Ltd. Developer is Chacha Ian |
//|                          https://www.mql5.com/en/users/chachaian |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd. Developer is Chacha Ian"
#property link      "https://www.mql5.com/en/users/chachaian"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Libraries                                                        |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| User Input Variables                                             |
//+------------------------------------------------------------------+
input group "Information"
input bool  manageTradesByMagicNumber   = true;
input ulong tradeSelectionMagicNumber   = 254700680002;

input group "Break-Even"
input bool  enableBreakEven            = false;
input int   breakEvenTriggerPoints     = 50;
input int   breakEvenLockPoints        = 0;

input group "Trailing Stop"
input bool  enableTrailingStop         = false;
input int   trailingStartPoints        = 50;
input int   trailingStepPoints         = 20;
input int   trailingDistancePoints     = 100;

input group "Partial Close"
input bool   enablePartialClose        = false;
input int    partialCloseTriggerPoints = 100;
input double partialClosePercent       = 50.0;

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade Trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){
   // ---
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason){
   //---
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
   //---
}

//+------------------------------------------------------------------+
//| Trade Transaction Events handler                                 |
//+------------------------------------------------------------------+
void OnTradeTransaction(
   const MqlTradeTransaction &trans,
   const MqlTradeRequest     &request,
   const MqlTradeResult      &result){
      // ----
}
//+------------------------------------------------------------------+
```

This structure provides all the essential event handlers needed for an Expert Advisor in MQL5, including initialization, deinitialization, tick handling, and trade transaction tracking.

To enable the automation of trade actions such as modifying stops or closing positions, we first include the Trade.mqh standard library. This library provides access to the CTrade class, which simplifies order management through its built-in methods like PositionModify and PositionClosePartial.

Next, we create a global instance of the CTrade class named Trade. This object will be used throughout the Expert Advisor to execute all trade operations, ensuring that we can interact with the trading environment efficiently and consistently.

By defining CTrade Trade globally, any functions in our EA can easily access it, whether it's adjusting stop-loss levels, activating break-even logic, or trailing positions.

Trade Selection Logic

The AutoProtect Expert Advisor manages only trades that are opened on the same chart where it is launched. This design decision ensures consistent behavior, as different symbols have unique point values that could lead to calculation errors if handled universally. Restricting trade management to the chart symbol guarantees precision and prevents conflicts across different markets.

To provide more control, the EA also allows filtering trades using a specific magic number. When enabled, the Expert Advisor only manages trades that match the specified number. This feature is particularly helpful when a trader runs multiple EAs on the same symbol but wants AutoProtect to handle trades from one specific strategy only.

Another key aspect of the design is that AutoProtect reacts only to newly opened positions after it is launched. It does not attempt to manage pre-existing trades that were active before the EA was attached to the chart. This approach avoids inconsistencies and ensures that each managed position is properly registered and tracked from the moment of its creation.

To make this possible, a custom structure called MqlTradeInfo is defined. This structure holds essential trade details such as ticket number, position type, entry price, stop levels, and trailing parameters. We then maintain an array of these structures globally to keep track of all positions managed by the EA.

```
...

//+------------------------------------------------------------------+
//| User Input Variables                                             |
//+------------------------------------------------------------------+
input group "Information"
input bool  manageTradesByMagicNumber   = true;
input ulong tradeSelectionMagicNumber   = 254700680002;

...

//+------------------------------------------------------------------+
//| Data Structures                                                  |
//+------------------------------------------------------------------+
struct MqlTradeInfo{
   ulong ticket;
   ENUM_POSITION_TYPE positionType;
   double openPrice;
   double originalStopLevel;
   double currentStopLevel;
   double nextStopLevel;
   double originalTargetLevel;
   double nextTrailTriggerPrice;
   bool   isMovedToBreakEven;
   bool   isPartialProfitsSecured;
};

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade Trade;
MqlTradeInfo tradeInfo[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   //---
   return(INIT_SUCCEEDED);
}

...
```

Next, the OnTradeTransaction event handler is used to detect when new trades are opened. Whenever a new position is created, the EA confirms it through the trading history and then records its details in the tradeInfo array. Only these tracked trades will later be processed by the trailing stop function.

```
...

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
   //---
}

//+------------------------------------------------------------------+
//| Trade Transaction Events handler                                 |
//+------------------------------------------------------------------+
void OnTradeTransaction(
   const MqlTradeTransaction &trans,
   const MqlTradeRequest     &request,
   const MqlTradeResult      &result){

   // When a new position is opened
   if(trans.type   != TRADE_TRANSACTION_DEAL_ADD){
      return;
   }

   if(trans.symbol != _Symbol){
      return;
   }

   if(trans.deal   == 0){
      return;
   }

   bool selected = false;

   HistorySelect(TimeCurrent() - 60, TimeCurrent());
   for(int i = 0; i < 6; i++){
      if(HistoryDealSelect(trans.deal)){
         selected = true;
         break;
      }
      Sleep(5);
      HistorySelect(TimeCurrent() - 60, TimeCurrent());
   }

   long entry = -1;
   if(selected){
      entry = (long)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   }

   if(selected && entry == DEAL_ENTRY_IN){

      ulong  positionTicket      = trans.position;
      double openPrice           = 0.000000;
      double originalStopLevel   = 0.000000;
      double originalTargetLevel = 0.000000;
      ulong  magicNumber         = 0;

      if(PositionSelectByTicket(positionTicket)){
         openPrice           = PositionGetDouble(POSITION_PRICE_OPEN);
         originalStopLevel   = PositionGetDouble(POSITION_SL);
         originalTargetLevel = PositionGetDouble(POSITION_TP);
         magicNumber         = (ulong)PositionGetInteger(POSITION_MAGIC);
      }

      if(manageTradesByMagicNumber){
         if(magicNumber != tradeSelectionMagicNumber){
            return;
         }
      }

      if(trans.deal_type == DEAL_TYPE_BUY){
         Print("NEW LONG opened (confirmed via history): deal=", trans.deal, " pos=", trans.position);
         ArrayResize(tradeInfo, ArraySize(tradeInfo) + 1);
         tradeInfo[ArraySize(tradeInfo) - 1].ticket                  = positionTicket;
         tradeInfo[ArraySize(tradeInfo) - 1].positionType            = POSITION_TYPE_BUY;
         tradeInfo[ArraySize(tradeInfo) - 1].openPrice               = openPrice;
         tradeInfo[ArraySize(tradeInfo) - 1].originalStopLevel       = originalStopLevel;
         tradeInfo[ArraySize(tradeInfo) - 1].currentStopLevel        = originalStopLevel;
         tradeInfo[ArraySize(tradeInfo) - 1].nextStopLevel           = originalStopLevel + trailingStepPoints * pointValue;
         tradeInfo[ArraySize(tradeInfo) - 1].originalTargetLevel     = originalTargetLevel;
         tradeInfo[ArraySize(tradeInfo) - 1].nextTrailTriggerPrice   = openPrice + usefulTrailingStartPoints  * pointValue;
         tradeInfo[ArraySize(tradeInfo) - 1].isMovedToBreakEven      = false;
         tradeInfo[ArraySize(tradeInfo) - 1].isPartialProfitsSecured = false;
      }

      if(trans.deal_type == DEAL_TYPE_SELL){
         Print("NEW SHORT opened (confirmed via history): deal=", trans.deal, " pos=", trans.position);
         tradeInfo[ArraySize(tradeInfo) - 1].ticket                  = positionTicket;
         tradeInfo[ArraySize(tradeInfo) - 1].positionType            = POSITION_TYPE_SELL;
         tradeInfo[ArraySize(tradeInfo) - 1].openPrice               = openPrice;
         tradeInfo[ArraySize(tradeInfo) - 1].originalStopLevel       = originalStopLevel;
         tradeInfo[ArraySize(tradeInfo) - 1].currentStopLevel        = originalStopLevel;
         tradeInfo[ArraySize(tradeInfo) - 1].nextStopLevel           = originalStopLevel - trailingStepPoints * pointValue;
         tradeInfo[ArraySize(tradeInfo) - 1].originalTargetLevel     = originalTargetLevel;
         tradeInfo[ArraySize(tradeInfo) - 1].nextTrailTriggerPrice   = openPrice - usefulTrailingStartPoints  * pointValue;
         tradeInfo[ArraySize(tradeInfo) - 1].isMovedToBreakEven      = false;
         tradeInfo[ArraySize(tradeInfo) - 1].isPartialProfitsSecured = false;
      }
   }

   if(!selected){

      if(trans.deal_type == DEAL_TYPE_BUY  && trans.position != 0){
         Print("Probable NEW LONG (fallback): deal=", trans.deal, " pos=", trans.position);
      }

      if(trans.deal_type == DEAL_TYPE_SELL && trans.position != 0){
         Print("Probable NEW SHORT (fallback): deal=", trans.deal, " pos=", trans.position);
      }

   }

}
//+------------------------------------------------------------------+
```

This block of code ensures that AutoProtect automatically registers each new position that qualifies under the symbol and (if set) Magic Number filters. From here on, these trades are officially tracked and can be modified safely by the EA's trailing stop module.

If you try to compile the EA at this stage, you will notice a few compile-time errors. This happens because the program is referencing some global variables that have not been declared yet. To fix these errors, we need to define those variables before using them.

We now add a few more global variables that will store important trading data used across different parts of the Expert Advisor. These include freezeLevelPoints, askPrice, bidPrice, and pointValue.

```
...

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade Trade;
MqlTradeInfo tradeInfo[];
double closePriceMinutesData [];
int usefulTrailingStartPoints = trailingStartPoints;
long freezeLevelPoints;
double askPrice;
double bidPrice;
double pointValue;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){
   // ---
   return(INIT_SUCCEEDED);
}

...
```

The freezeLevelPoints variable holds the broker's freeze level in points. This value defines the minimum distance from the current market price where stop levels (stop loss and take profit) can be modified. Since this value does not change during program execution, we can safely initialize it inside the OnInit function.

```
...

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   //--- Initialize global variables
   freezeLevelPoints  = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);

   return(INIT_SUCCEEDED);
}

...
```

The askPrice and bidPrice variables represent the current market prices for buying and selling. Because these values change with every new tick, we should update them inside the OnTick function.

```
...

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   //--- Scope variables
   askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);

}

...
```

pointValue represents the smallest price movement for the current symbol. It remains constant, so we can initialize it together with freezeLevelPoints in the OnInit function.

```
...

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
int OnInit(){

   //--- Initialize global variables
   freezeLevelPoints  = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   pointValue         = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   return(INIT_SUCCEEDED);
}

...
```

Before we move on, we need to define two more global variables that will play important roles in the EA's operation. These are:

```
double closePriceMinutesData[];
int usefulTrailingStartPoints = trailingStartPoints;
```

The closePriceMinutesData array is used to store recent closing prices from the 1-minute (M1) timeframe. This data will later help the EA detect conditions such as price crossovers and crossunders, which are important for some trade management functions that will be discussed in later sections.

The usefulTrailingStartPoints variable helps coordinate the behavior between the break-even and trailing stop functions. Normally, it takes the value of trailingStartPoints, which defines how many points into profit the trailing should begin. However, when both the break-even and trailing stop features are enabled at the same time, it is logical that the trailing stop should only start after the break-even has been triggered. To handle this automatically, the following code is placed in the OnInit function.

```
...

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   ...

   //--- If both BreakEven and Trailing are enabled,trailing should only start after break-even has triggered.
   if(enableBreakEven && enableTrailingStop){
      usefulTrailingStartPoints = breakEvenTriggerPoints;
   }

   // Set arrays as series
   ArraySetAsSeries(closePriceMinutesData, true);

   return(INIT_SUCCEEDED);
}

...
```

The ArraySetAsSeries function ensures that the most recent data point of the array closePriceMinutesData is always stored at index 0 of the array, which makes accessing the latest prices straightforward.

Then, inside the OnTick function, we load the most recent seven closing prices from the M1 timeframe using the CopyClose function:

```
...

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   ...
   bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // Get some minutes data
   if(CopyClose(_Symbol, PERIOD_M1, 0, 7, closePriceMinutesData) == -1){
      Print("Error while copying minutes datas ", GetLastError());
      return;
   }

}

...
```

By updating this array every tick, the EA keeps a small rolling window of recent market data, which can be referenced whenever crossover-based logic is needed.

### Break-even Functionality

Before implementing the break-even logic, it is important to define a few helper functions that make our code cleaner and easier to maintain. These small utility functions perform specific tasks that the main logic relies on. We'll place them in a separate section dedicated to utility functions, right below our existing code structure.

```
...

//--- UTILITY FUNCTIONS
//+------------------------------------------------------------------+
//| Checks if position modification is allowed                       |
//+------------------------------------------------------------------+
bool IsTradeModificationAllowed(long freezeLevelPts, ENUM_ORDER_TYPE action, double askPr, double bidPr, double stopLossLvl, double takeProfitLvl){
   double freezeDistance = freezeLevelPts * pointValue;

   if(freezeLevelPts == 0){
      return true;
   }

   if(action == ORDER_TYPE_BUY) {
      double distanceFromSpotPriceToTP = takeProfitLvl - bidPr;
      double distanceFromSpotPriceToSL = bidPr - stopLossLvl;

      if(distanceFromSpotPriceToTP > freezeDistance && distanceFromSpotPriceToSL > freezeDistance){
         return true;
      }
   }

   if(action == ORDER_TYPE_SELL){
      double distanceFromSpotPriceToTP = askPr - takeProfitLvl;
      double distanceFromSpotPriceToSL = stopLossLvl - askPr;

      if(distanceFromSpotPriceToTP > freezeDistance && distanceFromSpotPriceToSL > freezeDistance){
         return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| To detect a crossover                                            |
//+------------------------------------------------------------------+
bool IsCrossOver(const double price, const double &closePriceMinsData[]){
   if(closePriceMinsData[1] <= price && closePriceMinsData[0] > price){
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| To detect a crossunder                                           |
//+------------------------------------------------------------------+
bool IsCrossUnder(const double price, const double &closePriceMinsData[]){
   if(closePriceMinsData[1] >= price && closePriceMinsData[0] < price){
      return true;
   }
   return false;
}
```

The first function, IsTradeModificationAllowed, checks whether a position can be modified based on the broker's freeze level. A freeze level defines how close an order or stop level can be to the current market price. If an order is too close to the price within the restricted zone, any modification is rejected by the broker. This function calculates the distance between the current price and both the stop-loss and take-profit levels. It then ensures these distances are greater than the freeze distance before allowing any update.

Next, we define two simple helper functions: IsCrossOver and IsCrossUnder. These detect when the most recent one-minute close price crosses over or under a given price level. They are used to confirm that price has moved past our break-even trigger level before modifying any stop loss. By keeping these functions separate, we make the program modular, readable, and easy to extend later. For example, we can reuse the same crossover logic for trailing stops or partial closes.

After defining the helper functions, we create the ManageBreakEven function. This function is responsible for automatically moving a trade's stop-loss to the break-even point once the price moves a certain distance in profit.

```
...

//--- UTILITY FUNCTIONS

...

//+------------------------------------------------------------------+
//| To mange the trade break even functionality                      |
//+------------------------------------------------------------------+
void ManageBreakEven   (){
   int totalPositions = PositionsTotal();
   for(int i = totalPositions - 1; i >= 0; i--){
      ulong ticket = PositionGetTicket(i);
      if(ticket != 0){
         // Get some useful position properties
         ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         double openPrice                = PositionGetDouble(POSITION_PRICE_OPEN);
         double currentPrice             = PositionGetDouble(POSITION_PRICE_CURRENT);
         string symbol                   = PositionGetString(POSITION_SYMBOL);
         double stopLevel                = PositionGetDouble(POSITION_SL);
         double takeProfitLevel          = PositionGetDouble(POSITION_TP);
         ulong magicNumber               = PositionGetInteger(POSITION_MAGIC);

         if(symbol == _Symbol){

            if(!manageTradesByMagicNumber){

               if(positionType == POSITION_TYPE_BUY ){
                  for(int j = ArraySize(tradeInfo) - 1; j >= 0; j--){
                     if(tradeInfo[j].ticket == ticket && tradeInfo[j].isMovedToBreakEven == false){
                        if(IsCrossOver(openPrice + breakEvenTriggerPoints * pointValue, closePriceMinutesData)){
                           // Move SL to breakeven + breakEvenLockPoints
                           double newStopLevel = openPrice + (breakEvenLockPoints * pointValue);
                           if(IsTradeModificationAllowed(freezeLevelPoints, ORDER_TYPE_BUY, askPrice, bidPrice, stopLevel, takeProfitLevel)){
                              if(!Trade.PositionModify(ticket, newStopLevel, takeProfitLevel)){
                                 Print("Error while moving Stop Loss to breakeven! ", GetLastError());
                                 Print(Trade.ResultComment());
                              }
                              tradeInfo[j].isMovedToBreakEven = true;
                           }
                        }
                     }
                  }
               }

               if(positionType == POSITION_TYPE_SELL){
                  for(int j = ArraySize(tradeInfo) - 1; i >= 0; i--){
                     if(tradeInfo[j].ticket == ticket && tradeInfo[j].isMovedToBreakEven == false){
                        if(IsCrossUnder(openPrice - breakEvenTriggerPoints * pointValue, closePriceMinutesData)){
                           // Move SL to breakeven + breakEvenLockPoints
                           double newStopLevel = openPrice - (breakEvenLockPoints * pointValue);
                           if(IsTradeModificationAllowed(freezeLevelPoints, ORDER_TYPE_BUY, askPrice, bidPrice, stopLevel, takeProfitLevel)){
                              if(!Trade.PositionModify(ticket, newStopLevel, takeProfitLevel)){
                                 Print("Error while moving Stop Loss to breakeven! ", GetLastError());
                                 Print(Trade.ResultComment());
                              }
                              tradeInfo[j].isMovedToBreakEven = true;
                           }
                        }
                     }
                  }
               }

            }

            if(manageTradesByMagicNumber){
               if(magicNumber == tradeSelectionMagicNumber){
                  if(positionType == POSITION_TYPE_BUY ){
                     for(int j = ArraySize(tradeInfo) - 1; j >= 0; j--){
                        if(tradeInfo[j].ticket == ticket && tradeInfo[j].isMovedToBreakEven == false){
                           if(IsCrossOver(openPrice + breakEvenTriggerPoints * pointValue, closePriceMinutesData)){
                              // Move SL to breakeven + breakEvenLockPoints
                              double newStopLevel = openPrice + (breakEvenLockPoints * pointValue);
                              if(IsTradeModificationAllowed(freezeLevelPoints, ORDER_TYPE_BUY, askPrice, bidPrice, stopLevel, takeProfitLevel)){
                                 if(!Trade.PositionModify(ticket, newStopLevel, takeProfitLevel)){
                                    Print("Error while moving Stop Loss to breakeven! ", GetLastError());
                                    Print(Trade.ResultComment());
                                 }
                                 tradeInfo[j].isMovedToBreakEven = true;
                              }
                           }
                        }
                     }
                  }

                  if(positionType == POSITION_TYPE_SELL){
                     for(int j = ArraySize(tradeInfo) - 1; i >= 0; i--){
                        if(tradeInfo[j].ticket == ticket && tradeInfo[j].isMovedToBreakEven == false){
                           if(IsCrossUnder(openPrice - breakEvenTriggerPoints * pointValue, closePriceMinutesData)){
                              // Move SL to breakeven + breakEvenLockPoints
                              double newStopLevel = openPrice - (breakEvenLockPoints * pointValue);
                              if(IsTradeModificationAllowed(freezeLevelPoints, ORDER_TYPE_BUY, askPrice, bidPrice, stopLevel, takeProfitLevel)){
                                 if(!Trade.PositionModify(ticket, newStopLevel, takeProfitLevel)){
                                    Print("Error while moving Stop Loss to breakeven! ", GetLastError());
                                    Print(Trade.ResultComment());
                                 }
                                 tradeInfo[j].isMovedToBreakEven = true;
                              }
                           }
                        }
                     }
                  }
               }
            }

         }else{
            continue;
         }
      }else{
         Print("Error while getting a position ticket!", GetLastError());
         continue;
      }
   }
}
```

Inside the function, the EA loops through all open positions on the account. For each trade, it first checks that the symbol matches the chart symbol where the EA is running. If the user enabled trade filtering by magic number, it further ensures that only trades matching the specified magic number are managed.

For a buy position, the EA waits until the market price crosses above the breakeven trigger level. When this happens, the function calculates a new stop-loss level at the open price plus the user-defined lock points, which represent a small buffer beyond breakeven. It then calls the IsTradeModificationAllowed function to make sure the broker permits modification before updating the stop loss.

The same logic applies for a sell position, but in the opposite direction. Once the price crosses below the breakeven trigger level, the EA moves the stop loss to the open price minus the buffer points.

This design ensures that the breakeven function reacts only to active trades opened after the EA has been launched. It avoids touching any historical or unrelated trades. If modification fails, the EA prints clear error messages to help with debugging.

In short, this function's goal is to automatically protect the trader's capital by ensuring that once a trade moves favorably by the defined distance in points, it can no longer turn into a loss. This small but crucial feature is one of the cornerstones of the AutoProtect Smart Trade Manager EA.

### Trailing Stop functionality

The trailing stop feature allows the Expert Advisor to automatically move the stop-loss level, progressively securing gains as price moves further into profit. This helps to lock in profits without the trader having to monitor positions constantly. The trailing process is gradual, ensuring that profitable trades are not closed prematurely while still protecting gains if the market reverses. We are going to define the function ManageTrailingStop just below the ManageBreakEvenFunction.

```
...

//+------------------------------------------------------------------+
//| To mange the trade break even functionality                      |
//+------------------------------------------------------------------+
void ManageBreakEven   (){
   ...
}

//+------------------------------------------------------------------+
//| To manage the trade trailing stop functionality                  |
//+------------------------------------------------------------------+
void ManageTrailingStop(){
   int totalPositions = PositionsTotal();
   // Loop through all open positions
   for(int i = totalPositions - 1; i >= 0; i--){
      ulong ticket = PositionGetTicket(i);
      if(ticket != 0){
         // Get some useful position properties
         ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         double openPrice                = PositionGetDouble (POSITION_PRICE_OPEN);
         double currentPrice             = PositionGetDouble (POSITION_PRICE_CURRENT);
         string symbol                   = PositionGetString (POSITION_SYMBOL);
         double stopLevel                = PositionGetDouble (POSITION_SL);
         double takeProfitLevel          = PositionGetDouble (POSITION_TP);
         ulong magicNumber               = PositionGetInteger(POSITION_MAGIC);
         // Skip positions not matching this financial security
         if(symbol == _Symbol){
            if(!manageTradesByMagicNumber){

               if(positionType == POSITION_TYPE_BUY ){

                  // 07-10-2025
                  for(int i = ArraySize(tradeInfo) - 1; i >= 0; i--){
                     if(tradeInfo[i].ticket == ticket){
                        if(IsCrossOver(tradeInfo[i].nextTrailTriggerPrice, closePriceMinutesData)){
                           // Physically trail SL only when next SL level is above current SL Level
                           if(tradeInfo[i].nextStopLevel > tradeInfo[i].currentStopLevel){
                              if(IsTradeModificationAllowed(freezeLevelPoints, ORDER_TYPE_BUY, askPrice, bidPrice, stopLevel, takeProfitLevel)){
                                 if(!Trade.PositionModify(ticket, tradeInfo[i].nextStopLevel, takeProfitLevel)){
                                    Print("Error while trailing Stop Loss! ", GetLastError());
                                    Print(Trade.ResultComment());
                                    Print(Trade.ResultRetcode());
                                 }
                              }
                           }
                           tradeInfo[i].currentStopLevel = tradeInfo[i].currentStopLevel + trailingStepPoints * pointValue;
                           tradeInfo[i].nextStopLevel    = tradeInfo[i].nextStopLevel    + trailingStepPoints * pointValue;
                        }
                     }
                  }
               }

               if(positionType == POSITION_TYPE_SELL){
                  // 07-10-2025
                  for(int i = ArraySize(tradeInfo) - 1; i >= 0; i--){
                     if(tradeInfo[i].ticket == ticket){
                        if(IsCrossUnder(tradeInfo[i].nextTrailTriggerPrice, closePriceMinutesData)){
                           // Physically trail SL only when next SL level is above current SL Level
                           if(tradeInfo[i].nextStopLevel < tradeInfo[i].currentStopLevel){
                              if(IsTradeModificationAllowed(freezeLevelPoints, ORDER_TYPE_SELL, askPrice, bidPrice, stopLevel, takeProfitLevel)){
                                 if(!Trade.PositionModify(ticket, tradeInfo[i].nextStopLevel, takeProfitLevel)){
                                    Print("Error while trailing Stop Loss! ", GetLastError());
                                    Print(Trade.ResultComment());
                                    Print(Trade.ResultRetcode());
                                 }
                              }
                           }
                           tradeInfo[i].currentStopLevel = tradeInfo[i].currentStopLevel - trailingStepPoints * pointValue;
                           tradeInfo[i].nextStopLevel    = tradeInfo[i].nextStopLevel    - trailingStepPoints * pointValue;
                        }
                     }
                  }
               }
            }

            if( manageTradesByMagicNumber){
               if(magicNumber == tradeSelectionMagicNumber){

                  if(positionType == POSITION_TYPE_BUY ){
                     // 07-10-2025
                     for(int i = ArraySize(tradeInfo) - 1; i >= 0; i--){
                        if(tradeInfo[i].ticket == ticket){
                           if(IsCrossOver(tradeInfo[i].nextTrailTriggerPrice, closePriceMinutesData)){
                              // Physically trail SL only when next SL level is above current SL Level
                              if(tradeInfo[i].nextStopLevel > tradeInfo[i].currentStopLevel){
                                 if(IsTradeModificationAllowed(freezeLevelPoints, ORDER_TYPE_BUY, askPrice, bidPrice, stopLevel, takeProfitLevel)){
                                    if(!Trade.PositionModify(ticket, tradeInfo[i].nextStopLevel, takeProfitLevel)){
                                       Print("Error while trailing Stop Loss! ", GetLastError());
                                       Print(Trade.ResultComment());
                                       Print(Trade.ResultRetcode());
                                    }
                                 }
                              }
                              tradeInfo[i].currentStopLevel = tradeInfo[i].currentStopLevel + trailingStepPoints * pointValue;
                              tradeInfo[i].nextStopLevel    = tradeInfo[i].nextStopLevel    + trailingStepPoints * pointValue;
                           }
                        }
                     }
                  }
                  if(positionType == POSITION_TYPE_SELL){
                     // 07-10-2025
                     for(int i = ArraySize(tradeInfo) - 1; i >= 0; i--){
                        if(tradeInfo[i].ticket == ticket){
                           if(IsCrossUnder(tradeInfo[i].nextTrailTriggerPrice, closePriceMinutesData)){
                              // Physically trail SL only when next SL level is above current SL Level
                              if(tradeInfo[i].nextStopLevel < tradeInfo[i].currentStopLevel){
                                 if(IsTradeModificationAllowed(freezeLevelPoints, ORDER_TYPE_SELL, askPrice, bidPrice, stopLevel, takeProfitLevel)){
                                    if(!Trade.PositionModify(ticket, tradeInfo[i].nextStopLevel, takeProfitLevel)){
                                       Print("Error while trailing Stop Loss! ", GetLastError());
                                       Print(Trade.ResultComment());
                                       Print(Trade.ResultRetcode());
                                    }
                                 }
                              }
                              tradeInfo[i].currentStopLevel = tradeInfo[i].currentStopLevel - trailingStepPoints * pointValue;
                              tradeInfo[i].nextStopLevel    = tradeInfo[i].nextStopLevel    - trailingStepPoints * pointValue;
                           }
                        }
                     }
                  }
               }else{
                  continue;
               }
            }

         }else{
            continue;
         }
      }else{
         Print("Error while getting a position ticket! ", GetLastError());
         continue;
      }
   }
}
```

The ManageTrailingStop function goes through all open positions and checks if any of them meet the conditions for trailing. For each position, it first identifies whether it belongs to the same chart symbol and whether it should be managed by a specific magic number, depending on user settings. This ensures that only relevant trades are modified.

Inside the function, the program compares recent market prices using the IsCrossOver and IsCrossUnder helper functions. These checks determine when the price has crossed specific levels that trigger a stop-loss adjustment. When a crossover or crossunder happens, the EA calculates a new stop-loss level and moves it by a predefined step size (trailingStepPoints) in the position's direction.

Before modifying the trade, the function also checks if it is safe to do so using the IsTradeModificationAllowed function. This prevents changes when the price is too close to the freeze level, which could cause broker errors. Once verified, the EA modifies the stop-loss using Trade.PositionModify method and logs any errors that may occur during the process.

Each time the trailing condition is met, the EA updates the trade's stored data inside the tradeInfo structure. It increases both the currentStopLevel and the nextStopLevel so that future trailing continues smoothly. The function works for both buy and sell positions, applying the same logic but in opposite directions.

With this functionality, traders no longer need to manually trail stop losses. The EA keeps on adjusting them automatically, letting profitable trades run while minimizing risk exposure.

### Partial profits close functionality

Immediately below the ManageTrailingStop function, add the ManagePartialClose function.

```
...

//+------------------------------------------------------------------+
//| To manage the trade trailing stop functionality                  |
//+------------------------------------------------------------------+
void ManageTrailingStop(){
   ...
}

//+------------------------------------------------------------------+
//| To manage the trade partial close functionality                  |
//+------------------------------------------------------------------+
void ManagePartialClose(){
   int totalPositions = PositionsTotal();

   for(int i = totalPositions - 1; i >= 0; i--){
      ulong ticket = PositionGetTicket(i);
      if(ticket != 0){
         // Get some useful position properties
         ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         double openPrice                = PositionGetDouble (POSITION_PRICE_OPEN);
         double currentPrice             = PositionGetDouble (POSITION_PRICE_CURRENT);
         string symbol                   = PositionGetString (POSITION_SYMBOL);
         ulong magicNumber               = PositionGetInteger(POSITION_MAGIC);
         double positionVolume           = PositionGetDouble (POSITION_VOLUME);
         if(symbol == _Symbol){

            double volumeToDecrease = NormalizeDouble((partialClosePercent / 100.0) * positionVolume, 2);

            if(!manageTradesByMagicNumber){
               if(positionType == POSITION_TYPE_BUY ){
                  for(int j = ArraySize(tradeInfo) - 1; j >= 0; j--){
                     if(tradeInfo[j].ticket == ticket && tradeInfo[j].isPartialProfitsSecured == false){
                        if(IsCrossOver(openPrice + partialCloseTriggerPoints * pointValue, closePriceMinutesData)){
                           if(!Trade.PositionClosePartial(ticket, volumeToDecrease)){
                              Print("Error closing partial volume! ", GetLastError());
                              Print(Trade.ResultComment());
                           }
                           tradeInfo[j].isPartialProfitsSecured = true;
                        }
                     }
                  }
               }

               if(positionType == POSITION_TYPE_SELL){
                  for(int j = ArraySize(tradeInfo) - 1; j >= 0; j--){
                     if(tradeInfo[j].ticket == ticket && tradeInfo[j].isPartialProfitsSecured == false){
                        if(IsCrossOver(openPrice - partialCloseTriggerPoints * pointValue, closePriceMinutesData)){
                           if(!Trade.PositionClosePartial(ticket, volumeToDecrease)){
                              Print("Error closing partial volume! ", GetLastError());
                              Print(Trade.ResultComment());
                           }
                           tradeInfo[j].isPartialProfitsSecured = true;
                        }
                     }
                  }
               }

            }

            if( manageTradesByMagicNumber){
               if(magicNumber == tradeSelectionMagicNumber){

                  if(positionType == POSITION_TYPE_BUY ){
                     for(int j = ArraySize(tradeInfo) - 1; j >= 0; j--){
                        if(tradeInfo[j].ticket == ticket && tradeInfo[j].isPartialProfitsSecured == false){
                           if(IsCrossOver(openPrice + partialCloseTriggerPoints * pointValue, closePriceMinutesData)){
                              if(!Trade.PositionClosePartial(ticket, volumeToDecrease)){
                                 Print("Error closing partial volume! ", GetLastError());
                                 Print(Trade.ResultComment());
                              }
                              tradeInfo[j].isPartialProfitsSecured = true;
                           }
                        }
                     }
                  }

                  if(positionType == POSITION_TYPE_SELL){
                     for(int j = ArraySize(tradeInfo) - 1; j >= 0; j--){
                        if(tradeInfo[j].ticket == ticket && tradeInfo[j].isPartialProfitsSecured == false){
                           if(IsCrossOver(openPrice - partialCloseTriggerPoints * pointValue, closePriceMinutesData)){
                              if(!Trade.PositionClosePartial(ticket, volumeToDecrease)){
                                 Print("Error closing partial volume! ", GetLastError());
                                 Print(Trade.ResultComment());
                              }
                              tradeInfo[j].isPartialProfitsSecured = true;
                           }
                        }
                     }
                  }

               }
            }

         }

      }else{
         Print("Error while getting a position ticket!", GetLastError());
         continue;
      }
   }
}
```

This function handles the process of partially closing open trades once they reach a defined profit level. It allows the Expert Advisor to secure part of the profit while keeping the remaining position open to capture further gains.

The function begins by looping through all open positions. For each position, it retrieves useful information such as the position type (buy or sell), open price, current price, symbol, magic number, and trade volume.

It then checks if the position belongs to the current chart symbol. If trade management is filtered by magic number, it also verifies that the position matches the specified magic number before proceeding.

The function calculates the volume to close by applying the partial close percentage to the current trade volume. Once the price moves in profit by the specified trigger distance (in points), the EA closes the calculated portion of the trade using the CTrade.PositionClosePartial method.

For buy positions, the partial close is triggered when the current price rises above the open price by the defined threshold. For sell positions, it is triggered when the current price falls below the open price by the same distance.

If an error occurs during the partial close process, an error message and the reason for failure are printed in the journal for debugging.

### Testing and Validation

After completing the development of our Trade Manager Expert Advisor, the next step is to test its features. This process helps to confirm that every function behaves as expected under real trading conditions. Each feature, such as break-even, trailing stop, and partial close, should be tested separately to ensure proper operation.

To achieve this, we will use a simple testing EA that automatically opens trades when launched. We will then enable one feature at a time and observe its effect on the trade, recording screenshots and notes for reference. This approach provides a clear and reliable way to verify that all functionalities in our EA are working correctly.

To test our protection functions effectively, we will create a new Expert Advisor file named AutoProtect.mq5. This file will open a trade automatically when launched and allow us to observe how each feature behaves in a controlled environment. Below is an explanation of the additional functions and  code used in this testing EA.

1\. Checking for Active Positions

We will start by defining the following two functions just below our other existing utility functions: IsThereAnActiveBuyPosition and IsThereAnActiveSellPosition.

```
...

//+------------------------------------------------------------------+
//| To manage the trade partial close functionality                  |
//+------------------------------------------------------------------+
void ManagePartialClose(){
   ...
}

//+------------------------------------------------------------------+
//| To check if there is an active buy position opened by this EA    |
//+------------------------------------------------------------------+
bool IsThereAnActiveBuyPosition(ulong magicNm){
   for(int i = PositionsTotal() - 1; i >= 0; i--){
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0){
         Print("Error while fetching position ticket ", _LastError);
         continue;
      }else{
         if(PositionGetInteger(POSITION_MAGIC) == magicNm && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY){
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| To check if there is an active sell position opened by this EA   |
//+------------------------------------------------------------------+
bool IsThereAnActiveSellPosition(ulong mgcNumber){
   for(int i = PositionsTotal() - 1; i >= 0; i--){
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0){
         Print("Error while fetching position ticket ", _LastError);
         continue;
      }else{
         if(PositionGetInteger(POSITION_MAGIC) == mgcNumber && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL){
            return true;
         }
      }
   }
   return false;
}

...
```

These functions are used to check whether there is an active buy or sell position that was opened by our EA. Each function loops through all open positions on the account using PositionsTotal and PositionGetTicket. It then compares the magic number and position type to determine if the position belongs to our EA. If a matching position is found, the function returns true; otherwise, it returns false. This ensures that the EA does not open multiple trades of the same type unnecessarily.

2\. Allowing Trade Execution Once

Next, we introduce a new global variable.

```
bool isNewTradeAllowed;
```

This variable is used as a flag to control whether a new trade can be opened. Inside the OnInit function, we initialize it as:

```
isNewTradeAllowed  = true;
```

This means that when the EA starts, it is allowed to open one trade. After the first trade is opened, the flag will be set to false, preventing additional trades from being opened automatically.

3\. Automatically Opening a Test Trade

In the OnTick function, we add logic to open a buy position immediately after launch.

```
...

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   ...

   //--- Get some minutes data
   if(CopyClose(_Symbol, PERIOD_M1, 0, 7, closePriceMinutesData) == -1){
      Print("Error while copying minutes datas ", GetLastError());
      return;
   }

   //--- Open a long position immediately after launch
   if(isNewTradeAllowed){
      if(!IsThereAnActiveBuyPosition(tradeSelectionMagicNumber) && !IsThereAnActiveSellPosition(tradeSelectionMagicNumber)){
         Trade.Buy(1.0, _Symbol, askPrice, askPrice - 400 * pointValue, askPrice + 800 * pointValue);
         isNewTradeAllowed = false;
      }
   }
}

...
```

Here, the EA checks if there are no existing trades for the given magic number. If no open positions are found, it opens a buy trade using predefined stop-loss and take-profit levels. Once the trade is opened, isNewTradeAllowed becomes false to stop further trades.

4\. Calling the Protection Functions

Both AutoProtect.mq5 and AutoProtectTest.mq5 call the main protection functions as shown below:

```
...

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   ...

   //--- Open a long position immediately after launch
   if(isNewTradeAllowed){
      ...
   }

   if(enableBreakEven){
      ManageBreakEven();
   }

   if(enableTrailingStop){
      ManageTrailingStop();
   }

   if(enablePartialClose){
      ManagePartialClose();
   }

}

...
```

Each condition checks whether a specific feature is enabled. If it is, the corresponding function is executed. This approach allows us to test one feature at a time by simply toggling its input parameter in the EA settings.

5\. Configuring the Chart Appearance

To make our test environment clearer, we define a helper function named ConfigureChartAppearance.

```
...

//+------------------------------------------------------------------+
//| To check if there is an active sell position opened by this EA   |
//+------------------------------------------------------------------+
bool IsThereAnActiveSellPosition(ulong mgcNumber){
   ...
}

//+------------------------------------------------------------------+
//| This function configures the chart's appearance                  |
//+------------------------------------------------------------------+
bool ConfigureChartAppearance(){
   if(!ChartSetInteger(0, CHART_COLOR_BACKGROUND, clrWhiteSmoke)){
      Print("Error while setting chart background, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_SHOW_GRID, false)){
      Print("Error while setting chart grid, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_MODE, CHART_CANDLES)){
      Print("Error while setting chart mode, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_COLOR_CANDLE_BULL, clrGreen)){
      Print("Error while setting bullish candles color, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_COLOR_CANDLE_BEAR, clrDarkRed)){
      Print("Error while setting bearish candles color, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_COLOR_CHART_UP, clrGreen)){
      Print("Error while setting bearish candles color, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_COLOR_CHART_DOWN, clrDarkRed)){
      Print("Error while setting bearish candles color, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_COLOR_BID, clrDarkRed)){
      Print("Error while setting chart bid line, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_COLOR_ASK, clrGreen)){
      Print("Error while setting chart ask line, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_SHOW_ONE_CLICK, true)){
      Print("Error while setting one click buttons, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_COLOR_STOP_LEVEL, clrDarkBlue)){
      Print("Error while setting stop levels, ", GetLastError());
      return false;
   }
   if(!ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrBlack)){
      Print("Error while setting chart foreground, ", GetLastError());
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
```

This function customizes the look of the chart for better visibility during testing. It adjusts elements such as the chart background, candle colors, grid lines and stop-level lines.

For example:

- The background color is set to clrWhiteSmoke.
- Bullish candles are colored green, while bearish candles are dark red.
- The bid and ask lines are also colored for easy identification

Each setting is applied using the ChartSetInteger MQL5 function. If an error occurs, the function prints an error message and returns false. We then call this function inside OnInit.

```
...

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){

   if(!ConfigureChartAppearance()){
      Print("Error while configuring the chart's appearance, ", GetLastError());
      return(INIT_FAILED);
   }

   ...

}

...
```

This ensures that the chart is correctly configured before the EA starts running. If configuration fails, the EA stops initialization to prevent visual confusion.

This setup provides a clean and automated environment for testing each feature. It ensures that our tests are repeatable, controlled, and easy to observe, especially when verifying the behavior of the BreakEven, Trailing Stop, and Partial Close functionalities.

We will begin this process by testing the breakeven functionality to confirm that it adjusts the stop-loss to a risk-free level once the trade moves in profit.

We will disable both the trailing stop and partial close functionalities, leaving only the break-even feature enabled. Next, we will set the input variable breakEvenTriggerPoints to 200 and launch the EA on the EURUSD chart to observe its behavior.

![input parameters for breakeven](https://c.mql5.com/2/176/input_parameters_for_breakeven.png)

Immediately after launching the EA, you will notice that it opens a new long position with a stop loss set 400 points below and a take profit set 800 points above the entry price.

![Position Opening ](https://c.mql5.com/2/176/position_opening.png)

Once the position moves 200 points into profit, you will notice that the stop loss has been automatically adjusted to the breakeven level.

![sl moved to breakeven](https://c.mql5.com/2/176/sl_moved_to_breakeven.png)

Next, we will test the trailing Stop functionality. We will disable all other features and enable only the trailing stop in the input parameters. After that, we will launch the EA on the chart and observe how the stop loss is adjusted as the price continues to move further into profit.

![testing trailing stop](https://c.mql5.com/2/176/testing_the_trailing_stop_functionality.png)

We set the trailingStartPoints to 50 and the trailingStepPoints to 20. Next, we will launch the EA on the EURUSD chart and observe its behavior. Immediately after launch, you will notice that the EA opens a long position with a stop-loss 400 points away below the entry price and a take-profit set 800 points above it.

![trailing stop functionality first time launch](https://c.mql5.com/2/176/trailing_stop_functionality_first_time_launch.png)

In this case, the trailing stop was moved from 1.05945 to 1.05985.

![testing the trailing stop function step 1](https://c.mql5.com/2/176/testing_the_trailing_stop_functionality_step_1.png)

Next, the stop level was adjusted from 1.05985 to 1.06025.

![testing the trailing stop functionality step 2](https://c.mql5.com/2/176/testing_the_trailing_stop_functionality_step_2.png)

Again, the trailing stop was adjusted from 1.0625 to 1.06165 here.

![testing the trailing stop functionality step 3](https://c.mql5.com/2/176/testing_the_trailing_stop_functionality_step_3.png)

This clearly shows that the trailing stop functionality is working as expected. Finally, let us proceed to test the partial close functionality. We will disable all other functions and enable only the Partial Close functionality. Set PartialTriggerPoints to 200 and PartialClosePercent to 50.0. This means that 50% of the position volume will be closed once the price moves 200 points into profit.

![Partial Close Settings](https://c.mql5.com/2/176/Partial_Close_Settings.png)

Let us now launch the EA again on the EURUSD chart and observe its behavior. Once again, you will notice that the EA opens a long position immediately after launch. This position has a stop loss set 400 points below the entry price and a take profit set 800 points above the entry price. Also, notice that the position is opened with a volume of 1 standard lot.

![Partial Close In Action](https://c.mql5.com/2/176/Partial_Close_In_Action__1.png)

After the position moves 200 points into profit, you will observe that 50% of the initial trade volume has been closed. This confirms that our partial close functionality works as expected, marking successful completion of our testing and validation phase.

### Conclusion

Most traders begin by managing trades manually — watching charts, adjusting stop-losses, and taking profits by intuition. This often leads to emotional errors and missed opportunities.

In this article, we built a Smart Trade Manager EA that automates these tasks: moving stop-losses to breakeven, trailing profits, and closing partial positions. Each function was tested and proved reliable in real-time conditions.

Automation brings not just convenience but also discipline and consistency. With AutoProtect EA, traders can protect capital, secure profits, and manage positions efficiently — turning trading from emotional reaction into a structured, rule-based process.


**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19911.zip "Download all attachments in the single ZIP archive")

[AutoProtect.mq5](https://www.mql5.com/en/articles/download/19911/AutoProtect.mq5 "Download AutoProtect.mq5")(30.1 KB)

[AutoProtectTest.mq5](https://www.mql5.com/en/articles/download/19911/AutoProtectTest.mq5 "Download AutoProtectTest.mq5")(34.25 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Larry Williams Market Secrets (Part 6): Measuring Volatility Breakouts Using Market Swings](https://www.mql5.com/en/articles/20862)
- [Larry Williams Market Secrets (Part 5): Automating the Volatility Breakout Strategy in MQL5](https://www.mql5.com/en/articles/20745)
- [Larry Williams Market Secrets (Part 4): Automating Short-Term Swing Highs and Lows in MQL5](https://www.mql5.com/en/articles/20716)
- [Larry Williams Market Secrets (Part 3): Proving Non-Random Market Behavior with MQL5](https://www.mql5.com/en/articles/20510)
- [Larry Williams Market Secrets (Part 2): Automating a Market Structure Trading System](https://www.mql5.com/en/articles/20512)
- [Larry Williams Market Secrets (Part 1): Building a Swing Structure Indicator in MQL5](https://www.mql5.com/en/articles/20511)
- [Mastering Kagi Charts in MQL5 (Part 2): Implementing Automated Kagi-Based Trading](https://www.mql5.com/en/articles/20378)

**[Go to discussion](https://www.mql5.com/en/forum/498958)**

![Automating Trading Strategies in MQL5 (Part 37): Regular RSI Divergence Convergence with Visual Indicators](https://c.mql5.com/2/176/20031-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 37): Regular RSI Divergence Convergence with Visual Indicators](https://www.mql5.com/en/articles/20031)

In this article, we build an MQL5 EA that detects regular RSI divergences using swing points with strength, bar limits, and tolerance checks. It executes trades on bullish or bearish signals with fixed lots, SL/TP in pips, and optional trailing stops. Visuals include colored lines on charts and labeled swings for better strategy insights.

![Price Action Analysis Toolkit Development (Part 47): Tracking Forex Sessions and Breakouts in MetaTrader 5](https://c.mql5.com/2/177/19944-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 47): Tracking Forex Sessions and Breakouts in MetaTrader 5](https://www.mql5.com/en/articles/19944)

Global market sessions shape the rhythm of the trading day, and understanding their overlap is vital to timing entries and exits. In this article, we’ll build an interactive trading sessions  EA that brings those global hours to life directly on your chart. The EA automatically plots color‑coded rectangles for the Asia, Tokyo, London, and New York sessions, updating in real time as each market opens or closes. It features on‑chart toggle buttons, a dynamic information panel, and a scrolling ticker headline that streams live status and breakout messages. Tested on different brokers, this EA combines precision with style—helping traders see volatility transitions, identify cross‑session breakouts, and stay visually connected to the global market’s pulse.

![Machine Learning Blueprint (Part 4): The Hidden Flaw in Your Financial ML Pipeline — Label Concurrency](https://c.mql5.com/2/175/19850-machine-learning-blueprint-logo.png)[Machine Learning Blueprint (Part 4): The Hidden Flaw in Your Financial ML Pipeline — Label Concurrency](https://www.mql5.com/en/articles/19850)

Discover how to fix a critical flaw in financial machine learning that causes overfit models and poor live performance—label concurrency. When using the triple-barrier method, your training labels overlap in time, violating the core IID assumption of most ML algorithms. This article provides a hands-on solution through sample weighting. You will learn how to quantify temporal overlap between trading signals, calculate sample weights that reflect each observation's unique information, and implement these weights in scikit-learn to build more robust classifiers. Learning these essential techniques will make your trading models more robust, reliable and profitable.

![Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://c.mql5.com/2/177/20026-statistical-arbitrage-through-logo__1.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 6): Scoring System](https://www.mql5.com/en/articles/20026)

In this article, we propose a scoring system for mean-reversion strategies based on statistical arbitrage of cointegrated stocks. The article suggests criteria that go from liquidity and transaction costs to the number of cointegration ranks and time to mean-reversion, while taking into account the strategic criteria of data frequency (timeframe) and the lookback period for cointegration tests, which are evaluated before the score ranking properly. The files required for the reproduction of the backtest are provided, and their results are commented on as well.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/19911&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068267860117092205)

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