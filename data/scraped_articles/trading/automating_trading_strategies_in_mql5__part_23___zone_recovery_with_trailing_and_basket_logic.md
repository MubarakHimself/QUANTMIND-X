---
title: Automating Trading Strategies in MQL5 (Part 23): Zone Recovery with Trailing and Basket Logic
url: https://www.mql5.com/en/articles/18778
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:54:04.856222
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/18778&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068776697777552778)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 22)](https://www.mql5.com/en/articles/18720), we developed a [Zone Recovery System](https://www.mql5.com/go?link=https://tradingliteracy.com/zone-recovery-in-trading/ "https://tradingliteracy.com/zone-recovery-in-trading/") for Envelopes Trend Trading in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5), using Relative Strength Index (RSI) and Envelopes indicators to automate trades and manage losses through structured recovery zones. In Part 23, we refine this strategy by incorporating trailing stops to dynamically secure profits and a multi-basket system to efficiently handle multiple trade signals, thereby enhancing adaptability in volatile markets. We will cover the following topics:

1. [Understanding the Enhanced Trailing Stop and Multi-Basket Architecture](https://www.mql5.com/en/articles/18778#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/18778#para2)
3. [Backtesting](https://www.mql5.com/en/articles/18778#para3)
4. [Conclusion](https://www.mql5.com/en/articles/18778#para4)

By the end, you’ll have a refined MQL5 trading system with advanced features, ready for testing and further customization—let’s dive in!

### Understanding the Enhanced Trailing Stop and Multi-Basket Architecture

The [zone recovery strategy](https://www.mql5.com/go?link=https://tradingliteracy.com/zone-recovery-in-trading/ "https://tradingliteracy.com/zone-recovery-in-trading/") we’re enhancing is designed to turn potential losses into wins by placing counter-trades within a defined price range when the market moves against us. We’re now strengthening it with two key improvements: trailing stops and multi-basket trading. Trailing stops are necessary because they allow us to lock in profits as the market moves in our favor, protecting gains without closing trades too early, which is critical in trending markets where prices can run significantly. Multi-basket trading is equally important, as it lets us manage multiple independent trade signals simultaneously, increasing our ability to capture more opportunities while keeping risk organized across separate trade groups. See below.

![TRAILING STOP ARCHITECTURE](https://c.mql5.com/2/155/Screenshot_2025-07-07_220509.png)

We will achieve these enhancements by integrating a trailing stop mechanism that adjusts the stop-loss level dynamically based on market movement, ensuring we secure profits while giving trades room to grow. For multi-basket trading, we will introduce a system to handle multiple trade instances, each with its unique identifier, allowing us to track and manage several zone recovery cycles at once without overlap. We plan to combine these features with the existing [Relative Strength Indicator](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI) and [Envelopes](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes") indicators to maintain precise trade entries, while the trailing stops and basket system work together to optimize profit protection and trade capacity, making the strategy more robust and adaptable to various market conditions. Stay with us as we bring these improvements to life!

### Implementation in MQL5

To implement the enhancements in MQL5, we will add some extra user inputs for the trailing stop feature, and rename the maximum cap order limit since we are now dealing with multiple recovery instances.

```
input group "======= EA GENERAL SETTINGS ======="
input TradingLotSizeOptions lotOption = UNFIXED_LOTSIZE;               // Lot Size Option
input double initialLotSize = 0.01;                                    // Initial Lot Size
input double riskPercentage = 1.0;                                     // Risk Percentage (%)
input int    riskPoints = 300;                                         // Risk Points
input int    baseMagicNumber = 123456789;                              // Base Magic Number
input int    maxInitialPositions = 1;                                  // Maximum Initial Positions (Baskets/Signals)
input double zoneTargetPoints = 600;                                   // Zone Target Points
input double zoneSizePoints = 300;                                     // Zone Size Points
input bool   enableInitialTrailing = true;                             // Enable Trailing Stop for Initial Positions
input int    trailingStopPoints = 50;                                  // Trailing Stop Points
input int    minProfitPoints = 50;                                     // Minimum Profit Points to Start Trailing
```

We start the enhancement of our Zone Recovery System for Envelopes Trend Trading in MQL5 by updating the [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) parameters under the "EA GENERAL SETTINGS" group to support trailing stops and multi-basket trading. We make four key changes to the inputs. First, we rename "magicNumber" to "baseMagicNumber", set to 123456789, to serve as a starting point for generating unique magic numbers for multiple trade baskets, ensuring each basket is tracked separately for our multi-basket system. Second, we replace "maxOrders" with "maxInitialPositions", set to 1, to limit the number of initial trade baskets, allowing us to manage multiple trade signals efficiently.

Third, we add "enableInitialTrailing", a boolean set to true, to let us enable or disable trailing stops for initial positions, providing control over our new profit-locking feature. Fourth, we introduce "trailingStopPoints" set to 50 and "minProfitPoints" set to 50, defining the trailing stop distance and the minimum profit needed to activate it, respectively, to implement dynamic profit protection. These changes will enable our system to handle multiple trade baskets and protect profits effectively, setting the stage for further enhancements. We will be highlighting changes to enable easier tracking of the changes and avoid confusion. Upon compilation, we have the following input set.

![NEW INPUTS SET](https://c.mql5.com/2/155/Screenshot_2025-07-07_224808.png)

After adding the inputs, we can now forward declare the "MarketZoneTrader" class so it can be accessed by the base class, since we now want to handle multiple trade instances.

```
//--- Forward Declaration of MarketZoneTrader
class MarketZoneTrader;
```

Here, we introduce a forward declaration of the "MarketZoneTrader" [class](https://www.mql5.com/en/docs/basis/types/classes). We add it before the "BasketManager" class definition, which we will define just after this class, to allow it to reference "MarketZoneTrader" without requiring its full definition yet. This change is necessary because our new multi-basket system, managed by "BasketManager", will need to create and handle multiple instances of "MarketZoneTrader" for different trade baskets. By declaring "MarketZoneTrader" first, we ensure the compiler recognizes it when used in the new class, enabling our system to support multiple simultaneous trade cycles efficiently. We can then define the manager class.

```
//--- Basket Manager Class to Handle Multiple Traders
class BasketManager {
private:
   MarketZoneTrader* m_traders[];                                        //--- Array of trader instances
   int               m_handleRsi;                                        //--- RSI indicator handle
   int               m_handleEnvUpper;                                   //--- Upper Envelopes handle
   int               m_handleEnvLower;                                   //--- Lower Envelopes handle
   double            m_rsiBuffer[];                                     //--- RSI data buffer
   double            m_envUpperBandBuffer[];                            //--- Upper Envelopes buffer
   double            m_envLowerBandBuffer[];                            //--- Lower Envelopes buffer
   string            m_symbol;                                          //--- Trading symbol
   int               m_baseMagicNumber;                                 //--- Base magic number
   int               m_maxInitialPositions;                             //--- Maximum baskets (signals)

   //--- Initialize Indicators
   bool initializeIndicators() {
      m_handleRsi = iRSI(m_symbol, PERIOD_CURRENT, 8, PRICE_CLOSE);
      if (m_handleRsi == INVALID_HANDLE) {
         Print("Failed to initialize RSI indicator");
         return false;
      }
      m_handleEnvUpper = iEnvelopes(m_symbol, PERIOD_CURRENT, 150, 0, MODE_SMA, PRICE_CLOSE, 0.1);
      if (m_handleEnvUpper == INVALID_HANDLE) {
         Print("Failed to initialize upper Envelopes indicator");
         return false;
      }
      m_handleEnvLower = iEnvelopes(m_symbol, PERIOD_CURRENT, 95, 0, MODE_SMA, PRICE_CLOSE, 1.4);
      if (m_handleEnvLower == INVALID_HANDLE) {
         Print("Failed to initialize lower Envelopes indicator");
         return false;
      }
      ArraySetAsSeries(m_rsiBuffer, true);
      ArraySetAsSeries(m_envUpperBandBuffer, true);
      ArraySetAsSeries(m_envLowerBandBuffer, true);
      return true;
   }

}
```

To help manage basket trades, we define the "BasketManager" [class](https://www.mql5.com/en/docs/basis/types/classes) with private members to manage multiple instances of the "MarketZoneTrader" class and indicator data. We create "m\_traders", an array of "MarketZoneTrader" [pointers](https://www.mql5.com/en/docs/basis/types/object_pointers), to store individual trade baskets, each representing a separate zone recovery cycle. This change is critical as it allows us to manage multiple trade signals simultaneously, unlike the single-instance approach in the prior version. We also declare "m\_handleRsi", "m\_handleEnvUpper", and "m\_handleEnvLower" to hold indicator handles, and "m\_rsiBuffer", "m\_envUpperBandBuffer", and "m\_envLowerBandBuffer" arrays to store RSI and Envelopes data, moving indicator management from "MarketZoneTrader" to "BasketManager" for centralized control across baskets.

Additionally, we add "m\_symbol" to store the trading symbol, "m\_baseMagicNumber" for generating unique magic numbers per basket, and "m\_maxInitialPositions" to limit the number of active baskets, aligning with the new "maxInitialPositions" input. In the "initializeIndicators" function, we set up the RSI indicator with [iRSI](https://www.mql5.com/en/docs/indicators/irsi) using an 8-period setting and [Envelopes](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes") indicators with iEnvelopes (150-period with 0.1 deviation and 95-period with 1.4 deviation), checking for "INVALID\_HANDLE" and logging failures with [Print](https://www.mql5.com/en/docs/common/print). We configure "m\_rsiBuffer", "m\_envUpperBandBuffer", and "m\_envLowerBandBuffer" as time-series arrays using [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries). This new class structure will enable us to coordinate multiple trade baskets efficiently, centralizing indicator data for consistent signal generation across all baskets. We then need to have a logic to count all the individual basket positions for easier tracking, and clean the baskets.

```
//--- Count Active Baskets
int countActiveBaskets() {
   int count = 0;
   for (int i = 0; i < ArraySize(m_traders); i++) {
      if (m_traders[i] != NULL && m_traders[i].getCurrentState() != MarketZoneTrader::INACTIVE) {
         count++;
      }
   }
   return count;
}

//--- Cleanup Terminated Baskets
void cleanupTerminatedBaskets() {
   int newSize = 0;
   for (int i = 0; i < ArraySize(m_traders); i++) {
      if (m_traders[i] != NULL && m_traders[i].getCurrentState() == MarketZoneTrader::INACTIVE) {
         delete m_traders[i];
         m_traders[i] = NULL;
      }
      if (m_traders[i] != NULL) newSize++;
   }
   MarketZoneTrader* temp[];
   ArrayResize(temp, newSize);
   int index = 0;
   for (int i = 0; i < ArraySize(m_traders); i++) {
      if (m_traders[i] != NULL) {
         temp[index] = m_traders[i];
         index++;
      }
   }
   ArrayFree(m_traders);
   ArrayResize(m_traders, newSize);
   for (int i = 0; i < newSize; i++) {
      m_traders[i] = temp[i];
   }
   ArrayFree(temp);
}
```

Here, we add two new functions to the "BasketManager" class, "countActiveBaskets" and "cleanupTerminatedBaskets". We start with the "countActiveBaskets" function to track the number of active trade baskets. We initialize a "count" variable to 0 and loop through the "m\_traders" array using the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function. For each non-null "m\_traders" entry, we check if its state, obtained via "getCurrentState", is not "MarketZoneTrader::INACTIVE". If active, we increment "count". We return "count" to monitor how many baskets are currently running, which is crucial for ensuring we stay within the "m\_maxInitialPositions" limit when opening new baskets.

Next, we create the "cleanupTerminatedBaskets" function to remove inactive baskets and optimize memory. We first count non-null entries in "m\_traders" by looping through the array. If a trader is not null and its "getCurrentState" returns "MarketZoneTrader::INACTIVE", we use "delete" to free its memory and set the entry to "NULL". We track the number of remaining non-null traders in "newSize". Then, we create a temporary "temp" array, resize it to "newSize" with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), and copy non-null traders from "m\_traders" to "temp" using an "index" counter. We clear "m\_traders" with "ArrayFree", resize it to "newSize", and transfer the traders back from "temp". Finally, we free "temp" with [ArrayFree](https://www.mql5.com/en/docs/array/arrayfree). This cleanup ensures we remove terminated baskets, keeping our system efficient and ready for new trades. We then move to the public access modifier, where we will change how we handle the [constructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor) and [destructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor) in initializing and destroying the class members and elements.

```
public:
   BasketManager(string symbol, int baseMagic, int maxInitPos) {
      m_symbol = symbol;
      m_baseMagicNumber = baseMagic;
      m_maxInitialPositions = maxInitPos;
      ArrayResize(m_traders, 0);
      m_handleRsi = INVALID_HANDLE;
      m_handleEnvUpper = INVALID_HANDLE;
      m_handleEnvLower = INVALID_HANDLE;
   }

   ~BasketManager() {
      for (int i = 0; i < ArraySize(m_traders); i++) {
         if (m_traders[i] != NULL) delete m_traders[i];
      }
      ArrayFree(m_traders);
      cleanupIndicators();
   }
```

We start with the "BasketManager" [constructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor), which takes "symbol", "baseMagic", and "maxInitPos" as parameters. We assign these to "m\_symbol", "m\_baseMagicNumber", and "m\_maxInitialPositions", respectively, to set the trading symbol, base magic number for unique basket identification, and the maximum number of active baskets. We initialize the "m\_traders" array to zero size using the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function and set indicator handles—"m\_handleRsi", "m\_handleEnvUpper", and "m\_handleEnvLower"—to "INVALID\_HANDLE" to prepare for indicator setup later. This constructor is crucial for configuring the multi-basket system.

Next, we create the "~BasketManager" [destructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor) to clean up resources. Typically, destructors have the tilde sign as the prefix, just as a reminder. We loop through the "m\_traders" array using [ArraySize](https://www.mql5.com/en/docs/array/arraysize) and delete any non-null "MarketZoneTrader" instances with [delete](https://www.mql5.com/en/docs/basis/operators/deleteoperator) to free their memory. We then clear the "m\_traders" array with [ArrayFree](https://www.mql5.com/en/docs/array/arrayfree) and call "cleanupIndicators" to release indicator handles and buffers. This ensures our system shuts down cleanly, preventing memory leaks when the EA stops. In the prior version, we had to add the deletion logic in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler directly after realizing that there was a memory leak, but here, we can add it early since we already know we need to take care of memory leaks. We then need to alter the initialization logic so that it can load existing positions into respective baskets. Here is the logic we implement to achieve that.

```
bool initialize() {
   if (!initializeIndicators()) return false;
   //--- Load existing positions into baskets
   int totalPositions = PositionsTotal();
   for (int i = 0; i < totalPositions; i++) {
      ulong ticket = PositionGetTicket(i);
      if (PositionSelectByTicket(ticket)) {
         if (PositionGetString(POSITION_SYMBOL) == m_symbol) {
            long magic = PositionGetInteger(POSITION_MAGIC);
            if (magic >= m_baseMagicNumber && magic < m_baseMagicNumber + m_maxInitialPositions) {
               //--- Check if basket already exists for this magic
               bool exists = false;
               for (int j = 0; j < ArraySize(m_traders); j++) {
                  if (m_traders[j] != NULL && m_traders[j].getMagicNumber() == magic) {
                     exists = true;
                     break;
                  }
               }
               if (!exists && countActiveBaskets() < m_maxInitialPositions) {
                  createNewBasket(magic, ticket);
               }
            }
         }
      }
   }
   Print("BasketManager initialized with ", ArraySize(m_traders), " existing baskets");
   return true;
}

/*
//--- PREVIOUS INITIALIZATION
int initialize() {
   //--- Initialization Start
   m_tradeExecutor.SetExpertMagicNumber(m_tradeConfig.tradeIdentifier); //--- Set magic number
   int totalPositions = PositionsTotal();                           //--- Get total positions

   for (int i = 0; i < totalPositions; i++) {                       //--- Iterate positions
      ulong ticket = PositionGetTicket(i);                          //--- Get ticket
      if (PositionSelectByTicket(ticket)) {                         //--- Select position
         if (PositionGetString(POSITION_SYMBOL) == m_tradeConfig.marketSymbol && PositionGetInteger(POSITION_MAGIC) == m_tradeConfig.tradeIdentifier) { //--- Check symbol and magic
            if (activateTrade(ticket)) {                            //--- Activate position
               Print("Existing position activated: Ticket=", ticket); //--- Log activation
            } else {
               Print("Failed to activate existing position: Ticket=", ticket); //--- Log failure
            }
         }
      }
   }

   m_handleRsi = iRSI(m_tradeConfig.marketSymbol, PERIOD_CURRENT, 8, PRICE_CLOSE); //--- Initialize RSI
   if (m_handleRsi == INVALID_HANDLE) {                             //--- Check RSI
      Print("Failed to initialize RSI indicator");                  //--- Log failure
      return INIT_FAILED;                                           //--- Return failure
   }

   m_handleEnvUpper = iEnvelopes(m_tradeConfig.marketSymbol, PERIOD_CURRENT, 150, 0, MODE_SMA, PRICE_CLOSE, 0.1); //--- Initialize upper Envelopes
   if (m_handleEnvUpper == INVALID_HANDLE) {                        //--- Check upper Envelopes
      Print("Failed to initialize upper Envelopes indicator");      //--- Log failure
      return INIT_FAILED;                                           //--- Return failure
   }

   m_handleEnvLower = iEnvelopes(m_tradeConfig.marketSymbol, PERIOD_CURRENT, 95, 0, MODE_SMA, PRICE_CLOSE, 1.4); //--- Initialize lower Envelopes
   if (m_handleEnvLower == INVALID_HANDLE) {                        //--- Check lower Envelopes
      Print("Failed to initialize lower Envelopes indicator");      //--- Log failure
      return INIT_FAILED;                                           //--- Return failure
   }

   ArraySetAsSeries(m_rsiBuffer, true);                             //--- Set RSI buffer
   ArraySetAsSeries(m_envUpperBandBuffer, true);                    //--- Set upper Envelopes buffer
   ArraySetAsSeries(m_envLowerBandBuffer, true);                    //--- Set lower Envelopes buffer

   Print("EA initialized successfully");                            //--- Log success
   return INIT_SUCCEEDED;                                           //--- Return success
   //--- Initialization End
}
*/
```

Here, we implement the updated "initialize" function in the "BasketManager" class to support our multi-basket trading improvement by initializing indicators and loading existing positions into separate baskets. We start by calling "initializeIndicators" to set up RSI and Envelopes indicators, returning false if it fails, ensuring our system has the necessary market data. Unlike the previous version, where we handled indicator setup directly in "MarketZoneTrader"’s "initialize" function, we now centralize this in "BasketManager" to share indicator data across multiple baskets. Next, we check for existing positions using the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function and loop through each position, grabbing its "ticket" with the [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) function.

If [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) succeeds and the position’s symbol matches "m\_symbol" (via [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring)), we verify its magic number, obtained with "PositionGetInteger", falls within the range of "m\_baseMagicNumber" to "m\_baseMagicNumber + m\_maxInitialPositions". We then check if a basket already exists for this magic number by looping through "m\_traders" and calling "getMagicNumber" on non-null entries. If no basket exists and "countActiveBaskets" is below "m\_maxInitialPositions", we call "createNewBasket" with the magic number and "ticket" to load the position into a new basket. Finally, we log the number of initialized baskets with "Print" using [ArraySize](https://www.mql5.com/en/docs/array/arraysize) of "m\_traders" and return true. When we run the program, we get the following result.

![BASKETS INITIALIZATION](https://c.mql5.com/2/155/Screenshot_2025-07-07_235830.png)

We can now move on to processing ticks, where we need to process the existing baskets on every tick and create new baskets when new signals are confirmed in the "processTick" function, unlike in the previous version, where we only needed to initiate trades based on confirmed signals.

```
void processTick() {
   //--- Process existing baskets
   for (int i = 0; i < ArraySize(m_traders); i++) {
      if (m_traders[i] != NULL) {
         m_traders[i].processTick(m_rsiBuffer, m_envUpperBandBuffer, m_envLowerBandBuffer);
      }
   }
   cleanupTerminatedBaskets();

   //--- Check for new signals on new bar
   if (!isNewBar()) return;

   if (!CopyBuffer(m_handleRsi, 0, 0, 3, m_rsiBuffer)) {
      Print("Error loading RSI data. Reverting.");
      return;
   }
   if (!CopyBuffer(m_handleEnvUpper, 0, 0, 3, m_envUpperBandBuffer)) {
      Print("Error loading upper envelopes data. Reverting.");
      return;
   }
   if (!CopyBuffer(m_handleEnvLower, 1, 0, 3, m_envLowerBandBuffer)) {
      Print("Error loading lower envelopes data. Reverting.");
      return;
   }

   const int rsiOverbought = 70;
   const int rsiOversold = 30;
   int ticket = -1;
   ENUM_ORDER_TYPE signalType = (ENUM_ORDER_TYPE)-1;

   double askPrice = NormalizeDouble(SymbolInfoDouble(m_symbol, SYMBOL_ASK), Digits());
   double bidPrice = NormalizeDouble(SymbolInfoDouble(m_symbol, SYMBOL_BID), Digits());

   if (m_rsiBuffer[1] < rsiOversold && m_rsiBuffer[2] > rsiOversold && m_rsiBuffer[0] < rsiOversold) {
      if (askPrice > m_envUpperBandBuffer[0]) {
         if (countActiveBaskets() < m_maxInitialPositions) {
            signalType = ORDER_TYPE_BUY;
         }
      }
   } else if (m_rsiBuffer[1] > rsiOverbought && m_rsiBuffer[2] < rsiOverbought && m_rsiBuffer[0] > rsiOverbought) {
      if (bidPrice < m_envLowerBandBuffer[0]) {
         if (countActiveBaskets() < m_maxInitialPositions) {
            signalType = ORDER_TYPE_SELL;
         }
      }
   }

   if (signalType != (ENUM_ORDER_TYPE)-1) {
      //--- Create new basket with unique magic number
      int newMagic = m_baseMagicNumber + ArraySize(m_traders);
      if (newMagic < m_baseMagicNumber + m_maxInitialPositions) {
         MarketZoneTrader* newTrader = new MarketZoneTrader(lotOption, initialLotSize, riskPercentage, riskPoints, zoneTargetPoints, zoneSizePoints, newMagic);
         ticket = newTrader.openInitialOrder(signalType); //--- Open INITIAL position
         if (ticket > 0 && newTrader.activateTrade(ticket)) {
            int size = ArraySize(m_traders);
            ArrayResize(m_traders, size + 1);
            m_traders[size] = newTrader;
            Print("New basket created: Magic=", newMagic, ", Ticket=", ticket, ", Type=", EnumToString(signalType));
         } else {
            delete newTrader;
            Print("Failed to create new basket: Ticket=", ticket);
         }
      } else {
         Print("Maximum initial positions (baskets) reached: ", m_maxInitialPositions);
      }
   }
}
```

In the function, we start by looping through the "m\_traders" array using the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function and, for each non-null "MarketZoneTrader" instance, we call its "processTick" function, passing "m\_rsiBuffer", "m\_envUpperBandBuffer", and "m\_envLowerBandBuffer" to handle individual basket logic. This differs from the previous version, where "processTick" directly managed a single trade cycle. We then call "cleanupTerminatedBaskets" to remove inactive baskets, ensuring efficient resource use. Next, we check for new trade signals only on a new bar using "isNewBar", exiting if false to save resources.

We load indicator data with [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) for "m\_handleRsi", "m\_handleEnvUpper", and "m\_handleEnvLower" into their respective buffers, logging errors with "Print" and exiting if any fail, unlike the previous version, where this was done in "MarketZoneTrader". We set "rsiOverbought" to 70 and "rsiOversold" to 30, and initialize "ticket" and "signalType". We fetch "askPrice" and "bidPrice" using [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) with "SYMBOL\_ASK" and [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double), normalized with the [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) function.

For a buy signal, if "m\_rsiBuffer" indicates oversold conditions and "askPrice" exceeds "m\_envUpperBandBuffer", we set "signalType" to [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) if "countActiveBaskets" is below "m\_maxInitialPositions". For a sell signal, if "m\_rsiBuffer" shows overbought conditions and "bidPrice" is below "m\_envLowerBandBuffer", we set "signalType" to [ORDER\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type). If a valid "signalType" exists, we create a unique magic number with "m\_baseMagicNumber" plus "ArraySize(m\_traders)", and if within "m\_maxInitialPositions", we instantiate a new "MarketZoneTrader" with input parameters and the new magic number.

We call "openInitialOrder" with "signalType", and if the returned "ticket" is valid and "activateTrade" succeeds, we add the new trader to "m\_traders" using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) and log success with "Print" and the [EnumToString](https://www.mql5.com/en/docs/convert/enumtostring) function. Otherwise, we delete the trader and log the failure, or note if the basket limit is reached. Once the new trades are opened, we will need to create new baskets for them. Here is the logic we use to achieve that.

```
private:
   void createNewBasket(long magic, ulong ticket) {
      MarketZoneTrader* newTrader = new MarketZoneTrader(lotOption, initialLotSize, riskPercentage, riskPoints, zoneTargetPoints, zoneSizePoints, magic);
      if (newTrader.activateTrade(ticket)) {
         int size = ArraySize(m_traders);
         ArrayResize(m_traders, size + 1);
         m_traders[size] = newTrader;
         Print("Existing position loaded into basket: Magic=", magic, ", Ticket=", ticket);
      } else {
         delete newTrader;
         Print("Failed to load existing position into basket: Ticket=", ticket);
      }
   }
```

We implement the "createNewBasket" function in the private section of the "BasketManager" class, a new addition to support our multi-basket trading improvement by creating and managing new trade baskets for existing positions. We start by creating a new "MarketZoneTrader" instance, named "newTrader", using the input parameters "lotOption", "initialLotSize", "riskPercentage", "riskPoints", "zoneTargetPoints", "zoneSizePoints", and the provided "magic" number to configure a unique trade basket. Recall that we had this user input in the initialization stage in the prior version because we just needed one instance of the zone, so it did apply to all new positions, but in this case, we organize it in new class instances. Here is the code for that for quicker comparison.

```
//--- PREVIOUS VERSION OF NEW CLASS INSTANCE
//--- Global Instance
MarketZoneTrader *trader = NULL;                                        //--- Declare trader instance

int OnInit() {
   //--- EA Initialization Start
   trader = new MarketZoneTrader(lotOption, initialLotSize, riskPercentage, riskPoints, maxOrders, restrictMaxOrders, zoneTargetPoints, zoneSizePoints); //--- Create trader instance
   return trader.initialize();                                           //--- Initialize EA
   //--- EA Initialization End
}
```

We then call "activateTrade" on "newTrader" with the given "ticket" to load the existing position into the basket. If successful, we get the current size of the "m\_traders" array using [ArraySize](https://www.mql5.com/en/docs/array/arraysize), expand it by one with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), and add "newTrader" to the new slot. We log the success with "Print", including the "magic" and "ticket" values. If "activateTrade" fails, we delete "newTrader" to free memory and log the failure with "Print". The function will now enable us to organize existing positions into separate baskets, a key feature of our multi-basket system, unlike the single-instance approach in the previous version. That class will now enable us to manage the trade baskets effectively. Let us then graduate to modifying the base class so that it can contain the new multiple baskets and trailing stop features. Let us start with its members.

```
//--- Modified MarketZoneTrader Class
class MarketZoneTrader {
private:
   enum TradeState { INACTIVE, RUNNING, TERMINATING };

   struct TradeMetrics {
      bool   operationSuccess;
      double totalVolume;
      double netProfitLoss;
   };

   struct ZoneBoundaries {
      double zoneHigh;
      double zoneLow;
      double zoneTargetHigh;
      double zoneTargetLow;
   };

   struct TradeConfig {
      string         marketSymbol;
      double         openPrice;
      double         initialVolume;
      long           tradeIdentifier;
      string         initialTradeLabel;  //--- Label for initial positions
      string         recoveryTradeLabel; //--- Label for recovery positions
      ulong          activeTickets[];
      ENUM_ORDER_TYPE direction;
      double         zoneProfitSpan;
      double         zoneRecoverySpan;
      double         accumulatedBuyVolume;
      double         accumulatedSellVolume;
      TradeState     currentState;
      bool           hasRecoveryTrades;  //--- Flag to track recovery trades
      double         trailingStopLevel;  //--- Virtual trailing stop level
   };

   struct LossTracker {
      double tradeLossTracker;
   };

   TradeConfig           m_tradeConfig;
   ZoneBoundaries        m_zoneBounds;
   LossTracker           m_lossTracker;
   string                m_lastError;
   int                   m_errorStatus;
   CTrade                m_tradeExecutor;
   TradingLotSizeOptions m_lotOption;
   double                m_initialLotSize;
   double                m_riskPercentage;
   int                   m_riskPoints;
   double                m_zoneTargetPoints;
   double                m_zoneSizePoints;
}
```

Here, we enhance our program by modifying the "MarketZoneTrader" class, specifically its private section, to include new features supporting trailing stops and improved trade labeling. We retain the core structure but introduce key changes to the "TradeConfig" [structure](https://www.mql5.com/en/docs/basis/types/classes) to align with our enhanced strategy. We keep the "TradeState" [enumeration](https://www.mql5.com/en/docs/basis/types/integer/enumeration) with "INACTIVE", "RUNNING", and "TERMINATING" states, and the "TradeMetrics", "ZoneBoundaries", and "LossTracker" structures unchanged from the previous version, as they continue to manage trade states, performance metrics, zone boundaries, and loss tracking.

In the "TradeConfig" structure, we add two new string variables: "initialTradeLabel" and "recoveryTradeLabel". These labels allow us to tag initial and recovery trades separately, improving trade identification and tracking within each basket, especially useful for managing multiple baskets in our new system. We also introduce "hasRecoveryTrades", a [boolean](https://www.mql5.com/en/docs/basis/operations/bool) to track whether a basket includes recovery trades, which is critical for enabling or disabling trailing stops appropriately. Additionally, we add "trailingStopLevel", a double to store the virtual trailing stop level for each basket, enabling dynamic profit protection for initial trades.

Among the member variables, we retain "m\_tradeConfig", "m\_zoneBounds", "m\_lossTracker", "m\_lastError", "m\_errorStatus", "m\_tradeExecutor", "m\_lotOption", "m\_initialLotSize", "m\_riskPercentage", "m\_riskPoints", "m\_zoneTargetPoints", and "m\_zoneSizePoints" as they were, but their roles now support the new trailing stop and multi-basket functionality within each "MarketZoneTrader" instance. Notably, we remove the indicator-related variables like "m\_handleRsi" and "m\_rsiBuffer" from the class, as these are now managed centrally by the "BasketManager" class, streamlining each trader’s focus on individual basket operations. In the constructor and destructor, we will need to slightly change some variables so that they handle the new features.

```
public:
   MarketZoneTrader(TradingLotSizeOptions lotOpt, double initLot, double riskPct, int riskPts, double targetPts, double sizePts, long magic) {
      m_tradeConfig.currentState = INACTIVE;
      ArrayResize(m_tradeConfig.activeTickets, 0);
      m_tradeConfig.zoneProfitSpan = targetPts * _Point;
      m_tradeConfig.zoneRecoverySpan = sizePts * _Point;
      m_lossTracker.tradeLossTracker = 0.0;
      m_lotOption = lotOpt;
      m_initialLotSize = initLot;
      m_riskPercentage = riskPct;
      m_riskPoints = riskPts;
      m_zoneTargetPoints = targetPts;
      m_zoneSizePoints = sizePts;
      m_tradeConfig.marketSymbol = _Symbol;
      m_tradeConfig.tradeIdentifier = magic;
      m_tradeConfig.initialTradeLabel = "EA_INITIAL_" + IntegerToString(magic); //--- Label for initial positions
      m_tradeConfig.recoveryTradeLabel = "EA_RECOVERY_" + IntegerToString(magic); //--- Label for recovery positions
      m_tradeConfig.hasRecoveryTrades = false; //--- Initialize recovery flag
      m_tradeConfig.trailingStopLevel = 0.0; //--- Initialize trailing stop
      m_tradeExecutor.SetExpertMagicNumber(magic);
   }

   ~MarketZoneTrader() {
      ArrayFree(m_tradeConfig.activeTickets);
   }
```

We start with the "MarketZoneTrader" [constructor](https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor), now accepting an additional "magic" parameter to assign a unique magic number for each trade basket, unlike the previous version that used a fixed magic number. To support improved trade labeling, we add "m\_tradeConfig.initialTradeLabel" as "EA\_INITIAL" plus "magic" (via [IntegerToString](https://www.mql5.com/en/docs/convert/IntegerToString)) and "m\_tradeConfig.recoveryTradeLabel" as "EA\_RECOVERY" plus "magic", enabling distinct identification of initial and recovery trades within a basket. We initialize "m\_tradeConfig.hasRecoveryTrades" to false to track recovery trade status and set "m\_tradeConfig.trailingStopLevel" to 0.0 for the virtual trailing stop, both new features. Finally, we configure "m\_tradeExecutor" with "SetExpertMagicNumber" using "magic". We have highlighted the major changes for quick identification.

Next, we simplify the "~MarketZoneTrader" destructor compared to the previous version, which was called "cleanup". We now only clear "m\_tradeConfig.activeTickets" with [ArrayFree](https://www.mql5.com/en/docs/array/arrayfree), as indicator cleanup is handled by "BasketManager", reducing the destructor’s scope to focus on basket-specific resources. We can then update the function responsible for activating trades so that it can initialize the trailing stop level and recovery state for initial trades.

```
bool activateTrade(ulong ticket) {

   m_tradeConfig.hasRecoveryTrades = false;
   m_tradeConfig.trailingStopLevel = 0.0;

   //--- THE REST OF THE LOGIC REMAINS

   return true;
}
```

Here, we just add the logic to initialize the first trade's trailing stop level to 0 and recovery state to false to indicate it is the first position in the basket. Finally, we can add a function to open the initial position.

```
int openInitialOrder(ENUM_ORDER_TYPE orderType) {
   //--- Open INITIAL position based on signal
   int ticket;
   double openPrice;
   if (orderType == ORDER_TYPE_BUY) {
      openPrice = NormalizeDouble(getMarketAsk(), Digits());
   } else if (orderType == ORDER_TYPE_SELL) {
      openPrice = NormalizeDouble(getMarketBid(), Digits());
   } else {
      Print("Invalid order type [Magic=", m_tradeConfig.tradeIdentifier, "]");
      return -1;
   }
   double lotSize = 0;
   if (m_lotOption == FIXED_LOTSIZE) {
      lotSize = m_initialLotSize;
   } else if (m_lotOption == UNFIXED_LOTSIZE) {
      lotSize = calculateLotSize(m_riskPercentage, m_riskPoints);
   }
   if (lotSize <= 0) {
      Print("Invalid lot size [Magic=", m_tradeConfig.tradeIdentifier, "]: ", lotSize);
      return -1;
   }
   if (m_tradeExecutor.PositionOpen(m_tradeConfig.marketSymbol, orderType, lotSize, openPrice, 0, 0, m_tradeConfig.initialTradeLabel)) {
      ticket = (int)m_tradeExecutor.ResultOrder();
      Print("INITIAL trade opened [Magic=", m_tradeConfig.tradeIdentifier, "]: Ticket=", ticket, ", Type=", EnumToString(orderType), ", Volume=", lotSize);
   } else {
      ticket = -1;
      Print("Failed to open INITIAL order [Magic=", m_tradeConfig.tradeIdentifier, "]: Type=", EnumToString(orderType), ", Volume=", lotSize);
   }
   return ticket;
}
```

We implement a new "openInitialOrder" function in the public section of the "MarketZoneTrader" class to support our multi-basket and improved trade labeling enhancements by opening initial positions for a specific trade basket with distinct identification. We start by initializing "ticket" and "openPrice". For "orderType" set to [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type), we set "openPrice" using "getMarketAsk" and normalize it with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) and "Digits". For "ORDER\_TYPE\_SELL", we use "getMarketBid". If "orderType" is invalid, we log an error with "Print", including "m\_tradeConfig.tradeIdentifier", and return -1.

We determine "lotSize" based on "m\_lotOption": for "FIXED\_LOTSIZE", we use "m\_initialLotSize"; for "UNFIXED\_LOTSIZE", we call "calculateLotSize" with "m\_riskPercentage" and "m\_riskPoints". If "lotSize" is invalid, we log the error with "Print" and return -1. We then open the position using "m\_tradeExecutor.PositionOpen" with "m\_tradeConfig.marketSymbol", "orderType", "lotSize", "openPrice", and "m\_tradeConfig.initialTradeLabel" for clear labeling of initial trades. On success, we set "ticket" with "ResultOrder" and log the trade with "Print", including "m\_tradeConfig.tradeIdentifier" and the [EnumToString](https://www.mql5.com/en/docs/convert/enumtostring) function. On failure, we set "ticket" to -1 and log the error. Finally, we return the "ticket". Unlike the previous version’s "openOrder" function, this function uses the new "initialTradeLabel" and focuses solely on initial positions, aligning with our multi-basket system. Upon compilation, we get the following outcome.

![INITIAL BASKET](https://c.mql5.com/2/155/Screenshot_2025-07-08_003516.png)

From the image, we can see that we can open the initial trade and create a new basket instance for it. We now need to have trailing logic so that we can manage the trailing stop feature for the positions.

```
void evaluateMarketTick() {
   if (m_tradeConfig.currentState == INACTIVE) return;
   if (m_tradeConfig.currentState == TERMINATING) {
      finalizePosition();
      return;
   }
   double currentPrice;
   double profitPoints = 0.0;

   //--- Handle BUY initial position
   if (m_tradeConfig.direction == ORDER_TYPE_BUY) {
      currentPrice = getMarketBid();
      profitPoints = (currentPrice - m_tradeConfig.openPrice) / _Point;

      //--- Trailing Stop Logic for Initial Position
      if (enableInitialTrailing && !m_tradeConfig.hasRecoveryTrades && profitPoints >= minProfitPoints) {
         //--- Calculate desired trailing stop level
         double newTrailingStop = currentPrice - trailingStopPoints * _Point;
         //--- Start or update trailing stop if profit exceeds minProfitPoints + trailingStopPoints
         if (profitPoints >= minProfitPoints + trailingStopPoints) {
            if (m_tradeConfig.trailingStopLevel == 0.0 || newTrailingStop > m_tradeConfig.trailingStopLevel) {
               m_tradeConfig.trailingStopLevel = newTrailingStop;
               Print("Trailing stop updated [Magic=", m_tradeConfig.tradeIdentifier, "]: Level=", m_tradeConfig.trailingStopLevel, ", Profit=", profitPoints, " points");
            }
         }
         //--- Check if price has hit trailing stop
         if (m_tradeConfig.trailingStopLevel > 0.0 && currentPrice <= m_tradeConfig.trailingStopLevel) {
            Print("Trailing stop triggered [Magic=", m_tradeConfig.tradeIdentifier, "]: Bid=", currentPrice, " <= TrailingStop=", m_tradeConfig.trailingStopLevel);
            finalizePosition();
            return;
         }
      }

      //--- Zone Recovery Logic
      if (currentPrice > m_zoneBounds.zoneTargetHigh) {
         Print("Closing position [Magic=", m_tradeConfig.tradeIdentifier, "]: Bid=", currentPrice, " > TargetHigh=", m_zoneBounds.zoneTargetHigh);
         finalizePosition();
         return;
      } else if (currentPrice < m_zoneBounds.zoneLow) {
         Print("Triggering RECOVERY trade [Magic=", m_tradeConfig.tradeIdentifier, "]: Bid=", currentPrice, " < ZoneLow=", m_zoneBounds.zoneLow);
         triggerRecoveryTrade(ORDER_TYPE_SELL, currentPrice);
      }
   }
   //--- Handle SELL initial position
   else if (m_tradeConfig.direction == ORDER_TYPE_SELL) {
      currentPrice = getMarketAsk();
      profitPoints = (m_tradeConfig.openPrice - currentPrice) / _Point;

      //--- Trailing Stop Logic for Initial Position
      if (enableInitialTrailing && !m_tradeConfig.hasRecoveryTrades && profitPoints >= minProfitPoints) {
         //--- Calculate desired trailing stop level
         double newTrailingStop = currentPrice + trailingStopPoints * _Point;
         //--- Start or update trailing stop if profit exceeds minProfitPoints + trailingStopPoints
         if (profitPoints >= minProfitPoints + trailingStopPoints) {
            if (m_tradeConfig.trailingStopLevel == 0.0 || newTrailingStop < m_tradeConfig.trailingStopLevel) {
               m_tradeConfig.trailingStopLevel = newTrailingStop;
               Print("Trailing stop updated [Magic=", m_tradeConfig.tradeIdentifier, "]: Level=", m_tradeConfig.trailingStopLevel, ", Profit=", profitPoints, " points");
            }
         }
         //--- Check if price has hit trailing stop
         if (m_tradeConfig.trailingStopLevel > 0.0 && currentPrice >= m_tradeConfig.trailingStopLevel) {
            Print("Trailing stop triggered [Magic=", m_tradeConfig.tradeIdentifier, "]: Ask=", currentPrice, " >= TrailingStop=", m_tradeConfig.trailingStopLevel);
            finalizePosition();
            return;
         }
      }

      //--- Zone Recovery Logic
      if (currentPrice < m_zoneBounds.zoneTargetLow) {
         Print("Closing position [Magic=", m_tradeConfig.tradeIdentifier, "]: Ask=", currentPrice, " < TargetLow=", m_zoneBounds.zoneTargetLow);
         finalizePosition();
         return;
      } else if (currentPrice > m_zoneBounds.zoneHigh) {
         Print("Triggering RECOVERY trade [Magic=", m_tradeConfig.tradeIdentifier, "]: Ask=", currentPrice, " > ZoneHigh=", m_zoneBounds.zoneHigh);
         triggerRecoveryTrade(ORDER_TYPE_BUY, currentPrice);
      }
   }
}
```

Here, we enhance the program by updating the "evaluateMarketTick" function to incorporate trailing stop logic while maintaining the existing zone recovery logic. We start by checking if "m\_tradeConfig.currentState" is "INACTIVE" or "TERMINATING", exiting or calling "finalizePosition" as before. For a buy position ("m\_tradeConfig.direction" as [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type)), we get "currentPrice" with "getMarketBid" and calculate "profitPoints" as the difference between "currentPrice" and "m\_tradeConfig.openPrice" divided by "\_Point". The new trailing stop logic checks if "enableInitialTrailing" is true, "m\_tradeConfig.hasRecoveryTrades" is false, and "profitPoints" meets or exceeds "minProfitPoints". If so, we calculate "newTrailingStop" by subtracting "trailingStopPoints" times "\_Point" from "currentPrice". If "profitPoints" also exceeds "minProfitPoints" plus "trailingStopPoints" and "m\_tradeConfig.trailingStopLevel" is either 0.0 or less than "newTrailingStop", we update "m\_tradeConfig.trailingStopLevel" and log it with "Print".

If "m\_tradeConfig.trailingStopLevel" is set and "currentPrice" falls below it, we log the trigger and call "finalizePosition" to close the trade. The zone recovery logic remains unchanged, closing the position if "currentPrice" exceeds "m\_zoneBounds.zoneTargetHigh" or triggering a sell recovery trade with "triggerRecoveryTrade" if it falls below "m\_zoneBounds.zoneLow".

For a sell position ("m\_tradeConfig.direction" as [ORDER\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type)), we fetch "currentPrice" with "getMarketAsk" and calculate "profitPoints" inversely. The trailing stop logic mirrors the buy case, setting "newTrailingStop" by adding "trailingStopPoints" times [\_Point](https://www.mql5.com/en/docs/predefined/_point) to "currentPrice", updating "m\_tradeConfig.trailingStopLevel" if conditions are met, and closing the position if "currentPrice" exceeds it. The zone recovery logic closes the position if "currentPrice" is below "m\_zoneBounds.zoneTargetLow" or triggers a buy recovery trade if above "m\_zoneBounds.zoneHigh". We don't include a physical trailing stop because we want to have full control of the system. That way, we are able to keep all instances monitored and managed. Here is the output after running the program for the trailing stop feature.

![TRAILING STOP INSTANCE](https://c.mql5.com/2/155/Screenshot_2025-07-08_005743.png)

From the image, we can see that we can trail the position and close it when the price falls back the trailing level. Finally, we just create an instance of the basket manager and then use it for the management globally.

```
//--- Global Instance
BasketManager *manager = NULL;

int OnInit() {
   manager = new BasketManager(_Symbol, baseMagicNumber, maxInitialPositions);
   if (!manager.initialize()) {
      delete manager;
      manager = NULL;
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if (manager != NULL) {
      delete manager;
      manager = NULL;
      Print("EA deinitialized");
   }
}

void OnTick() {
   if (manager != NULL) {
      manager.processTick();
   }
}
```

We update the global instance and event handlers to use the new "BasketManager" class, replacing the previous version’s use of the "MarketZoneTrader" [class](https://www.mql5.com/en/docs/basis/types/classes) to support our multi-basket trading improvement by centralizing the management of multiple trade baskets. We start by declaring a global "manager" [pointer](https://www.mql5.com/en/docs/basis/types/object_pointers) to the "BasketManager" class, initialized to "NULL", instead of the previous "trader" pointer to "MarketZoneTrader". This shift is crucial as it allows us to manage multiple trade baskets through a single manager, unlike the single-instance approach in the prior version.

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we create a new "BasketManager" instance for "manager", passing "\_Symbol", "baseMagicNumber", and "maxInitialPositions" to configure it for the current chart, unique basket identification, and the maximum number of baskets. We call "manager.initialize" to set up indicators and load existing positions, and if it fails, we delete "manager", set it to "NULL", and return [INIT\_FAILED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode). On success, we return "INIT\_SUCCEEDED".

In the "OnDeinit" event handler, we check if "manager" is not "NULL", then delete it with "delete", set it to "NULL", and log the deinitialization with "Print". In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick), we check if "manager" is not "NULL" and call "manager.processTick" to handle market ticks across all baskets, replacing the previous call to "trader.processTick". This centralizes tick processing for multiple baskets, enhancing the system’s ability to manage concurrent trade signals. Upon compilation, we have the following outcome.

![FINAL TRADES](https://c.mql5.com/2/155/Screenshot_2025-07-08_010754.png)

From the image, we can see that we can create separate signal baskets and manage them, with different labels constructed from the magic number provided. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/155/Screenshot_2025-07-08_012609.png)

Backtest report:

![REPORT](https://c.mql5.com/2/155/Screenshot_2025-07-08_012643.png)

### Conclusion

In conclusion, we have enhanced our [Zone Recovery System](https://www.mql5.com/go?link=https://tradingliteracy.com/zone-recovery-in-trading/ "https://tradingliteracy.com/zone-recovery-in-trading/") for Envelopes Trend Trading in [MQL5](https://www.mql5.com/) by introducing [trailing stops](https://www.mql5.com/go?link=https://www.investopedia.com/terms/t/trailingstop.asp "https://www.investopedia.com/terms/t/trailingstop.asp") and a multi-basket trading system, building on the foundation from [Part 22](https://www.mql5.com/en/articles/18720) with new components like the "BasketManager" [class](https://www.mql5.com/en/docs/basis/types/classes) and updated "MarketZoneTrader" functions. These improvements offer a more flexible and robust trading framework that you can customize further by adjusting parameters like "trailingStopPoints" or "maxInitialPositions".

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may lead to losses. Thorough backtesting and careful risk management are essential before deploying this program in live markets.

With these enhancements, you can refine this system or adapt its architecture to create new strategies, advancing your algorithmic trading journey. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18778.zip "Download all attachments in the single ZIP archive")

[Envelopes\_Trend\_Bounce\_with\_Zone\_Recovery\_Trailing\_Stop\_EA.mq5](https://www.mql5.com/en/articles/download/18778/envelopes_trend_bounce_with_zone_recovery_trailing_stop_ea.mq5 "Download Envelopes_Trend_Bounce_with_Zone_Recovery_Trailing_Stop_EA.mq5")(28.24 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/490716)**
(1)


![Tyrone Chan](https://c.mql5.com/avatar/2024/7/668972EE-FF50.png)

**[Tyrone Chan](https://www.mql5.com/en/users/tyronechan)**
\|
31 Aug 2025 at 07:20

Sir, No initial sell trades open.

Is it related to trading logic?

![Graph Theory: Dijkstra's Algorithm Applied in Trading](https://c.mql5.com/2/155/18760-graph-theory-dijkstra-s-algorithm-logo.png)[Graph Theory: Dijkstra's Algorithm Applied in Trading](https://www.mql5.com/en/articles/18760)

Dijkstra's algorithm, a classic shortest-path solution in graph theory, can optimize trading strategies by modeling market networks. Traders can use it to find the most efficient routes in the candlestick chart data.

![Neural Networks in Trading: Hyperbolic Latent Diffusion Model (Final Part)](https://c.mql5.com/2/101/Neural_Networks_in_Trading__Hyperbolic_Latent_Diffusion_Model___LOGO2__1.png)[Neural Networks in Trading: Hyperbolic Latent Diffusion Model (Final Part)](https://www.mql5.com/en/articles/16323)

The use of anisotropic diffusion processes for encoding the initial data in a hyperbolic latent space, as proposed in the HypDIff framework, assists in preserving the topological features of the current market situation and improves the quality of its analysis. In the previous article, we started implementing the proposed approaches using MQL5. Today we will continue the work we started and will bring it to its logical conclusion.

![Developing a Replay System (Part 74): New Chart Trade (I)](https://c.mql5.com/2/101/Desenvolvendo_um_sistema_de_Replay_Parte_74___LOGO.png)[Developing a Replay System (Part 74): New Chart Trade (I)](https://www.mql5.com/en/articles/12413)

In this article, we will modify the last code shown in this series about Chart Trade. These changes are necessary to adapt the code to the current replay/simulation system model. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Arithmetic Optimization Algorithm (AOA): From AOA to SOA (Simple Optimization Algorithm)](https://c.mql5.com/2/103/Simple_Optimization_Algorithm___LOGO.png)[Arithmetic Optimization Algorithm (AOA): From AOA to SOA (Simple Optimization Algorithm)](https://www.mql5.com/en/articles/16364)

In this article, we present the Arithmetic Optimization Algorithm (AOA) based on simple arithmetic operations: addition, subtraction, multiplication and division. These basic mathematical operations serve as the foundation for finding optimal solutions to various problems.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pejnbducghjhdjcvwsgbaczpgxnjudgl&ssn=1769180042270989068&ssn_dr=1&ssn_sr=0&fv_date=1769180042&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F18778&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2023)%3A%20Zone%20Recovery%20with%20Trailing%20and%20Basket%20Logic%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918004301732497&fz_uniq=5068776697777552778&sv=2552)

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