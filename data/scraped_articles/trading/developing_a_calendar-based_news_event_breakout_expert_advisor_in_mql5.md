---
title: Developing a Calendar-Based News Event Breakout Expert Advisor in MQL5
url: https://www.mql5.com/en/articles/16752
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 10
scraped_at: 2026-01-22T17:19:26.449668
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/16752&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049044866400363500)

MetaTrader 5 / Examples


### Introduction

Volatility tends to peak around high-impact news events, creating significant breakout opportunities. In this article, we will outline the implementation process of a calendar-based breakout strategy in MQL5. We'll cover everything from creating a class to interpret and store calendar data, developing realistic backtests using this data, and finally, implementing execution code for live trading.

### Motivation

While the MQL5 community offers numerous articles and codebases on handling MetaTrader 5 calendars in backtesting, these resources can be overly complex for beginners aiming to develop a simple breakout strategy. This article seeks to simplify the process of creating a strategy using the news calendar in MQL5 and provide a comprehensive guide for traders.

The motivation for creating a calendar news breakout trading strategy lies in leveraging the predictable timing of scheduled news events—such as economic reports, earnings releases, or geopolitical announcements—that often trigger significant market volatility and price movements. By anticipating these events, traders aim to capitalize on breakout opportunities when prices move decisively beyond established support or resistance levels following the news. This strategy seeks to maximize profits from increased liquidity and momentum around news releases while employing disciplined risk management to navigate the heightened uncertainty. Ultimately, it provides a structured approach to exploit the patterns and reactions that typically occur in the markets around key calendar events.

### Calendar News Backtest

MQL5 has default operations for handling News Calendar data provided by brokers. However, this data cannot be fetched in the strategy tester using the existing code. Therefore, we can create a calendar history **include file** adapted from [Rene Balke](https://www.mql5.com/en/users/bmtrading) that processes the news history data and stores it in a binary file, which we will use later.

- The CCalendarEntry class represents a single economic calendar event with various properties related to the country, event details, and its associated values (e.g., forecast, actual, previous values, etc.).
- The Compare() method compares two calendar events based on their time and importance, returning a value indicating which event is considered greater.
- The ToString() method converts the event data into a human-readable string format, including the event’s importance and other relevant properties.

```
//+------------------------------------------------------------------+
//| A class to represent a single economic calendar event            |
//+------------------------------------------------------------------+
class CCalendarEntry :public CObject {
public:
   ulong country_id;
   string country_name;
   string country_code;
   string country_currency;
   string country_currency_symbol;
   string country_url_name;

   ulong event_id;
   ENUM_CALENDAR_EVENT_TYPE event_type;
   ENUM_CALENDAR_EVENT_SECTOR event_sector;
   ENUM_CALENDAR_EVENT_FREQUENCY event_frequency;
   ENUM_CALENDAR_EVENT_TIMEMODE event_time_mode;
   ENUM_CALENDAR_EVENT_UNIT event_unit;
   ENUM_CALENDAR_EVENT_IMPORTANCE event_importance;
   ENUM_CALENDAR_EVENT_MULTIPLIER event_multiplier;
   uint event_digits;
   string event_source_url;
   string event_event_code;
   string event_name;

   ulong value_id;
   datetime value_time;
   datetime value_period;
   int value_revision;
   long value_actual_value;
   long value_prev_value;
   long value_revised_prev_value;
   long value_forecast_value;
   ENUM_CALENDAR_EVENT_IMPACT value_impact_type;

//+------------------------------------------------------------------+
//| Compare news importance function                                 |
//+------------------------------------------------------------------+
   int Compare(const CObject *node, const int mode = 0) const{
      CCalendarEntry* other = (CCalendarEntry*)node;
      if (value_time==other.value_time){
         return event_importance-other.event_importance;
      }
      return (int)(value_time -other.value_time);
   }

//+------------------------------------------------------------------+
//| Convert data to string function                                  |
//+------------------------------------------------------------------+
   string ToString(){
      string txt;
      string importance = "None";
      if(event_importance==CALENDAR_IMPORTANCE_HIGH)importance="High";
      else if(event_importance==CALENDAR_IMPORTANCE_MODERATE) importance = "Moderate";
      else if(event_importance==CALENDAR_IMPORTANCE_LOW)importance = "Low";
      StringConcatenate(txt,value_time,">",event_name,"(",country_code,"|",country_currency,")",importance);
      return txt;
     }

};
```

- The **CCalendarHistory** class manages a collection of CCalendarEnrtry class objects, extending CArrayObj for array-like functionality, and provides methods for accessing and manipulating the calendar event data.
- The **operator\[\]** method is overridden to return a CCalendarEnrtry object at a specific index in the collection, enabling array-style access to calendar entries.
- The **At()** method returns a pointer to a CCalendarEnrtry at a specified index. It ensures that the index is valid before accessing the array.
- The **LoadCalendarEntriesFromFile()** method loads calendar entries from a binary file, reading the relevant data (e.g., country information, event details) and populating CCalendarEnrtry objects.

```
//+------------------------------------------------------------------+
//| A class to manage a collection of CCalendarEntry objects         |
//+------------------------------------------------------------------+
class CCalendarHistory :public CArrayObj{
public:
//overriding existing operators to better deal with calendar format data
   CCalendarEntry *operator[](const int index) const{return(CCalendarEntry*)At(index);}
   CCalendarEntry *At (const int index) const;
   bool LoadCalendarEntriesFromFile(string fileName);
   bool SaveCalendarValuesToFile(string filename);

};

CCalendarEntry *CCalendarHistory::At(const int index)const{
   if(index<0||index>=m_data_total)return(NULL);
   return (CCalendarEntry*)m_data[index];

}

//+------------------------------------------------------------------+
//| A function to load calendar events from your saved binary file   |
//+------------------------------------------------------------------+
bool CCalendarHistory::LoadCalendarEntriesFromFile(string fileName){
   CFileBin file;
   if(file.Open(fileName,FILE_READ|FILE_COMMON)>0){
      while(!file.IsEnding()){
         CCalendarEntry*entry = new CCalendarEntry();
         int len;
         file.ReadLong(entry.country_id);
         file.ReadInteger(len);
         file.ReadString(entry.country_name,len);
         file.ReadInteger(len);
         file.ReadString(entry.country_code,len);
         file.ReadInteger(len);
         file.ReadString(entry.country_currency,len);
         file.ReadInteger(len);
         file.ReadString(entry.country_currency_symbol,len);
         file.ReadInteger(len);
         file.ReadString(entry.country_url_name,len);

         file.ReadLong(entry.event_id);
         file.ReadEnum(entry.event_type);
         file.ReadEnum(entry.event_sector);
         file.ReadEnum(entry.event_frequency);
         file.ReadEnum(entry.event_time_mode);
         file.ReadEnum(entry.event_unit);
         file.ReadEnum(entry.event_importance);
         file.ReadEnum(entry.event_multiplier);
         file.ReadInteger(entry.event_digits);
         file.ReadInteger(len);
         file.ReadString(entry.event_source_url,len);
         file.ReadInteger(len);
         file.ReadString(entry.event_event_code,len);
         file.ReadInteger(len);
         file.ReadString(entry.event_name,len);

         file.ReadLong(entry.value_id);
         file.ReadLong(entry.value_time);
         file.ReadLong(entry.value_period);
         file.ReadInteger(entry.value_revision);
         file.ReadLong(entry.value_actual_value);
         file.ReadLong(entry.value_prev_value);
         file.ReadLong(entry.value_revised_prev_value);
         file.ReadLong(entry.value_forecast_value);
         file.ReadEnum(entry.value_impact_type);

         CArrayObj::Add(entry);
      }
      Print(__FUNCTION__,">Loaded",CArrayObj::Total(),"Calendar Entries From",fileName,"...");
      CArray::Sort();
      file.Close();
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| A function to save calendar values into a binary file            |
//+------------------------------------------------------------------+
bool CCalendarHistory::SaveCalendarValuesToFile(string fileName){
   CFileBin file;
   if(file.Open(fileName,FILE_WRITE|FILE_COMMON)>0){
     datetime chunk_end   = TimeTradeServer();
      // Let's do ~12 months (adjust as needed).
      int months_to_fetch  = 12*25;

      while(months_to_fetch > 0)
      {
         // For each month, we go back ~30 days
         datetime chunk_start = chunk_end - 30*24*60*60;
         if(chunk_start < 1) // Just a safety check
            chunk_start = 1;
         MqlCalendarValue values[];
         if(CalendarValueHistory(values, chunk_start, chunk_end))
         {
            // Write to file
            for(uint i = 0; i < values.Size(); i++)
            {
               MqlCalendarEvent event;
               if(!CalendarEventById(values[i].event_id,event))
                  continue;  // skip if not found

               MqlCalendarCountry country;
               if(!CalendarCountryById(event.country_id,country))
                  continue;  // skip if not found

       file.WriteLong(country.id);
       file.WriteInteger(country.name.Length());
       file.WriteString(country.name,country.name.Length());
       file.WriteInteger(country.code.Length());
       file.WriteString(country.code,country.code.Length());
       file.WriteInteger(country.currency.Length());
       file.WriteString(country.currency,country.currency.Length());
       file.WriteInteger(country.currency_symbol.Length());
       file.WriteString(country.currency_symbol, country.currency_symbol.Length());
       file.WriteInteger(country.url_name.Length());
       file.WriteString(country.url_name,country.url_name.Length());

       file.WriteLong(event.id);
       file.WriteEnum(event.type);
       file.WriteEnum(event.sector);
       file.WriteEnum(event.frequency);
       file.WriteEnum(event.time_mode);
       file.WriteEnum(event.unit);
       file.WriteEnum(event.importance);
       file.WriteEnum(event.multiplier);
       file.WriteInteger(event.digits);
       file.WriteInteger(event.source_url.Length());
       file.WriteString(event.source_url,event.source_url.Length());
       file.WriteInteger(event.event_code.Length());
       file.WriteString(event.event_code,event.event_code.Length());
       file.WriteInteger(event.name.Length());
       file.WriteString(event.name,event.name.Length());

       file.WriteLong(values[i].id);
       file.WriteLong(values[i].time);
       file.WriteLong(values[i].period);
       file.WriteInteger(values[i].revision);
       file.WriteLong(values[i].actual_value);
       file.WriteLong(values[i].prev_value);
       file.WriteLong(values[i].revised_prev_value);
       file.WriteLong(values[i].forecast_value);
       file.WriteEnum(values[i].impact_type);
     }
            Print(__FUNCTION__, " >> chunk ",
                  TimeToString(chunk_start), " - ", TimeToString(chunk_end),
                  ": saved ", values.Size(), " events.");
         }

         // Move to the previous chunk
         chunk_end = chunk_start;
         months_to_fetch--;

         // short pause to avoid spamming server:
         Sleep(500);
      }

     file.Close();
     return true;
   }
   return false;
}
```

Next, we will create the expert advisor responsible for obtaining backtest results.

For this expert advisor, we will run it on a 5-minute timeframe. For each closed bar, we check if there is a high-impact news event within the next 5 minutes. If so, we place buy stop and sell stop orders within a given deviation from the current bid price, with an optional stop loss. Additionally, we will exit all open positions once a specified time is reached.

Developing a news breakout trading strategy—where buy/sell stops are placed at key levels just before high-impact news events—can be motivated by several strategic and tactical considerations:

- **Explosive Price Movement:** High-impact news often triggers significant price swings. Positioning stops near key levels can help traders enter as soon as a breakout occurs, capturing large price moves.
- **Enhanced Risk-Reward:** Quick, volatile moves can present favorable risk-reward setups if the trader’s stops are triggered in the direction of the breakout.

- **Predictability of News Releases:** Since the timing of high-impact news is known in advance, traders can plan entries and exits more precisely, reducing the uncertainty around market timing.
- **Anticipating Liquidity Surge:** News releases often attract increased market participation, which can lead to more reliable breakouts when key levels are breached.

- **Pre-Planned Execution:** Setting stops at key technical levels before the news helps remove emotional decision-making at the moment of market shock, leading to more disciplined execution.
- **Automation Possibility:** Placing orders in advance can allow for automated execution as soon as the news hits, ensuring rapid response to market moves without the need for manual intervention.

By combining these motivations, we aim to develop a systematic approach that capitalizes on the predictability and volatility of high-impact news events while maintaining disciplined risk management and clear execution rules.

The expert advisor starts by including the necessary helper files and creating objects for the related classes. It also declares the relevant global variables that will be used later.

```
#define FILE_NAME "CalendarHistory.bin"
#include <Trade/Trade.mqh>
#include <CalendarHistory.mqh>
#include <Arrays/ArrayString.mqh>
CCalendarHistory calendar;
CTrade trade;
CArrayString curr;

ulong poss, buypos = 0, sellpos=0;
input int Magic = 0;
int barsTotal = 0;
int currentIndex = 0;
datetime s_lastUpdate = 0;
input int closeTime = 18;
input int slp = 1000;
input int Deviation = 1000;
input string Currencies = "USD";
input ENUM_CALENDAR_EVENT_IMPORTANCE Importance = CALENDAR_IMPORTANCE_HIGH;
input bool saveFile = true;
```

The OnInit() initializer function assures the following things:

- If _saveFile_ is true, the calendar entries are saved to a file named "CalendarHistory.bin".
- The calendar events are then loaded from this file. But you can't save and load at the same time because the saving method closes the file in the end.
- The _Currencies_ variable input string is split into an array of individual currencies, and the array is sorted. So, if you want both USD and EUR currency related events, simply input "USD";"EUR".
- The Magic number is assigned to the CTrade object to identify trades initiated by this EA.

```
//+------------------------------------------------------------------+
//| Initializer function                                             |
//+------------------------------------------------------------------+
int OnInit() {
   if(saveFile==true)calendar.SaveCalendarValuesToFile(FILE_NAME);
   calendar.LoadCalendarEntriesFromFile(FILE_NAME);
   string arr[];
   StringSplit(Currencies,StringGetCharacter(";",0),arr);
   curr.AddArray(arr);
   curr.Sort();
   trade.SetExpertMagicNumber(Magic);
   return(INIT_SUCCEEDED);
}
```

Here are the necessary functions to perform the execution tasks.

- **OnTradeTransaction**: Monitors incoming trade transactions and updates the buypos or sellpos with the order ticket when a buy or sell order with the specified magic number is added.
- **executeBuy**: Places a buy stop order at the given price with a calculated stop loss and records the resulting order ticket in buypos.
- **executeSell**: Places a sell stop order at the given price with a calculated stop loss and records the resulting order ticket in sellpos.
- **IsCloseTime**: Checks the current server time to determine if it has passed the predefined closing hour.

```
//+------------------------------------------------------------------+
//| A function for handling trade transaction                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans, const MqlTradeRequest& request, const MqlTradeResult& result) {
    if (trans.type == TRADE_TRANSACTION_ORDER_ADD) {
        COrderInfo order;
        if (order.Select(trans.order)) {
            if (order.Magic() == Magic) {
                if (order.OrderType() == ORDER_TYPE_BUY) {
                    buypos = order.Ticket();
                } else if (order.OrderType() == ORDER_TYPE_SELL) {
                    sellpos = order.Ticket();
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Buy execution function                                           |
//+------------------------------------------------------------------+
void executeBuy(double price) {
       double sl = price- slp*_Point;
       sl = NormalizeDouble(sl, _Digits);
       double lots=0.1;
       trade.BuyStop(lots,price,_Symbol,sl,0,ORDER_TIME_DAY,1);
       buypos = trade.ResultOrder();
       }

//+------------------------------------------------------------------+
//| Sell execution function                                          |
//+------------------------------------------------------------------+
void executeSell(double price) {
       double sl = price + slp * _Point;
       sl = NormalizeDouble(sl, _Digits);
       double lots=0.1;
       trade.SellStop(lots,price,_Symbol,sl,0,ORDER_TIME_DAY,1);
       sellpos = trade.ResultOrder();
       }

//+------------------------------------------------------------------+
//| Exit time boolean function                                       |
//+------------------------------------------------------------------+
bool IsCloseTime(){
   datetime currentTime = TimeTradeServer();
   MqlDateTime timeStruct;
   TimeToStruct(currentTime,timeStruct);
   int currentHour =timeStruct.hour;
   return(currentHour>closeTime);
}
```

Finally, we just implement the execution logic in the OnTick() function.

```
//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
    int bars = iBars(_Symbol,PERIOD_CURRENT);

    if (barsTotal!= bars){
      barsTotal = bars;
      double bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);
   datetime now = TimeTradeServer();
   datetime horizon = now + 5*60; // 5 minutes from now

   while (currentIndex < calendar.Total())
   {
         CCalendarEntry*entry=calendar.At(currentIndex);
         if (entry.value_time < now)
         {
            currentIndex++;
            continue;
         }
         // Now if the next event time is beyond horizon, break out
         if (entry.value_time > horizon)
            break;

         // If it is within the next 5 minutes, check other conditions:
         if (entry.event_importance >= Importance && curr.SearchFirst(entry.country_currency) >= 0 && buypos == sellpos )
         {
             double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
             executeBuy(bid + Deviation*_Point);
             executeSell(bid - Deviation*_Point);
         }
         currentIndex++;
      }
    if(IsCloseTime()){
       for(int i = 0; i<PositionsTotal(); i++){
         poss = PositionGetTicket(i);
         if(PositionGetInteger(POSITION_MAGIC) == Magic) trade.PositionClose(poss);
      }
    }
     if(buypos>0&&(!PositionSelectByTicket(buypos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      buypos = 0;
      }
     if(sellpos>0&&(!PositionSelectByTicket(sellpos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      sellpos = 0;
      }

   }
}
```

This assures that we check the condition upon new closed bars. If the current bar is the same as the last saved bar, that implies it's not a new bar yet, so we would return without executing the remainder of the trading logic.

```
    int bars = iBars(_Symbol,PERIOD_CURRENT);

    if (barsTotal!= bars){
      barsTotal = bars;
```

The while loop logic ensures efficient backtesting by starting from the beginning of the data. If the current time is later than a given event, we increment the global variable index, preventing the need to loop from the beginning again. This helps reduce computing time and memory usage during our backtesting, dramatically speeding up the process and saving a lot of time—especially over long test periods.

```
while (currentIndex < calendar.Total())
   {
         CCalendarEntry*entry=calendar.At(currentIndex);
         if (entry.value_time < now)
         {
            currentIndex++;
            continue;
         }
         // Now if the next event time is beyond horizon, break out
         if (entry.value_time > horizon)
            break;

         // If it is within the next 5 minutes, check other conditions:
         if (entry.event_importance >= Importance && curr.SearchFirst(entry.country_currency) >= 0 && buypos == sellpos )
         {
             double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
             executeBuy(bid + Deviation*_Point);
             executeSell(bid - Deviation*_Point);
         }
         currentIndex++;
      }
```

This part checks if the current hour is past the close time. If so, it loops through the portfolio, checks for open positions with the EA's magic number, and closes them. Once closed, the position ticket is reset to zero.

```
    if(IsCloseTime()){
       for(int i = 0; i<PositionsTotal(); i++){
         poss = PositionGetTicket(i);
         if(PositionGetInteger(POSITION_MAGIC) == Magic) trade.PositionClose(poss);
      }
    }
     if(buypos>0&&(!PositionSelectByTicket(buypos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      buypos = 0;
      }
     if(sellpos>0&&(!PositionSelectByTicket(sellpos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      sellpos = 0;
      }
```

Now compile the program and head to the MetaTrader 5 terminal. Open any chart and drag this EA onto the chart like this:

![Dragging EA](https://c.mql5.com/2/110/Drag_EA.jpg)

Make sure that the setting saveFile is set to "true".

![set to true](https://c.mql5.com/2/110/saveFile.jpg)

The news event data provided by your broker is the same across all charts, so the symbols don't matter. The EA will initialize immediately upon attachment to your chart, which means the file will be saved at that point. After a few seconds, you can remove the EA. The file will be saved to the [common files path](https://www.mql5.com/en/docs/constants/io_constants/fileflags) on your computer, and you can navigate there to verify that the binary file was saved.

Now you can test the strategy in the strategy tester.

![parameters](https://c.mql5.com/2/110/parameters_news.png)

Important notes about your backtest parameters:

- Set the _saveFile_ variable to "false" to prevent the binary data file from closing upon initialization.

- Select a reasonable deviation and stop loss. A deviation that is too wide will fail to capture the news event volatility, while a deviation or stop loss that is too narrow will be prone to slippage during news spikes.

- Choose a reasonable position closing time. I recommend selecting an hour close to the market close or the end of the trading day, so you can capture the entire news-driven move. The corresponding input hour will depend on your broker's server time.


Here is my backtest on SPIUSDc from 2019.1.1 – 2024.12.1 on the 5-minute timeframe.

![setting](https://c.mql5.com/2/110/setting_news.png)

![equity curve](https://c.mql5.com/2/110/curve_news.png)

![result](https://c.mql5.com/2/110/result_news.png)

Key results:

- Profit factor: 1.26
- Sharpe ratio: 2.66
- Number of trades: 1604

Finally, I suggest that readers use real ticks if available. Select stress testing on high latency to account for slippage and high spreads during high-impact news events. Last but not least, demo trade in live conditions to validate that your backtest results are trustworthy. Some brokers have significant slippage, while others may have minimal slippage. When trading a strategy that relies on high volatility, always take additional measures to ensure the strategy is profitable in live trading.

### Live Trading Implementation

For live trading, we will use a separate expert advisor code. We can now utilize the calendar operations from MQL5, implementing the same logic as before. The only difference is that we create a function to update the upcoming news events and store them in the array of the calendar object we created, updating it every hour.

The logic of this code updates a calendar history by fetching new event data from a calendar API if more than an hour has passed since the last update, then processes and stores the event details, including country, value, and forecast data, into a new CCalendarEntry object for each event.

```
//+------------------------------------------------------------------+
//| Update upcoming news events                                      |
//+------------------------------------------------------------------+
void UpdateCalendarHistory(CCalendarHistory &history)
{
   //upcoming event in the next hour
   datetime fromTime = TimeTradeServer()+3600;
   // For example, if it's been > 1hr since last update:
   if(fromTime - s_lastUpdate > 3600)
   {
      // Determine the time range to fetch new events
      // For instance, from s_lastUpdate to 'now'
      MqlCalendarValue values[];
      if(CalendarValueHistory(values, s_lastUpdate, fromTime))
      {
         for(uint i = 0; i < values.Size(); i++)
         {
            MqlCalendarEvent event;
            if(!CalendarEventById(values[i].event_id,event))
               continue;
            MqlCalendarCountry country;
            if(!CalendarCountryById(event.country_id, country))
               continue;
            // Create a new CCalendarEntry and fill from 'values[i]', 'event', 'country'
            CCalendarEntry *entry = new CCalendarEntry();
            entry.country_id = country.id;
            entry.value_time               = values[i].time;
            entry.value_period             = values[i].period;
            entry.value_revision           = values[i].revision;
            entry.value_actual_value       = values[i].actual_value;
            entry.value_prev_value         = values[i].prev_value;
            entry.value_revised_prev_value = values[i].revised_prev_value;
            entry.value_forecast_value     = values[i].forecast_value;
            entry.value_impact_type        = values[i].impact_type;
            // event data
            entry.event_id             = event.id;
            entry.event_type           = event.type;
            entry.event_sector         = event.sector;
            entry.event_frequency      = event.frequency;
            entry.event_time_mode      = event.time_mode;
            entry.event_unit           = event.unit;
            entry.event_importance     = event.importance;
            entry.event_multiplier     = event.multiplier;
            entry.event_digits         = event.digits;
            entry.event_source_url     = event.source_url;
            entry.event_event_code     = event.event_code;
            entry.event_name           = event.name;
            // country data
            entry.country_name         = country.name;
            entry.country_code         = country.code;
            entry.country_currency     = country.currency;
            entry.country_currency_symbol = country.currency_symbol;
            entry.country_url_name     = country.url_name;
            // Add to your in-memory calendar
            history.Add(entry);
         }
      }
      // Sort to keep chronological order
      history.Sort();
      // Mark the last update time
      s_lastUpdate = fromTime;
   }
}
```

The rest of the code is almost the exact same as the backtest EA. You can simply integrate the code into the execution EA like this, and we're done.

```
#include <Trade/Trade.mqh>
#include <CalendarHistory.mqh>
#include <Arrays/ArrayString.mqh>
CCalendarHistory calendar;
CArrayString curr;
CTrade trade;

ulong poss, buypos = 0, sellpos=0;
input int Magic = 0;
int barsTotal = 0;
datetime s_lastUpdate = 0;
input int closeTime = 18;
input int slp = 1000;
input int Deviation = 1000;
input string Currencies = "USD";
input ENUM_CALENDAR_EVENT_IMPORTANCE Importance = CALENDAR_IMPORTANCE_HIGH;

//+------------------------------------------------------------------+
//| Initializer function                                             |
//+------------------------------------------------------------------+
int OnInit() {
   trade.SetExpertMagicNumber(Magic);
   string arr[];
   StringSplit(Currencies,StringGetCharacter(";",0),arr);
   curr.AddArray(arr);
   curr.Sort();
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Destructor function                                              |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {

  }

//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
    int bars = iBars(_Symbol,PERIOD_CURRENT);

    if (barsTotal!= bars){
      barsTotal = bars;
      UpdateCalendarHistory(calendar);
      double bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);
   datetime now = TimeTradeServer();
   datetime horizon = now + 5*60; // 5 minutes from now

   // Loop over all loaded events
   for(int i = 0; i < calendar.Total(); i++)
   {
      CCalendarEntry *entry = calendar.At(i);

      // If event time is between 'now' and 'now+5min'
      if(entry.value_time > now && entry.value_time <= horizon&&buypos==sellpos&&entry.event_importance>=Importance&&curr.SearchFirst(entry.country_currency)>=0)
      {
        executeBuy(bid+Deviation*_Point);
        executeSell(bid-Deviation*_Point);
        }
     }
    if(IsCloseTime()){
       for(int i = 0; i<PositionsTotal(); i++){
         poss = PositionGetTicket(i);
         if(PositionGetInteger(POSITION_MAGIC) == Magic) trade.PositionClose(poss);
      }
    }
     if(buypos>0&&(!PositionSelectByTicket(buypos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      buypos = 0;
      }
     if(sellpos>0&&(!PositionSelectByTicket(sellpos)|| PositionGetInteger(POSITION_MAGIC) != Magic)){
      sellpos = 0;
      }
   }
}

//+------------------------------------------------------------------+
//| A function for handling trade transaction                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans, const MqlTradeRequest& request, const MqlTradeResult& result) {
    if (trans.type == TRADE_TRANSACTION_ORDER_ADD) {
        COrderInfo order;
        if (order.Select(trans.order)) {
            if (order.Magic() == Magic) {
                if (order.OrderType() == ORDER_TYPE_BUY) {
                    buypos = order.Ticket();
                } else if (order.OrderType() == ORDER_TYPE_SELL) {
                    sellpos = order.Ticket();
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Buy execution function                                           |
//+------------------------------------------------------------------+
void executeBuy(double price) {
       double sl = price- slp*_Point;
       sl = NormalizeDouble(sl, _Digits);
       double lots=0.1;
       trade.BuyStop(lots,price,_Symbol,sl,0,ORDER_TIME_DAY,1);
       buypos = trade.ResultOrder();
       }

//+------------------------------------------------------------------+
//| Sell execution function                                          |
//+------------------------------------------------------------------+
void executeSell(double price) {
       double sl = price + slp * _Point;
       sl = NormalizeDouble(sl, _Digits);
       double lots=0.1;
       trade.SellStop(lots,price,_Symbol,sl,0,ORDER_TIME_DAY,1);
       sellpos = trade.ResultOrder();
       }

//+------------------------------------------------------------------+
//| Update upcoming news events                                      |
//+------------------------------------------------------------------+
void UpdateCalendarHistory(CCalendarHistory &history)
{
   //upcoming event in the next hour
   datetime fromTime = TimeTradeServer()+3600;
   // For example, if it's been > 1hr since last update:
   if(fromTime - s_lastUpdate > 3600)
   {
      // Determine the time range to fetch new events
      // For instance, from s_lastUpdate to 'now'
      MqlCalendarValue values[];
      if(CalendarValueHistory(values, s_lastUpdate, fromTime))
      {
         for(uint i = 0; i < values.Size(); i++)
         {
            MqlCalendarEvent event;
            if(!CalendarEventById(values[i].event_id,event))
               continue;
            MqlCalendarCountry country;
            if(!CalendarCountryById(event.country_id, country))
               continue;
            // Create a new CCalendarEntry and fill from 'values[i]', 'event', 'country'
            CCalendarEntry *entry = new CCalendarEntry();
            entry.country_id = country.id;
            entry.value_time               = values[i].time;
            entry.value_period             = values[i].period;
            entry.value_revision           = values[i].revision;
            entry.value_actual_value       = values[i].actual_value;
            entry.value_prev_value         = values[i].prev_value;
            entry.value_revised_prev_value = values[i].revised_prev_value;
            entry.value_forecast_value     = values[i].forecast_value;
            entry.value_impact_type        = values[i].impact_type;
            // event data
            entry.event_id             = event.id;
            entry.event_type           = event.type;
            entry.event_sector         = event.sector;
            entry.event_frequency      = event.frequency;
            entry.event_time_mode      = event.time_mode;
            entry.event_unit           = event.unit;
            entry.event_importance     = event.importance;
            entry.event_multiplier     = event.multiplier;
            entry.event_digits         = event.digits;
            entry.event_source_url     = event.source_url;
            entry.event_event_code     = event.event_code;
            entry.event_name           = event.name;
            // country data
            entry.country_name         = country.name;
            entry.country_code         = country.code;
            entry.country_currency     = country.currency;
            entry.country_currency_symbol = country.currency_symbol;
            entry.country_url_name     = country.url_name;
            // Add to your in-memory calendar
            history.Add(entry);
         }
      }
      // Sort to keep chronological order
      history.Sort();
      // Mark the last update time
      s_lastUpdate = fromTime;
   }
}

//+------------------------------------------------------------------+
//| Exit time boolean function                                       |
//+------------------------------------------------------------------+
bool IsCloseTime(){
   datetime currentTime = TimeTradeServer();
   MqlDateTime timeStruct;
   TimeToStruct(currentTime,timeStruct);
   int currentHour =timeStruct.hour;
   return(currentHour>closeTime);
}
```

For future strategy development, you can build on the foundation outlined in this article and explore ideas focused on news-based trading. For example:

**Trading Breakouts at Key Levels**

Instead of relying on fixed deviations, this approach focuses on price movements triggered by significant news breaking through key support or resistance levels. For instance, when major economic reports or corporate announcements cause a breakout, you can enter trades in the direction of the move. To implement this, monitor the news schedule and identify critical price levels in advance.

**Fading the News**

This strategy involves trading against the market's initial reaction to a news event, assuming that the first move is an overreaction. After a sharp price spike following the news, you wait for the market to correct itself and then enter a trade in the opposite direction.

**Filtering News Events**

If your strategy works best in low-volatility markets, you can avoid trading during high-impact news events. By checking the news calendar in advance, you can program your trading system to close open positions or pause new trades until after the event, ensuring more stable market conditions.

**News Scalping**

This idea focuses on capturing small profits from short-term price movements caused by news events. It involves quick entries and exits, tight stop-losses, and fast profit-taking. This strategy is particularly effective during volatile events that cause rapid price swings.

**Economic Calendar Trading**

This approach revolves around scheduled events in the economic calendar, such as interest rate decisions, GDP announcements, or employment reports. By analyzing how markets have historically reacted to similar news and factoring in current expectations, you can anticipate potential price movements and prepare accordingly.

Each of these strategies relies on gathering and analyzing relevant news data in advance, allowing you to make the most of market volatility caused by important events.

### Conclusion

In this article, we created a helper include file to interpret, format, and store calendar news event data. Next, we developed a backtest expert advisor to fetch the news event data provided by brokers, implement the strategy logic, and test the results of our trading strategy using this data. The strategy showed promising profitability, with over 1,600 samples across 5 years of tick data. Finally, we shared the live execution expert advisor code for live trading and outlined future aspirations, encouraging further development of strategies built on the framework introduced in this article.

**File Table**

| File Name | File Usage |
| --- | --- |
| CalendarHistory.mqh | The include helper file for handling calendar news event data. |
| News Breakout Backtest.mq5 | The expert advisor for storing news event data and performing backtests. |
| News Breakout.mq5 | The expert advisor for live trading. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16752.zip "Download all attachments in the single ZIP archive")

[The\_Calendar\_News\_Event\_Breakout\_Strategy.zip](https://www.mql5.com/en/articles/download/16752/the_calendar_news_event_breakout_strategy.zip "Download The_Calendar_News_Event_Breakout_Strategy.zip")(6.77 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Decoding Opening Range Breakout Intraday Trading Strategies](https://www.mql5.com/en/articles/17745)
- [Day Trading Larry Connors RSI2 Mean-Reversion Strategies](https://www.mql5.com/en/articles/17636)
- [Exploring Advanced Machine Learning Techniques on the Darvas Box Breakout Strategy](https://www.mql5.com/en/articles/17466)
- [The Kalman Filter for Forex Mean-Reversion Strategies](https://www.mql5.com/en/articles/17273)
- [Robustness Testing on Expert Advisors](https://www.mql5.com/en/articles/16957)
- [Trend Prediction with LSTM for Trend-Following Strategies](https://www.mql5.com/en/articles/16940)
- [The Inverse Fair Value Gap Trading Strategy](https://www.mql5.com/en/articles/16659)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/480045)**
(6)


![Zhuo Kai Chen](https://c.mql5.com/avatar/2024/11/6743e84b-8a3d.jpg)

**[Zhuo Kai Chen](https://www.mql5.com/en/users/sicklemql)**
\|
16 May 2025 at 10:28

**hrawoodward [#](https://www.mql5.com/en/forum/480045#comment_56714474):**

Hi, this is great thanks! I'm a bit confused about entering multiple currencies. I've tried:

"USD";"GBP"

"USD";"GBP.

"USD" "GBP";

Only the last one doesn't produce an error, but i'm not sure it works correctly. Maybe it only picks up the USD. Can you advise?

Hello, if you look at the code in the initialization function, it will split the colon and store different currencies into the curr object attribute. You first one should work, although you don't need to add the quotations. The storing process will store all the events into the binary file regardless of their attributes. Only in trade logic will we filter for the attributes. Here's what I have run just now:

[![settings](https://c.mql5.com/3/464/Screenshot_2025-05-16_at_18.23.38__1.png)](https://c.mql5.com/3/464/Screenshot_2025-05-16_at_18.23.38.png "https://c.mql5.com/3/464/Screenshot_2025-05-16_at_18.23.38.png")

[![result](https://c.mql5.com/3/464/Screenshot_2025-05-16_at_18.23.18__1.png)](https://c.mql5.com/3/464/Screenshot_2025-05-16_at_18.23.18.png "https://c.mql5.com/3/464/Screenshot_2025-05-16_at_18.23.18.png")

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
16 May 2025 at 16:22

It looks like this implementation does not look at timezone switches (DST) on the broker server and hence produces inaccurate results during backtesting and optimizations.


![Zhuo Kai Chen](https://c.mql5.com/avatar/2024/11/6743e84b-8a3d.jpg)

**[Zhuo Kai Chen](https://www.mql5.com/en/users/sicklemql)**
\|
17 May 2025 at 03:17

**Stanislav Korotky [#](https://www.mql5.com/en/forum/480045#comment_56719040):**

It looks like this implementation does not look at timezone switches (DST) on the broker server and hence produces inaccurate results during backtesting and optimizations.

Thanks for reminding me! I forgot to consider it in the article cuz I used a broker that doesn't have DST for demonstration.

[https://www.mql5.com/en/book/advanced/calendar](https://www.mql5.com/en/book/advanced/calendar "https://www.mql5.com/en/book/advanced/calendar")

From this source we know that the calendar data is provided from the MQL5 side, and it is automatically adjusted to the broker's **current** Timetradeserver() time zone, meaning that for brokers with DST would need to adjust my code and take it into account.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
17 May 2025 at 13:05

**Zhuo Kai Chen [#](https://www.mql5.com/en/forum/480045#comment_56720891):**

From this source we know that the calendar data is provided from the MQL5 side, and it is automatically adjusted to the broker's **current** Timetradeserver() time zone, meaning that for brokers with DST would need to adjust my code and take it into account.

Since the published in the book implementation is a bit outdated, the actual (updated) story can be found in the [blog](https://www.mql5.com/en/blogs/post/759851) and the [codebase](https://www.mql5.com/en/code/53393) (indicator) and in the [codebase](https://www.mql5.com/en/code/52977) (script).

![TL_TL_TL](https://c.mql5.com/avatar/avatar_na2.png)

**[TL\_TL\_TL](https://www.mql5.com/en/users/tl_tl_tl)**
\|
27 Sep 2025 at 19:41

Brother, this CalendarHistory.mqh file does not compile, 4 errors, 94, 106, 114, 122 lines.


![Price Action Analysis Toolkit Development (Part 8): Metrics Board](https://c.mql5.com/2/112/Price_Action_Analysis_Toolkit_Development_Part_8___LOGO2.png)[Price Action Analysis Toolkit Development (Part 8): Metrics Board](https://www.mql5.com/en/articles/16584)

As one of the most powerful Price Action analysis toolkits, the Metrics Board is designed to streamline market analysis by instantly providing essential market metrics with just a click of a button. Each button serves a specific function, whether it’s analyzing high/low trends, volume, or other key indicators. This tool delivers accurate, real-time data when you need it most. Let’s dive deeper into its features in this article.

![Introduction to MQL5 (Part 11): A Beginner's Guide to Working with Built-in Indicators in MQL5 (II)](https://c.mql5.com/2/112/Introduction_to_MQL5_Part_10___LOGO.png)[Introduction to MQL5 (Part 11): A Beginner's Guide to Working with Built-in Indicators in MQL5 (II)](https://www.mql5.com/en/articles/16956)

Discover how to develop an Expert Advisor (EA) in MQL5 using multiple indicators like RSI, MA, and Stochastic Oscillator to detect hidden bullish and bearish divergences. Learn to implement effective risk management and automate trades with detailed examples and fully commented source code for educational purposes!

![MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://c.mql5.com/2/112/MQL5_Trading_Toolkit_Part_7___LOGO.png)[MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://www.mql5.com/en/articles/16906)

Learn how to complete the creation of the final module in the History Manager EX5 library, focusing on the functions responsible for handling the most recently canceled pending order. This will provide you with the tools to efficiently retrieve and store key details related to canceled pending orders with MQL5.

![Implementing the SHA-256 Cryptographic Algorithm from Scratch in MQL5](https://c.mql5.com/2/112/Implementing_the_SHA-256_Cryptographic_Algorithm_from_Scratch_in_MQL5__LOGO.png)[Implementing the SHA-256 Cryptographic Algorithm from Scratch in MQL5](https://www.mql5.com/en/articles/16357)

Building DLL-free cryptocurrency exchange integrations has long been a challenge, but this solution provides a complete framework for direct market connectivity.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dnaarpyouwihtzbvhxcagtetjztptnzu&ssn=1769091564587357447&ssn_dr=1&ssn_sr=0&fv_date=1769091564&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16752&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Calendar-Based%20News%20Event%20Breakout%20Expert%20Advisor%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909156502487578&fz_uniq=5049044866400363500&sv=2552)

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