---
title: Developing a multi-currency Expert Advisor (Part 16): Impact of different quote histories on test results
url: https://www.mql5.com/en/articles/15330
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:27:58.764156
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pmsarrzzwujajgxabxqzyogwocvkluon&ssn=1769092076240774016&ssn_dr=0&ssn_sr=0&fv_date=1769092076&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15330&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20multi-currency%20Expert%20Advisor%20(Part%2016)%3A%20Impact%20of%20different%20quote%20histories%20on%20test%20results%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909207650496468&fz_uniq=5049152094553875948&sv=2552)

MetaTrader 5 / Tester


### Introduction

In the [previous](https://www.mql5.com/en/articles/15294) article, we started preparing the multi-currency EA for trading on a real account. As part of the preparation process, we added support for different names of trading instruments, automatic completion of trading when you want to change the settings of trading strategies, and correct resumption of the EA after restarting for various reasons.

The preparation activities do not end there. We have outlined a few more necessary steps, but we will return to them later. Now let's look at such an important aspect as ensuring similar results across different brokers. It is known that the quotes for trading instruments at different brokers are not identical. Therefore, by testing and optimizing on some quotes, we select the optimal parameters specifically for them. Of course, we hope that when we start trading on other quotes, their differences from the quotes used for testing will be minor, and therefore, the differences in trading results will also be insignificant.

However, this is too important a question to be left without detailed examination. So let's see how our EA behaves when tested on quotes from different brokers.

### Comparing results

First, let's launch our EA on quotes from the MetaQuotes-Demo server. The first launch was with the risk manager enabled. However, looking ahead, we will say that on other quotes the risk manager completed trading significantly earlier than the end of the test period, so we will disable it to get the full picture. This way we can ensure a fairer comparison of results. Here's what got:

![](https://c.mql5.com/2/115/992868497315__1.png)

![](https://c.mql5.com/2/115/3713696569130__2.png)

Fig. 1. Test results on MetaQuotes-Demo server quotes without the risk manager

Now let's connect the terminal to the real server of another broker and run the EA testing again with the same parameters:

![](https://c.mql5.com/2/115/1139766110287__1.png)

![](https://c.mql5.com/2/115/3613388691857__1.png)

Fig. 2. Test results on quotes of a real server of another broker without the risk manager

This is an unexpected turn of events. The account was completely drained in less than a year. Let's try to understand the reasons behind this behavior so that we can understand whether it is possible to somehow correct the situation.

### Looking for the reason

Let's save the tester reports for the completed passes as XML files, open them and find the place where the list of completed deals begins. Arrange the open file windows so that we can see the top parts of the deal lists for both reports at the same time:

![](https://c.mql5.com/2/115/5863796149063__1.png)

Fig. 3. The top parts of the lists of deals performed by the EA when testing on quotes from different servers

Even from the first few lines of the reports it is clear that the positions were opened at different times. Therefore, if there were any differences in quotes for the same moments of time on different servers, they most likely would not have such a destructive effect as different open times.

Let's see where position opening moments are determined in our strategies. We should have a look at the file implementing the class of a single instance of the _SimpleVolumesStrategy.mqh_ trading strategy. If we look into the code, we will find the _SignalForOpen()_ method returning the open signal:

```
//+------------------------------------------------------------------+
//| Signal for opening pending orders                                |
//+------------------------------------------------------------------+
int CSimpleVolumesStrategy::SignalForOpen() {
// By default, there is no signal
   int signal = 0;

// Copy volume values from the indicator buffer to the receiving array
   int res = CopyBuffer(m_iVolumesHandle, 0, 0, m_signalPeriod, m_volumes);

// If the required amount of numbers have been copied
   if(res == m_signalPeriod) {
      // Calculate their average value
      double avrVolume = ArrayAverage(m_volumes);

      // If the current volume exceeds the specified level, then
      if(m_volumes[0] > avrVolume * (1 + m_signalDeviation + m_ordersTotal * m_signaAddlDeviation)) {
         // if the opening price of the candle is less than the current (closing) price, then
         if(iOpen(m_symbol, m_timeframe, 0) < iClose(m_symbol, m_timeframe, 0)) {
            signal = 1; // buy signal
         } else {
            signal = -1; // otherwise, sell signal
         }
      }
   }

   return signal;
}
```

We see that the opening signal is determined by the tick volume values for the current trading instrument. Prices (both current and past) do not participate in the formation of the opening signal. More precisely, their participation is present after it has been determined that a position needs to be opened, and it only influences the direction of opening. Therefore, it seems that the issue is precisely in the strong differences in the tick volume values received from different servers.

This is quite possible, since in order for different brokers to visually match candlestick price charts, it is sufficient to give only four correct ticks per minute to build Open, Close, High and Low prices for the candle of the shortest period M1. The number of intermediate ticks, during which the price was within the specified limits between Low and High, is of no importance. This means that brokers are free to decide how many ticks to store in history and how they are to be distributed over time within one candle. It is also worth remembering that even with one broker, servers for demo accounts and for real accounts may not show exactly the same picture.

If this is really the case, then we can easily get around this obstacle. But to implement such a workaround, we would first want to make sure that we have correctly identified the cause of the observed discrepancies, so that our efforts are not wasted.

### Mapping out the path

To test our assumption, we will need the following tools:

- **Saving history**. Let's add to our EA the ability to save the history of deals (opening and closing positions) at the end of the tester run. Saving can be done either to a file or to a database. Since this tool will only be used as an auxiliary one for now, it is probably easier to use saving to a file. If we want to use it on a more permanent basis in the future, we can expand it to include the ability to save history to a database.

- **Trading replay**. Let's create a new EA that does not contain any rules for opening positions, but will only reproduce the opening and closing of positions, reading them from the history saved by another EA. Since we have decided to save the history to a file for now, this EA will accept the name of the file with the history of deals as an input, and then read and execute the deals saved in it.

Having made these tools, we will first launch our EA in the tester, using quotes from the MetaQuotes-Demo server and save the trading history of this test pass to the file. This will be the first pass. Then we will launch a new trading replay EA in the tester on quotes from another server using the saved history file. This will be the second pass. If the differences in trading results obtained earlier are indeed due to very different tick volume data, and the prices themselves are approximately the same, then in the second pass we should get results similar to the results of the first pass.

### Saving history

There are different ways to implement history saving. For example, we can add the method to the _CVirtualAdvisor_ class, which will be called from the _OnTester()_ event. This method forces us to extend an existing class, adding functionality that it can actually do without. So, let's make a separate class _CExpertHistory_ to solve this specific problem. We do not need to create multiple objects of this class, so we can make it static, that is, containing only static properties and methods.

There will be only one main public method of the class — _Export()_. The remaining methods will play a supporting role. The _Export()_ method receives two parameters: the name of the file to write history to and the flag to use the shared terminal data folder. The default file name may be an empty string. In this case, an auxiliary method will be used to generate the _GetHistoryFileName()_ file. Using the flag for writing to the shared folder, we can choose where the history file will be saved - to the shared data folder or to the local terminal data folder. By default, the flag value will be set for writing to the shared folder, since when running in the tester, it is more difficult to open the local folder of the test agent than the shared folder.

As class properties, we will need the separator character specified when opening a CSV file for writing, the handle of the opened file itself so that it can be used in auxiliary methods, and the array of column names of the data being saved.

```
//+------------------------------------------------------------------+
//| Export trade history to file                                     |
//+------------------------------------------------------------------+
class CExpertHistory {
private:
   static string     s_sep;            // Separator character
   static int        s_file;           // File handle for writing
   static string     s_columnNames[];  // Array of column names

   // Write deal history to file
   static void       WriteDealsHistory();

   // Write one row of deal history to file
   static void       WriteDealsHistoryRow(const string &fields[]);

   // Get the first deal date
   static datetime   GetStartDate();

   // Form a file name
   static string     GetHistoryFileName();

public:
   // Export deal history
   static void       Export(
      string exportFileName = "",   // File name for export. If empty, the name is generated
      int commonFlag = FILE_COMMON  // Save the file in shared data folder
   );
};

// Static class variables
string CExpertHistory::s_sep = ",";
int    CExpertHistory::s_file;
string CExpertHistory::s_columnNames[] = {"DATE", "TICKET", "TYPE",
                                          "SYMBOL", "VOLUME", "ENTRY", "PRICE",
                                          "STOPLOSS", "TAKEPROFIT", "PROFIT",
                                          "COMMISSION", "FEE", "SWAP",
                                          "MAGIC", "COMMENT"
                                         };
```

In the main _Export()_ method, we will create and open a file for writing with a specified or generated name. If the file is successfully opened, call the deal history saving method and close the file.

```
//+------------------------------------------------------------------+
//| Export deal history                                              |
//+------------------------------------------------------------------+
void CExpertHistory::Export(string exportFileName = "", int commonFlag = FILE_COMMON) {
   // If the file name is not specified, then generate it
   if(exportFileName == "") {
      exportFileName = GetHistoryFileName();
   }

   // Open the file for writing in the desired data folder
   s_file = FileOpen(exportFileName, commonFlag | FILE_WRITE | FILE_CSV | FILE_ANSI, s_sep);

   // If the file is open,
   if(s_file > 0) {
      // Set the deal history
      WriteDealsHistory();

      // Close the file
      FileClose(s_file);
   } else {
      PrintFormat(__FUNCTION__" | ERROR: Can't open file [%s]. Last error: %d",  exportFileName, GetLastError());
   }
}
```

In the _GetHistoryFileName()_ method, the file name is made up of several fragments. First, add the EA name and version to the beginning of the name, if it is specified in the _\_\_VERSION\_\__ constant. Second, add the start and end dates of the deal history. We will determine the start date by the date of the first deal in history by calling the _GetStartDate()_ method. The end date will be determined by the current time, since the history is exported after the test run is completed. In other words, the current time at the moment of calling the history saving method is precisely the test end time. Third, add the values of some pass characteristics to the file name: initial balance, final balance, drawdown and Sharpe ratio.

If the name turns out to be too long, shorten it to the acceptable length and add the .history.csv extension.

```
//+------------------------------------------------------------------+
//| Form the file name                                               |
//+------------------------------------------------------------------+
string CExpertHistory::GetHistoryFileName() {
   // Take the EA name
   string fileName = MQLInfoString(MQL_PROGRAM_NAME);

   // If a version is specified, add it
#ifdef __VERSION__
   fileName += "." + __VERSION__;
#endif

   fileName += " ";

   // Add the history start and end date
   fileName += "[" + TimeToString(GetStartDate(), TIME_DATE);\
   fileName += " - " + TimeToString(TimeCurrent(), TIME_DATE) + "]";

   fileName += " ";

   // Add some statistical characteristics
   fileName += "[" + DoubleToString(TesterStatistics(STAT_INITIAL_DEPOSIT), 0);\
   fileName += ", " + DoubleToString(TesterStatistics(STAT_INITIAL_DEPOSIT) + TesterStatistics(STAT_PROFIT), 0);\
   fileName += ", " + DoubleToString(TesterStatistics(STAT_EQUITY_DD_RELATIVE), 0);\
   fileName += ", " + DoubleToString(TesterStatistics(STAT_SHARPE_RATIO), 2);\
   fileName += "]";

   // If the name is too long, shorten it
   if(StringLen(fileName) > 255 - 13) {
      fileName = StringSubstr(fileName, 0, 255 - 13);
   }

   // Add extension
   fileName += ".history.csv";

   return fileName;
}
```

In the method of writing history to a file, first write the header, which is the row with the names of data columns. Then we select all available history and start iterating through all deals. Get the properties of each deal. If this is a deal opening or balance operation, form an array with the values of all the deal properties and pass it to the _WriteDealsHistoryRow()_ method for writing a single deal.

```
//+------------------------------------------------------------------+
//| Write deal history to file                                       |
//+------------------------------------------------------------------+
void CExpertHistory::WriteDealsHistory() {
   // Write a header with column names
   WriteDealsHistoryRow(s_columnNames);

   // Variables for each deal properties
   uint     total;
   ulong    ticket = 0;
   long     entry;
   double   price;
   double   sl, tp;
   double   profit, commission, fee, swap;
   double   volume;
   datetime time;
   string   symbol;
   long     type, magic;
   string   comment;

   // Take the entire history
   HistorySelect(0, TimeCurrent());
   total = HistoryDealsTotal();

   // For all deals
   for(uint i = 0; i < total; i++) {
      // If the deal is successfully selected,
      if((ticket = HistoryDealGetTicket(i)) > 0) {
         // Get the values of its properties
         time  = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
         type  = HistoryDealGetInteger(ticket, DEAL_TYPE);
         symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
         volume = HistoryDealGetDouble(ticket, DEAL_VOLUME);
         entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
         price = HistoryDealGetDouble(ticket, DEAL_PRICE);
         sl = HistoryDealGetDouble(ticket, DEAL_SL);
         tp = HistoryDealGetDouble(ticket, DEAL_TP);
         profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         fee = HistoryDealGetDouble(ticket, DEAL_FEE);
         swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
         magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
         comment = HistoryDealGetString(ticket, DEAL_COMMENT);

         if(type == DEAL_TYPE_BUY || type == DEAL_TYPE_SELL || type == DEAL_TYPE_BALANCE) {
            // Replace the separator characters in the comment with a space
            StringReplace(comment, s_sep, " ");

            // Form an array of values for writing one deal to the file string
            string fields[] = {TimeToString(time, TIME_DATE | TIME_MINUTES | TIME_SECONDS),
                               IntegerToString(ticket), IntegerToString(type), symbol, DoubleToString(volume), IntegerToString(entry),
                               DoubleToString(price, 5), DoubleToString(sl, 5), DoubleToString(tp, 5), DoubleToString(profit),
                               DoubleToString(commission), DoubleToString(fee), DoubleToString(swap), IntegerToString(magic), comment
                              };

            // Set the values of a single deal to the file
            WriteDealsHistoryRow(fields);
         }
      }
   }
}
```

In the _WriteDealsHistoryRow()_ method, we simply combine all the values from the passed array into one string via the specified separator and write it to the open CSV file. For connection, we used a new macro _JOIN_, which was added to our collection of macros in the _Macros.mqh_ file.

```
//+------------------------------------------------------------------+
//| Write one row of deal history to the file                        |
//+------------------------------------------------------------------+
void CExpertHistory::WriteDealsHistoryRow(const string &fields[]) {
   // Row to be set
   string row = "";

   // Concatenate all array values into one row using a separator
   JOIN(fields, row, ",");

   // Write a row to the file
   FileWrite(s_file, row);
}
```

Save the changes in the _ExpertHistory.mqh_ file in the current folder.

Now we only need to connect the file to the EA file and add calling the _CExpertHistory::Export()_ method to the _OnTester()_ event handler:

```
...

#include "ExpertHistory.mqh"

...

//+------------------------------------------------------------------+
//| Test results                                                     |
//+------------------------------------------------------------------+
double OnTester(void) {
   CExpertHistory::Export();
   return expert.Tester();
}
```

Save the changes in the _SimpleVolumesExpert.mq5_ file in the current folder.

Let's start testing the EA. After the test is complete, a file with the following name appeared in the shared data folder

SimpleVolumesExpert.1.19 \[2021.01.01 - 2022.12.30\] \[10000, 34518, 1294, 3.75\].history.csv

The name reveals that the deal history covers two years (2021 and 2022), the starting account balance is USD 10,000, while the final one is USD 34,518. During the test interval, the maximum relative drawdown by equity was USD 1294, while the Sharpe ratio was 3.75. If we open the resulting file in Excel, we will see the following:

![](https://c.mql5.com/2/115/6329997143012__1.png)

Fig. 4. Result of unloading deal history into a CSV file

The data looks valid. Let's now move on to developing an EA that will be able to reproduce trading on another account using the CSV file.

**Trading replay**

Let's start implementing the new EA by creating a trading strategy. Indeed, following other people's instructions on when and what positions to open can also be called a trading strategy. If the signals source of the signals is trustworthy, then why not use it. Therefore, let's create a new class _CHistoryStrategy_ inheriting it from _CVirtualStrategy_. As for the methods, we will definitely need to implement a constructor, a tick handling method and a method for converting to a string. Although we will not need the last one, its presence is required due to inheritance, since this method is abstract in the parent class.

We only need to add the following properties to the new class:

- _m\_symbols_ — array of symbol names (trading instruments);
- _m\_history_ —  two-dimensional array for reading from the deal history file (N rows \* 15 columns);
- _m\_totalDeals_ — number of deals in history;
- _m\_currentDeal_ —  current deal index;
- _m\_symbolInfo_ — object for obtaining data on the symbol properties.

The initial values for these properties will be set in the constructor.

```
//+------------------------------------------------------------------+
//| Trading strategy for reproducing the history of deals            |
//+------------------------------------------------------------------+
class CHistoryStrategy : public CVirtualStrategy {
protected:
   string            m_symbols[];            // Symbols (trading instruments)
   string            m_history[][15];        // Array of deal history (N rows * 15 columns)
   int               m_totalDeals;           // Number of deals in history
   int               m_currentDeal;          // Current deal index

   CSymbolInfo       m_symbolInfo;           // Object for getting information about the symbol properties

public:
                     CHistoryStrategy(string p_params);        // Constructor
   virtual void      Tick() override;        // OnTick event handler
   virtual string    operator~() override;   // Convert object to string
};
```

The strategy constructor should accept one argument - the initialization string. This requirement also follows from inheritance. The initialization string should pack all the necessary values. The constructor reads them from the string and use them as needed. It just so happens that for this simple strategy we only need to pass one value in the initialization string - the name of the history file. All further data for the strategy will be obtained from the history file. Then the constructor can be implemented the following way:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHistoryStrategy::CHistoryStrategy(string p_params) {
   m_params = p_params;

// Read the file name from the parameters
   string fileName = ReadString(p_params);

// If the name is read, then
   if(IsValid()) {
      // Attempting to open a file in the data folder
      int f = FileOpen(fileName, FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');

      // If failed to open a file, then try to open the file from the shared folder
      if(f == INVALID_HANDLE) {
         f = FileOpen(fileName, FILE_COMMON | FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');
      }

      // If this does not work, report an error and exit
      if(f == INVALID_HANDLE) {
         SetInvalid(__FUNCTION__,
                    StringFormat("ERROR: Can't open file %s from common folder %s, error code: %d",
                                 fileName, TerminalInfoString(TERMINAL_COMMONDATA_PATH), GetLastError()));
         return;
      }

      // Read the file up to the header string (usually it comes first)
      while(!FileIsEnding(f)) {
         string s = FileReadString(f);
         // If we find a header string, read the names of all columns without saving them
         if(s == "DATE") {
            FORI(14, FileReadString(f));
            break;
         }
      }

      // Read the remaining rows until the end of the file
      while(!FileIsEnding(f)) {
         // If the array for storing the read history is filled, increase its size
         if(m_totalDeals == ArraySize(m_history)) {

            ArrayResize(m_history, ArraySize(m_history) + 10000, 100000);
         }

         // Read 15 values from the next file string into the array string
         FORI(15, m_history[m_totalDeals][i] = FileReadString(f));

         // If the deal symbol is not empty,
         if(m_history[m_totalDeals][SYMBOL] != "") {
            // Add it to the symbol array if there is no such symbol there yet
            ADD(m_symbols, m_history[m_totalDeals][SYMBOL]);
         }

         // Increase the counter of read deals
         m_totalDeals++;
      }

      // Close the file
      FileClose(f);

      PrintFormat(__FUNCTION__" | OK: Found %d rows in %s", m_totalDeals, fileName);

      // If there are read deals except for the very first one (account top-up), then
      if(m_totalDeals > 1) {
         // Set the exact size for the history array
         ArrayResize(m_history, m_totalDeals);

         // Current time
         datetime ct = TimeCurrent();

         PrintFormat(__FUNCTION__" |\n"
                     "Start time in tester:  %s\n"
                     "Start time in history: %s",
                     TimeToString(ct, TIME_DATE), m_history[0][DATE]);

         // If the test start date is greater than the history start date, then report an error
         if(StringToTime(m_history[0][DATE]) < ct) {
            SetInvalid(__FUNCTION__,
                       StringFormat("ERROR: For this history file [%s] set start date less than %s",
                                    fileName, m_history[0][DATE]));
         }
      }

      // Create virtual positions for each symbol
      CVirtualReceiver::Get(GetPointer(this), m_orders, ArraySize(m_symbols));

      // Register the event handler for a new bar on the minimum timeframe
      FOREACH(m_symbols, IsNewBar(m_symbols[i], PERIOD_M1));
   }
}
```

In the constructor, we read the file name from the initialization string and try to open it. If the file was successfully opened from a local or shared data folder, then we read its contents, filling the _m\_history_ array with it. As we read, we also fill the _m\_symbols_ array of symbol names: as soon as a new name is encountered, we immediately add it to the array. This is done by the _ADD()_ macro.

Along the way, we count the number of read deal entries in the _m\_totalDeals_ property using it as the index of the first dimension of the _m\_history_ array, which should be used to record information about the next deal. After all the contents of the file have been read, we close it.

Next we check if the test start date is greater than the history start date. We cannot allow such a situation, since in this case it will not be possible to model some of the deals from the beginning of the history. This may well lead to distorted trading results during the test. Therefore, we allow the constructor to create a valid object only if the deal history starts no earlier than the test start date.

The key point in the constructor is the allocation of virtual positions strictly according to the number of different symbol names encountered in history. Since the objective of the strategy is to provide the required open volume of positions for each symbol, this can be done using only a single virtual position per symbol.

The tick handling method will only work with the array of read deals. Since we can open/close several symbols at once at one moment in time, we arrange a loop that handles all rows from the history of deals whose time is not greater than the current one. The remaining deal entries are handled on the following ticks, when the current time increases and new deals appear whose time has already arrived.

If at least one deal is found that needs to be handled, we first find its symbol and its index in the m\_symbols array. Using this index, we will determine, which virtual position from the _m\_orders_ array is responsible for this symbol. If the index is not found for some reason (this should not happen yet if everything is working correctly), then we will simply skip the deal. We will also skip deals that reflect balance deals on the account.

Now the most interesting part begins. We need to handle the read deal. There are two possible cases here: there is no open virtual position for this symbol, or a virtual position is open.

In the former case, everything is simple: we open a position in the direction of the deal with the appropriate volume. In the second case, we may need to either increase the volume of the current position for a given symbol or decrease it. Moreover, it may be necessary to reduce it so much that the direction of the open position changes.

To simplify the calculations, we will do the following:

- Convert the volume of the new deal into the "signed" format. That is, if it was in the SELL direction, then we will make its volume negative.
- We will get the volume of the open deal for the same symbol as in the new one. The _CVirtualOrder::Volume()_ method immediately returns the volume in signed format.
- Add the volume of the already open position to the volume of the new one. Get a new volume that should remain open after taking the new deal into account. This volume will also be in the "signed" format.
- Close the open virtual position.
- If the new volume is not equal to zero, open a new virtual position for the symbol. We determine its direction by the sign of the new volume (positive - BUY, negative - SELL). The modulus of the new volume is passed to the virtual position opening method as a volume.

After this procedure, increase the counter of handled deals from the history and move on to the next loop iteration. If at this point in time there are no more deals to handle or the deals in the history have ended, then the tick handling is complete.

```
//+------------------------------------------------------------------+
//| OnTick event handler                                             |
//+------------------------------------------------------------------+
void CHistoryStrategy::Tick() override {
//---
   while(m_currentDeal < m_totalDeals && StringToTime(m_history[m_currentDeal][DATE]) <= TimeCurrent()) {
      // Deal symbol
      string symbol = m_history[m_currentDeal][SYMBOL];

      // Find the index of the current deal symbol in the array of symbols
      int index;
      FIND(m_symbols, symbol, index);

      // If not found, then skip the current deal
      if(index == -1) {
         m_currentDeal++;
         continue;
      }

      // Deal type
      ENUM_DEAL_TYPE type = (ENUM_DEAL_TYPE) StringToInteger(m_history[m_currentDeal][TYPE]);

      // Current deal volume
      double volume = NormalizeDouble(StringToDouble(m_history[m_currentDeal][VOLUME]), 2);

      // If this is a top-up/withdrawal, skip the deal
      if(volume == 0) {
         m_currentDeal++;
         continue;
      }

      // Report information about the read deal
      PrintFormat(__FUNCTION__" | Process deal #%d: %s %.2f %s",
                  m_currentDeal, (type == DEAL_TYPE_BUY ? "BUY" : (type == DEAL_TYPE_SELL ? "SELL" : EnumToString(type))),
                  volume, symbol);

      // If this is a sell deal, then make the volume negative
      if(type == DEAL_TYPE_SELL) {
         volume *= -1;
      }

      // If the virtual position for the current deal symbol is open,
      if(m_orders[index].IsOpen()) {
         // Add its volume to the volume of the current trade
         volume += m_orders[index].Volume();

         // Close the virtual position
         m_orders[index].Close();
      }

      // If the volume for the current symbol is not 0,
      if(MathAbs(volume) > 0.00001) {
         // Open a virtual position of the required volume and direction
         m_orders[index].Open(symbol, (volume > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL), MathAbs(volume));
      }

      // Increase the counter of handled deals
      m_currentDeal++;
   }
}
```

Save the obtained code in the _HistoryStrategy.mqh_ file of the current folder.

Now let's create an EA file based on the existing _SimpleVolumesExpert.mq5_. To get the desired result, we need to add an input to the EA, in which we can specify the name of the file with the history.

```
input group "::: Testing the deal history"
input string historyFileName_    = "";    // File with history
```

The part of the code responsible for loading strategy initialization strings from the database is no longer needed, so we remove it.

We need to set the creation of a single instance of the _CHistoryStrategy_ class strategy in the initialization string. The strategy receives the file name with history as an argument:

```
// Prepare the initialization string for an EA with a group of several strategies
   string expertParams = StringFormat(
                            "class CVirtualAdvisor(\n"
                            "    class CVirtualStrategyGroup(\n"
                            "       [\n"\
                            "        class CHistoryStrategy(\"%s\")\n"\
                            "       ],%f\n"
                            "    ),\n"
                            "    class CVirtualRiskManager(\n"
                            "       %d,%.2f,%d,%.2f,%.2f,%d,%.2f,%.2f,%d,%.2f,%d,%.2f,%.2f"
                            "    )\n"
                            "    ,%d,%s,%d\n"
                            ")",
                            historyFileName_, scale_,
                            rmIsActive_, rmStartBaseBalance_,
                            rmCalcDailyLossLimit_, rmMaxDailyLossLimit_, rmCloseDailyPart_,
                            rmCalcOverallLossLimit_, rmMaxOverallLossLimit_, rmCloseOverallPart_,
                            rmCalcOverallProfitLimit_, rmMaxOverallProfitLimit_, rmMaxOverallProfitDate_,
                            rmMaxRestoreTime_, rmLastVirtualProfitFactor_,
                            magic_, "HistoryReceiver", useOnlyNewBars_
                         );
```

This completes the changes to the EA file. Save it as _HistoryReceiverExpert.mq5_ in the current folder.

Now we have a working EA that can reproduce the history of deals. In fact, its capabilities are somewhat broader. We can easily see what the trading results will look like when increasing the volume of opened positions with an increase in the account balance, despite the fact that the deals in the history are set based on trading with a fixed balance. We can apply different risk manager parameters to evaluate its impact on trading, despite the fact that the deal history was set with different risk manager parameters (or even with the risk manager disabled). After passing through the tester, the deal history is automatically saved to a new file.

But if we do not need all these additional features yet, do not want to use the risk manager and do not like the bunch of unused inputs associated with it, then we can create a new EA class that will not have additional features. In this class, we can also get rid of saving the status and of the interface for drawing virtual positions on charts, as well as of other things that are not used much yet.

Implementing such a class might look something like this:

```
//+------------------------------------------------------------------+
//| Trade history replay EA class                                    |
//+------------------------------------------------------------------+
class CVirtualHistoryAdvisor : public CAdvisor {
protected:
   CVirtualReceiver *m_receiver;       // Receiver object that brings positions to the market
   bool              m_useOnlyNewBar;  // Handle only new bar ticks
   datetime          m_fromDate;       // Test start time

public:
   CVirtualHistoryAdvisor(string p_param);   // Constructor
   ~CVirtualHistoryAdvisor();                // Destructor

   virtual void      Tick() override;        // OnTick event handler
   virtual double    Tester() override;      // OnTester event handler

   virtual string    operator~() override;   // Convert object to string
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CVirtualHistoryAdvisor::CVirtualHistoryAdvisor(string p_params) {
// Save the initialization string
   m_params = p_params;

// Read the file name from the initialization string
   string fileName = ReadString(p_params);

// Read the work flag only at the bar opening
   m_useOnlyNewBar = (bool) ReadLong(p_params);

// If there are no read errors,
   if(IsValid()) {
      if(!MQLInfoInteger(MQL_TESTER)) {
         // Otherwise, set the object state to invalid
         SetInvalid(__FUNCTION__, "ERROR: This expert can run only in tester");
         return;
      }

      if(fileName == "") {
         // Otherwise, set the object state to invalid
         SetInvalid(__FUNCTION__, "ERROR: Set file name with deals history in ");
         return;
      }

      string strategyParams = StringFormat("class CHistoryStrategy(\"%s\")", fileName);

      CREATE(CHistoryStrategy, strategy, strategyParams);

      Add(strategy);

      // Initialize the receiver with the static receiver
      m_receiver = CVirtualReceiver::Instance(65677);

      // Save the work (test) start time
      m_fromDate = TimeCurrent();
   }
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
void CVirtualHistoryAdvisor::~CVirtualHistoryAdvisor() {
   if(!!m_receiver)     delete m_receiver;      // Remove the recipient
   DestroyNewBar();           // Remove the new bar tracking objects
}

//+------------------------------------------------------------------+
//| OnTick event handler                                             |
//+------------------------------------------------------------------+
void CVirtualHistoryAdvisor::Tick(void) {
// Define a new bar for all required symbols and timeframes
   bool isNewBar = UpdateNewBar();

// If there is no new bar anywhere, and we only work on new bars, then exit
   if(!isNewBar && m_useOnlyNewBar) {
      return;
   }

// Start handling in strategies
   CAdvisor::Tick();

// Receiver handles virtual positions
   m_receiver.Tick();

// Adjusting market volumes
   m_receiver.Correct();
}

//+------------------------------------------------------------------+
//| OnTester event handler                                           |
//+------------------------------------------------------------------+
double CVirtualHistoryAdvisor::Tester() {
// Maximum absolute drawdown
   double balanceDrawdown = TesterStatistics(STAT_EQUITY_DD);

// Profit
   double profit = TesterStatistics(STAT_PROFIT);

// Fixed balance for trading from settings
   double fixedBalance = CMoney::FixedBalance();

// The ratio of possible increase in position sizes for the drawdown of 10% of fixedBalance_
   double coeff = fixedBalance * 0.1 / MathMax(1, balanceDrawdown);

// Calculate the profit in annual terms
   long totalSeconds = TimeCurrent() - m_fromDate;
   double totalYears = totalSeconds / (365.0 * 24 * 3600);
   double fittedProfit = profit * coeff / totalYears;

// If it is not specified, then take the initial balance (although this will give a distorted result)
   if(fixedBalance < 1) {
      fixedBalance = TesterStatistics(STAT_INITIAL_DEPOSIT);
      balanceDrawdown = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
      coeff = 0.1 / balanceDrawdown;
      fittedProfit = fixedBalance * MathPow(1 + profit * coeff / fixedBalance, 1 / totalYears);
   }

   return fittedProfit;
}

//+------------------------------------------------------------------+
//| Convert an object to a string                                    |
//+------------------------------------------------------------------+
string CVirtualHistoryAdvisor::operator~() {
   return StringFormat("%s(%s)", typename(this), m_params);
}
//+------------------------------------------------------------------+
```

The EA of this class will accept only two parameters in the initialization string: the name of the history file and the flag of working only at the opening of a minute bar. Save this code in the _VirtualHistoryAdvisor.mqh_ file of the current folder.

The EA file that uses this class can also be shortened somewhat compared to the previous version:

```
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "::: Testing the deal history"
input string historyFileName_    = "";    // File with history
input group "::: Money management"
sinput double fixedBalance_      = 10000; // - Used deposit (0 - use all) in the account currency
input  double scale_             = 1.00;  // - Group scaling multiplier

input group "::: Other parameters"
input bool     useOnlyNewBars_   = true;  // - Work only at bar opening

datetime fromDate = TimeCurrent();        // Operation start time

CVirtualHistoryAdvisor     *expert;       // EA object

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
// Set parameters in the money management class
   CMoney::DepoPart(scale_);
   CMoney::FixedBalance(fixedBalance_);

// Prepare the initialization string for the deal history replay EA
   string expertParams = StringFormat(
                            "class CVirtualHistoryAdvisor(\"%s\",%f,%d)",
                            historyFileName_, useOnlyNewBars_
                         );

// Create an EA handling virtual positions
   expert = NEW(expertParams);

// If the EA is not created, then return an error
   if(!expert) return INIT_FAILED;

// Successful initialization
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   expert.Tick();
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if(!!expert) delete expert;
}

//+------------------------------------------------------------------+
//| Test results                                                     |
//+------------------------------------------------------------------+
double OnTester(void) {
   return expert.Tester();
}
//+------------------------------------------------------------------+
```

Save this code in the _SimpleHistoryReceiverExpert.mq5_ file of the current folder.

### Test results

Let's launch one of the created EAs specifying the correct name of the file with the saved deal history. First, let's launch it on the same quotes server that was used to get the history (MetaQuotes-Demo). The obtained test results perfectly match the original results! I must admit, this is even a somewhat unexpectedly good result, indicating the correct implementation of the plan.

Now let's see what happens when we run the EA on another server:

![](https://c.mql5.com/2/115/217184439914__1.png)

![](https://c.mql5.com/2/115/5921277156970__1.png)

Fig. 5. Results of reproducing the history of deals on quotes of another broker's real server

The balance curve chart is almost indistinguishable from the chart for the initial trading results on MetaQuotes-Demo. However, the numerical values are slightly different. Let's look at the original values again for comparison:

![](https://c.mql5.com/2/115/3713696569130__3.png)

Fig. 6. Initial test results on MetaQuotes-Demo server quotes

We see a slight decrease in total and normalized average annual profit and the Sharpe ratio, as well as a slight increase in drawdown. However, these results are not comparable to the loss of the entire deposit we initially saw when running the EA on the quotes of another broker's real server. This is very encouraging and opens up a new layer of tasks we may have to solve while preparing the EA for real trading.

### Conclusion

It is time for some interim conclusions. We were able to show that for a particular trading strategy used, changing the quote server can have very dire consequences. But having understood the reasons for such behavior, we were able to show that if we leave the logic of signals for opening positions on the server with the original quotes, and pass only the operations of opening and closing positions to the new server, then the trading results again become comparable.

To do this, we have developed two new tools allowing us to save the history of deals after the tester pass and then play back the deals based on the saved history. But these tools can only be used in the tester. In real trading, they are meaningless. Now we can start the implementation of such a division of responsibilities between EAs for real trading as well, since the test results confirm the validity of using such an approach.

We will need to split the EA into two separate ones. The first one will make decisions about opening positions and open them, while working on the quotes server that seems most convenient to us. At the same time, it will have to ensure that the list of open positions is broadcast in a form that the second EA can accept. The second EA will work in another terminal, connected to another quotes server if necessary. It will constantly maintain the volume of open positions corresponding to the values broadcast by the first EA. This will help bypass the limitation we identified at the beginning of this article.

We can go even further. The mentioned work layout implies that both terminals should work on one computer. But this is not necessary. The terminals can work on different computers. The main thing is that the first EA can pass information about positions to the second EA via certain channels. Clearly, this will not allow successful operation of trading strategies, for which it is very critical to adhere to the exact position opening time and price. But we were initially focused on using other strategies, for which high precision of entries is not required. Therefore, communication channel delays should not become an obstacle when arranging such a work layout.

But let's not get too ahead of ourselves. We will continue our systematic movement in the chosen direction in the following articles.

Thank you for your attention! See you soon!

### Archive contents

| # | Name | Version | Description | Recent changes |
| --- | --- | --- | --- | --- |
|  | MQL5/Experts/Article.15330 |
| --- | --- |
| 1 | Advisor.mqh | 1.04 | EA base class | [Part 10](https://www.mql5.com/en/articles/14739) |
| --- | --- | --- | --- | --- |
| 2 | Database.mqh | 1.03 | Class for handling the database | [Part 13](https://www.mql5.com/en/articles/14982) |
| --- | --- | --- | --- | --- |
| 3 | ExpertHistory.mqh | 1.00 | Class for exporting trade history to file | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 4 | Factorable.mqh | 1.01 | Base class of objects created from a string | [Part 10](https://www.mql5.com/en/articles/14739) |
| --- | --- | --- | --- | --- |
| 5 | HistoryReceiverExpert.mq5 | 1.00 | EA for replaying the history of deals with the risk manager | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 6 | HistoryStrategy.mqh | 1.00 | Class of the trading strategy for replaying the history of deals | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 7 | Interface.mqh | 1.00 | Basic class for visualizing various objects | [Part 4](https://www.mql5.com/en/articles/14246) |
| --- | --- | --- | --- | --- |
| 8 | Macros.mqh | 1.02 | Useful macros for array operations | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 9 | Money.mqh | 1.01 | Basic money management class | [Part 12](https://www.mql5.com/en/articles/14764) |
| --- | --- | --- | --- | --- |
| 10 | NewBarEvent.mqh | 1.00 | Class for defining a new bar for a specific symbol | [Part 8](https://www.mql5.com/en/articles/14574) |
| --- | --- | --- | --- | --- |
| 11 | Receiver.mqh | 1.04 | Base class for converting open volumes into market positions | [Part 12](https://www.mql5.com/en/articles/14764) |
| --- | --- | --- | --- | --- |
| 12 | SimpleHistoryReceiverExpert.mq5 | 1.00 | Simplified EA for replaying the history of deals | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 13 | SimpleVolumesExpert.mq5 | 1.19 | EA for parallel operation of several groups of model strategies. Parameters should be loaded from the optimization database. | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 14 | SimpleVolumesStrategy.mqh | 1.09 | Class of trading strategy using tick volumes | [Part 15](https://www.mql5.com/en/articles/15294) |
| --- | --- | --- | --- | --- |
| 15 | Strategy.mqh | 1.04 | Trading strategy base class | [Part 10](https://www.mql5.com/en/articles/14739) |
| --- | --- | --- | --- | --- |
| 16 | TesterHandler.mqh | 1.02 | Optimization event handling class | [Part 13](https://www.mql5.com/en/articles/14982) |
| --- | --- | --- | --- | --- |
| 17 | VirtualAdvisor.mqh | 1.06 | Class of the EA handling virtual positions (orders) | [Part 15](https://www.mql5.com/en/articles/15294) |
| --- | --- | --- | --- | --- |
| 18 | VirtualChartOrder.mqh | 1.00 | Graphical virtual position class | [Part 4](https://www.mql5.com/en/articles/14246) |
| --- | --- | --- | --- | --- |
| 19 | VirtualFactory.mqh | 1.04 | Object factory class | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 20 | VirtualHistoryAdvisor.mqh | 1.00 | Trade history replay EA class | [Part 16](https://www.mql5.com/en/articles/15330) |
| --- | --- | --- | --- | --- |
| 21 | VirtualInterface.mqh | 1.00 | EA GUI class | [Part 4](https://www.mql5.com/en/articles/14246) |
| --- | --- | --- | --- | --- |
| 22 | VirtualOrder.mqh | 1.04 | Class of virtual orders and positions | [Part 8](https://www.mql5.com/en/articles/14574) |
| --- | --- | --- | --- | --- |
| 23 | VirtualReceiver.mqh | 1.03 | Class for converting open volumes to market positions (receiver) | [Part 12](https://www.mql5.com/en/articles/14764) |
| --- | --- | --- | --- | --- |
| 24 | VirtualRiskManager.mqh | 1.02 | Risk management class (risk manager) | [Part 15](https://www.mql5.com/en/articles/15294) |
| --- | --- | --- | --- | --- |
| 25 | VirtualStrategy.mqh | 1.05 | Class of a trading strategy with virtual positions | [Part 15](https://www.mql5.com/en/articles/15294) |
| --- | --- | --- | --- | --- |
| 26 | VirtualStrategyGroup.mqh | 1.00 | Class of trading strategies group(s) | [Part 11](https://www.mql5.com/en/articles/14741) |
| --- | --- | --- | --- | --- |
| 27 | VirtualSymbolReceiver.mqh | 1.00 | Symbol receiver class | [Part 3](https://www.mql5.com/en/articles/14148) |
| --- | --- | --- | --- | --- |
|  | MQL5/Files |
| --- | --- |
| 1 | SimpleVolumesExpert.1.19 \[2021.01.01 - 2022.12.30\] \[10000, 34518, 1294, 3.75\].history.csv |  | History of SimpleVolumesExpert.mq5 EA deals obtained after the export. It can bse used to replay the deals in the tester using SimpleHistoryReceiverExpert.mq5 or HistoryReceiverExpert.mq5 EAs |  |
| --- | --- | --- | --- | --- |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15330](https://www.mql5.com/ru/articles/15330)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15330.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/15330/mql5.zip "Download MQL5.zip")(163.78 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/480613)**
(4)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
31 Jul 2024 at 13:24

That's exactly what Signal Service does.


![Amir Jafary](https://c.mql5.com/avatar/2024/11/674AFC80-99F4.png)

**[Amir Jafary](https://www.mql5.com/en/users/eyas1370)**
\|
30 Jan 2025 at 14:00

i download last files how i can run the advisor in my meta! i cant compile advisor file and have error


![Cristian-bogdan Buzatu](https://c.mql5.com/avatar/avatar_na2.png)

**[Cristian-bogdan Buzatu](https://www.mql5.com/en/users/buza20)**
\|
3 Feb 2025 at 23:13

I got this error when I tried to run a backtest on the EA:

2025.02.04 01:11:13.690Core 012021.01.01 00:00:00   database error, no such table: passes

2025.02.04 01:11:13.690Core 01 [tester stopped](https://www.mql5.com/en/docs/common/TesterStop "MQL5 Documentation: TesterStop function") because OnInit returns non-zero code 1

Any help please?

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
4 Feb 2025 at 11:52

**Cristian-bogdan Buzatu [#](https://www.mql5.com/ru/forum/470758#comment_55813684):**

I got this error when I tried to run a backtest on EA

Most likely the reason is that you have not created a database and have not performed the first two steps of optimisation, which will fill the database with information about the performed passes [(part 9](https://www.mql5.com/ru/articles/14680), [part 11](https://www.mql5.com/ru/articles/14741), [part 13](https://www.mql5.com/ru/articles/14892)). Unfortunately, at the time of writing this article, there is not yet a simple tool to create a database, create an optimisation project and export its results to the final EA. We revisited this issue in [part 21](https://www.mql5.com/ru/articles/16373), but did not finish addressing it. It will be continued in parts 22 and 23 (not ready for publication yet).

![Price Action Analysis Toolkit Development (Part 10): External Flow (II) VWAP](https://c.mql5.com/2/115/Price_Action_Analysis_Toolkit_Development_Part_10____LOGO.png)[Price Action Analysis Toolkit Development (Part 10): External Flow (II) VWAP](https://www.mql5.com/en/articles/16984)

Master the power of VWAP with our comprehensive guide! Learn how to integrate VWAP analysis into your trading strategy using MQL5 and Python. Maximize your market insights and improve your trading decisions today.

![Automating Trading Strategies in MQL5 (Part 4): Building a Multi-Level Zone Recovery System](https://c.mql5.com/2/115/Automating_Trading_Strategies_in_MQL5_Part_4__LOGO.png)[Automating Trading Strategies in MQL5 (Part 4): Building a Multi-Level Zone Recovery System](https://www.mql5.com/en/articles/17001)

In this article, we develop a Multi-Level Zone Recovery System in MQL5 that utilizes RSI to generate trading signals. Each signal instance is dynamically added to an array structure, allowing the system to manage multiple signals simultaneously within the Zone Recovery logic. Through this approach, we demonstrate how to handle complex trade management scenarios effectively while maintaining a scalable and robust code design.

![Data Science and ML (Part 33): Pandas Dataframe in MQL5, Data Collection for ML Usage made easier](https://c.mql5.com/2/115/Data_Science_and_ML_Part_33___LOGO.png)[Data Science and ML (Part 33): Pandas Dataframe in MQL5, Data Collection for ML Usage made easier](https://www.mql5.com/en/articles/17030)

When working with machine learning models, it’s essential to ensure consistency in the data used for training, validation, and testing. In this article, we will create our own version of the Pandas library in MQL5 to ensure a unified approach for handling machine learning data, for ensuring the same data is applied inside and outside MQL5, where most of the training occurs.

![Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs(IV) — Test Trading Strategy](https://c.mql5.com/2/114/Integrate_Your_Own_LLM_into_EA__Part_5_____IV___LOGO.png)[Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs(IV) — Test Trading Strategy](https://www.mql5.com/en/articles/13506)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/15330&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049152094553875948)

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