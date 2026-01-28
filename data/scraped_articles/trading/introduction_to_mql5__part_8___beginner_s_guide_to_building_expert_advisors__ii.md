---
title: Introduction to MQL5 (Part 8): Beginner's Guide to Building Expert Advisors (II)
url: https://www.mql5.com/en/articles/15299
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:28:53.558256
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=lughnklqusabvqqpxgvoprdzuciaxzaz&ssn=1769092132409166342&ssn_dr=0&ssn_sr=0&fv_date=1769092132&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15299&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%208)%3A%20Beginner%27s%20Guide%20to%20Building%20Expert%20Advisors%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909213238863704&fz_uniq=5049162243561596451&sv=2552)

MetaTrader 5 / Trading


### Introduction

Having already studied the fundamentals of MQL5, you are now prepared to take on one of the most important tasks associated with algorithmic trading: creating a working Expert Advisor. As I indicated in the previous article, we will use a project-based approach for this series. This method helps in both comprehending abstract ideas and recognizing how they are used in practical situations. You will have a firm grasp of how to automate trading decisions based on candlestick patterns and predetermined conditions by the time you finish this guide.

A lot of beginners frequently post questions on the forum, and although the amazing community members respond with great advice, some self-taught users find it difficult to incorporate these answers into their programs. While the answer to a specific question is provided, it is frequently impractical to provide a detailed explanation of the entire code. Even though the code snippets and tips can be helpful, as a self-taught novice, you might still find it difficult to put everything together. With a project-based approach, this article seeks to answer these frequently asked questions and guarantees that the solutions can be applied by any EA.

In this article, we will focus on developing an EA that uses the candlestick analysis from the previous day to determine its trading direction. The EA will concentrate on selling for the day if the most recent daily candlestick is bearish and buying if it is bullish. The EA will also verify its trading signals by utilizing the close price of the day's first 1-hour candlestick. There won't be more than one open position at any given time, and the daily maximum of two trades will be enforced. It will function under stringent trade limits. Furthermore, its operation will be limited to the designated trading hours of Monday to Wednesday.

Throughout this project, we will address several common questions that beginners often have, especially those frequently posed on MQL5 forums. Some of these questions include:

- How to buy and sell in MQL5?

- How to get the open and close prices of candlesticks?

- How to avoid trading on every tick?

- How can I limit an EA to only entering one trade at a time?

- How to set the trading period for an EA?

- How to specify the days of the week an EA can trade?

- How to set profit and loss limits for trades?


By adopting this project-based approach, I aim to provide clear, practical answers to these questions, enabling you to implement the solutions directly into your Expert Advisor. This hands-on method not only helps in understanding theoretical concepts but also in seeing their application in real-world scenarios. Let's go into the world of MQL5 and start building your trading tools!

### **1\. Setting up the Project**

**1.1. Pseudocode**

It's crucial to use pseudocode to describe our Expert Advisor's logic before we start writing code. The significance of pseudocode was covered in the previous article, emphasizing how it facilitates clear planning and organization of your ideas, which speeds up the actual coding process. To learn MQL5, keep in mind that project-based learning is preferable to studying everything at once. You get better the more projects you work on. This is our EA's basic pseudocode:

**1\. Initialize the EA:**

- Set the magic number for trade identification.
- Define start and end times for trading.
- Initialize variables for storing prices.

**2\. On Every Tick:**

- Make sure the time is within trading hours by checking the current one.
- Get the open and close prices from the previous day.
- Make a sell trade if the close of the prior day was lower than the open (bearish).
- Make a buy trade if the close of the prior day was higher than the open (bullish).
- Ensure only one trade is open at a time.
- Only trade from Monday to Thursday.
- Ensure a maximum of one open position
- Daily limits of two trades per day.
- Daily profit limit.
- Close trades at the end of the trading period.

**1.2. Importing Required Libraries**

Libraries are sets of pre-written classes and functions in MQL5 that make coding easier. They enable you to concentrate on the distinctive features of your Expert Advisor (EA) rather than having to rewrite common functionalities. In one of our earlier articles, we discussed the idea of including files. Don't worry if you don't fully understand Include Files; you can still follow this article. You don't have to know everything at once to use MQL5; learning through project-based learning allows you to advance gradually. You can use the code snippets and explanations in your programs with effectiveness if you just follow the instructions.

**Analogy**

Consider the work you do as a coder as a vast, enchanted bookshelf. This bookshelf has numerous books. Every book contains unique guidelines and narratives that assist you in achieving various goals. Similar to these rare books on your enchanted bookshelf are MQL5 libraries. The pre-written stories (or code) in these books can be used to build your trading robot. You can just pick up the appropriate book and use the stories within to guide you, saving you the trouble of having to start from scratch every time you write.

So you do not need to write every single detail yourself when developing your Expert Advisor. These particular books contain instructions that you can use to simplify your coding. This will allow you to concentrate on the special and interesting aspects of your trading robot rather than getting bogged down in the mundane details.

**1.2.1. Include the Trade Library**

You must incorporate the Trade library into your EA to manage trading operations. A collection of classes and functions for effective trade management are offered by this library.

**Analogy**

Imagine owning an expansive bookcase brimming with volumes. Every book functions as a kind of toolkit, offering assistance in various areas. To build something unique, such as an awesome trading robot, you pull the appropriate book off the shelf. It is as though a special book has been taken off the shelf to include the Trade Library. You will find all the necessary tools in this book to manage trading operations. This book has already written all the instructions, so you don't have to do it yourself!

You include this special book in your project with the following line of code:

```
#include <Trade/Trade.mqh> // Include the trade library for trading functions
```

Your trading robot will learn from this line to "go grab the book called Trade.mqh from the shelf." This book contains all sorts of useful tools that streamline the process of opening, modifying, and closing trades. It ensures everything runs smoothly and saves you a ton of time.

**Example:**

```
#include <Trade/Trade.mqh> // Include the trade library for trading functions

// Create an instance of the CTrade class for trading operations
CTrade trade;
//magic number
int MagicNumber = 103432;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
// Set the magic number for the EA's trades
   trade.SetExpertMagicNumber(MagicNumber);

// Return initialization success
   return(INIT_SUCCEEDED);
  }
```

**1.2.2.1. Create an Instance of the CTrade Class**

The CTrade class, part of the Trade library, encapsulates methods for executing trading operations. Creating an instance of this class allows you to use its functions to manage your trades.

**Analogy**

Imagine your bookshelf again. You've already picked out the special book (Trade.mqh) that has all the instructions for trading. Now, think of the CTrade class as a special character in this book, like a superhero who knows everything about trading. To use this superhero, you need to bring him into your story. You do this by creating an instance of the CTrade class. It's like inviting the superhero to help you with your project.

Here's how you invite the superhero with a simple line of code:

```
CTrade trade;
```

In this code, CTrade is the superhero, and trade is your invitation. By creating this instance called trade, you're saying, "Superhero CTrade, please join my project and help me manage trades!"

Now, whenever you need to place or manage a trade, you can ask your superhero trade to do it for you. This makes your job much easier and ensures everything is done correctly.

**1.2.2.2. Setting Up the Magic Number**

A magic number is a unique identifier assigned to each trade placed by your EA. This identifier helps you track and manage trades, ensuring that your EA can distinguish its own trades from those placed manually or by other EAs. Think of each book on your shelf as a trade. Sometimes, you might have several books (trades) that look similar, but each one has a special sticker on it with a unique number. This sticker helps you quickly find and manage your books without getting confused.

This unique sticker is known to your Expert Advisor (EA) as a magic number. It is a track-able and manageable way to identify each trade your EA places, ensuring it can differentiate its trades from those placed manually or by other EAs.

**Example:**

```
int MagicNumber = 103432;
```

In the context of the CTrade class, you set the magic number as follows:

```
trade.SetExpertMagicNumber(MagicNumber);
```

This line ensures that every trade placed by the trade instance will have the specified magic number, making it easy to identify and manage.

To use pre-written code that streamlines complex tasks, you must import the necessary libraries into MQL5. Libraries are similar to toolkits that are stocked with time and effort-saving items. Using these libraries will allow you to concentrate on the distinctive features of your Expert Advisor (EA) rather than having to write all the code from scratch.

### **2\. Retrieving and Analyzing Candlestick Data**

It’s essential to retrieve and analyze candlesticks' open and close prices. This information helps us understand market trends and decide whether to buy or sell. We need to access candlesticks' open and close prices to get a clear picture of market movements. Using functions like CopyOpen and CopyClose, we can efficiently fetch this data. With this, we can identify whether a candlestick is bullish or bearish by comparing the open and close prices.

**2.1. CopyClose and CopyOpen Functions in MQL5**

To get the close and open prices of candlesticks for a given symbol and timeframe, utilize MQL5's CopyClose and CopyOpen functions. These functions have multiple ways to copy data based on your needs.

**2.1.1. Call by the First Position and Number of Required Elements**

**Analogy**

Consider the bookshelf as your source of price data, where each book serves as a candlestick representing the open and closed prices. You would count the books from the left and select the starting point when you wanted to begin copying books from a specific spot on the shelf. You can begin, for example, with the third book if that is where you would like to start.

Next, you select the quantity of books you wish to take if you wish to copy a specific number of books beginning at that position. For instance, you will end up with a total of five books in hand if you decide to take five books, beginning with the third book. Similar to choosing a starting book on the shelf and determining how many books to take from that point, this method uses the CopyClose function to allow you to specify a starting position in the price data and the number of elements (candlesticks) you want to copy.

**Syntax:**

```
CopyClose(symbol_name, timeframe,  start_pos, count, close_array);
CopyOpen(symbol_name, timeframe,  start_pos, count, open_array);
```

**Example:**

```
double close_prices[];
double open_prices[];
CopyClose(_Symbol, PERIOD_D1, 2, 5, close_prices); // Copy the close prices of 5 daily candlesticks starting from the 3rd candlestick
CopyOpen(_Symbol, PERIOD_D1, 2, 5, open_prices); // Copy the open prices of 5 daily candlesticks starting from the 3rd candlestick
```

The price data for a particular symbol and timeframe is represented by the bookshelf in this analogy, which is symbolized by the symbol\_name and timeframe. The decision to take five books from the starting position (count) is equivalent to selecting the third book on the shelf. The starting position (start\_pos) is similar to that decision. Like holding the chosen books in your hands, you store the copied data in the target array (close\_array).

**2.1.2. Call by the Start Date and Number of Required Elements**

**Analogy**

Consider your bookshelf as your price data, where a single book corresponds to a day's worth of candlestick readings. To begin copying close prices from a given date, locate the book on the shelf that corresponds to that date. The date "June 1st," for instance, is where you should begin if you wish to begin at that point.

Next, you select the quantity of books you wish to take if you wish to duplicate a specific number of close prices beginning on that date. If you were to take, for example, the close prices of five books beginning with "June 1st," you would obtain the close prices of those five books. This method, which is similar to choosing a starting book on the shelf and determining how many books to take from that point, uses the CopyClose function to allow you to specify a starting date in the price data and the number of elements (candlesticks) you want to copy.

Likewise, you would use the CopyOpen function in the same way to get the open prices. You select the book that corresponds to a given date and choose the quantity of books you wish to take in order to obtain the open prices.

**Syntax:**

```
CopyClose(symbol_name, timeframe,  timeframe, count, close_array[]);
CopyOpen(symbol_name, timeframe,  timeframe, count, open_array[]);
```

**Example:**

```
close_prices[];
double open_prices[];
datetime start_date = D'2023.06.01 00:00';  // Starting from June 1st, 2023
// Copy the close prices of 5 daily candlesticks starting from June 1st, 2023
CopyClose(_Symbol, PERIOD_D1, start_date, 5, close_prices);
// Copy the open prices of 5 daily candlesticks starting from June 1st, 2023 CopyOpen(_Symbol, PERIOD_D1, start_date, 5, open_prices);
```

The price data for a particular symbol and timeframe is represented by the bookshelf in this analogy, which is symbolized by the symbol\_name and timeframe. Selecting the book on the shelf that corresponds to "June 1st" is similar to the starting date (start\_time), and selecting five books from that starting date is analogous to the number of elements (count). Like holding the selected books in your hands, you store the copied data in the target arrays (close\_array and open\_array).

By following this analogy, you will be able to see how the CopyClose and CopyOpen functions facilitate the organized and intuitive retrieval of specific data from your price data bookshelf based on a starting date.

**2.1.3. Call by the Start and End Dates of a Required Time Interval**

**Analogy**

Consider that every book on your bookshelf is a candlestick that shows the open and close prices for a specific period of time. You would look for books that cover the full-time period if you wanted to retrieve close prices for a particular time frame.

As an illustration, let's say you wish to duplicate the close prices for June 1st through June 5th. You would locate the book that corresponds to your beginning point on June 1st and your ending point on June 5th. The close prices for that time period can be obtained by choosing every book published between these two dates.

Similar to picking out a beginning and an ending book to cover a certain range on the shelf, this method lets you specify both a beginning and an ending date. Likewise, you can use the CopyOpen function in the same way to get the open prices.

**Syntax:**

```
CopyClose( symbol_name, timeframe, start_time, stop_time, close_array[]);
CopyOpen(symbol_name, timeframe, start_time, stop_time, open_array[]);
```

**Example:**

```
double close_prices[];
double open_prices[];
datetime start_date = D'2023.06.01 00:00'; // Starting from June 1st, 2023
datetime end_date = D'2023.06.05 00:00'; // Ending on June 5th, 2023
// Copy the close prices from June 1st to June 5th
CopyClose(_Symbol, PERIOD_D1, start_date, end_date, close_prices);
// Copy the open prices from June 1st to June 5th
CopyOpen(_Symbol, PERIOD_D1, start_date, end_date, open_prices);
```

The price data for a particular symbol and timeframe is represented by the bookshelf in this analogy, which is symbolized by the symbol\_name and timeframe. The starting date (start\_time) is akin to choosing the book corresponding to "June 1st" on the shelf, and the ending date (stop\_time) is like finding the book corresponding to "June 5th". Like holding the selected books in your hands, you store the copied data in the target arrays (close\_array and open\_array).

By understanding this analogy, you can see how the CopyClose and CopyOpen functions help you retrieve specific data from your price data bookshelf based on a start and end date, making the data retrieval process intuitive and organized.

**Converting String to Time**

It can be difficult to use the method of calling by the start and end dates of a required time interval when working with Expert Advisors that operate within a specific time window during the day. The main problem is that dates change, but time does not. We automate our Expert Advisors, so it is not feasible to manually enter the date each day.

**Analogy**

Consider a library where the shelves are arranged according to the date and the time of day. You must be aware of the day and time if you wish to read books (or retrieve data) at a particular time of day. It can be laborious to update the date each day in order to retrieve the appropriate books from the shelves.  Consider the time strings as labels you put on the particular hours you are interested in reading in order to streamline this process (retrieving data).

**Example:**

```
// Declaring time strings
string start_time_str = "00:00";  // Start time
string end_time_str = "20:00";    // End time

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {

// Converting time strings to datetime values
   datetime start_time = StringToTime(start_time_str);
   datetime end_time = StringToTime(end_time_str);

   Print("start_time: ", start_time,"\nend_time: ",end_time);

  }
```

**Explanation:**

// Declaring time strings string start\_time\_str = "00:00"; // Start time string end\_time\_str = "20:00"; // End time

- Consider the time strings as labels you put to the particular hours you are interested in reading in order to streamline this process (retrieving data).

// Converting time strings to datetime values datetime start\_time = StringToTime(start\_time\_str); datetime end\_time = StringToTime(end\_time\_str);

- When you use the StringToTime function, it's as if you have a magical bookmark that, on any given day, knows exactly which shelf to point to. This is true regardless of the date. You won't have to bother about changing the date every day in this way. By guaranteeing that start\_time and end\_time consistently refer to the current time on the shelf, this makes it simple to retrieve the appropriate books (data) without the need for manual updates.

**Output:**

![Figure 1. Code Output](https://c.mql5.com/2/112/fff111.png)

**Implementation in the EA**

**Example:**

```
#include <Trade/Trade.mqh>

// Create an instance of the CTrade class for trading operations
CTrade trade;

// Unique identifier for the EA's trades
int MagicNumber = 103432;

// Arrays to store the previous day's open and close prices
double daily_close[];
double daily_open[];

// Arrays to store the first H1 bar's open and close prices of the day
double first_h1_price_close[];

// Arrays to store H1 bars' open and close prices
double H1_price_close[];
double H1_price_open[];

// Strings to define the trading start and end times and the first trade time
string start = "00:00";
string end = "20:00";
string firsttrade  = "02:00";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(MagicNumber);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Convert time strings to datetime format
   datetime start_time = StringToTime(start); // Convert start time string to datetime
   datetime end_time = StringToTime(end); // Convert end time string to datetime
   datetime current_time = TimeCurrent(); // Get the current time
   datetime first_tradetime = StringToTime(firsttrade); // Convert first trade time string to datetime

// Copy daily close and open prices
   CopyClose(_Symbol, PERIOD_D1, 1, 1, daily_close); // Copy the close price of the previous day
   CopyOpen(_Symbol, PERIOD_D1, 1, 1, daily_open); // Copy the open price of the previous day

// Set the arrays to be copied from right to left (latest to oldest)
   ArraySetAsSeries(daily_close, true); // Set daily_close array as series
   ArraySetAsSeries(daily_open, true); // Set daily_open array as series

// Copy close and open prices for the first H1 bar of the day
   CopyClose(_Symbol, PERIOD_H1, start_time, 1, first_h1_price_close); // Copy the close price of the first H1 bar

// Copy close prices for the latest 5 H1 bars
   CopyClose(_Symbol, PERIOD_H1, 0, 5, H1_price_close); // Copy the close prices of the latest 5 H1 bars
   CopyOpen(_Symbol, PERIOD_H1, 0, 5, H1_price_open); // Copy the open prices of the latest 5 H1 bars

// Set the arrays to be copied from right to left (latest to oldest)
   ArraySetAsSeries(H1_price_close, true); // Set H1_price_close array as series
   ArraySetAsSeries(H1_price_open, true); // Set H1_price_open array as series

// If the last daily bar is bearish
   if(daily_close[0] < daily_open[0])
     {
      // Check specific conditions for a sell trade
      if(H1_price_close[2] >= first_h1_price_close[0] && H1_price_close[1] < first_h1_price_close[0] && current_time >= first_tradetime)
        {

         Comment("Its a sell");

        }

     }

// If the last daily bar is bullish
   if(daily_close[0] > daily_open[0])
     {
      // Check specific conditions for a buy trade
      if(H1_price_close[2] <= first_h1_price_close[0] && H1_price_close[1] > first_h1_price_close[0] && current_time >= first_tradetime)
        {
         Comment("Its a buy");
        }
     }

  }
```

**Explanation:**

To begin, we import the required Trade library with #include <Trade/Trade.mqh>, allowing us to use trading-related functions. Next, we create an instance of the CTrade class, CTrade trade;, which will facilitate our trading operations. We also define a unique identifier for our EA's trades with int MagicNumber = 103432;.

We then declare arrays to store the previous day's open and close prices, the first H1 bar's open and close prices of the day, and the H1 bars' open and close prices. These arrays will hold the retrieved price data, which we will analyze to make trading decisions. The trading start and end times and the first trade time are defined as strings, which we will later convert to datetime format. In the OnInit function, we set the expert's magic number for identifying our trades.In the OnTick function, we convert the time strings to datetime format and retrieve the current time. We then use the CopyClose and CopyOpen functions to copy the daily and H1 candlestick data into the respective arrays. The ArraySetAsSeries function sets the arrays to be copied from right to left, ensuring the most recent data is at index 0.

Finally, we analyze the retrieved candlestick data to determine the market sentiment. If the last daily bar is bearish, we check specific conditions to decide on a sell trade. Similarly, if the last daily bar is bullish, we check conditions for a buy trade. The Comment function displays the trading decision on the chart. This example code sets up the structure to retrieve and analyze the open and close prices of daily and 1-hour candlesticks, then uses this analysis to determine the trade direction.

### **3\. Implementing Trade Execution**

**3.1 How to Buy and Sell in MQL5**

This section will cover the basic steps required to execute buy and sell orders in MQL5, including determining the levels of stop-loss and take-profit. The CTrade class from the Trade.mqh library, which offers basic methods for carrying out trades, will be utilized.

**Placing Buy and Sell Orders:**

We use the Buy and Sell methods of the CTrade class to place buy and sell orders. We can execute trades with little code thanks to these techniques. Consider the CTrade class as a special trading function-focused shelf in a library. We only need to grab the required book (method) off the shelf and apply it when we want to execute a trade.

**Setting Stop-Loss and Take-Profit:**

TP and SL levels are critical for risk management. Whereas TP specifies the desired profit, SL specifies the maximum loss you are willing to accept. Price points in relation to the current market price are specified for both. Consider SL and TP as bookmarks in a book that indicate where you will stop (SL) in the event that something goes wrong and where you will take your reward (TP) in the event that everything goes well.

**Example:**

```
#include <Trade/Trade.mqh> // Include the trade library for trading functions

// Create an instance of the CTrade class for trading operations
CTrade trade;

// Unique identifier for the EA's trades
int MagicNumber = 103432;

// Arrays to store the previous day's open and close prices
double daily_close[];
double daily_open[];

// Arrays to store the first H1 bar's open and close prices of the day
double first_h1_price_close[];

// Arrays to store H1 bars' open and close prices
double H1_price_close[];
double H1_price_open[];

// Strings to define the trading start and end times and the first trade time
string start = "00:00";
string end = "20:00";
string firsttrade  = "02:00";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(MagicNumber);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Convert time strings to datetime format
   datetime start_time = StringToTime(start); // Convert start time string to datetime
   datetime end_time = StringToTime(end); // Convert end time string to datetime
   datetime current_time = TimeCurrent(); // Get the current time
   datetime first_tradetime = StringToTime(firsttrade); // Convert first trade time string to datetime

// Copy daily close and open prices
   CopyClose(_Symbol, PERIOD_D1, 1, 1, daily_close); // Copy the close price of the previous day
   CopyOpen(_Symbol, PERIOD_D1, 1, 1, daily_open); // Copy the open price of the previous day

// Set the arrays to be copied from right to left (latest to oldest)
   ArraySetAsSeries(daily_close, true); // Set daily_close array as series
   ArraySetAsSeries(daily_open, true); // Set daily_open array as series

// Copy close and open prices for the first H1 bar of the day
   CopyClose(_Symbol, PERIOD_H1, start_time, 1, first_h1_price_close); // Copy the close price of the first H1 bar

// Copy close prices for the latest 5 H1 bars
   CopyClose(_Symbol, PERIOD_H1, 0, 5, H1_price_close); // Copy the close prices of the latest 5 H1 bars
   CopyOpen(_Symbol, PERIOD_H1, 0, 5, H1_price_open); // Copy the open prices of the latest 5 H1 bars

// Set the arrays to be copied from right to left (latest to oldest)
   ArraySetAsSeries(H1_price_close, true); // Set H1_price_close array as series
   ArraySetAsSeries(H1_price_open, true); // Set H1_price_open array as series

// Get the symbol point size
   double symbol_point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

// Get the current Bid price
   double Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

// Calculate the stop loss and take profit prices for sell
   double tp_sell = Bid - 400 * symbol_point;
   double sl_sell = Bid + 100 * symbol_point;

// Get the current Ask price
   double Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

// Calculate the stop loss and take profit prices for buy
   double tp_buy = Ask + 400 * symbol_point;
   double sl_buy = Ask - 100 * symbol_point;

// If the last daily bar is bearish
   if(daily_close[0] < daily_open[0])
     {
      // Check specific conditions for a sell trade
      if(H1_price_close[2] >= first_h1_price_close[0] && H1_price_close[1] < first_h1_price_close[0] && current_time >= first_tradetime)
        {
         // Execute the sell trade
         trade.Sell(1.0, _Symbol, Bid, sl_sell, tp_sell); // Replace with your lot size
         Comment("It's a sell");
        }
     }

// If the last daily bar is bullish
   if(daily_close[0] > daily_open[0])
     {
      // Check specific conditions for a buy trade
      if(H1_price_close[2] <= first_h1_price_close[0] && H1_price_close[1] > first_h1_price_close[0] && current_time >= first_tradetime)
        {
         // Execute the buy trade
         trade.Buy(1.0, _Symbol, Ask, sl_buy, tp_buy); // Replace with your lot size
         Comment("It's a buy");
        }
     }
  }
```

**Explanation:**

- Using #include, we include the trade library.
- We construct a trade instance of the CTrade class.
- Based on the current Bid and Ask prices, as well as the symbol's point size, we determine the SL and TP levels.
- We use the Buy and Sell methods to place orders with the specified SL and TP levels.

**Parameters of trade.Buy and trade.Sell:**

- _lotsize:_ The volume of the trade.
- _\_Symbol:_ The symbol on which the trade is executed (e.g., EURUSD).
- _Ask/Bid:_ The current Ask price for buy trades or Bid price for sell trades.
- _sl:_ The stop-loss price level.
- _tp:_ The take-profit price level.

In this article, we have discussed obtaining certain candlestick data, like the open and close prices, and how to place buy and sell orders. This progression is crucial for you as a beginner because it explains the fundamentals of incorporating trades into your program. The orders might, however, continue to be sent with each tick because of the OnTick event handler.

It's critical to restrict the Expert Advisor (EA) to one trade at a time in order to control risk and prevent over-trading. In order to do this, logic must be implemented to determine whether any open positions exist before making a new trade. This strategy helps prevent over-trading by ensuring that trades are only executed when necessary.

**Preventing Trades on Every Tick:**

Event handling by OnTick could lead to new positions for each tick. We make certain that trades are only made when necessary by putting in place a check for new bars and open positions. Consider every new bar as a book that has been added to our collection. We make sure no other book (trade) is being read at the moment and only choose to "read" (trade) when a new book (bar) is added.

**Example:**

```
// Flag to indicate a new bar has formed
bool newBar;

// Variable to store the time of the last bar
datetime lastBarTime;

// Array to store bar data (OHLC)
MqlRates bar[];

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

// Check for a new bar
   CopyRates(_Symbol, PERIOD_H1, 0, 3, bar); // Copy the latest 3 H1 bars
   if(bar[0].time > lastBarTime)  // Check if the latest bar time is greater than the last recorded bar time
     {
      newBar = true; // Set the newBar flag to true
      lastBarTime = bar[0].time; // Update the last bar time
     }
   else
     {
      newBar = false; // Set the newBar flag to false
     }

// If a new bar has formed
   if(newBar == true)
     {
      // If the last daily bar is bearish
      if(daily_close[0] < daily_open[0])
        {
         // Check specific conditions for a sell trade
         if(H1_price_close[2] >= first_h1_price_close[0] && H1_price_close[1] < first_h1_price_close[0] && current_time >= first_tradetime)
           {
            // Execute the sell trade
            trade.Sell(1.0, _Symbol, Bid, sl_sell, tp_sell); // Replace with your lot size
            Comment("It's a sell");
           }
        }

      // If the last daily bar is bullish
      if(daily_close[0] > daily_open[0])
        {
         // Check specific conditions for a buy trade
         if(H1_price_close[2] <= first_h1_price_close[0] && H1_price_close[1] > first_h1_price_close[0] && current_time >= first_tradetime)
           {
            // Execute the buy trade
            trade.Buy(1.0, _Symbol, Ask, sl_buy, tp_buy); // Replace with your lot size
            Comment("It's a buy");
           }
        }

     }

  }
```

**Explanation:**

bool newBar;

- In order to indicate whether a new bar (candlestick) has formed, this line declares a boolean variable called newBar.

datetime lastBarTime;

- In order to store the time of the most recent processed bar, this line declares a variable called lastBarTime of type datetime.

MqlRates bar\[\];

- This line declares an array bar of type MqlRates. This array will be used to store bar data, such as Open, High, Low, and Close (OHLC) prices.

CopyRates(\_Symbol, PERIOD\_H1, 0, 3, bar); // Copy the latest 3 H1 bars

- The most recent three hourly (H1) bars for the current symbol (\_Symbol) are copied into the bar array using the CopyRates function in this line.

if(bar\[0\].time > lastBarTime)

- This line determines whether the most recent bar's time (bar\[0\].time) exceeds the previous bar's time. Assuming it's accurate, a new bar has emerged.

newBar = true;

- The newBar flag is set to true if a new bar has formed.

lastBarTime = bar\[0\].time;

- The time of the most recent bar is updated in the lastBarTime variable.

else {

newBar = false;

        }

- If the time of the latest bar is not greater than lastBarTime, the newBar flag is set to false.

if(newBar == true)

{

// do this

}

- If a new bar has formed, do this.

This code makes sure that the code inside the if statement runs only when a new bar forms, not every tick. It does this by determining whether the time of the most recent bar is longer than the last bar time that was recorded. Based on this, we can determine when a new bar is formed and set the newBar flag appropriately. This avoids the code inside the ‘if(newBar == true)’ block from running repeatedly on each tick, which improves the expert advisor's performance and eliminates needless actions.

**3.2 Limiting the EA to One Open Position at a Time**

As the trade execution is now only initiated upon the formation of a new bar, we can improve our EA even more by restricting the quantity of open positions. This way, the EA will only initiate a new trade in the event that no other trades are pending.

This is the logic:

- _Check for Open Positions:_ We look to see if there are any open positions prior to making a new trade. The EA will place a trade if there are no open positions.
- _Stopping Trades at Every Tick:_ We make sure that trades are only made when they are absolutely necessary by adding a check for new bars and open positions.

Consider each new bar to be a book that has been added to our library. We only decide to "read" (trade) when a new book (bar) is added, after making sure that no other book (trade) is currently being read.

**Example:**

```
// Initialize the total number of positions being held
int totalPositions = 0;
for(int i = 0; i < PositionsTotal(); i++)
  {
// Get the ticket number for the position
   ulong ticket = PositionGetTicket(i);

// Check if the position's magic number matches
   if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
     {
      // Increment the total positions count
      totalPositions++;
     }
  }
```

**Explanation:**

int totalPositions = 0;

- In this line, the integer variable totalPositions is declared and set to zero. The number of open positions that the Expert Advisor (EA) currently holds will be counted using this variable.

for(int i = 0; i < PositionsTotal(); i++)

- Every open position is iterated over in this for loop. The total number of open positions in the trading terminal is returned by the PositionsTotal() function.
- i: An iterative loop counter that starts at 0 and increases by 1 each time. As long as i is less than PositionsTotal(), the loop will keep going.

ulong ticket = PositionGetTicket(i);

- The ticket number for the position at index i is retrieved on this line. The ticket number for the position at the given index is returned by the PositionGetTicket(i) function.
- ticket: A variable of type ulong (unsigned long) that stores the ticket number of the current position being examined.

if(PositionGetInteger(POSITION\_MAGIC) == MagicNumber)

- This if statement determines if the magic number assigned to the EA (MagicNumber) and the magic number of the current position match
- PositionGetInteger(POSITION\_MAGIC): A function that retrieves the integer value of the specified property (in this case, the magic number) for the current position.
- Magic Number: a predetermined constant that serves as this EA's unique trade identifier. It guarantees that only positions that the EA has opened are counted.

totalPositions++;

- This line increases the totalPositions counter by 1 if the magic number of the current position matches MagicNumber. In essence, this is a count of the number of positions that this particular EA has opened.

This block of code is used to count the number of positions currently held by the EA. It does this by:

- Initializing a counter (totalPositions) to zero.
- Looping through all open positions using a for loop.
- For each position, retrieving its ticket number.
- Checking if the position's magic number matches the EA's magic number.
- If there's a match, incrementing the totalPositions counter.

**3.3 Limiting the EA to a Maximum of Two Trades per Day**

We will make sure that the EA only opens a maximum of two trades per day in order to prevent over-trading now that we have the EA executing trades based on specific conditions on each new bar.

**Implementation Details**

This section of the code filters the trading history for the current day using the HistorySelect function. Next, we'll use the EA's distinct magic number to loop through the chosen trading history and count the number of trades the algorithm has made. We only count the entry trades by verifying the nature of each trade. The EA will not open new trades for the remainder of the day if there are two entry trades in total.

Consider it similar to overseeing a bookshelf, where each book is a trade. There's only room for two new books per day. You don't put any more books on the shelf until the following day, after you put those two on there. This ensures you maintain a tidy and manageable collection, preventing clutter (over-trading) and maintaining order (risk management).

**Example:**

```
// Select the trading history within the specified time range
bool success = HistorySelect(start_time, end_time); // Select the trading history

// Initialize the total number of trades for the day
int totalDeal = 0;
if(success)
  {
   for(int i = 0; i < HistoryDealsTotal(); i++)
     {
      // Get the ticket number for the deal
      ulong ticket = HistoryDealGetTicket(i);

      // Check if the deal's magic number matches
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == MagicNumber)
        {
         // Check if the deal was an entry
         if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN)
           {
            // Increment the total deal count
            totalDeal++;
           }
        }
     }
  }
```

**Explanation:**

- _HistorySelect:_ filters the day's trading history.
- _totalDeal:_ counter for keeping account of trade volume.
- _Loop through history:_ To count the trades, iterate through the trading history.
- _Verify the magic number:_ Verify that our EA owns the trade.
- Entry trades are counted by increasing the counter.

This code ensures that the EA only trades up to two times per day, maintaining a disciplined trading approach and reducing the risk of overtrading.

**3.4 Limiting Profit or Loss for the Day**

In this section, we'll make sure Expert Advisor doesn't exceed daily total profit or loss.

```
// Initialize the total profit
double totalProfit = 0;
long dealsMagic = 0;
double profit = 0;
if(success)
  {
   for(int i = 0; i < HistoryDealsTotal(); i++)
     {
      // Get the ticket number for the deal
      ulong ticket = HistoryDealGetTicket(i);

      // Check if the deal was an entry
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN)
        {
         // Get the magic number of the deal
         dealsMagic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
        }

      // Check if the deal was an exit
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
        {
         // Get the profit of the deal
         profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);

         // Check if the magic number matches
         if(MagicNumber == dealsMagic)
           {
            // Add the profit to the total profit
            totalProfit += profit;
           }
        }
     }
  }
```

**Explanation:**

- _HistorySelect:_ filters the day's trading history.
- _totalProfit:_ Counter to track the total profit or loss for the day.
- Iterates through the trading history in a loop to total the gains and losses.
- _Verify the entry deal:_ ascertains the magic number for every transaction.
- Verify the exit deal by computing the profit on each trade that is closed.
- _Add to total profit:_ Accumulates the profit or loss for trades with the matching magic number.

By halting additional trades after a predetermined profit or loss threshold is reached for the day, this strategy reduces risk. It guarantees controlled trading and aids in avoiding excessive trading, which raises the risk.

**3.5 Closing all Open Positions at a Specified Time**

This section will cover using the MQL5 framework to implement such a feature. As a result, there is less chance of unanticipated market fluctuations because no trades are left open at night or during off-peak trading hours. Consider it the end of the library's hours. We make sure that all positions are closed at the designated end time, just like a librarian makes sure that all books (trades) are returned and accounted for at the end of the day.

```
// Close trades at the specified end time
for(int i = 0; i < PositionsTotal(); i++)
  {
// Get the ticket number for the position
   ulong ticket = PositionGetTicket(i);

// Check if the position's magic number matches and if it's the end time
   if(PositionGetInteger(POSITION_MAGIC) == MagicNumber && current_time == end_time)
     {
      // Close the position
      trade.PositionClose(ticket);
     }
  }
```

**Explanation:**

This part retrieves the distinct ticket number for each open position by iterating through them all. Next, it verifies that the magic number of the position corresponds with the magic number of the EA and that the current time is within the designated end time. The PositionClose function of the CTrade class is used to close the position if these requirements are satisfied.

**3.6 Specifying the Days of the Week an EA Can Trade**

The current day of the week must be obtained and compared to our trading rules in order to guarantee that our Expert Advisor (EA) only trades on designated days of the week.

**Example:**

```
//getting the day of week and month
MqlDateTime day; //Declare an MqlDateTime structure to hold the current time and date
TimeCurrent(day); // Get the current time and fill the MqlDateTime structure
int week_day = day.day_of_week; //Extract the day of the week (0 = Sunday, 1 = Monday, ..., 6 = Saturday)

//getting the current month
MqlDateTime month; //Declare a structure to hold current month information
TimeCurrent(month); //Get the current date and time
int year_month = month.mon; //Extract the month component (1 for January, 2 for February, ..., 12 for December)

if(week_day == 5)
  {
   Comment("No trades on fridays", "\nday of week: ",week_day);
  }
else
   if(week_day == 4)
     {
      Comment("No trades on Thursdays", "\nday of week: ",week_day);
     }
   else
     {
      Comment(week_day);
     }
```

**Explanation:**

We must ascertain the current day of the week and compare it to predetermined guidelines in order to regulate the days on which an Expert Advisor (EA) may trade. To store the current date and time, we first declare a MqlDateTime structure. We first fill this structure with the current month and day of the week using the TimeCurrent() function.

Next, we check the day of the week using conditional statements. The code shows a message indicating that no trades are permitted on Fridays if the current day is Friday (shown by the value 5). In a similar vein, Thursday (value 4) indicates that no trades are permitted on Thursdays. The code indicates that trading is allowed on all other days by simply displaying the current day of the week.

**Analogy**

Consider every day of the week as a different type of book found in a library.

- **Friday (5):** The code forbids trading on Fridays, just as it does not permit the loan of  mystery novels on Fridays.
- **Thursday (4):** The code ceases trading on Thursdays, much like science fiction novels cannot be checked out on Thursdays.
- **Other Days:** The code allows trading for all other genres (days), just as it does for book genres.

### **Conclusion**

We've used a project-based learning methodology in this article to make it easy for anyone to understand the fundamentals of algorithmic trading with MQL5, even complete beginners. By following along, you have gained knowledge on how to do fundamental tasks like purchasing and selling in MQL5, obtaining the open and close prices of candlesticks, and putting strategies into place to prevent trading at every tick. You have also learned how to control important aspects of automated trading, such as restricting an EA to one trade at a time, defining trading days and periods, and establishing profit and loss limits.

Please feel free to ask questions regarding the subjects this article covers as you proceed on your journey. I'm here to help, whether it's with understanding particular code implementations or providing clarification on how these concepts relate to various trading strategies. Get in touch with us, and together we can discuss ways to improve your comprehension and utilization of MQL5 algorithmic trading. Recall that mastering any programming language, including MQL5, is a journey towards algorithmic trading. Naturally, one cannot understand everything at once. You will gradually increase your knowledge and abilities by adopting project-based learning, in which you apply concepts to real-world projects similar to the ones we've studied. Every project serves as a stepping stone that gradually builds your proficiency and confidence.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15299.zip "Download all attachments in the single ZIP archive")

[MQL5Project2.mq5](https://www.mql5.com/en/articles/download/15299/mql5project2.mq5 "Download MQL5Project2.mq5")(10.02 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Introduction to MQL5 (Part 36): Mastering API and WebRequest Function in MQL5 (X)](https://www.mql5.com/en/articles/20938)
- [Introduction to MQL5 (Part 35): Mastering API and WebRequest Function in MQL5 (IX)](https://www.mql5.com/en/articles/20859)
- [Introduction to MQL5 (Part 34): Mastering API and WebRequest Function in MQL5 (VIII)](https://www.mql5.com/en/articles/20802)
- [Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://www.mql5.com/en/articles/20700)
- [Introduction to MQL5 (Part 32): Mastering API and WebRequest Function in MQL5 (VI)](https://www.mql5.com/en/articles/20591)
- [Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://www.mql5.com/en/articles/20546)
- [Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://www.mql5.com/en/articles/20425)

**[Go to discussion](https://www.mql5.com/en/forum/470069)**

![DoEasy. Service functions (Part 1): Price patterns](https://c.mql5.com/2/71/DoEasy._Service_functions_Part_1___LOGO.png)[DoEasy. Service functions (Part 1): Price patterns](https://www.mql5.com/en/articles/14339)

In this article, we will start developing methods for searching for price patterns using timeseries data. A pattern has a certain set of parameters, common to any type of patterns. All data of this kind will be concentrated in the object class of the base abstract pattern. In the current article, we will create an abstract pattern class and a Pin Bar pattern class.

![SP500 Trading Strategy in MQL5 For Beginners](https://c.mql5.com/2/84/SP500_Trading_Strategy_in_MQL5____LOGO.png)[SP500 Trading Strategy in MQL5 For Beginners](https://www.mql5.com/en/articles/14815)

Discover how to leverage MQL5 to forecast the S&P 500 with precision, blending in classical technical analysis for added stability and combining algorithms with time-tested principles for robust market insights.

![Data Science and ML (Part 26): The Ultimate Battle in Time Series Forecasting — LSTM vs GRU Neural Networks](https://c.mql5.com/2/84/Data_Science_and_ML_Part_26__LOGO.png)[Data Science and ML (Part 26): The Ultimate Battle in Time Series Forecasting — LSTM vs GRU Neural Networks](https://www.mql5.com/en/articles/15182)

In the previous article, we discussed a simple RNN which despite its inability to understand long-term dependencies in the data, was able to make a profitable strategy. In this article, we are discussing both the Long-Short Term Memory(LSTM) and the Gated Recurrent Unit(GRU). These two were introduced to overcome the shortcomings of a simple RNN and to outsmart it.

![Developing an Expert Advisor (EA) based on the Consolidation Range Breakout strategy in MQL5](https://c.mql5.com/2/84/Developing_an_Expert_Advisor_based_on_the_Consolidation_Range_Breakout_strategy_in_MQL5___LOGO.png)[Developing an Expert Advisor (EA) based on the Consolidation Range Breakout strategy in MQL5](https://www.mql5.com/en/articles/15311)

This article outlines the steps to create an Expert Advisor (EA) that capitalizes on price breakouts after consolidation periods. By identifying consolidation ranges and setting breakout levels, traders can automate their trading decisions based on this strategy. The Expert Advisor aims to provide clear entry and exit points while avoiding false breakouts

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=noqldmgittrgnycybgfwwuziyzxqhcdz&ssn=1769092132409166342&ssn_dr=0&ssn_sr=0&fv_date=1769092132&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15299&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%208)%3A%20Beginner%27s%20Guide%20to%20Building%20Expert%20Advisors%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909213238858902&fz_uniq=5049162243561596451&sv=2552)

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