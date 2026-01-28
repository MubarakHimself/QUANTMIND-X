---
title: Price Action Analysis Toolkit Development Part (4): Analytics Forecaster EA
url: https://www.mql5.com/en/articles/16559
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:08:28.701706
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ibceevwtxspquhszwwvupjgexndvlren&ssn=1769191707623435260&ssn_dr=0&ssn_sr=0&fv_date=1769191707&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16559&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Price%20Action%20Analysis%20Toolkit%20Development%20Part%20(4)%3A%20Analytics%20Forecaster%20EA%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919170758311890&fz_uniq=5071605560807205659&sv=2552)

MetaTrader 5 / Integration


### Contents:

- [Introduction](https://www.mql5.com/en/articles/16559#para2)
- [Previous Article Review](https://www.mql5.com/en/articles/16559#para3)
- [Project Overview](https://www.mql5.com/en/articles/16559#para4)
- [Building a Telegram Bot and Fetching Chart IDs](https://www.mql5.com/en/articles/16559#para5)
- [Implementing Telegram Integration to the EA (Expert Advisor)](https://www.mql5.com/en/articles/16559#para6)
- [Testing](https://www.mql5.com/en/articles/16559#para7)
- [Conclusion](https://www.mql5.com/en/articles/16559#para8)

### Introduction

Having transitioned from a [script](https://www.mql5.com/en/articles/15927/170545#!tab=article) to an Expert Advisor (EA), the previous tool, [Analytics Master EA](https://www.mql5.com/en/articles/16434), was designed to analyze key metrics and provide continuous updates directly on the chart. While it served as a foundational asset, its capabilities were limited to accessing analysis information solely within the chart itself. Before we proceed with the development of more advanced analysis tools, I believe it is essential to enhance our analytics information broadcast.

In this article, we will focus on integrating our broadcast with Telegram to ensure wider access to analyzed information. I chose Telegram for this purpose due to its popularity and accessible interface among traders. Integrating MetaTrader 5 chart updates with Telegram offers significant advantages for active traders. This setup provides immediate access to crucial market information, enhances the user experience, and improves communication. Consequently, you can develop more effective strategies and respond swiftly in a rapidly changing market. With this integration, you can increase your chances of success and make more informed, timely decisions.

### Previous Article Review

Let's take a look at our previous tool, Analytics Master EA, for a recap. We are integrating the same analytics information from this tool into Telegram. To learn more about it, follow the link [https://www.mql5.com/en/articles/16434](https://www.mql5.com/en/articles/16434). The Analytics Master EA was designed to analyze and calculate the following key market metrics:

- Previous day's open and close
- Previous day's volume
- Current day's volume
- Previous day's high and low
- Key support and resistance levels
- Account balance
- Account equity
- Market spread
- Minimum and maximum lot size
- Market volatility

Understanding these metric values is crucial for traders, as they provide insights into market behavior and trends. Previous day metrics help establish a context for current market conditions, while current metrics assist in gauging performance. By identifying support and resistance levels, traders can make more informed decisions on entry and exit points. Moreover, knowing account balance and equity ensures that trading risk is managed effectively. Insights into market spread, lot sizes, and volatility are essential for optimizing trade execution and maximizing potential profits while minimizing risks. Overall, a solid grasp of these metrics empowers traders to devise informed strategies and enhance overall trading performance.

The EA drew trendlines for key support and resistance levels and provided anticipated market direction based on the calculated metrics. All this information was presented on the chart in a table-like format, updated regularly every two hours. See Fig. 1 below.

![Analytics Master Result](https://c.mql5.com/2/135/P009.png)

Fig 1. Analysis Result

The analysis section also includes the last update time for easy reference to the most recent information. The tool is designed solely for market analysis; it does not execute trades automatically. Users must manually execute their trades based on the analyzed information and signals generated. For optimal results, it is important to combine this data with the user’s own trading strategy.

### Project Overview

Telegram integration with MetaTrader 5 involves connecting the MetaTrader 5 trading platform to Telegram, a messaging service, allowing traders to receive instant notifications, alerts, and analytics about their trading activities directly in their Telegram chat. This integration utilizes the Telegram Bot API, enabling automated communication from the trading algorithm or Expert Advisor (EA) running on MetaTrader 5 to a designated Telegram chat. The diagram below summarizes everything.

![](https://c.mql5.com/2/135/image1-2.png)

Fig 2. Integration Pathway

Key Components of the Integration

- Telegram Bot: Create a Telegram bot using the BotFather on Telegram, which provides an access token needed to authenticate requests sent to the Telegram API.
- Chat ID: Identify the chat ID where messages should be sent. This can be a personal chat or a group chat, and it is used to target where the alerts will be delivered.
- [MetaTrader 5](https://www.metatrader5.com/en "https://www.metatrader5.com/en"): Utilize the [MQL5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") programming language to develop or modify EAs or scripts that can connect with the Telegram API. This generally involves using HTTP POST requests to send messages.

### Building a Telegram Bot and Fetching Chart IDs

Telegram bots are automated software applications that operate within the Telegram messaging platform, enabling interaction with users through automated tasks. One of their primary functions is to streamline communication by providing quick and relevant responses to user inquiries without human intervention. This capability allows businesses and developers to automate various tasks, such as sending notifications and processing commands, which enhances user experience and engagement.

Additionally, Telegram bots excel in information retrieval; they can pull data from external sources, such as market data or news feeds, and deliver it directly to users. In the context of trading, bots are particularly valuable for alerting users about specific market events, price changes, or signals based on predefined criteria. Their ability to integrate with APIs and other services further empowers them to perform advanced functions like data analysis and reporting, making them versatile tools for various applications. Follow the following steps to create your telegram bot:

Step 1: Open Telegram App

Make sure you have the Telegram app installed on your device.

Step 2: Search for the BotFather

![Step 2](https://c.mql5.com/2/135/Step2.png)

Fig 3. Botfather

In the search bar of the app, type BotFather.

BotFather is an official Telegram bot that allows you to create and manage other bots.

Step 3: Start a Chat with BotFather

![Step3/4](https://c.mql5.com/2/135/Step_3_and_4.png)

Fig 4. Step 3 and 4

Click on the BotFather result to open the chat.

Click the Start button, or type /start, to initiate the conversation.

Step 4: Create a New Bot

Type the command /newbot and send it.

BotFather will prompt you to choose a name for your bot. This is the display name that users will see.

After entering the name, you’ll be asked to provide a username for your bot. The username must end in "bot" (e.g., MyCoolBot).

Step 5: Receive Your Bot Token

![Step4/5](https://c.mql5.com/2/135/Step4_and_5.png)

Fig 5. Step 5

Once you have completed the previous steps, BotFather will create your bot and provide you with a unique API token. This token is essential for interacting with the Telegram Bot API, so save it somewhere secure.

After creating your Telegram bot, the next step is to find your [chat ID](https://www.mql5.com/en/docs/chart_operations/chartid). Chart IDs, on the other hand, serve as unique identifiers assigned to specific data visualizations or charts within trading applications or charting tools. These IDs play a crucial role in the identification and retrieval of charts, enabling users and developers to reference specific visualizations easily. Chart IDs facilitate the extraction of current or historical data related to particular charts, allowing for a tailored approach to data analysis. This aspect is especially beneficial in trading, as it empowers users to access relevant information quickly, leading to informed decision-making.

Furthermore, chart IDs enable customization, allowing developers to modify parameters and settings according to individual user preferences or trading strategies. When integrated with Telegram bots, chart IDs can significantly enhance functionality; they enable bots to provide users with specific data visualizations directly within their chat interface, streamlining the process of obtaining insights and making trading decisions. Below are two methods for acquiring it using Telegram bots.

Method 1: Using Get ID Bots

This is a straightforward approach. You can search for and open bots like @get\_id\_bot or @userinfobot. Once you start the bot by clicking the Start button or typing /start, it will respond with your chat ID, which you can then note down.

![Get Id Bot](https://c.mql5.com/2/135/user_infor.png)

Fig 6. Get ID

Method 2: Using a Web Browser

Begin by sending any message to your bot in Telegram. Then, open a web browser and enter the following URL, replacing <YourBotToken> with your actual bot token:

_https://api.telegram.org/bot<YourBotToken>/getUpdates_

After hitting Enter, examine the response returned by the API. Your chat ID will be located in the "result" section of the response.

### Implementing Telegram Integration to the EA (Expert Advisor)

Implementing Telegram Integration into the EA involves incorporating Telegram's messaging features into our trading Expert Advisor (EA) on MetaTrader. This integration enables the EA to send instantaneous notifications and alerts directly to a Telegram account, keeping users informed about market conditions, key metrics, and other essential trading information. By utilizing Telegram's API, I can enhance the EA's functionality, ensuring that users receive important updates without the need to constantly check the trading platform. This improves responsiveness to market changes, ultimately making the trading experience more efficient.

The code from the Analytics Master EA has been improved by incorporating commands that enable it to relay analyzed metrics to the Telegram app. I will provide the complete integrated MQL5 code and guide you step-by-step through the integration process.

Analytics Forecaster (EA) code

```
//+-------------------------------------------------------------------+
//|                                        Analytics Forecaster EA.mq5|
//|                                 Copyright 2024, Christian Benjamin|
//|                                               https://www.mql5.com|
//+-------------------------------------------------------------------+
#property copyright   "2024, MetaQuotes Software Corp."
#property link        "https://www.mql5.com/en/users/lynnchris"
#property description "EA for market analysis,commenting and Telegram Integeration"
#property version     "1.1"
#property strict

// Inputs for risk management
input double RiskPercentage = 1.0;        // Percentage of account balance to risk per trade
input double StopLossMultiplier = 1.0;    // Multiplier for determining the stop loss distance
input int ATR_Period = 14;                // Period for ATR calculation

// Telegram configuration
input string TelegramToken = "YOUR BOT TOKEN"; // Your Telegram bot token
input string ChatID = "YOUR CHART ID"; // Your chat ID
input bool SendTelegramAlerts = true; // Option to enable/disable Telegram notifications

// Global variables for storing values
datetime lastUpdateTime = 0;
double previousDayOpen, previousDayClose, previousDayHigh, previousDayLow;
double previousDayVolume;
double currentDayVolume;
double support, resistance;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   lastUpdateTime = 0; // Set the initial update time
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0); // Clean up any drawn objects on the current chart
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   UpdateMetrics(); // Call to the function that fetches and displays the metrics
  }

//+------------------------------------------------------------------+
//| Update metrics and display them                                  |
//+------------------------------------------------------------------+
void UpdateMetrics()
  {
// Check if 2 hours have passed since the last update
   if(TimeCurrent() - lastUpdateTime >= 2 * 3600)
     {
      // Fetch previous day's data
      datetime prevDay = iTime(NULL, PERIOD_D1, 1);
      previousDayOpen = iOpen(NULL, PERIOD_D1, 1);
      previousDayClose = iClose(NULL, PERIOD_D1, 1);
      previousDayHigh = iHigh(NULL, PERIOD_D1, 1);
      previousDayLow = iLow(NULL, PERIOD_D1, 1);
      previousDayVolume = iVolume(NULL, PERIOD_D1, 1);

      // Fetch current day's volume
      currentDayVolume = iVolume(NULL, PERIOD_D1, 0); // Volume for today

      // Calculate support and resistance
      support = previousDayLow - (previousDayHigh - previousDayLow) * 0.382; // Fibonacci level
      resistance = previousDayHigh + (previousDayHigh - previousDayLow) * 0.382; // Fibonacci level

      // Determine market direction
      string marketDirection = AnalyzeMarketDirection(previousDayOpen, previousDayClose, previousDayHigh, previousDayLow);

      // Calculate possible lot size based on risk management
      double lotSize = CalculateLotSize(support, resistance);

      // Retrieve account metrics
      double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      double accountEquity = AccountInfoDouble(ACCOUNT_EQUITY);

      // Calculate market spread manually
      double marketBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double marketAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double marketSpread = marketAsk - marketBid; // Calculate spread

      double minLotSize = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double maxLotSize = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

      // Calculate market volatility using ATR
      int atrHandle = iATR(NULL, PERIOD_H1, ATR_Period); // Get the ATR handle
      double atrValue = 0.0;

      if(atrHandle != INVALID_HANDLE)   // Check if the handle is valid
        {
         double atrBuffer[]; // Array to hold the ATR values
         if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) > 0)   // Copy the latest ATR value
           {
            atrValue = atrBuffer[0]; // Retrieve the ATR value from the buffer
           }
         IndicatorRelease(atrHandle); // Release the indicator handle
        }

      // Create the output string, including pair name and last update time
      string pairName = Symbol(); // Get the current symbol name
      string lastUpdateStr = TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES);
      string infoStr = StringFormat("Pair: %s\nPrev Day Open: %.2f\nPrev Day Close: %.2f\nPrev Day High: %.2f\nPrev Day Low: %.2f\n"
                                    "Prev Day Volume: %.0f\nCurrent Day Volume: %.0f\nMarket Direction: %s\n"
                                    "Support: %.2f\nResistance: %.2f\nAccount Balance: %.2f\nAccount Equity: %.2f\n"
                                    "Market Spread: %.2f\nMin Lot Size: %.2f, Max Lot Size: %.2f\n"
                                    "Market Volatility (ATR): %.2f\nLast Update Time: %s\nPossible Lot Size: %.2f",
                                    pairName, previousDayOpen, previousDayClose, previousDayHigh, previousDayLow,
                                    previousDayVolume, currentDayVolume, marketDirection,
                                    support, resistance, accountBalance, accountEquity, marketSpread,
                                    minLotSize, maxLotSize, atrValue, lastUpdateStr, lotSize);

      // Log the information
      Print(infoStr);

      // Display information on the chart
      Comment(infoStr);

      // Send Telegram notification
      if(SendTelegramAlerts)
         SendTelegramMessage(infoStr);

      // Remove old trend lines and create new ones for previous day's high/low
      ObjectsDeleteAll(0);

      // Draw continuous trend lines
      DrawContinuousTrendLine("PrevDayHigh", previousDayHigh);
      DrawContinuousTrendLine("PrevDayLow", previousDayLow);

      // Update last update time
      lastUpdateTime = TimeCurrent();
     }
  }

//+------------------------------------------------------------------+
//| Analyze market direction                                         |
//+------------------------------------------------------------------+
string AnalyzeMarketDirection(double open, double close, double high, double low)
  {
   string direction;

   if(close > open)
     {
      direction = "Bullish";
     }
   else
      if(close < open)
        {
         direction = "Bearish";
        }
      else
        {
         direction = "Neutral";
        }

// Include current trends or patterns based on high and low for further analysis
   if(high > open && high > close)
     {
      direction += " with bullish pressure"; // Example addition for context
     }
   else
      if(low < open && low < close)
        {
         direction += " with bearish pressure"; // Example addition for context
        }

   return direction;
  }

//+------------------------------------------------------------------+
//| Draw a continuous trend line to the left on the chart            |
//+------------------------------------------------------------------+
void DrawContinuousTrendLine(string name, double price)
  {
   datetime startTime = TimeCurrent() - 720 * 3600; // Extend 24 hours into the past
   ObjectCreate(0, name, OBJ_TREND, 0, startTime, price, TimeCurrent(), price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, (StringFind(name, "High") >= 0) ? clrRed : clrBlue);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 2); // Set thickness of the line
   ObjectSetInteger(0, name, OBJPROP_XSIZE, 0); // Set this property to extend the line infinitely to the left
  }

//+------------------------------------------------------------------+
//| Calculate the lot size based on risk management                  |
//+------------------------------------------------------------------+
double CalculateLotSize(double support, double resistance)
  {
   double stopLossDistance = MathAbs((support - resistance) * StopLossMultiplier);
   double riskAmount = AccountInfoDouble(ACCOUNT_BALANCE) * (RiskPercentage / 100.0);

// Get the tick size for the current symbol
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

// Calculate the lot size based on the stop loss and tick size
   double lotSize = riskAmount / (stopLossDistance / tickSize); // Adjusted for the correct pip size
   lotSize = NormalizeDouble(lotSize, 2); // Normalize the lot size to two decimal places

// Ensure lot size is above minimum lot size allowed by broker
   double minLotSize = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(lotSize < minLotSize)
      lotSize = minLotSize;

   return lotSize;
  }

//+------------------------------------------------------------------+
//| Send message to Telegram API                                     |
//+------------------------------------------------------------------+
void SendTelegramMessage(string message)
  {
   string url = StringFormat("https://api.telegram.org/bot%s/sendMessage", TelegramToken);
   string headers = "Content-Type: application/json\r\n"; // Content type for JSON
   int timeout = 1000; // Timeout in milliseconds

// Format the data as JSON
   string postData = StringFormat("{\"chat_id\":\"%s\",\"text\":\"%s\"}", ChatID, message);

// Convert the string to a char array
   char dataArray[];
   StringToCharArray(postData, dataArray);

// Prepare the result buffer and response headers
   char result[];
   string responseHeaders;

// Perform the web request
   int responseCode = WebRequest("POST", url, headers, timeout, dataArray, result, responseHeaders);
   if(responseCode == 200)   // HTTP 200 OK
     {
      Print("Message sent successfully!");
     }
   else
     {
      PrintFormat("Error sending message. HTTP Response Code: %d. Error: %s", responseCode, GetLastError());
     }
  }

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
```

1\. Declare Input Variables:

At the start of our EA, we should declare the necessary input variables. Input variables in MQL5 allow traders to customize the operation of the EA without digging into the code itself. This makes it easier to adapt trading strategies quickly. It's essential to name our variables intuitively. For example, TelegramToken clearly indicates its purpose. Consider using default values for your variables that align with common practice or settings in your trading strategy to reduce configuration complexity when testing.

```
input string TelegramToken = "YOUR_BOT_API_TOKEN"; // Replace with your actual bot token
input string ChatID = "YOUR_CHAT_ID"; // Replace with your actual chat ID
input bool SendTelegramAlerts = true; // Control whether alerts are sent
```

Remember to replace "YOUR\_BOT\_API\_TOKEN" and "YOUR\_CHAT\_ID" with the actual values.

2\. Create the Message-Sending Function:

The _SendTelegramMessage_ function effectively constructs a structured HTTP POST request to the Telegram API, sending a notification to the specified chat. By preparing the API URL, setting the headers, formatting the message data as JSON, executing the request, and handling the response, this function allows the EA to communicate instant updates to users via Telegram. This streamlined logic enables quick communication of trades and alerts and enhances the overall functionality of the EA as a trading assistant by keeping the user informed effortlessly.

- Purpose of the Function: This function handles the communication with the Telegram API. It encapsulates the logic needed to send a message to our Telegram bot. By creating reusable functions, we ensure code clarity and reduce duplication.
- Error Handling: Incorporating error handling is critical. Whenever we send a message, we should log not only successful sends, but also any errors that arise. This practice aids debugging and provides feedback.

```
void SendTelegramMessage(string message)
{
    string url = StringFormat("https://api.telegram.org/bot%s/sendMessage", TelegramToken);
    string headers = "Content-Type: application/json\r\n";
    int timeout = 1000;

    string postData = StringFormat("{\"chat_id\":\"%s\",\"text\":\"%s\"}", ChatID, message);

    char dataArray[];
    StringToCharArray(postData, dataArray);

    char result[];
    string responseHeaders;

    int responseCode = WebRequest("POST", url, headers, timeout, dataArray, result, responseHeaders);
    if (responseCode == 200)
    {
        Print("Message sent successfully! Response: ", CharArrayToString(result));
    }
    else
    {
        PrintFormat("Error sending message. HTTP Response Code: %d. Error: %s", responseCode, GetLastError());
    }
}
```

- Understanding WebRequest: The [WebRequest](https://www.mql5.com/en/docs/network/webrequest) function is crucial as it allows our EA to make HTTP requests to APIs. Ensure that the "Allow automated trading" option is enabled in the EA properties for proper functionality.

3. Triggering the Telegram Message

- Time Check and Fetching Data

The first part of the code initiates a time check to determine whether two hours have passed since the last update of market metrics. By using the _[TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent)()_ function, the code retrieves the current time and compares it with the _lastUpdateTime_ variable. If more than two hours have elapsed, the Expert Advisor (EA) proceeds to gather the latest data related to market conditions. This check is crucial for preventing the EA from flooding the Telegram chat with messages too frequently, which might be perceived as spam by the user.

```
// Check if 2 hours have passed since the last update
if (TimeCurrent() - lastUpdateTime >= 2 * 3600)
{
    // ... [Code that fetches data and calculates support/resistance, etc.]
}
```

- Creating the Output String

In the second part, a detailed output string, referred to as _infoStr_, is generated to consolidate the market metrics collected from the EA's operations. The code retrieves the current trading symbol with pair name, and additionally, it formats the current time for the message using _lastUpdateStr_. The _StringFormat_ function is then employed to construct the message, incorporating various placeholders that will be replaced with the specific metrics, such as previous day’s high, low, open, and close, current day volume, market direction, and other account details. This formatted string is integral to the operation, as it presents a clear and structured summary of the current market conditions, which will later be sent through Telegram.

```
// Create the output string, including pair name and last update time
string pairName = Symbol(); // Get the current symbol name
string lastUpdateStr = TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES);
string infoStr = StringFormat("Pair: %s\nPrev Day Open: %.2f\nPrev Day Close: %.2f\n"
                               "Prev Day High: %.2f\nPrev Day Low: %.2f\n"
                               "Prev Day Volume: %.0f\nCurrent Day Volume: %.0f\n"
                               "Market Direction: %s\nSupport: %.2f\nResistance: %.2f\n"
                               "Account Balance: %.2f\nAccount Equity: %.2f\n"
                               "Market Spread: %.2f\nMin Lot Size: %.2f, Max Lot Size: %.2f\n"
                               "Market Volatility (ATR): %.2f\nLast Update Time: %s\nPossible Lot Size: %.2f",
                               pairName, previousDayOpen, previousDayClose, previousDayHigh,
                               previousDayLow, previousDayVolume, currentDayVolume,
                               marketDirection, support, resistance, accountBalance,
                               accountEquity, marketSpread, minLotSize, maxLotSize,
                               atrValue, lastUpdateStr, lotSize);
```

-  Logging and Displaying Information

The third part centers around logging and displaying the constructed information. The [Print](https://www.mql5.com/en/docs/common/print) _(infoStr)_; function call serves to log the message to the Experts tab of the MetaTrader platform, allowing for visibility into the information being sent over Telegram. This provides a useful debugging tool to confirm that the metrics are being correctly formulated. Additionally, the Comment _(infoStr)_; command displays the same information directly on the trading chart, offering traders a visual confirmation of the metrics without needing to consult the logs. These steps keep the user informed about the reported metrics and verify the accuracy of the data before sending it to Telegram.

```
// Log the information
Print(infoStr); // Here the information is logged for debugging

// Display information on the chart
Comment(infoStr); // Display the same information on the chart
```

- Sending the Telegram Notification

In the final part, the code manages the actual sending of the Telegram notification. The _(SendTelegramAlerts)_ statement checks if the option to send alerts is enabled, allowing users to easily disable notifications without requiring code modifications. If alerts are enabled, the function _SendTelegramMessage(infoStr)_ is invoked, which sends the carefully constructed message to the specified Telegram chat. This step is critical, as it is the point where the market metrics are communicated effectively to the user. Following this, the code updates the _lastUpdateTime_ variable to the current time using _lastUpdateTime_ = [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent)();, thereby marking the conclusion of this update cycle and ensuring that the timing for the next update adheres to the two-hour interval established earlier.

```
// Send Telegram notification
if (SendTelegramAlerts) // Check if sending alerts is enabled
    SendTelegramMessage(infoStr); // Send the constructed message

// Update last update time
lastUpdateTime = TimeCurrent();
```

- Current Pair Name

I have also added a feature that displays the current pair name alongside the analyzed metrics, making it easier to identify which pair is being analyzed.

```
// Create the output string, including pair name and last update time
      string pairName = Symbol(); // Get the current symbol name
```

Below is how the information will appear on Telegram:

![Pair Name](https://c.mql5.com/2/135/PAIR_NAME.PNG)

Fig 7. Pair Result

### Testing

Before testing, a few adjustments need to be made in your MetaTrader 5 to ensure that information is relayed to Telegram smoothly.

Allow Web Requests:

- Open MetaTrader 5 and navigate to Tools > Options > Expert Advisors.

### ![Allowing Web Request](https://c.mql5.com/2/135/Capture.PNG)

Fig 8. Setting Web Requests

- Check the box for "Allow WebRequest for listed URL" and add _https://api.telegram.org_ to the list. This setting enables the EA to send requests to Telegram's API.

![Setting Web Request](https://c.mql5.com/2/135/saved.png)

Fig 9. Setting Web Request

For testing purposes, you can also adjust the update timing to a lower frequency. In this case, I will reduce it from 2 hours to 15 seconds.

```
//+------------------------------------------------------------------+
//| Update metrics and display them                                  |
//+------------------------------------------------------------------+
void UpdateMetrics()
  {
// Check if 15 seconds have passed since the last update
   if(TimeCurrent() - lastUpdateTime >= 15)
```

Now, proceed to [compile](https://www.mql5.com/en/book/intro/edit_compile_run) your MQL5 code in [MetaEditor](https://www.mql5.com/en/book/intro/edit_compile_run) and attach the Expert Advisor or script to a chart in MetaTrader 5. After a successful [compilation](https://www.mql5.com/en/book/intro/edit_compile_run), please drag your Expert Advisor (EA) onto the chart. From there, you should start receiving notifications in your Telegram regarding updates sent from the EA. Below, I have illustrated the test results from my Telegram.

![Telegram Result](https://c.mql5.com/2/135/Integration_Result.gif)

Fig 10. Telegram Result

The diagram below also shows that the information provided on the MetaTrader 5 chart is the same as the information related to Telegram.

![Result](https://c.mql5.com/2/135/Telegram_Integration.png)

Fig 11. Test Results

### Conclusion

In conclusion, the success of the Analytics Forecaster EA, as evidenced by the diagrams presented above, lies in its sophisticated approach to market analysis and timely notifications through Telegram integration. By utilizing various trading metrics such as the previous day's data, current volume comparisons, market direction, and risk management principles, the EA provides traders with valuable insights. The calculated support and resistance levels combined with automated lot size determination enable more informed trading decisions, ensuring that both novice and experienced traders can adapt strategies according to market conditions while managing their risk effectively.

Additionally, the seamless integration with Telegram enhances user interaction by delivering prompt updates, allowing traders to act swiftly in response to market changes. The ability to receive critical trading information on a mobile platform significantly improves the convenience of monitoring trades, while also fostering a more responsive trading environment. Overall, the Analytics Forecaster EA demonstrates noteworthy capabilities in automating analysis and risk management, thus empowering traders to optimize their performance in a dynamic market. With its focus on data accuracy and user-friendly features, this EA stands out as a powerful tool in the trader's arsenal, paving the way for future advancements in automated trading solutions.

| Date | Tool Name | Description | Version | Updates | Notes |
| --- | --- | --- | --- | --- | --- |
| 01/10/24 | Chart Projector | Script to overlay the previous day's price action with ghost effect. | 1.0 | Initial Release | First tool in Lynnchris Tool Chest |
| 18/11/24 | Analytical Comment | It provides previous day's information in a tabular format, as well as anticipates the future direction of the market. | 1.0 | Initial Release | Second tool in the Lynnchris Tool Chest |
| 27/11/24 | Analytics Master | Regular Update of market metrics after every two hours | 1.01 | Second Release | Third tool in the Lynnchris Tool Chest |
| 02/12/2024 | Analytics Forecaster | Regular Update of market metrics after every two hours with telegram integration | 1.1 | Third Edition | Tool number 5 |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16559.zip "Download all attachments in the single ZIP archive")

[Analytics\_Forecaster\_EA.mq5](https://www.mql5.com/en/articles/download/16559/analytics_forecaster_ea.mq5 "Download Analytics_Forecaster_EA.mq5")(10.54 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Price Action Analysis Toolkit (Part 55): Designing a CPI Mini-Candle Overlay for Intra-bar Pressure](https://www.mql5.com/en/articles/20949)
- [Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://www.mql5.com/en/articles/20851)
- [Price Action Analysis Toolkit Development (Part 53): Pattern Density Heatmap for Support and Resistance Zone Discovery](https://www.mql5.com/en/articles/20390)
- [Price Action Analysis Toolkit Development (Part 52): Master Market Structure with Multi-Timeframe Visual Analysis](https://www.mql5.com/en/articles/20387)
- [Price Action Analysis Toolkit Development (Part 51): Revolutionary Chart Search Technology for Candlestick Pattern Discovery](https://www.mql5.com/en/articles/20313)
- [Price Action Analysis Toolkit Development (Part 50): Developing the RVGI, CCI and SMA Confluence Engine in MQL5](https://www.mql5.com/en/articles/20262)
- [Price Action Analysis Toolkit Development (Part 49): Integrating Trend, Momentum, and Volatility Indicators into One MQL5 System](https://www.mql5.com/en/articles/20168)

**[Go to discussion](https://www.mql5.com/en/forum/477767)**

![Trading Insights Through Volume: Trend Confirmation](https://c.mql5.com/2/104/Trading_Insights_Through_Volume___LOGO.png)[Trading Insights Through Volume: Trend Confirmation](https://www.mql5.com/en/articles/16573)

The Enhanced Trend Confirmation Technique combines price action, volume analysis, and machine learning to identify genuine market movements. It requires both price breakouts and volume surges (50% above average) for trade validation, while using an LSTM neural network for additional confirmation. The system employs ATR-based position sizing and dynamic risk management, making it adaptable to various market conditions while filtering out false signals.

![Chemical reaction optimization (CRO) algorithm (Part II): Assembling and results](https://c.mql5.com/2/81/Algorithm_for_optimization_by_chemical_reactions__LOGO___1.png)[Chemical reaction optimization (CRO) algorithm (Part II): Assembling and results](https://www.mql5.com/en/articles/15080)

In the second part, we will collect chemical operators into a single algorithm and present a detailed analysis of its results. Let's find out how the Chemical reaction optimization (CRO) method copes with solving complex problems on test functions.

![Reimagining Classic Strategies (Part 12): EURUSD Breakout Strategy](https://c.mql5.com/2/104/Reimagining_Classic_Strategies_Part_12___LOGO__1.png)[Reimagining Classic Strategies (Part 12): EURUSD Breakout Strategy](https://www.mql5.com/en/articles/16569)

Join us today as we challenge ourselves to build a profitable break-out trading strategy in MQL5. We selected the EURUSD pair and attempted to trade price breakouts on the hourly timeframe. Our system had difficulty distinguishing between false breakouts and the beginning of true trends. We layered our system with filters intended to minimize our losses whilst increasing our gains. In the end, we successfully made our system profitable and less prone to false breakouts.

![Introduction to MQL5 (Part 10): A Beginner's Guide to Working with Built-in Indicators in MQL5](https://c.mql5.com/2/104/Introduction_to_MQL5_Part_10___LOGO__1.png)[Introduction to MQL5 (Part 10): A Beginner's Guide to Working with Built-in Indicators in MQL5](https://www.mql5.com/en/articles/16514)

This article introduces working with built-in indicators in MQL5, focusing on creating an RSI-based Expert Advisor (EA) using a project-based approach. You'll learn to retrieve and utilize RSI values, handle liquidity sweeps, and enhance trade visualization using chart objects. Additionally, the article emphasizes effective risk management, including setting percentage-based risk, implementing risk-reward ratios, and applying risk modifications to secure profits.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dpzyrjkthmwmagdhfrymwlilbpcesexa&ssn=1769191707623435260&ssn_dr=0&ssn_sr=0&fv_date=1769191707&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16559&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Price%20Action%20Analysis%20Toolkit%20Development%20Part%20(4)%3A%20Analytics%20Forecaster%20EA%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919170758346685&fz_uniq=5071605560807205659&sv=2552)

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