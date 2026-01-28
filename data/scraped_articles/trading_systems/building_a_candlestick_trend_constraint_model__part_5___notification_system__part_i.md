---
title: Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part I)
url: https://www.mql5.com/en/articles/14963
categories: Trading Systems, Integration, Indicators, Expert Advisors
relevance_score: 5
scraped_at: 2026-01-23T17:33:52.190755
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/14963&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068385211508521166)

MetaTrader 5 / Trading systems


Contents

- [Introduction](https://www.mql5.com/en/articles/14963#para1)
- [Terminal notifications](https://www.mql5.com/en/articles/14963#para2)
- [Push notifications](https://www.mql5.com/en/articles/14963#para3)
- [Email notifications](https://www.mql5.com/en/articles/14963#para4)
- [Debugging Trend Constraint V1.04](https://www.mql5.com/en/articles/14963#para8)
- [Integrating](https://www.mql5.com/en/articles/14963#para5) [the social network channels](https://www.mql5.com/en/articles/14963#para5) (e.g. Telegram, and WhatsApp)
- [Telegram Integration](https://www.mql5.com/en/articles/14963#para9)
- [WhatsApp Integration](https://www.mql5.com/en/articles/14963#para10)
- [The power of virtual private network (VPS) on our notification system](https://www.mql5.com/en/articles/14963edit#para6)
- [Conclusion](https://www.mql5.com/en/articles/14963#para7)

### Introduction

MetaTrader 5 provides various notification options for informing users about trading events, such as terminal, email, and push notifications. Integrating it with social platforms like Telegram and WhatsApp for signal sharing can be highly beneficial. Configuring notifications in MetaTrader 5 enables you to stay informed about your trading activities regardless of your location. By utilizing the aforementioned notification access methods, you can select the option that best fits your requirements. This guide will walk you through the setup and customization of MetaTrader 5 notifications, including integration with Telegram and other social media platforms. It will focus on configuration details and the initial steps for integration, paving the way for a more in-depth exploration in Part II of this series.

### Terminal notifications

Terminal notifications are internal alerts in the MetaTrader 5 platform. They cover all alert events triggered within MetaTrader 5. You can manually set alerts in MetaTrader 5 to trigger when specific conditions are met. Alert settings are typically found in the tools tab at the bottom of the default MetaTrader 5 window.

- Symbol: Select the trading instrument for which you want to set the alert.
- Condition: Choose the condition that will trigger the alert (e.g., Bid >, Ask <, Time =, etc.).
- Value: Specify the value that the condition will be compared against (e.g., a specific price level).
- Source: Select the type of alert (e.g., Sound, File, Mail, etc.). For terminal notifications, you can choose Sound.
- Action: Select the action to be taken when the alert is triggered (e.g., play a sound, send a notification, etc.).
- Sound: Choose a sound file to be played when the alert is triggered.
- Timeout: Set the time interval after which the alert will be checked again.
- Maximum Iterations: Specify the number of times the alert should be triggered.

Below is an example of setting up an alert using Step Index synthetics. The values shown in the table can be adjusted based on your needs. After configuring the settings, click Ok to create the alert and it will be ready for activation. See the animated demonstration image below the table for a quick summary. To begin, right-click on the alert tab in the tools window, select create from the menu that appears, and a dialogue box with the settings outlined in the table will open.

| Setting | Value |
| --- | --- |
| Symbol | Step Index |
| Condition | Bid> |
| Value | 9666 |
| Source | Sound |
| Action | Selected a sound file ( alert2.wav) |
| Timeout | Set to 60 seconds |
| Maximum Iterations | Set to 5 |

![How to setup terminal alert in MetaTrader 5](https://c.mql5.com/2/80/How_to_setup_alert_on_mt5.gif)

### Push notifications

It is  a feature in MetaTrader 5 which allows notifications generated on platform whether generated internally or by an indictor or an  expert advisor to be send to a mobile phone  MetaTrader 5 platform via the MetaQuotes ID of the mobile device. For mobile device to  receive push notification MetaTrader 5 must be installed from [playstore](https://www.mql5.com/go?link=https://play.google.com/store/apps/details?id=net.metaquotes.metatrader5%26hl=en "Download for Android") for Android  or [apple store](https://www.mql5.com/go?link=https://apps.apple.com/us/app/metatrader-5/id413251709 "Dowload for IOS") for IOS. On desktop MetaTrader 5 push notifications must be enabled  for alert to be received on mobile phone. A unique MetaQuotes ID is created on mobile phone soon after installation of mobile MetaTrader 5.

On the left, you can find an image illustrating how to find the MetaQuotes ID. On the right, there is an image demonstrating how to activate push notifications on the desktop MetaTrader 5 platform, along with the field for entering the MetaQuotes ID. By checking the box and entering the ID from your mobile platform, you can instantly begin receiving notifications in the messages section of your MetaTrader 5 mobile platform. You can add many MetaQuotes IDs from various mobile devices seperating them

![Locate metaquotes ID on Android](https://c.mql5.com/2/80/Locate_metaquotes_ID_on_android.gif)![Configure MetaTrader 5 Terminal Push Notifications](https://c.mql5.com/2/81/Configure_mt5_terminal_Push_notifications.gif)

### Email notifications

Email remains a powerful and versatile tool for communication due to its convenience, efficiency, and wide range of functionalities. Whether for personal use, business correspondence, or professional networking, email provides a reliable and effective means of staying connected and exchanging information. Setting up email notifications in MetaTrader 5 allows you to receive alerts via email for various trading activities, such as price movements, order executions, and custom events.

The advantage of emails is their nearly instantaneous delivery, enabling swift information exchange and efficient dissemination to large groups. Email services often provide encryption to secure content, protecting sensitive information. Direct delivery to recipient inboxes minimizes interception risks compared to other communication forms.

Here is a step by step guide for setting up email notifications:

> - Launch the MetaTrader 5 platform on your computer.
> - Go to Tools > Options > Email.
> - Check the box Enable.
> - Fill in SMTP Server Details:

> | Setting | Fill in details |
> | --- | --- |
> | SMTP server | The SMTP server address for your email provider (e.g., smtp.gmail.com for Gmail). |
> | SMTP login | Your email address (e.g., your-email@gmail.com). |
> | SMTP password | Your email password or app-specific password if using Gmail. |
> | From | Your email address (e.g., your-email@gmail.com). |
> | To | The email address where you want to receive notifications (can be the same as the From address or different). |

> ![Setting up email notifications on mt5](https://c.mql5.com/2/80/Setup_Email_notifications.gif)

What is SMTP server.

Based on my research, an Simple Mail Transfer Protocol (SMTP) server is a mail server that utilizes the (SMTP) protocol to send, receive, and relay outgoing emails. It functions alongside the Mail Transfer Agent (MTA) to direct emails from the sender's email client to the recipient's email server. Here is a list of mail providers with SMTP servers:

- gmail
- yahoo mail
- hotmail
- zohomail
- icloud mail

### Debugging Trend Constraint V1.04

We have successfully integrated the Draw\_Line style into our system's trends. However, I discovered that numerous alerts were being triggered after each bar due to this new feature. For instance, when a chart was set to a minute timeframe, alerts were being generated at every minute candle close, which is quite overwhelming.

![Alert after every bar notifications problem.](https://c.mql5.com/2/80/Trend_Constraint_V1.04_Errors.PNG)

Our aim is to obtain optimal signals, focusing on only a select few. To tackle this issue, I have contemplated resolving the matter by removing myalert() from the fifth and sixth buffers. It is essential that this code is rectified for the seamless integration of Telegram and WhatsApp as discussed in this article. Below is the revised version of our code:

Buffer 5 Modified:

```
// --- Buffer5 (Buy Trend)
      if(MA5[i] > MA6[i])
        {
         Buffer5[i] = Low[i] - 15 * myPoint;
         // Disabled myAlert from Buffer 5
         // myAlert("indicator", "BUY TREND | MA Fast: " + DoubleToString(MA5[i], 2) + " | MA Slow: " + DoubleToString(MA6[i], 2));
        }
```

Buffer 6 Modified:

```
// --- Buffer6 (Sell Trend)
      if(MA5[i] < MA6[i])
        {
         Buffer6[i] = High[i] + 15 * myPoint;
         // Disabled myAlert from Buffer 6
         // myAlert("indicator", "SELL TREND | MA Fast: " + DoubleToString(MA5[i], 2) + " | MA Slow: " + DoubleToString(MA6[i], 2));
        }
```

Trend Constraint V1.04 modified:

```
///Indicator Name: Trend Constraint
#property copyright "Clemence Benjamin"
#property link      "https://mql5.com"
#property version   "1.04"
#property description "A model that seek to produce sell signal when D1 candle is Bearish only and  buy signal when it is Bullish"
//--- indicator settings
#property indicator_chart_window
#property indicator_buffers 6
#property indicator_plots 6

#property indicator_type1 DRAW_ARROW
#property indicator_width1 5
#property indicator_color1 0xFF3C00
#property indicator_label1 "Buy"

#property indicator_type2 DRAW_ARROW
#property indicator_width2 5
#property indicator_color2 0x0000FF
#property indicator_label2 "Sell"

#property indicator_type3 DRAW_ARROW
#property indicator_width3 2
#property indicator_color3 0xE8351A
#property indicator_label3 "Buy Reversal"

#property indicator_type4 DRAW_ARROW
#property indicator_width4 2
#property indicator_color4 0x1A1AE8
#property indicator_label4 "Sell Reversal"

#property indicator_type5 DRAW_LINE
#property indicator_style5 STYLE_SOLID
#property indicator_width5 2
#property indicator_color5 0xFFAA00
#property indicator_label5 "Buy Trend"

#property indicator_type6 DRAW_LINE
#property indicator_style6 STYLE_SOLID
#property indicator_width6 2
#property indicator_color6 0x0000FF
#property indicator_label6 "Sell Trend"

#define PLOT_MAXIMUM_BARS_BACK 5000
#define OMIT_OLDEST_BARS 50

//--- indicator buffers
double Buffer1[];
double Buffer2[];
double Buffer3[];
double Buffer4[];
double Buffer5[];
double Buffer6[];

input double Oversold = 30;
input double Overbought = 70;
input int Slow_MA_period = 200;
input int Fast_MA_period = 100;
datetime time_alert; //used when sending alert
input bool Audible_Alerts = true;
input bool Push_Notifications = true;
double myPoint; //initialized in OnInit
int RSI_handle;
double RSI[];
double Open[];
double Close[];
int MA_handle;
double MA[];
int MA_handle2;
double MA2[];
int MA_handle3;
double MA3[];
int MA_handle4;
double MA4[];
double Low[];
double High[];
int MA_handle5;
double MA5[];
int MA_handle6;
double MA6[];

void myAlert(string type, string message)
  {
   if(type == "print")
      Print(message);
   else if(type == "error")
     {
      Print(type+" | Trend Constraint V1.04 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
     }
   else if(type == "order")
     {
     }
   else if(type == "modify")
     {
     }
   else if(type == "indicator")
     {
      if(Audible_Alerts) Alert(type+" | Trend Constraint V1.04 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
      if(Push_Notifications) SendNotification(type+" | Trend Constraint V1.04 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
     }
  }

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, Buffer1);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(0, PLOT_ARROW, 241);
   SetIndexBuffer(1, Buffer2);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(1, PLOT_ARROW, 242);
   SetIndexBuffer(2, Buffer3);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(2, PLOT_ARROW, 236);
   SetIndexBuffer(3, Buffer4);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(3, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(3, PLOT_ARROW, 238);
   SetIndexBuffer(4, Buffer5);
   PlotIndexSetDouble(4, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(4, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   SetIndexBuffer(5, Buffer6);
   PlotIndexSetDouble(5, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(5, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   //initialize myPoint
   myPoint = Point();
   if(Digits() == 5 || Digits() == 3)
     {
      myPoint *= 10;
     }
   RSI_handle = iRSI(NULL, PERIOD_CURRENT, 14, PRICE_CLOSE);
   if(RSI_handle < 0)
     {
      Print("The creation of iRSI has failed: RSI_handle=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle = iMA(NULL, PERIOD_CURRENT, 7, 0, MODE_SMMA, PRICE_CLOSE);
   if(MA_handle < 0)
     {
      Print("The creation of iMA has failed: MA_handle=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle2 = iMA(NULL, PERIOD_CURRENT, 400, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle2 < 0)
     {
      Print("The creation of iMA has failed: MA_handle2=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle3 = iMA(NULL, PERIOD_CURRENT, 100, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle3 < 0)
     {
      Print("The creation of iMA has failed: MA_handle3=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle4 = iMA(NULL, PERIOD_CURRENT, 200, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle4 < 0)
     {
      Print("The creation of iMA has failed: MA_handle4=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle5 = iMA(NULL, PERIOD_CURRENT, Fast_MA_period, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle5 < 0)
     {
      Print("The creation of iMA has failed: MA_handle5=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle6 = iMA(NULL, PERIOD_CURRENT, Slow_MA_period, 0, MODE_SMA, PRICE_CLOSE);
   if(MA_handle6 < 0)
     {
      Print("The creation of iMA has failed: MA_handle6=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime& time[],
                const double& open[],
                const double& high[],
                const double& low[],
                const double& close[],
                const long& tick_volume[],
                const long& volume[],
                const int& spread[])
  {
   int limit = rates_total - prev_calculated;
   //--- counting from 0 to rates_total
   ArraySetAsSeries(Buffer1, true);
   ArraySetAsSeries(Buffer2, true);
   ArraySetAsSeries(Buffer3, true);
   ArraySetAsSeries(Buffer4, true);
   ArraySetAsSeries(Buffer5, true);
   ArraySetAsSeries(Buffer6, true);
   //--- initial zero
   if(prev_calculated < 1)
     {
      ArrayInitialize(Buffer1, EMPTY_VALUE);
      ArrayInitialize(Buffer2, EMPTY_VALUE);
      ArrayInitialize(Buffer3, EMPTY_VALUE);
      ArrayInitialize(Buffer4, EMPTY_VALUE);
      ArrayInitialize(Buffer5, EMPTY_VALUE);
      ArrayInitialize(Buffer6, EMPTY_VALUE);
     }
   else
      limit++;
   datetime Time[];

   int RSIBuffer;
   int MABuffer;
   int RSIPeriod = 14;
   ArrayResize(RSI, rates_total);
   ArrayResize(Open, rates_total);
   ArrayResize(Close, rates_total);
   CopyOpen(NULL, 0, 0, rates_total, Open);
   CopyClose(NULL, 0, 0, rates_total, Close);
   if(CopyBuffer(RSI_handle, 0, 0, rates_total, RSI) < 0)
     {
      Print("Getting RSI values failed, not enough bars!");
     }
   ArrayResize(MA, rates_total);
   if(CopyBuffer(MA_handle, 0, 0, rates_total, MA) < 0)
     {
      Print("Getting MA values failed, not enough bars!");
     }
   ArrayResize(MA2, rates_total);
   if(CopyBuffer(MA_handle2, 0, 0, rates_total, MA2) < 0)
     {
      Print("Getting MA values failed, not enough bars!");
     }
   ArrayResize(MA3, rates_total);
   if(CopyBuffer(MA_handle3, 0, 0, rates_total, MA3) < 0)
     {
      Print("Getting MA values failed, not enough bars!");
     }
   ArrayResize(MA4, rates_total);
   if(CopyBuffer(MA_handle4, 0, 0, rates_total, MA4) < 0)
     {
      Print("Getting MA values failed, not enough bars!");
     }
   ArrayResize(Low, rates_total);
   if(CopyLow(NULL, 0, 0, rates_total, Low) < 0)
     {
      Print("Getting LOW values failed, not enough bars!");
     }
   ArrayResize(High, rates_total);
   if(CopyHigh(NULL, 0, 0, rates_total, High) < 0)
     {
      Print("Getting HIGH values failed, not enough bars!");
     }
   ArrayResize(MA5, rates_total);
   if(CopyBuffer(MA_handle5, 0, 0, rates_total, MA5) < 0)
     {
      Print("Getting MA values failed, not enough bars!");
     }
   ArrayResize(MA6, rates_total);
   if(CopyBuffer(MA_handle6, 0, 0, rates_total, MA6) < 0)
     {
      Print("Getting MA values failed, not enough bars!");
     }

   for(int i=limit-1; i>=0; i--)
     {
      if(i < rates_total-1 && time[i] != time[i+1]+PeriodSeconds())
        {
         continue;
        }
      Buffer1[i] = EMPTY_VALUE;
      Buffer2[i] = EMPTY_VALUE;
      Buffer3[i] = EMPTY_VALUE;
      Buffer4[i] = EMPTY_VALUE;
      Buffer5[i] = EMPTY_VALUE;
      Buffer6[i] = EMPTY_VALUE;

      // --- Indicator calculations
      // --- Buffer1 (Buy)
      if((Close[i] > MA[i] && MA[i] > MA2[i] && RSI[i] < Oversold) || (RSI[i] < Oversold && Close[i] > MA3[i]))
        {
         Buffer1[i] = Low[i] - 5 * myPoint;
         myAlert("indicator", "BUY OPPORTUNITY | RSI: " + DoubleToString(RSI[i], 2) + " | MA: " + DoubleToString(MA[i], 2));
        }

      // --- Buffer2 (Sell)
      if((Close[i] < MA[i] && MA[i] < MA2[i] && RSI[i] > Overbought) || (RSI[i] > Overbought && Close[i] < MA3[i]))
        {
         Buffer2[i] = High[i] + 5 * myPoint;
         myAlert("indicator", "SELL OPPORTUNITY | RSI: " + DoubleToString(RSI[i], 2) + " | MA: " + DoubleToString(MA[i], 2));
        }

      // --- Buffer3 (Buy Reversal)
      if(RSI[i] < Oversold && Close[i] > MA[i])
        {
         Buffer3[i] = Low[i] - 10 * myPoint;
         myAlert("indicator", "BUY REVERSAL | RSI: " + DoubleToString(RSI[i], 2) + " | MA: " + DoubleToString(MA[i], 2));
        }

      // --- Buffer4 (Sell Reversal)
      if(RSI[i] > Overbought && Close[i] < MA[i])
        {
         Buffer4[i] = High[i] + 10 * myPoint;
         myAlert("indicator", "SELL REVERSAL | RSI: " + DoubleToString(RSI[i], 2) + " | MA: " + DoubleToString(MA[i], 2));
        }

      // --- Buffer5 (Buy Trend)
      if(MA5[i] > MA6[i])
        {
         Buffer5[i] = Low[i] - 15 * myPoint;
         //Disabled myAlert from Buffer 5
         // myAlert("indicator", "BUY TREND | MA Fast: " + DoubleToString(MA5[i], 2) + " | MA Slow: " + DoubleToString(MA6[i], 2));
        }

      // --- Buffer6 (Sell Trend)
      if(MA5[i] < MA6[i])
        {
         Buffer6[i] = High[i] + 15 * myPoint;
         // Disabled myAlert from Buffer 6
         // myAlert("indicator", "SELL TREND | MA Fast: " + DoubleToString(MA5[i], 2) + " | MA Slow: " + DoubleToString(MA6[i], 2));
        }
     }
   return(rates_total);
  }
```

### Integrating the social network (e.g. Telegram and WhatsApp )

This article section will show the integration of Telegram and WhatsApp on MetaTrader 5 specifically for our Trend Constraint indicator. This process significantly boosts MetaTrader 5's capabilities by offering real-time, secure, and convenient notifications. These integrations enhance the efficiency, responsiveness, and effectiveness of trading activities, serving as valuable tools for traders today. With a vast user base on these platforms, sharing signals with these communities can be beneficial. Let's delve into how we can transmit the signals generated by our indicator to social media platforms. I have conducted research and tested this to confirm its functionality.

Requirements:

- latest [MetaTrader5](https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=www.mql5.com&utm_campaign=0685.mql5.chats.promo "Download MetaTrader 5") platform desktop.
- Verified Whatsapp  and Telegram accounts (download from playstore for android and app store for IOS).
- Web browser e.g. [Google Chrome](https://www.mql5.com/go?link=https://www.google.com/chrome/?brand=JJTC%26gad_source=1%26gclid=CjwKCAjw65-zBhBkEiwAjrqRMJDJawXPv8fxxHUVtQdu53ojiIt6XPcMAtXpSmDImq2C-laQTu93DRoCjD0QAvD_BwE%26gclsrc=aw.ds "Download Chrome").
- Server for hosting middleware scripts.

### Telegram Integration

Step 1: Create a Telegram Bot

> Create the Bot:

> > - Open Telegram and search for the "BotFather" bot.
> > - Start a chat with BotFather and use the _/newbot_ command to create a new bot.
> > - Follow the prompts to set a name and username for your bot.
> > - BotFather will provide a token, which you will use to interact with the Telegram API.

> Get Chat ID:

> > - Add your bot to a Telegram group or start a chat with it.
> > - Use the following URL in your browser to get updates and find your chat ID: _https://api.telegram.org/bot<YourBotToken>/getUpdates_
> > - Send a message in the chat and check the URL again to find the chat ID.

For a more comprehensive guide on getting a Chat ID visit [Github](https://www.mql5.com/go?link=https://gist.github.com/nafiesl/4ad622f344cd1dc3bb1ecbe468ff9f8a "Go on github for a guide")

Step 2: Create a Middleware Script

> You need a script to send messages via Telegram's API.

To create  a middleware script we are going to use Python language according to the research that I did. This Python script uses the requests library to send a message to a Telegram chat via a bot. Here is an explanation of each line of the script:

```
import requests
```

Imports the requests library used to make HTTP requests in Python. In this script, it will be used to send a POST request to the Telegram Bot API.

```
def send_telegram_message(chat_id, message, bot_token):
```

Defines a function named _send\_telegram\_message_: This function takes three parameters:

1. _chat\_id_: The unique identifier for the target chat or username of the target channel.
2. _message_: The text message to be sent.
3. _bot\_token_: The token for the Telegram bot, which is provided by _@BotFather_ when creating the bot.

```
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
```

This line constructs the URL for the Telegram Bot API method _sendMessage_ using the provided _bot\_token._

```
    payload = {
        'chat_id': chat_id,
        'text': message
    }
```

Creates the payload for the POST request. This dictionary contains the parameters required by the _sendMessage_ method:

> - _chat\_id_: The ID of the chat where the message will be sent.
> - _text_: The content of the message.

```
    response = requests.post(url, data=payload)
```

This line sends the HTTP POST request to the constructed URL with the payload data using the _requests.post_ method. The response from the API is stored in the response variable.

```
    return response.json()
```

This line converts the response to JSON format and returns it. This JSON response typically contains information about the sent message, including message ID, sender, chat details, and more.

Combining all the pieces of code together here is what we get:

```
import requests

def send_telegram_message(chat_id, message, bot_token):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, data=payload)
    return response.json()
```

Step 3: Configure MetaTrader 5 to Use the Middleware

> Create an MQL5 script to send alerts via HTTP requests to your middleware script

Lets go through the breakdown of the code and gain an understanding each line:

```
void SendTelegramMessage(string chat_id, string message, string bot_token)
```

Defines a function named _SendTelegramMessage._

Parameters within the function:

- _chat\_id_: A string representing the Telegram chat ID where the message will be sent.
- _message:_ A string containing the message to be sent.
- _bot\_token:_ A string representing the bot token required for authorization with the Telegram API.

Function Body opening brace:

```
{
    string url = "http://your-server-url/send_telegram_message";
```

Assigns the server endpoint URL to the variable url. This URL is the address of the server that handles forwarding the message to Telegram.

```
    char postData[];
    StringToCharArray("chat_id=" + chat_id + "&message=" + message + "&bot_token=" + bot_token, postData);
```

_postData_ Array:

- Declaration: Declares an array postData of type char to hold the POST request data.
- String to Char Array Conversion: Converts a concatenated string of the chat\_id, message, and bot\_token parameters into a character array and stores it in postData. The concatenated string forms the body of the POST request in the format required by the server.

```
    char result[];
```

Declares an array _result_ of type _char_ to hold the response from the web request.

```
    int res = WebRequest("POST", url, "", NULL, 0, postData, 0, result, NULL);
```

Web Request:

> - HTTP Method: "POST" indicates the request type.
> - URL: url is the endpoint to send the request.
> - Headers: An empty string "" means no additional headers.
> - Cookies: NULL indicates no cookies.
> - Timeout: 0 specifies no timeout.
> - Post Data: postData is the data to be sent in the POST request.
> - Result: result is where the response will be stored.
> - The function returns an integer res which is the HTTP status code of the response.

```
    if (res != 200)
```

Check Response Code: Compares the response code res to 200 (HTTP OK). If it is not 200, it indicates an error.

Error Handler:

```
    {
        Print("Error sending message: ", GetLastError());
    }
```

Print Error Message: If the response code is not 200, this block prints an error message along with the last error code using _GetLastError()._

Success Handler:

```
    else
    {
        Print("Message sent successfully.");
    }
```

Print Success Message: If the response code is 200, this block prints a success message indicating the message was sent successfully.

This line ends the _SendTelegramMessage_ function with closing brace.

MQL5 Code Sending Telegram Messages:

```
void SendTelegramMessage(string chat_id, string message, string bot_token)
{
    string url = "http://your-server-url/send_telegram_message";
    char postData[];
    StringToCharArray("chat_id=" + chat_id + "&message=" + message + "&bot_token=" + bot_token, postData);

    char result[];
    int res = WebRequest("POST", url, "", NULL, 0, postData, 0, result, NULL);
    if (res != 200)
    {
        Print("Error sending message: ", GetLastError());
    }
    else
    {
        Print("Message sent successfully.");
    }
}
```

This MQL5 script above is designed to send a message to a Telegram chat by making a web request to a server endpoint. The example usage shows how to use the _SendTelegramMessage_ function within the _OnStart_ event, which runs when the script is initiated. We look at WhatsApp integration in next section below and every we have explained the integration of telegram makes sense even when it is now a different social platform. Always remember to put substitute some part of the code with your actual credentials for them to work.

### WhatsApp Integration

Step 1: Register with a Messaging API Provider.

> - Choose a Provider: Twilio is a popular choice for WhatsApp integration.
> - Create an Account: Sign up for Twilio and complete any necessary verification.
> - Obtain API Credentials: Get your API credentials (account SID, auth token) from Twilio.

Step 2: Create a Middleware Script.

Middleware Python Script For WhatsApp Integration:

```
import requests

def send_whatsapp_message(to, message, account_sid, auth_token):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    payload = {
        'From': 'whatsapp:+14155238886',  # Twilio sandbox number
        'To': f'whatsapp:{to}',
        'Body': message
    }
    headers = {
        'Authorization': f'Basic {account_sid}:{auth_token}'
    }
    response = requests.post(url, data=payload, headers=headers)
    return response.json()
```

Step 3: Configure MetaTrader 5 to Use the Middleware

MQL5 code send WhatsApp Notifications:

```
void SendWhatsAppMessage(string to, string message, string account_sid, string auth_token)
{
    string url = "http://your-server-url/send_whatsapp_message";
    char postData[];
    StringToCharArray("to=" + to + "&message=" + message + "&account_sid=" + account_sid + "&auth_token=" + auth_token, postData);

    char result[];
    int res = WebRequest("POST", url, "", NULL, 0, postData, 0, result, NULL);
    if (res != 200)
    {
        Print("Error sending message: ", GetLastError());
    }
    else
    {
        Print("Message sent successfully.");
    }
}
```

### The Power of Virtual Private Service(VPS) on our notification system

A Virtual Private Service operates continuously without downtime, ensuring your MetaTrader 5 platform and notification systems are always active. This is critical for receiving real-time notifications without interruptions. VPS providers offer robust and stable internet connections, reducing the risk of disconnections that can occur with home or office networks.  VPS servers are often located in data centers with high-speed connections to major financial exchanges, reducing latency and improving the speed at which trading alerts and notifications are received and acted upon.  Unlike shared hosting, a VPS provides dedicated CPU, RAM, and storage resources, ensuring consistent performance for running MetaTrader 5 and handling notifications.  A VPS can be accessed remotely from any device with an internet connection, allowing you to manage your MetaTrader 5 platform and receive notifications from anywhere.

### Conclusion

We have successfully configured a robust notification system for our indicator, laying the groundwork for WhatsApp and Telegram integration. This enhancement is pivotal in attracting a large community, as these signals can be instantly shared. Developers can capitalize through marketing these signals to interested traders through these popular platforms, offering a swift and efficient means of transmission directly from the platform to social media channels.

This approach ensures signal accessibility anytime, anywhere with internet access, allowing for easy sharing with just a click. Traders can conveniently choose between Telegram or WhatsApp for signal access. Further integration details will be explored in Part II. Please find the reference files attached below. Feel free to engage in discussions in the comments section.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14963.zip "Download all attachments in the single ZIP archive")

[Telegram\_Middleware.py](https://www.mql5.com/en/articles/download/14963/telegram_middleware.py "Download Telegram_Middleware.py")(0.35 KB)

[Telegram\_mt5.mq5](https://www.mql5.com/en/articles/download/14963/telegram_mt5.mq5 "Download Telegram_mt5.mq5")(1.22 KB)

[WhatsApp\_Middleware.py](https://www.mql5.com/en/articles/download/14963/whatsapp_middleware.py "Download WhatsApp_Middleware.py")(0.55 KB)

[WhatsApp\_mt5.mq5](https://www.mql5.com/en/articles/download/14963/whatsapp_mt5.mq5 "Download WhatsApp_mt5.mq5")(1.26 KB)

[Trend\_Constraint\_V1.04.mq5](https://www.mql5.com/en/articles/download/14963/trend_constraint_v1.04.mq5 "Download Trend_Constraint_V1.04.mq5")(11.73 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/468875)**

![Multibot in MetaTrader (Part II): Improved dynamic template](https://c.mql5.com/2/71/Multibot_in_MetaTrader_Part_II_____LOGO__1.png)[Multibot in MetaTrader (Part II): Improved dynamic template](https://www.mql5.com/en/articles/14251)

Developing the theme of the previous article, I decided to create a more flexible and functional template that has greater capabilities and can be effectively used both in freelancing and as a base for developing multi-currency and multi-period EAs with the ability to integrate with external solutions.

![Data Science and Machine Learning (Part 24): Forex Time series Forecasting Using Regular AI Models](https://c.mql5.com/2/81/Data_Science_and_Machine_Learning_Part_24__LOGO.png)[Data Science and Machine Learning (Part 24): Forex Time series Forecasting Using Regular AI Models](https://www.mql5.com/en/articles/15013)

In the forex markets It is very challenging to predict the future trend without having an idea of the past. Very few machine learning models are capable of making the future predictions by considering past values. In this article, we are going to discuss how we can use classical(Non-time series) Artificial Intelligence models to beat the market

![The base class of population algorithms as the backbone of efficient optimization](https://c.mql5.com/2/71/The_basic_class_of_population_algorithms____LOGO_2_.png)[The base class of population algorithms as the backbone of efficient optimization](https://www.mql5.com/en/articles/14331)

The article represents a unique research attempt to combine a variety of population algorithms into a single class to simplify the application of optimization methods. This approach not only opens up opportunities for the development of new algorithms, including hybrid variants, but also creates a universal basic test stand. This stand becomes a key tool for choosing the optimal algorithm depending on a specific task.

![MQL5 Wizard Techniques you should know (Part 23): CNNs](https://c.mql5.com/2/81/MQL5_Wizard_Techniques_you_should_know_Part_23__LOGO.png)[MQL5 Wizard Techniques you should know (Part 23): CNNs](https://www.mql5.com/en/articles/15101)

Convolutional Neural Networks are another machine learning algorithm that tend to specialize in decomposing multi-dimensioned data sets into key constituent parts. We look at how this is typically achieved and explore a possible application for traders in another MQL5 wizard signal class.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/14963&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068385211508521166)

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