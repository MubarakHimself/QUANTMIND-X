---
title: Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part II)
url: https://www.mql5.com/en/articles/14968
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 5
scraped_at: 2026-01-23T17:32:05.082755
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/14968&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068347961257162851)

MetaTrader 5 / Trading


Contents

- [Introduction](https://www.mql5.com/en/articles/14968#para1)
- [Integration of Telegram in Trend Constraint](https://www.mql5.com/en/articles/14968#para2)

  - [Understanding the Telegram  BOT API](https://www.mql5.com/en/articles/14968#para3)
  - I [nstalling Telegram Python modules on Windows](https://www.mql5.com/en/articles/14968#para4)
  - [Understanding the send\_telegram\_messages python script](https://www.mql5.com/en/articles/14968#para5)
  - [Configuring the Trend Constraint Indicator for telegram](https://www.mql5.com/en/articles/14968#para6)
  - [Final code](https://www.mql5.com/en/articles/14968#para7)
  - [Error Handling](https://www.mql5.com/en/articles/14968#para8)

- [Testing Results](https://www.mql5.com/en/articles/14968#para9)
- [Conclusion](https://www.mql5.com/en/articles/14968#para10)

### Introduction

In the previous article, we briefly introduced the integration process. By splitting it into sub-sections, we aimed to simplify the understanding of the process step by step. A strong foundation is essential, and we hope the groundwork we've laid is solid as we delve deeper into making these integrations function seamlessly, particularly within our Trend Constraint model. Ultimately, our goal is to comfortably receive notifications on both Telegram and WhatsApp by the project's conclusion. This setup will ensure we stay informed of indicators without missing any signals, all while engaging with friends and family on social media. Sharing signals directly on the social platform will be effortless, eliminating the need to switch between applications.

The aim of this article is to guide us through each step comprehensively until we achieve the desired outcomes. With the foundational knowledge acquired in the previous article, everything is now clear. I will elucidate each line of code comprising the integrated program. There are four key components in this project related to telegram integration that need to be kept in mind consistently.

- Telegram bot API.
- Python Script.
- Dedicated server for hosting the script when dealing with Web Requests.
- Our indicator program with configured for telegram.

This is a basic overview of components involved in the main integration process. While I have specifically mentioned Telegram and WhatsApp, it should be noted that other social media platforms can also be integrated, as long as there is a programming language available to facilitate the process. Recognizing the importance of language compatibility in programming, we have incorporated Python and MQL5 in a single project. This highlights the benefit of being familiar with various languages like Python, C++, ONNX, and C#. Such knowledge can greatly assist MQL5 programmers in developing functionalities within the platform and integrating with other social APIs.

The practical integration of Telegram will guide us to the third subsection of Part 5 of the article series where will further integrate WhatsApp, following a structure similar to Telegram but utilizing a Messaging API instead of the bot API. Building upon the foundation we have established, this task will be easier as we now have the principles in mind.

### Integration of Telegram in Trend Constraint

Let's proceed with our significant project: at this point, we are practically continuing from where we paused. Based on my research, there are several methods to achieve this as long as the code logic is understood, but I have opted to concentrate on a method that involves using a Python script and the _ShellExecuteW_ function to facilitate interaction between a MetaTrader 5 indicator and telebot for Telegram. I will delve further into elaborating on these aspects. This approach offers an advantage as it is relatively straightforward for those familiar with Python and MQL5. It does not necessitate extensive modifications to the MQL5 indicator. The only downside is that it requires external dependencies such as Python script and libraries.

The aim is to enhance MetaTrader 5 signal accessibility on Telegram for a wider audience and facilitate easy sharing and forwarding through the implementation of telebots.

In a previous article (Part I), I discussed the integration method that utilizes the [WebRequest](https://www.mql5.com/en/search#!keyword=webrequest&module=mql5_module_documentation) function. However, upon reviewing MQL5 documentation, it was found that this method is not ideal for integrating indicators but functions well with robots.

Caution should be exercised when using shell DLL files as they may pose serious risks when utilized with untrusted applications. It is crucial to fully comprehend and trust the functionality of these systems on your computer in order to prevent attacks and hacks.

### Understanding the Telegram BOT API

I assume you are already an active Telegram user. The project requires more personalization and privacy, including details that I will keep to myself. Follow along to create your Telegram bot with a unique name, like I did using _@Botfather_. I named the bot Trend Constraint, with the username _@trend\_constraint\_bot_. You can create your own bot in a similar way with a unique name. Here is a brief overview of how to get started on Botfather. Follow Botfather's instructions to complete the process. Once done, you will receive a _bot token_ for accessing the bot API. After that, initiate a conversation with the bot, add it to a group, or a channel to start chatting. Each chat has a unique ID that the bot will use to interact with a specific user. This chat is also used to pass signals from MetaTrader 5 indicator to a Telegram user.

![Create a telegram bot using botfather](https://c.mql5.com/2/81/telebot.gif)

After  I had set everything, I visited the bot API using chrome browser. Remember the bot token you received from the BotFather, use the link to API _https://api.telegram.org/bot<your bot-token>/getUpdates ,_ replace the highlighted text with your bot-token for it to work.

Typical example here:  _https://api.telegram.org/bot9004946256:shTUYuq52f8CHLt8BLdYGHYJi2QM6H3donA/getUpdates_

Note: The highlighted API Token was randomly generated for educational purpose. Please use the one provided by @BotFather for your robot. Make sure to make a chat with the bot for the API to display what we want. Also refresh the API browser tab so that new chat updates can appear. Check the pretty-print checkbox on API browser tab to have an easy to follow layout.

The API will display a JSON code for the ongoing communication with the bot. Firstly lets define a JSON(JavaScript Object Notation) as a lightweight data-interchange format that's easy for humans to read and write, and easy for machines to parse and generate. It is commonly used for transmitting data in web applications (e.g., between a server and a client) and for configuring applications and data storage. It is structured by Object and Arrays.

Below is an API JSON that was displayed after sending "hey" message to the bot:

```
{
  "ok": true,
  "result": [\
    {\
      "update_id": 464310132,\
      "message": {\
        "message_id": 12,\
        "from": {\
          "id": 7049213628,\
          "is_bot": false,\
          "first_name": "Clemence",\
          "last_name": "Benjamin",\
          "username": "benjc_trade_advisor",\
          "language_code": "en"\
        },\
        "chat": {\
          "id": 7049213628,\
          "first_name": "Clemence",\
          "last_name": "Benjamin",\
          "username": "benjc_trade_advisor",\
          "type": "private"\
        },\
        "date": 1719044625,\
        "text": "hey"\
      }\
```\
\
Here is a detailed explanation of the JSON code:\
\
Top-Level Structure:\
\
- _ok_: This is a boolean value indicating the success of the API request. In this case, it is true, which means the request was successful.\
- _result_: This is an array containing one or more update objects. In this example, there is one update object.\
\
Inside the _result_ Array\
\
Each element in the result array represents an update. Here, we have a single update:\
\
- _update\_id_: This is a unique identifier for the update. It helps to keep track of updates and ensure that none are missed or processed multiple times. In this case, the update\_id is\
- 464310132.\
\
The _message_ Object\
\
This object contains information about the message that triggered the update:\
\
- _message\_id_: This is a unique identifier for the message within the chat. Here, it is 12.\
- _from_: This object contains information about the sender of the message:\
- _id_: The unique identifier for the user who sent the message. Here, it is 7049213628 .\
- _is\_bot_: This boolean value indicates whether the sender is a bot. In this case, it is false, meaning the sender is a human.\
- _first\_name_: The first name of the sender, which is Clemence.\
- _last\_name_: The last name of the sender, which is Benjamin.\
- _language\_code_: The language code representing the sender's language settings. Here, it is en for English.\
- _chat_: This object contains information about the chat where the message was sent.\
- _id_: The unique identifier for the chat. Since this is a private chat, it matches the user's ID ( 7049213628 ).\
- _first\_name:_ The first name of the chat participant, which is Clemence.\
- _last\_name_: The last name of the chat participant, which is Benjamin.\
- _type_: The type of chat. Here, it is private, indicating a one-on-one chat between the user and the bot.\
- _date_: The date and time when the message was sent, represented as a Unix timestamp (seconds since January 1, 1970). In this case, the timestamp is 1719044625.\
- _text_: The actual text content of the message, which is "hey".\
\
I decided to put aside the chat section of the JSON so that we can easily focus on the most important part which is the chat ID that we are going to need in our indicator program. Check the JSON snippet below:\
\
```\
"chat": {\
          "id": 7049213628,\
          "first_name": "Clemence",\
          "last_name": "Benjamin",\
          "username": "benjc_trade_advisor",\
          "type": "private"\
        }\
```\
\
The " _chat"_ object provides detailed information about the chat where the message was sent, including the unique identifier for the chat, the participant's first and last names, and the type of chat. In this case, it specifies a private chat involving a user named Clemence Benjamin. Let us look at a detailed explanation of the " _chat"_ Object:\
\
i _d:_\
\
- Description: This is the unique identifier for the chat.\
- Value: 7049213628\
- Significance: In the context of private chats, this ID usually matches the user ID of the participant involved in the chat.\
\
_first\_name:_\
\
- Description: The first name of the participant in the chat.\
- Value: Clemence\
- Significance: This helps identify the user by their first name within the chat.\
\
_last\_name:_\
\
- Description: The last name of the participant in the chat.\
- Value: Benjamin\
- Significance: This complements the first name to fully identify the user within the chat.\
\
_"username"_\
\
- Description: It is the key,\
- Value: "benjc\_trade\_advisor"\
- Significance: This line indicates that the username associated with the object (such as a user, bot, or chat) is "benjc\_trade\_advisor". This username is typically used to identify the entity in a recognizable format within applications or systems that use the JSON data\
\
_type:_\
\
- Description: The type of chat.\
- Value: private\
- Significance: Indicates that this chat is a one-on-one conversation between the user and the bot (as opposed to a group chat or a channel).\
\
Summary:\
\
The aim of the aforementioned section was to develop a working telegram bot and acquire the bot token and chat ID, both crucial elements for the main project. We delved into the API for a deeper understanding of each component. Having the bot token and Chat ID enables us to move forward with the integration process, where we will also explore various programming languages.\
\
### Installing Telegram Python modules on Windows\
\
We need to have python installed in your computer. Make sure your computer is connected with internet access. You can download it from [Python.org](https://www.mql5.com/go?link=https://www.python.org/ "Download Python"). I made this guide practically on windows computer. It might be a different approach on Mac, Linux and other frameworks. When done python installation the next step is to get python telegram API modules installed too, these will enable the python scripts  for telegram to run well. Open the _cmd.exe_(window command prompt) and run the code snippet below. Copy the code and paste in windows command prompt. Press Enter on keyboard to let the code initiate downloading, and wait a short while as the module finish installation.\
\
```\
pip install pyTelegramBotAPI\
```\
\
A screenshot part of the  command prompt is shown below.\
\
![Windows command prompt to install python telegram Api modules](https://c.mql5.com/2/81/windowscmd__1.PNG)\
\
When done you can close the window and restart the computer.\
\
After completing this step, your computer is now well-prepared to execute the Python scripts for interacting with the Telegram Bot API. In the following stage, we will examine the code closely to set up our system for the task.\
\
### Understanding the send\_telegram\_messages python script\
\
Lets look at  the construction of the script. And then I will give the final code in one snippet. The file should be named send\_telegram\_message.py\
\
The script begins by importing necessary modules as follows:\
\
- import telebot: Imports the telebot module, which provides the functions needed to interact with the Telegram Bot API.\
- import sys: Imports the sys module, which allows the script to use command-line arguments.\
\
```\
import telebot\
import sys\
```\
\
We move on to declare the API\_TOKEN and the Chat ID:\
\
- API\_TOKEN: This variable stores the bot's API token, which is used to authenticate the bot with the Telegram servers.\
- CHAT\_ID: A unique value of identity  of each chat between either the user and the telebot or, channel and groups.\
\
Make sure the  values for API\_TOKEN and CHAT\_ID are in single quotes as shown below\
\
```\
API_TOKEN = '9004946256:shTUYuq52f8CHLt8BLdYGHYJi2QM6H3donA' #Replace the API TOKEN with your bot tokrn from @BotFather\
CHAT_ID = '7049213628'  #Replace the ID with your actual Chat ID from the telebot API\
```\
\
We need to initialize the _TeleBot_ object with the provided API token, to allow interaction with the Telegram Bot API.\
\
```\
bot = telebot.TeleBot(API_TOKEN)\
```\
\
The next python code snippet defines a function to send a message via Telegram, as error handling exceptions associated with the API and the system.\
\
```\
def send_telegram_message(message):\
    try:\
        bot.send_message(CHAT_ID, message)\
        print("Message sent successfully!")\
    except telebot.apihelper.ApiTelegramException as e:\
        print(f"Failed to send message: {e}")\
    except Exception as e:\
        print(f"An error occurred: {e}")\
```\
\
The final part of the code  is a condition that ensures this block of code runs only if the script is executed directly, not if it's imported as a module. It retrieves the message from command-line arguments or sets a default message if no arguments are provided.\
\
```\
if __name__ == "__main__":\
    message = sys.argv[1] if len(sys.argv) > 1 else "Test message"\
    send_telegram_message(message)\
```\
\
Summing up everything  we have our final code, save the file  as send\_telegram\_message.py and in your python scripts folder. I found the scripts folder access working fine.\
\
```\
import telebot\
import sys\
\
API_TOKEN = '9004946256:shTUYuq52f8CHLt8BLdYGHYJi2QM6H3donA'#Replace with your API_TOKEN given by BotFather\
CHAT_ID = '7049213628' #Replace with your CHAT_ID\
\
bot = telebot.TeleBot(API_TOKEN)\
\
def send_telegram_message(message):\
    try:\
        bot.send_message(CHAT_ID, message)\
        print("Message sent successfully!")\
    except telebot.apihelper.ApiTelegramException as e:\
        print(f"Failed to send message: {e}")\
    except Exception as e:\
        print(f"An error occurred: {e}")\
\
if __name__ == "__main__":\
   message = sys.argv[1] if len(sys.argv) > 1 else "Test message"\
    send_telegram_message(message)\
```\
\
The next major step is to set up the MQL5 Indicator to call the Python Script.\
\
### Configuring the Trend Constraint Indicator for telegram\
\
Here we have to modify the _myAler_ t function in the MQL5 indicator to call the Python script using the _ShellExecuteW_ function. This function will execute the Python script and pass the alert message as an argument.\
\
Here is how we modify it in Trend Constraint, I have included the two code snippets before modification and after modification:\
\
Before Modification:\
\
```\
void myAlert(string type, string message)\
  {\
   if(type == "print")\
      Print(message);\
   else if(type == "error")\
     {\
      Print(type+" | Trend Constraint V1.05 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);\
     }\
   else if(type == "order")\
     {\
     }\
   else if(type == "modify")\
     {\
     }\
   else if(type == "indicator")\
     {\
      if(Audible_Alerts) Alert(type+" | Trend Constraint V1.05 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);\
      if(Push_Notifications) SendNotification(type+" | Trend Constraint V1.05 @ "+Symbol()+","+IntegerToString(Period())+" | "+message);\
     }\
  }\
```\
\
After modification:\
\
```\
//--- ShellExecuteW declaration ----------------------------------------------\
#import "shell32.dll"\
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);\
#import\
\
datetime last_alert_time;\
input int alert_cooldown_seconds = 60; // Cooldown period in seconds, this helps to avoid instant continuous alerting depending on indicator conditions\
\
//Modify the myAlert Function for telegram notification\
void myAlert(string type, string message) {\
    datetime current_time = TimeCurrent();\
    if (current_time - last_alert_time < alert_cooldown_seconds) {\
        // Skip alert if within cooldown period\
        return;\
    }\
\
    last_alert_time = current_time;\
    string full_message = type + " | Trend Constraint V1.04 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message;\
    if (type == "print") {\
        Print(message);\
    } else if (type == "error") {\
        Print(type + " | Trend Constraint V1.04 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message);\
    } else if (type == "order") {\
        // Add order alert handling if needed\
    } else if (type == "modify") {\
        // Add modify alert handling if needed\
    } else if (type == "indicator") {\
        if (Audible_Alerts) {\
            Alert(full_message);\
        }\
        if (Push_Notifications) {\
            SendNotification(full_message);\
        }\
\
        // Send to Telegram\
        string python_path = "C:\\Users\\Pro_tech\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";\
        string script_path = "C:\\Users\\Pro_tech\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_telegram_message.py";\
        string command = python_path + " \"" + script_path + "\" \"" + full_message + "\"";\
\
        Print("Executing command to send Telegram message: ", command);\
\
        // Use cmd.exe to execute the command and then wait for 5 seconds\
        string final_command = "/c " + command + " && timeout 5";\
        int result = ShellExecuteW(0, "open", "cmd.exe", final_command, NULL, 1);\
        if (result <= 32) {\
            int error_code = GetLastError();\
            Print("Failed to execute Python script. Error code: ", error_code);\
        } else {\
            Print("Successfully executed Python script. Result code: ", result);\
        }\
    }\
}\
\
//--- End of Telegram Integration functions ---------------------------------------\
```\
\
I will briefly explain what the modification is doing is doing.\
\
First we need to import the _ShellExecuteW_ function from the Windows _shell32.dll_ library, allowing the MQL5 program to execute external commands, in this case it is running the _send\_telegram\_message.py_ script. The program will not work without declaring the _ShellExecuteW_ function. We also imployed the cooldown to caters to resttrict continuous instant execution of the cmd.exe due to some wrongly configured indicator conditions. In my case as, I mentioned in the previous article, there was an alert condition in Buffer 5 and 6 of the Trend Constraint V1.04 which caused multiple signal alerts at a short time. The result was worse when I integrated the Telegram feature , the cmd.exe was launched several times at an instant and the computer froze. To alleviate I had to only allow the indicator to draw without _myAlert()_. In other words I muted it by turning it into comments as you know comments are not executed in the program.\
\
```\
//--- ShellExecuteW declaration ----------------------------------------------\
#import "shell32.dll"\
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);\
#import\
```\
\
The other crucial part of the code is:\
\
```\
// Send to Telegram\
string python_path = "C:\\Users\\Pro_tech\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";\
string script_path = "C:\\Users\\Pro_tech\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_telegram_message.py";\
string command = python_path + " \"" + script_path + "\" \"" + full_message + "\"";\
```\
\
The code above constructs a command to run a Python script from within a MetaTrader 5 indicator program, specifying the paths to the Python interpreter and the script, as well as the message to be sent. Please for the highlighted text you need to replace with own text for path depending on your computer. This I gave as an example from my computer.\
\
### Final Code\
\
Having explained everything and integrated successfully, we now have a new feature  and we upgrade to Trend Constraint V1.05\
\
```\
//+------------------------------------------------------------------+\
//|                                       Trend Constraint V1.05.mq5 |\
//|                                Copyright 2024, Clemence Benjamin |\
//|                                             https://www.mql5.com |\
//+------------------------------------------------------------------+\
#property copyright "Copyright 2024, Clemence Benjamin"\
#property link      "https://www.mql5.com"\
#property version   "1.05"\
#property description "A model that seeks to produce sell signals when D1 candle is Bearish only and buy signals when it is Bullish"\
\
//--- indicator settings\
#property indicator_chart_window\
#property indicator_buffers 6\
#property indicator_plots 6\
\
#property indicator_type1 DRAW_ARROW\
#property indicator_width1 5\
#property indicator_color1 0xFF3C00\
#property indicator_label1 "Buy"\
\
#property indicator_type2 DRAW_ARROW\
#property indicator_width2 5\
#property indicator_color2 0x0000FF\
#property indicator_label2 "Sell"\
\
#property indicator_type3 DRAW_ARROW\
#property indicator_width3 2\
#property indicator_color3 0xE8351A\
#property indicator_label3 "Buy Reversal"\
\
#property indicator_type4 DRAW_ARROW\
#property indicator_width4 2\
#property indicator_color4 0x1A1AE8\
#property indicator_label4 "Sell Reversal"\
\
#property indicator_type5 DRAW_LINE\
#property indicator_style5 STYLE_SOLID\
#property indicator_width5 2\
#property indicator_color5 0xFFAA00\
#property indicator_label5 "Buy"\
\
#property indicator_type6 DRAW_LINE\
#property indicator_style6 STYLE_SOLID\
#property indicator_width6 2\
#property indicator_color6 0x0000FF\
#property indicator_label6 "Sell"\
\
#define PLOT_MAXIMUM_BARS_BACK 5000\
#define OMIT_OLDEST_BARS 50\
\
//--- indicator buffers\
double Buffer1[];\
double Buffer2[];\
double Buffer3[];\
double Buffer4[];\
double Buffer5[];\
double Buffer6[];\
\
input double Oversold = 30;\
input double Overbought = 70;\
input int Slow_MA_period = 200;\
input int Fast_MA_period = 100;\
datetime time_alert; //used when sending alert\
input bool Audible_Alerts = true;\
input bool Push_Notifications = true;\
double myPoint; //initialized in OnInit\
int RSI_handle;\
double RSI[];\
double Open[];\
double Close[];\
int MA_handle;\
double MA[];\
int MA_handle2;\
double MA2[];\
int MA_handle3;\
double MA3[];\
int MA_handle4;\
double MA4[];\
double Low[];\
double High[];\
int MA_handle5;\
double MA5[];\
int MA_handle6;\
double MA6[];\
int MA_handle7;\
double MA7[];\
\
//--- ShellExecuteW declaration ----------------------------------------------\
#import "shell32.dll"\
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);\
#import\
\
//--- functions for telegram integration -----------------------------------------------\
datetime last_alert_time;\
input int alert_cooldown_seconds = 60; // Cooldown period in seconds\
\
void myAlert(string type, string message) {\
    datetime current_time = TimeCurrent();\
    if (current_time - last_alert_time < alert_cooldown_seconds) {\
        // Skip alert if within cooldown period\
        return;\
    }\
\
    last_alert_time = current_time;\
    string full_message = type + " | Trend Constraint V1.05 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message;\
    if (type == "print") {\
        Print(message);\
    } else if (type == "error") {\
        Print(type + " | Trend Constraint V1.05 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message);\
    } else if (type == "order") {\
        // Add order alert handling if needed\
    } else if (type == "modify") {\
        // Add modify alert handling if needed\
    } else if (type == "indicator") {\
        if (Audible_Alerts) {\
            Alert(full_message);\
        }\
        if (Push_Notifications) {\
            SendNotification(full_message);\
        }\
\
        // Send to Telegram //Remember to replace the storages path with your actual path.\
        string python_path = "C:\\Users\\Pro_tech\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";\
        string script_path = "C:\\Users\\Pro_tech\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_telegram_message.py";\
        string command = python_path + " \"" + script_path + "\" \"" + full_message + "\"";\
\
        // Debugging: Print the command being executed\
        Print("Executing command to send Telegram message: ", command);\
\
        // Use cmd.exe to execute the command and then wait for 5 seconds\
        string final_command = "/c " + command + " && timeout 5";\
        int result = ShellExecuteW(0, "open", "cmd.exe", final_command, NULL, 1);\
        if (result <= 32) {\
            int error_code = GetLastError();\
            Print("Failed to execute Python script. Error code: ", error_code);\
        } else {\
            Print("Successfully executed Python script. Result code: ", result);\
        }\
    }\
}\
\
//+------------------------------------------------------------------+\
//| Custom indicator initialization function                         |\
//+------------------------------------------------------------------+\
int OnInit()\
  {\
   SetIndexBuffer(0, Buffer1);\
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);\
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));\
   PlotIndexSetInteger(0, PLOT_ARROW, 241);\
   SetIndexBuffer(1, Buffer2);\
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);\
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));\
   PlotIndexSetInteger(1, PLOT_ARROW, 242);\
   SetIndexBuffer(2, Buffer3);\
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);\
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));\
   PlotIndexSetInteger(2, PLOT_ARROW, 236);\
   SetIndexBuffer(3, Buffer4);\
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, EMPTY_VALUE);\
   PlotIndexSetInteger(3, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));\
   PlotIndexSetInteger(3, PLOT_ARROW, 238);\
   SetIndexBuffer(4, Buffer5);\
   PlotIndexSetDouble(4, PLOT_EMPTY_VALUE, EMPTY_VALUE);\
   PlotIndexSetInteger(4, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));\
   SetIndexBuffer(5, Buffer6);\
   PlotIndexSetDouble(5, PLOT_EMPTY_VALUE, EMPTY_VALUE);\
   PlotIndexSetInteger(5, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));\
   //initialize myPoint\
   myPoint = Point();\
   if(Digits() == 5 || Digits() == 3)\
     {\
      myPoint *= 10;\
     }\
   RSI_handle = iRSI(NULL, PERIOD_CURRENT, 14, PRICE_CLOSE);\
   if(RSI_handle < 0)\
     {\
      Print("The creation of iRSI has failed: RSI_handle=", INVALID_HANDLE);\
      Print("Runtime error = ", GetLastError());\
      return(INIT_FAILED);\
     }\
\
   MA_handle = iMA(NULL, PERIOD_CURRENT, 7, 0, MODE_SMMA, PRICE_CLOSE);\
   if(MA_handle < 0)\
     {\
      Print("The creation of iMA has failed: MA_handle=", INVALID_HANDLE);\
      Print("Runtime error = ", GetLastError());\
      return(INIT_FAILED);\
     }\
\
   MA_handle2 = iMA(NULL, PERIOD_CURRENT, 400, 0, MODE_SMA, PRICE_CLOSE);\
   if(MA_handle2 < 0)\
     {\
      Print("The creation of iMA has failed: MA_handle2=", INVALID_HANDLE);\
      Print("Runtime error = ", GetLastError());\
      return(INIT_FAILED);\
     }\
\
   MA_handle3 = iMA(NULL, PERIOD_CURRENT, 100, 0, MODE_EMA, PRICE_CLOSE);\
   if(MA_handle3 < 0)\
     {\
      Print("The creation of iMA has failed: MA_handle3=", INVALID_HANDLE);\
      Print("Runtime error = ", GetLastError());\
      return(INIT_FAILED);\
     }\
\
   MA_handle4 = iMA(NULL, PERIOD_CURRENT, 200, 0, MODE_SMA, PRICE_CLOSE);\
   if(MA_handle4 < 0)\
     {\
      Print("The creation of iMA has failed: MA_handle4=", INVALID_HANDLE);\
      Print("Runtime error = ", GetLastError());\
      return(INIT_FAILED);\
     }\
\
   MA_handle5 = iMA(NULL, PERIOD_CURRENT, Fast_MA_period, 0, MODE_SMA, PRICE_CLOSE);\
   if(MA_handle5 < 0)\
     {\
      Print("The creation of iMA has failed: MA_handle5=", INVALID_HANDLE);\
      Print("Runtime error = ", GetLastError());\
      return(INIT_FAILED);\
     }\
\
   MA_handle6 = iMA(NULL, PERIOD_CURRENT, Slow_MA_period, 0, MODE_SMA, PRICE_CLOSE);\
   if(MA_handle6 < 0)\
     {\
      Print("The creation of iMA has failed: MA_handle6=", INVALID_HANDLE);\
      Print("Runtime error = ", GetLastError());\
      return(INIT_FAILED);\
     }\
\
   MA_handle7 = iMA(NULL, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE);\
   if(MA_handle7 < 0)\
     {\
      Print("The creation of iMA has failed: MA_handle7=", INVALID_HANDLE);\
      Print("Runtime error = ", GetLastError());\
      return(INIT_FAILED);\
     }\
\
   return(INIT_SUCCEEDED);\
  }\
\
//+------------------------------------------------------------------+\
//| Custom indicator iteration function                              |\
//+------------------------------------------------------------------+\
int OnCalculate(const int rates_total,\
                const int prev_calculated,\
                const datetime& time[],\
                const double& open[],\
                const double& high[],\
                const double& low[],\
                const double& close[],\
                const long& tick_volume[],\
                const long& volume[],\
                const int& spread[])\
  {\
   int limit = rates_total - prev_calculated;\
   //--- counting from 0 to rates_total\
   ArraySetAsSeries(Buffer1, true);\
   ArraySetAsSeries(Buffer2, true);\
   ArraySetAsSeries(Buffer3, true);\
   ArraySetAsSeries(Buffer4, true);\
   ArraySetAsSeries(Buffer5, true);\
   ArraySetAsSeries(Buffer6, true);\
   //--- initial zero\
   if(prev_calculated < 1)\
     {\
      ArrayInitialize(Buffer1, EMPTY_VALUE);\
      ArrayInitialize(Buffer2, EMPTY_VALUE);\
      ArrayInitialize(Buffer3, EMPTY_VALUE);\
      ArrayInitialize(Buffer4, EMPTY_VALUE);\
      ArrayInitialize(Buffer5, EMPTY_VALUE);\
      ArrayInitialize(Buffer6, EMPTY_VALUE);\
     }\
   else\
      limit++;\
   datetime Time[];\
\
   datetime TimeShift[];\
   if(CopyTime(Symbol(), PERIOD_CURRENT, 0, rates_total, TimeShift) <= 0) return(rates_total);\
   ArraySetAsSeries(TimeShift, true);\
   int barshift_M1[];\
   ArrayResize(barshift_M1, rates_total);\
   int barshift_D1[];\
   ArrayResize(barshift_D1, rates_total);\
   for(int i = 0; i < rates_total; i++)\
     {\
      barshift_M1[i] = iBarShift(Symbol(), PERIOD_M1, TimeShift[i]);\
      barshift_D1[i] = iBarShift(Symbol(), PERIOD_D1, TimeShift[i]);\
   }\
   if(BarsCalculated(RSI_handle) <= 0)\
      return(0);\
   if(CopyBuffer(RSI_handle, 0, 0, rates_total, RSI) <= 0) return(rates_total);\
   ArraySetAsSeries(RSI, true);\
   if(CopyOpen(Symbol(), PERIOD_M1, 0, rates_total, Open) <= 0) return(rates_total);\
   ArraySetAsSeries(Open, true);\
   if(CopyClose(Symbol(), PERIOD_D1, 0, rates_total, Close) <= 0) return(rates_total);\
   ArraySetAsSeries(Close, true);\
   if(BarsCalculated(MA_handle) <= 0)\
      return(0);\
   if(CopyBuffer(MA_handle, 0, 0, rates_total, MA) <= 0) return(rates_total);\
   ArraySetAsSeries(MA, true);\
   if(BarsCalculated(MA_handle2) <= 0)\
      return(0);\
   if(CopyBuffer(MA_handle2, 0, 0, rates_total, MA2) <= 0) return(rates_total);\
   ArraySetAsSeries(MA2, true);\
   if(BarsCalculated(MA_handle3) <= 0)\
      return(0);\
   if(CopyBuffer(MA_handle3, 0, 0, rates_total, MA3) <= 0) return(rates_total);\
   ArraySetAsSeries(MA3, true);\
   if(BarsCalculated(MA_handle4) <= 0)\
      return(0);\
   if(CopyBuffer(MA_handle4, 0, 0, rates_total, MA4) <= 0) return(rates_total);\
   ArraySetAsSeries(MA4, true);\
   if(CopyLow(Symbol(), PERIOD_CURRENT, 0, rates_total, Low) <= 0) return(rates_total);\
   ArraySetAsSeries(Low, true);\
   if(CopyHigh(Symbol(), PERIOD_CURRENT, 0, rates_total, High) <= 0) return(rates_total);\
   ArraySetAsSeries(High, true);\
   if(BarsCalculated(MA_handle5) <= 0)\
      return(0);\
   if(CopyBuffer(MA_handle5, 0, 0, rates_total, MA5) <= 0) return(rates_total);\
   ArraySetAsSeries(MA5, true);\
   if(BarsCalculated(MA_handle6) <= 0)\
      return(0);\
   if(CopyBuffer(MA_handle6, 0, 0, rates_total, MA6) <= 0) return(rates_total);\
   ArraySetAsSeries(MA6, true);\
   if(BarsCalculated(MA_handle7) <= 0)\
      return(0);\
   if(CopyBuffer(MA_handle7, 0, 0, rates_total, MA7) <= 0) return(rates_total);\
   ArraySetAsSeries(MA7, true);\
   if(CopyTime(Symbol(), Period(), 0, rates_total, Time) <= 0) return(rates_total);\
   ArraySetAsSeries(Time, true);\
   //--- main loop\
   for(int i = limit-1; i >= 0; i--)\
     {\
      if (i >= MathMin(PLOT_MAXIMUM_BARS_BACK-1, rates_total-1-OMIT_OLDEST_BARS)) continue; //omit some old rates to prevent "Array out of range" or slow calculation\
\
      if(barshift_M1[i] < 0 || barshift_M1[i] >= rates_total) continue;\
      if(barshift_D1[i] < 0 || barshift_D1[i] >= rates_total) continue;\
\
      //Indicator Buffer 1\
      if(RSI[i] < Oversold\
      && RSI[i+1] > Oversold //Relative Strength Index crosses below fixed value\
      && Open[barshift_M1[i]] >= Close[1+barshift_D1[i]] //Candlestick Open >= Candlestick Close\
      && MA[i] > MA2[i] //Moving Average > Moving Average\
      && MA3[i] > MA4[i] //Moving Average > Moving Average\
      )\
        {\
         Buffer1[i] = Low[1+i]; //Set indicator value at Candlestick Low\
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy"); //Alert on next bar open\
         time_alert = Time[1];\
        }\
      else\
        {\
         Buffer1[i] = EMPTY_VALUE;\
        }\
      //Indicator Buffer 2\
      if(RSI[i] > Overbought\
      && RSI[i+1] < Overbought //Relative Strength Index crosses above fixed value\
      && Open[barshift_M1[i]] <= Close[1+barshift_D1[i]] //Candlestick Open <= Candlestick Close\
      && MA[i] < MA2[i] //Moving Average < Moving Average\
      && MA3[i] < MA4[i] //Moving Average < Moving Average\
      )\
        {\
         Buffer2[i] = High[1+i]; //Set indicator value at Candlestick High\
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell"); //Alert on next bar open\
         time_alert = Time[1];\
        }\
      else\
        {\
         Buffer2[i] = EMPTY_VALUE;\
        }\
      //Indicator Buffer 3\
      if(MA5[i] > MA6[i]\
      && MA5[i+1] < MA6[i+1] //Moving Average crosses above Moving Average\
      )\
        {\
         Buffer3[i] = Low[i]; //Set indicator value at Candlestick Low\
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy Reversal"); //Alert on next bar open\
         time_alert = Time[1];\
        }\
      else\
        {\
         Buffer3[i] = EMPTY_VALUE;\
        }\
      //Indicator Buffer 4\
      if(MA5[i] < MA6[i]\
      && MA5[i+1] > MA6[i+1] //Moving Average crosses below Moving Average\
      )\
        {\
         Buffer4[i] = High[i]; //Set indicator value at Candlestick High\
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell Reversal"); //Alert on next bar open\
         time_alert = Time[1];\
        }\
      else\
        {\
         Buffer4[i] = EMPTY_VALUE;\
        }\
      //Indicator Buffer 5, Alert muted by turning it into a comment\
      if(MA3[i] > MA7[i] //Moving Average > Moving Average\
      )\
        {\
         Buffer5[i] = MA3[i]; //Set indicator value at Moving Average\
         //if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy"); //Alert on next bar open\
         //time_alert = Time[1];\
        }\
      else\
        {\
         Buffer5[i] = EMPTY_VALUE;\
        }\
      //Indicator Buffer 6, Alert muted by turning it into a comment\
      if(MA3[i] < MA7[i] //Moving Average < Moving Average\
      )\
        {\
         Buffer6[i] = MA3[i]; //Set indicator value at Moving Average\
         //if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell"); //Alert on next bar open\
         //time_alert = Time[1];\
        }\
      else\
        {\
         Buffer6[i] = EMPTY_VALUE;\
        }\
     }\
   return(rates_total);\
  }\
//+------------------------------------------------------------------+\
```\
\
### Error Handling\
\
```\
Failed to send message: A request to the Telegram API was unsuccessful. Error code: 401. Description: Unauthorized\
```\
\
According to my testing the error code above was due an dysfunctional API\_TOKEN such as the one we used as an example earlier. You need to use a working  API\_TOKEN value. I deleted most of the errors to arrive at a clean working code for this guide. However, you may make mistakes when editing or changing your code, so you need to carefully check every step you take.\
\
### Testing Results\
\
After adding the indicator to the chart, I enabled the allow DLL option to allow our indicator to execute scripts via command prompt. The animated image demonstrates how the indicator is added and its appearance on the chart.\
\
> > ![lauching the indicator](https://c.mql5.com/2/81/Launch_trend_constraint.gif)\
\
You can test if the script is working by running the file in its path via command prompt see the image below. With Command Prompt open to the folder with your script type in _python send\_telegram\_message.py ._ A message send successful, response show the script is working and the test message is also forwarded to the chat.\
\
![test message cmd](https://c.mql5.com/2/81/message_sent_success.PNG)\
\
The result image below show the beginning of a conversation with the bot which enabled us to get the Chat ID in the bot API. It also shows the incoming signal sent by the bot from Trend Constraint V1.05. The Signals came in immediately as they were generated in the MetaTrader 5 platform.\
\
![Telegram Chat With Trend  Contraint telebot](https://c.mql5.com/2/81/telegram_chat.png)\
\
### Conclusion\
\
We have successfully integrated telegram into our model. Trend Constraint V1.05 has advanced significantly, now able to pass signals internally and externally, benefiting traders worldwide with Telegram access. The signal transmission is fast, with no delays due to the algorithm's efficient execution. The system is dedicated to a specific indicator within the platform, ensuring no interference with other functions. Signals are securely transmitted directly to a specified ID. These systems can be hosted on Virtual Private Server for continuous operation, providing users with a stable signal supply. Such projects may encounter errors during development, but I am pleased to have resolved them successfully.\
\
I hope this project has inspired you in some way. If you are working on a project and have encountered challenges with this type of integration, please feel free to share your thoughts in the discussion section below. Attached are the source files that you can modify for your projects and explore some ideas with the help of the comments provided for educational purposes. Next, we plan to integrate another popular social platform, WhatsApp.\
\
| Attachment | Description |\
| --- | --- |\
| send\_telegram\_message.py | The script to let the indicator pass notifications to telegram it contains API\_Token and Chat ID |\
| Trend Constraint V1.05.mq5 | The main MQL5 indicator program source code |\
| Telebot\_API.txt | Telegram Bot API structure |\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/14968.zip "Download all attachments in the single ZIP archive")\
\
[send\_telegram\_message.py](https://www.mql5.com/en/articles/download/14968/send_telegram_message.py "Download send_telegram_message.py")(0.66 KB)\
\
[Trend\_Constraint\_V1.05.mq5](https://www.mql5.com/en/articles/download/14968/trend_constraint_v1.05.mq5 "Download Trend_Constraint_V1.05.mq5")(16.05 KB)\
\
[Telebot\_API.txt](https://www.mql5.com/en/articles/download/14968/telebot_api.txt "Download Telebot_API.txt")(0.61 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)\
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)\
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)\
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)\
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)\
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)\
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)\
\
**[Go to discussion](https://www.mql5.com/en/forum/469209)**\
\
![Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status](https://c.mql5.com/2/71/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status](https://www.mql5.com/en/articles/14246)\
\
Having started developing a multi-currency EA, we have already achieved some results and managed to carry out several code improvement iterations. However, our EA was unable to work with pending orders and resume operation after the terminal restart. Let's add these features.\
\
![Integrate Your Own LLM into EA (Part 4): Training Your Own LLM with GPU](https://c.mql5.com/2/82/Integrate_Your_Own_LLM_into_EA_Part_4____LOGO.png)[Integrate Your Own LLM into EA (Part 4): Training Your Own LLM with GPU](https://www.mql5.com/en/articles/13498)\
\
With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.\
\
![Population optimization algorithms: Resistance to getting stuck in local extrema (Part I)](https://c.mql5.com/2/72/Population_optimization_algorithms__Resistance_to_getting_stuck_in_local_extrema__LOGO.png)[Population optimization algorithms: Resistance to getting stuck in local extrema (Part I)](https://www.mql5.com/en/articles/14352)\
\
This article presents a unique experiment that aims to examine the behavior of population optimization algorithms in the context of their ability to efficiently escape local minima when population diversity is low and reach global maxima. Working in this direction will provide further insight into which specific algorithms can successfully continue their search using coordinates set by the user as a starting point, and what factors influence their success.\
\
![MQL5 Wizard Techniques you should know (Part 24): Moving Averages](https://c.mql5.com/2/82/MQL5_Wizard_Techniques_you_should_know_Part_24__LOGO.png)[MQL5 Wizard Techniques you should know (Part 24): Moving Averages](https://www.mql5.com/en/articles/15135)\
\
Moving Averages are a very common indicator that are used and understood by most Traders. We explore possible use cases that may not be so common within MQL5 Wizard assembled Expert Advisors.\
\
[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/14968&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068347961257162851)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)