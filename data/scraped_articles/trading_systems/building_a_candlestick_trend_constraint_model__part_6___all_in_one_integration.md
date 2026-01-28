---
title: Building A Candlestick Trend Constraint Model (Part 6): All in one integration
url: https://www.mql5.com/en/articles/15143
categories: Trading Systems, Integration, Indicators
relevance_score: 8
scraped_at: 2026-01-22T17:46:02.684507
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/15143&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049363338225363496)

MetaTrader 5 / Trading systems


1. [Introduction](https://www.mql5.com/en/articles/15143#para1)
2. [Major Integration Sections](https://www.mql5.com/en/articles/15143#para2)
3. [Combining the Integration logic](https://www.mql5.com/en/articles/15143#para3)
4. [The Comment Function](https://www.mql5.com/en/articles/15143#para4)
5. [Testing the effects of the Comment Function](https://www.mql5.com/en/articles/15143#para5)
6. [Conclusion](https://www.mql5.com/en/articles/15143#para6)

### Introduction

Our signal system has consistently demonstrated exceptional performance throughout each stage of its development. The current objective is to merge the existing programs into a single, unified signal system. Remember, the prior two versions of the Trend Constraint indicator each had their specific integrations in [Part 5](https://www.mql5.com/en/articles/14969). This consolidation aims to harness the full power of programming, which significantly reduces human workload by allowing computers to execute complex, repetitive tasks at incredible speeds.

Given that we have two programs with similar logic but distinct features, the integration process involves more than simply copying and pasting source code. Instead, we will strategically retain certain elements that have consistent effects across both programs, ensuring optimal functionality. This process, which I refer to as merging, requires careful consideration and precision.

In this article, we will break down the MQL5 code sections where integration occurs and discuss the key lines that remain global throughout the merging process. This meticulous approach is essential when combining multiple code snippets to create a cohesive and efficient program.

From [Part 5](https://www.mql5.com/en/articles/14969) and its subsections, we had two major integrations to summarize:

1. [Integration of Telegram](https://www.mql5.com/en/articles/14968) with MetaTrader 5 for notifications.
2. [Integration of WhatsApp](https://www.mql5.com/en/articles/14969) with MetaTrader 5 for notifications.

One challenge is that our integration performs tasks in the Command Prompt with the window hidden to avoid interfering with other processes on the computer screen. As a result, there is no confirmation of whether the signals have been successfully sent to the target platforms. We want our system to comment on the chart window for every successful signal broadcast or, at the very least, print it in the platform journal.

Let's delve deeper and discuss further insights in the next sections of this article.

### Our major integration sections

I have extracted the main points of interest from the code specifically for this article's discussion. For more detailed discussions, please revisit [Part 5](https://www.mql5.com/en/articles/14963). Here are the programs we are comparing to facilitate the merging process. Take a look at each program and compare their similarities and differences. This comparison will help us identify the key areas where integration is required, ensuring a seamless and efficient merge.

- Integration of Telegram:

```
/--- ShellExecuteW declaration ----------------------------------------------
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import

//--- functions for telegram integration -----------------------------------------------
datetime last_alert_time;
input int alert_cooldown_seconds = 60; // Cooldown period in seconds

void myAlert(string type, string message) {
    datetime current_time = TimeCurrent();
    if (current_time - last_alert_time < alert_cooldown_seconds) {
        // Skip alert if within cooldown period
        return;
    }

    last_alert_time = current_time;
    string full_message = type + " | Trend Constraint V1.05 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message;
    if (type == "print") {
        Print(message);
    } else if (type == "error") {
        Print(type + " | Trend Constraint V1.05 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message);
    } else if (type == "order") {
        // Add order alert handling if needed
    } else if (type == "modify") {
        // Add modify alert handling if needed
    } else if (type == "indicator") {
        if (Audible_Alerts) {
            Alert(full_message);
        }
        if (Push_Notifications) {
            SendNotification(full_message);
        }

        // Send to Telegram
        string python_path = "C:\\Users\\your_computer_name\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";
        string script_path = "C:\\Users\\your_computer_name\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_telegram_message.py";
        string command = python_path + " \"" + script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed
        Print("Executing command to send Telegram message: ", command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_command = "/c " + command + " && timeout 5";
        int result = ShellExecuteW(0, "open", "cmd.exe", final_command, NULL, 1);
        if (result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute Python script. Error code: ", error_code);
        } else {
            Print("Successfully executed Python script. Result code: ", result);
        }
    }
}
```

- Integration of WhatsApp:

```
//--- ShellExecuteW declaration ----------------------------------------------
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import

//--- global variables ------------------------------------------------------
datetime last_alert_time;
input int alert_cooldown_seconds = 60; // Cooldown period in seconds

//--- myAlert function ------------------------------------------------------
void myAlert(string type, string message) {
    datetime current_time = TimeCurrent();
    if (current_time - last_alert_time < alert_cooldown_seconds) {
        // Skip alert if within cooldown period
        return;
    }

    last_alert_time = current_time;
    string full_message = type + " | Trend Constraint V1.06 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message;
    if (type == "print") {
        Print(message);
    } else if (type == "error") {
        Print(type + " | Trend Constraint V1.06 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message);
    } else if (type == "order") {
        // Add order alert handling if needed
    } else if (type == "modify") {
        // Add modify alert handling if needed
    } else if (type == "indicator" || type == "info") {
        if (Audible_Alerts) {
            Alert(full_message);
        }
        if (Push_Notifications) {
            SendNotification(full_message);
        }

        // Send to WhatsApp
        string python_path = "C:\\Users\\Your_Computer_Name\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";
        string script_path = "C:\\Users\\Your_Computer_Name\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_whatsapp_message.py";
        string command = python_path + " \"" + script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed
        Print("Executing command to send WhatsApp message: ", command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_command = "/c " + command + " && timeout 5";
        int result = ShellExecuteW(0, "open", "cmd.exe", final_command, NULL, 1);
        if (result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute Python script. Error code: ", error_code);
        } else {
            Print("Successfully executed Python script. Result code: ", result);
        }
    }
}
```

Combining the Integration Logic

To create a single program that integrates both [WhatsApp](https://www.mql5.com/en/articles/14969) and [Telegram](https://www.mql5.com/en/articles/14968) using the two provided code snippets, we will combine the logic from each snippet into one cohesive function. Here’s the plan:

1. Combine Global Variables and Declarations: We will consolidate the declarations and global variables.
2. Merge the _[myAlert](https://www.mql5.com/en/docs/common/alert)_ Function: We will extend the _myAlert_ function to handle sending messages to both WhatsApp and Telegram.
3. Adjust the Command Execution Logic: We will ensure that both commands (WhatsApp and Telegram) are executed within the same function.
4. Ensure Cooldown Period is Maintained: We will keep the logic that ensures alerts are not sent too frequently.

To combine declarations and global variables, both snippets had the _ShellExecuteW_ declaration and a cooldown period variable, which are unified at the top of the code to avoid redundancy. The myAlert function is expanded to include logic for both [WhatsApp](https://www.mql5.com/en/articles/14969) and [Telegram](https://www.mql5.com/en/articles/14968) notifications, with alert cooldown logic ensuring messages are not sent too frequently. In summary, for WhatsApp, the path to the Python executable and the WhatsApp script is defined, and a command string is constructed to execute the WhatsApp message-sending script. This command is executed using _ShellExecuteW_, with a result check to log any errors. Similarly, for Telegram, the path to the Python executable and the Telegram script is defined, a command string is constructed to execute the Telegram message-sending script, and the command is executed using _ShellExecuteW_, with a result check to log any errors.

Here's the combined program:

```
//--- ShellExecuteW declaration ----------------------------------------------
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import

//--- global variables ------------------------------------------------------
datetime last_alert_time;
input int alert_cooldown_seconds = 60; // Cooldown period in seconds

//--- myAlert function ------------------------------------------------------
void myAlert(string type, string message) {
    datetime current_time = TimeCurrent();
    if (current_time - last_alert_time < alert_cooldown_seconds) {
        // Skip alert if within cooldown period
        return;
    }

    last_alert_time = current_time;
    string full_message = type + " | Trend Constraint V1.06 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message;
    if (type == "print") {
        Print(message);
    } else if (type == "error") {
        Print(type + " | Trend Constraint V1.06 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message);
    } else if (type == "order") {
        // Add order alert handling if needed
    } else if (type == "modify") {
        // Add modify alert handling if needed
    } else if (type == "indicator" || type == "info") {
        if (Audible_Alerts) {
            Alert(full_message);
        }
        if (Push_Notifications) {
            SendNotification(full_message);
        }

        // Send to WhatsApp
        string python_path = "C:\\Users\\Your_Computer_Name\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";
        string whatsapp_script_path = "C:\\Users\\Your_Computer_Name\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_whatsapp_message.py";
        string whatsapp_command = python_path + " \"" + whatsapp_script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed for WhatsApp
        Print("Executing command to send WhatsApp message: ", whatsapp_command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_whatsapp_command = "/c " + whatsapp_command + " && timeout 5";
        int whatsapp_result = ShellExecuteW(0, "open", "cmd.exe", final_whatsapp_command, NULL, 1);
        if (whatsapp_result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute WhatsApp Python script. Error code: ", error_code);
        } else {
            Print("Successfully executed WhatsApp Python script. Result code: ", whatsapp_result);
        }

        // Send to Telegram
        string telegram_script_path = "C:\\Users\\Your_Computer_Name\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_telegram_message.py";
        string telegram_command = python_path + " \"" + telegram_script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed for Telegram
        Print("Executing command to send Telegram message: ", telegram_command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_telegram_command = "/c " + telegram_command + " && timeout 5";
        int telegram_result = ShellExecuteW(0, "open", "cmd.exe", final_telegram_command, NULL, 1);
        if (telegram_result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute Telegram Python script. Error code: ", error_code);
        } else {
            Print("Successfully executed Telegram Python script. Result code: ", telegram_result);
        }
    }
}
```

At this stage let's break the code into sections to explain their functionalities:

```
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import
```

This section imports the _ShellExecuteW_ function from the shell32.dll Windows library. _ShellExecuteW_ is a Windows API function that performs an operation on a specified file. By importing this function, the MQL5 code can execute external commands or scripts, such as Python scripts for sending messages via WhatsApp and Telegram.

```
datetime last_alert_time;
input int alert_cooldown_seconds = 60; // Cooldown period in seconds
```

The above code snippet makes the Global Variables of the integration algorithm.

- _last\_alert\_time_: A global variable that stores the timestamp of the last alert sent. This helps in implementing a cooldown period between alerts.
- _alert\_cooldown\_seconds_: An input variable (user-configurable) that specifies the cooldown period in seconds. This determines how frequently alerts can be sent to avoid spamming.

```
void myAlert(string type, string message) {
    datetime current_time = TimeCurrent();
    if (current_time - last_alert_time < alert_cooldown_seconds) {
        // Skip alert if within cooldown period
        return;
    }

    last_alert_time = current_time;
    string full_message = type + " | Trend Constraint V1.06 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message;
    if (type == "print") {
        Print(message);
    } else if (type == "error") {
        Print(type + " | Trend Constraint V1.06 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message);
    } else if (type == "order") {
        // Add order alert handling if needed
    } else if (type == "modify") {
        // Add modify alert handling if needed
    } else if (type == "indicator" || type == "info") {
        if (Audible_Alerts) {
            Alert(full_message);
        }
        if (Push_Notifications) {
            SendNotification(full_message);
        }

        // Send to WhatsApp
        string python_path = "C:\\Users\\Your_Computer_Name\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";
        string whatsapp_script_path = "C:\\Users\\Your_Computer_Name\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_whatsapp_message.py";
        string whatsapp_command = python_path + " \"" + whatsapp_script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed for WhatsApp
        Print("Executing command to send WhatsApp message: ", whatsapp_command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_whatsapp_command = "/c " + whatsapp_command + " && timeout 5";
        int whatsapp_result = ShellExecuteW(0, "open", "cmd.exe", final_whatsapp_command, NULL, 1);
        if (whatsapp_result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute WhatsApp Python script. Error code: ", error_code);
        } else {
            Print("Successfully executed WhatsApp Python script. Result code: ", whatsapp_result);
        }

        // Send to Telegram
        string telegram_script_path = "C:\\Users\\Your_Computer_Name\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_telegram_message.py";
        string telegram_command = python_path + " \"" + telegram_script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed for Telegram
        Print("Executing command to send Telegram message: ", telegram_command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_telegram_command = "/c " + telegram_command + " && timeout 5";
        int telegram_result = ShellExecuteW(0, "open", "cmd.exe", final_telegram_command, NULL, 1);
        if (telegram_result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute Telegram Python script. Error code: ", error_code);
        } else {
            Print("Successfully executed Telegram Python script. Result code: ", telegram_result);
        }
    }
}
```

-  The myAlert function is designed to send alerts based on the type and message provided. It manages the cooldown period, constructs the alert message, and sends it to both WhatsApp and Telegram using external Python scripts. It is the largest section of the code as you can see.

```
datetime current_time = TimeCurrent();
if (current_time - last_alert_time < alert_cooldown_seconds) {
    // Skip alert if within cooldown period
    return;
}
last_alert_time = current_time;
```

- This section checks if the current time minus the last alert time is less than the cooldown period. If true, it skips sending the alert. This prevents frequent alerts within a short period.

To make sure our scripts are working, we obtain the following results in the Command Prompt:

```
C:\Users\Your_Computer_Name\AppData\Local\Programs\Python\Python312\Scripts>python send_telegram_message.py "Trend Constraint V1.07 testing"
Message sent successfully!

C:\Users\Your_Computer_Name\AppData\Local\Programs\Python\Python312\Scripts>python send_whatsapp_message.py "Trend Constraint V1.07 testing"
Message sent successfully
```

The highlighted text is the positive response from the Command Prompt, confirming that our scripts are working fine. It is very important to make add the file path correctly in the main program.

On the other side, we also receive the signals on our social platforms. Below, on the left, is an image showing a Telegram test message from the Command Prompt, and on the right, is a WhatsApp test message from the Command Prompt. We are now certain that our program is working fine, and we can proceed to our main program.

![Telegram script test](https://c.mql5.com/2/84/Telegram_Integration.png)![Whatsapp Script Test](https://c.mql5.com/2/84/Whatsappscript_Integration.png)

In the illustration above, the sandbox connection provided by the [Twilio API](https://www.mql5.com/go?link=https://console.twilio.com/us1/develop/sms "https://console.twilio.com/us1/develop/sms") for WhatsApp integration expires within 72 hours. It is important to reconnect by sending a unique message to be re-added for receiving API messages. In this case, the message to get reconnected is "join so-cave." To acquire a non-expiring service, you can purchase a Twilio number.

Let's proceed and integrate everything into one program using the Trend Constraint indicator logic. This advances us to Trend Constraint V1.07:

```
//+------------------------------------------------------------------+
//|                                       Trend Constraint V1.07.mq5 |
//|                                Copyright 2024, Clemence Benjamin |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com"
#property version   "1.07"
#property description "A model that seeks to produce sell signals when D1 candle is Bearish only and buy signals when it is Bullish"

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
#property indicator_label5 "Buy"

#property indicator_type6 DRAW_LINE
#property indicator_style6 STYLE_SOLID
#property indicator_width6 2
#property indicator_color6 0x0000FF
#property indicator_label6 "Sell"

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
int MA_handle7;
double MA7[];

//--- ShellExecuteW declaration ----------------------------------------------
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import

//--- global variables ------------------------------------------------------
datetime last_alert_time;
input int alert_cooldown_seconds = 60; // Cooldown period in seconds

//--- myAlert function ------------------------------------------------------
void myAlert(string type, string message) {
    datetime current_time = TimeCurrent();
    if (current_time - last_alert_time < alert_cooldown_seconds) {
        // Skip alert if within cooldown period
        return;
    }

    last_alert_time = current_time;
    string full_message = type + " | Trend Constraint V1.07 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message;
    if (type == "print") {
        Print(message);
    } else if (type == "error") {
        Print(type + " | Trend Constraint V1.07 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message);
    } else if (type == "order") {
        // Add order alert handling if needed
    } else if (type == "modify") {
        // Add modify alert handling if needed
    } else if (type == "indicator" || type == "info") {
        if (Audible_Alerts) {
            Alert(full_message);
        }
        if (Push_Notifications) {
            SendNotification(full_message);
        }

        // Send to WhatsApp //Replace your_computer_name with the your actual computer name. //Make sure the path to your python and scripts is correct.
        string python_path = "C:\\Users\\Your_Computer\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";
        string whatsapp_script_path = "C:\\Users\\Your_computer_name\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_whatsapp_message.py";
        string whatsapp_command = python_path + " \"" + whatsapp_script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed for WhatsApp
        Print("Executing command to send WhatsApp message: ", whatsapp_command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_whatsapp_command = "/c " + whatsapp_command + " && timeout 5";
        int whatsapp_result = ShellExecuteW(0, "open", "cmd.exe", final_whatsapp_command, NULL, 1);
        if (whatsapp_result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute WhatsApp Python script. Error code: ", error_code);
        } else {
            Print("Successfully executed WhatsApp Python script. Result code: ", whatsapp_result);
        }

        // Send to Telegram
        string telegram_script_path = "C:\\Users\\protech\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_telegram_message.py";
        string telegram_command = python_path + " \"" + telegram_script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed for Telegram
        Print("Executing command to send Telegram message: ", telegram_command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_telegram_command = "/c " + telegram_command + " && timeout 5";
        int telegram_result = ShellExecuteW(0, "open", "cmd.exe", final_telegram_command, NULL, 1);
        if (telegram_result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute Telegram Python script. Error code: ", error_code);
        } else {
            Print("Successfully executed Telegram Python script. Result code: ", telegram_result);
        }
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
   // Send test message on launch
   myAlert("info", "Thank you for subscribing. You shall be receiving Trend Constraint signal alerts via Whatsapp.");
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

   MA_handle7 = iMA(NULL, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle7 < 0)
     {
      Print("The creation of iMA has failed: MA_handle7=", INVALID_HANDLE);
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

   datetime TimeShift[];
   if(CopyTime(Symbol(), PERIOD_CURRENT, 0, rates_total, TimeShift) <= 0) return(rates_total);
   ArraySetAsSeries(TimeShift, true);
   int barshift_M1[];
   ArrayResize(barshift_M1, rates_total);
   int barshift_D1[];
   ArrayResize(barshift_D1, rates_total);
   for(int i = 0; i < rates_total; i++)
     {
      barshift_M1[i] = iBarShift(Symbol(), PERIOD_M1, TimeShift[i]);
      barshift_D1[i] = iBarShift(Symbol(), PERIOD_D1, TimeShift[i]);
   }
   if(BarsCalculated(RSI_handle) <= 0)
      return(0);
   if(CopyBuffer(RSI_handle, 0, 0, rates_total, RSI) <= 0) return(rates_total);
   ArraySetAsSeries(RSI, true);
   if(CopyOpen(Symbol(), PERIOD_M1, 0, rates_total, Open) <= 0) return(rates_total);
   ArraySetAsSeries(Open, true);
   if(CopyClose(Symbol(), PERIOD_D1, 0, rates_total, Close) <= 0) return(rates_total);
   ArraySetAsSeries(Close, true);
   if(BarsCalculated(MA_handle) <= 0)
      return(0);
   if(CopyBuffer(MA_handle, 0, 0, rates_total, MA) <= 0) return(rates_total);
   ArraySetAsSeries(MA, true);
   if(BarsCalculated(MA_handle2) <= 0)
      return(0);
   if(CopyBuffer(MA_handle2, 0, 0, rates_total, MA2) <= 0) return(rates_total);
   ArraySetAsSeries(MA2, true);
   if(BarsCalculated(MA_handle3) <= 0)
      return(0);
   if(CopyBuffer(MA_handle3, 0, 0, rates_total, MA3) <= 0) return(rates_total);
   ArraySetAsSeries(MA3, true);
   if(BarsCalculated(MA_handle4) <= 0)
      return(0);
   if(CopyBuffer(MA_handle4, 0, 0, rates_total, MA4) <= 0) return(rates_total);
   ArraySetAsSeries(MA4, true);
   if(CopyLow(Symbol(), PERIOD_CURRENT, 0, rates_total, Low) <= 0) return(rates_total);
   ArraySetAsSeries(Low, true);
   if(CopyHigh(Symbol(), PERIOD_CURRENT, 0, rates_total, High) <= 0) return(rates_total);
   ArraySetAsSeries(High, true);
   if(BarsCalculated(MA_handle5) <= 0)
      return(0);
   if(CopyBuffer(MA_handle5, 0, 0, rates_total, MA5) <= 0) return(rates_total);
   ArraySetAsSeries(MA5, true);
   if(BarsCalculated(MA_handle6) <= 0)
      return(0);
   if(CopyBuffer(MA_handle6, 0, 0, rates_total, MA6) <= 0) return(rates_total);
   ArraySetAsSeries(MA6, true);
   if(BarsCalculated(MA_handle7) <= 0)
      return(0);
   if(CopyBuffer(MA_handle7, 0, 0, rates_total, MA7) <= 0) return(rates_total);
   ArraySetAsSeries(MA7, true);
   if(CopyTime(Symbol(), Period(), 0, rates_total, Time) <= 0) return(rates_total);
   ArraySetAsSeries(Time, true);
   //--- main loop
   for(int i = limit-1; i >= 0; i--)
     {
      if (i >= MathMin(PLOT_MAXIMUM_BARS_BACK-1, rates_total-1-OMIT_OLDEST_BARS)) continue; //omit some old rates to prevent "Array out of range" or slow calculation

      if(barshift_M1[i] < 0 || barshift_M1[i] >= rates_total) continue;
      if(barshift_D1[i] < 0 || barshift_D1[i] >= rates_total) continue;

      //Indicator Buffer 1
      if(RSI[i] < Oversold
      && RSI[i+1] > Oversold //Relative Strength Index crosses below fixed value
      && Open[barshift_M1[i]] >= Close[1+barshift_D1[i]] //Candlestick Open >= Candlestick Close
      && MA[i] > MA2[i] //Moving Average > Moving Average
      && MA3[i] > MA4[i] //Moving Average > Moving Average
      )
        {
         Buffer1[i] = Low[1+i]; //Set indicator value at Candlestick Low
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer1[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 2
      if(RSI[i] > Overbought
      && RSI[i+1] < Overbought //Relative Strength Index crosses above fixed value
      && Open[barshift_M1[i]] <= Close[1+barshift_D1[i]] //Candlestick Open <= Candlestick Close
      && MA[i] < MA2[i] //Moving Average < Moving Average
      && MA3[i] < MA4[i] //Moving Average < Moving Average
      )
        {
         Buffer2[i] = High[1+i]; //Set indicator value at Candlestick High
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer2[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 3
      if(MA5[i] > MA6[i]
      && MA5[i+1] < MA6[i+1] //Moving Average crosses above Moving Average
      )
        {
         Buffer3[i] = Low[i]; //Set indicator value at Candlestick Low
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy Reversal"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer3[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 4
      if(MA5[i] < MA6[i]
      && MA5[i+1] > MA6[i+1] //Moving Average crosses below Moving Average
      )
        {
         Buffer4[i] = High[i]; //Set indicator value at Candlestick High
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell Reversal"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer4[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 5, Alert muted by turning it into a comment
      if(MA3[i] > MA7[i] //Moving Average > Moving Average
      )
        {
         Buffer5[i] = MA3[i]; //Set indicator value at Moving Average
         //if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy"); //Alert on next bar open
         //time_alert = Time[1];
        }
      else
        {
         Buffer5[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 6, Alert muted by turning it into a comment
      if(MA3[i] < MA7[i] //Moving Average < Moving Average
      )
        {
         Buffer6[i] = MA3[i]; //Set indicator value at Moving Average
         //if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell"); //Alert on next bar open
         //time_alert = Time[1];
        }
      else
        {
         Buffer6[i] = EMPTY_VALUE;
        }
     }
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

If you observed carefully, we updated the indicator from V1.06 to V1.07. After compiling the program, we encountered no errors, and our program is now operating smoothly on MetaTrader 5. Below are images of the test messages sent upon launching the indicator on MT5: on the far left are Push Notifications on MetaTrader 5 Android mobile, in the centre is a Telegram test notification, and on the right is a WhatsApp test message.

![MetaTrade 5 push notification on Android](https://c.mql5.com/2/84/mt5pushh.png)![Telegram, Trend Constraint V1.07 test](https://c.mql5.com/2/84/telegramtest.png)![Whatsapp, Trend Constraint V1.07 test](https://c.mql5.com/2/84/Whatsapp.png)

### The Comment Function

The [Comment function](https://www.mql5.com/en/docs/common/comment) in MQL5 is a built-in function used to display custom text messages directly on the chart. This function helps us in providing real-time visual feedback by displaying messages that can be continuously updated during the execution of an indicator or an Expert Advisor. In this case, our aim is to use it to achieve the following:

- Notifying the user of the successful launch of the indicator.
- Confirming the successful sending of alert messages.
- Alerting the user to failures in sending alert messages.

We will target three areas in the code to incorporate the feature:

```
int OnInit() {
    // Initialization code here
    Comment("Indicator successfully launched.");
    return INIT_SUCCEEDED;
}

```

The purpose is of the above code snippet is to notify us that the indicator has been successfully launched. Upon successful initialization of the indicator, the Comment function displays the message, " _Indicator successfully launched_" on the chart. This provides immediate feedback that the indicator is active and running.

```
if (result > 32) {
    Print("Successfully executed Python script. Result code: ", result);
    Comment("Success message sent: " + message);
}
```

This is to inform us that an alert message has been successfully sent. When an alert message is successfully sent using the _myAlert_ function, the function displays the message " _Success message sent_ _\[message\]_" on the chart, where _\[message\]_ is the actual alert content. This provides confirmation to the us that the alert has been dispatched correctly.

```
if (result <= 32) {
    int error_code = GetLastError();
    Print("Failed to execute Python script. Error code: ", error_code);
    Comment("Failed to send message: " + message);
}
```

Finally, we also want to be informed of a failed launch and this enhances the functionality of our program. This snippet notifies us of a failure in sending an alert message. If there is a failure in sending an alert message, it displays the message, " _Failed to send message \[message\]_" on the chart, where _\[message\]_ is the intended alert content. This alerts us to the failure, allowing them to take corrective actions.

To leverage the new capabilities introduced by the [_Comment function_](https://www.mql5.com/en/docs/common/comment), I have upgraded our program to Trend Constraint V1.08. By strategically integrating this function into the relevant sections of the code, I successfully updated the program, ensuring its smooth operation. Below, you will find the source code with the modified sections highlighted, showcasing the enhancements made.

```
//+------------------------------------------------------------------+
//|                                       Trend Constraint V1.08.mq5 |
//|                                Copyright 2024, Clemence Benjamin |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

#property indicator_chart_window
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com"
#property version   "1.08"
#property description "A model that seeks to produce sell signals when D1 candle is Bearish only and buy signals when it is Bullish"

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
#property indicator_label5 "Buy"

#property indicator_type6 DRAW_LINE
#property indicator_style6 STYLE_SOLID
#property indicator_width6 2
#property indicator_color6 0x0000FF
#property indicator_label6 "Sell"

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
int MA_handle7;
double MA7[];

//--- ShellExecuteW declaration ----------------------------------------------
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import

//--- global variables ------------------------------------------------------
datetime last_alert_time;
input int alert_cooldown_seconds = 60; // Cooldown period in seconds

//--- myAlert function ------------------------------------------------------
void myAlert(string type, string message) {
    datetime current_time = TimeCurrent();
    if (current_time - last_alert_time < alert_cooldown_seconds) {
        // Skip alert if within cooldown period
        return;
    }

    last_alert_time = current_time;
    string full_message = type + " | Trend Constraint V1.08 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message;

    string comment = "Alert triggered by Trend Constraint V1.08 | Symbol: " + Symbol() + " | Period: " + IntegerToString(Period()) + " | Message: " + message;

    if (type == "print") {
        Print(message);
    } else if (type == "error") {
        Print(type + " | Trend Constraint V1.08 @ " + Symbol() + "," + IntegerToString(Period()) + " | " + message);
    } else if (type == "order") {
        // Add order alert handling if needed
    } else if (type == "modify") {
        // Add modify alert handling if needed
    } else if (type == "indicator" || type == "info") {
        if (Audible_Alerts) {
            Alert(full_message);
        }
        if (Push_Notifications) {
            SendNotification(full_message);
        }

        // Send to WhatsApp
        string python_path = "C:\\Users\\protech\\AppData\\Local\\Programs\\Python\\Python312\\python.exe";
        string whatsapp_script_path = "C:\\Users\\protech\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_whatsapp_message.py";
        string whatsapp_command = python_path + " \"" + whatsapp_script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed for WhatsApp
        Print("Executing command to send WhatsApp message: ", whatsapp_command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_whatsapp_command = "/c " + whatsapp_command + " && timeout 5";
        int whatsapp_result = ShellExecuteW(0, "open", "cmd.exe", final_whatsapp_command, NULL, 0);
        if (whatsapp_result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute WhatsApp Python script. Error code: ", error_code);
            Comment("Failed to send message: " + message);
        } else {
            Print("Successfully executed WhatsApp Python script. Result code: ", whatsapp_result);
            Comment("Success message sent: " + message);
        }

        // Send to Telegram
        string telegram_script_path = "C:\\Users\\protech\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\send_telegram_message.py";
        string telegram_command = python_path + " \"" + telegram_script_path + "\" \"" + full_message + "\"";

        // Debugging: Print the command being executed for Telegram
        Print("Executing command to send Telegram message: ", telegram_command);

        // Use cmd.exe to execute the command and then wait for 5 seconds
        string final_telegram_command = "/c " + telegram_command + " && timeout 5";
        int telegram_result = ShellExecuteW(0, "open", "cmd.exe", final_telegram_command, NULL, 0);
        if (telegram_result <= 32) {
            int error_code = GetLastError();
            Print("Failed to execute Telegram Python script. Error code: ", error_code);
            Comment("Failed to send message: " + message);
        } else {
            Print("Successfully executed Telegram Python script. Result code: ", telegram_result);
            Comment("Success message sent: " + message);
        }
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
   // Send test message on launch
   myAlert("info", "Thank you for subscribing. You shall be receiving Trend Constraint signal alerts via Whatsapp.");
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

   MA_handle7 = iMA(NULL, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle7 < 0)
     {
      Print("The creation of iMA has failed: MA_handle7=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }
   Comment("Indicator successfully launched.");
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

   datetime TimeShift[];
   if(CopyTime(Symbol(), PERIOD_CURRENT, 0, rates_total, TimeShift) <= 0) return(rates_total);
   ArraySetAsSeries(TimeShift, true);
   int barshift_M1[];
   ArrayResize(barshift_M1, rates_total);
   int barshift_D1[];
   ArrayResize(barshift_D1, rates_total);
   for(int i = 0; i < rates_total; i++)
     {
      barshift_M1[i] = iBarShift(Symbol(), PERIOD_M1, TimeShift[i]);
      barshift_D1[i] = iBarShift(Symbol(), PERIOD_D1, TimeShift[i]);
   }
   if(BarsCalculated(RSI_handle) <= 0)
      return(0);
   if(CopyBuffer(RSI_handle, 0, 0, rates_total, RSI) <= 0) return(rates_total);
   ArraySetAsSeries(RSI, true);
   if(CopyOpen(Symbol(), PERIOD_M1, 0, rates_total, Open) <= 0) return(rates_total);
   ArraySetAsSeries(Open, true);
   if(CopyClose(Symbol(), PERIOD_D1, 0, rates_total, Close) <= 0) return(rates_total);
   ArraySetAsSeries(Close, true);
   if(BarsCalculated(MA_handle) <= 0)
      return(0);
   if(CopyBuffer(MA_handle, 0, 0, rates_total, MA) <= 0) return(rates_total);
   ArraySetAsSeries(MA, true);
   if(BarsCalculated(MA_handle2) <= 0)
      return(0);
   if(CopyBuffer(MA_handle2, 0, 0, rates_total, MA2) <= 0) return(rates_total);
   ArraySetAsSeries(MA2, true);
   if(BarsCalculated(MA_handle3) <= 0)
      return(0);
   if(CopyBuffer(MA_handle3, 0, 0, rates_total, MA3) <= 0) return(rates_total);
   ArraySetAsSeries(MA3, true);
   if(BarsCalculated(MA_handle4) <= 0)
      return(0);
   if(CopyBuffer(MA_handle4, 0, 0, rates_total, MA4) <= 0) return(rates_total);
   ArraySetAsSeries(MA4, true);
   if(CopyLow(Symbol(), PERIOD_CURRENT, 0, rates_total, Low) <= 0) return(rates_total);
   ArraySetAsSeries(Low, true);
   if(CopyHigh(Symbol(), PERIOD_CURRENT, 0, rates_total, High) <= 0) return(rates_total);
   ArraySetAsSeries(High, true);
   if(BarsCalculated(MA_handle5) <= 0)
      return(0);
   if(CopyBuffer(MA_handle5, 0, 0, rates_total, MA5) <= 0) return(rates_total);
   ArraySetAsSeries(MA5, true);
   if(BarsCalculated(MA_handle6) <= 0)
      return(0);
   if(CopyBuffer(MA_handle6, 0, 0, rates_total, MA6) <= 0) return(rates_total);
   ArraySetAsSeries(MA6, true);
   if(BarsCalculated(MA_handle7) <= 0)
      return(0);
   if(CopyBuffer(MA_handle7, 0, 0, rates_total, MA7) <= 0) return(rates_total);
   ArraySetAsSeries(MA7, true);
   if(CopyTime(Symbol(), Period(), 0, rates_total, Time) <= 0) return(rates_total);
   ArraySetAsSeries(Time, true);
   //--- main loop
   for(int i = limit-1; i >= 0; i--)
     {
      if (i >= MathMin(PLOT_MAXIMUM_BARS_BACK-1, rates_total-1-OMIT_OLDEST_BARS)) continue; //omit some old rates to prevent "Array out of range" or slow calculation

      if(barshift_M1[i] < 0 || barshift_M1[i] >= rates_total) continue;
      if(barshift_D1[i] < 0 || barshift_D1[i] >= rates_total) continue;

      //Indicator Buffer 1
      if(RSI[i] < Oversold
      && RSI[i+1] > Oversold //Relative Strength Index crosses below fixed value
      && Open[barshift_M1[i]] >= Close[1+barshift_D1[i]] //Candlestick Open >= Candlestick Close
      && MA[i] > MA2[i] //Moving Average > Moving Average
      && MA3[i] > MA4[i] //Moving Average > Moving Average
      )
        {
         Buffer1[i] = Low[1+i]; //Set indicator value at Candlestick Low
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer1[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 2
      if(RSI[i] > Overbought
      && RSI[i+1] < Overbought //Relative Strength Index crosses above fixed value
      && Open[barshift_M1[i]] <= Close[1+barshift_D1[i]] //Candlestick Open <= Candlestick Close
      && MA[i] < MA2[i] //Moving Average < Moving Average
      && MA3[i] < MA4[i] //Moving Average < Moving Average
      )
        {
         Buffer2[i] = High[1+i]; //Set indicator value at Candlestick High
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer2[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 3
      if(MA5[i] > MA6[i]
      && MA5[i+1] < MA6[i+1] //Moving Average crosses above Moving Average
      )
        {
         Buffer3[i] = Low[i]; //Set indicator value at Candlestick Low
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy Reversal"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer3[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 4
      if(MA5[i] < MA6[i]
      && MA5[i+1] > MA6[i+1] //Moving Average crosses below Moving Average
      )
        {
         Buffer4[i] = High[i]; //Set indicator value at Candlestick High
         if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell Reversal"); //Alert on next bar open
         time_alert = Time[1];
        }
      else
        {
         Buffer4[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 5, Alert muted by turning it into a comment
      if(MA3[i] > MA7[i] //Moving Average > Moving Average
      )
        {
         Buffer5[i] = MA3[i]; //Set indicator value at Moving Average
         //if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Buy"); //Alert on next bar open
         //time_alert = Time[1];
        }
      else
        {
         Buffer5[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 6, Alert muted by turning it into a comment
      if(MA3[i] < MA7[i] //Moving Average < Moving Average
      )
        {
         Buffer6[i] = MA3[i]; //Set indicator value at Moving Average
         //if(i == 1 && Time[1] != time_alert) myAlert("indicator", "Sell"); //Alert on next bar open
         //time_alert = Time[1];
        }
      else
        {
         Buffer6[i] = EMPTY_VALUE;
        }
     }
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

### Testing the effects of the Comment Function

Implementing the _[Comment](https://www.mql5.com/en/docs/common/comment)_ function in MQL5 is a straightforward way to enhance the interactivity and formativeness of trading charts. By integrating this function, you can provide traders with essential, real-time updates directly on their charts, improving their overall trading experience. This function allows for the display of dynamic data, such as current prices, indicator values, and custom messages, in a clear and concise manner. As a result, traders can make more informed decisions without the need to switch between multiple windows or external tools.

The _[Comment](https://www.mql5.com/en/docs/common/comment)_ function's ease of use and flexibility make it an invaluable tool for developing more user-friendly and efficient trading algorithms. By incorporating real-time, context-specific information directly onto the trading interface, the function enhances situational awareness and streamlines the trading process, contributing to a more effective and satisfying user experience. Here's an image, showing successful launch of Trend Constraint V1.07:

![](https://c.mql5.com/2/84/terminal64_qbzO50kVb7__1.gif)

### Conclusion

In the software development journey, innovation often comes from the seamless integration of existing solutions to create more robust and feature-rich applications. This article explored the process of merging two programs into a single cohesive unit, demonstrating the power of combining functionalities to enhance overall performance and user experience.

We began by understanding the core functionalities of the two separate programs, each with its unique strengths. By carefully analyzing their codebases and identifying points of synergy, we successfully merged them into a unified program. This merger not only streamlined operations but also reduced redundancy and potential conflicts, paving the way for a more efficient execution.

Furthermore, the incorporation of the [C _omment function_](https://www.mql5.com/en/docs/common/comment) in the MQL5 program added a new dimension to our combined application. By leveraging MQL5's robust alert system, we implemented a feature that allows for real-time notifications through various channels, including WhatsApp and Telegram. This enhancement ensures that users are always informed of critical events, thereby improving responsiveness and decision-making.

As we move forward, the possibilities for further enhancements and customizations remain vast, inviting continuous improvement and innovation. By building on existing technologies and thoughtfully integrating new features, we can create powerful tools that drive efficiency, enhance user engagement, and ultimately lead to better outcomes.

See attached files below. Comments and views are always welcome.

| Attachment | Description |
| --- | --- |
| Trend Constraint V1.07.mq5 | Integration of two platforms in one program. |
| Trend Constraint V1.08.mq5 | Incorporating the command function. |
| [Send\_telegram\_message.py](https://www.mql5.com/en/articles/14968) | Script for sending Telegram messages. |
| [send\_whatsapp\_message.py](https://www.mql5.com/en/articles/14969) | Script for sending WhatsApp messages. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15143.zip "Download all attachments in the single ZIP archive")

[Trend\_Constraint\_V1.07.mq5](https://www.mql5.com/en/articles/download/15143/trend_constraint_v1.07.mq5 "Download Trend_Constraint_V1.07.mq5")(17.46 KB)

[Trend\_Constraint\_V1.08.mq5](https://www.mql5.com/en/articles/download/15143/trend_constraint_v1.08.mq5 "Download Trend_Constraint_V1.08.mq5")(17.89 KB)

[send\_telegram\_message.py](https://www.mql5.com/en/articles/download/15143/send_telegram_message.py "Download send_telegram_message.py")(0.65 KB)

[send\_whatsapp\_message.py](https://www.mql5.com/en/articles/download/15143/send_whatsapp_message.py "Download send_whatsapp_message.py")(0.96 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/470155)**
(2)


![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
14 Jan 2025 at 16:32

can be simpler and more universal. The common practice in the big world is not to reinvent the wheel, but to use a data-collector/router

for example metrics and events via Socket() or WebRequest() to telegraf [https://www.influxdata.com/time-series-platform/telegraf/](https://www.mql5.com/go?link=https://www.influxdata.com/time-series-platform/telegraf/ "https://www.influxdata.com/time-series-platform/telegraf/").

and it will handle all (any) recipients.

The code is much less and it works more stable.

--

telegraf is listed as an example, freely available to everyone. There are alternatives both commercial and open source

![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
29 Jan 2025 at 08:28

**Maxim Kuznetsov [#](https://www.mql5.com/en/forum/470155#comment_55632367):**

can be simpler and more universal. The common practice in the big world is not to reinvent the wheel, but to use a data-collector/router

for example metrics and events via Socket() or WebRequest() to telegraf [https://www.influxdata.com/time-series-platform/telegraf/](https://www.mql5.com/go?link=https://www.influxdata.com/time-series-platform/telegraf/ "https://www.influxdata.com/time-series-platform/telegraf/").

and it will handle all (any) recipients.

The code is much less and it works more stable.

--

telegraf is listed as an example, freely available to everyone. There are alternatives both commercial and open source

Thank you for sharing. I will take a look at how the server works.

![MQL5 Wizard Techniques you should know (Part 28): GANs Revisited with a Primer on Learning Rates](https://c.mql5.com/2/85/MQL5_Wizard_Techniques_you_should_know_Part_28____LOGO.png)[MQL5 Wizard Techniques you should know (Part 28): GANs Revisited with a Primer on Learning Rates](https://www.mql5.com/en/articles/15349)

The Learning Rate, is a step size towards a training target in many machine learning algorithms’ training processes. We examine the impact its many schedules and formats can have on the performance of a Generative Adversarial Network, a type of neural network that we had examined in an earlier article.

![Population optimization algorithms: Resistance to getting stuck in local extrema (Part II)](https://c.mql5.com/2/72/Population_optimization_algorithms__Resistance_to_getting_stuck_in_local_extrema__LOGO__1.png)[Population optimization algorithms: Resistance to getting stuck in local extrema (Part II)](https://www.mql5.com/en/articles/14212)

We continue our experiment that aims to examine the behavior of population optimization algorithms in the context of their ability to efficiently escape local minima when population diversity is low and reach global maxima. Research results are provided.

![Data Science and ML (Part 27): Convolutional Neural Networks (CNNs) in MetaTrader 5 Trading Bots — Are They Worth It?](https://c.mql5.com/2/84/Data_Science_and_ML_Part_27.png)[Data Science and ML (Part 27): Convolutional Neural Networks (CNNs) in MetaTrader 5 Trading Bots — Are They Worth It?](https://www.mql5.com/en/articles/15259)

Convolutional Neural Networks (CNNs) are renowned for their prowess in detecting patterns in images and videos, with applications spanning diverse fields. In this article, we explore the potential of CNNs to identify valuable patterns in financial markets and generate effective trading signals for MetaTrader 5 trading bots. Let us discover how this deep machine learning technique can be leveraged for smarter trading decisions.

![GIT: What is it?](https://c.mql5.com/2/69/GIT__Mas_que_coisa_2_esta___LOGO.png)[GIT: What is it?](https://www.mql5.com/en/articles/12516)

In this article, I will introduce a very important tool for developers. If you are not familiar with GIT, read this article to get an idea of what it is and how to use it with MQL5.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=orljcxkgcjhltlisvsgyvoltzqotgahs&ssn=1769093160572603670&ssn_dr=0&ssn_sr=0&fv_date=1769093160&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15143&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Building%20A%20Candlestick%20Trend%20Constraint%20Model%20(Part%206)%3A%20All%20in%20one%20integration%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909316092799504&fz_uniq=5049363338225363496&sv=2552)

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