---
title: Creating a Trading Administrator Panel in MQL5 (Part II): Enhancing Responsiveness and Quick Messaging
url: https://www.mql5.com/en/articles/15418
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:39:32.021052
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=gsktttwoucaqmbdeemtozowixivhpyxg&ssn=1769157570338615413&ssn_dr=0&ssn_sr=0&fv_date=1769157570&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15418&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20Trading%20Administrator%20Panel%20in%20MQL5%20(Part%20II)%3A%20Enhancing%20Responsiveness%20and%20Quick%20Messaging%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915757076182218&fz_uniq=5062624444888819137&sv=2552)

MetaTrader 5 / Examples


### Content:

1. [Introduction](https://www.mql5.com/en/articles/15418#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/15418#para2)
3. [Conclusion](https://www.mql5.com/en/articles/15418#para3)

### Introduction:

Imagine missing a crucial market signal due to a delayed message – a common hurdle that costs traders valuable opportunities and profits in the fast-paced trading environment. In this context, administrator insights are as vital as the market signals themselves. While algorithmic systems are quick and emotionally intelligent, they cannot replace the keen oversight of skilled traders, who continuously monitor system performance and make critical decisions.

![Think about this.](https://c.mql5.com/2/92/rect7.png)

Comparing On-click message and type in message.

![Limitations on the current Admin Panel](https://c.mql5.com/2/91/terminal64_PuHAjN9a6q.gif)

In the image above, there is an issue: when attempting to drag the panel, the chart moves instead, and the minimize button is not responding

As algorithmic trading dominates financial markets, efficient communication between trading systems users (Traders) and human administrators (man behind the trading system) has become imperative. Previously, we created Admin Panel messaging interfaces that have limited responsiveness required for real-time tasks such as quick messaging and clicking and dragging the panel itself, creating a significant challenge for administrators who need to react swiftly.

![Improvements for the current Admin Panel:](https://c.mql5.com/2/91/New_Panel.png)

Admin Panel: Improvements aimed at the previous Admin Panel

This article aims to break these communication barriers by introducing responsiveness to the Administrator Messaging Interface using MQL5. Additionally, it highlights the significance of quick messaging to support agile trading decisions and operational efficiency.

In this discussion, we will explore how MQL5 can be utilized to enhance message responsiveness within trading platforms. We’ll guide you through key implementation steps, helping you develop a more in-depth understanding of the possibilities offered by MQL5 programming. Together, we'll create a more effective messaging interface that meets the demands of modern trading.

To keep track of this discussion from start to finish, I will address the following questions:

1. What is responsiveness in GUI?
2. What are quick messages?

**Responsiveness:**

In MQL5, responsiveness in the GUI (Graphical User Interface) refers to how quickly and smoothly the interface reacts to user interactions, such as clicking buttons, moving sliders, or resizing panels. A responsive GUI provides immediate feedback to the user, ensuring that the interface feels intuitive and easy to use. This is especially important in trading applications, where timely actions can be crucial.

Achieving responsiveness often involves optimizing the code to reduce the execution time of GUI-related functions, minimizing the number of objects drawn on the chart, and utilizing asynchronous processing when possible. This ensures that the main thread remains responsive to user input, allowing for a seamless user experience during critical trading activities.

Let me outline in detail the key aspects of responsiveness in an MQL5 GUI:

- Immediate Feedback: The interface should respond instantly to user actions, such as clicking buttons or entering text. There should be no noticeable delay between the user's action and the system's response.

- Smooth Performance: Even with multiple GUI elements and complex logic, the interface should run smoothly without lagging or freezing. This involves efficient coding practices to minimize the load on the CPU and ensure quick execution of user commands.

- Dynamic Updates: The GUI should be able to update elements dynamically without requiring a full redraw of the entire interface. For example, if a new price level is reached, the corresponding elements (e.g., labels, lines) should update smoothly without flickering.

- Scalability: The interface should handle changes in size or resolution well. For example, if the user resizes a panel, the content should adjust automatically and maintain usability.

- Error Handling: The GUI should manage errors gracefully, providing clear and immediate feedback to the user if something goes wrong, without crashing or becoming unresponsive.

**Quick Messages:**

Refer to predefined, commonly used messages that can be sent with just a single click or minimal interaction. These messages are typically configured in advance to meet frequent communication needs, enabling users to quickly respond or send standard messages without the need for manual typing.

**Use Cases for Quick Messages**

- Standard Responses: Quick messages can be used for standard replies or commands, such as acknowledging a trade signal, confirming an action, or notifying a team about a specific event.
- Error Notifications: If an error occurs in trading, a brief message like "Invalid signal" or "Error detected" can be sent immediately.
- Routine Commands: Quick messages can include routine commands or instructions that are frequently used in trading operations, like "Close all positions" or "Activate EA."

**Example:**

Suppose you have an Admin Panel in MQL5 with several Quick Messages for an automated trading system. These might include:

- Start monitoring: A brief message to begin monitoring market conditions.
- Stop monitoring: A brief message to halt monitoring.
- Invalid signal: A message to notify the user that an invalid trade signal has been detected.

Each of these could be tied to a button on the panel. When the button is clicked, the pre-defined message is instantly sent out, saving time and ensuring consistent communication.

### Implementation in MQL5

In MQL5, Quick Messages can be implemented in a Messaging Panel by creating buttons or dropdown menus linked to specific pre-written text strings. When the user clicks one of these buttons, the corresponding message is automatically sent via the desired communication channel, such as [Telegram](https://www.mql5.com/go?link=http://api.telegram.org/ "http://api.telegram.org/"), email, or another messaging API. Some obvious drawbacks noticeable in the animated image on introduction showing our panel covering some part of the chart can be annoying when we want a have a bigger picture to analyze the chart. To conquer this segment, I will divide it into two:

1.  Logically placing Panel control buttons.
2.  Coding our quick messages buttons using the repeat function.

I assume you have read [Part I](https://www.mql5.com/en/articles/15417) and have gained a sense of where we are coming from and where we are going.

**1\. Logically placing Panel control buttons**

![The close, minimize and maximize buttons.](https://c.mql5.com/2/92/Buttons.PNG)

Minimize, Maximize and Close Button

- Button Declaration:

Here, we show how we declare the buttons before we go further:

```
///Global variables
CButton minimizeButton;
CButton maximizeButton;
CButton closeButton;
```

- Minimize button:


For the minimize button, we created it using the **CButton** class, positioning it at (375, -22) on the chart with a size that spans (30, 22) pixels. The button displays an underscore \_, a common symbol for minimizing windows. We added it to the Admin Panel using " **adminPanel. Add(minimizeButton)**". The purpose of this button is to allow users to temporarily hide the Admin Panel from view without closing it completely. In the " **OnMinimizeButtonClick()**" function, we programmed the button to hide the Admin Panel and only show minimize, maximize, and close buttons. This simulates minimizing the window while keeping essential controls available.

```
// Create the minimize button
if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
{
    Print("Failed to create minimize button");
    return INIT_FAILED;
}
minimizeButton.Text("_");
adminPanel.Add(minimizeButton);

// Function to handle minimize button click
void OnMinimizeButtonClick()
{
    minimized = true;

    // Hide the full admin panel
    adminPanel.Hide();
    minimizeButton.Show();
    maximizeButton.Show();
    closeButton.Show();
}
```

- Maximize button:

For the maximize button, we used the same **"CButton"** class and positioned it next to the minimize button at (405, -22). The button displays **\[ \]**, a common symbol for maximizing or restoring windows, and we added it to the Admin Panel using **"adminPanel. Add(maximizeButton)"**. This button allows users to restore the Admin Panel to its full size after it has been minimized. In the " **OnMaximizeButtonClick()"** function, clicking this button restores the Admin Panel to its original size and hides minimize, maximize, and close buttons. This mimics the behavior of maximizing a minimized window.

```
// Create the maximize button
if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
{
    Print("Failed to create maximize button");
    return INIT_FAILED;
}
maximizeButton.Text("[ ]");
adminPanel.Add(maximizeButton);

// Function to handle maximize button click
void OnMaximizeButtonClick()
{
    if (minimized)
    {
        minimizeButton.Hide();
        maximizeButton.Hide();
        closeButton.Hide();
        adminPanel.Show();
    }
}
```

- Close button:

For the close button, we created it in the same manner as the other two buttons, positioning it at (435, -22) and displaying an X, a universal symbol for closing windows. We added this button to the Admin Panel using " **adminPanelAdd(closeButton**)". This button allows users to completely remove the Expert Advisor (EA) from the chart by calling " **ExpertRemove()"** in the " **OnCloseButtonClick()"** function. It provides a straightforward way to close the Admin Panel and stop the EA when the user is done using the panel.

```
// Create the close button
if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
{
    Print("Failed to create close button");
    return INIT_FAILED;
}
closeButton.Text("X");
adminPanel.Add(closeButton);

// Function to handle close button click
void OnCloseButtonClick()
{
    ExpertRemove(); // Completely remove the EA
    Print("Admin Panel closed.");
}
```

**2\. Coding our quick messages buttons using the loop (repeat) function**

![Quick Messaging Buttons.](https://c.mql5.com/2/92/Quick_M_Buttons.PNG)

Multiple Quick Message Button interface

Inputs for Quick Messages:

-  We used of input variables (QuickMessage1 to QuickMessage8) to make the messages customizable. These inputs allow users to modify the text of each quick message directly from the Expert Advisor's settings, without altering the core code. This flexibility makes it easy to tailor the messages to different trading scenarios or user preferences. Additionally, the button placement is dynamic, meaning you can adjust the number of buttons, their size, or their position by modifying the loop parameters or the " **quickMessages"** array. This structure ensures that the Admin Panel can be easily adapted to different needs, providing a robust and user-friendly interface.

```
//+------------------------------------------------------------------+
//| Inputs                                                                       |
//+------------------------------------------------------------------+
input string QuickMessage1 = "Updates";
input string QuickMessage2 = "Close all";
input string QuickMessage3 = "In deep profits";
input string QuickMessage4 = "Hold position";
input string QuickMessage5 = "Swing Entry";
input string QuickMessage6 = "Scalp Entry";
input string QuickMessage7 = "Book profit";
input string QuickMessage8 = "Invalid Signal";
input string InputChatId = "Enter Chat ID from Telegram bot API";  // User's Telegram chat ID
input string InputBotToken = "Enter BOT TOKEN from your Telegram bot"; // User's Telegram bot token
```

Loop Implementation:

- We established multiple quick message buttons by creating an array of ( **CButton)** objects ( **quickMessageButtons\[8\]**) and initializing them in a loop. The loop iterates through the ( **quickMessages)** array, which contains predefined messages. Each iteration creates a button, assigns a label from **(quickMessages)**, and positions the buttons dynamically based on their index. The essence of the repeat function is captured by the loop structure, which replicates the creation and setup process for each button, ensuring consistency and efficiency. In MQL5, this approach minimizes redundancy by using a loop to handle repetitive tasks like creating multiple buttons with similar characteristics, thus simplifying the code and reducing the potential for errors.

```
// Array of predefined quick messages
string quickMessages[8] = {
    QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4,
    QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8
};

// Coordinates and dimensions for the buttons
int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;

// Loop to create and configure quick message buttons
for (int i = 0; i < 8; i++)
{
    if (!quickMessageButtons[i].Create(chart_id, "QuickMessageButton" + IntegerToString(i + 1), 0,
        startX + (i % 2) * (width + spacing),
        startY + (i / 2) * (height + spacing),
        startX + (i % 2) * (width + spacing) + width,
        startY + (i / 2) * (height + spacing) + height))
    {
        Print("Failed to create quick message button ", i + 1);
        return INIT_FAILED;
    }
    quickMessageButtons[i].Text(quickMessages[i]);
    adminPanel.Add(quickMessageButtons[i]);
}
```

Managing Message Length:

- The implementation of character length involves a character counter that tracks the number of characters typed into the input box and updates a label to display the current length alongside the maximum allowed message length. The **(OnInputChange)** function is triggered whenever the input text changes, retrieving the text, calculating its length using ( **StringLen)**, and then updating the ( **charCounter)** label with the format **"current\_length/MAX\_MESSAGE\_LENGTH**". This ensures users know how many characters they have left while composing their message, preventing them from exceeding the allowed limit. Optionally, I used 600 as maximum character length.

```
// Maximum number of characters allowed in a message
int MAX_MESSAGE_LENGTH = 600;

// Function to update the character counter
void OnInputChange()
{
    string text = inputBox.Text();
    int currentLength = StringLen(text);
    charCounter.Text(IntegerToString(currentLength) + "/" + IntegerToString(MAX_MESSAGE_LENGTH));
}
```

Our fully integrated program is here:

```
//+------------------------------------------------------------------+
//|                                          Admin Panel.mq5         |
//|                   Copyright 2024, Clemence Benjamin              |
//|       https://www.mql5.com/en/users/billionaire2024/seller       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property description "A responsive Admin Panel. Send messages to your telegram clients without leaving MT5"
#property version   "1.09"

#include <Trade\Trade.mqh>
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Label.mqh>  // Use CLabel for displaying text

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input string QuickMessage1 = "Updates";
input string QuickMessage2 = "Close all";
input string QuickMessage3 = "In deep profits";
input string QuickMessage4 = "Hold position";
input string QuickMessage5 = "Swing Entry";
input string QuickMessage6 = "Scalp Entry";
input string QuickMessage7 = "Book profit";
input string QuickMessage8 = "Invalid Signal";
input string InputChatId = "Enter Chat ID from Telegram bot API";  // User's Telegram chat ID
input string InputBotToken = "Enter BOT TOKEN from your Telegram bot"; // User's Telegram bot token

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
CDialog adminPanel;
CButton sendButton;
CButton clearButton;
CButton minimizeButton;
CButton maximizeButton;
CButton closeButton;
CButton quickMessageButtons[8];
CEdit inputBox;
CLabel charCounter;  // Use CLabel for the character counter
bool minimized = false;
int MAX_MESSAGE_LENGTH = 600;  // Maximum number of characters

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    long chart_id = ChartID();

    // Create the dialog
    if (!adminPanel.Create(chart_id, "Admin Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create dialog");
        return INIT_FAILED;
    }

    // Create the input box
    if (!inputBox.Create(chart_id, "InputBox", 0, 5, 5, 460, 75))
    {
        Print("Failed to create input box");
        return INIT_FAILED;
    }
    adminPanel.Add(inputBox);

    // Create the clear button for the input box
    if (!clearButton.Create(chart_id, "ClearButton", 0, 180, 75, 270, 105))
    {
        Print("Failed to create clear button");
        return INIT_FAILED;
    }
    clearButton.Text("Clear");
    adminPanel.Add(clearButton);

    // Create the send button for custom messages
    if (!sendButton.Create(chart_id, "SendButton", 0, 270, 75, 460, 105))
    {
        Print("Failed to create send button");
        return INIT_FAILED;
    }
    sendButton.Text("Send Message");
    adminPanel.Add(sendButton);

    // Create the character counter label
    if (!charCounter.Create(chart_id, "CharCounter", 0, 380, 110, 460, 130))
    {
        Print("Failed to create character counter label");
        return INIT_FAILED;
    }
    charCounter.Text("0/" + IntegerToString(MAX_MESSAGE_LENGTH));
    adminPanel.Add(charCounter);

    // Create the quick message buttons
    string quickMessages[8] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;

    for (int i = 0; i < 8; i++)
    {
        if (!quickMessageButtons[i].Create(chart_id, "QuickMessageButton" + IntegerToString(i + 1), 0, startX + (i % 2) * (width + spacing), startY + (i / 2) * (height + spacing), startX + (i % 2) * (width + spacing) + width, startY + (i / 2) * (height + spacing) + height))
        {
            Print("Failed to create quick message button ", i + 1);
            return INIT_FAILED;
        }
        quickMessageButtons[i].Text(quickMessages[i]);
        adminPanel.Add(quickMessageButtons[i]);
    }

    adminPanel.Show();
     // Create the minimize button
    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return INIT_FAILED;
    }
    minimizeButton.Text("_");
    adminPanel.Add(minimizeButton);

    // Create the maximize button
    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return INIT_FAILED;
    }
    maximizeButton.Text("[ ]");
    adminPanel.Add(maximizeButton);

    // Create the close button
    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return INIT_FAILED;
    }
    closeButton.Text("X");
    adminPanel.Add(closeButton);

    adminPanel.Show();

    // Enable chart events
    ChartSetInteger(ChartID(), CHART_EVENT_OBJECT_CREATE, true);
    ChartSetInteger(ChartID(), CHART_EVENT_OBJECT_DELETE, true);
    ChartSetInteger(ChartID(), CHART_EVENT_MOUSE_WHEEL, true);
    ChartRedraw();

    Print("Initialization complete");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    adminPanel.Destroy();
    Print("Deinitialization complete");
}

//+------------------------------------------------------------------+
//| Expert event handling function                                   |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    // Handle different types of events
    switch (id)
    {
        case CHARTEVENT_OBJECT_CLICK:
            if (sparam == "SendButton")
            {
                OnSendButtonClick();
            }
            else if (sparam == "ClearButton")
            {
                OnClearButtonClick();
            }
            else if (sparam == "MinimizeButton")
            {
                OnMinimizeButtonClick();
            }
            else if (sparam == "MaximizeButton")
            {
                OnMaximizeButtonClick();
            }
            else if (sparam == "CloseButton")
            {
                OnCloseButtonClick();
            }
            else if (StringFind(sparam, "QuickMessageButton") >= 0)
            {
                int index = StringToInteger(StringSubstr(sparam, StringLen("QuickMessageButton")));
                OnQuickMessageButtonClick(index - 1);
            }
            break;

        case CHARTEVENT_OBJECT_CHANGE:
            if (sparam == "InputBox")
            {
                OnInputChange();
            }
            break;

        default:
            break;
    }
}

//+------------------------------------------------------------------+
//| Function to handle custom message send button click              |
//+------------------------------------------------------------------+
void OnSendButtonClick()
{
    string message = inputBox.Text();
    if (message != "")
    {
        if (SendMessageToTelegram(message))
            Print("Custom message sent: ", message);
        else
            Print("Failed to send custom message.");
    }
    else
    {
        Print("No message entered.");
    }
}

//+------------------------------------------------------------------+
//| Function to handle clear button click                            |
//+------------------------------------------------------------------+
void OnClearButtonClick()
{
    inputBox.Text(""); // Clear the text in the input box
    OnInputChange();   // Update the character counter
    Print("Input box cleared.");
}

//+------------------------------------------------------------------+
//| Function to handle quick message button click                    |
//+------------------------------------------------------------------+
void OnQuickMessageButtonClick(int index)
{
    string quickMessages[8] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    string message = quickMessages[index];

    if (SendMessageToTelegram(message))
        Print("Quick Message Button Clicked - Quick message sent: ", message);
    else
        Print("Failed to send quick message.");
}

//+------------------------------------------------------------------+
//| Function to update the character counter                         |
//+------------------------------------------------------------------+
void OnInputChange()
{
    string text = inputBox.Text();
    int currentLength = StringLen(text);
    charCounter.Text(IntegerToString(currentLength) + "/" + IntegerToString(MAX_MESSAGE_LENGTH));
}
//+------------------------------------------------------------------+
//| Function to handle minimize button click                         |
//+------------------------------------------------------------------+
void OnMinimizeButtonClick()
{
    minimized = true;

    // Hide the full admin panel
    adminPanel.Hide();
    minimizeButton.Show();
    maximizeButton.Show();
    closeButton.Show();

}



//+------------------------------------------------------------------+
//| Function to handle maximize button click                         |
//+------------------------------------------------------------------+
void OnMaximizeButtonClick()
{
    if (minimized)
    {


        minimizeButton.Hide();
        maximizeButton.Hide();
        closeButton.Hide();
        adminPanel.Show();
    }

}

//+------------------------------------------------------------------+
//| Function to handle close button click                            |
//+------------------------------------------------------------------+
void OnCloseButtonClick()
{
    ExpertRemove(); // Completely remove the EA
    Print("Admin Panel closed.");
}

//+------------------------------------------------------------------+
//| Function to send the message to Telegram                         |
//+------------------------------------------------------------------+
bool SendMessageToTelegram(string message)
{
    // Use the input values for bot token and chat ID
    string botToken = InputBotToken;
    string chatId = InputChatId;

    string url = "https://api.telegram.org/bot" + botToken + "/sendMessage";
    char post_data[];

    // Prepare the message data
    string jsonMessage = "{\"chat_id\":\"" + chatId + "\", \"text\":\"" + message + "\"}";

    // Resize the character array to fit the JSON payload
    ArrayResize(post_data, StringToCharArray(jsonMessage, post_data));

    int timeout = 5000;
    char result[];
    string responseHeaders;

    // Make the WebRequest
    int res = WebRequest("POST", url, "Content-Type: application/json\r\n", timeout, post_data, result, responseHeaders);

    if (res == 200) // HTTP 200 OK
    {
        Print("Message sent successfully: ", message);
        return true;
    }
    else
    {
        Print("Failed to send message. HTTP code: ", res, " Error code: ", GetLastError());
        Print("Response: ", CharArrayToString(result));
        return false;
    }
}
```

Testing the advanced and responsive Admin Panel:

Here, I launched the Admin Panel, and it was performing well with little drawbacks. See the image below.

![Testing the advanced Admin Panel](https://c.mql5.com/2/92/terminal64_AZzKnfcLv9.gif)

Volatility 150s index: Admin Panel testing

Telegram integration worked fine, our messages are coming at a click!

![Telegram Quick Messages Incoming](https://c.mql5.com/2/92/Telegram_Quick_Messages.PNG)

Telegram Quick Messages Incoming

### Conclusion

In conclusion, the integration of responsiveness and quick messaging features into the Admin Panel Expert Advisor (EA) represents a significant enhancement in its utility and user experience. The newly added minimize, maximize, and close buttons provide a seamless and intuitive interface, allowing users to manage the panel’s visibility and operation with ease. These features ensure that the panel is not only functional but also adaptable to the user’s needs, whether they require a full view or a compact, unobtrusive display.

The implementation of quick messaging further streamlines communication by allowing users to send predefined messages instantly to their Telegram clients without leaving the MetaTrader 5 environment. This feature is particularly valuable in fast-paced trading scenarios where time is critical. The panel's ability to send custom messages, paired with the convenience of quick message buttons, empowers users to maintain effective communication with minimal disruption to their trading activities.

Overall, these enhancements make the Admin Panel a more powerful tool for traders and administrators, improving both efficiency and flexibility. This evolution reflects our commitment to providing solutions that address real-world challenges in algorithmic trading, ensuring that users can manage their trading operations with greater control and convenience.

More could be done, but today we have made it here. Attached is a source file below. Happy developing and trading, fellows!

[Back to the Top](https://www.mql5.com/en/articles/15418#content)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15418.zip "Download all attachments in the single ZIP archive")

[Admin\_Panel.mq5](https://www.mql5.com/en/articles/download/15418/admin_panel.mq5 "Download Admin_Panel.mq5")(11.83 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/472830)**
(2)


![Hamed Sadeghi](https://c.mql5.com/avatar/avatar_na2.png)

**[Hamed Sadeghi](https://www.mql5.com/en/users/sadeghi26187)**
\|
11 Sep 2024 at 22:26

This article is incredibly useful and highly practical. Thank you


![Clemence Benjamin](https://c.mql5.com/avatar/2025/3/67df27c6-2936.png)

**[Clemence Benjamin](https://www.mql5.com/en/users/billionaire2024)**
\|
12 Sep 2024 at 08:04

**Hamed Sadeghi [#](https://www.mql5.com/en/forum/472830#comment_54548767):**

This article is incredibly useful and highly practical. Thank you

Thank you for your valuable feedback, it's greatly appreciated!

![How to add Trailing Stop using Parabolic SAR](https://c.mql5.com/2/76/How_to_add_a_Trailing_Stop_using_the_Parabolic_SAR_indicator__LOGO.png)[How to add Trailing Stop using Parabolic SAR](https://www.mql5.com/en/articles/14782)

When creating a trading strategy, we need to test a variety of protective stop options. Here is where a dynamic pulling up of the Stop Loss level following the price comes to mind. The best candidate for this is the Parabolic SAR indicator. It is difficult to think of anything simpler and visually clearer.

![Example of Stochastic Optimization and Optimal Control](https://c.mql5.com/2/92/Example_of_Stochastic_Optimization_and_Optimal_Control__LOGO.png)[Example of Stochastic Optimization and Optimal Control](https://www.mql5.com/en/articles/15720)

This Expert Advisor, named SMOC (likely standing for Stochastic Model Optimal Control), is a simple example of an advanced algorithmic trading system for MetaTrader 5. It uses a combination of technical indicators, model predictive control, and dynamic risk management to make trading decisions. The EA incorporates adaptive parameters, volatility-based position sizing, and trend analysis to optimize its performance across varying market conditions.

![Neural Networks Made Easy (Part 87): Time Series Patching](https://c.mql5.com/2/76/Neural_networks_are_easy_fPart_87k____LOGO.png)[Neural Networks Made Easy (Part 87): Time Series Patching](https://www.mql5.com/en/articles/14798)

Forecasting plays an important role in time series analysis. In the new article, we will talk about the benefits of time series patching.

![Self Optimizing Expert Advisor with MQL5 And Python (Part III): Cracking The Boom 1000 Algorithm](https://c.mql5.com/2/92/Self_Optimizing_Expert_Advisor_with_MQL5_And_Python_Part_III____LOGO.png)[Self Optimizing Expert Advisor with MQL5 And Python (Part III): Cracking The Boom 1000 Algorithm](https://www.mql5.com/en/articles/15781)

In this series of articles, we discuss how we can build Expert Advisors capable of autonomously adjusting themselves to dynamic market conditions. In today's article, we will attempt to tune a deep neural network to Deriv's synthetic markets.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=aqzikjowaajlejibiduqjwjyejlceiqy&ssn=1769157570338615413&ssn_dr=0&ssn_sr=0&fv_date=1769157570&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15418&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20Trading%20Administrator%20Panel%20in%20MQL5%20(Part%20II)%3A%20Enhancing%20Responsiveness%20and%20Quick%20Messaging%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915757076181207&fz_uniq=5062624444888819137&sv=2552)

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