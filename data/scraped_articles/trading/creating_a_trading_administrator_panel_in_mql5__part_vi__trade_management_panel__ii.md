---
title: Creating a Trading Administrator Panel in MQL5 (Part VI):Trade Management Panel (II)
url: https://www.mql5.com/en/articles/16328
categories: Trading, Integration
relevance_score: 6
scraped_at: 2026-01-22T17:59:48.040171
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/16328&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049544302377413985)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/16328#para1)
- [Enhancing The Trade Management Panel](https://www.mql5.com/en/articles/16328#para2)

  - [Adjusting The Trade Management Panel Layout](https://www.mql5.com/en/articles/16328#para3)
  - [Helper Function and New button Integration](https://www.mql5.com/en/articles/16328#para4)

- [Coding Button Handlers](https://www.mql5.com/en/articles/16328#para5)
- [Testing](https://www.mql5.com/en/articles/16328#para6)
- [Conclusion](https://www.mql5.com/en/articles/16328#para7)


### Introduction

Building on our ability to pass security tests and access the Admin Home Panel, where we have primary access to key features, our goal today is to continue promoting the implementation of MQL5 as we enhance the performance of our multi-functional interface in an actively developed project. Our previous publication served as an introduction to the multiple features of the panel, unveiling the latest capabilities without detailing the functionality of each button within the Trade Management Panel. We will ensure that every button in the Trade Management Panel produces a response upon clicking.

Additionally, I would like to address a question: Are we limited by the current panel layout? To explore this, we will consider alternative layout ideas that can help us better utilize the capabilities of the dialog class in the renowned trading algorithm development language, MQL5. Below, I have included an image demonstrating the ability to adjust the vertical scale of the chart, which I find useful for maximizing the space available for our panel. I also believe it would be beneficial to allow both panels to be displayed simultaneously while still maintaining a clear view of the chart. To facilitate this feature, we will add a button in the Admin Home Panel that enables users to view all panels at once.

![MetaTrader 5 Vertical Space](https://c.mql5.com/2/129/terminal64_moRZft4Hsf__3.gif)

MetaTrader 5 chart vertical scale creates spaces and makes the view of candlestick range clearer

Before we proceed further, I want to give everyone an opportunity to reconnect with the various elements of our multi-functional interface. This will help address some of the unspoken challenges encountered while working through this article series.

1. For those experiencing compilation errors, it's likely due to running the code with the default library files that haven't been extended. To resolve this issue, make sure to overwrite the contents of the MQL5/Include/controls directory with the files from the [Extended Header](https://www.mql5.com/en/articles/download/16045/extended_mql5_header_files.zip) folder provided in the [previous articles.](https://www.mql5.com/en/articles/16045)
2. The other challenge involves ensuring that the chat ID and bot token are updated to match your own, so that the verification code is sent to your account.

For the second challenge, the solution involves accessing the Telegram app to create two bots: one dedicated to authentication during application launch and the other for communication through individual chats, groups, and channels within the Communications Panel of the EA program.

Below, please enter your actual chat ID and bot token. Ensure that the chat ID is not from a group or channel, as doing so would share your code with multiple users; it should remain highly personal. To verify that you have the correct chat ID, consider starting a chat with the bot and checking the API for the chat ID. Keep in mind that these values are hard-coded into your source code before compilation, and they differ from the chat ID and bot token you enter when launching the application.

We [previously](https://www.mql5.com/en/articles/16240) discussed this in our earlier writings, and the steps are easy to follow.

STEP 1:

Ensure that you are a registered Telegram user. You can do this by downloading the [Telegram app for your phone or desktop](https://www.mql5.com/go?link=https://api.telegram.org/ "https://api.telegram.org/") and then following the in-app instructions to sign up.

STEP 2:

Start a chat with BotFather to create two bots: one for two-factor authentication (2FA) and the other for communications. You will receive a unique bot token for each bot, which can be used to access the Telegram API and obtain chat IDs. Each bot has its own distinct token, and you're free to name them as you wish, ensuring you can easily identify their purposes.

STEP 3:

Access the [API](https://www.mql5.com/go?link=https://api.telegram.org/ "https://api.telegram.org/") by clicking on the following link: https://api.telegram.org/botReplaceMeWithTheTokenFromTheBotFather/getUpdates.

![Empty API](https://c.mql5.com/2/129/API_empty1__3.png)

[Telegram API](https://www.mql5.com/go?link=https://api.telegram.org/ "https://api.telegram.org/") accessed using [Chrome Browser](https://www.mql5.com/go?link=https://www.googleadservices.com/pagead/aclk?sa=L%26ai=DChcSEwjcrqOR6uWJAxU7k1AGHXT_NosYABACGgJkZw%26co=1%26ase=2%26gclid=Cj0KCQiA6Ou5BhCrARIsAPoTxrDEZJANmoBot0usPzUKkJ3ITYHna3_hwg708Mzlvi6WfCb7FPF8IjQaArp8EALw_wcB%26ohost=www.google.com%26cid=CAESV-D2m7mzu_bHJPcOBdoqN46mSJXXi9e7Gni-v_xfGffijnof45Dk7fJ9EIpJPIBPyiaiCaqE7xv4IZRU-U-rM4q-GFne8ykDnvAYnjl21VEJABCgIieMzg%26sig=AOD64_3CxDDdOSLGgCo_Ax8GS2kJ4rDapg%26q%26nis=4%26adurl%26ved=2ahUKEwisgZ-R6uWJAxWmWEEAHeJcNboQqyQoAXoECAoQDQ "https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwjcrqOR6uWJAxU7k1AGHXT_NosYABACGgJkZw&co=1&ase=2&gclid=Cj0KCQiA6Ou5BhCrARIsAPoTxrDEZJANmoBot0usPzUKkJ3ITYHna3_hwg708Mzlvi6WfCb7FPF8IjQaArp8EALw_wcB&ohost=www.google.com&cid=CAESV-D2m7mzu_bHJPcOBdoqN46mSJXXi9e7Gni-v_xfGffijnof45Dk7fJ9EIpJPIBPyiaiCaqE7xv4IZRU-U-rM4q-GFne8ykDnvAYnjl21VEJABCgIieMzg&sig=AOD64_3CxDDdOSLGgCo_Ax8GS2kJ4rDapg&q&nis=4&adurl&ved=2ahUKEwisgZ-R6uWJAxWmWEEAHeJcNboQqyQoAXoECAoQDQ")

STEP 4:

Start a conversation with your dedicated bot for 2FA code delivery. You can also add the communications bot to any groups or channels where you intend to share trading insights and set it as an admin to ensure it functions properly. Once you’ve completed the setup, initiate a conversation within the groups to retrieve the chat ID. Afterward, refresh your browser in the API tab to see the updated information.

![Telegram Message sent to a bot](https://c.mql5.com/2/129/Telegram_Ms__3.png)

Telegram Chat: Start a conversation with the bot (In this case I named my bot Admin Panel)

![API messages](https://c.mql5.com/2/129/Api1__3.png)

Telegram API : Here we can obtain our Chat ID for 2FA

In the image above, I initiated a personal chat, primarily to obtain my chat ID for 2FA delivery. You can also add the same bot to groups and channels; when you start conversations there, the chat IDs will automatically populate in the API, just as shown above, typically appearing below the first conversation. For example, I added the bot to one of my educational channels, and the API message I received is displayed in the image below.

To summarize, the same bot can be used for multiple channels, each with its own unique chat ID. Alternatively, you may choose to use a different bot for each channel if you prefer.

![Adding your bot as Admin in a Channel](https://c.mql5.com/2/129/Telegram_6gClRhTkKx__3.gif)

Adding bot as admin in a Telegram Channel

Finally, the API JSON with the channel chat ID is shown below.

![API JASON](https://c.mql5.com/2/129/API_Jason1__3.png)

Channel Chat generated JASON

This code snippet is where you should input the bot credentials for 2FA code delivery:

```
// Constants for 2FA. Here put the chat ID dedicated for verification code delivery. It must be kept secure always.
const string Hardcoded2FAChatId = "REPLACE WITH YOUR CHAT ID";
const string Hardcoded2FABotToken = "REPLACE WITH YOU BOT TOKEN";  //Obtain your bot token from a telegram bot father.

//If you don't already have a bot you can start the chat with the bot father to create one.
```

Below is an image that illustrates where to input the Telegram credentials for trading communications during app initialization. You can also customize quick messages as needed:

![input launch settings](https://c.mql5.com/2/129/Launch_input_settings__5.png)

Initialization inputs settings explained

Please note that the code snippet and image above display two different chat IDs, as mentioned earlier. One is used for verification code delivery, while the other is for broadcasting administrative communications through personal chats or Telegram groups. Each user, group, and channel connected to the bot has a unique chat ID.

Finally, we have been using the same PIN, 2024, throughout these discussions; however, you can customize it for your own use within the source code. For 2FA, the six-digit code generator algorithm will handle this automatically; you only need to ensure you have your unique chat ID for the secure delivery of the code.

With this brief recap, I hope you are prepared and ready for today's progress. In summary, we will focus on the handlers for the Trade Management Buttons and then enhance their layout regarding position and dimensions on the chart.

### Enhancing The Trade Management Panel

First, I would like you to understand the difference between the Communication Panel buttons and the Trading Management Panel. I know that most of us are familiar with the bulk operations functionality available in both the desktop and mobile versions of MetaTrader 5.

As the panel names suggest, their functions are self-explanatory; however, you may find them confusing due to the similar titles or descriptions. For example, the buttons in the Communication Panel are solely for communication, while those in the Trading Management Panel are specifically programmed for handling trades.

It is possible to implement trade handlers on the same buttons used in the Communication Panel, allowing for both tasks to be executed simultaneously when clicked. However, there are several reasons to keep the Communication buttons distinct from the Trade Management buttons. One key reason is that the Admin might want to manage trades that are not necessarily intended for broadcasting.

Please see the images below, which highlight the most significant differences.

![Communications Panel](https://c.mql5.com/2/129/new_Interface__6.PNG)

Communications Panel: Messaging buttons

Almost all the buttons here are designed for communication, except for navigation buttons. Additionally, the quick messages they send can be customized during initialization. The 'Close All' quick message button is designed solely to send the message to the designated Telegram client and does not perform any other actions.

![Trade Management Panel](https://c.mql5.com/2/129/Trade_Management_Panel__3.PNG)

Trade Management Panel: (These buttons must execute trade operations when clicked)

From the images above, we have clarified the purpose of each functionality. Now, turning our attention to the Trade Management Panel, there is considerable work to be done. Our panel currently has limited buttons, and we aim to enhance it to maximize space for chart viewing. New buttons are essential for some critical trade operations, such as a 'Close All Orders' button.

### Ajusting The Trade Management Layout

The new interface layout is an innovative design aimed at creating more space for chart visibility while still allowing for the usual administrative operations within this development series. This approach leverages the position coordinates of the panels in relation to the MetaTrader 5 chart.

Currently, the Admin Home Panel is too large for its content, and we will also be adjusting it. To make the process more presentable, we will break it down into three sections.

**Adjusting The Trade Management Panel**

During initialization, our panel is created using the following piece of code:

```
if (!tradeManagementPanel.Create(ChartID(), "Trade Management Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communictions panel dialog");
        return INIT_FAILED;
    }
```

In the above code, our:

x1 =30

x2 =500

Therefore, width = X2 -X1 = 500-30 = 470.

If we increase our X2 value, the new width will expand in accordance with the design concept. For example, I will increase it by 50% of the original width. Please see the code snippet below.

```
// Let's increase our panel width by 50% of the former panel and is likely to be 250px but we will add an extra 30px to cover for the initial px on x1 co-ordinate.
// For the y co-ordinate we will reduce so much to 150px
if (!tradeManagementPanel.Create(ChartID(), "Trade Management Panel", 0, 30, 30, 780, 150))
    {
        Print("Failed to create Communictions panel dialog");
        return INIT_FAILED;
    }
```

The result after compiling and testing is as follows:

![new panel layout](https://c.mql5.com/2/129/new_Interface__7.PNG)

New Trading Panel Layout

The recent change left some of the buttons out of our desired order, causing them to float over the chart instead of remaining within the panel borders. To resolve this issue, we will navigate to the CreateTradeManagementControls() function and adjust the button coordinates as shown in the code snippet below

```
// Create the Trade Management Panel controls
// Here we adjusted our button coordinates to fit well in the new Trade Management Panel
bool CreateTradeManagementControls()
{
    long chart_id = ChartID();

    // Buy Button
    if (!buyButton.Create(chart_id, "BuyButton", 0, 130, 5, 210, 40))
    {
        Print("Failed to create Buy button");
        return false;
    }
    buyButton.Text("Buy");
    tradeManagementPanel.Add(buyButton);

    // Sell Button
    if (!sellButton.Create(chart_id, "SellButton", 0, 220, 5, 320, 40))
    {
        Print("Failed to create Sell button");
        return false;
    }
    sellButton.Text("Sell");
    tradeManagementPanel.Add(sellButton);

    // Close Position Button
    if (!closePosButton.Create(chart_id, "ClosePosButton", 0, 130, 50, 260, 70))
    {
        Print("Failed to create Close Position button");
        return false;
    }
    closePosButton.Text("Close Position");
    tradeManagementPanel.Add(closePosButton);

    // Modify Position Button
    if (!modifyPosButton.Create(chart_id, "ModifyPosButton", 0, 270, 50, 410, 70))
    {
        Print("Failed to create Modify Position button");
        return false;
    }
    modifyPosButton.Text("Modify Position");
    tradeManagementPanel.Add(modifyPosButton);

    // Set Stop-Loss Button
    if (!setSLButton.Create(chart_id, "SetSLButton", 0, 330, 5, 430, 40))
    {
        Print("Failed to create Set Stop-Loss button");
        return false;
    }
    setSLButton.Text("Set SL");
    tradeManagementPanel.Add(setSLButton);

    // Set Take-Profit Button
    if (!setTPButton.Create(chart_id, "SetTPButton", 0, 440, 5, 540, 40))
    {
        Print("Failed to create Set Take-Profit button");
        return false;
    }
    setTPButton.Text("Set TP");
    tradeManagementPanel.Add(setTPButton);

    return true;
}
```

Compiling and running the code produced the following image, with all the buttons now properly placed.

![Buttons now well arranged.](https://c.mql5.com/2/129/new_Interface_buttons__3.PNG)

Trade Management Panel: Buttons well arranged

Now that we have completed the layout changes and arranged the buttons, let's move on to editing the existing buttons and adding new ones.

Another important consideration is to avoid conflicts with the internal buttons. For example, see the image below, which illustrates how the native buttons overlap with our admin panel.

![](https://c.mql5.com/2/129/ShareX_d7gBsCJE3k__3.gif)

Quick buttons overlapping admin panel

To address this issue, we will apply a shift to our x-coordinates, moving them to the right to translate the panel's position. Please see the updated value in the code snippet below:

```
//1
//2
//2
if (!tradeManagementPanel.Create(ChartID(), "Trade Management Panel", 0, 260, 30, 1040, 150))
    {
        Print("Failed to create Communictions panel dialog");
        return INIT_FAILED;
    }
```

The compilation was successful, and the problem has been resolved. See the layout below:

![After shifting the panel along x-Axis](https://c.mql5.com/2/129/terminal64_XEXWI4Aefp__3.gif)

The Quick buttons well arranged and not overlapping

### Helper Function And New Button Creation

During my C++ classes, my lecturer often emphasized the importance of the repeat function, considering it a fundamental concept for anyone aspiring to call themselves a programmer. Later, while researching helper functions in MQL5, I initially thought I had stumbled upon something similar to C++'s repeat function. However, I soon realized that while these functions share certain advantages, they serve distinct purposes.

To summarize, MQL5 helper functions can be likened to "function objects" that parameterize behavior, whereas C++ repetition utilities are more about applying consistent transformations or patterns.

Here are some similarities between the two:

- Both aim to reduce repetitive code.
- Both enhance code clarity and maintainability.

In this discussion, we introduced the concept of Helper Functions as we prepare to implement multiple buttons to keep our work smart and presentable. Stay tuned as we delve into its implementation!

One feature I particularly enjoy using when managing multiple trades is the ability to streamline the process, allowing me to handle numerous trades efficiently all at once.

![Bulk Trade Operations](https://c.mql5.com/2/129/Bulk_Operations__3.png)

Bulk Operations: Used in MetaTrader 5 to manage many trades

We will prioritize trading operations that require a single click to execute. Later, we will shift our focus to trade modification features, such as adjusting stop levels and configuring them directly from the admin panel. From the image above, we can also draw inspiration for button names to use in our project.

Below is an outline of the buttons:

- Close all Positions
- Close Profitable Positions
- Close Losing Positions
- Close Buy Positions
- Close Sell Positions
- Delete All Orders
- Delete Limit Orders
- Delete Stop Orders
- Delete Stop Limit Orders

By now, we are already familiar with buttons from our previous discussions. The procedure for creating individual buttons is largely the same, differing only in their coordinates. To minimize repetitive tasks, we utilize a Helper Function, which is custom-built to streamline the process. This approach reduces the size of the code and efficiently incorporates all the new buttons.

First, we declare our buttons as global variables:

```
// Button Declarations
CButton buyButton;                 // Button for Buy operations
CButton sellButton;                // Button for Sell operations
CButton closeAllButton;            // Button for closing all positions
CButton closeProfitButton;         // Button for closing profitable positions
CButton closeLossButton;           // Button for closing losing positions
CButton closeBuyButton;            // Button for closing Buy positions
CButton closeSellButton;           // Button for closing Sell positions
CButton deleteAllOrdersButton;     // Button for deleting all orders
CButton deleteLimitOrdersButton;   // Button for deleting limit orders
CButton deleteStopOrdersButton;    // Button for deleting stop orders
CButton deleteStopLimitOrdersButton; // Button for deleting stop limit orders

```

With our declarations complete, it's time to roll out the helper function for creating those buttons, as I have implemented it. The helper function, CreateButton, is a streamlined utility designed to simplify the repetitive task of creating and configuring buttons in the Trade Management Panel.

This function takes parameters for the button's reference, name, label text, and coordinates while handling the underlying creation and setup logic, including error handling. By centralizing this process, we eliminate redundant code and ensure that all buttons are created consistently with minimal effort.

This modularity is crucial because it enhances code readability and maintainability, making it easier to extend the panel with additional buttons or adjust functionality in one central location instead of across multiple instances. Essentially, the helper function acts as a bridge between the panel design and the button creation process, ensuring seamless integration.

Here is the code for our helper function:

```
//Helper Function For seamless Button creation
bool CreateButton(CButton &button, const string name, const string text, int x1, int y1, int x2, int y2)
{
    long chart_id = ChartID();

    if (!button.Create(chart_id, name, 0, x1, y1, x2, y2))
    {
        Print("Failed to create button: ", name);
        return false;
    }

    button.Text(text);
    tradeManagementPanel.Add(button);
    return true;
}
```

The above code snippet will encapsulate the whole button creation process. The CreateTradeManagementControls function serves as the master organizer, calling CreateButton repeatedly to define and position each button logically within the Trade Management Panel. Instead of duplicating button creation logic for each control, this function focuses solely on specifying unique details like coordinates, labels, and button types.

The modular design enabled by CreateButton keeps this higher-level function concise and focused on its primary purpose: structuring the layout of the panel. Together, these two functions work in harmony—CreateTradeManagementControls handles the structure while delegating repetitive tasks to CreateButton—to produce a clean, efficient, and adaptable implementation of the Trade Management Panel. And the code for all the buttons is right here

```
//+------------------------------------------------------------------+
//| Create Trade Management Controls                                 |
//+------------------------------------------------------------------+
bool CreateTradeManagementControls()
{
    // Coordinates for buttons (adjust as needed)
    const int Y1_TOP = 5, Y2_TOP = 40;
    const int Y1_MID = 50, Y2_MID = 70;
    const int Y1_BOTTOM = 80, Y2_BOTTOM = 100;

    // Buy Button
    if (!CreateButton(buyButton, "BuyButton", "Buy", 130, Y1_TOP, 210, Y2_TOP)) return false;

    // Sell Button
    if (!CreateButton(sellButton, "SellButton", "Sell", 220, Y1_TOP, 320, Y2_TOP)) return false;

    // Close All Positions Button
    if (!CreateButton(closeAllButton, "CloseAllButton", "Close All", 130, Y1_MID, 230, Y2_MID)) return false;

    // Close Profitable Positions Button
    if (!CreateButton(closeProfitButton, "CloseProfitButton", "Close Profitable", 240, Y1_MID, 380, Y2_MID)) return false;

    // Close Losing Positions Button
    if (!CreateButton(closeLossButton, "CloseLossButton", "Close Losing", 390, Y1_MID, 510, Y2_MID)) return false;

    // Close Buy Positions Button
    if (!CreateButton(closeBuyButton, "CloseBuyButton", "Close Buys", 520, Y1_MID, 620, Y2_MID)) return false;

    // Close Sell Positions Button
    if (!CreateButton(closeSellButton, "CloseSellButton", "Close Sells", 630, Y1_MID, 730, Y2_MID)) return false;

    // Delete All Orders Button
    if (!CreateButton(deleteAllOrdersButton, "DeleteAllOrdersButton", "Delete All Orders", 130, Y1_BOTTOM , 270, Y2_BOTTOM )) return false;

    // Delete Limit Orders Button
    if (!CreateButton(deleteLimitOrdersButton, "DeleteLimitOrdersButton", "Delete Limits", 275, Y1_BOTTOM , 385, Y2_BOTTOM )) return false;

    // Delete Stop Orders Button
    if (!CreateButton(deleteStopOrdersButton, "DeleteStopOrdersButton", "Delete Stops", 390, Y1_BOTTOM , 515, Y2_BOTTOM )) return false;

    // Delete Stop Limit Orders Button
    if (!CreateButton(deleteStopLimitOrdersButton, "DeleteStopLimitOrdersButton", "Delete Stop Limits", 520, Y1_BOTTOM , 660, Y2_BOTTOM )) return false;

    return true; // All buttons created successfully
}
```

Here is the outcome of the new layout

![New Layout Trade Management Panel](https://c.mql5.com/2/129/terminal64_xKjfkqlXDg__3.gif)

New Layout of after new buttons integration

During the process of adding new buttons, some older ones were removed to maintain uniformity. For now, our focus is on operations requiring instant execution, without the need to input additional data—for example, tasks like order modification

### Coding Button Handlers

To enhance the functionality of the Trade Management Panel, we implemented dedicated handler functions for each button to enable specific trading operations. Here's an explanation for each code snippet:

1\. Buy Button Handler ( _OnBuyButtonClick_)

The _OnBuyButtonClick_ function enables the creation of a market order to buy the specified asset. By utilizing the _CTrade_ class, it handles the essential trade parameters such as lot size, slippage, stop loss, and take profit, ensuring precise execution. This is critical for traders who want to quickly open buy positions in a programmatically controlled environment.

```
//+------------------------------------------------------------------+
//| Handle Buy button click                                          |
//+------------------------------------------------------------------+
void OnBuyButtonClick()
{
    CTrade trade;
    double lotSize = 0.1; // Example lot size
    double slippage = 3;  // Example slippage
    double stopLoss = 0;  // Example stop loss (in points)
    double takeProfit = 0; // Example take profit (in points)

    // Open Buy order
 double askPrice;
if (SymbolInfoDouble(Symbol(), SYMBOL_ASK, askPrice) && askPrice > 0)
{
    if (trade.Buy(lotSize, Symbol(), askPrice, slippage, stopLoss, takeProfit))
    {
        Print("Buy order executed successfully.");
    }
    else
    {
        Print("Failed to execute Buy order. Error: ", GetLastError());
    }
}
else
{
    Print("Failed to retrieve Ask price. Error: ", GetLastError());
}

    // Execute Buy order logic here
    Print("Executing Buy operation");
}
```

2\. Sell Button Handler ( _OnSellButtonClick_)

The _OnSellButtonClick_ function mirrors the buy handler, enabling the user to sell an asset via a market order. By structuring the sell logic, it ensures consistent parameter handling, such as lot size and slippage, making the trading panel efficient for initiating sell orders on demand.

```
//+------------------------------------------------------------------+
//| Handle Sell button click                                         |
//+------------------------------------------------------------------+
void OnSellButtonClick()
{
    CTrade trade;
    double lotSize = 0.1; // Example lot size
    double slippage = 3;  // Example slippage
    double stopLoss = 0;  // Example stop loss (in points)
    double takeProfit = 0; // Example take profit (in points)

    double bidPrice;
if (SymbolInfoDouble(Symbol(), SYMBOL_BID, bidPrice) && bidPrice > 0)
{
    // Open Sell order
    if (trade.Sell(lotSize, Symbol(), bidPrice, slippage, stopLoss, takeProfit))
    {
        Print("Sell order opened successfully.");
    }
    else
    {
        Print("Error opening sell order: ", trade.ResultRetcode());
    }
}
else
{
    Print("Failed to retrieve Bid price. Error: ", GetLastError());
}
}
```

3\. Close All Positions Handler ( _OnCloseAllButtonClick_)

This function automates the closure of all active positions, iterating through open trades and using _CTrade_. _PositionClose_ for execution. It's particularly useful for traders seeking to exit all trades quickly, safeguarding against sudden market volatility or fulfilling a strategy's exit requirements.

```
//+------------------------------------------------------------------+
//| Handle Close All button click                                    |
//+------------------------------------------------------------------+
void OnCloseAllButtonClick()
{
    CPositionInfo position;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        if (position.SelectByIndex(i))
        {
            CTrade trade;
            if (position.Type() == POSITION_TYPE_BUY)
                trade.PositionClose(position.Ticket());
            else if (position.Type() == POSITION_TYPE_SELL)
                trade.PositionClose(position.Ticket());
        }
    }
    Print("All positions closed.");
}
```

4\. Close Profitable Positions Handler ( _OnCloseProfitButtonClick_)

With the _OnCloseProfitButtonClick_ function, traders can secure gains by closing only profitable positions. It filters trades based on their profit value and ensures selective closures, aligning with strategies focused on locking in profits while keeping loss-making trades for further evaluation.

```
//+------------------------------------------------------------------+
//| Handle Close Profitable button click                             |
//+------------------------------------------------------------------+
void OnCloseProfitButtonClick()
{
    CPositionInfo position;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        if (position.SelectByIndex(i) && position.Profit() > 0)
        {
            CTrade trade;
            if (position.Type() == POSITION_TYPE_BUY)
                trade.PositionClose(position.Ticket());
            else if (position.Type() == POSITION_TYPE_SELL)
                trade.PositionClose(position.Ticket());
        }
    }
    Print("Profitable positions closed.");
}
```

5\. Close Losing Positions Handler ( _OnCloseLossButtonClick_)

This handler provides a risk-management tool by closing all positions that are incurring losses. By targeting only negative profit trades, it helps in mitigating further drawdowns, which is vital for maintaining account equity and adhering to predefined loss limits.

```
//+------------------------------------------------------------------+
//| Handle Close Losing button click                                 |
//+------------------------------------------------------------------+
void OnCloseLossButtonClick()
{
    CPositionInfo position;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        if (position.SelectByIndex(i) && position.Profit() < 0)
        {
            CTrade trade;
            if (position.Type() == POSITION_TYPE_BUY)
                trade.PositionClose(position.Ticket());
            else if (position.Type() == POSITION_TYPE_SELL)
                trade.PositionClose(position.Ticket());
        }
    }
    Print("Losing positions closed.");
}

void OnCloseBuyButtonClick()
{
    // Close Buy positions logic
    Print("Closing Buy positions");
}

void OnCloseSellButtonClick()
{
    // Close Sell positions logic
    Print("Closing Sell positions");
}
```

6\. Delete All Orders Handler ( _OnDeleteAllOrdersButtonClick_)

This function deletes all pending orders, ensuring no residual limit or stop orders affect the account. By utilizing the COrderInfo class to retrieve and cancel orders, it helps maintain a clean order book and prevents unintended executions.

```
//+------------------------------------------------------------------+
//| Handle Delete All Orders button click                            |
//+------------------------------------------------------------------+
void OnDeleteAllOrdersButtonClick()
{
    COrderInfo order;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (order.SelectByIndex(i))
        {
            CTrade trade;
            trade.OrderDelete(order.Ticket());
        }
    }
    Print("All orders deleted.");
}
```

7\. Delete Limit Orders Handler ( _OnDeleteLimitOrdersButtonClick)_

The _OnDeleteLimitOrdersButtonClick_ function focuses exclusively on canceling limit orders. This is essential for traders who need to adjust their strategy while preserving stop or other order types, allowing precise control over order management.

```
//+------------------------------------------------------------------+
//| Handle Delete Limit Orders button click                          |
//+------------------------------------------------------------------+
void OnDeleteLimitOrdersButtonClick()
{
    COrderInfo order;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (order.SelectByIndex(i) && order.Type() == ORDER_TYPE_BUY_STOP_LIMIT)
        {
            CTrade trade;
            trade.OrderDelete(order.Ticket());
        }
    }
    Print("All limit orders deleted.");
}
```

8\. Delete Stop Orders Handler ( _OnDeleteStopOrdersButtonClick_)

This handler targets the removal of all stop orders, ensuring they do not trigger undesired trades in volatile markets. By isolating stop orders, it provides traders with a granular level of control over pending order management.

```
//+------------------------------------------------------------------+
//| Handle Delete Stop Orders button click                           |
//+------------------------------------------------------------------+
void OnDeleteStopOrdersButtonClick()
{
    COrderInfo order;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (order.SelectByIndex(i) && order.Type() == ORDER_TYPE_BUY_STOP && ORDER_TYPE_SELL_STOP)
        {
            CTrade trade;
            trade.OrderDelete(order.Ticket());
        }
    }
    Print("All stop orders deleted.");
}
```

9\. Delete Stop Limit Orders Handler ( _OnDeleteStopLimitOrdersButtonClick_)

The function manages the deletion of stop-limit orders, useful for strategies that involve hybrid order types. It reinforces flexibility and aligns order handling with the trader's updated market outlook or strategy changes.

```
//+------------------------------------------------------------------+
//| Handle Delete Stop Limit Orders button click                     |
//+------------------------------------------------------------------+
void OnDeleteStopLimitOrdersButtonClick()
{
    COrderInfo order;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (order.SelectByIndex(i) && order.Type() == ORDER_TYPE_BUY_LIMIT && ORDER_TYPE_SELL_STOP_LIMIT)
        {
            CTrade trade;
            trade.OrderDelete(order.Ticket());
        }
    }
    Print("All stop limit orders deleted.");
}
```

**Integration into [OnChartEvent:](https://www.mql5.com/en/docs/standardlibrary/controls/cwnd/cwndonevent)**

To connect button clicks to these functions, we integrate their calls within the _OnChartEvent_ function. By linking the button's _sparam_ value with its corresponding handler, the program ensures seamless interaction between the graphical user interface and backend trade logic, making the panel responsive and user-friendly.

```
//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if (id == CHARTEVENT_OBJECT_CLICK)
    {
        // Panel navigation buttons
        if (sparam == "HomeButtonComm")
        {
            adminHomePanel.Show();
            communicationsPanel.Hide();
        }
        else if (sparam == "HomeButtonTrade")
        {
            adminHomePanel.Show();
            tradeManagementPanel.Hide();
        }

        if (sparam == "TradeMgmtAccessButton")
        {
            tradeManagementPanel.Show();
            adminHomePanel.Hide();
        }
        else if (sparam == "CommunicationsPanelAccessButton")
        {
            communicationsPanel.Show();
            adminHomePanel.Hide();
        }

        // Control buttons for panel resizing and closing
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
            ExpertRemove();

        }
        {
        if (sparam == "LoginButton")
        {
            OnLoginButtonClick();
        }
        else if (sparam == "CloseAuthButton")
        {
            OnCloseAuthButtonClick();
        }
        else if (sparam == "TwoFALoginButton")
        {
            OnTwoFALoginButtonClick();
        }
        else if (sparam == "Close2FAButton")
        {
            OnClose2FAButtonClick();
        }
    }

    switch (id)
    {
        case CHARTEVENT_OBJECT_CLICK:
            if (sparam == "SendButton") OnSendButtonClick();
            else if (sparam == "ClearButton") OnClearButtonClick();
            else if (sparam == "ChangeFontButton") OnChangeFontButtonClick();
            else if (sparam == "ToggleThemeButton") OnToggleThemeButtonClick();
            else if (sparam == "MinimizeButton") OnMinimizeButtonClick();
            else if (sparam == "MaximizeButton") OnMaximizeButtonClick();
            else if (sparam == "CloseButton") OnCloseButtonClick();
            else if (StringFind(sparam, "QuickMessageButton") != -1)
            {
                long index = StringToInteger(StringSubstr(sparam, 18));
                OnQuickMessageButtonClick(index - 1);
            }
            break;

        case CHARTEVENT_OBJECT_ENDEDIT:
            if (sparam == "InputBox") OnInputChange();
            break;
    }

  }

        // Trade management buttons
        if (sparam == "BuyButton") OnBuyButtonClick();
        else if (sparam == "SellButton") OnSellButtonClick();
        else if (sparam == "CloseAllButton") OnCloseAllButtonClick();
        else if (sparam == "CloseProfitButton") OnCloseProfitButtonClick();
        else if (sparam == "CloseLossButton") OnCloseLossButtonClick();
        else if (sparam == "CloseBuyButton") OnCloseBuyButtonClick();
        else if (sparam == "CloseSellButton") OnCloseSellButtonClick();
        else if (sparam == "DeleteAllOrdersButton") OnDeleteAllOrdersButtonClick();
        else if (sparam == "DeleteLimitOrdersButton") OnDeleteLimitOrdersButtonClick();
        else if (sparam == "DeleteStopOrdersButton") OnDeleteStopOrdersButtonClick();
        else if (sparam == "DeleteStopLimitOrdersButton") OnDeleteStopLimitOrdersButtonClick();
    }
```

Then our final code is here and we now have many lines:

```
//+------------------------------------------------------------------+
//|                                             Admin Panel.mq5      |
//|                           Copyright 2024, Clemence Benjamin      |
//|        https://www.mql5.com/en/users/billionaire2024/seller      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property description "A secure and responsive Admin Panel. Send messages to your telegram clients without leaving MT5"
#property version   "1.22"

#include <Trade\Trade.mqh>
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Label.mqh>

// Input parameters for quick messages
input string QuickMessage1 = "Updates";
input string QuickMessage2 = "Close all";
input string QuickMessage3 = "In deep profits";
input string QuickMessage4 = "Hold position";
input string QuickMessage5 = "Swing Entry";
input string QuickMessage6 = "Scalp Entry";
input string QuickMessage7 = "Book profit";
input string QuickMessage8 = "Invalid Signal";
input string InputChatId = "YOUR_CHAT_ID";
input string InputBotToken = "YOUR_BOT_TOKEN";

// Constants for 2FA
const string Hardcoded2FAChatId = "Replace chat ID with yours";
const string Hardcoded2FABotToken = "Replace with your bot token";

// Global variables
CDialog adminHomePanel, tradeManagementPanel, communicationsPanel;
CDialog authentication, twoFactorAuth;
CButton homeButtonComm, homeButtonTrade;

CButton sendButton, clearButton, changeFontButton, toggleThemeButton;
CButton loginButton, closeAuthButton, twoFALoginButton, close2FAButton;
CButton quickMessageButtons[8], minimizeButton, maximizeButton, closeButton;
CButton tradeMgmtAccessButton, communicationsPanelAccessButton;
CEdit inputBox, passwordInputBox, twoFACodeInput;
CLabel charCounter, passwordPromptLabel, feedbackLabel, twoFAPromptLabel, twoFAFeedbackLabel;
bool minimized = false;
bool darkTheme = false;
int MAX_MESSAGE_LENGTH = 4096;
string availableFonts[] = { "Arial", "Courier New", "Verdana", "Times New Roman" };
int currentFontIndex = 0;
string Password = "2024"; // Hardcoded password
string twoFACode = "";

// Button Declarations for Trade Management
CButton buyButton;                 // Button for Buy operations
CButton sellButton;                // Button for Sell operations
CButton closeAllButton;            // Button for closing all positions
CButton closeProfitButton;         // Button for closing profitable positions
CButton closeLossButton;           // Button for closing losing positions
CButton closeBuyButton;            // Button for closing Buy positions
CButton closeSellButton;           // Button for closing Sell positions
CButton deleteAllOrdersButton;     // Button for deleting all orders
CButton deleteLimitOrdersButton;   // Button for deleting limit orders
CButton deleteStopOrdersButton;    // Button for deleting stop orders
CButton deleteStopLimitOrdersButton; // Button for deleting stop limit orders

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if (!ShowAuthenticationPrompt())
    {
        Print("Authorization failed. Exiting...");
        return INIT_FAILED;
    }

    if (!adminHomePanel.Create(ChartID(), "Admin Home Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Admin Home Panel");
        return INIT_FAILED;
    }

    if (!CreateAdminHomeControls())
    {
        Print("Home panel control creation failed");
        return INIT_FAILED;
    }

    if (!communicationsPanel.Create(ChartID(), "Communications Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communications panel dialog");
        return INIT_FAILED;
    }

    if (!tradeManagementPanel.Create(ChartID(), "Trade Management Panel", 0,260, 30, 1040, 170))
    {
        Print("Failed to create Trade Management panel dialog");
        return INIT_FAILED;
    }

    if (!CreateControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }

    if (!CreateTradeManagementControls())
    {
        Print("Trade management control creation failed");
        return INIT_FAILED;
    }

    adminHomePanel.Hide(); // Hide home panel by default on initialization
    communicationsPanel.Hide(); // Hide the Communications Panel
    tradeManagementPanel.Hide(); // Hide the Trade Management Panel
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if (id == CHARTEVENT_OBJECT_CLICK)
    {
        // Panel navigation buttons
        if (sparam == "HomeButtonComm")
        {
            adminHomePanel.Show();
            communicationsPanel.Hide();
        }
        else if (sparam == "HomeButtonTrade")
        {
            adminHomePanel.Show();
            tradeManagementPanel.Hide();
        }

        if (sparam == "TradeMgmtAccessButton")
        {
            tradeManagementPanel.Show();
            adminHomePanel.Hide();
        }
        else if (sparam == "CommunicationsPanelAccessButton")
        {
            communicationsPanel.Show();
            adminHomePanel.Hide();
        }

        // Control buttons for panel resizing and closing
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
            ExpertRemove();

        }
        {
        if (sparam == "LoginButton")
        {
            OnLoginButtonClick();
        }
        else if (sparam == "CloseAuthButton")
        {
            OnCloseAuthButtonClick();
        }
        else if (sparam == "TwoFALoginButton")
        {
            OnTwoFALoginButtonClick();
        }
        else if (sparam == "Close2FAButton")
        {
            OnClose2FAButtonClick();
        }
    }

    switch (id)
    {
        case CHARTEVENT_OBJECT_CLICK:
            if (sparam == "SendButton") OnSendButtonClick();
            else if (sparam == "ClearButton") OnClearButtonClick();
            else if (sparam == "ChangeFontButton") OnChangeFontButtonClick();
            else if (sparam == "ToggleThemeButton") OnToggleThemeButtonClick();
            else if (sparam == "MinimizeButton") OnMinimizeButtonClick();
            else if (sparam == "MaximizeButton") OnMaximizeButtonClick();
            else if (sparam == "CloseButton") OnCloseButtonClick();
            else if (StringFind(sparam, "QuickMessageButton") != -1)
            {
                long index = StringToInteger(StringSubstr(sparam, 18));
                OnQuickMessageButtonClick(index - 1);
            }
            break;

        case CHARTEVENT_OBJECT_ENDEDIT:
            if (sparam == "InputBox") OnInputChange();
            break;
    }

  }

        // Trade management buttons
        if (sparam == "BuyButton") OnBuyButtonClick();
        else if (sparam == "SellButton") OnSellButtonClick();
        else if (sparam == "CloseAllButton") OnCloseAllButtonClick();
        else if (sparam == "CloseProfitButton") OnCloseProfitButtonClick();
        else if (sparam == "CloseLossButton") OnCloseLossButtonClick();
        else if (sparam == "CloseBuyButton") OnCloseBuyButtonClick();
        else if (sparam == "CloseSellButton") OnCloseSellButtonClick();
        else if (sparam == "DeleteAllOrdersButton") OnDeleteAllOrdersButtonClick();
        else if (sparam == "DeleteLimitOrdersButton") OnDeleteLimitOrdersButtonClick();
        else if (sparam == "DeleteStopOrdersButton") OnDeleteStopOrdersButtonClick();
        else if (sparam == "DeleteStopLimitOrdersButton") OnDeleteStopLimitOrdersButtonClick();
    }

//+------------------------------------------------------------------+
//| Trade management button handlers                                 |
//+------------------------------------------------------------------+
void OnBuyButtonClick()
{
    CTrade trade;
    double lotSize = 0.1; //  lot size
    double slippage = 3;  // slippage
    double stopLoss = 0;  //  stop loss (in points)
    double takeProfit = 0; //  take profit (in points)

    // Open Buy order
 double askPrice;
if (SymbolInfoDouble(Symbol(), SYMBOL_ASK, askPrice) && askPrice > 0)
{
    if (trade.Buy(lotSize, Symbol(), askPrice, slippage, stopLoss, takeProfit))
    {
        Print("Buy order executed successfully.");
    }
    else
    {
        Print("Failed to execute Buy order. Error: ", GetLastError());
    }
}
else
{
    Print("Failed to retrieve Ask price. Error: ", GetLastError());
}

    // Execute Buy order logic here
    Print("Executing Buy operation");
}

//+------------------------------------------------------------------+
//| Handle Sell button click                                         |
//+------------------------------------------------------------------+
void OnSellButtonClick()
{
    CTrade trade;
    double lotSize = 0.1; // lot size
    double slippage = 3;  //  slippage
    double stopLoss = 0;  // stop loss (in points)
    double takeProfit = 0; //  take profit (in points)

    double bidPrice;
if (SymbolInfoDouble(Symbol(), SYMBOL_BID, bidPrice) && bidPrice > 0)
{
    // Open Sell order
    if (trade.Sell(lotSize, Symbol(), bidPrice, slippage, stopLoss, takeProfit))
    {
        Print("Sell order opened successfully.");
    }
    else
    {
        Print("Error opening sell order: ", trade.ResultRetcode());
    }
}
else
{
    Print("Failed to retrieve Bid price. Error: ", GetLastError());
}
}

//+------------------------------------------------------------------+
//| Handle Close All button click                                    |
//+------------------------------------------------------------------+
void OnCloseAllButtonClick()
{
    CPositionInfo position;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        if (position.SelectByIndex(i))
        {
            CTrade trade;
            if (position.Type() == POSITION_TYPE_BUY)
                trade.PositionClose(position.Ticket());
            else if (position.Type() == POSITION_TYPE_SELL)
                trade.PositionClose(position.Ticket());
        }
    }
    Print("All positions closed.");
}

//+------------------------------------------------------------------+
//| Handle Close Profitable button click                             |
//+------------------------------------------------------------------+
void OnCloseProfitButtonClick()
{
    CPositionInfo position;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        if (position.SelectByIndex(i) && position.Profit() > 0)
        {
            CTrade trade;
            if (position.Type() == POSITION_TYPE_BUY)
                trade.PositionClose(position.Ticket());
            else if (position.Type() == POSITION_TYPE_SELL)
                trade.PositionClose(position.Ticket());
        }
    }
    Print("Profitable positions closed.");
}

//+------------------------------------------------------------------+
//| Handle Close Losing button click                                 |
//+------------------------------------------------------------------+
void OnCloseLossButtonClick()
{
    CPositionInfo position;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        if (position.SelectByIndex(i) && position.Profit() < 0)
        {
            CTrade trade;
            if (position.Type() == POSITION_TYPE_BUY)
                trade.PositionClose(position.Ticket());
            else if (position.Type() == POSITION_TYPE_SELL)
                trade.PositionClose(position.Ticket());
        }
    }
    Print("Losing positions closed.");
}

void OnCloseBuyButtonClick()
{
    // Close Buy positions logic
    Print("Closing Buy positions");
}

void OnCloseSellButtonClick()
{
    // Close Sell positions logic
    Print("Closing Sell positions");
}

//+------------------------------------------------------------------+
//| Handle Delete All Orders button click                            |
//+------------------------------------------------------------------+
void OnDeleteAllOrdersButtonClick()
{
    COrderInfo order;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (order.SelectByIndex(i))
        {
            CTrade trade;
            trade.OrderDelete(order.Ticket());
        }
    }
    Print("All orders deleted.");
}

//+------------------------------------------------------------------+
//| Handle Delete Limit Orders button click                          |
//+------------------------------------------------------------------+
void OnDeleteLimitOrdersButtonClick()
{
    COrderInfo order;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (order.SelectByIndex(i) && order.Type() == ORDER_TYPE_BUY_STOP_LIMIT)
        {
            CTrade trade;
            trade.OrderDelete(order.Ticket());
        }
    }
    Print("All limit orders deleted.");
}

//+------------------------------------------------------------------+
//| Handle Delete Stop Orders button click                           |
//+------------------------------------------------------------------+
void OnDeleteStopOrdersButtonClick()
{
    COrderInfo order;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (order.SelectByIndex(i) && order.Type() == ORDER_TYPE_BUY_STOP && ORDER_TYPE_SELL_STOP)
        {
            CTrade trade;
            trade.OrderDelete(order.Ticket());
        }
    }
    Print("All stop orders deleted.");
}

//+------------------------------------------------------------------+
//| Handle Delete Stop Limit Orders button click                     |
//+------------------------------------------------------------------+
void OnDeleteStopLimitOrdersButtonClick()
{
    COrderInfo order;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (order.SelectByIndex(i) && order.Type() == ORDER_TYPE_BUY_LIMIT && ORDER_TYPE_SELL_STOP_LIMIT)
        {
            CTrade trade;
            trade.OrderDelete(order.Ticket());
        }
    }
    Print("All stop limit orders deleted.");
}

//+------------------------------------------------------------------+
//| Show authentication input dialog                                 |
//+------------------------------------------------------------------+
bool ShowAuthenticationPrompt()
{
    if (!authentication.Create(ChartID(), "Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create authentication dialog");
        return false;
    }

    if (!passwordInputBox.Create(ChartID(), "PasswordInputBox", 0, 20, 70, 260, 95))
    {
        Print("Failed to create password input box");
        return false;
    }
    authentication.Add(passwordInputBox);

    if (!passwordPromptLabel.Create(ChartID(), "PasswordPromptLabel", 0, 20, 20, 260, 40))
    {
        Print("Failed to create password prompt label");
        return false;
    }
    passwordPromptLabel.Text("Enter password: Access Admin Panel");
    authentication.Add(passwordPromptLabel);

    if (!feedbackLabel.Create(ChartID(), "FeedbackLabel", 0, 20, 140, 380, 160))
    {
        Print("Failed to create feedback label");
        return false;
    }
    feedbackLabel.Text("");
    feedbackLabel.Color(clrRed); // Red color for incorrect attempts
    authentication.Add(feedbackLabel);

    if (!loginButton.Create(ChartID(), "LoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create login button");
        return false;
    }
    loginButton.Text("Login");
    authentication.Add(loginButton);

    if (!closeAuthButton.Create(ChartID(), "CloseAuthButton", 0, 120, 120, 200, 140))
    {
        Print("Failed to create close button for authentication");
        return false;
    }
    closeAuthButton.Text("Close");
    authentication.Add(closeAuthButton);

    authentication.Show();
    ChartRedraw();
    return true;
}

//+------------------------------------------------------------------+
//| Show two-factor authentication input dialog                      |
//+------------------------------------------------------------------+
void ShowTwoFactorAuthPrompt()
{
    if (!twoFactorAuth.Create(ChartID(), "Two-Factor Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create 2FA dialog");
        return;
    }

    if (!twoFACodeInput.Create(ChartID(), "TwoFACodeInput", 0, 20, 70, 260, 95))
    {
        Print("Failed to create 2FA code input box");
        return;
    }
    twoFactorAuth.Add(twoFACodeInput);

    if (!twoFAPromptLabel.Create(ChartID(), "TwoFAPromptLabel", 0, 20, 20, 380, 40))
    {
        Print("Failed to create 2FA prompt label");
        return;
    }
    twoFAPromptLabel.Text("Enter the 2FA code sent to your Telegram:");
    twoFactorAuth.Add(twoFAPromptLabel);

    if (!twoFAFeedbackLabel.Create(ChartID(), "TwoFAFeedbackLabel", 0, 20, 140, 380, 160))
    {
        Print("Failed to create 2FA feedback label");
        return;
    }
    twoFAFeedbackLabel.Text("");
    twoFAFeedbackLabel.Color(clrRed); // Red color for incorrect 2FA attempts
    twoFactorAuth.Add(twoFAFeedbackLabel);

    if (!twoFALoginButton.Create(ChartID(), "TwoFALoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create 2FA login button");
        return;
    }
    twoFALoginButton.Text("Verify");
    twoFactorAuth.Add(twoFALoginButton);

    if (!close2FAButton.Create(ChartID(), "Close2FAButton", 0, 120, 120, 200, 140))
    {
        Print("Failed to create close button for 2FA");
        return;
    }
    close2FAButton.Text("Close");
    twoFactorAuth.Add(close2FAButton);

    twoFactorAuth.Show();
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Admin Home Panel controls creation                               |
//+------------------------------------------------------------------+
bool CreateAdminHomeControls()
{
    long chart_id = ChartID();

    if (!tradeMgmtAccessButton.Create(chart_id, "TradeMgmtAccessButton", 0, 50, 50, 250, 90))
    {
        Print("Failed to create Trade Management Access button");
        return false;
    }
    tradeMgmtAccessButton.Text("Trade Management Panel");
    adminHomePanel.Add(tradeMgmtAccessButton);

    if (!communicationsPanelAccessButton.Create(chart_id, "CommunicationsPanelAccessButton", 0, 50, 100, 250, 140))
    {
        Print("Failed to create Communications Panel Access button");
        return false;
    }
    communicationsPanelAccessButton.Text("Communications Panel");
    adminHomePanel.Add(communicationsPanelAccessButton);

    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    adminHomePanel.Add(minimizeButton);

    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    adminHomePanel.Add(maximizeButton);

    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    adminHomePanel.Add(closeButton);

    return true;
}
//Helper Function seamless Button creation
bool CreateButton(CButton &button, const string name, const string text, int x1, int y1, int x2, int y2)
{
    long chart_id = ChartID();

    if (!button.Create(chart_id, name, 0, x1, y1, x2, y2))
    {
        Print("Failed to create button: ", name);
        return false;
    }

    button.Text(text);
    tradeManagementPanel.Add(button);
    return true;
}

//+------------------------------------------------------------------+
//| Create Trade Management Controls (Buttons)                       |
//+------------------------------------------------------------------+
bool CreateTradeManagementControls()
{
    // Coordinates for buttons (adjust as needed)
    const int Y1_TOP = 5, Y2_TOP = 40;
    const int Y1_MID = 50, Y2_MID = 70;
    const int Y1_BOTTOM = 80, Y2_BOTTOM = 100;

    // Create Buttons
    if (!CreateButton(buyButton, "BuyButton", "Buy", 130, Y1_TOP, 210, Y2_TOP)) return false;
    if (!CreateButton(sellButton, "SellButton", "Sell", 220, Y1_TOP, 320, Y2_TOP)) return false;
    if (!CreateButton(closeAllButton, "CloseAllButton", "Close All", 130, Y1_MID, 230, Y2_MID)) return false;
    if (!CreateButton(closeProfitButton, "CloseProfitButton", "Close Profitable", 240, Y1_MID, 380, Y2_MID)) return false;
    if (!CreateButton(closeLossButton, "CloseLossButton", "Close Losing", 390, Y1_MID, 510, Y2_MID)) return false;
    if (!CreateButton(closeBuyButton, "CloseBuyButton", "Close Buys", 520, Y1_MID, 620, Y2_MID)) return false;
    if (!CreateButton(closeSellButton, "CloseSellButton", "Close Sells", 630, Y1_MID, 730, Y2_MID)) return false;
    if (!CreateButton(deleteAllOrdersButton, "DeleteAllOrdersButton", "Delete All Orders", 130, Y1_BOTTOM , 270, Y2_BOTTOM )) return false;
    if (!CreateButton(deleteLimitOrdersButton, "DeleteLimitOrdersButton", "Delete Limits", 275, Y1_BOTTOM , 385, Y2_BOTTOM )) return false;
    if (!CreateButton(deleteStopOrdersButton, "DeleteStopOrdersButton", "Delete Stops", 390, Y1_BOTTOM , 515, Y2_BOTTOM )) return false;
    if (!CreateButton(deleteStopLimitOrdersButton, "DeleteStopLimitOrdersButton", "Delete Stop Limits", 520, Y1_BOTTOM , 660, Y2_BOTTOM )) return false;

    return true; // All buttons created successfully
}

//+------------------------------------------------------------------+
//| Handle login button click                                        |
//+------------------------------------------------------------------+
void OnLoginButtonClick()
{
    string enteredPassword = passwordInputBox.Text();
    if (enteredPassword == Password)
    {
        twoFACode = GenerateRandom6DigitCode();
        SendMessageToTelegram("A login attempt was made on the Admin Panel. Please use this code to verify your identity: " + twoFACode, Hardcoded2FAChatId, Hardcoded2FABotToken);

        authentication.Destroy();
        ShowTwoFactorAuthPrompt();
        Print("Password authentication successful. A 2FA code has been sent to your Telegram.");
    }
    else
    {
        feedbackLabel.Text("Wrong password. Try again.");
        passwordInputBox.Text("");
    }
    ///Handlers for the trade management

}

//+------------------------------------------------------------------+
//| Handle 2FA login button click                                    |
//+------------------------------------------------------------------+
void OnTwoFALoginButtonClick()
{
    // If 2FA is successful, show the trade management panel
    string enteredCode = twoFACodeInput.Text();
    if (enteredCode == twoFACode)
    {
        twoFactorAuth.Destroy();
        adminHomePanel.Show();
        Print("2FA authentication successful. Access granted to Trade Management Panel.");
    }
    else
    {
        twoFAFeedbackLabel.Text("Wrong code. Try again.");
        twoFACodeInput.Text("");
    }
}

//+------------------------------------------------------------------+
//| Handle close button for authentication                           |
//+------------------------------------------------------------------+
void OnCloseAuthButtonClick()
{
    authentication.Destroy();
    ExpertRemove(); // Exit the expert
    Print("Authentication dialog closed.");
}

//+------------------------------------------------------------------+
//| Handle close button for 2FA                                      |
//+------------------------------------------------------------------+
void OnClose2FAButtonClick()
{
    twoFactorAuth.Destroy();
    ExpertRemove();
    Print("2FA dialog closed.");
}

//+------------------------------------------------------------------+
//| Create necessary UI controls                                     |
//+------------------------------------------------------------------+
bool CreateControls()
{
    long chart_id = ChartID();


    if (!inputBox.Create(chart_id, "InputBox", 0, 5, 25, 460, 95))
    {
        Print("Failed to create input box");
        return false;
    }
    communicationsPanel.Add(inputBox);

    // Create Home Button for Communications Panel
    if (!homeButtonComm.Create(chart_id,  "HomeButtonComm", 0, 20, 120, 120,150))
    {
        Print("Failed to create Home button for Communications Panel");
        return false;
    }
    homeButtonComm.Text("Home 🏠");
    communicationsPanel.Add(homeButtonComm);

    // Create Home Button for Trade Management Panel
    if (!homeButtonTrade.Create(chart_id, "HomeButtonTrade", 0, 20, 10, 120, 30))
    {
        Print("Failed to create Home button for Trade Management Panel");
        return false;
    }
    homeButtonTrade.Text("Home 🏠");
    tradeManagementPanel.Add(homeButtonTrade);

    if (!charCounter.Create(chart_id, "CharCounter", 0, 380, 5, 460, 25))
    {
        Print("Failed to create character counter");
        return false;
    }
    charCounter.Text("0/" + IntegerToString(MAX_MESSAGE_LENGTH));
    communicationsPanel.Add(charCounter);

    if (!clearButton.Create(chart_id, "ClearButton", 0, 235, 95, 345, 125))
    {
        Print("Failed to create clear button");
        return false;
    }
    clearButton.Text("Clear");
    communicationsPanel.Add(clearButton);

    if (!sendButton.Create(chart_id, "SendButton", 0, 350, 95, 460, 125))
    {
        Print("Failed to create send button");
        return false;
    }
    sendButton.Text("Send");
    communicationsPanel.Add(sendButton);

    if (!changeFontButton.Create(chart_id, "ChangeFontButton", 0, 95, 95, 230, 115))
    {
        Print("Failed to create change font button");
        return false;
    }
    changeFontButton.Text("Font<>");
    communicationsPanel.Add(changeFontButton);

    if (!toggleThemeButton.Create(chart_id, "ToggleThemeButton", 0, 5, 95, 90, 115))
    {
        Print("Failed to create toggle theme button");
        return false;
    }
    toggleThemeButton.Text("Theme<>");
    communicationsPanel.Add(toggleThemeButton);

    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    communicationsPanel.Add(minimizeButton);

    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    communicationsPanel.Add(maximizeButton);

    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    communicationsPanel.Add(closeButton);

    return CreateQuickMessageButtons();

}

//+------------------------------------------------------------------+
//| Create quick message buttons                                     |
//+------------------------------------------------------------------+
bool CreateQuickMessageButtons()
{
    string quickMessages[] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;

    for (int i = 0; i < ArraySize(quickMessages); i++)
    {
        bool created = quickMessageButtons[i].Create(ChartID(), "QuickMessageButton" + IntegerToString(i + 1), 0,
            startX + (i % 2) * (width + spacing), startY + (i / 2) * (height + spacing),
            startX + (i % 2) * (width + spacing) + width, startY + (i / 2) * (height + spacing) + height);

        if (!created)
        {
            Print("Failed to create quick message button ", i + 1);
            return false;
        }
        quickMessageButtons[i].Text(quickMessages[i]);
        communicationsPanel.Add(quickMessageButtons[i]);
    }

    return true;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    communicationsPanel.Destroy();
    Print("Deinitialization complete");
}

//+------------------------------------------------------------------+
//| Handle custom message send button click                          |
//+------------------------------------------------------------------+
void OnSendButtonClick()
{
    string message = inputBox.Text();
    if (StringLen(message) > 0)
    {
        if (SendMessageToTelegram(message, InputChatId, InputBotToken))
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
//| Handle clear button click                                        |
//+------------------------------------------------------------------+
void OnClearButtonClick()
{
    inputBox.Text("");
    OnInputChange();
    Print("Input box cleared.");
}

//+------------------------------------------------------------------+
//| Handle quick message button click                                |
//+------------------------------------------------------------------+
void OnQuickMessageButtonClick(long index)
{
    string quickMessages[] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    string message = quickMessages[(int)index];

    if (SendMessageToTelegram(message, InputChatId, InputBotToken))
        Print("Quick message sent: ", message);
    else
        Print("Failed to send quick message.");
}

//+------------------------------------------------------------------+
//| Update character counter                                         |
//+------------------------------------------------------------------+
void OnInputChange()
{
    int currentLength = StringLen(inputBox.Text());
    charCounter.Text(IntegerToString(currentLength) + "/" + IntegerToString(MAX_MESSAGE_LENGTH));
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Handle toggle theme button click                                 |
//+------------------------------------------------------------------+
void OnToggleThemeButtonClick()
{
    darkTheme = !darkTheme;
    UpdateThemeColors();
    Print("Theme toggled: ", darkTheme ? "Dark" : "Light");
}

//+------------------------------------------------------------------+
//| Update theme colors for the panel                                |
//+------------------------------------------------------------------+
void UpdateThemeColors()
{
    color textColor = darkTheme ? clrWhite : clrBlack;
    color buttonBgColor = darkTheme ? clrDarkSlateGray : clrGainsboro;
    color borderColor = darkTheme ? clrSlateGray : clrGray;
    color bgColor = darkTheme ? clrDarkBlue : clrWhite;


    UpdateButtonTheme(clearButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(sendButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(toggleThemeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(changeFontButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(minimizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(maximizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(closeButton, textColor, buttonBgColor, borderColor);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        UpdateButtonTheme(quickMessageButtons[i], textColor, buttonBgColor, borderColor);
    }

    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Apply theme settings to a button                                 |
//+------------------------------------------------------------------+
void UpdateButtonTheme(CButton &button, color textColor, color bgColor, color borderColor)
{
    button.SetTextColor(textColor);
    button.SetBackgroundColor(bgColor);
    button.SetBorderColor(borderColor);
}

//+------------------------------------------------------------------+
//| Handle change font button click                                  |
//+------------------------------------------------------------------+
void OnChangeFontButtonClick()
{
    currentFontIndex = (currentFontIndex + 1) % ArraySize(availableFonts);
    SetFontForAll(availableFonts[currentFontIndex]);
    Print("Font changed to: ", availableFonts[currentFontIndex]);
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Set font for all input boxes and buttons                         |
//+------------------------------------------------------------------+
void SetFontForAll(string fontName)
{
    inputBox.Font(fontName);
    clearButton.Font(fontName);
    sendButton.Font(fontName);
    toggleThemeButton.Font(fontName);
    changeFontButton.Font(fontName);
    minimizeButton.Font(fontName);
    maximizeButton.Font(fontName);
    closeButton.Font(fontName);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        quickMessageButtons[i].Font(fontName);
    }
}

//+------------------------------------------------------------------+
//| Generate a random 6-digit code for 2FA                           |
//+------------------------------------------------------------------+
string GenerateRandom6DigitCode()
{
    int code = MathRand() % 1000000; // Produces a 6-digit number
    return StringFormat("%06d", code); // Ensures leading zeros
}

//+------------------------------------------------------------------+
//| Handle minimize button click                                     |
//+------------------------------------------------------------------+
void OnMinimizeButtonClick()
{
    minimized = true;
    communicationsPanel.Hide();
    minimizeButton.Hide();
    maximizeButton.Show();
    closeButton.Show();
    Print("Panel minimized.");
}

//+------------------------------------------------------------------+
//| Handle maximize button click                                     |
//+------------------------------------------------------------------+
void OnMaximizeButtonClick()
{
    if (minimized)
    {
        communicationsPanel.Show();
        minimizeButton.Show();
        maximizeButton.Hide();
        closeButton.Hide();
        minimized = false;
        Print("Panel maximized.");
    }
}

//+------------------------------------------------------------------+
//| Handle close button click for admin panel                        |
//+------------------------------------------------------------------+
void OnCloseButtonClick()
{
    ExpertRemove();
    Print("Admin panel closed.");
}

//+------------------------------------------------------------------+
//| Send the message to Telegram                                     |
//+------------------------------------------------------------------+
bool SendMessageToTelegram(string message, string chatId, string botToken)
{
    string url = "https://api.telegram.org/bot" + botToken + "/sendMessage";
    string jsonMessage = "{\"chat_id\":\"" + chatId + "\", \"text\":\"" + message + "\"}";

    char postData[];
    ArrayResize(postData, StringToCharArray(jsonMessage, postData) - 1);

    int timeout = 5000;
    char result[];
    string responseHeaders;
    int responseCode = WebRequest("POST", url, "Content-Type: application/json\r\n", timeout, postData, result, responseHeaders);

    if (responseCode == 200)
    {
        Print("Message sent successfully: ", message);
        return true;
    }
    else
    {
        Print("Failed to send message. HTTP code: ", responseCode, " Error code: ", GetLastError());
        Print("Response: ", CharArrayToString(result));
        return false;
    }
}

//+------------------------------------------------------------------+
```

### Testing

Upon testing, all the buttons are functioning correctly and responding as programmed, making our Trading Management Panel a valuable component of the Trading Administrator's toolkit. The integration of new buttons and the implementation of their event handlers were both successfully completed.

![Test results](https://c.mql5.com/2/129/terminal64_4VYts6woUy__3.gif)

Trade Management Panel button handlers working

![Admin Panel V1.22 buttons working](https://c.mql5.com/2/129/results1__3.png)

These are Experts comments showing successful order execution and closing

### Conclusion

In this discussion, we revisited the Telegram configuration and enhanced the Trade Management Panel with new trade management buttons. We also improved the layout by reducing the vertical scale and increasing the horizontal scale, including a positional shift in the x-direction. This adjustment resolved overlapping issues between our panel and the inbuilt quick trade buttons, improving chart visibility while keeping the trading buttons easily accessible.

After finalizing the layout, we integrated button event handlers to ensure they respond appropriately when clicked.

I hope this discussion has shed light on the broader possibilities of GUI programming in MQL5, demonstrating the creative potential when designing various GUI components. With skilled and imaginative minds, there's so much more to explore beyond this foundation.

Happy developing, fellow traders!

[Back to Contents](https://www.mql5.com/en/articles/16328#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16328.zip "Download all attachments in the single ZIP archive")

[Admin\_Panel\_V1.22.mq5](https://www.mql5.com/en/articles/download/16328/admin_panel_v1.22.mq5 "Download Admin_Panel_V1.22.mq5")(73.66 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/477004)**

![Developing a Replay System (Part 53): Things Get Complicated (V)](https://c.mql5.com/2/81/Desenvolvendo_um_sistema_de_Replay1Parte_53__LOGO.png)[Developing a Replay System (Part 53): Things Get Complicated (V)](https://www.mql5.com/en/articles/11932)

In this article, we'll cover an important topic that few people understand: Custom Events. Dangers. Advantages and disadvantages of these elements. This topic is key for those who want to become a professional programmer in MQL5 or any other language. Here we will focus on MQL5 and MetaTrader 5.

![Price Action Analysis Toolkit Development (Part 2):  Analytical Comment Script](https://c.mql5.com/2/102/Price_Action_Analysis_Toolkit_Development_Part_2____LOGO.png)[Price Action Analysis Toolkit Development (Part 2): Analytical Comment Script](https://www.mql5.com/en/articles/15927)

Aligned with our vision of simplifying price action, we are pleased to introduce another tool that can significantly enhance your market analysis and help you make well-informed decisions. This tool displays key technical indicators such as previous day's prices, significant support and resistance levels, and trading volume, while automatically generating visual cues on the chart.

![Trading Insights Through Volume: Moving Beyond OHLC Charts](https://c.mql5.com/2/102/Trading_Insights_Through_Volume_Moving_Beyond_OHLC_Charts___LOGO.png)[Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)

Algorithmic trading system that combines volume analysis with machine learning techniques, specifically LSTM neural networks. Unlike traditional trading approaches that primarily focus on price movements, this system emphasizes volume patterns and their derivatives to predict market movements. The methodology incorporates three main components: volume derivatives analysis (first and second derivatives), LSTM predictions for volume patterns, and traditional technical indicators.

![Mutual information as criteria for Stepwise Feature Selection](https://c.mql5.com/2/102/Mutual_information_as_criteria_for_Stepwise_Feature_Selection___LOGO2.png)[Mutual information as criteria for Stepwise Feature Selection](https://www.mql5.com/en/articles/16416)

In this article, we present an MQL5 implementation of Stepwise Feature Selection based on the mutual information between an optimal predictor set and a target variable.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/16328&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049544302377413985)

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