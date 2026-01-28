---
title: How to Create an Interactive MQL5 Dashboard/Panel Using the Controls Class (Part 2): Adding Button Responsiveness
url: https://www.mql5.com/en/articles/16146
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T17:59:43.172253
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/16146&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068889028352212699)

MetaTrader 5 / Trading


### Introduction

In our [previous article](https://www.mql5.com/en/articles/16084), we successfully set up the core components of our [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") dashboard panel. At that stage, the buttons and labels we assembled remained static, providing a foundational yet inactive structure. Now, it’s time to move beyond a mere visual framework. In this next part, we’ll focus on making the panel truly interactive. We will breathe life into the components by adding the ability to respond to user inputs and clicks, turning our dashboard into a dynamic tool that’s ready for real-time trading interaction.

In this article, essentially Part 2, we will explore how to automate the functionality of the buttons created in [Part 1](https://www.mql5.com/en/articles/16084), ensuring they react when clicked and edited. We’ll learn how to set up events that trigger specific actions, allowing the user to interact with the panel in a meaningful way. We will cover key topics, including:

1. Illustration of Elements to be Automated: A detailed overview of the components that will gain functionality.
2. Automating the GUI Interactions in MQL5: Implementing the necessary code to ensure buttons respond to user inputs and clicks effectively.
3. Conclusion: Summarizing the advancements made in creating the interactive dashboard panel.

Let’s dive into these topics to enhance our trading interface!

### Illustration of Elements to be Automated

We will concentrate on the automation of the buttons we created in the first part of our MQL5 panel. Each button has a specific function, and we want to make sure they react intuitively to the user's commands. This reaction is essential because, unlike a program that runs in the background, a trading panel needs to be user-friendly and accessible. First, we have the button in the upper right-hand corner of the panel, which is designed to close the entire interface. So, if the trading environment is open on the [MetaTrader 5](https://www.metatrader5.com/en "https://www.metatrader5.com/en") chart, it should be possible to dismiss the panel in the same way one would close an application.

While the trade button is active, we will put buttons in place that execute certain trading operations. These include "Open Buy", "Sell", "Sell Stop", "Sell Limit", "Buy Stop", and "Buy Limit". These buttons will allow for quick order placements and facilitate immediate reactions to the ever-changing market. We will also automate the closure buttons that will practically manage trades when the close button is active. They include "Close All", and "Close All Profit Trades", among many others, and one that almost our fingers cringe to mention, "Close All Pending Orders". When you click a button, it will do what it says it will do.

Lastly, we will automate the information button that, when pressed, unfurls an interface of buttons that details the user's account information and background information. We hope this will aid in keeping the traders informed about pertinent details related to their accounts, thereby helping them to make better decisions. The aim, of all of this, is to make a responsive trading panel, that makes the sort of operations that a trader needs to do easy, and that also, in some small way, tries to engage the user more than the last panel did.

To easily aid in understanding these automation processes and components, below is a detailed description of them featuring the previous milestone.

![COMPONENTS ILLUSTRATION](https://c.mql5.com/2/98/Screenshot_2024-10-16_203236.png)

With the insight of what we will be doing, let us start the automation right away. Please refer to the [previous article](https://www.mql5.com/en/articles/16084) where we created the static assembly of the GUI elements if you still have not gone through it so that you will be on the right track with us. Let's do it.

### Automating the GUI Interactions in MQL5

We will go from simple to complex processes so that our structure is arranged in chronological order. Thus, we will update the account information on each tick or price quote change. To achieve this, we will need the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, an in-built MQL5 function that is typically called when the price quotes change. The function is a void data type which means that it handles executions directly and does not have to return any output. Your function should resemble this as below.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

  ...
}
//+------------------------------------------------------------------+
```

This is the event handler that is responsible for price updates and thus the heart of our logic. We will add the control logic to this function as below:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   //--- Start of the OnTick function, called on every price tick

   //--- Check if the background color of the INFO button is yellow
   if (obj_Btn_INFO.ColorBackground() == clrYellow) {
      //--- Update the account equity display on the panel
      obj_Btn_ACC_EQUITY.Text(DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2));

      //--- Update the account balance display on the panel
      obj_Btn_ACC_BALANCE.Text(DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));

      //--- Update the server trade time display on the panel
      obj_Btn_TIME.Text(TimeToString(TimeTradeServer(), TIME_DATE | TIME_SECONDS));
   }

   //--- End of the OnTick function
}
//+------------------------------------------------------------------+
```

The first action that we do within the OnTick function is to check the background color of the information button ("obj\_Btn\_INFO"). If the button's background is set to yellow, it indicates that we have activated the information display mode. Upon meeting this condition, we will update various account-related displays on the panel. Specifically, we update the account equity display by fetching the current account equity using the [AccountInfoDouble](https://www.mql5.com/en/docs/constants/environment_state/accountinformation) function and passing the property "ACCOUNT\_EQUITY" as the input parameter and formatting it to two decimal places with the [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) function. We then assign this value to the text of the "obj\_Btn\_ACC\_EQUITY" button, ensuring that we have the latest equity information at our fingertips.

Next, we similarly update the account balance display by retrieving the account balance using the "ACCOUNT\_BALANCE" parameter property, formatting it to two decimal places, and setting it to the text of the "obj\_Btn\_ACC\_BALANCE" button. Finally, we refresh the server trade time display by obtaining the current server trade time using the [TimeTradeServer](https://www.mql5.com/en/docs/dateandtime/timetradeserver) function, formatting it to include both date and seconds and updating the text of the "obj\_Btn\_TIME" button with this value. This ensures we are always aware of the latest trade time, essential for timely decision-making in trading scenarios. These are the results.

![ONTICK UPDATE](https://c.mql5.com/2/98/PANEL_ONTICK_GIF.gif)

From the visualization, we can see that the time field is updated accordingly, which is a success. Now, the first automation component is done. That was easy, right? We proceed to the other components of our GUI panel then. The automation of the rest elements will be done inside the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) function handler, so let us take a deeper view of its input parameters as well as its functions.

```
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
      ...
}
```

The purpose of the function is to handle chart changes made by either a user or an MQL5 program. Thus, the interactions that the user will be making such as moving the mouse, editing the button fields, and clicking labels and buttons will be captured and handled by this event handler. Let’s break down its arguments for more interpretation:

- **id:** This parameter represents the event ID and corresponds to one of the 11 predefined event types. These include events like key presses, mouse movements, object creation, chart changes, and custom events. For custom events, you can use IDs from CHARTEVENT\_CUSTOM to CHARTEVENT\_CUSTOM\_LAST. The 11 event types are as shown below;

![CHART EVENT TYPES](https://c.mql5.com/2/98/Screenshot_2024-07-09_185053.png)

- **lparam:** A long-type event parameter. Its value depends on the specific event being handled. For example, it could represent a key code during a key press event.
- **dparam:** A double-type event parameter. Similar to lparam, its value varies based on the event type. For instance, during a mouse movement event, it might convey the mouse cursor’s position.
- **sparam:** A string-type event parameter. Again, its meaning depends on the event. For instance, during object creation, it could hold the name of the newly created object.

To easily showcase this more understandably, inside the function, let us have a printout that contains all four arguments to the journal.

```
// Print the 4 function parameters
Print("ID = ",id,", LPARAM = ",lparam,", DPARAM = ",dparam,", SPARAM = ",sparam);
```

This function will print the chart event ID, its long-type event value, double-type event value, and the string-type value. Let us have a look at the following GIF for easier referencing.

![GENERAL CHARTEVENTS](https://c.mql5.com/2/98/CHART_EVENTS_1.gif)

From the GIF provided, everything should now be clear. We now graduate to capturing chart click events on the GUI panel elements. Thus, our ID will be [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents).

```
   //Print("ID = ",id,", LPARAM = ",lparam,", DPARAM = ",dparam,", SPARAM = ",sparam);

   if (id==CHARTEVENT_OBJECT_CLICK){

        //---
   }
```

We first comment out the previous line of code because we do not want to spam our journal with irrelevant information. The two slashes (//) used are called single-line comments and comment out a code from their start and continue until the end of the line, hence their name 'single-line' comment. Comments are particularly ignored by the computer during execution. We use the if statement to check out whether there was an object click. This is achieved by equating chart event ID to the object click enumerations. If we did click an object, let us print the arguments and see what we get. The following code is used.

```
   if (id==CHARTEVENT_OBJECT_CLICK){
      Print("ID = ",id,", LM = ",lparam,", DM = ",dparam,", SPARAM = ",sparam);

     ...
   }
```

In the printout function, we just changed the "LPARAM" to "LP" and "DPARAM" to "DP" so that we can only concentrate on the chart event ID and the name of the clicked object, from there we will get the object's ID and take action if necessary. Below is an illustration of the logic:

![OBJECT CLICK GIF](https://c.mql5.com/2/98/OBJECT_CLICK_GIF.gif)

The first button that we automate is the trade button.

```
      if (sparam==obj_Btn_TRADE.Name()){
         // Print to the log which object was clicked for debugging purposes
         Print("OBJECT CLICKED = ", obj_Btn_TRADE.Name());

         // Reset the pressed states of all buttons to ensure only one button appears pressed at a time
         obj_Btn_TRADE.Pressed(false);
         obj_Btn_CLOSE.Pressed(false);
         obj_Btn_INFO.Pressed(false);

         // Change the background color of the Trade button to yellow to indicate it is active
         obj_Btn_TRADE.ColorBackground(clrYellow);
         // Set the background color of the Close and Info buttons to silver to indicate they are inactive
         obj_Btn_CLOSE.ColorBackground(clrSilver);
         obj_Btn_INFO.ColorBackground(clrSilver);

         // Set the border color of the Trade button to yellow to match its background
         obj_Btn_TRADE.ColorBorder(clrYellow);
         // Set the border color of the Close and Info buttons to silver to indicate they are inactive
         obj_Btn_CLOSE.ColorBorder(clrSilver);
         obj_Btn_INFO.ColorBorder(clrSilver);

         // Call a function to destroy the Close section if it exists
         destroySection_Close();
         // Call a function to destroy the Information section if it exists
         destroySection_Information();

         // Create the Trade section, bringing it to the forefront
         createSection_Trade();
      }
```

Here, we manage the interaction when the "Trade" button is clicked by checking if the string parameter matches the name of the "Trade" button using the function "Name". First, we use the function [Print](https://www.mql5.com/en/docs/common/print) to log the name of the clicked object, which helps in debugging by confirming which button was clicked. Next, we reset the pressed state of all the relevant buttons— "obj\_Btn\_TRADE", "obj\_Btn\_CLOSE", and "obj\_Btn\_INFO"—by calling the function "Pressed" for each button and passing a false flag. This ensures that only one button is visually pressed at a time. After resetting, we visually indicate that the "Trade" button is active by changing its background color to yellow using the "ColorBackground" function while setting the background colors of the "Close" and "Info" buttons to silver to show they are inactive. Similarly, we update the border colors using the "ColorBorder" function—yellow for the "Trade" button and silver for the others.

Then, we clean up the interface by calling the functions "destroySection\_Close" and "destroySection\_Information" to remove any content displayed for the "Close" or "Information" sections. Finally, we invoke the function "createSection\_Trade" to dynamically create and display the trading section, ensuring the user has access to the trading interface. We do a similar thing to the "Close" button as well.

```
      // Check if the clicked object is the Close button
      else if (sparam==obj_Btn_CLOSE.Name()){
         // Print to the log which object was clicked for debugging purposes
         Print("OBJECT CLICKED = ", obj_Btn_CLOSE.Name());

         // Reset the pressed states of all buttons
         obj_Btn_TRADE.Pressed(false);
         obj_Btn_CLOSE.Pressed(false);
         obj_Btn_INFO.Pressed(false);

         // Set the background color of the Trade button to silver, indicating it's inactive
         obj_Btn_TRADE.ColorBackground(clrSilver);
         // Change the background color of the Close button to yellow to indicate it is active
         obj_Btn_CLOSE.ColorBackground(clrYellow);
         obj_Btn_INFO.ColorBackground(clrSilver);

         // Set the border color of the Trade button to silver
         obj_Btn_TRADE.ColorBorder(clrSilver);
         // Set the border color of the Close button to yellow
         obj_Btn_CLOSE.ColorBorder(clrYellow);
         obj_Btn_INFO.ColorBorder(clrSilver);

         // Call a function to destroy the Trade section if it exists
         destroySection_Trade();
         // Call a function to destroy the Information section if it exists
         destroySection_Information();

         // Create the Close section, bringing it to the forefront
         createSection_Close();
      }
```

Here, we do almost the exact logic as we did with the "Trade" button. We handle the scenario where the "Close" button is clicked by checking if the string parameter matches the name of the "Close" button using the "Name" function. We first log the name of the clicked button with the [Print](https://www.mql5.com/en/docs/common/print) function for debugging purposes. This helps us confirm that the correct button was clicked. Next, we reset the pressed states of the "obj\_Btn\_TRADE", "obj\_Btn\_CLOSE", and "obj\_Btn\_INFO" buttons using the "Pressed" function. This ensures none of the buttons remain pressed visually after clicking. We then update the interface by changing the background color of the "Close" button to yellow using the "ColorBackground" function, indicating it is active while setting the background color of the "Trade" and "Info" buttons to silver to show they are inactive.

Similarly, we adjust the border colors of the buttons using the "ColorBorder" function, making the "Close" button's border yellow to match its background, while the borders of the "Trade" and "Info" buttons are set to silver to indicate inactivity. To clean up, we call the functions "destroySection\_Trade" and "destroySection\_Information" to remove any existing content for the trading or information sections. Finally, we call the "createSection\_Close" function to dynamically generate and display the interface related to closing positions or orders, ensuring the user can interact with the appropriate close options on the panel. The information button also adopts the same logic. Its code snippet is as below:

```
      // Check if the clicked object is the Information button
      else if (sparam==obj_Btn_INFO.Name()){
         // Print to the log which object was clicked for debugging purposes
         Print("OBJECT CLICKED = ", obj_Btn_INFO.Name());

         // Reset the pressed states of all buttons
         obj_Btn_TRADE.Pressed(false);
         obj_Btn_CLOSE.Pressed(false);
         obj_Btn_INFO.Pressed(false);

         // Set the background color of the Trade and Close buttons to silver, indicating they are inactive
         obj_Btn_TRADE.ColorBackground(clrSilver);
         obj_Btn_CLOSE.ColorBackground(clrSilver);
         // Change the background color of the Info button to yellow to indicate it is active
         obj_Btn_INFO.ColorBackground(clrYellow);

         // Set the border color of the Trade and Close buttons to silver
         obj_Btn_TRADE.ColorBorder(clrSilver);
         obj_Btn_CLOSE.ColorBorder(clrSilver);
         // Set the border color of the Info button to yellow
         obj_Btn_INFO.ColorBorder(clrYellow);

         // Call a function to destroy the Trade section if it exists
         destroySection_Trade();
         // Call a function to destroy the Close section if it exists
         destroySection_Close();

         // Create the Information section, bringing it to the forefront
         createSection_Information();
      }
```

When we compile and run the program, we get the following output on the control button interaction.

![CONTROL CLICKS](https://c.mql5.com/2/98/CONTROL_CLICKS.gif)

We can see that was a success. We now want to destroy the panel when the windows button is clicked.

```
      // Check if the clicked object is the exit button (X button)
      else if (sparam==obj_Btn_X.Name()){
         // Print to the log which object was clicked for debugging purposes
         Print("OBJECT CLICKED = ", obj_Btn_X.Name());

         // Call functions to destroy all sections, effectively closing the entire panel
         destroySection_Trade();
         destroySection_Close();
         destroySection_Information();

         // Call a function to destroy the main panel itself
         destroySection_Main_Panel();
      }
```

Here, we handle the case where we click the exit button by checking if the string parameter matches the name of the exit button using the "Name" function. We first log the name of the clicked object for debugging purposes with the [Print](https://www.mql5.com/en/docs/common/print) function, which helps confirm that the exit button was clicked.

Next, we proceed to close the entire panel by calling several functions. First, we call "destroySection\_Trade", "destroySection\_Close", and "destroySection\_Information" to remove any sections related to trading, closing positions, or displaying account information, if they are currently displayed. These functions ensure that all interactive sections of the panel are properly cleared. Finally, we call the "destroySection\_Main\_Panel" function, which is responsible for destroying the main panel itself. This effectively removes all elements of the panel from the interface, closing the panel completely. Here is an illustration.

![PANEL CLOSE](https://c.mql5.com/2/98/PANEL_CLOSE.gif)

We now need to handle the trading operations when the respective buttons are clicked. To easily do that, we need to include a class instance that aids the process. Thus, we include a trade instance by using [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) at the beginning of the source code. This gives us access to the "CTrade" class, which we use to create a trade object. This is crucial as we need it to do the trade operations.

```
#include <Trade/Trade.mqh>
CTrade obj_Trade;
```

The preprocessor will replace the line #include <Trade/Trade.mqh> with the content of the file Trade.mqh. Angle brackets indicate that the Trade.mqh file will be taken from the standard directory (usually it is terminal\_installation\_directory\\MQL5\\Include). The current directory is not included in the search. The line can be placed anywhere in the program, but usually, all inclusions are placed at the beginning of the source code, for a better code structure and easier reference. Declaration of the obj\_Trade object of the CTrade class will give us access to the methods contained in that class easily, thanks to the MQL5 developers.

![CTRADE CLASS](https://c.mql5.com/2/98/j._INCLUDE_CTRADE_CLASS.png)

After including the trade library, we can start with the logic to open a sell position when the sell button is clicked. Here is the logic we need to adopt.

```
      else if (sparam==obj_Btn_SELL.Name()){ //--- Check if the Sell button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_SELL.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
         double entry_price = Bid; //--- Set the entry price for selling to the current bid price
         double stopLoss = Ask+StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
         double takeprofit = Ask-StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

         Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
         obj_Trade.Sell(lots,_Symbol,entry_price,stopLoss,takeprofit); //--- Execute the sell order
      }
```

Here, once we confirm that the sell button was clicked, we log the button click event using the [Print](https://www.mql5.com/en/docs/common/print) function for debugging purposes, which outputs the name of the clicked button. We then proceed to gather essential trading details. First, we retrieve and normalize the current "ask" and "bid" prices of the symbol using the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function, which fetches the market prices, and the [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) function ensures these prices are properly formatted according to the number of decimal places defined by [\_Digits](https://www.mql5.com/en/docs/predefined/_Digits).

Next, we obtain the lot size from the input field by converting the text from the "Lots" field into a number using the [StringToDouble](https://www.mql5.com/en/docs/convert/stringtodouble) function. The entry price is set to the current "bid" price, as a sell order uses this price for execution. We also calculate the stop loss and take profit values based on the user input. The stop loss is calculated by adding the user-defined value (from the "SL" input field) to the "ask" price, adjusted by the symbol's minimum price increment, [\_Point](https://www.mql5.com/en/docs/predefined/_point). Similarly, the take profit is calculated by subtracting the user-defined value (from the "TP" input field) from the "ask" price.

Finally, we log the order details such as the lot size, entry price, stop loss, and take profit using the Print function. The sell order is executed by calling the "Sell" function from the "obj\_Trade" object, passing the required parameters: the lot size, symbol, entry price, stop loss, and take profit. For the buy position, a similar logic is used as below.

```
      else if (sparam==obj_Btn_BUY.Name()){ //--- Check if the Buy button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_BUY.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
         double entry_price = Ask; //--- Set the entry price for buying to the current ask price
         double stopLoss = Bid-StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
         double takeprofit = Bid+StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

         Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
         obj_Trade.Buy(lots,_Symbol,entry_price,stopLoss,takeprofit); //--- Execute the buy order
      }
```

When we compile and test the progress, we get the following output.

![SELL BUY BUTTONS](https://c.mql5.com/2/98/SELL_BUY_GIF.gif)

To open the limit and stop orders, a similar logic is used. However, since these do not directly use the current market quotes, we will need to configure and incorporate an extra algorithm for security checks. Let's start with the sell-stop button.

```
      else if (sparam==obj_Btn_SELLSTOP.Name()){ //--- Check if the Sell Stop button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_SELLSTOP.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double user_price = StringToDouble(obj_Edit_PRICE.Text()); //--- Get the user-defined price from input field
         long stopslevel = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL); //--- Get the minimum stop level for the symbol
         double valid_price = Bid - stopslevel*_Point; //--- Calculate the valid price for placing a sell stop order

         if (user_price > valid_price){ //--- Check if the user-defined price is valid
            Print("ERROR: INVALID STOPS PRICE. ",user_price," > ",valid_price); //--- Log an error message
         }
         else if (user_price <= valid_price){ //--- If the user-defined price is valid
            double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
            double entry_price = user_price; //--- Set the entry price for the sell stop order
            double stopLoss = user_price+StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
            double takeprofit = user_price-StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

            Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
            obj_Trade.SellStop(lots,entry_price,_Symbol,stopLoss,takeprofit); //--- Execute the sell stop order
         }
      }
```

Here, we handle the automation logic for when the Sell Stop button is clicked by checking if the variable string parameter matches the name of the Sell Stop button using the "Name" function. Once we confirm this, we log the event using the [Print](https://www.mql5.com/en/docs/common/print) function as usual to output the clicked button's name for debugging purposes. We start by retrieving and normalizing the "ask" and "bid" prices using the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function to fetch the market prices and [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) to format them according to the symbol's decimal precision defined by "\_Digits".

Next, we obtain the "user-defined price" from the "Price" input field, converting it from text into a numerical value using the [StringToDouble](https://www.mql5.com/en/docs/convert/stringtodouble) function. Additionally, we retrieve the minimum stop level for the symbol, which is necessary for validating stop orders, using [SymbolInfoInteger](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger) with the "SYMBOL\_TRADE\_STOPS\_LEVEL" parameter. We calculate the valid price by subtracting the stop level (converted to points) from the "bid" price.

We then check if the "user-defined price" is valid by comparing it to the calculated valid price. If the user price exceeds the valid range, we log an error message indicating an "invalid stops price". However, if the user-defined price is valid, we proceed with the following steps. We retrieve the lot size from the input field using StringToDouble to convert the text into a number. The entry price is set to the user-defined price. We then calculate the stop loss by adding the user-defined stop loss value (from the "SL" field) to the entry price and adjusting it based on the symbol's price increment [\_Point](https://www.mql5.com/en/docs/predefined/_point). Similarly, the take profit is calculated by subtracting the user-defined take profit value (from the "TP" field) from the entry price.

Finally, we log the order details, including the lot size, "entry price", "stop loss", and "take profit" using the Print function. The sell stop order is executed by calling the "Sell Stop" function from the "obj\_Trade" object, passing the relevant parameters such as the lot size, entry price, symbol, stop loss, and take profit, placing the sell stop order based on the user’s instructions. To execute the other orders, we use a similar approach.

```
      else if (sparam==obj_Btn_BUYSTOP.Name()){ //--- Check if the Buy Stop button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_BUYSTOP.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double user_price = StringToDouble(obj_Edit_PRICE.Text()); //--- Get the user-defined price from input field
         long stopslevel = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL); //--- Get the minimum stop level for the symbol
         double valid_price = Ask + stopslevel*_Point; //--- Calculate the valid price for placing a buy stop order

         if (user_price < valid_price){ //--- Check if the user-defined price is valid
            Print("ERROR: INVALID STOPS PRICE. ",user_price," < ",valid_price); //--- Log an error message
         }
         else if (user_price >= valid_price){ //--- If the user-defined price is valid
            double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
            double entry_price = user_price; //--- Set the entry price for the buy stop order
            double stopLoss = user_price-StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
            double takeprofit = user_price+StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

            Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
            obj_Trade.BuyStop(lots,entry_price,_Symbol,stopLoss,takeprofit); //--- Execute the buy stop order
         }
      }
      else if (sparam==obj_Btn_SELLLIMIT.Name()){ //--- Check if the Sell Limit button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_SELLLIMIT.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double user_price = StringToDouble(obj_Edit_PRICE.Text()); //--- Get the user-defined price from input field
         long stopslevel = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL); //--- Get the minimum stop level for the symbol
         double valid_price = Bid + stopslevel*_Point; //--- Calculate the valid price for placing a sell limit order

         if (user_price < valid_price){ //--- Check if the user-defined price is valid
            Print("ERROR: INVALID STOPS PRICE. ",user_price," < ",valid_price); //--- Log an error message
         }
         else if (user_price >= valid_price){ //--- If the user-defined price is valid
            double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
            double entry_price = user_price; //--- Set the entry price for the sell limit order
            double stopLoss = user_price+StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
            double takeprofit = user_price-StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

            Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
            obj_Trade.SellLimit(lots,entry_price,_Symbol,stopLoss,takeprofit); //--- Execute the sell limit order
         }
      }
      else if (sparam==obj_Btn_BUYLIMIT.Name()){ //--- Check if the Buy Limit button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_BUYLIMIT.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double user_price = StringToDouble(obj_Edit_PRICE.Text()); //--- Get the user-defined price from input field
         long stopslevel = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL); //--- Get the minimum stop level for the symbol
         double valid_price = Ask - stopslevel*_Point; //--- Calculate the valid price for placing a buy limit order

         if (user_price > valid_price){ //--- Check if the user-defined price is valid
            Print("ERROR: INVALID STOPS PRICE. ",user_price," > ",valid_price); //--- Log an error message
         }
         else if (user_price <= valid_price){ //--- If the user-defined price is valid
            double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
            double entry_price = user_price; //--- Set the entry price for the buy limit order
            double stopLoss = user_price-StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
            double takeprofit = user_price+StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

            Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
            obj_Trade.BuyLimit(lots,entry_price,_Symbol,stopLoss,takeprofit); //--- Execute the buy limit order
         }
      }
```

When we test the buttons, we get the following output data.

![PENDING ORDERS](https://c.mql5.com/2/98/PENDING_ORDERS.gif)

That was a success. We now graduate to closing positions and deleting pending orders. Let us start with the logic to close all the open market positions.

```
      else if (sparam==obj_Btn_CLOSE_ALL.Name()){ //--- Check if the Close All button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     obj_Trade.PositionClose(pos_ticket); //--- Close the position
                  }
               }
            }
         }
      }
```

Here, we just handle the logic for automating the "Close All" button, which will close all active positions when clicked. We begin by checking if the button is clicked and log the event by printing the name of the clicked button to the log for debugging purposes using the Print function. Next, we initiate a [for loop](https://www.mql5.com/en/book/basis/statements/statements_for) that iterates through all open positions in the account by using the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function, which returns the total number of open positions. The loop begins from the last position in the list (represented by "PositionsTotal() - 1") and works backward, decrementing "i" after each iteration to ensure that we handle all positions.

For each iteration, we use the [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) function to retrieve the ticket number of the current position. The ticket number uniquely identifies each position. We then check if the ticket is valid by ensuring it's greater than 0. If valid, we use the [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) function to select the position associated with that ticket.

Once the position is selected, we further check if the position matches the symbol for which we're managing trades by using [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) function with the "POSITION\_SYMBOL" parameter and comparing it to [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), which represents the current trading symbol. If the position matches the symbol, we call the "PositionClose" function from the "obj\_Trade" object and pass the ticket number as a parameter to close the position. This process continues until all positions that match the trading symbol are closed. To close all the sell positions, we will use the same function but add an extra control logic to it as follows.

```
      else if (sparam==obj_Btn_CLOSE_ALL_SELL.Name()){ //--- Check if the Close All Sell button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL_SELL.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_SELL){ //--- Check if the position is a sell position
                        obj_Trade.PositionClose(pos_ticket); //--- Close the sell position
                     }
                  }
               }
            }
         }
      }
```

This is a similar approach to the previous one that we used to close all the positions. Thus, we have highlighted the extra section so we focus on the additional logic used to close only sell positions when the "Close All Sell" button is clicked. After looping through all the positions, we retrieve the position type for each position using the function [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) with the constant [POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer). This value is then cast into the enumeration [ENUM\_POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer) to make it easier to understand, a process called [typecasting](https://www.mql5.com/en/docs/basis/types/casting). Once we have the position type, we specifically check whether the current position is a sell position by comparing it to the [POSITION\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer) property. If the condition evaluates to true, meaning that the position is indeed a sell, we proceed to close it using the "PositionClose" function from the object "obj\_Trade". We do the same when closing the buy positions.

```
      else if (sparam==obj_Btn_CLOSE_ALL_BUY.Name()){ //--- Check if the Close All Buy button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL_BUY.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_BUY){ //--- Check if the position is a buy position
                        obj_Trade.PositionClose(pos_ticket); //--- Close the buy position
                     }
                  }
               }
            }
         }
      }
```

When closing all sell positions that are in a loss, we still need to expand the current approach to incorporate the profit of the selected position, which we can then use to make comparisons and determine whether they are gainers or losers. Here is the logic.

```
      else if (sparam==obj_Btn_CLOSE_LOSS_SELL.Name()){ //--- Check if the Close Loss Sell button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_LOSS_SELL.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_SELL){ //--- Check if the position is a sell position
                        double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                        if (profit_loss < 0){ //--- Check if the position is at a loss
                           obj_Trade.PositionClose(pos_ticket); //--- Close the losing sell position
                        }
                     }
                  }
               }
            }
         }
      }
```

Here, after looping through all the positions and ensuring that the current position is of type sell, we introduce a check for the profit/loss of each position. The [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) function retrieves the current profit/loss of the selected position, using the constant [POSITION\_PROFIT](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer) to determine whether the position is in profit or loss. We then add a conditional statement to check if the profit/loss value is less than zero, which would indicate that the position is in a loss. If this condition is met, we proceed to close the losing sell position by calling the "PositionClose" function. This extra logic, which we have highlighted in yellow, ensures that only sell positions that are currently at a loss will be closed. We use a similar approach for the other buttons.

```
      else if (sparam==obj_Btn_CLOSE_LOSS_BUY.Name()){ //--- Check if the Close Loss Buy button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_LOSS_BUY.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_BUY){ //--- Check if the position is a buy position
                        double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                        if (profit_loss < 0){ //--- Check if the position is at a loss
                           obj_Trade.PositionClose(pos_ticket); //--- Close the losing buy position
                        }
                     }
                  }
               }
            }
         }
      }

      else if (sparam==obj_Btn_CLOSE_PROFIT_SELL.Name()){ //--- Check if the Close Profit Sell button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_PROFIT_SELL.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_SELL){ //--- Check if the position is a sell position
                        double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                        if (profit_loss >= 0){ //--- Check if the position is profitable
                           obj_Trade.PositionClose(pos_ticket); //--- Close the profitable sell position
                        }
                     }
                  }
               }
            }
         }
      }

      else if (sparam==obj_Btn_CLOSE_PROFIT_BUY.Name()){ //--- Check if the Close Profit Buy button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_PROFIT_BUY.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_BUY){ //--- Check if the position is a buy position
                        double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                        if (profit_loss >= 0){ //--- Check if the position is profitable
                           obj_Trade.PositionClose(pos_ticket); //--- Close the profitable buy position
                        }
                     }
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_ALL_LOSS.Name()){ //--- Check if the Close All Loss button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL_LOSS.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                     if (profit_loss < 0){ //--- Check if the position is at a loss
                        obj_Trade.PositionClose(pos_ticket); //--- Close the losing position
                     }
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_ALL_PROFIT.Name()){ //--- Check if the Close All Profit button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL_PROFIT.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                     if (profit_loss >= 0){ //--- Check if the position is profitable
                        obj_Trade.PositionClose(pos_ticket); //--- Close the profitable position
                     }
                  }
               }
            }
         }
      }
```

After we are done with the logic of the positions, we can now graduate to the logic of the pending orders. Here, we delete all the pending orders. This is achieved via the following logic.

```
      else if (sparam==obj_Btn_CLOSE_PENDING.Name()){ //--- Check if the Close Pending button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_PENDING.Name()); //--- Log the button click event

         for (int i = OrdersTotal() -1; i >= 0; i--){ //--- Loop through all pending orders
            ulong order_ticket = OrderGetTicket(i); //--- Get the ticket of the order
            if (order_ticket > 0){ //--- Check if the order ticket is valid
               if (OrderSelect(order_ticket)){ //--- Select the order by ticket
                  if (OrderGetString(ORDER_SYMBOL)==_Symbol){ //--- Check if the order matches the symbol
                     obj_Trade.OrderDelete(order_ticket); //--- Delete the pending order
                  }
               }
            }
         }
      }
```

Here, we focus on the functionality of closing pending orders when the "Close Pending" button is clicked. We start by checking if the string parameter matches the name of the Close Pending button, which signifies that the button has been activated. Upon confirming the button click, we log the action for debugging purposes using the [Print](https://www.mql5.com/en/docs/common/print) function, which outputs the name of the clicked button. Next, we enter a loop that iterates through all pending orders by utilizing the [OrdersTotal](https://www.mql5.com/en/docs/trading/orderstotal) function to get the total number of orders and decrementing the index in reverse order to avoid skipping any orders during deletion. For each order, we retrieve the order ticket number using the [OrderGetTicket](https://www.mql5.com/en/docs/trading/ordergetticket) function. We then check whether the ticket is valid by ensuring it is greater than zero. If valid, we proceed to select the order using the [OrderSelect](https://www.mql5.com/en/docs/trading/orderselect) function.

Once the order is selected, we verify if it corresponds to the symbol specified in [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol) by calling the [OrderGetString](https://www.mql5.com/en/docs/trading/ordergetstring) function with the constant "ORDER\_SYMBOL". If the order matches, we invoke the [OrderDelete](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderdelete) function to delete the pending order associated with the retrieved ticket. This entire process allows for efficiently closing all pending orders linked to the specified trading symbol when the button is activated, ensuring the user's intention to manage their orders effectively. After all that, we just refresh the chart using the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function as below to make sure the changes take effect on the chart.

```
   ChartRedraw(0);
```

That is all that we need to automate the panel. When we run the program and test the closure interface, we get the following output.

![CLOSURE SECTION](https://c.mql5.com/2/98/CLOSURE_GUI_GIF.gif)

The [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) event handler logic that we have incorporated into the system to handle chart click events is as follows:

```
//+------------------------------------------------------------------+
//| Handling chart events                                            |
//+------------------------------------------------------------------+
void OnChartEvent(
   const int       id,       // Event ID indicating the type of event (e.g., mouse click, timer, etc.)
   const long&     lparam,   // Long type parameter associated with the event, usually containing data like mouse coordinates or object IDs
   const double&   dparam,   // Double type parameter associated with the event, used for floating-point values related to the event
   const string&   sparam    // String type parameter associated with the event, typically the name of the object that triggered the event
){
   // Print the 4 function parameters
   //Print("ID = ",id,", LPARAM = ",lparam,", DPARAM = ",dparam,", SPARAM = ",sparam);

   // Check if the event is a click on a chart object
   if (id == CHARTEVENT_OBJECT_CLICK){
      //Print("ID = ",id,", LM = ",lparam,", DM = ",dparam,", SPARAM = ",sparam);
      // Check if the clicked object is the Trade button
      if (sparam==obj_Btn_TRADE.Name()){
         // Print to the log which object was clicked for debugging purposes
         Print("OBJECT CLICKED = ", obj_Btn_TRADE.Name());

         // Reset the pressed states of all buttons to ensure only one button appears pressed at a time
         obj_Btn_TRADE.Pressed(false);
         obj_Btn_CLOSE.Pressed(false);
         obj_Btn_INFO.Pressed(false);

         // Change the background color of the Trade button to yellow to indicate it is active
         obj_Btn_TRADE.ColorBackground(clrYellow);
         // Set the background color of the Close and Info buttons to silver to indicate they are inactive
         obj_Btn_CLOSE.ColorBackground(clrSilver);
         obj_Btn_INFO.ColorBackground(clrSilver);

         // Set the border color of the Trade button to yellow to match its background
         obj_Btn_TRADE.ColorBorder(clrYellow);
         // Set the border color of the Close and Info buttons to silver to indicate they are inactive
         obj_Btn_CLOSE.ColorBorder(clrSilver);
         obj_Btn_INFO.ColorBorder(clrSilver);

         // Call a function to destroy the Close section if it exists
         destroySection_Close();
         // Call a function to destroy the Information section if it exists
         destroySection_Information();

         // Create the Trade section, bringing it to the forefront
         createSection_Trade();
      }
      // Check if the clicked object is the Close button
      else if (sparam==obj_Btn_CLOSE.Name()){
         // Print to the log which object was clicked for debugging purposes
         Print("OBJECT CLICKED = ", obj_Btn_CLOSE.Name());

         // Reset the pressed states of all buttons
         obj_Btn_TRADE.Pressed(false);
         obj_Btn_CLOSE.Pressed(false);
         obj_Btn_INFO.Pressed(false);

         // Set the background color of the Trade button to silver, indicating it's inactive
         obj_Btn_TRADE.ColorBackground(clrSilver);
         // Change the background color of the Close button to yellow to indicate it is active
         obj_Btn_CLOSE.ColorBackground(clrYellow);
         obj_Btn_INFO.ColorBackground(clrSilver);

         // Set the border color of the Trade button to silver
         obj_Btn_TRADE.ColorBorder(clrSilver);
         // Set the border color of the Close button to yellow
         obj_Btn_CLOSE.ColorBorder(clrYellow);
         obj_Btn_INFO.ColorBorder(clrSilver);

         // Call a function to destroy the Trade section if it exists
         destroySection_Trade();
         // Call a function to destroy the Information section if it exists
         destroySection_Information();

         // Create the Close section, bringing it to the forefront
         createSection_Close();
      }
      // Check if the clicked object is the Information button
      else if (sparam==obj_Btn_INFO.Name()){
         // Print to the log which object was clicked for debugging purposes
         Print("OBJECT CLICKED = ", obj_Btn_INFO.Name());

         // Reset the pressed states of all buttons
         obj_Btn_TRADE.Pressed(false);
         obj_Btn_CLOSE.Pressed(false);
         obj_Btn_INFO.Pressed(false);

         // Set the background color of the Trade and Close buttons to silver, indicating they are inactive
         obj_Btn_TRADE.ColorBackground(clrSilver);
         obj_Btn_CLOSE.ColorBackground(clrSilver);
         // Change the background color of the Info button to yellow to indicate it is active
         obj_Btn_INFO.ColorBackground(clrYellow);

         // Set the border color of the Trade and Close buttons to silver
         obj_Btn_TRADE.ColorBorder(clrSilver);
         obj_Btn_CLOSE.ColorBorder(clrSilver);
         // Set the border color of the Info button to yellow
         obj_Btn_INFO.ColorBorder(clrYellow);

         // Call a function to destroy the Trade section if it exists
         destroySection_Trade();
         // Call a function to destroy the Close section if it exists
         destroySection_Close();

         // Create the Information section, bringing it to the forefront
         createSection_Information();
      }
      // Check if the clicked object is the exit button (X button)
      else if (sparam==obj_Btn_X.Name()){
         // Print to the log which object was clicked for debugging purposes
         Print("OBJECT CLICKED = ", obj_Btn_X.Name());

         // Call functions to destroy all sections, effectively closing the entire panel
         destroySection_Trade();
         destroySection_Close();
         destroySection_Information();

         // Call a function to destroy the main panel itself
         destroySection_Main_Panel();
      }
      else if (sparam==obj_Btn_SELL.Name()){ //--- Check if the Sell button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_SELL.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
         double entry_price = Bid; //--- Set the entry price for selling to the current bid price
         double stopLoss = Ask+StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
         double takeprofit = Ask-StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

         Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
         obj_Trade.Sell(lots,_Symbol,entry_price,stopLoss,takeprofit); //--- Execute the sell order
      }
      else if (sparam==obj_Btn_BUY.Name()){ //--- Check if the Buy button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_BUY.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
         double entry_price = Ask; //--- Set the entry price for buying to the current ask price
         double stopLoss = Bid-StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
         double takeprofit = Bid+StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

         Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
         obj_Trade.Buy(lots,_Symbol,entry_price,stopLoss,takeprofit); //--- Execute the buy order
      }
      else if (sparam==obj_Btn_SELLSTOP.Name()){ //--- Check if the Sell Stop button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_SELLSTOP.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double user_price = StringToDouble(obj_Edit_PRICE.Text()); //--- Get the user-defined price from input field
         long stopslevel = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL); //--- Get the minimum stop level for the symbol
         double valid_price = Bid - stopslevel*_Point; //--- Calculate the valid price for placing a sell stop order

         if (user_price > valid_price){ //--- Check if the user-defined price is valid
            Print("ERROR: INVALID STOPS PRICE. ",user_price," > ",valid_price); //--- Log an error message
         }
         else if (user_price <= valid_price){ //--- If the user-defined price is valid
            double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
            double entry_price = user_price; //--- Set the entry price for the sell stop order
            double stopLoss = user_price+StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
            double takeprofit = user_price-StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

            Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
            obj_Trade.SellStop(lots,entry_price,_Symbol,stopLoss,takeprofit); //--- Execute the sell stop order
         }
      }
      else if (sparam==obj_Btn_BUYSTOP.Name()){ //--- Check if the Buy Stop button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_BUYSTOP.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double user_price = StringToDouble(obj_Edit_PRICE.Text()); //--- Get the user-defined price from input field
         long stopslevel = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL); //--- Get the minimum stop level for the symbol
         double valid_price = Ask + stopslevel*_Point; //--- Calculate the valid price for placing a buy stop order

         if (user_price < valid_price){ //--- Check if the user-defined price is valid
            Print("ERROR: INVALID STOPS PRICE. ",user_price," < ",valid_price); //--- Log an error message
         }
         else if (user_price >= valid_price){ //--- If the user-defined price is valid
            double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
            double entry_price = user_price; //--- Set the entry price for the buy stop order
            double stopLoss = user_price-StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
            double takeprofit = user_price+StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

            Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
            obj_Trade.BuyStop(lots,entry_price,_Symbol,stopLoss,takeprofit); //--- Execute the buy stop order
         }
      }
      else if (sparam==obj_Btn_SELLLIMIT.Name()){ //--- Check if the Sell Limit button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_SELLLIMIT.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double user_price = StringToDouble(obj_Edit_PRICE.Text()); //--- Get the user-defined price from input field
         long stopslevel = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL); //--- Get the minimum stop level for the symbol
         double valid_price = Bid + stopslevel*_Point; //--- Calculate the valid price for placing a sell limit order

         if (user_price < valid_price){ //--- Check if the user-defined price is valid
            Print("ERROR: INVALID STOPS PRICE. ",user_price," < ",valid_price); //--- Log an error message
         }
         else if (user_price >= valid_price){ //--- If the user-defined price is valid
            double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
            double entry_price = user_price; //--- Set the entry price for the sell limit order
            double stopLoss = user_price+StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
            double takeprofit = user_price-StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

            Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
            obj_Trade.SellLimit(lots,entry_price,_Symbol,stopLoss,takeprofit); //--- Execute the sell limit order
         }
      }
      else if (sparam==obj_Btn_BUYLIMIT.Name()){ //--- Check if the Buy Limit button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_BUYLIMIT.Name()); //--- Log the button click event

         double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits); //--- Get and normalize the ask price
         double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits); //--- Get and normalize the bid price

         double user_price = StringToDouble(obj_Edit_PRICE.Text()); //--- Get the user-defined price from input field
         long stopslevel = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL); //--- Get the minimum stop level for the symbol
         double valid_price = Ask - stopslevel*_Point; //--- Calculate the valid price for placing a buy limit order

         if (user_price > valid_price){ //--- Check if the user-defined price is valid
            Print("ERROR: INVALID STOPS PRICE. ",user_price," > ",valid_price); //--- Log an error message
         }
         else if (user_price <= valid_price){ //--- If the user-defined price is valid
            double lots = StringToDouble(obj_Edit_LOTS.Text()); //--- Get the lot size from input field
            double entry_price = user_price; //--- Set the entry price for the buy limit order
            double stopLoss = user_price-StringToDouble(obj_Edit_SL.Text())*_Point; //--- Calculate stop loss based on user input
            double takeprofit = user_price+StringToDouble(obj_Edit_TP.Text())*_Point; //--- Calculate take profit based on user input

            Print("Lots = ",lots,", Entry = ",entry_price,", SL = ",stopLoss,", TP = ",takeprofit); //--- Log order details
            obj_Trade.BuyLimit(lots,entry_price,_Symbol,stopLoss,takeprofit); //--- Execute the buy limit order
         }
      }


      else if (sparam==obj_Btn_CLOSE_ALL.Name()){ //--- Check if the Close All button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     obj_Trade.PositionClose(pos_ticket); //--- Close the position
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_ALL_SELL.Name()){ //--- Check if the Close All Sell button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL_SELL.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_SELL){ //--- Check if the position is a sell position
                        obj_Trade.PositionClose(pos_ticket); //--- Close the sell position
                     }
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_ALL_BUY.Name()){ //--- Check if the Close All Buy button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL_BUY.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_BUY){ //--- Check if the position is a buy position
                        obj_Trade.PositionClose(pos_ticket); //--- Close the buy position
                     }
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_LOSS_SELL.Name()){ //--- Check if the Close Loss Sell button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_LOSS_SELL.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_SELL){ //--- Check if the position is a sell position
                        double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                        if (profit_loss < 0){ //--- Check if the position is at a loss
                           obj_Trade.PositionClose(pos_ticket); //--- Close the losing sell position
                        }
                     }
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_LOSS_BUY.Name()){ //--- Check if the Close Loss Buy button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_LOSS_BUY.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_BUY){ //--- Check if the position is a buy position
                        double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                        if (profit_loss < 0){ //--- Check if the position is at a loss
                           obj_Trade.PositionClose(pos_ticket); //--- Close the losing buy position
                        }
                     }
                  }
               }
            }
         }
      }

      else if (sparam==obj_Btn_CLOSE_PROFIT_SELL.Name()){ //--- Check if the Close Profit Sell button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_PROFIT_SELL.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_SELL){ //--- Check if the position is a sell position
                        double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                        if (profit_loss >= 0){ //--- Check if the position is profitable
                           obj_Trade.PositionClose(pos_ticket); //--- Close the profitable sell position
                        }
                     }
                  }
               }
            }
         }
      }

      else if (sparam==obj_Btn_CLOSE_PROFIT_BUY.Name()){ //--- Check if the Close Profit Buy button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_PROFIT_BUY.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); //--- Get the position type
                     if (pos_type == POSITION_TYPE_BUY){ //--- Check if the position is a buy position
                        double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                        if (profit_loss >= 0){ //--- Check if the position is profitable
                           obj_Trade.PositionClose(pos_ticket); //--- Close the profitable buy position
                        }
                     }
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_ALL_LOSS.Name()){ //--- Check if the Close All Loss button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL_LOSS.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                     if (profit_loss < 0){ //--- Check if the position is at a loss
                        obj_Trade.PositionClose(pos_ticket); //--- Close the losing position
                     }
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_ALL_PROFIT.Name()){ //--- Check if the Close All Profit button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_ALL_PROFIT.Name()); //--- Log the button click event

         for (int i = PositionsTotal() -1; i >= 0; i--){ //--- Loop through all positions
            ulong pos_ticket = PositionGetTicket(i); //--- Get the ticket of the position
            if (pos_ticket > 0){ //--- Check if the position ticket is valid
               if (PositionSelectByTicket(pos_ticket)){ //--- Select the position by ticket
                  if (PositionGetString(POSITION_SYMBOL)==_Symbol){ //--- Check if the position matches the symbol
                     double profit_loss = PositionGetDouble(POSITION_PROFIT); //--- Get the profit/loss of the position
                     if (profit_loss >= 0){ //--- Check if the position is profitable
                        obj_Trade.PositionClose(pos_ticket); //--- Close the profitable position
                     }
                  }
               }
            }
         }
      }
      else if (sparam==obj_Btn_CLOSE_PENDING.Name()){ //--- Check if the Close Pending button is clicked
         Print("OBJECT CLICKED = ",obj_Btn_CLOSE_PENDING.Name()); //--- Log the button click event

         for (int i = OrdersTotal() -1; i >= 0; i--){ //--- Loop through all pending orders
            ulong order_ticket = OrderGetTicket(i); //--- Get the ticket of the order
            if (order_ticket > 0){ //--- Check if the order ticket is valid
               if (OrderSelect(order_ticket)){ //--- Select the order by ticket
                  if (OrderGetString(ORDER_SYMBOL)==_Symbol){ //--- Check if the order matches the symbol
                     obj_Trade.OrderDelete(order_ticket); //--- Delete the pending order
                  }
               }
            }
         }
      }


   }


   ChartRedraw(0);

}
```

In a nutshell, this is what we have achieved.

![FULL PANEL ILLUSTRATION](https://c.mql5.com/2/98/FULL_PANEL_GUI.gif)

This was fantastic! We have successfully brought our panel to life, making it fully interactive and responsive. It now features button clicks, live data updates, and responsiveness to active states, enhancing the overall user experience and functionality of our trading interface.

### Conclusion

In conclusion, the enhancements that we have implemented in our [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5")  GUI panel significantly elevate its interactivity and functionality, creating a more engaging user experience. With the addition of live data updates, and responsive button clicks, we can now interact with the panel seamlessly and intuitively. These features not only make it easier to execute buy and sell orders but also ensure that we have immediate access to real-time trading account information, empowering them to make swift and informed trading decisions as market conditions change.

Moreover, the automation of various components, such as position management and account information display, adds a layer of convenience and efficiency to the trading process. By enabling users to close positions and orders with just a click and offering customizable options, the GUI panel becomes a powerful tool for modern traders. This transformation fosters a cleaner, more organized workspace that allows for better focus and productivity. We trust that this article has provided valuable insights into enhancing MQL5 GUI panels and hope you found the explanations clear and informative. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16146.zip "Download all attachments in the single ZIP archive")

[CONTROL\_PANEL\_PART\_2.mq5](https://www.mql5.com/en/articles/download/16146/control_panel_part_2.mq5 "Download CONTROL_PANEL_PART_2.mq5")(74.16 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/475155)**
(9)


![Sergei Poliukhov](https://c.mql5.com/avatar/avatar_na2.png)

**[Sergei Poliukhov](https://www.mql5.com/en/users/operlay)**
\|
27 May 2025 at 20:52

**Allan Munene Mutiiria [#](https://www.mql5.com/ru/forum/487535#comment_56800392):**

Did you even read the article?

I'm looking for such a panel. What I did before stopped working. I'm left with either finding a ready-made one, or using global variables or files and a Python application...

I just read a little bit.

![Sergei Poliukhov](https://c.mql5.com/avatar/avatar_na2.png)

**[Sergei Poliukhov](https://www.mql5.com/en/users/operlay)**
\|
27 May 2025 at 20:53

The panel is beautiful and functional. Thank you.


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
27 May 2025 at 20:57

**Sergei Poliukhov [#](https://www.mql5.com/en/forum/475155#comment_56800531):**

The panel is beautiful and functional. Thank you.

Welcome

![SERGEI NAIDENOV](https://c.mql5.com/avatar/2019/11/5DE2C04F-1280.jpg)

**[SERGEI NAIDENOV](https://www.mql5.com/en/users/leonsi)**
\|
28 May 2025 at 08:04

and there are plans to (minimise/unmount) the panel!? and it would be nice to implement [moving](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "MetaTrader 5 Help: Moving Average indicator") the panel window on the graph!

![Line00](https://c.mql5.com/avatar/2021/1/60049DE9-DE73.png)

**[Line00](https://www.mql5.com/en/users/line00)**
\|
2 Jun 2025 at 10:20

Awesome panel. Great article! Not only does the article describes how to create a perfectly working panel for a trader, but the information about it is presented so clearly that it can be used by beginners as a guide. And not only a guide not only for creating a panel, but also a guide on how to correctly and competently write codes for MQL5. Very valuable and informative article. Thank you to the author, Allan Munene Mutiiria!!!

Regards,

B.V. Dolgikh

![Neural networks made easy (Part 89): Frequency Enhanced Decomposition Transformer (FEDformer)](https://c.mql5.com/2/77/Neural_networks_are_easy_cPart_89q___LOGO.png)[Neural networks made easy (Part 89): Frequency Enhanced Decomposition Transformer (FEDformer)](https://www.mql5.com/en/articles/14858)

All the models we have considered so far analyze the state of the environment as a time sequence. However, the time series can also be represented in the form of frequency features. In this article, I introduce you to an algorithm that uses frequency components of a time sequence to predict future states.

![How to integrate Smart Money Concepts (OB) coupled with Fibonacci indicator for Optimal Trade Entry](https://c.mql5.com/2/98/How_to_integrate_Smart_Money_Concepts__coupled_with_Fibonacci_indicator_for_Optimal_Trade_Entry__LOG.png)[How to integrate Smart Money Concepts (OB) coupled with Fibonacci indicator for Optimal Trade Entry](https://www.mql5.com/en/articles/13396)

The SMC (Order Block) are key areas where institutional traders initiate significant buying or selling. After a significant price move, fibonacci helps to identify potential retracement from a recent swing high to a swing low to identify optimal trade entry.

![Developing a Replay System (Part 49): Things Get Complicated (I)](https://c.mql5.com/2/77/Desenvolvendo_um_sistema_de_Replay_oParte_49q_____LOGO.png)[Developing a Replay System (Part 49): Things Get Complicated (I)](https://www.mql5.com/en/articles/11820)

In this article, we'll complicate things a little. Using what was shown in the previous articles, we will start to open up the template file so that the user can use their own template. However, I will be making changes gradually, as I will also be refining the indicator to reduce the load on MetaTrader 5.

![Creating an MQL5 Expert Advisor Based on the Daily Range Breakout Strategy](https://c.mql5.com/2/98/Creating_an_MQL5_Expert_Advisor_Based_on_the_Daily_Range_Breakout.png)[Creating an MQL5 Expert Advisor Based on the Daily Range Breakout Strategy](https://www.mql5.com/en/articles/16135)

In this article, we create an MQL5 Expert Advisor based on the Daily Range Breakout strategy. We cover the strategy’s key concepts, design the EA blueprint, and implement the breakout logic in MQL5. In the end, we explore techniques for backtesting and optimizing the EA to maximize its effectiveness.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=cnhhiheplnymtytxfhytolccorxncpvx&ssn=1769180380777737859&ssn_dr=0&ssn_sr=0&fv_date=1769180380&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16146&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Create%20an%20Interactive%20MQL5%20Dashboard%2FPanel%20Using%20the%20Controls%20Class%20(Part%202)%3A%20Adding%20Button%20Responsiveness%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918038028475208&fz_uniq=5068889028352212699&sv=2552)

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