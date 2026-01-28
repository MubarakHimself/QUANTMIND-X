---
title: Creating Active Control Panels in MQL5 for Trading
url: https://www.mql5.com/en/articles/62
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:31:14.504089
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=hzchaneyzrseezcoqpnyrrfwugaexxri&ssn=1769178672369337916&ssn_dr=0&ssn_sr=0&fv_date=1769178672&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F62&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20Active%20Control%20Panels%20in%20MQL5%20for%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917867216811998&fz_uniq=5068330330416412719&sv=2552)

MetaTrader 5 / Expert Advisors


### Introduction

The efficiency is very essential in a working environment, especially in the work of traders, where speed and accuracy play a great role. While preparing the terminal for work, each one makes his workplace as comfortable as possible for himself, in order to implement the analyses and enter the market as soon as possible. But the reality of the matter is that developers can not always please everyone and it's impossible to tune to one's desire certain functions.

For example, for a scalper, each fraction of a second and each hit of the "New Order" key is important, and the subsequent setting of all of the parameters might be time critical.

So how do we find a solution? The solution lays in the customize of the elements, since MetaTrader 5 provides such wonderful components as the "Button", "Edit" and "Label". Let us do it.

### 2\. Panel Options

First of all, let's decide what type of functions are essential for a panel. We'll place the main emphasis on trading, using the panel and, therefore, include the following functions:

- Opening position

- Placing of a pending order
- Modification of the position/order
- Closing of the position

- Deleting a pending order

Also, no harm will be done by adding the ability to customize the color scheme panel, font sizes, and saving settings. Let's give a more detailed description of all the elements of the future panel. We'll specify the name of the object, its type and description of its purpose, for each of the panel's functions. The name of each object will begin with "ActP" - this will be a kind of key, indicating that the object belongs to the panel.

**2.1.** **Open Positions**

Below we will introduce all of the necessary parameters for the opening of the position, and will implement it by clicking a button. The auxiliary lines, which are activated by checking a box, will assist us in setting Stop Loss and Take Profit levels. The selection of the execution type will be done using the radio buttons.

| Name | Type | Description |
| --- | --- | --- |
| ActP\_buy\_button1 | Button | Button for a Buy trade |
| ActP\_sell\_button1 | Button | Button for a Sale trade |
| ActP\_DealLines\_check1 | Flag | Set/reset flag of the auxiliary lines |
| ActP\_Exe\_radio1 | Radio button | Group of radio buttons for selecting the trade type |
| ActP\_SL\_edit1 | Input field | Field for inputting a Stop Loss |
| ActP\_TP\_edit1 | Input field | Field for inputting a Take Profit |
| ActP\_Lots\_edit1 | Input field | Field for entering the amount |
| ActP\_dev\_edit1 | Input field | Field for entering a tolerable deviation during the opening |
| ActP\_mag\_edit1 | Input field | Field for entering a number |
| ActP\_comm\_edit1 | Input field | Filed for entering comments |

Table 1 List of the panel elements, "Trade opening"

**2.2 Placing a Pending Order**

Below we will introduce all of the necessary parameters for the placing of a pending order, and place them by pressing a key. Supporting lines, which are activated by checking a flag, will help to set Stop Loss, Take Profit, stop-limit levels and expiration times. Selection of the type of execution and the type of expiration time, will be performed with the help of a groups of radio buttons.

| Name | Type | Description |
| --- | --- | --- |
| ActP\_buy\_button2 | Button | Button for setting a Buy order |
| ActP\_sell\_button2 | Button | Button for setting a trade order |
| ActP\_DealLines\_check2 | Flag | The auxiliary lines set / reset flag |
| ActP\_lim\_check2 | Flag | Order stop-limit set/reset flag |
| ActP\_Exe\_radio2 | Radio button | Group of radio button for selecting the type of order execution |
| ActP\_exp\_radio2 | Radio button | Group of radio button for selecting the type of the order expiration |
| ActP\_SL\_edit2 | Input field | Field for inputting a Stop Loss |
| ActP\_TP\_edit2 | Input field | Field for inputting a Take Profit |
| ActP\_Lots\_edit2 | Input field | Field for entering the amount |
| ActP\_limpr\_edit2 | Input field | Field for entering the price of a stop-limit order |
| ActP\_mag\_edit2 | Input field | Field for entering the magic number |
| ActP\_comm\_edit2 | Input field | Field for comments |
| ActP\_exp\_edit2 | Input field | Field for entering the expiration time |
| ActP\_Pr\_edit2 | Input field | Field for entering the price of the order execution |

Table 2 List of the elements of the "Placing pending orders" panel

**2.3.** **Modification / Closing of Trades**

Below we will introduce all of the necessary parameters for the modification and the closing of a trade. The auxiliary lines, which are activated by checking a box, will assist us in the installation of Stop Loss and Take Profit levels. The selection of trades will be generated from a dropdown list.

| Name | Type | Description |
| --- | --- | --- |
| ActP\_ord\_button5 | Dropdown list | List of selections for a trade |
| ActP\_mod\_button4 | Button | Trade modification button |
| ActP\_del\_button4 | Button | Trade closing button |
| ActP\_DealLines\_check4 | Flag | Auxiliary lines set/reset flag |
| ActP\_SL\_edit4 | Input field | Field for inputting a Stop Loss |
| ActP\_TP\_edit4 | Input field | Field for inputting a Take Profit |
| ActP\_Lots\_edit4 | Input field | Field for entering the amount |
| ActP\_dev\_edit4 | Input field | Field for entering a tolerable deviation |
| ActP\_mag\_edit4 | Input field | Field for displaying the magic number (read only) |
| ActP\_Pr\_edit4 | Input field | Field to display the opening price (read only) |

Table 3. List of the elements of the "Trade modification / closing" panel

**2.4.** **Modification / Removal of Orders**

Below we will introduce all of the necessary parameters for modification and removal of pending orders. Supporting lines, which are activated by checking a box, will assist in the installation of stops, takes, stop-limit levels, and expiration times. Selecting the type of expiration times will be generated with the help of a groups of radio buttons. Selection of orders will be generated from a dropdown list.

| Name | Type | Description |
| --- | --- | --- |
| ActP\_ord\_button5 | Dropdown list | List to select the order |
| ActP\_mod\_button3 | Button | Order modification button |
| ActP\_del\_button3 | Button | Order removal button |
| ActP\_DealLines\_check3 | Flag | Auxiliary lines set/reset flag |
| ActP\_exp\_radio3 | Radio button | Group of radio buttons for selecting the type of expiration of an order |
| ActP\_SL\_edit3 | Input field | Field for inputting a Stop Loss |
| ActP\_TP\_edit3 | Input field | Field for inputting a take |
| ActP\_Lots\_edit3 | Input field | Field that displays volume (read only) |
| ActP\_limpr\_edit3 | Input field | Field for inputting the price for a stoplimit order |
| ActP\_mag\_edit3 | Input field | Field that displays magic numbers (read only) |
| ActP\_comm\_edit3 | Input field | Field for comments |
| ActP\_exp\_edit3 | Input field | Field for inputting the expiration time |
| ActP\_Pr\_edit3 | Input field | Field for entering the price of order execution |
| ActP\_ticket\_edit3 | Input field | Field that displays the order ticket (read only) |

Table 4. List of the elements of the "Modification / removal of orders" panel

**2.5 Settings**

Below we will chose the color of buttons, labels, and texts from the dropdown list, as well as set up various font sized.

| Name | Type | Description |
| --- | --- | --- |
| ActP\_col1\_button6 | Dropdown list | List of color selections for buttons |
| ActP\_col2\_button6 | Dropdown list | List of color selection for tags |
| ActP\_col3\_button6 | Dropdown list | List of text color selection |
| ActP\_font\_edit6 | Input field | Field for specifying text size |

Table 5. List of elements of the "Settings" panel

A button is also added to create the possibility of minimizing the panel if it isn't being used. You might have noticed the presence of such an instrument as "support lines". What are they and why do we need them? Through the use of these lines, we will be able to set up a Stop Loss, Take Profit, the price of triggering of a pending order, the price of a stop-limit order (horizontal lines), as well as the expiration time of a postponed order (vertical line), simply by using the mouse to drag these lines to the desired price/time.

After all, a visual installation is more convenient than a textual one (manually inputting prices / time into the appropriate field ). Also, these lines will serve us as "highlights" of parameter of a selected order. Since there can be a lot of orders, the standard terminal shaded lines, which usually display prices, can become very confusing.

### 3\. The General Approach to the Interface Creation

So we have successfully set forth our objective - to create a form of graphical assistant within the trade. For this purpose, we need the most user-friendly interface. First, it must be clear that all elements of control (and there will be many) will have to be created using software, and therefore the position and size of objects needs to be precalculated.

Now, imagine that we went through a long, tedious and hard time, working out the coordinates of the objects, making sure that they don't overlap one another and are clearly visible; and then there is a need to add a new object, and our whole scheme now needs to be rebuilt!

Those who are familiar with the Rapid Application Development environment (Delphi, C + + Builder, etc.) know how quickly the most complicated user interface can be created.

Let us try to implement it using MQL5. First, using a mouse, we locate the objects of control in the most appropriate manner, and adjust their sizes. Then, we write a simple script, which reads the properties of all objects on the chart, and records them to a file, and when needed, we will easily be able to retrieve those properties and completely reconstruct the objects on any chart.

The code of the script may look like this:

```
//+------------------------------------------------------------------+
//|                                  Component properties writer.mq5 |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
#property version   "1.00"
#property script_show_inputs

input int interfaceID=1; //input parameter - the identifier of the stored interface
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   //Open file for writing
   int handle=FileOpen("Active_Panel_scheme_"+IntegerToString(interfaceID)+".bin", FILE_WRITE|FILE_BIN);
   if(handle!=INVALID_HANDLE)
     {
      //We will go all the objects on the chart
      for(int i=0;i<ObjectsTotal(0);i++)
        {
         string name=ObjectName(0,i);
         //And write their properties in the file
         FileWriteString(handle,name,100);
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_TYPE));

         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_XDISTANCE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_YDISTANCE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_XSIZE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_YSIZE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_COLOR));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_STYLE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_WIDTH));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_BACK));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_SELECTED));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_SELECTABLE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_READONLY));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_FONTSIZE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_STATE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_BGCOLOR));

         FileWriteString(handle,ObjectGetString(0,name,OBJPROP_TEXT),100);
         FileWriteString(handle,ObjectGetString(0,name,OBJPROP_FONT),100);
         FileWriteString(handle,ObjectGetString(0,name,OBJPROP_BMPFILE,0),100);
         FileWriteString(handle,ObjectGetString(0,name,OBJPROP_BMPFILE,1),100);

         FileWriteDouble(handle,ObjectGetDouble(0,name,OBJPROP_PRICE));
        }
      //Close file
      FileClose(handle);
      Alert("Done!");
     }
  }
//+------------------------------------------------------------------+
```

As you can see, the code is extremely simple, it writes to a binary file some properties of all chart objects. The most important thing is not to forget the sequence order of the recorded properties while reading the file.

The script is ready, so let's turn to the creation of the interface.

And the first thing we will do is organize the main menu by the type of its tabs. Why do we need tabs? Because there are a lot of objects, and fitting them all on the screen would be problematic. And since the objects are grouped accordingly (see table above), it is easier to place each group on a separate tab.

Thus, using the terminal menu Insert -> Objects -> Button, we will create five buttons at the top of the chart, which will serve as our main menu.

![Figure 1. Panel tabs](https://c.mql5.com/2/1/figure1__2.png)

Fig. 1 Panel tabs

Let's not forget that objects can be easily duplicated, by selecting one, and then dragging it with a mouse, while holding down the "Ctrl" key. By doing this we will create a copy of the object rather than relocate its original.

Special attention should be given to the names of the objects, not forgetting that they must all begin with "ActP". In addition, we add "main" to the name of the string, which indicates that that the object belongs to the main menu bar.

![Figure 2. List of objects (panel tabs)](https://c.mql5.com/2/1/figure2__2.png)

Figure 2\. List of objects
(panel tabs)

Similarly, let's apply the tab contents to the new chart. The contents of each tab should be placed on a separate chart!

**Tab "Market":**

![Figure 3. Elements of the "Market" tab](https://c.mql5.com/2/1/figure3__1.png)

Figure 3. Elements of the "Market" tab

**Tab "Pending":**

![Figure 4. Elements of the "Pending" tab](https://c.mql5.com/2/1/figure4__1.png)

Figure 4. Elements of the "Pending" tab

**Settings tab:**

![Figure 5. Elements of the "Settings" tab](https://c.mql5.com/2/1/figure5.png)

Figure 5\. Elements of the "Settings" tab

The last **tab "Modify / close" is different,** it will serve to modify / delete pending orders, as well as modify and close trading deals. It will be reasonable to divide the work with ttrades and the work with orders into two separate sub-tabs. First, let's create a button which will activate the drop-down list, from which we will choose an order or a trade to work with.

![Figure 6. Elements of the  "Modify/Close" tab](https://c.mql5.com/2/1/figure6.png)

Figure
6\. Elements of the
"Modify/Close" tab

Afterwards we create sub-tabs. To work with trades:

![Figure 7. Elements for working with positions](https://c.mql5.com/2/1/figure7.png)

Figure

7\. Elements for
working with positions

And for working with orders:

![Figure 8. Sub-tab for working with orders](https://c.mql5.com/2/1/figure8.png)

Figure 8\. Sub-tab for working with orders

That's it, the interface is created.

We apply the script to each of the charts to save each tab in a separate file. The input parameter "interfaceID" must be different for each tab:

- 0 - Home page
- 1 - Market
- 2 - Pending
- 3 - Button for activating trade / order selection list

- 4 - Settings
- 6 - Sub-tab to work with trades
- 7 - Sub-tab to work with orders

Tab number 5 corresponds to the button of "Minimize window" on the main menu, so there are no objects on it, and we can skip it.

After all of these manipulations, the following files will appear in the directory folder of the terminal -> MQL5 ->:

![Figure 8. List of files of panels schemes](https://c.mql5.com/2/1/figure8__1.png)

Figure 9\. List of files with panels schemes

### 4\. Downloading Interface Elements

Now the interface elements are stored in files and are ready to be put to work. To start with, let's determine the place where our panel will be located.  If we locate it directly on the main chart, it will block the prices chart, which is very inconvenient. Therefore, it will be most reasonable to place the panel in the sub-window of the main chart. An indicator can create this pane.

Let's create it:

```
#property copyright "Copyright 2010, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
#property version   "1.00"
#property indicator_separate_window //place the indicator in a separate window

int OnInit()
  {
//--- indicator buffers mapping
   //Set the short name of the indicator
   IndicatorSetString(INDICATOR_SHORTNAME, "AP");
//---
   return(0);
  }

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
//---
//--- return value of prev_calculated for next call
   return(rates_total);
  }
```

The code is very simple because the main function of this indicator is the creation of sub-windows, rather than making various calculations. The one thing that we will do is install a "short" name of the indicator, by which we can find its sub-window. We'll compile and apply a chart onto the indicator, and a window will appear.

Now let's focus on the Expert Advisor panel. We'll create a new Expert Advisor.

The OnInit () function will contain the following operators:

```
double Bid,Ask;         //variables for current prices
datetime time_current;  //time of last tick
int wnd=-1;             //index of the window with an indicator
bool last_loaded=false; //flag indicating whether it's a first initialization or not
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   //Start the timer at intervals of 1 second
   EventSetTimer(1);
   //Get the latest prices
   get_prices();
   //Define the window with an indicator
   wnd=ChartWindowFind(0,"AP");
   //If the first initialization - create interface
   if(!last_loaded) create_interface();
//---
   return(0);
  }
```

Here we launch a timer (why that's done will be explained below), obtain the latest prices from the market, using the ChartWindowFind, locate the indicator window and save it as a variable. Flag last\_loaded - indicates whether or not this is the first time the Expert Advisor is intialized. This information will be needed in order to avoid reloading the interface during a reinitialization.

The create\_interface () function looks the following way:

```
//+------------------------------------------------------------------+
//| Function of the interface creation                               |
//+------------------------------------------------------------------+
void create_interface()
  {
   //if reset settings is selected
   if(Reset_Expert_Settings)
     {
     //Reset
      GlobalVariableDel("ActP_buttons_color");
      GlobalVariableDel("ActP_label_color");
      GlobalVariableDel("ActP_text_color");
      GlobalVariableDel("ActP_font_size");
     }

   //Create the main menu interface
   ApplyScheme(0);
   //Create the interface tab "Market"
   ApplyScheme(1);
   //Set all objects as unmarked
   Objects_Selectable("ActP",false);
   //redraw the chart
   ChartRedraw();
  }
```

The first step is to check the input parameter "reset settings", and if it is installed, remove the global variables, responsible for the settings. How this action affects the panel will be described below. Further, the ApplyScheme () function will create an interface from a file.

```
//+------------------------------------------------------------------+
//| The function for the interface loading                           |
//| ID - ID of the saved interface                                   |
//+------------------------------------------------------------------+
bool ApplyScheme(int ID)
  {
   string fname="Active_Panel_scheme_custom_"+IntegerToString(ID)+".bin";
   //download the standard scheme if there isn't saved scheme
   if(!FileIsExist(fname)) fname="Active_Panel_scheme_"+IntegerToString(ID)+".bin";
   //open file for reading
   int handle=FileOpen(fname,FILE_READ|FILE_BIN);
   //file opened
   if(handle!=INVALID_HANDLE)
     {
      //Loading all objects
      while(!FileIsEnding(handle))
        {
         string obj_name=FileReadString(handle,100);
         int _wnd=wnd;
         //the auxiliary lines are in the main window
         if(StringFind(obj_name,"line")>=0) _wnd=0;
         ENUM_OBJECT obj_type=FileReadInteger(handle);
         //creating object
         ObjectCreate(0, obj_name, obj_type, _wnd, 0, 0);
         //and apply the properties
         ObjectSetInteger(0,obj_name,OBJPROP_XDISTANCE,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_YDISTANCE,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_XSIZE,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_YSIZE,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_COLOR,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_STYLE,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_WIDTH,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_BACK,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_SELECTED,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_SELECTABLE,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_READONLY,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_FONTSIZE,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_STATE,FileReadInteger(handle));
         ObjectSetInteger(0,obj_name,OBJPROP_BGCOLOR,FileReadInteger(handle));

         ObjectSetString(0,obj_name,OBJPROP_TEXT,FileReadString(handle,100));
         ObjectSetString(0,obj_name,OBJPROP_FONT,FileReadString(handle,100));
         ObjectSetString(0,obj_name,OBJPROP_BMPFILE,0,FileReadString(handle,100));
         ObjectSetString(0,obj_name,OBJPROP_BMPFILE,1,FileReadString(handle,100));

         ObjectSetDouble(0,obj_name,OBJPROP_PRICE,FileReadDouble(handle));

         //Set color for the objects
         if(GlobalVariableCheck("ActP_buttons_color") && obj_type==OBJ_BUTTON)
            ObjectSetInteger(0,obj_name,OBJPROP_BGCOLOR,GlobalVariableGet("ActP_buttons_color"));
         if(GlobalVariableCheck("ActP_label_color") && obj_type==OBJ_LABEL)
            ObjectSetInteger(0,obj_name,OBJPROP_COLOR,GlobalVariableGet("ActP_label_color"));
         if(GlobalVariableCheck("ActP_text_color") && (obj_type==OBJ_EDIT || obj_type==OBJ_BUTTON))
            ObjectSetInteger(0,obj_name,OBJPROP_COLOR,GlobalVariableGet("ActP_text_color"));
         if(GlobalVariableCheck("ActP_font_size") && (obj_type==OBJ_EDIT || obj_type==OBJ_LABEL))
            ObjectSetInteger(0,obj_name,OBJPROP_FONTSIZE,GlobalVariableGet("ActP_font_size"));
         //Set global variable font size
         if(obj_name=="ActP_font_edit6" && GlobalVariableCheck("ActP_font_size"))
            ObjectSetString(0,obj_name,OBJPROP_TEXT,IntegerToString(GlobalVariableGet("ActP_font_size")));
        }
      //Close file
      FileClose(handle);
      return(true);
     }
   return(false);
  }
```

Once again, there is nothing complicated about this. The function will open the desired file, with a pre-saved interface scheme and will create it in the window, which we previously identified (indicator window). Also we select the colors of objects and font sizes from the global variables of the terminal.

The Objects\_Selectable () function makes all of the objects, except for auxiliary lines, unmarked, in order to turn on the animations of buttons and avoid accidently deleting a necessary object.

```
//+------------------------------------------------------------------+
//| Function of setting objects as unselectable                      |
//+------------------------------------------------------------------+
void  Objects_Selectable(string IDstr,bool flag)
  {
   //Check all the objects
   for(int i=ObjectsTotal(0);i>=0;i--)
     {
      string n=ObjectName(0,i);
      //If the object belongs to the panel
      if(StringFind(n,IDstr)>=0)
        {
         //Lines remain untouched
         if(!flag)
            if(StringFind(n,"line")>-1) continue;
         //Set everything unselectable except the lines
         ObjectSetInteger(0,n,OBJPROP_SELECTABLE,flag);
        }
     }
  }
```

Now let's look at the OnTick() function. It will serve us to obtain the latest prices on the market.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //Get the latest prices
   get_prices();
  }
```

Function get\_prices() has the form:

```
//+------------------------------------------------------------------+
//| Function obtain information on tick                              |
//+------------------------------------------------------------------+
void get_prices()
  {
   MqlTick tick;
   //if the tick was
   if(SymbolInfoTick(Symbol(),tick))
     {
      //obtain information
      Bid=tick.bid;
      Ask=tick.ask;
      time_current=tick.time;
     }
  }
```

And do not forget about OnDeinit ():

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   //if the deinitialisation reason isn't the timeframe or symbol change
   if(reason!=REASON_CHARTCHANGE)
     {
      //reset initialization flag
      last_loaded=false;
      //Delete all panel objects
      ObjectsDeleteAll_my("ActP");
      //Delete files with the saved state of the tabs
      FileDelete("Active_Panel_scheme_custom_1.bin");
      FileDelete("Active_Panel_scheme_custom_2.bin");
      FileDelete("Active_Panel_scheme_custom_3.bin");
      FileDelete("Active_Panel_scheme_custom_4.bin");
      FileDelete("Active_Panel_scheme_custom_5.bin");
     }
   //otherwise set a flag
   else last_loaded=true;
   //stop the timer
   EventKillTimer();
  }
```

First check the cause of deinitialisation: if it's due to a change of a timeframe and / or symbols, we will not delete the panel item. In all other cases, remove all items, using the ObjectsDeleteAll\_my () function.

```
//+------------------------------------------------------------------+
//| The function deletes all panel objects                           |
//| IDstr - object identifier                                        |
//+------------------------------------------------------------------+
void  ObjectsDeleteAll_my(string IDstr)
  {
   //check all the objects
   for(int i=ObjectsTotal(0);i>=0;i--)
     {
      string n=ObjectName(0,i);
      //if the name contains the identifier - remove the object
      if(StringFind(n,IDstr)>=0) ObjectDelete(0,n);
     }
  }
```

After compiling and running the Expert Advisor, we obtain the following result:

![Figure 10. Example of Expert Advisor work](https://c.mql5.com/2/1/figure10x.png)

Figure 10\. Example of Expert
Advisor work

However there is little use from all of this until we are able to make these objects respond to our manipulation.

### 5\. Event Handling

The interface is created, now we have to get it to work. All of our actions with objects generate specific events. The OnChartEvent function OnChartEvent(const int id, const long& lparam, const
double& dparam, const string& sparam) is the handling mechanism of events [ChartEvent](https://www.mql5.com/en/docs/runtime/event_fire#chartevent) . Of all the events we are interested in the following:

- CHARTEVENT\_CLICK - click on the chart
- CHARTEVENT\_OBJECT\_ENDEDIT - finished editing the input field
- CHARTEVENT\_OBJECT\_CLICK - click on the graphic object

In our case, the parameter of the id function indicates the ID of the event, sparam - indicates the name of the object, which generates this event, and all other parameters are not of interest to us.

The first event that we will explore is the - click on the main menu button.

**5.1.** **Handling Main Menu Events**

Recall that the main menu consists of five buttons. When one of them is clicked on, it should go into a pressed mode, direct us to the right interface and upload the appropriate tabs. Then all of the other menu buttons should go into the unpressed mode.

```
//+------------------------------------------------------------------+
//| Event handlers                                                   |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on a graphic object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //main menu button click
      if(sparam=="ActP_main_1") {Main_controls_click(1); ChartRedraw(); return;}
       //Here we execute the corresponding operators
      if(sparam=="ActP_main_2") {Main_controls_click(2); ChartRedraw(); return;}
      if(sparam=="ActP_main_3") {Main_controls_click(3); ChartRedraw(); return;}
      if(sparam=="ActP_main_4") {Main_controls_click(4); ChartRedraw(); return;}
      if(sparam=="ActP_main_5") {Main_controls_click(5); ChartRedraw(); return;}
   ...
   }
...
}
```

If there was a click on the menu button, then we have performed the Main\_controls\_click() function. Let's redraw the chart using ChartRedraw(), and complete the function. We should complete the execution because only one object can be clicked on at one time, and therefore, all further implementations will lead to a waste of CPU time.

```
//+------------------------------------------------------------------+
//| Tab processor                                                    |
//| ID - index of clicked tab                                        |
//+------------------------------------------------------------------+
void Main_controls_click(int ID)
  {
   int loaded=ID;
   //we will go all tabs
   for(int i=1;i<6;i++)
     {
      //for all except the selected set inactive
      if(i!=ID)
        {
         //also remember the last active tab
         if(ObjectGetInteger(0,"ActP_main_"+IntegerToString(i),OBJPROP_STATE)==1) loaded=i;
         ObjectSetInteger(0,"ActP_main_"+IntegerToString(i),OBJPROP_STATE,0);
        }
     }
//if(loaded==ID) return;
   //set an active state for the selected
   ObjectSetInteger(0,"ActP_main_"+IntegerToString(ID),OBJPROP_STATE,1);
   //delete the drop-down lists
   DeleteLists("ActP_orders_list_");
   DeleteLists("ActP_color_list_");
   //and set the list buttons to the unpressed state
   ObjectSetInteger(0,"ActP_ord_button5",OBJPROP_STATE,0);
   ObjectSetInteger(0,"ActP_col1_button6",OBJPROP_STATE,0);
   ObjectSetInteger(0,"ActP_col2_button6",OBJPROP_STATE,0);
   ObjectSetInteger(0,"ActP_col3_button6",OBJPROP_STATE,0);
   //save state of the last active tab
   SaveScheme(loaded);
   //remove old tab
   DeleteScheme("ActP");
   //and load a new
   ApplyScheme(ID);
   //Set all objects as unselected
   Objects_Selectable("ActP",false);
  }
```

We have been introduced to the Objects\_Selectable() and ApplyScheme() functions, and we will later turn to the DeleteLists() function.

The SaveScheme() function saves an interface file, so that the objects retain all of their properties during a reload:

```
//+------------------------------------------------------------------+
//| Interface saving function                                        |
//+------------------------------------------------------------------+
void SaveScheme(int interfaceID)
  {
   //open file for writing
   int handle=FileOpen("Active_Panel_scheme_custom_"+IntegerToString(interfaceID)+".bin",FILE_WRITE|FILE_BIN);
   //if file opened
   if(handle!=INVALID_HANDLE)
     {
      //go all the chart objects
      for(int i=0;i<ObjectsTotal(0);i++)
        {
         string name=ObjectName(0,i);
         //if the object belongs to the panel
         if(StringFind(name,"ActP")<0) continue;
         //and it isn't a tab
         if(StringFind(name,"main")>=0) continue;
         //write the object properties to a file
         FileWriteString(handle,name,100);
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_TYPE));

         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_XDISTANCE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_YDISTANCE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_XSIZE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_YSIZE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_COLOR));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_STYLE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_WIDTH));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_BACK));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_SELECTED));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_SELECTABLE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_READONLY));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_FONTSIZE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_STATE));
         FileWriteInteger(handle,ObjectGetInteger(0,name,OBJPROP_BGCOLOR));

         FileWriteString(handle,ObjectGetString(0,name,OBJPROP_TEXT),100);
         FileWriteString(handle,ObjectGetString(0,name,OBJPROP_FONT),100);
         FileWriteString(handle,ObjectGetString(0,name,OBJPROP_BMPFILE,0),100);
         FileWriteString(handle,ObjectGetString(0,name,OBJPROP_BMPFILE,1),100);

         FileWriteDouble(handle,ObjectGetDouble(0,name,OBJPROP_PRICE));
        }
      //Close file
      FileClose(handle);
     }
  }
```

The DeleteScheme() function removes the tab objects.

```
//+------------------------------------------------------------------+
//| Function to delete all the panel objects, except tabs            |
//+------------------------------------------------------------------+
void  DeleteScheme(string IDstr)
  {
   //we will go through all the objects
   for(int i=ObjectsTotal(0);i>=0;i--)
     {
      string n=ObjectName(0,i);
      //remove everything but the tab
      if(StringFind(n,IDstr)>=0 && StringFind(n,"main")<0) ObjectDelete(0,n);
     }
  }
```

Thus, by performing the Main\_controls\_click() function, we will remove the old tab, saving it beforehand, and load a new one.

By compiling the Expert Advisor, we'll see the results.

Now we will click the main menu button, load the new tabs, keeping them in the state of the original tabs.

![Figure 11. Items of the "Pending" tab](https://c.mql5.com/2/1/figure10_.png)

Figure 11\. Items of the "Pending" tab

![Figure 12. Elements of the "Modify/Close" tab](https://c.mql5.com/2/1/figure11_.png)

Figure 12\. Elements of the "Modify/Close" tab

![Figure 12. Elements of the "Settings" tab](https://c.mql5.com/2/1/figure12.png)

Figure 13\. Elements of the "Settings" tab

With this we can finish the manipulation of the main menu, since it now fully serves its functions.

**5.2. H** **andling of the "Flag" Component Event**

The setting of auxiliary lines and stoplimit orders is made by using the "flag" components, but it is not in the list of graphical objects of MT5. So let's create it. There is a "graphic label" object, which factually is an image which has a state of "on" and a state of "off". The state can be varied by clicking on the object. A separate image can be set for each state. Choose an image for each of the states:

-  Enabled ![](https://c.mql5.com/2/1/check_on.bmp)
-  Disabled ![](https://c.mql5.com/2/1/check_off.bmp)

Let's set the pictures in the properties of the object:

![Figure 13. Setting the properties of the "flag" element](https://c.mql5.com/2/1/f13.png)

Figure 13. Setting the properties of the "flag" element

It must be reminded that for the pictures to be available in the list, they need to be located in the folder "Terminal folder-> MQL5-> Images" and have the extension ".Bmp".

Let's turn to processing events, which occur when you click on an object. We'll use the example of the flag, which is responsible for the placing of auxiliary lines at the opening of the trade.

```
      //click on the flag of the setting of auxiliary lines during transaction opening
      if(sparam=="ActP_DealLines_check1")
      {
         //Check the flag state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         //If the flag is set
         if(selected)
         {
            //Retrieve the value of the stop loss and take profit from the input fields
            string SL_txt=ObjectGetString(0, "ActP_SL_edit1", OBJPROP_TEXT);
            string TP_txt=ObjectGetString(0, "ActP_TP_edit1", OBJPROP_TEXT);
            double val_SL, val_TP;

            //If the Stop field is not empty
            //save the value
            if(SL_txt!="")
               val_SL=StringToDouble(SL_txt);

            //if empty
            else
            {
               //Take the max. and min. prices of chart
               double pr_max=ChartGetDouble(0, CHART_PRICE_MAX);
               double pr_min=ChartGetDouble(0, CHART_PRICE_MIN);
               //Set the stop at the 1/3 of the chart price range
               val_SL=pr_min+(pr_max-pr_min)*0.33;
            }

            //Similarly processes the Take
            if(TP_txt!="")
               val_TP=StringToDouble(TP_txt);
            else
            {
               double pr_max=ChartGetDouble(0, CHART_PRICE_MAX);
               double pr_min=ChartGetDouble(0, CHART_PRICE_MIN);
               val_TP=pr_max-(pr_max-pr_min)*0.33;
            }
            //Move the line to new positions
            ObjectSetDouble(0, "ActP_SL_line1", OBJPROP_PRICE, val_SL);
            ObjectSetDouble(0, "ActP_TP_line1", OBJPROP_PRICE, val_TP);
         }
          //If the flag is unset
         else
         {
             //remove the lines
            ObjectSetDouble(0, "ActP_SL_line1", OBJPROP_PRICE, 0);
            ObjectSetDouble(0, "ActP_TP_line1", OBJPROP_PRICE, 0);
         }
          //redraw the chart
         ChartRedraw();
          //and finish the function
         return;
      }
```

The same method is used for the flags, which are responsible for the processing and installation of auxiliary lines on the closing / modification of pending orders tab. Therefore, we will not go into details about them in this article. Those who wish to familiarize themselves with them can use the Expert Advisor code.

The setting of stoplimit orders flag on the "Pending" tab has the following handler:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...

   //Event - click on a graphic object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
      //Click on the orders stoplimit check box
      if(sparam=="ActP_limit_check2")
      {
         //Check the flag state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         if(selected) //if flag is set
         {
            //set the new color for the price edit
            ObjectSetInteger(0, "ActP_limpr_edit2", OBJPROP_BGCOLOR, White);
            //enable it for the edit
            ObjectSetInteger(0, "ActP_limpr_edit2", OBJPROP_READONLY, false);
            //установим в поле значение текущей цены
            //Set the current price as the field value
            ObjectSetString(0, "ActP_limpr_edit2", OBJPROP_TEXT, DoubleToString(Bid, _Digits));
            //if the auxiliary lines are allowed
            //move them
            if(ObjectGetInteger(0, "ActP_DealLines_check2", OBJPROP_STATE)==1)
               ObjectSetDouble(0, "ActP_lim_line2", OBJPROP_PRICE, Bid);
         }
          //if flag is unset
         else
         {
            //set the field unavailable for editing
            ObjectSetInteger(0, "ActP_limpr_edit2", OBJPROP_BGCOLOR, LavenderBlush);
            //set the field color
            ObjectSetInteger(0, "ActP_limpr_edit2", OBJPROP_READONLY, true);
            //and "empty" text
            ObjectSetString(0, "ActP_limpr_edit2", OBJPROP_TEXT, "");
            //if the auxiliary lines are allowed
            //move them to the zero point
            if(ObjectGetInteger(0, "ActP_DealLines_check2", OBJPROP_STATE)==1)
               ObjectSetDouble(0, "ActP_lim_line2", OBJPROP_PRICE, 0);
         }
      }
   ...
   }
...
}
```

We have now finished working with flags. Let's consider the following object of our own production - "radio buttons group".

**5.3.** **Handling of the "Radio buttons Group" Component Event**

Using this component, we select the type of a trade and the type of order expiration. Just like in the case with the flags, we will use graphic tags, but this time, with new pictures.

- Enabled ![](https://c.mql5.com/2/1/radio_on.bmp)
- Disabled ![](https://c.mql5.com/2/1/radio_off.bmp)

But here the problem is complicated by the need to reset all the radio buttons, except for the one you click, to an inactive state. Consider the example of the radio button of order execution type:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //event - click on a graphic object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
      //click on radion button 1 - order execution type
      if(sparam=="ActP_Exe1_radio2")
      {
         //check the radio button state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         //set the appropriate state
         ObjectSetInteger(0,sparam,OBJPROP_STATE, 1);
         //if it selected
         if(selected)
         {
            //reset the other radio buttons
            ObjectSetInteger(0, "ActP_Exe2_radio2", OBJPROP_STATE, false);
            ObjectSetInteger(0, "ActP_Exe3_radio2", OBJPROP_STATE, false);
            //redraw the chart
            ChartRedraw();
            //finish the execution of function
            return;
         }
         //redraw the chart
         ChartRedraw();
         //finish the execution of function
         return;
      }

       //Similarly for the radio button 2
      if(sparam=="ActP_Exe2_radio2")
      {
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         ObjectSetInteger(0,sparam,OBJPROP_STATE, 1);
         if(selected)
         {
            ObjectSetInteger(0, "ActP_Exe1_radio2", OBJPROP_STATE, false);
            ObjectSetInteger(0, "ActP_Exe3_radio2", OBJPROP_STATE, false);
            ChartRedraw();
            return;
         }
         ChartRedraw();
         return;
      }

       //Similarly for the radio button 3
      if(sparam=="ActP_Exe3_radio2")
      {
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         ObjectSetInteger(0,sparam,OBJPROP_STATE, 1);
         if(selected)
         {
            ObjectSetInteger(0, "ActP_Exe1_radio2", OBJPROP_STATE, false);
            ObjectSetInteger(0, "ActP_Exe2_radio2", OBJPROP_STATE, false);
            ChartRedraw();
            return;
         }
         ChartRedraw();
         return;
      }
   ...
   }
...
}
```

The  order expiration type radio buttons differ only in the fact that when you click on the third one, you must perform an additional step - you need to set a new date in the entry time of the expiration of an order:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on a graphic object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //Click on the 3rd radio button - order expiration date
      if(sparam=="ActP_exp3_radio2")
      {
         //checking it state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         ObjectSetInteger(0,sparam,OBJPROP_STATE, 1);
         //if it selected
         if(selected)
         {
            //reset the remained radio buttons
            ObjectSetInteger(0, "ActP_exp1_radio2", OBJPROP_STATE, false);
            ObjectSetInteger(0, "ActP_exp2_radio2", OBJPROP_STATE, false);
            //set the new date to the date edit field
            ObjectSetInteger(0, "ActP_exp_edit2", OBJPROP_BGCOLOR, White);
            ObjectSetInteger(0, "ActP_exp_edit2", OBJPROP_READONLY, false);
            ObjectSetString(0, "ActP_exp_edit2", OBJPROP_TEXT, TimeToString(time_current));
            //if auxiliary lines are allowed
            //set the new time line
            if(ObjectGetInteger(0, "ActP_DealLines_check2", OBJPROP_STATE)==1)
               ObjectSetInteger(0, "ActP_exp_line2", OBJPROP_TIME, time_current);
            ChartRedraw();
            return;
         }

          //if it isn't selected
         else
         {
            //set the edit field as not available for editing
            ObjectSetInteger(0, "ActP_exp_edit2", OBJPROP_BGCOLOR, LavenderBlush);
            ObjectSetInteger(0, "ActP_exp_edit2", OBJPROP_READONLY, true);
            //remove the auxiliary line
            if(ObjectGetInteger(0, "ActP_DealLines_check2", OBJPROP_STATE)==1)
               ObjectSetInteger(0, "ActP_exp_line2", OBJPROP_TIME, 0);
         }
         ChartRedraw();
         return;
   ...
   }
...
}
```

Now we have finished working with the radio buttons.

**5.4.** **Creating and handling events of dropdown lists**

We'll be using the dropdown list for choosing orders / trades for modification / closing / removal and color selections panel. Let's start with the list of trades / orders.

The first thing that appears on the "Modification / closure" tab is a button labeled "Select an order -->", this will be the button that activates the list. When you click on it, the dropdown list should unfold, and after we make our selection, it should fold up once again.  Let's take a look at the CHARTEVENT\_OBJECT\_CLICK handler of this button:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //event - click on a graphic object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
      //click on the drop-down list activate button (order select)

      if(sparam=="ActP_ord_button5")
      {
         //check status
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         //the list is activated
         if(selected)// the list is selected
         {
            //delete interface
            DeleteScheme("ActP", true);
            //arrays for serving the information about the orders
            string info[100];
            //array for the tickets
            int tickets[100];
            //initialize it
            ArrayInitialize(tickets, -1);
            //get orders info
            get_ord_info(info, tickets);
            //create the list
            create_list(info, tickets);
         }
          //the list isn't active
         else
         {
            //delete it
            DeleteLists("ActP_orders_list_");
         }
          //redraw the chart
         ChartRedraw();
          //finish the function
         return;
      }
   ...
   }
...
}
```

Our primary goal is to determine whether the trades/orders are on the market, and if so, to extract information from them and display it in the list. The get\_ord\_info() function executes this role:

```
//+------------------------------------------------------------------+
//| The function for obtaining the information about orders          |
//+------------------------------------------------------------------+
void get_ord_info(string &info[],int &tickets[])
  {
   //initialize the counter
   int cnt=0;
   string inf;
   //if there is an open position
   if(PositionSelect(Symbol()))
     {
     //combine all order infomation in a single line
      double vol=PositionGetDouble(POSITION_VOLUME);
      int typ=PositionGetInteger(POSITION_TYPE);
      if(typ==POSITION_TYPE_BUY) inf+="BUY ";
      if(typ==POSITION_TYPE_SELL) inf+="SELL ";
      inf+=DoubleToString(vol, MathCeil(MathAbs(MathLog(vol)/MathLog(10))))+" lots";
      inf+=" at "+DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), Digits());
      //write the results
      info[cnt]=inf;
      tickets[cnt]=0;
      //increment the counter
      cnt++;
     }

   //all orders
   for(int i=0;i<OrdersTotal();i++)
     {
      //get ticket
      int ticket=OrderGetTicket(i);
      //if order symbol is equal to chart symbol
      if(OrderGetString(ORDER_SYMBOL)==Symbol())
        {
         //combine all order infomation in a single line
         inf="#"+IntegerToString(ticket)+" ";
         int typ=OrderGetInteger(ORDER_TYPE);
         double vol=OrderGetDouble(ORDER_VOLUME_CURRENT);
         if(typ==ORDER_TYPE_BUY_LIMIT) inf+="BUY LIMIT ";
         if(typ==ORDER_TYPE_SELL_LIMIT) inf+="SELL LIMIT ";
         if(typ==ORDER_TYPE_BUY_STOP) inf+="BUY STOP ";
         if(typ==ORDER_TYPE_SELL_STOP) inf+="SELL STOP ";
         if(typ==ORDER_TYPE_BUY_STOP_LIMIT) inf+="BUY STOP LIMIT ";
         if(typ==ORDER_TYPE_SELL_STOP_LIMIT) inf+="SELL STOP LIMIT ";
         inf+=DoubleToString(vol, MathCeil(MathAbs(MathLog(vol)/MathLog(10))))+" lots";
         inf+=" at "+DoubleToString(OrderGetDouble(ORDER_PRICE_OPEN), Digits());
         //write the results
         info[cnt]=inf;
         tickets[cnt]=ticket;
         //increment the counter
         cnt++;
        }
     }
  }
```

It will combine into a block information and order tickets and trades.

Further, the create\_list() function will create a list on the basis of this information:

```
//+------------------------------------------------------------------+
//| The function creates list of positions                           |
//| info - array for the positions                                   |
//| tickets - array for the tickets                                  |
//+------------------------------------------------------------------+
void create_list(string &info[],int &tickets[])
  {
   //get the coordinates of the list activation button
   int x=ObjectGetInteger(0,"ActP_ord_button5",OBJPROP_XDISTANCE);
   int y=ObjectGetInteger(0, "ActP_ord_button5", OBJPROP_YDISTANCE)+ObjectGetInteger(0, "ActP_ord_button5", OBJPROP_YSIZE);
   //get colors
   color col=ObjectGetInteger(0,"ActP_ord_button5",OBJPROP_COLOR);
   color bgcol=ObjectGetInteger(0,"ActP_ord_button5",OBJPROP_BGCOLOR);
   //get window height
   int wnd_height=ChartGetInteger(0,CHART_HEIGHT_IN_PIXELS,wnd);
   int y_cnt=0;
   //proceed arrays
   for(int i=0;i<100;i++)
     {
      //break if end reached
      if(tickets[i]==-1) break;
      //calculate the list item coordinates
      int y_pos=y+y_cnt*20;
      //if the windiow limits are reachedl, start a new column
      if(y_pos+20>wnd_height) {x+=300; y_cnt=0;}
      y_pos=y+y_cnt*20;
      y_cnt++;
      string name="ActP_orders_list_"+IntegerToString(i)+" $"+IntegerToString(tickets[i]);
      //create element
      create_button(name,info[i],x,y_pos,300,20);
      //and set its properties
      ObjectSetInteger(0,name,OBJPROP_COLOR,col);
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,0);
      ObjectSetInteger(0,name,OBJPROP_STATE,0);
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,8);
      ObjectSetInteger(0,name,OBJPROP_BGCOLOR,bgcol);
     }
  }
```

And finally, the DeleteLists () functions removes the elements of the list:

```
//+------------------------------------------------------------------+
//| The function for the list deletion                               |
//+------------------------------------------------------------------+
void  DeleteLists(string IDstr)
  {
   //proceed all objects
   for(int i=ObjectsTotal(0);i>=0;i--)
     {
      string n=ObjectName(0,i);
      //delete lists
      if(StringFind(n,IDstr)>=0 && StringFind(n,"main")<0) ObjectDelete(0,n);
     }
  }
```

So now when you click on the activation button a list is created.
We need to make it work, since with every click on any element of the list, some specific action must take place. Specifically: the loading of an interface for working with an order, and the filling of this interface with information on the order/trade.

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   // Event - click on a graphic object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //Click not on an item of order selection list
      if(StringFind(sparam, "ActP_orders_list_")<0)
      {
          //Remove it
         DeleteLists("ActP_orders_list_");
          //Set the activation button to "unpressed"
         ObjectSetInteger(0, "ActP_ord_button5", OBJPROP_STATE, 0);
          //redraw chart
         ChartRedraw();
      }
       //Click on the order selection list item
      else
      {
          //Set a new name for the activation button
         ObjectSetString(0, "ActP_ord_button5", OBJPROP_TEXT, ObjectGetString(0, sparam, OBJPROP_TEXT));
          //Set the activation button to "unpressed"
         ObjectSetInteger(0, "ActP_ord_button5", OBJPROP_STATE, 0);
          //get ticket from the list item description
         int ticket=StringToInteger(StringSubstr(sparam, StringFind(sparam, "$")+1));
          //Load the interface
         SetScheme(ticket);
          //and delete the list
         DeleteLists("ActP_orders_list_");
          //chart redraw
         ChartRedraw();
      }
   ...
   }
...
}
```

This is where it gets complicated. Since we do not know in advance the size of the list and the names of its objects, we will have to retrieve information from it by retrieving the name of the element of the list. The SetScheme() function will set up the appropriate interface - for working with a trade, or with a pending order:

```
//+------------------------------------------------------------------+
//| The function sets the interface depending on type:               |
//| position or pending order                                        |
//| t - ticket                                                       |
//+------------------------------------------------------------------+
void SetScheme(int t)
  {
   //if position
   if(t==0)
     {
      //check for its presence
      if(PositionSelect(Symbol()))
        {
         //delete old scheme
         DeleteScheme("ActP",true);
         //and apply new
         ApplyScheme(6);
         //set position parameters
         SetPositionParams();
         //the objects are unavailable for the selection
         Objects_Selectable("ActP",false);
        }
     }
   //if order
   if(t>0)
     {
      //check for its presence
      if(OrderSelect(t))
        {
         //delete old scheme
         DeleteScheme("ActP",true);
         //and apply new
         ApplyScheme(7);
         //set order parameters
         SetOrderParams(t);
         //the objects are unavailable for the selection
         Objects_Selectable("ActP",false);
        }
     }
  }
```

The SetPositionParams() and SetOrderParams() functions install the required properties of the loaded interface:

```
//+------------------------------------------------------------------+
//| Set position parameters for the objects                          |
//+------------------------------------------------------------------+
void SetPositionParams()
  {
   //if position is exists
   if(PositionSelect(Symbol()))
     {
      //get its parameters
      double pr=PositionGetDouble(POSITION_PRICE_OPEN);
      double lots=PositionGetDouble(POSITION_VOLUME);
      double sl=PositionGetDouble(POSITION_SL);
      double tp=PositionGetDouble(POSITION_TP);
      double mag=PositionGetInteger(POSITION_MAGIC);
      //and set new values to the objects
      ObjectSetString(0,"ActP_Pr_edit4",OBJPROP_TEXT,str_del_zero(DoubleToString(pr)));
      ObjectSetString(0,"ActP_lots_edit4",OBJPROP_TEXT,str_del_zero(DoubleToString(lots)));
      ObjectSetString(0,"ActP_SL_edit4",OBJPROP_TEXT,str_del_zero(DoubleToString(sl)));
      ObjectSetString(0,"ActP_TP_edit4",OBJPROP_TEXT,str_del_zero(DoubleToString(tp)));
      if(mag!=0) ObjectSetString(0,"ActP_mag_edit4",OBJPROP_TEXT,IntegerToString(mag));
      //redraw chart
      ChartRedraw();
     }
   //if there isn't position, show message
   else MessageBox("There isn't open position for "+Symbol());
  }
//+------------------------------------------------------------------+
//| Set pending order parameters for the objects                     |
//| ticket - order ticket                                            |
//+------------------------------------------------------------------+
void SetOrderParams(int ticket)
  {
   //if order exists
   if(OrderSelect(ticket) && OrderGetString(ORDER_SYMBOL)==Symbol())
     {
      //get its parameters
      double pr=OrderGetDouble(ORDER_PRICE_OPEN);
      double lots=OrderGetDouble(ORDER_VOLUME_CURRENT);
      double sl=OrderGetDouble(ORDER_SL);
      double tp=OrderGetDouble(ORDER_TP);
      double mag=OrderGetInteger(ORDER_MAGIC);
      double lim=OrderGetDouble(ORDER_PRICE_STOPLIMIT);
      datetime expir=OrderGetInteger(ORDER_TIME_EXPIRATION);
      ENUM_ORDER_TYPE type=OrderGetInteger(ORDER_TYPE);
      ENUM_ORDER_TYPE_TIME expir_type=OrderGetInteger(ORDER_TYPE_TIME);

      //of order type is stoplimit, modify the interface
      if(type==ORDER_TYPE_BUY_STOP_LIMIT || type==ORDER_TYPE_SELL_STOP_LIMIT)
        {
         //set new value to the order price edit
         ObjectSetString(0,"ActP_limpr_edit3",OBJPROP_TEXT,DoubleToString(lim,_Digits));
         ObjectSetInteger(0,"ActP_limpr_edit3",OBJPROP_BGCOLOR,White);
         //set order price available for edit
         ObjectSetInteger(0,"ActP_limpr_edit3",OBJPROP_READONLY,false);
        }
      //if order type isn't stoplimit, modify the interface
      else
        {
         ObjectSetString(0,"ActP_limpr_edit3",OBJPROP_TEXT,"");
         ObjectSetInteger(0,"ActP_limpr_edit3",OBJPROP_BGCOLOR,LavenderBlush);
         ObjectSetInteger(0,"ActP_limpr_edit3",OBJPROP_READONLY,true);
        }

      //check expiration type
      //and set interface elements
      switch(expir_type)
        {
         case ORDER_TIME_GTC:
           {
            ObjectSetInteger(0,"ActP_exp1_radio3",OBJPROP_STATE,1);
            ObjectSetInteger(0,"ActP_exp2_radio3",OBJPROP_STATE,0);
            ObjectSetInteger(0,"ActP_exp3_radio3",OBJPROP_STATE,0);
            break;
           }
         case ORDER_TIME_DAY:
           {
            ObjectSetInteger(0,"ActP_exp1_radio3",OBJPROP_STATE,0);
            ObjectSetInteger(0,"ActP_exp2_radio3",OBJPROP_STATE,1);
            ObjectSetInteger(0,"ActP_exp3_radio3",OBJPROP_STATE,0);
            break;
           }
         case ORDER_TIME_SPECIFIED:
           {
            ObjectSetInteger(0,"ActP_exp1_radio3",OBJPROP_STATE,0);
            ObjectSetInteger(0,"ActP_exp2_radio3",OBJPROP_STATE,0);
            ObjectSetInteger(0,"ActP_exp3_radio3",OBJPROP_STATE,1);
            //in addition, set new value to the edit
            ObjectSetString(0,"ActP_exp_edit3",OBJPROP_TEXT,TimeToString(expir));
            break;
           }
        }
      //set new values for the objects
      ObjectSetString(0,"ActP_Pr_edit3",OBJPROP_TEXT,str_del_zero(DoubleToString(pr)));
      ObjectSetString(0,"ActP_lots_edit3",OBJPROP_TEXT,str_del_zero(DoubleToString(lots)));
      ObjectSetString(0,"ActP_SL_edit3",OBJPROP_TEXT,str_del_zero(DoubleToString(sl)));
      ObjectSetString(0,"ActP_TP_edit3",OBJPROP_TEXT,str_del_zero(DoubleToString(tp)));
      ObjectSetString(0,"ActP_ticket_edit3",OBJPROP_TEXT,IntegerToString(ticket));
      if(mag!=0) ObjectSetString(0,"ActP_mag_edit3",OBJPROP_TEXT,IntegerToString(mag));
      ChartRedraw();
     }
   //if there isn't such order, show message
   else MessageBox("There isn't an order with ticket "+IntegerToString(ticket)+" for "+Symbol());
  }
```

And the final touch - the list should be removed when you click on the chart, using CHARTEVENT\_CLICK for this event:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - is click on the chart
   if(id==CHARTEVENT_CLICK)
   {
       //delete all lists
      DeleteLists("ActP_orders_list_");
      DeleteLists("ActP_color_list_");
       //Set the activate buttons to the unpressed state
      ObjectSetInteger(0, "ActP_ord_button5", OBJPROP_STATE, 0);
      ObjectSetInteger(0, "ActP_col1_button6", OBJPROP_STATE, 0);
      ObjectSetInteger(0, "ActP_col2_button6", OBJPROP_STATE, 0);
      ObjectSetInteger(0, "ActP_col3_button6", OBJPROP_STATE, 0);
      ChartRedraw();
      return;
   }
...
}
```

As a result, we have a nice drop-down list:

![Figure 14. An example of the drop-down list panel "Modify/Close"](https://c.mql5.com/2/1/figure14.png)

Figure
14. An
example of the drop-down list panel "Modify/Close"

Now we need to create a list of color selection on the Settings tab.

Consider the handlers of the activation buttons:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on the chart
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //Click on the button to activate the colors drop-down list
      if(sparam=="ActP_col1_button6")
      {
          //check state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
          //the list is active
         if(selected)//the list is active
         {
             //creat list
            create_color_list(100, "ActP_col1_button6", 1);
             //Set the position of the remaining buttons to "unpressed"
            ObjectSetInteger(0, "ActP_col2_button6", OBJPROP_STATE, 0);
            ObjectSetInteger(0, "ActP_col3_button6", OBJPROP_STATE, 0);
             //delete other lists
            DeleteLists("ActP_color_list_2");
            DeleteLists("ActP_color_list_3");
         }
          //the list isn't selected
         else
         {
             //delete it
            DeleteLists("ActP_color_list_");
         }
          //redraw chart
         ChartRedraw();
          //finish the execution of function
         return;
      }
   ...
   }
...
}
```

Here we follow the same method as with the order selection list.

The function of creating a list differs:

```
//+------------------------------------------------------------------+
//| Function for creating the colors list                            |
//| y_max - maximal list widthа                                      |
//| ID - list ID                                                     |
//| num - interface number                                           |
//+------------------------------------------------------------------+
void create_color_list(int y_max,string ID,int num)
  {
  //Get the coordinates of the list activation button
   int x=ObjectGetInteger(0,ID,OBJPROP_XDISTANCE);
   int y=ObjectGetInteger(0, ID, OBJPROP_YDISTANCE)+ObjectGetInteger(0, ID, OBJPROP_YSIZE);
   //get color
   color col=ObjectGetInteger(0,ID,OBJPROP_COLOR);
   //and window width
   int wnd_height=ChartGetInteger(0,CHART_HEIGHT_IN_PIXELS,wnd);
   y_max+=y;
   int y_cnt=0;
   //We will go through the colors array
   for(int i=0;i<132;i++)
     {
      color bgcol=colors[i];
      //calculate list item coordinates
      int y_pos=y+y_cnt*20;
      //if we reached the boundaries of the window, start new column
      if(y_pos+20>wnd_height || y_pos+20>y_max) {x+=20; y_cnt=0;}
      y_pos=y+y_cnt*20;
      y_cnt++;
      //create new element
      string name="ActP_color_list_"+IntegerToString(num)+ID+IntegerToString(i);
      create_button(name,"",x,y_pos,20,20);
      //and set its properties
      ObjectSetInteger(0,name,OBJPROP_COLOR,col);
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,0);
      ObjectSetInteger(0,name,OBJPROP_STATE,0);
      ObjectSetInteger(0,name,OBJPROP_BGCOLOR,bgcol);
     }
  }
```

Further let's work out the click process for the list element:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on chart
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //click isn't on the color list button
      if(StringFind(sparam, "ActP_color_list_1")<0)
      {
          //delete list
         DeleteLists("ActP_color_list_1");
          //set color list activation button to "unpressed"
         ObjectSetInteger(0, "ActP_col1_button6", OBJPROP_STATE, 0);
          //redraw chart
         ChartRedraw();
      }
       //click on the color list
      else
      {
          //get color from the list
         color col=ObjectGetInteger(0, sparam, OBJPROP_BGCOLOR);
          //set it for all the buttons
         SetButtonsColor(col);
          //set button to unpressed
         ObjectSetInteger(0, "ActP_col1_button6", OBJPROP_STATE, 0);
          //delete list
         DeleteLists("ActP_color_list_1");
          //redraw chart
         ChartRedraw();
      }
   ...
   }
...
}
```

The SetButtonsColor() function sets the color of buttons:

```
//+------------------------------------------------------------------+
//| The function sets color for all buttons                          |
//| col - color                                                      |
//+------------------------------------------------------------------+
void SetButtonsColor(color col)
  {
   //We will go through all the objects
   for(int i=ObjectsTotal(0);i>=0;i--)
     {
      string n=ObjectName(0,i);
      //If the object belongs to the panel and its has a button type
      //set color
      if(StringFind(n,"ActP")>=0 && ObjectGetInteger(0,n,OBJPROP_TYPE)==OBJ_BUTTON)
         ObjectSetInteger(0,n,OBJPROP_BGCOLOR,col);
     }
   //set global variable
   GlobalVariableSet("ActP_buttons_color",col);
  }
```

Let's view the results below:

![Figure 15. Setting the colors of buttons](https://c.mql5.com/2/1/figure15.png)

Figure
15\. Setting the
colors of buttons

The lists of color selection and text labels are similar. As a result, we can make the panel nicely colorful in just a few clicks:

![Figure 16. Changed colors of panels, buttons, and text](https://c.mql5.com/2/1/figure16_.png)

Figure 16. Changed colors of panels, buttons, and text

We are now finished with lists. Let's move on to input fields.

**5.5. Handling the Entry Field** **Event**

An entry field will generate an event CHARTEVENT\_OBJECT\_ENDEDIT, which occurs at the completion of editing the text in the field. The only reason we need to handle this event is due to the setting of auxiliary lines for prices, relevant to the prices in the entry fields.

Let us consider the example of a stop line:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //End edit event
   if(id==CHARTEVENT_OBJECT_ENDEDIT)//end edit event
   {
   ...
      //if edit field is SL field
      if(sparam=="ActP_SL_edit1")
      {
        //and auxiliary lines are enabled
         if(ObjectGetInteger(0,"ActP_DealLines_check1",OBJPROP_STATE)==1)
         {
            //get text from the field
            double sl_val=StringToDouble(ObjectGetString(0, "ActP_SL_edit1", OBJPROP_TEXT));
            //move lines at new position
            ObjectSetDouble(0, "ActP_SL_line1", OBJPROP_PRICE, sl_val);
         }
         //redraw chart
         ChartRedraw();
         //it ins't necessary to proceed the other objects, because the event from the one
         return;
      }
   ...
   }
...
}
```

Other entry fields are processed similarly.

**5.6 Handling Timer Events**

The timer is used to monitor the auxiliary lines. This way, when you move the lines, the values of prices to which they are linked, automatically move to the input field. With each tick of the timer, the OnTimer() function is executed.

Consider the example of the placing of Stop Loss and Take Profit lines tracking with the active "Market" tab:

```
void OnTimer()// Timer handler
{
   //panel 1 is active
   if(ObjectGetInteger(0, "ActP_main_1", OBJPROP_STATE)==1)
   {
      //if auxiliary lines are allowed
      if(ObjectGetInteger(0,"ActP_DealLines_check1",OBJPROP_STATE)==1)
      {
         //set new values to the edit fields
         double sl_pr=NormalizeDouble(ObjectGetDouble(0, "ActP_SL_line1", OBJPROP_PRICE), _Digits);
         //stop loss
         ObjectSetString(0, "ActP_SL_edit1", OBJPROP_TEXT, DoubleToString(sl_pr, _Digits));
         //take profit
         double tp_pr=NormalizeDouble(ObjectGetDouble(0, "ActP_TP_line1", OBJPROP_PRICE), _Digits);
         ObjectSetString(0, "ActP_TP_edit1", OBJPROP_TEXT, DoubleToString(tp_pr, _Digits));
      }
   }
   ...
   //redraw chart
   ChartRedraw();
}
//+------------------------------------------------------------------+
```

Tracking other lines is implemented similarly.

### 6\. Performing Trade Operations

So at this point we have filled in all the required entry fields, check boxes, lines, and radio buttons. Now is the time to try some trading based on all of the data we have.

**6.1.** **Opening Deal**

The "From the market" tab contains buttons "Buy" and "Sell". If all fields are filled correctly, a trade should be implemented when we click on either of the buttons.

Let's look at the handlers of these buttons:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on the object on the chart
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //click on the Buy button
      if(sparam=="ActP_buy_button1")
      {
          //check its state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
          //if it "pressed"
         if(selected)
         {
             //try to perform a deal
            deal(ORDER_TYPE_BUY);
             //and set the button to the unpressed state
            ObjectSetInteger(0, sparam, OBJPROP_STATE, 0);
         }

          //redraw chart
         ChartRedraw();
          //and finish the function execution
         return;
      }
      //******************************************
       //the similar for the sell button
      if(sparam=="ActP_sell_button1")
      {
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         if(selected)
         {
            deal(ORDER_TYPE_SELL);
            ObjectSetInteger(0, sparam, OBJPROP_STATE, 0);
         }
         ChartRedraw();
         return;
      }
   ...
   }
...
}
```

You see, the deal() function is working.

```
//+------------------------------------------------------------------+
//| Deal function                                                    |
//+------------------------------------------------------------------+
int deal(ENUM_ORDER_TYPE typ)
  {
   //get the data from the objects
   double SL=StringToDouble(ObjectGetString(0,"ActP_SL_edit1",OBJPROP_TEXT));
   double TP=StringToDouble(ObjectGetString(0, "ActP_TP_edit1", OBJPROP_TEXT));
   double lots=StringToDouble(ObjectGetString(0,"ActP_Lots_edit1",OBJPROP_TEXT));
   int mag=StringToInteger(ObjectGetString(0, "ActP_Magic_edit1", OBJPROP_TEXT));
   int dev=StringToInteger(ObjectGetString(0, "ActP_Dev_edit1", OBJPROP_TEXT));
   string comm=ObjectGetString(0,"ActP_Comm_edit1",OBJPROP_TEXT);
   ENUM_ORDER_TYPE_FILLING filling=ORDER_FILLING_FOK;
   if(ObjectGetInteger(0,"ActP_Exe2_radio1",OBJPROP_STATE)==1) filling=ORDER_FILLING_IOC;
   //prepare request
   MqlTradeRequest req;
   MqlTradeResult res;
   req.action=TRADE_ACTION_DEAL;
   req.symbol=Symbol();
   req.volume=lots;
   req.price=Ask;
   req.sl=NormalizeDouble(SL, Digits());
   req.tp=NormalizeDouble(TP, Digits());
   req.deviation=dev;
   req.type=typ;
   req.type_filling=filling;
   req.magic=mag;
   req.comment=comm;
   //send order
   OrderSend(req,res);
   //show message with the result
   MessageBox(RetcodeDescription(res.retcode),"Message");
   //return retcode
   return(res.retcode);
  }
```

Nothing supernatural. We first read required information from the objects and, on their basis, create a trade request.

Let's check our work:

![Figure 17. Trading operations - the result of Buy trade execution](https://c.mql5.com/2/1/figure17_.png)

Figure 17. Trading operations - the result of Buy trade execution

As you can see, the Buy trade is successfully acomplished.

**6.2.** **Setting a Pending Order**

The "Buy" and "Sell" buttons on the "Pending" tab are responsible for the placing of pending orders.

Let's consider the handlers:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on the chart object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //click on the pending order set button
      if(sparam=="ActP_buy_button2")
      {
         //check the button state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         //if it pressed
         if(selected)
         {
            ENUM_ORDER_TYPE typ;
            //get the pending order from the edit
            double pr=NormalizeDouble(StringToDouble(ObjectGetString(0, "ActP_Pr_edit2", OBJPROP_TEXT)), Digits());
            //if it isn't stoplimit order
            if(ObjectGetInteger(0, "ActP_limit_check2", OBJPROP_STATE)==0)
            {
               //if the order price is below the current price, set limit order
               if(Ask>pr) typ=ORDER_TYPE_BUY_LIMIT;
               //overwise - stop order
               else typ=ORDER_TYPE_BUY_STOP;
            }
              //if stoplimit order is specified
            else
            {
               //set operation type
               typ=ORDER_TYPE_BUY_STOP_LIMIT;
            }
              //try to place order
            order(typ);
              //set button to the unpressed state
            ObjectSetInteger(0, sparam, OBJPROP_STATE, 0);
         }
          //redraw chart
         ChartRedraw();
          //and finish the execution of function
         return;
      }
      //******************************************

       //similar for the sell pending order
      if(sparam=="ActP_sell_button2")
      {
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         if(selected)
         {
            ENUM_ORDER_TYPE typ;
            double pr=NormalizeDouble(StringToDouble(ObjectGetString(0, "ActP_Pr_edit2", OBJPROP_TEXT)), Digits());
            if(ObjectGetInteger(0, "ActP_limit_check2", OBJPROP_STATE)==0)
            {
               if(Bid<pr) typ=ORDER_TYPE_SELL_LIMIT;
               else typ=ORDER_TYPE_SELL_STOP;
            }
            else
            {
               typ=ORDER_TYPE_SELL_STOP_LIMIT;
            }
            order(typ);
            ObjectSetInteger(0, sparam, OBJPROP_STATE, 0);
         }
         ChartRedraw();
         return;
      }
   ...
   }
...
}
```

Here we determine the type of future orders on the basis of the relation of the current market price to the set price, after which the order() function determines the order:

```
//+------------------------------------------------------------------+
//| The function places an order                                     |
//+------------------------------------------------------------------+
int order(ENUM_ORDER_TYPE typ)
  {
   //get the order details from the objects
   double pr=StringToDouble(ObjectGetString(0,"ActP_Pr_edit2",OBJPROP_TEXT));
   double stoplim=StringToDouble(ObjectGetString(0,"ActP_limpr_edit2",OBJPROP_TEXT));
   double SL=StringToDouble(ObjectGetString(0, "ActP_SL_edit2", OBJPROP_TEXT));
   double TP=StringToDouble(ObjectGetString(0, "ActP_TP_edit2", OBJPROP_TEXT));
   double lots=StringToDouble(ObjectGetString(0,"ActP_Lots_edit2",OBJPROP_TEXT));
   datetime expir=StringToTime(ObjectGetString(0,"ActP_exp_edit2",OBJPROP_TEXT));
   int mag=StringToInteger(ObjectGetString(0,"ActP_Magic_edit2",OBJPROP_TEXT));
   string comm=ObjectGetString(0,"ActP_Comm_edit2",OBJPROP_TEXT);
   ENUM_ORDER_TYPE_FILLING filling=ORDER_FILLING_FOK;
   if(ObjectGetInteger(0, "ActP_Exe2_radio2", OBJPROP_STATE)==1) filling=ORDER_FILLING_IOC;
   if(ObjectGetInteger(0, "ActP_Exe3_radio2", OBJPROP_STATE)==1) filling=ORDER_FILLING_RETURN;
   ENUM_ORDER_TYPE_TIME expir_type=ORDER_TIME_GTC;
   if(ObjectGetInteger(0, "ActP_exp2_radio2", OBJPROP_STATE)==1) expir_type=ORDER_TIME_DAY;
   if(ObjectGetInteger(0, "ActP_exp3_radio2", OBJPROP_STATE)==1) expir_type=ORDER_TIME_SPECIFIED;

   //prepare request
   MqlTradeRequest req;
   MqlTradeResult res;
   req.action=TRADE_ACTION_PENDING;
   req.symbol=Symbol();
   req.volume=lots;
   req.price=NormalizeDouble(pr,Digits());
   req.stoplimit=NormalizeDouble(stoplim,Digits());
   req.sl=NormalizeDouble(SL, Digits());
   req.tp=NormalizeDouble(TP, Digits());
   req.type=typ;
   req.type_filling=filling;
   req.type_time=expir_type;
   req.expiration=expir;
   req.comment=comm;
   req.magic=mag;
   //place order
   OrderSend(req,res);
   //show message with the result
   MessageBox(RetcodeDescription(res.retcode),"Message");
   //return retcode
   return(res.retcode);
  }
```

Let's check our work:

![Figure 18. Trading operations - the result pending order placing](https://c.mql5.com/2/1/figure18_.png)

Figure 18. Trading operations - the
result pending order placing

Buy stoplimit is set successfully.

**6.3. Position** **Modification**

The Edit button on the "Modify/Close" tab is responsible for the modification of the selected position:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on the graphic object on the chart
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //click on the modify position button
      if(sparam=="ActP_mod_button4")
      {
          //check the button state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
          //if it pressed
         if(selected)//if pressed
         {
            //modify position
            modify_pos();
            //delete the elements of the scheme
            DeleteScheme("ActP" ,true);
            //and reset it (update the interface)
            SetScheme(0);
            //set the button to the unpressed state
            ObjectSetInteger(0, sparam, OBJPROP_STATE, 0);
         }
          //redraw chart
         ChartRedraw();
          //finish the execution of function
         return;
      }
   ...
   }
...
}
```

The Modify\_pos() function is directly responsible for the modification:

```
//+------------------------------------------------------------------+
//| The function modifies the position parameters                    |
//+------------------------------------------------------------------+
int modify_pos()
  {
   if(!PositionSelect(Symbol())) MessageBox("There isn't open position for symbol "+Symbol(),"Message");
   //get the details from the edit objects
   double SL=StringToDouble(ObjectGetString(0,"ActP_SL_edit4",OBJPROP_TEXT));
   double TP=StringToDouble(ObjectGetString(0, "ActP_TP_edit4", OBJPROP_TEXT));
   int dev=StringToInteger(ObjectGetString(0,"ActP_dev_edit4",OBJPROP_TEXT));
   //prepare request
   MqlTradeRequest req;
   MqlTradeResult res;
   req.action=TRADE_ACTION_SLTP;
   req.symbol=Symbol();
   req.sl=NormalizeDouble(SL, _Digits);
   req.tp=NormalizeDouble(TP, _Digits);
   req.deviation=dev;
   //send request
   OrderSend(req,res);
   //show message with the result
   MessageBox(RetcodeDescription(res.retcode),"Message");
   //return retcode
   return(res.retcode);
  }
```

Results:

![Figure 19. Trading operations - the result of modifying the properties of the trade (set TP and SL)](https://c.mql5.com/2/1/figure19_.png)

Figure
19\. Trading
operations - the result of modifying the properties of the trade (TP
and SL)

Stop Loss and Take Profit levels are changed successfully.

**6.4.** **Closing Position**

The Close button on the tab "Modify/Close" is responsible for the closure (possibly partial) of the position:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on the chart object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
      //click on the close button
      if(sparam=="ActP_del_button4")
      {
         //check the button state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
          //if pressed
         if(selected)
         {
            //try to close position
            int retcode=close_pos();
            //if successful
            if(retcode==10009)
            {
               //delete scheme elements
               DeleteScheme("ActP" ,true);
               //set the new text for the list activisation
               ObjectSetString(0, "ActP_ord_button5", OBJPROP_TEXT, "Select order -->");
            }
            //set button state to unpressed
            ObjectSetInteger(0, sparam, OBJPROP_STATE, 0);
         }
          //redraw chart
         ChartRedraw();
          //finish the execution of function
         return;
      }
   ...
   }
...
}
```

The close\_pos() function is responsible for the closure:

```
//+------------------------------------------------------------------+
//| Closes the position                                              |
//+------------------------------------------------------------------+
int close_pos()
  {
   if(!PositionSelect(Symbol())) MessageBox("There isn't open position for symbol "+Symbol(),"Message");
   //get the position details from the objects
   double lots=StringToDouble(ObjectGetString(0,"ActP_lots_edit4",OBJPROP_TEXT));
   if(lots>PositionGetDouble(POSITION_VOLUME)) lots=PositionGetDouble(POSITION_VOLUME);
   int dev=StringToInteger(ObjectGetString(0, "ActP_dev_edit4", OBJPROP_TEXT));
   int mag=StringToInteger(ObjectGetString(0, "ActP_mag_edit4", OBJPROP_TEXT));

   //prepare request
   MqlTradeRequest req;
   MqlTradeResult res;

   //the opposite deal is dependent on position type
   if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
     {
      req.price=Bid;
      req.type=ORDER_TYPE_SELL;
     }
   else
     {
      req.price=Ask;
      req.type=ORDER_TYPE_BUY;
     }

   req.action=TRADE_ACTION_DEAL;
   req.symbol=Symbol();
   req.volume=lots;
   req.sl=0;
   req.tp=0;
   req.deviation=dev;
   req.type_filling=ORDER_FILLING_FOK;
   req.magic=mag;
   //send request
   OrderSend(req,res);
   //show message with the result
   MessageBox(RetcodeDescription(res.retcode),"Message");
   //return retcode
   return(res.retcode);
  }
```

The result - closed 1.5 lots of three of the selected transaction:

![Figure 20. Trading - partial position closure](https://c.mql5.com/2/1/figure20.png)

Figure
20\. Trading -
partial position closure

**6.5.** **Modification of a Pending Order**

The Edit button on the "Modification/closure" tab is responsible for modifying the selected order:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on the chart graphic object
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
      //click on the order modify button
      if(sparam=="ActP_mod_button3")
      {
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
         if(selected)
         {
            //get the order ticket from the edit
            string button_name=ObjectGetString(0, "ActP_ord_button5", OBJPROP_TEXT);
            long ticket=StringToInteger(StringSubstr(button_name, 1, StringFind(button_name, " ")-1));
            //modifying an order
            modify_order(ticket);
            //update interface
            DeleteScheme("ActP" ,true);
            SetScheme(ticket);
            //set button to unpressed state
            ObjectSetInteger(0, sparam, OBJPROP_STATE, 0);
         }
          //redraw chart
         ChartRedraw();
          //and finish the execution of function
         return;
      }
   ...
   }
...
}
```

The Modify\_order () function is responsible for the modification:

```
//+------------------------------------------------------------------+
//| The function modifies an order                                   |
//| ticket - order ticket                                            |
//+------------------------------------------------------------------+
int modify_order(int ticket)
  {
   //get the order details from the corresponding chart objects
   double pr=StringToDouble(ObjectGetString(0,"ActP_Pr_edit3",OBJPROP_TEXT));
   double stoplim=StringToDouble(ObjectGetString(0,"ActP_limpr_edit3",OBJPROP_TEXT));
   double SL=StringToDouble(ObjectGetString(0, "ActP_SL_edit3", OBJPROP_TEXT));
   double TP=StringToDouble(ObjectGetString(0, "ActP_TP_edit3", OBJPROP_TEXT));
   double lots=StringToDouble(ObjectGetString(0,"ActP_Lots_edit3",OBJPROP_TEXT));
   datetime expir=StringToTime(ObjectGetString(0,"ActP_exp_edit3",OBJPROP_TEXT));
   ENUM_ORDER_TYPE_TIME expir_type=ORDER_TIME_GTC;
   if(ObjectGetInteger(0, "ActP_exp2_radio3", OBJPROP_STATE)==1) expir_type=ORDER_TIME_DAY;
   if(ObjectGetInteger(0, "ActP_exp3_radio3", OBJPROP_STATE)==1) expir_type=ORDER_TIME_SPECIFIED;

   //prepare request to modify
   MqlTradeRequest req;
   MqlTradeResult res;
   req.action=TRADE_ACTION_MODIFY;
   req.order=ticket;
   req.volume=lots;
   req.price=NormalizeDouble(pr,Digits());
   req.stoplimit=NormalizeDouble(stoplim,Digits());
   req.sl=NormalizeDouble(SL, Digits());
   req.tp=NormalizeDouble(TP, Digits());
   req.type_time=expir_type;
   req.expiration=expir;
   //send request
   OrderSend(req,res);
   //show message with the result
   MessageBox(RetcodeDescription(res.retcode),"Message");
   //return retcode
   return(res.retcode);
  }
```

Let's see the result - an order is modified successfully:

![Figure 21. Modification of pending order](https://c.mql5.com/2/1/figure21.png)

Figure 21\. Modification of pending order

**6.6.** **Deleting a Pending Order**

The Delete button on the "Modification/Closure" tab is responsible for deleting the selected order:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
...
   //Event - click on the graphic object on the chart
   if(id==CHARTEVENT_OBJECT_CLICK)
   {
   ...
       //click on the order delete button
      if(sparam=="ActP_del_button3")
      {
         //check the button state
         bool selected=ObjectGetInteger(0,sparam,OBJPROP_STATE);
          //if pressed
         if(selected)
         {
            //get the ticket from the list
            string button_name=ObjectGetString(0, "ActP_ord_button5", OBJPROP_TEXT);
            long ticket=StringToInteger(StringSubstr(button_name, 1, StringFind(button_name, " ")-1));
            //try to delete order
            int retcode=del_order(ticket);
            //if successful
            if(retcode==10009)
            {
               //delete all objects of the scheme
               DeleteScheme("ActP" ,true);
               //set new text for the list activation button
               ObjectSetString(0, "ActP_ord_button5", OBJPROP_TEXT, "Select an order -->");
            }
             //set button state to unpressed
            ObjectSetInteger(0, sparam, OBJPROP_STATE, 0);
         }
          //redraw chart
         ChartRedraw();
          //and finish the execution of function
         return;
      }
   ...
   }
...
}
```

The del\_order() function is responsible for the removal of orders:

```
//+------------------------------------------------------------------+
//| The function for pending order deletion                          |
//| ticket - order ticket                                            |
//+------------------------------------------------------------------+
int del_order(int ticket)
  {
   //prepare request for deletion
   MqlTradeRequest req;
   MqlTradeResult res;
   req.action=TRADE_ACTION_REMOVE;
   req.order=ticket;
   //send request
   OrderSend(req,res);
   //show message with the result
   MessageBox(RetcodeDescription(res.retcode),"Message");
   //return retcode
   return(res.retcode);
  }
```

Let's see the result - the order is removed.

![Figure 22. Trading - removal of a pending order](https://c.mql5.com/2/1/figure22.png)

Fig. 22 Trading operations - removal of a pending order

### Conclusion

Finally all of the functions of the panel have been tested and are working successfully.

Hopefully the knowledge gained from reading this article will help you with the development of active control panels, which will serve you as irreplaceable aids for working on the market.

To get started with the panel you need to unzip the archive into a folder with the terminal, then apply the **AP** indicator to the chart, and **only then** launch the **Active Panel** Expert Advisor.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/62](https://www.mql5.com/ru/articles/62)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/62.zip "Download all attachments in the single ZIP archive")

[active\_panel\_mql5\_en.zip](https://www.mql5.com/en/articles/download/62/active_panel_mql5_en.zip "Download active_panel_mql5_en.zip")(20.11 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [A Library for Constructing a Chart via Google Chart API](https://www.mql5.com/en/articles/114)
- [Creating Information Boards Using Standard Library Classes and Google Chart API](https://www.mql5.com/en/articles/102)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1055)**
(11)


![Vitaliy Kostrubko](https://c.mql5.com/avatar/2016/8/579E94F7-83FB.png)

**[Vitaliy Kostrubko](https://www.mql5.com/en/users/bbk30)**
\|
16 Nov 2024 at 23:22

... as for the creation of "alternative Panels in the indicator window = so in general the topic is TOP (!) :)

... but for the classical approach --> when the Panel is made for the MAIN window of the Graphics - please tell me how to rewrite the code ?!

\-\-\--------------

in short, Dear Experts _(or in case the Author @space\_cowboy sees it)_ -->\> please Rewrite half an article for "dummies" -->> how to write your Panel in the Main Graphics window (?!) :)))

\-\-\--------------

Thank you \_/\\\_ :)

![Vitaliy Kostrubko](https://c.mql5.com/avatar/2016/8/579E94F7-83FB.png)

**[Vitaliy Kostrubko](https://www.mql5.com/en/users/bbk30)**
\|
17 Nov 2024 at 01:05

... and a question to the Author :

why does the Script saves [object properties](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property "MQL5 Documentation: Object Properties") not in a .CSV file, but in a .BIN (???!!).

CSV is easier to work with (!).

![lynxntech](https://c.mql5.com/avatar/2022/7/62CF9DBF-A3CD.png)

**[lynxntech](https://www.mql5.com/en/users/lynxntech)**
\|
17 Nov 2024 at 02:34

**Vitaliy Kostrubko object properties not in .CSV file but in .BIN (???!!!).**
**CSV is easier to work with (!).**

it is necessary to make it a rule to look at the dates of the last posts, there 2014 year

![streamtear](https://c.mql5.com/avatar/avatar_na2.png)

**[streamtear](https://www.mql5.com/en/users/streamtear)**
\|
26 Feb 2025 at 06:36

Can you please tell me how to make the click panel effective in the historical [backtesting](https://www.mql5.com/en/articles/2612 "Article: Testing trading strategies on real ticks ") of this EA.


![Эдуард Шубников](https://c.mql5.com/avatar/2023/6/649B05B3-F355.png)

**[Эдуард Шубников](https://www.mql5.com/en/users/shubnikov)**
\|
31 May 2025 at 11:33

Can you tell me where to find the radio button in the terminal?


![MQL for "Dummies": How to Design and Construct Object Classes](https://c.mql5.com/2/0/cube__1.png)[MQL for "Dummies": How to Design and Construct Object Classes](https://www.mql5.com/en/articles/53)

By creating a sample program of visual design, we demonstrate how to design and construct classes in MQL5. The article is written for beginner programmers, who are working on MT5 applications. We propose a simple and easy grasping technology for creating classes, without the need to deeply immerse into the theory of object-oriented programming.

![Practical Application Of Databases For Markets Analysis](https://c.mql5.com/2/0/dar.png)[Practical Application Of Databases For Markets Analysis](https://www.mql5.com/en/articles/69)

Working with data has become the main task for modern software - both for standalone and network applications. To solve this problem a specialized software were created. These are Database Management Systems (DBMS), that can structure, systematize and organize data for their computer storage and processing. As for trading, the most of analysts don't use databases in their work. But there are tasks, where such a solution would have to be handy. This article provides an example of indicators, that can save and load data from databases both with client-server and file-server architectures.

![OOP in MQL5 by Example: Processing Warning and Error Codes](https://c.mql5.com/2/0/mistake.png)[OOP in MQL5 by Example: Processing Warning and Error Codes](https://www.mql5.com/en/articles/70)

The article describes an example of creating a class for working with the trade server return codes and all the errors that occur during the MQL-program run. Read the article, and you will learn how to work with classes and objects in MQL5. At the same time, this is a convenient tool for handling errors; and you can further change this tool according to your specific needs.

![Migrating from MQL4 to MQL5](https://c.mql5.com/2/0/logo__4.png)[Migrating from MQL4 to MQL5](https://www.mql5.com/en/articles/81)

This article is a quick guide to MQL4 language functions, it will help you to migrate your programs from MQL4 to MQL5. For each MQL4 function (except trading functions) the description and MQL5 implementation are presented, it allows you to reduce the conversion time significantly. For convenience, the MQL4 functions are divided into groups, similar to MQL4 Reference.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/62&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068330330416412719)

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