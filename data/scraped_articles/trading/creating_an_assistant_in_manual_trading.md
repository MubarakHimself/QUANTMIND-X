---
title: Creating an assistant in manual trading
url: https://www.mql5.com/en/articles/2281
categories: Trading, Expert Advisors
relevance_score: -2
scraped_at: 2026-01-24T14:16:19.104332
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=jnpugrvsbvrefemzoqgzbnvtifnobtfi&ssn=1769253377648480430&ssn_dr=0&ssn_sr=0&fv_date=1769253377&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2281&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20an%20assistant%20in%20manual%20trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925337762414522&fz_uniq=5083453116424133633&sv=2552)

MetaTrader 5 / Examples


### Introduction

In this article, I provide another example of creating a fully operational trading panel from scratch in order to assist those who trade Forex manually.

### 1\. Identify functionality of a trading panel

First, we need to set for ourselves the final results that we wish to achieve. We will have to decide what functionality we expect from our panel, and what design will be the most convenient for us. My vision of a trading panel is shared with you here, but I am also happy to take your suggestions on board and, hopefully, cover them in my new articles.

So, our panel must certainly include the following elements.

1. Buy and Sell buttons.
2. Buttons to close all positions by symbol or account or in different directions (Buy/Sell orders).
3. Option to display Stop Loss and Take Profit levels in points and currency of deposit (when entering one parameter, the other parameter should be automatically corrected).
4. Automatically calculates Stop Loss and Take Profit levels using manually set parameters (p.2), and displays them on chart.
5. Enables traders to move Stop Loss and/or Take Profit on the chart. All changes should be displayed on the panel with a change of relevant values.
6. Calculates an expected trading volume by set risk parameters (in currency of deposit or in percentage from the current balance).
7. Allows traders to set a trade volume himself/herself. All relevant parameters that depend on him/her must be automatically re-calculated at the same time.
8. Memorizes which parameters are entered by a trader, and which are automatically calculated. This is important, so parameters entered by a trader remain the same during further re-calculations.
9. Stores all entered parameters in order to avoid repeatedly entering them after rebooting.

### 2\. Create a graphical presentation of the panel

Let's take a new sheet and draw our future trading panel by placing all necessary elements on it.

When developing a design for the trading panel, it should be taken into consideration how practical is the implementation. First, the trade panel should contain sufficient information, be easily readable and not overloaded with extra elements. We should always remember that it is not just a nice image on the screen, but an essential tool for a trader.

Here is my variation.

![Design](https://c.mql5.com/2/23/TradePanel__1.png)

### 3\. Build a panel model in MQL5

### 3.1.  Template

Now that we have set the target, it will be implemented in the MQL5 code. For this purpose, we will use standard libraries that will facilitate work to maximum. MQL5 has CAppDialog that is a basic class for creating dialog windows. We are going to build our panel based on it.

Therefore, we will create a duplicate of the class and analyze it in the OnInit() function.

```
#include <Controls\Dialog.mqh>

class CTradePanel : public CAppDialog
  {
public:
                     CTradePanel(void){};
                    ~CTradePanel(void){};
  };

CTradePanel TradePanel;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   // Create Trade Panel
   if(!TradePanel.Create(ChartID(),"Trade Panel",0,20,20,320,420))
     {
      return (INIT_FAILED);
     }
   // Run Trade Panel
   TradePanel.Run();
//---
   return(INIT_SUCCEEDED);
  }
```

As a result of this relatively simple manipulation we obtain a template for our future panel.

![Template](https://c.mql5.com/2/23/iut7ytkfp.png)

### 3.2. Declare all necessary objects

Now we are going to apply all required controls to our template. Objects of relevant classes for every element of control are created for this purpose. We are going to create objects with standard classes CLabel, CEdit, CButton and CBmpButton.

We add necessary Include files and create a Creat() function for the CTradePanel class:

```
#include <Controls\Dialog.mqh>
#include <Controls\Label.mqh>
#include <Controls\Button.mqh>
```

"Edit.mqh" and "BmpButton.mqh" files haven't been included on purpose, since they are called from "Dialog.mqh".

The following step involves declaration of the relevant type variables for every object on the panel in the CTradePanel class. Here we also declare the Creat(..) procedure where all elements are duly arranged. Please note: declaration of variables and other actions within the CTradePanel class will be declared in the "private" block. However, functions available for calling beyond the class, such as Creat(...), are declared in the "public" block.

```
class CTradePanel : public CAppDialog
  {
private:

   CLabel            ASK, BID;                        // Display Ask and Bid prices
   CLabel            Balance_label;                   // Display label "Account Balance"
   CLabel            Balance_value;                   // Display Account balance
   CLabel            Equity_label;                    // Display label "Account Equity"
   CLabel            Equity_value;                    // Display Account Equity
   CLabel            PIPs;                            // Display label "Pips"
   CLabel            Currency;                        // Display Account currency
   CLabel            ShowLevels;                      // Display label "Show"
   CLabel            StopLoss;                        // Display label "Stop Loss"
   CLabel            TakeProfit;                      // Display label "TakeProfit"
   CLabel            Risk;                            // Display label "Risk"
   CLabel            Equity;                          // Display label "% to Equity"
   CLabel            Currency2;                       // Display Account currency
   CLabel            Orders;                          // Display label "Opened Orders"
   CLabel            Buy_Lots_label;                  // Display label "Buy Lots"
   CLabel            Buy_Lots_value;                  // Display Buy Lots value
   CLabel            Sell_Lots_label;                 // Display label "Sell Lots"
   CLabel            Sell_Lots_value;                 // Display Sell Lots value
   CLabel            Buy_profit_label;                // Display label "Buy Profit"
   CLabel            Buy_profit_value;                // Display Buy Profit value
   CLabel            Sell_profit_label;               // Display label "Sell Profit"
   CLabel            Sell_profit_value;               // Display Sell profit value
   CEdit             Lots;                            // Display volume of next order
   CEdit             StopLoss_pips;                   // Display Stop loss in pips
   CEdit             StopLoss_money;                  // Display Stop loss in accaunt currency
   CEdit             TakeProfit_pips;                 // Display Take profit in pips
   CEdit             TakeProfit_money;                // Display Take profit in account currency
   CEdit             Risk_percent;                    // Display Risk percent to equity
   CEdit             Risk_money;                      // Display Risk in account currency
   CBmpButton        StopLoss_line;                   // Check to display StopLoss Line
   CBmpButton        TakeProfit_line;                 // Check to display TakeProfit Line
   CBmpButton        StopLoss_pips_b;                 // Select Stop loss in pips
   CBmpButton        StopLoss_money_b;                // Select Stop loss in accaunt currency
   CBmpButton        TakeProfit_pips_b;               // Select Take profit in pips
   CBmpButton        TakeProfit_money_b;              // Select Take profit in account currency
   CBmpButton        Risk_percent_b;                  // Select Risk percent to equity
   CBmpButton        Risk_money_b;                    // Select Risk in account currency
   CBmpButton        Increase,Decrease;               // Increase and Decrease buttons
   CButton           SELL,BUY;                        // Sell and Buy Buttons
   CButton           CloseSell,CloseBuy,CloseAll;     // Close buttons

public:
                     CTradePanel(void){};
                    ~CTradePanel(void){};
  //--- Create function
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);

  };
```

### 3.3. Create initialization procedures for groups of objects

It is time to specify the body of the Creat(...) function. Please note that we should initialize all the above declared objects in this function. It's easy to calculate that we have declared 45 objects of 4 types. Therefore, it is useful to specify 4 procedures of initializing one object per each type. Class initialization functions are declared in the "private" block.

Certainly, objects could have been declared in the array, but then we would risk loosing connection between the name of the object's variable and its functionality, and that potentially could complicate work with objects. Therefore, a selection was made towards code transparency and application simplicity (to avoid obstacles).

### The CLabel class

The CLabel class will be used for displaying the informative text on our panel. When creating the initialization function, we need to determine which functions will be the same for all elements of this class, and which functions will be different. In this case, the differences are the following:

- object's name;
- displayed text;
- coordinates of an element;
- object alignment according to the anchor point.


After identifying differences, we determine which functions require to be passed by function parameters to make it universal, and which functions are generated in the actual process.

When working with objects, you should bear in mind that all chart objects must have separate names. As always, it is up to a programmer, whether each object name should be given independently, or generated by the program. By creating a universal function, I have chosen to generate names of objects within the program. Therefore, I have identified the object name for the class by adding a sequence number.

```
string name=m_name+"Label"+(string)ObjectsTotal(chart,-1,OBJ_LABEL);
```

The displayed text, object coordinates and object alignment, according to the anchor point, will be passed to the function using parameters. We will create enumerations for aligning the object to ensure the ease of reading the code and handling work for a programmer:

```
  enum label_align
     {
      left=-1,
      right=1,
      center=0
     };
```

Also, we should indicate a chart code, subwindow number and a link to the created object in the parameters of the procedure.

Within the actual function, we specify procedures that must be performed with every object of such class.

- We create object using the Create(...) function of the parent class.
- The required text is then placed in the object.
- The object is aligned according to the anchor point.
- We add the object in the "container" of the dialog window.

```
bool CTradePanel::CreateLabel(const long chart,const int subwindow,CLabel &object,const string text,const uint x,const uint y,label_align align)
  {
   // All objects must to have separate name
   string name=m_name+"Label"+(string)ObjectsTotal(chart,-1,OBJ_LABEL);
   //--- Call Create function
   if(!object.Create(chart,name,subwindow,x,y,0,0))
     {
      return false;
     }
   //--- Addjust text
   if(!object.Text(text))
     {
      return false;
     }
   //--- Aling text to Dialog box's grid
   ObjectSetInteger(chart,object.Name(),OBJPROP_ANCHOR,(align==left ? ANCHOR_LEFT_UPPER : (align==right ? ANCHOR_RIGHT_UPPER : ANCHOR_UPPER)));
   //--- Add object to controls
   if(!Add(object))
     {
      return false;
     }
   return true;
  }
```

### The CButton class

The CButton class is aimed for creating rectangular-shaped buttons with a label. These are our standard buttons for opening and closing orders.

Starting work with this class of objects, we use the same approach as in the previous case. Specifics of its operation should be considered, though. First of all, there is no need to align the button text, since it has a center alignment in the parent class. The button size that we will pass in parameters appears here already.

Also, the button's state appears: pressed or depressed. Furthermore, a pressed button can be locked down. Therefore, these additional options have to be described in the object initialization process. We will disable LockDown for our buttons and set them in the "Depressed" state.

```
bool CTradePanel::CreateButton(const long chart,const int subwindow,CButton &object,const string text,const uint x,const uint y,const uint x_size,const uint y_size)
  {
   // All objects must to have separate name
   string name=m_name+"Button"+(string)ObjectsTotal(chart,-1,OBJ_BUTTON);
   //--- Call Create function
   if(!object.Create(chart,name,subwindow,x,y,x+x_size,y+y_size))
     {
      return false;
     }
   //--- Addjust text
   if(!object.Text(text))
     {
      return false;
     }
   //--- set button flag to unlock
   object.Locking(false);
   //--- set button flag to unpressed
   if(!object.Pressed(false))
     {
      return false;
     }
   //--- Add object to controls
   if(!Add(object))
     {
      return false;
     }
   return true;
  }
```

### The CEdit class

The CEdit class is used for creating data entry objects. Cells for entering volumes of deals, Stop Loss and Take Profit (in points and in currency of deposit) and risk levels apply to such objects.

The same approach as with two previously described classes is applied. But, unlike with buttons, it is required to indicate how to align the cell text in a process of initializing this class. It should be remembered that any information that is entered or passed to a cell is always interpreted as a text. Therefore, when passing numbers to display an object, they should be first converted to a text.

Objects of CEdit, unlike buttons, don't have "Pressed" / "Depressed" states, but at the same time this class creates objects that are unavailable for editing by a user during the program's operation. In our case, all objects should enable a user to edit them. We will indicate this in our initialization function.

```
bool CTradePanel::CreateEdit(const long chart,const int subwindow,CEdit &object,const string text,const uint x,const uint y,const uint x_size,const uint y_size)
  {
   // All objects must to have separate name
   string name=m_name+"Edit"+(string)ObjectsTotal(chart,-1,OBJ_EDIT);
   //--- Call Create function
   if(!object.Create(chart,name,subwindow,x,y,x+x_size,y+y_size))
     {
      return false;
     }
   //--- Addjust text
   if(!object.Text(text))
     {
      return false;
     }
   //--- Align text in Edit box
   if(!object.TextAlign(ALIGN_CENTER))
     {
      return false;
     }
   //--- set Read only flag to false
   if(!object.ReadOnly(false))
     {
      return false;
     }
   //--- Add object to controls
   if(!Add(object))
     {
      return false;
     }
   return true;
  }
```

### The CBmpButton class

The CBmpButton class is used to create non-standard buttons by using graphic objects, instead of labels. These buttons that are understandable for any user are used when creating standardized controls for various applications. In our case, using this class we create the following:

- radio buttons to select the terms of Stop Loss, Take Profit and risk: in currency or points (or percentage - for risk);

- check boxes to fix the display of Stop Loss and Take Profit levels;
- buttons to increase or decrease the volume of deal.

Working with this class of objects is similar to working with the CButton class. The difference is in passing graphic objects for pressed and depressed button states instead of a text. For our panel we will use the image of buttons provided with MQL5. In order to distribute a ready software product with one file, we will specify these images as resources.

```
#resource "\\Include\\Controls\\res\\RadioButtonOn.bmp"
#resource "\\Include\\Controls\\res\\RadioButtonOff.bmp"
#resource "\\Include\\Controls\\res\\CheckBoxOn.bmp"
#resource "\\Include\\Controls\\res\\CheckBoxOff.bmp"
#resource "\\Include\\Controls\\res\\SpinInc.bmp"
#resource "\\Include\\Controls\\res\\SpinDec.bmp"
```

It should be also noted that all elements of this class, **apart from increase and decrease lot** buttons, maintain their "Pressed" or "Depressed" state. Therefore, we add additional parameters to the initialization function.

```
//+------------------------------------------------------------------+
//| Create BMP Button                                                |
//+------------------------------------------------------------------+
bool CTradePanel::CreateBmpButton(const long chart,const int subwindow,CBmpButton &object,const uint x,const uint y,string BmpON,string BmpOFF,bool lock)
  {
   // All objects must to have separate name
   string name=m_name+"BmpButton"+(string)ObjectsTotal(chart,-1,OBJ_BITMAP_LABEL);
   //--- Calculate coordinates
   uint y1=(uint)(y-(Y_STEP-CONTROLS_BUTTON_SIZE)/2);
   uint y2=y1+CONTROLS_BUTTON_SIZE;
   //--- Call Create function
   if(!object.Create(m_chart_id,name,m_subwin,x-CONTROLS_BUTTON_SIZE,y1,x,y2))
      return(false);
   //--- Assign BMP pictures to button status
   if(!object.BmpNames(BmpOFF,BmpON))
      return(false);
   //--- Add object to controls
   if(!Add(object))
      return(false);
   //--- set Lock flag to true
   object.Locking(lock);
//--- succeeded
   return(true);
  }
```

By specifying functions of creating objects, it is mandatory to declare these functions in the "private" block of our class.

```
private:

   //--- Create Label object
   bool              CreateLabel(const long chart,const int subwindow,CLabel &object,const string text,const uint x,const uint y,label_align align);
   //--- Create Button
   bool              CreateButton(const long chart,const int subwindow,CButton &object,const string text,const uint x,const uint y,const uint x_size,const uint y_size);
   //--- Cleate Edit object
   bool              CreateEdit(const long chart,const int subwindow,CEdit &object,const string text,const uint x,const uint y,const uint x_size,const uint y_size);
   //--- Create BMP Button
   bool              CreateBmpButton(const long chart,const int subwindow,CBmpButton &object,const uint x,const uint y,string BmpON,string BmpOFF,bool lock);
```

### 3.4. Arrange all elements in order

Now that we wrote the initialization function for each class of objects, it is time to write it for our trading panel also. The main targets of this function are: calculation of coordinates for every panel object and step by step creation of all objects by calling the relevant initialization function.

Let me remind you that elements on the panel should be conveniently located for a user and have an aesthetic appeal. We have focused on this matter when creating the model of our panel, and will adhere this concept now. At the same time it is important to understand that when using our class in the final program, the sizes of panels may differ. In order to maintain the concept of our design when the sizes change, we must calculate coordinates of every object, instead of indicating clearly. For this purpose we will create a peculiar lighthouse:

- distance from the window border until the first element of control;
- distance between elements of control in height;
- height of control.

```
   #define  Y_STEP   (int)(ClientAreaHeight()/18/4)      // height step between elements
   #define  Y_WIDTH  (int)(ClientAreaHeight()/18)        // height of element
   #define  BORDER   (int)(ClientAreaHeight()/24)        // distance between border and elements
```

This way, we will be able to calculate the coordinates of the first control and every following control relative to a previous one.

Also, by defining optimal sizes of our panel, we can indicate them as default values for parameters passed to the function.

```
bool CTradePanel::Create(const long chart,const string name,const int subwin=0,const int x1=20,const int y1=20,const int x2=320,const int y2=420)
  {
      // At first call create function of parents class
   if(!CAppDialog::Create(chart,name,subwin,x1,y1,x2,y2))
     {
      return false;
     }
   // Calculate coordinates and size of BID object
   // Coordinates calculation in dialog box, not in chart
   int l_x_left=BORDER;
   int l_y=BORDER;
   int y_width=Y_WIDTH;
   int y_sptep=Y_STEP;
   // Create object
   if(!CreateLabel(chart,subwin,BID,DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits),l_x_left,l_y,left))
     {
      return false;
     }
   // Adjust font size for object
   if(!BID.FontSize(Y_WIDTH))
     {
      return false;
     }
   // Repeat same functions for other objects
   int l_x_right=ClientAreaWidth()-20;
   if(!CreateLabel(chart,subwin,ASK,DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits),l_x_right,l_y,right))
     {
      return false;
     }
   if(!ASK.FontSize(Y_WIDTH))
     {
      return false;
     }
   l_y+=2*Y_WIDTH;
...................
  }
```

You can see the entire code of the function in the attached example.

The following panel is the result of our efforts.

![Panel](https://c.mql5.com/2/23/Final.png)

So far it is just a model — a beautiful image on the chart, but we will "bring it to life" at the next stage.

### 4\. Bring the image to life

Now that we have created a graphic model of our trading panel, it is time to teach it to react to events. Accordingly, in order to create and set up the event handler, we need to figure out to what events and how it should react.

### 4.1. Change the tool price

When changing the tool's price with the МТ5 terminal, the NewTick event that runs the Expert Advisor's OnTick() function is generated. Therefore, we should call a relevant function of our class from this function that will handle this event. We will give it a similar name — OnTick(), and declare it in the "public" block, since it will be called from the external program.

```
public:

   virtual void      OnTick(void);
```

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   TradePanel.OnTick();
  }
```

What changes occur on the panel, if the tool price changes? The first thing we should do is change Ask and Bid values on our panel.

```
//+------------------------------------------------------------------+
//| Event "New Tick                                                  |
//+------------------------------------------------------------------+
void CTradePanel::OnTick(void)
  {
   //--- Change Ask and Bid prices on panel
   ASK.Text(DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_ASK),(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS)));
   BID.Text(DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_BID),(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS)));
```

Then, in case there are open positions, we change Equity on the panel. As a precaution I have added a compatibility of the figure displayed on the panel to Equity, even when there are no open positions. This will allow to display the actual funds after the "emergency situations". This way, there is no need to check for open positions: we check the conformity between the current equity on the account and the figure displayed on the panel. If required, we display the true value on the panel.

```
//--- Сheck and change (if necessary) equity
   if(Equity_value.Text()!=DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2)+" "+AccountInfoString(ACCOUNT_CURRENCY))
     {
      Equity_value.Text(DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2)+" "+AccountInfoString(ACCOUNT_CURRENCY));
     }

```

Similar iterations will be also created for displaying balance.

I can see this question coming: "Why to check for balance on every tick, it is only changed when performing trading operations?" Yes, this is correct, and later we will discuss how to react to trading events. But there is a small chance to trade when our panel is not launched or there is no connection between the terminal and the server. I have added this operation specifically to display the actual balance on the panel at all times, even after various emergency situations.

The next step when changing the price calls for checking the presence of open position based on the current instrument, and in case it is present, we check and correct the value of open volume and current profit of positions in Buy or Sell fields.

```
//--- Check and change (if neccesory) Buy and Sell lots and profit value.
   if(PositionSelect(_Symbol))
     {
      switch((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
           Buy_profit_value.Text(DoubleToString(PositionGetDouble(POSITION_PROFIT),2)+" "+AccountInfoString(ACCOUNT_CURRENCY));
           if(Buy_Lots_value.Text()!=DoubleToString(PositionGetDouble(POSITION_VOLUME),2))
              {
               Buy_Lots_value.Text(DoubleToString(PositionGetDouble(POSITION_VOLUME),2));
              }
           if(Sell_profit_value.Text()!=DoubleToString(0,2)+" "+AccountInfoString(ACCOUNT_CURRENCY))
              {
               Sell_profit_value.Text(DoubleToString(0,2)+" "+AccountInfoString(ACCOUNT_CURRENCY));
              }
           if(Sell_Lots_value.Text()!=DoubleToString(0,2))
              {
               Sell_Lots_value.Text(DoubleToString(0,2));
              }
           break;
         case POSITION_TYPE_SELL:
           Sell_profit_value.Text(DoubleToString(PositionGetDouble(POSITION_PROFIT),2)+" "+AccountInfoString(ACCOUNT_CURRENCY));
           if(Sell_Lots_value.Text()!=DoubleToString(PositionGetDouble(POSITION_VOLUME),2))
              {
               Sell_Lots_value.Text(DoubleToString(PositionGetDouble(POSITION_VOLUME),2));
              }
           if(Buy_profit_value.Text()!=DoubleToString(0,2)+" "+AccountInfoString(ACCOUNT_CURRENCY))
              {
               Buy_profit_value.Text(DoubleToString(0,2)+" "+AccountInfoString(ACCOUNT_CURRENCY));
              }
           if(Buy_Lots_value.Text()!=DoubleToString(0,2))
              {
               Buy_Lots_value.Text(DoubleToString(0,2));
              }
           break;
        }
     }
   else
     {
      if(Buy_Lots_value.Text()!=DoubleToString(0,2))
        {
         Buy_Lots_value.Text(DoubleToString(0,2));
        }
      if(Sell_Lots_value.Text()!=DoubleToString(0,2))
        {
         Sell_Lots_value.Text(DoubleToString(0,2));
        }
      if(Buy_profit_value.Text()!=DoubleToString(0,2)+" "+AccountInfoString(ACCOUNT_CURRENCY))
        {
         Buy_profit_value.Text(DoubleToString(0,2)+" "+AccountInfoString(ACCOUNT_CURRENCY));
        }
      if(Sell_profit_value.Text()!=DoubleToString(0,2)+" "+AccountInfoString(ACCOUNT_CURRENCY))
        {
         Sell_profit_value.Text(DoubleToString(0,2)+" "+AccountInfoString(ACCOUNT_CURRENCY));
        }
     }
```

Also, we should not forget to check the state of check boxes for displaying Stop Loss and Take Profit levels on the chart. If required, we can correct the position of lines. Calls of these functions will be added to the code. Further information is provided below.

```
   //--- Move SL and TP lines if necessary
   if(StopLoss_line.Pressed())
     {
      UpdateSLLines();
     }
   if(TakeProfit_line.Pressed())
     {
      UpdateTPLines();
     }
   return;
  }
```

### 4.2. Enter values into editable fields.

There is whole range of editable fields on our panel, and, certainly, we need to set up the receipt and handling of entered information.

Entering information into editable fields is an event of changing the graphic object that belongs to the ChartEvent group. This group events are handled with the OnChartEvent function. It has 4 input parameters: event identifier and 3 parameters that characterize an event, that refer to long, double and string types. As in the previous case, we will create the event handler in our class and call it from the OnChartEvent function with passing all input parameters that characterize an event. Going a bit forward, I would like to say that events of pressing the trading panel buttons will also be handled with this function. Therefore, this function will act as a dispatcher that calls the function of handling a specific event after analyzing it. Information about event will then be passed to a parent class function for performing procedures specified in a parent class.

```
public:

   virtual bool      OnEvent(const int id,const long &lparam, const double &dparam, const string &sparam);

//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   TradePanel.OnEvent(id, lparam, dparam, sparam);
  }
```

Macro substitution will be applied for building such dispatcher.

```
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CTradePanel)
   ON_EVENT(ON_END_EDIT,Lots,LotsEndEdit)
   ON_EVENT(ON_END_EDIT,StopLoss_pips,SLPipsEndEdit)
   ON_EVENT(ON_END_EDIT,TakeProfit_pips,TPPipsEndEdit)
   ON_EVENT(ON_END_EDIT,StopLoss_money,SLMoneyEndEdit)
   ON_EVENT(ON_END_EDIT,TakeProfit_money,TPMoneyEndEdit)
   ON_EVENT(ON_END_EDIT,Risk_percent,RiskPercentEndEdit)
   ON_EVENT(ON_END_EDIT,Risk_money,RiskMoneyEndEdit)
EVENT_MAP_END(CAppDialog)
```

Accordingly, all functions of handling events should be declared in the "private" block of our class.

```
private:

   //--- On Event functions
   void              LotsEndEdit(void);                              // Edit Lot size
   void              SLPipsEndEdit(void);                            // Edit Stop Loss in pips
   void              TPPipsEndEdit(void);                            // Edit Take Profit in pips
   void              SLMoneyEndEdit(void);                           // Edit Stop Loss in money
   void              TPMoneyEndEdit(void);                           // Edit Take Profit in money
   void              RiskPercentEndEdit(void);                       // Edit Risk in percent
   void              RiskMoneyEndEdit(void);                         // Edit Risk in money
```

In order to store data obtained from editable fields, we enter additional variables in the "private" block.

```
private:

   //--- variables of current values
   double            cur_lot;                         // Lot of next order
   int               cur_sl_pips;                     // Stop Loss in pips
   double            cur_sl_money;                    // Stop Loss in money
   int               cur_tp_pips;                     // Take Profit in pips
   double            cur_tp_money;                    // Take Profit in money
   double            cur_risk_percent;                // Risk in percent
   double            cur_risk_money;                  // Risk in money
```

Let's analyze the example of a specific event — entering volume of a prepared trade. Let me remind you that entering any information in similar fields is interpreted as a text, despite of its content. In fact, when entering a text in a field a whole range of events is generated: mouse pointer going over an object, pressing the mouse button, start of editing a field, clicking the keyboard keys, end of editing a field etc. We are interested only in the last event when the process of entering information is finished. Therefore, calling the function will be performed based on the "ON\_END\_EDIT" event.

The first thing we should do in the event handling function is to read the entered text and to transform it to the double type value.

Then we have to "normalize" the obtained value, which implies setting it in accordance with the conditions of a trading instrument (minimum and maximum volume of one order, and change step of volume). In order to perform this operation we will write a separate function, because it will be needed when pressing the buttons for increasing and decreasing the deal volume. The obtained value should be returned to the panel in order to inform a trader about a factual volume of a future deal.

```
//+------------------------------------------------------------------+
//| Read lots value after edit                                       |
//+------------------------------------------------------------------+
void CTradePanel::LotsEndEdit(void)
  {
   //--- Read and normalize lot value
   cur_lot=NormalizeLots(StringToDouble(Lots.Text()));
   //--- Output lot value to panel
   Lots.Text(DoubleToString(cur_lot,2));
```

Apart from that, we will have to recalculate and change values of the remaining editable fields on the panel, depending on current settings of radio buttons. This is obligatory because the risk total is changed when deals are closed by Stop Loss (in case of indicating Stop Loss in points) or Stop Loss level in points (in case of indicating Stop Loss in money) when changing trade volume. The risk level will be followed after Stop Loss. The similar situation applies to Take Profit values. Certainly, these operations will be organized through relevant functions.

```
   //--- Check and modify value of other labels
   if(StopLoss_money_b.Pressed())
     {
      StopLossPipsByMoney();
     }
   if(TakeProfit_money_b.Pressed())
     {
      TakeProfitPipsByMoney();
     }
   if(StopLoss_pips_b.Pressed())
     {
      StopLossMoneyByPips();
     }
   if(TakeProfit_pips_b.Pressed())
     {
      TakeProfitMoneyByPips();
     }
```

When we create a tool for a user's daily operation, we should always bear in mind the "usability" term (use of convenience). And here we need to remember about p.8 described in the beginning of our trading panel's functionality: "The panel should memorize which parameters are entered by a trader, and which of them are automatically calculated. This is required so the parameters entered by a trader don't change, if possible, at subsequent recalculations. In other words, when changing Stop Loss in points in the future, we should remember that a trader has changed deal volume and risk level. If necessary, last entered data can be requested to remain the same.

For this purpose we will enter the RiskByValue variable into the "private" block, and assign it with true value in the event handling function.

```
private:
   bool              RiskByValue;                     // Flag: Risk by Value or Value by Risk
   RiskByValue=true;
   return;
  }
```

We will consider the principles of organizing a function of correcting connected editable fields based on the StopLossMoneyByPips function, since it has a more comprehensive functionality.

1\. In fact, this function will be called in three cases: when changing a lot size, entering value in the Stop Loss field in peeps and moving the Stop Loss line. So, the first thing we should do — check the current volume of the expected deal. If it doesn't match the tool specifications and the market, the value displayed on the panel should be corrected.

```
//+------------------------------------------------------------------+
//|  Modify SL money by Order lot and SL pips                        |
//+------------------------------------------------------------------+
void CTradePanel::StopLossMoneyByPips(void)
  {
   //--- Read and normalize lot value
   cur_lot=NormalizeLots(StringToDouble(Lots.Text()));
   //--- Output lot value to panel
   Lots.Text(DoubleToString(cur_lot,2));
```

2\. The second component for calculating a monetary value of a possible risk is a sum of changing funds when the instrument's price changes by one tick with an open position of 1 lot. For this purpose, we will obtain the price of one tick and a minimum size of changing the instrument's price:

```
   double tick_value=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double tick_size=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
```

3\. From the obtained data we calculate possible losses, and the obtained value is displayed on the panel in a relevant field.

```
   cur_sl_money=NormalizeDouble(tick_value*cur_lot*(tick_size/_Point)*cur_sl_pips,2);
   StopLoss_money.Text(DoubleToString(cur_sl_money,2));
```

4 Please note that a sum of possible loses when closing an order by Stop Loss, in fact, is our risk in a monetary expression. Therefore, we should duplicate the calculated value in the risk field in monetary terms, and then calculate the relative value of risk (in percentage).

```
   cur_risk_money=cur_sl_money;
   Risk_money.Text(DoubleToString(cur_risk_money,2));
   cur_risk_percent=NormalizeDouble(cur_risk_money/AccountInfoDouble(ACCOUNT_BALANCE)*100,2);
   Risk_percent.Text(DoubleToString(cur_risk_percent,2));
  return;
 }
```

The function of calculating Stop Loss in points based on the monetary value is contrary from the function described above, except that the risk is not changed, but the position of Stop Loss levels on the chart.

Functions of correcting Take Profit values are specified the same way.

We similarly create functions for handling events of editing other fields. Please remember that when editing fields we will also have to change the state of radio buttons. In order to avoid duplicating the specification of button states in every function, we will call a handling event function for pressing the relevant button.

### 4.3. Handle events of pressing radio buttons.

Radio button is an element of the interface that allows users to select one option (point) from the pre-selected set (group).

Therefore, when pressing one radio button, we should change states of interconnected buttons. At the same time, the switch of radio buttons doesn't lead to re-calculation of any parameters.

This way, functions of handling events of pressing radio buttons will only change the state of interconnected radio buttons, i.e. bring the pressed radio button to the "Pressed" state, and other dependent buttons — to the "Depressed" state.

As for the technical aspect, pressing the button refers to the ChartEvent group of events. Therefore, handling will be performed the same way as editing fields. We declare functions of handling events in the "private" block:

```
private:

   //--- On Event functions
   void              SLPipsClick();                                  // Click Stop Loss in pips
   void              TPPipsClick();                                  // Click Take Profit in pips
   void              SLMoneyClick();                                 // Click Stop Loss in money
   void              TPMoneyClick();                                 // Click Take Profit in money
   void              RiskPercentClick();                             // Click Risk in percent
   void              RiskMoneyClick();                               // Click Risk in money
```

We add macro substitution of the event handler:

```
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CTradePanel)
   ON_EVENT(ON_END_EDIT,Lots,LotsEndEdit)
   ON_EVENT(ON_END_EDIT,StopLoss_pips,SLPipsEndEdit)
   ON_EVENT(ON_END_EDIT,TakeProfit_pips,TPPipsEndEdit)
   ON_EVENT(ON_END_EDIT,StopLoss_money,SLMoneyEndEdit)
   ON_EVENT(ON_END_EDIT,TakeProfit_money,TPMoneyEndEdit)
   ON_EVENT(ON_END_EDIT,Risk_percent,RiskPercentEndEdit)
   ON_EVENT(ON_END_EDIT,Risk_money,RiskMoneyEndEdit)
   ON_EVENT(ON_CLICK,StopLoss_pips_b,SLPipsClick)
   ON_EVENT(ON_CLICK,TakeProfit_pips_b,TPPipsClick)
   ON_EVENT(ON_CLICK,StopLoss_money_b,SLMoneyClick)
   ON_EVENT(ON_CLICK,TakeProfit_money_b,TPMoneyClick)
   ON_EVENT(ON_CLICK,Risk_percent_b,RiskPercentClick)
   ON_EVENT(ON_CLICK,Risk_money_b,RiskMoneyClick)
EVENT_MAP_END(CAppDialog)
```

The function of handling events will look as follows:

```
//+------------------------------------------------------------------+
//| Click Stop Loss in pips                                          |
//+------------------------------------------------------------------+
void CTradePanel::SLPipsClick(void)
  {
   StopLoss_pips_b.Pressed(cur_sl_pips>0);
   StopLoss_money_b.Pressed(false);
   Risk_money_b.Pressed(false);
   Risk_percent_b.Pressed(false);
   return;
  }
```

See the attached code to get acquainted with all handling events functions.

### 4.4. Press buttons for changing the deal volume.

Unlike radio buttons, the program must perform the whole range of operations that we should specify in a code when pressing the buttons for changing the deal volume. First of all, this is an increase or decrease of the cur\_lot variable by the size of the change step of the deal volume. Then you must compare the obtained value with maximum and minimum possible value for a tool. For an additional option, I would suggest to check the presence of available funds for opening the order of such volume, because afterwards, when a trader opens a new order, account funds may be insufficient. Then, we will have to display a new value of the deal volume on the panel and edit the relevant values, as in the case of manually entering the deal volume in the editable field.

The same as previously, we will declare our functions in the private block:

```
private:
................
   //--- On Event functions
................
   void              IncreaseLotClick();                             // Click Increase Lot
   void              DecreaseLotClick();                             // Click Decrease Lot
```

We add the function of handling interruptions with micro substitutions:

```
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CTradePanel)
   ON_EVENT(ON_END_EDIT,Lots,LotsEndEdit)
   ON_EVENT(ON_END_EDIT,StopLoss_pips,SLPipsEndEdit)
   ON_EVENT(ON_END_EDIT,TakeProfit_pips,TPPipsEndEdit)
   ON_EVENT(ON_END_EDIT,StopLoss_money,SLMoneyEndEdit)
   ON_EVENT(ON_END_EDIT,TakeProfit_money,TPMoneyEndEdit)
   ON_EVENT(ON_END_EDIT,Risk_percent,RiskPercentEndEdit)
   ON_EVENT(ON_END_EDIT,Risk_money,RiskMoneyEndEdit)
   ON_EVENT(ON_CLICK,StopLoss_pips_b,SLPipsClick)
   ON_EVENT(ON_CLICK,TakeProfit_pips_b,TPPipsClick)
   ON_EVENT(ON_CLICK,StopLoss_money_b,SLMoneyClick)
   ON_EVENT(ON_CLICK,TakeProfit_money_b,TPMoneyClick)
   ON_EVENT(ON_CLICK,Risk_percent_b,RiskPercentClick)
   ON_EVENT(ON_CLICK,Risk_money_b,RiskMoneyClick)
   ON_EVENT(ON_CLICK,Increase,IncreaseLotClick)
   ON_EVENT(ON_CLICK,Decrease,DecreaseLotClick)
EVENT_MAP_END(CAppDialog)
```

Let's look into the event handling function:

```
//+------------------------------------------------------------------+
//|  Increase Lot Click                                              |
//+------------------------------------------------------------------+
void CTradePanel::IncreaseLotClick(void)
  {
   //--- Read and normalize lot value
   cur_lot=NormalizeLots(StringToDouble(Lots.Text())+SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP));
   //--- Output lot value to panel
   Lots.Text(DoubleToString(cur_lot,2));
   //--- Call end edit lot function
   LotsEndEdit();
   return;
  }
```

First we read the current value of deal volume and increase it by a step from the tool specification. Then, straight away we bring the obtained value in accordance with specification of the tool with the NormalizeLots function that we have already encountered.

Further, we call the function of handling changes of the lot volume in the entry window, as we have previously specified all necessary procedures in this function.

The function of decreasing a lot is created similarly.

### 4.5. Change states of check boxes.

At the following stage we will create the event handler for reacting to pressing check boxes. There are two panels on our check box for switching on/off the display of Stop Loss and Take Profit levels on the chart.

What should occur when changing the state of a check box? In fact, the main function of this event involves displaying the chart lines. There are two ways to solve this problem:

1. create and remove lines every time you press;
2. create lines on the chart together with all objects of the panel once, and to display or hide check boxes when their state changes.

I chose the second option. For this purpose, one more library will be connected:

```
#include <ChartObjects\ChartObjectsLines.mqh>
```

Then we declare objects of horizontal lines in the private block and declare the function of their initialization:

```
private:
.................
   CChartObjectHLine BuySL, SellSL, BuyTP, SellTP;    // Stop Loss and Take Profit Lines

   //--- Create Horizontal line
   bool              CreateHLine(long chart, int subwindow,CChartObjectHLine &object,color clr, string comment);
```

We specify the procedure of initializing horizontal lines. First, we create a line on the chart.

```
//+------------------------------------------------------------------+
//| Create horizontal line                                           |
//+------------------------------------------------------------------+
bool CTradePanel::CreateHLine(long chart, int subwindow,CChartObjectHLine &object,color clr, string comment)
  {
   // All objects must to have separate name
   string name="HLine"+(string)ObjectsTotal(chart,-1,OBJ_HLINE);
   //--- Create horizontal line
   if(!object.Create(chart,name,subwindow,0))
      return false;
```

Then, we set color, type of line and add comment that is displayed when hovering over an object.

```
   //--- Set color of lin
   if(!object.Color(clr))
      return false;
   //--- Set dash style to line
   if(!object.Style(STYLE_DASH))
      return false;
   //--- Add comment to line
   if(!object.Tooltip(comment))
      return false;
```

We hide the line from the chart and move the line to the background.

```
   //--- Hide line
   if(!object.Timeframes(OBJ_NO_PERIODS))
      return false;
   //--- Move line to background
   if(!object.Background(true))
      return false;
```

Since one of the panel's options is to allow traders to move Stop Loss or Take Profit lines on the chart, we will enable highlighting lines:

```
   if(!object.Selectable(true))
      return false;
   return true;
  }
```

Now, we add initialization of lines to the function of creating our trading panel.

```
//+------------------------------------------------------------------+
//| Creat Trade Panel function                                       |
//+------------------------------------------------------------------+
bool CTradePanel::Create(const long chart,const string name,const int subwin=0,const int x1=20,const int y1=20,const int x2=320,const int y2=420)
  {
...................
...................
   //--- Create horizontal lines of SL & TP
   if(!CreateHLine(chart,subwin,BuySL,SL_Line_color,"Buy Stop Loss"))
     {
      return false;
     }
   if(!CreateHLine(chart,subwin,SellSL,SL_Line_color,"Sell Stop Loss"))
     {
      return false;
     }
   if(!CreateHLine(chart,subwin,BuyTP,TP_Line_color,"Buy Take Profit"))
     {
      return false;
     }
   if(!CreateHLine(chart,subwin,SellTP,TP_Line_color,"Sell Take Profit"))
     {
      return false;
     }
    return true;
  }
```

After we have created lines, we will specify the event handling function. It will be built according to the same scheme applied previously with the function of handling previous events. We declare functions of handling events in the private block:

```
private:
...............
   void              StopLossLineClick();                            // Click StopLoss Line
   void              TakeProfitLineClick();                          // Click TakeProfit Line
```

Calling the function is added to the even handler:

```
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CTradePanel)
   ON_EVENT(ON_END_EDIT,Lots,LotsEndEdit)
   ON_EVENT(ON_END_EDIT,StopLoss_pips,SLPipsEndEdit)
   ON_EVENT(ON_END_EDIT,TakeProfit_pips,TPPipsEndEdit)
   ON_EVENT(ON_END_EDIT,StopLoss_money,SLMoneyEndEdit)
   ON_EVENT(ON_END_EDIT,TakeProfit_money,TPMoneyEndEdit)
   ON_EVENT(ON_END_EDIT,Risk_percent,RiskPercentEndEdit)
   ON_EVENT(ON_END_EDIT,Risk_money,RiskMoneyEndEdit)
   ON_EVENT(ON_CLICK,StopLoss_pips_b,SLPipsClick)
   ON_EVENT(ON_CLICK,TakeProfit_pips_b,TPPipsClick)
   ON_EVENT(ON_CLICK,StopLoss_money_b,SLMoneyClick)
   ON_EVENT(ON_CLICK,TakeProfit_money_b,TPMoneyClick)
   ON_EVENT(ON_CLICK,Risk_percent_b,RiskPercentClick)
   ON_EVENT(ON_CLICK,Risk_money_b,RiskMoneyClick)
   ON_EVENT(ON_CLICK,Increase,IncreaseLotClick)
   ON_EVENT(ON_CLICK,Decrease,DecreaseLotClick)
   ON_EVENT(ON_CLICK,StopLoss_line,StopLossLineClick)
   ON_EVENT(ON_CLICK,TakeProfit_line,TakeProfitLineClick)
EVENT_MAP_END(CAppDialog)
```

And, finally, we write the function of handling the event. In the beginning of the function we test the state of the check box. Further actions will depend on it. If pressed, then levels of display should be updated before displaying the lines. Then we apply lines to the chart.

```
//+------------------------------------------------------------------+
//| Show and Hide Stop Loss Lines                                    |
//+------------------------------------------------------------------+
void CTradePanel::StopLossLineClick()
  {
   if(StopLoss_line.Pressed()) // Button pressed
     {
      if(BuySL.Price(0)<=0)
        {
         UpdateSLLines();
        }
      BuySL.Timeframes(OBJ_ALL_PERIODS);
      SellSL.Timeframes(OBJ_ALL_PERIODS);
     }
```

If check box is not pressed, the lines are hidden.

```
   else                         // Button unpressed
     {
      BuySL.Timeframes(OBJ_NO_PERIODS);
      SellSL.Timeframes(OBJ_NO_PERIODS);
     }
   ChartRedraw();
   return;
  }
```

At the end of the function we call the chart re-drawing.

### 4.6. Trade

Now that functions of handling events for main control of the panel are described, we proceed with handling events of pressing the buttons of trading operations. For trading on the account we also use the standard library MQL5 "Trade.mqh", where the CTrade class of trading operations is described.

```
#include <Trade\Trade.mqh>
```

We declare the class of trading operations in the private block:

```
private:
................
   CTrade            Trade;                           // Class of trade operations
```

And we perform initialization of the trading class in the function of initializing our class. Here we set a magic number of deals, slippage level for trades and policy of performing trading orders.

```
//+------------------------------------------------------------------+
//| Class initialization function                                    |
//+------------------------------------------------------------------+
CTradePanel::CTradePanel(void)
  {
   Trade.SetExpertMagicNumber(0);
   Trade.SetDeviationInPoints(5);
   Trade.SetTypeFilling((ENUM_ORDER_TYPE_FILLING)0);
   return;
  }
```

If you wish, you can add additional functions here for setting magic number and slippage level from the external program. Don't forget that these functions should be declared in the public block.

After all preparation we write the functions of handling events of pressing buttons. First, same as before, we declare functions in the private block:

```
private:
.....................
   void              BuyClick();                                     // Click BUY button
   void              SellClick();                                    // Click SELL button
   void              CloseBuyClick();                                // Click CLOSE BUY button
   void              CloseSellClick();                               // Click CLOSE SELL button
   void              CloseClick();                                   // Click CLOSE ALL button
```

Then we add new functions to the dispatcher of handling events:

```
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CTradePanel)
...................
   ON_EVENT(ON_CLICK,BUY,BuyClick)
   ON_EVENT(ON_CLICK,SELL,SellClick)
   ON_EVENT(ON_CLICK,CloseBuy,CloseBuyClick)
   ON_EVENT(ON_CLICK,CloseSell,CloseSellClick)
   ON_EVENT(ON_CLICK,CloseAll,CloseClick)
EVENT_MAP_END(CAppDialog)
```

And, certainly, we specify the functions of handling events. Let's consider a Buy option, for example. Which actions should our program perform when the "BUY" button is pressed?

Possibly, first we should actualize the expected deal volume. We read the value in the lot field, adjust it to the specification of the instrument and check the sufficiency of funds for opening an order, and then return the updated value to the panel.

```
void CTradePanel::BuyClick(void)
  {
   cur_lot=NormalizeLots(StringToDouble(Lots.Text()));
   Lots.Text(DoubleToString(cur_lot,2));
```

At the next stage we obtain the market price of the tool and calculate the Stop Loss and Take Profit price levels according to parameters set on the panel:

```
   double price=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double SL=(cur_sl_pips>0 ? NormalizeDouble(price-cur_sl_pips*_Point,_Digits) : 0);
   double TP=(cur_tp_pips>0 ? NormalizeDouble(price+cur_tp_pips*_Point,_Digits) : 0);
```

And, finally, we send the request for placing the order to the broker's server. If there is error, a function of notifying traders must be added.

```
   if(!Trade.Buy(NormalizeLots(cur_lot),_Symbol,price,SL,TP,"Trade Panel"))
      MessageBox("Error of open BUY ORDER "+Trade.ResultComment(),"Trade Panel Error",MB_ICONERROR|MB_OK);;
   return;
  }
```

Similarly, we create functions to handle pressing on other trading buttons. You can learn more about the code of these functions from the attached file.

### 5\. Move Stop Loss and Take Profit levels "manually".

We remember that fairly frequently traders move Stop Loss and Take Profit levels to some meaningful levels on the chart. In my opinion, it is not right to make users calculate the number of points from the current price to this level. Therefore, we will allow users to simply move the line to the necessary point, and the program will do the rest.

I have decided not to overload the program with the code of handling mouse movements on the chart, and applied the terminal's standard function for moving objects. For this purpose, we left the opportunity for users to select and move the horizontal lines. The "CHARTEVENT\_OBJECT\_DRAG" event will be handled using the program.

As always, first we declare the function of handling events in the public block, i.e. this function will be called from the external program:

```
public:
................
   virtual bool      DragLine(string name);
```

Calling this function will be performed from the OnChartEvent function of the main program at the arrival of the event with transfer of the object's name.

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id==CHARTEVENT_OBJECT_DRAG)
     {
      if(TradePanel.DragLine(sparam))
        {
         ChartRedraw();
        }
     }
...........
```

In the actual function of handling the event we must:

- define, which line exactly was moved (Stop Loss or Take Profit);
- calculate pointer value in points;
- display the obtained value in a relevant cell of the panel;
- calculate value of all interconnected pointers on the panel;
- change value of radio buttons, if necessary.

The first three points we perform in the event handling function, and for the last points we call the "manual" editing function of the relevant field. And, certainly, after handling the event we remove highlighting of the line.

```
//+------------------------------------------------------------------+
//| Function of moving horizontal lines                              |
//+------------------------------------------------------------------+
bool CTradePanel::DragLine(string name)
  {
   if(name==BuySL.Name())
     {
      StopLoss_pips.Text(DoubleToString(MathAbs(BuySL.Price(0)-SymbolInfoDouble(_Symbol,SYMBOL_ASK))/_Point,0));
      SLPipsEndEdit();
      BuySL.Selected(false);
      return true;
     }
   if(name==SellSL.Name())
     {
      StopLoss_pips.Text(DoubleToString(MathAbs(SellSL.Price(0)-SymbolInfoDouble(_Symbol,SYMBOL_BID))/_Point,0));
      SLPipsEndEdit();
      SellSL.Selected(false);
      return true;
     }
   if(name==BuyTP.Name())
     {
      TakeProfit_pips.Text(DoubleToString(MathAbs(BuyTP.Price(0)-SymbolInfoDouble(_Symbol,SYMBOL_ASK))/_Point,0));
      TPPipsEndEdit();
      BuyTP.Selected(false);
      return true;
     }
   if(name==SellTP.Name())
     {
      TakeProfit_pips.Text(DoubleToString(MathAbs(SellTP.Price(0)-SymbolInfoDouble(_Symbol,SYMBOL_BID))/_Point,0));
      TPPipsEndEdit();
      SellTP.Selected(false);
      return true;
     }
   return false;
  }
```

### 6\. Save current parameters after restart

I would like to remind you that after re-starting the program a user probably wouldn't enjoy entering all values on the panel once again. And frequently the majority would feel frustrated having to constantly pull the panel to the chart area that is convenient for a user. Possibly, it would be OK for someone if this action was forced only when the terminal is restarted. Don't forget that restarting the program occurs after a simple change of the chart's time frame. It happens a lot more frequently. Furthermore, many trading systems require analyzing charts on several time frames. Therefore, it is essential to save the states of radio buttons and check boxes, and also values of all fields that are manually entered by a user. And, of course, the panel must memorize the state and the position of a window.

As for the last action, it is already implemented in a mother class. It simply remains to read the saved information when starting the program.

As per editable fields and button states, certain efforts will be required here. Let me tell you right away though that the most load of work is already completed by developers, and I am eternally grateful to them for it.

I won't go into details about the class inheritance, but will say that starting from the "father" of the CObject class, all inherited classes have Save and Load functions. And our CTradePanel class has inherited the call of function for saving all enabled objects from its parent class during the class deinitialization. However, we have an unpleasant surprise here — the CEdit and CBmpButton classes have inherited "empty" functions:

```
   //--- methods for working with files
   virtual bool      Save(const int file_handle)                         { return(true);   }
   virtual bool      Load(const int file_handle)                         { return(true);   }
```

Therefore, we should re-write these functions for those objects whose data we wish to save. For such purpose we create two new classes — CEdit\_new and CBmpButton\_new that will be successors of the CEdit and CBmpButton class, respectively. We will specify functions of saving and reading data there.

```
class CEdit_new : public CEdit
  {
public:
                     CEdit_new(void){};
                    ~CEdit_new(void){};
   virtual bool      Save(const int file_handle)
     {
      if(file_handle==INVALID_HANDLE)
        {
         return false;
        }
      string text=Text();
      FileWriteInteger(file_handle,StringLen(text));
      return(FileWriteString(file_handle,text)>0);
     }
   virtual bool      Load(const int file_handle)
     {
      if(file_handle==INVALID_HANDLE)
        {
         return false;
        }
      int size=FileReadInteger(file_handle);
      string text=FileReadString(file_handle,size);
      return(Text(text));
     }

  };

class CBmpButton_new : public CBmpButton
  {
public:
                     CBmpButton_new(void){};
                    ~CBmpButton_new(void){};
   virtual bool      Save(const int file_handle)
    {
     if(file_handle==INVALID_HANDLE)
        {
         return false;
        }
      return(FileWriteInteger(file_handle,Pressed()));
     }
   virtual bool      Load(const int file_handle)
     {
      if(file_handle==INVALID_HANDLE)
        {
         return false;
        }
      return(Pressed((bool)FileReadInteger(file_handle)));
     }
  };
```

And, certainly, we will change the types of saved objects to the new ones.

```
   CEdit_new         Lots;                            // Display volume of next order
   CEdit_new         StopLoss_pips;                   // Display Stop loss in pips
   CEdit_new         StopLoss_money;                  // Display Stop loss in accaunt currency
   CEdit_new         TakeProfit_pips;                 // Display Take profit in pips
   CEdit_new         TakeProfit_money;                // Display Take profit in account currency
   CEdit_new         Risk_percent;                    // Display Risk percent to equity
   CEdit_new         Risk_money;                      // Display Risk in account currency
   CBmpButton_new    StopLoss_line;                   // Check to display StopLoss Line
   CBmpButton_new    TakeProfit_line;                 // Check to display TakeProfit Line
   CBmpButton_new    StopLoss_pips_b;                 // Select Stop loss in pips
   CBmpButton_new    StopLoss_money_b;                // Select Stop loss in accaunt currency
   CBmpButton_new    TakeProfit_pips_b;               // Select Take profit in pips
   CBmpButton_new    TakeProfit_money_b;              // Select Take profit in account currency
   CBmpButton_new    Risk_percent_b;                  // Select Risk percent to equity
   CBmpButton_new    Risk_money_b;                    // Select Risk in account currency
```

It is not enough only to save information, you also need to read it. For this purpose we re-write the function of launching our trading panel:

```
public:
.................
   virtual bool      Run(void);
```

First, we read the saved data:

```
//+------------------------------------------------------------------+
//| Run of Trade Panel                                               |
//+------------------------------------------------------------------+
bool CTradePanel::Run(void)
  {
   IniFileLoad();
```

Then, we update the variable values:

```
   cur_lot=StringToDouble(Lots.Text());
   cur_sl_pips=(int)StringToInteger(StopLoss_pips.Text());     // Stop Loss in pips
   cur_sl_money=StringToDouble(StopLoss_money.Text());         // Stop Loss in money
   cur_tp_pips=(int)StringToInteger(TakeProfit_pips.Text());   // Take Profit in pips
   cur_tp_money=StringToDouble(TakeProfit_money.Text());       // Take Profit in money
   cur_risk_percent=StringToDouble(Risk_percent.Text());       // Risk in percent
   cur_risk_money=StringToDouble(Risk_money.Text());           // Risk in money
   RiskByValue=true;
```

And, finally we call functions to handle the pressing on check boxes that actualize states of Stop Loss and Take Profit levels:

```
   StopLossLineClick();
   TakeProfitLineClick();
   return(CAppDialog::Run());
  }
```

### 7\. Clean up

We have done a great deal of work and hopefully users will remain pleased. But there comes time when for various reasons they switch off the program. And before we leave, we have to clean up after ourselves: remove all objects that we have created from the chart, and keep those objects created by a user or third party programs.

At the deinitialization of the program, the Deinit event is generated, and that calls the OnDeinit  function to indicate the reason of deinitialization. Therefore, we must call the function of deinitializing our class from the indicated function of the main program:

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   TradePanel.Destroy(reason);
   return;
  }
```

This function should be declared in the public block of our class:

```
public:
.............
   virtual void      Destroy(const int reason);
```

In this function's body, we will delete horizontal lines from the chart and call the deinitialization function of the parent class that will keep all necessary information and remove the trading panel's objects from the chart.

```
//+------------------------------------------------------------------+
//| Application deinitialization function                            |
//+------------------------------------------------------------------+
void CTradePanel::Destroy(const int reason)
  {
   BuySL.Delete();
   SellSL.Delete();
   BuyTP.Delete();
   SellTP.Delete();
   CAppDialog::Destroy(reason);
   return;
  }
```

### Conclusion

Dear readers, colleagues and friends!

I truly hope that you read my article until the end and found it useful.

I tried to share my experience of creating trading panels and offer you a ready tool for operating on the market.

I would appreciate if you would send me your ideas and suggestions with what you wish to see on our trading panel. On my behalf, I promise to fulfill the most interesting ideas and cover them in my future articles.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2281](https://www.mql5.com/ru/articles/2281)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2281.zip "Download all attachments in the single ZIP archive")

[tradepanel.ex5](https://www.mql5.com/en/articles/download/2281/tradepanel.ex5 "Download tradepanel.ex5")(319.32 KB)

[tradepanel.mq5](https://www.mql5.com/en/articles/download/2281/tradepanel.mq5 "Download tradepanel.mq5")(56.85 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/89205)**
(94)


![SkyWalkerFX0305](https://c.mql5.com/avatar/avatar_na2.png)

**[SkyWalkerFX0305](https://www.mql5.com/en/users/skywalkerfx0305)**
\|
31 Jul 2023 at 11:24

Thank you for the article.

I will modify the program to increase the panel size.

"y2" in the Create part of panel creation is set to a larger value, but it does not grow like the image.

Do you know how to resize it?

![Xiang Zeng Chen](https://c.mql5.com/avatar/avatar_na2.png)

**[Xiang Zeng Chen](https://www.mql5.com/en/users/chinafxer)**
\|
23 Aug 2023 at 19:14

The function "Close all profit orders" and "Close all loss orders" should be added, enter the [stop loss](https://www.metatrader5.com/en/terminal/help/trading/general_concept "User Guide: Types of orders") percentage and the amount and number of pips can be displayed.


![sydongjsh](https://c.mql5.com/avatar/avatar_na2.png)

**[sydongjsh](https://www.mql5.com/en/users/sydongjsh)**
\|
15 Mar 2024 at 11:44

How does this trading panel go to the bottom right corner of the screen when minimised? How to change the code? Please advise, thanks!


![kazu](https://c.mql5.com/avatar/avatar_na2.png)

**[kazu](https://www.mql5.com/en/users/kazusa0123)**
\|
24 Oct 2024 at 16:03

Even when I press the button to close all positions, only one position gets closed if I have multiple positions. Also, when I hold multiple positions, the profit and loss are only reflected for the first position (in the bottom section).


![Cristobal Hidalgo Soriano](https://c.mql5.com/avatar/2023/7/64a4ab58-556d.jpg)

**[Cristobal Hidalgo Soriano](https://www.mql5.com/en/users/mrcuervo)**
\|
29 Dec 2024 at 21:33

Hello

yes I saw it as you say good job

but I see it lacks to be able to do the BT with it.

can you do something in this sense ?

Regards

![How to create bots for Telegram in MQL5](https://c.mql5.com/2/22/telegram-avatar.png)[How to create bots for Telegram in MQL5](https://www.mql5.com/en/articles/2355)

This article contains step-by-step instructions for creating bots for Telegram in MQL5. This information may prove useful for users who wish to synchronize their trading robot with a mobile device. There are samples of bots in the article that provide trading signals, search for information on websites, send information about the account balance, quotes and screenshots of charts to you smart phone.

![Universal Expert Advisor: A Custom Trailing Stop (Part 6)](https://c.mql5.com/2/23/63vov3f0bdp_1sl2.png)[Universal Expert Advisor: A Custom Trailing Stop (Part 6)](https://www.mql5.com/en/articles/2411)

The sixth part of the article about the universal Expert Advisor describes the use of the trailing stop feature. The article will guide you through how to create a custom trailing stop module using unified rules, as well as how to add it to the trading engine so that it would automatically manage positions.

![Creating a trading robot for Moscow Exchange. Where to start?](https://c.mql5.com/2/23/expert-moex-avatar.png)[Creating a trading robot for Moscow Exchange. Where to start?](https://www.mql5.com/en/articles/2513)

Many traders on Moscow Exchange would like to automate their trading algorithms, but they do not know where to start. The MQL5 language offers a huge range of trading functions, and it additionally provides ready classes that help users to make their first steps in algo trading.

![Graphical Interfaces VI: the Slider and the Dual Slider Controls (Chapter 2)](https://c.mql5.com/2/23/avad1j__1.png)[Graphical Interfaces VI: the Slider and the Dual Slider Controls (Chapter 2)](https://www.mql5.com/en/articles/2468)

In the previous article, we have enriched our library with four controls frequently used in graphical interfaces: checkbox, edit, edit with checkbox and check combobox. The second chapter of the sixth part will be dedicated to the slider and the dual slider controls.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=cxtrmoussewsvvaescwqizmkppjhiisx&ssn=1769253377648480430&ssn_dr=0&ssn_sr=0&fv_date=1769253377&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2281&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20an%20assistant%20in%20manual%20trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925337762432961&fz_uniq=5083453116424133633&sv=2552)

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