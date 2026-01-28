---
title: Quick Manual Trading Toolkit: Basic Functionality
url: https://www.mql5.com/en/articles/7892
categories: Trading, Expert Advisors
relevance_score: -5
scraped_at: 2026-01-24T14:18:23.609509
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/7892&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083478151788502139)

MetaTrader 5 / Trading


### Contents

- [Introduction](https://www.mql5.com/en/articles/7892#intro)
- [The concept of the toolkit](https://www.mql5.com/en/articles/7892#concept)
- [Implementation of basic functionality](https://www.mql5.com/en/articles/7892#prog)
- [Conclusion](https://www.mql5.com/en/articles/7892#final)

### Introduction

Today, many traders switch to automated trading systems which can require additional setup or can be fully automated and ready to use. However, there is a considerable part of traders who prefer trading manually, in the old fashioned way. They prefer to have human expert judgment when making decisions in each specific trading situation. Sometimes such situations develop rapidly, so the trader needs to react quickly. Also, some trading styles (for example, scalping), require accuracy in market entry timing. This is where additional tools may come in handy: they can provide the fastest possible implementation of desired actions, such as market entry or exit. Therefore, I decided to implement basic functionality for quick manual trading needs.

### The concept of the toolkit

First, it is necessary to determine a set of basic actions which may be needed in manual trading. Based on this, we will develop a tool that will enable efficient and fast execution of appropriate actions. Manual trading has an underlying trading system, which implies one of the following market entry methods: using market or pending orders. Therefore, the main criteria for the toolkit is the ability to work with these two order types. We can also select tasks which can be performed by a trader in the process of trading — tools can assist in reducing the time and number of actions required to execute these tasks.

![](https://c.mql5.com/2/40/001.jpg)

Fig. 1 The main toolkit window

Figure 1 shows two categories: Market Order and Pending Order. I have also selected three basic tasks which sometimes require quick execution, but which cannot be performed in one action. Many applications, including the MetaTrader 5 terminal, have a basic set of hotkeys enabling quick execution of certain commands or actions. Our application will take this fact into account. A hotkey is indicated in brackets — when pressed, either the specified action is performed, or - when working with orders - an appropriate window opens. It will also be possible to perform actions using the mouse. For example, it will be possible to close all profitable orders by pressing the C key or by clicking on "Close all profitable".

Pressing M or clicking "Market order" will open the window "Settings: Market order". The window contains options for entering data for placing market buy or sell orders.

![](https://c.mql5.com/2/40/002.jpg)

Fig. 2 Window for configuring and creating market orders.

In addition to basic setup, similar to the one available in the terminal, it will be possible to select the lot not only as a numeric value but also as a percent of account balance. This also concerns Take Profit and Stop Loss: not only in the price format, but also in points. The Buy and Sell actions can also be performed in two ways: by clicking on the appropriate button or by pressing a hotkey specified in brackets.

Press P to open the pending order setup window. Four pending order types are supported.

![](https://c.mql5.com/2/40/003.jpg)

Fig. 3 Window for configuring and creating pending orders

Similar to market orders, pending orders support lot as a numerical value or as a percent of balance, as well as selection of Take Profit and Stop Loss in price or in points.

### Implementation of tools

First, let's create an initial project structure. Open the Experts directory and create the 'Simple Trading' folder with a few files, as is shown in fig. 4 below.

![](https://c.mql5.com/2/39/004.jpg)

Fig. 4 Project file structure

Let's consider the purpose of the created files:

- **SimpleTrading.mq5** — the file of the Expert Advisor, in which the GUI will be created and which will contain initial application settings.
- **Program.mqh** — the include file to be connected to the Expert Advisor, which will contain the **CFastTrading** class, its fields and methods. It will also contain their partial implementation.
- **MainWindow.mqh** — the include file to be connected to **Program.mqh**, which will contain the implementation methods of GUI elements.
- **Defines.mqh** —  the include file to be connected to **Program.mqh**; it contains a set of macro substitutions for GUI elements, used to implement English and Russian versions.

Firstly, open **Program.mqh**, connect the libraries required for the implementation of the interface and of trading functions, and create the **CFastTrading** class.

```
//+------------------------------------------------------------------+
//|                                                      Program.mqh |
//|                                                         Alex2356 |
//|                    https://www.mql5.com/en/users/alex2356/seller |
//+------------------------------------------------------------------+
#include <EasyAndFastGUI\WndEvents.mqh>
#include <DoEasy25\Engine.mqh>
#include "Defines.mqh"
//+------------------------------------------------------------------+
//| Enumeration for switching the interface language                 |
//+------------------------------------------------------------------+
enum LANG
{
   RUSSIAN,       // Russian
   ENGLISH        // English
};
//+------------------------------------------------------------------+
//| Class for creating an application                                |
//+------------------------------------------------------------------+
class CFastTrading : public CWndEvents
{
public:
                     CFastTrading(void);
                    ~CFastTrading(void);
   //--- Initialization/deinitialization
   void              OnInitEvent(void);
   void              OnDeinitEvent(const int reason);
   //--- Timer
   void              OnTimerEvent(void);
   //--- Chart event handler
   virtual void      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
};
//+------------------------------------------------------------------+
//| Adding GUI elements                                              |
//+------------------------------------------------------------------+
#include "MainWindow.mqh"
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CFastTrading::CFastTrading(void)
{
}
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CFastTrading::~CFastTrading(void)
{
}
//+------------------------------------------------------------------+
//| Initialization                                                    |
//+------------------------------------------------------------------+
void CFastTrading::OnInitEvent(void)
{
}
//+------------------------------------------------------------------+
//| Deinitialization                                                 |
//+------------------------------------------------------------------+
void CFastTrading::OnDeinitEvent(const int reason)
{
//--- Remove the interface
   CWndEvents::Destroy();
}
//+------------------------------------------------------------------+
//| Timer                                                            |
//+------------------------------------------------------------------+
void CFastTrading::OnTimerEvent(void)
{
   CWndEvents::OnTimerEvent();
//---
}
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void CFastTrading::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
{
}
//+------------------------------------------------------------------+
```

Now, open the EA file **SimpleTrading.mq5**, connect to it **Program.mqh** and create an instance if the newly created class. Also, set input parameters, which include the following:

- Base FontSize — the base font size of the application.
- Caption Color — caption color of the main application window.
- Back color — background color.
- Interface language — interface language.
- Magic Number — unique number for the orders created by this Expert Advisor.

```
//+------------------------------------------------------------------+
//|                                                SimpleTrading.mq5 |
//|                                                         Alex2356 |
//|                    https://www.mql5.com/en/users/alex2356/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, Alexander Fedosov"
#property link      "https://www.mql5.com/en/users/alex2356"
#property version   "1.00"
//--- Include application class
#include "Program.mqh"
//+------------------------------------------------------------------+
//| Expert Advisor input parameters                                  |
//+------------------------------------------------------------------+
input int                  Inp_BaseFont      =  10;                  // Base FontSize
input color                Caption           =  C'0,130,225';        // Caption Color
input color                Background        =  clrWhite;            // Back color
input LANG                 Language          =  ENGLISH;             // Interface language
input ulong                MagicNumber       =  1111;                // Magic Number
//---
CFastTrading program;
ulong tick_counter;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
//---
   tick_counter=GetTickCount();
//---
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   program.OnDeinitEvent(reason);
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
//---
}
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer(void)
{
   program.OnTimerEvent();
}
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int    id,
                  const long   &lparam,
                  const double &dparam,
                  const string &sparam)
{
   program.ChartEvent(id,lparam,dparam,sparam);
//---
   if(id==CHARTEVENT_CUSTOM+ON_END_CREATE_GUI)
   {
      Print("End in ",GetTickCount()-tick_counter," ms");
   }
}
//+------------------------------------------------------------------+
```

In order for the EA input parameters to be available for this class, it is necessary to create variables to which EA settings values will be assigned, and methods which will be used for implementation. Create variables in the private section of the class.

```
private:
//---
   color             m_caption_color;
   color             m_background_color;
//---
   int               m_base_font_size;
   int               m_m_edit_index;
   int               m_p_edit_index;
//---
   ulong             m_magic_number;
//---
   string            m_base_font;
//---
   LANG              m_language;
```

Create methods in the public section:

```
   //--- Caption color
   void              CaptionColor(const color clr);
   //--- Background color
   void              BackgroundColor(const color clr);
   //--- Font size
   void              FontSize(const int font_size);
   //--- Font name
   void              FontName(const string font_name);
   //--- Setting the interface language
   void              SetLanguage(const LANG lang);
   //--- Setting the magic number
   void              SetMagicNumber(ulong magic_number);
```

Here is the implementation:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::CaptionColor(const color clr)
{
   m_caption_color=clr;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::BackgroundColor(const color clr)
{
   m_background_color=clr;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::FontSize(const int font_size)
{
   m_base_font_size=font_size;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::FontName(const string font_name)
{
   m_base_font=font_name;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::SetLanguage(const LANG lang)
{
   m_language=lang;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::SetMagicNumber(const ulong magic_number)
{
   m_magic_number=magic_number;
}
```

Now, apply them in the EA initialization:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
//---
   tick_counter=GetTickCount();
//--- Initialize class variables
   program.FontSize(Inp_BaseFont);
   program.BackgroundColor(Background);
   program.CaptionColor(Caption);
   program.SetLanguage(Language);
   program.SetMagicNumber(MagicNumber);
//---
   return(INIT_SUCCEEDED);
}
```

Add the **CreateGUI()** method which will create the entire interface. As for now, it is empty. It will be filled further, when we create UI elements.

```
//--- Create the graphical interface of the program
   bool              CreateGUI(void);
```

The method can be added to the EA initialization:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
//---
   tick_counter=GetTickCount();
//--- Initialize class variables
   program.FontName("Trebuchet MS");
   program.FontSize(Inp_BaseFont);
   program.BackgroundColor(Background);
   program.CaptionColor(Caption);
   program.SetLanguage(Language);
   program.SetMagicNumber(MagicNumber);
//--- Set up the trading panel
   if(!program.CreateGUI())
   {
      Print(__FUNCTION__," > Failed to create graphical interface!");
      return(INIT_FAILED);
   }
//---
   return(INIT_SUCCEEDED);
}
```

Now, create the main application window. This can be done by adding the **CreateMainWindow()** method into the protected section of our class.

```
protected:
   //--- forms
   bool              CreateMainWindow(void);
```

Add its implementation to the **MainWindow.mqh** file and then call it in **CreateGUI()**. In this method implementation, I used the CAPTION\_NAME macro substitution, that is why it should be created in the special **Defines.mqh** file.

```
//+------------------------------------------------------------------+
//| Creates a form for orders                                        |
//+------------------------------------------------------------------+
bool CFastTrading::CreateMainWindow(void)
{
//--- Add a window pointer to the window array
   CWndContainer::AddWindow(m_main_window);
//--- Properties
   m_main_window.XSize(400);
   m_main_window.YSize(182);
//--- Coordinates
   int x=5;
   int y=20;
   m_main_window.CaptionHeight(22);
   m_main_window.IsMovable(true);
   m_main_window.CaptionColor(m_caption_color);
   m_main_window.CaptionColorLocked(m_caption_color);
   m_main_window.CaptionColorHover(m_caption_color);
   m_main_window.BackColor(m_background_color);
   m_main_window.FontSize(m_base_font_size);
   m_main_window.Font(m_base_font);
//--- Create the form
   if(!m_main_window.CreateWindow(m_chart_id,m_subwin,CAPTION_NAME,x,y))
      return(false);
//---
   return(true);
}
//+------------------------------------------------------------------+
//| Creates the graphical interface of the program                   |
//+------------------------------------------------------------------+
bool CFastTrading::CreateGUI(void)
{
//--- Create the main application window
   if(!CreateMainWindow())
      return(false);
//--- Complete GUI creation
   CWndEvents::CompletedGUI();
   return(true);
}
```

Now, create a new macro substitution.

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#include "Program.mqh"
#define CAPTION_NAME                   (m_language==RUSSIAN ? "Быстрый трейдинг" : "Fast Trading")
```

Compile the project to obtain the main application window. Now, let's add buttons (Fig.1), which will perform the described actions. To implement them, create the universal **CreateButton()** method in the protected section of our class.

```
//--- Buttons
   bool              CreateButton(CWindow &window,CButton &button,string text,color baseclr,int x_gap,int y_gap,int w_number);

```

Implement it in  **MainWindow.mqh** and apply in the main window creating method body.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateButton(CWindow &window,CButton &button,string text,color baseclr,int x_gap,int y_gap,int w_number)
{
//--- Store the window pointer
   button.MainPointer(window);
//--- Set properties before creation
   button.XSize(170);
   button.YSize(40);
   button.Font(m_base_font);
   button.FontSize(m_base_font_size);
   button.BackColor(baseclr);
   button.BackColorHover(baseclr);
   button.BackColorPressed(baseclr);
   button.BorderColor(baseclr);
   button.BorderColorHover(baseclr);
   button.BorderColorPressed(baseclr);
   button.LabelColor(clrWhite);
   button.LabelColorPressed(clrWhite);
   button.LabelColorHover(clrWhite);
   button.IsCenterText(true);
//--- Create a control element
   if(!button.CreateButton(text,x_gap,window.CaptionHeight()+y_gap))
      return(false);
//--- Add a pointer to the element to the database
   CWndContainer::AddToElementsArray(w_number,button);
   return(true);
}
//+------------------------------------------------------------------+
//| Creates a form for orders                                        |
//+------------------------------------------------------------------+
bool CFastTrading::CreateMainWindow(void)
{
//--- Add a window pointer to the window array
   CWndContainer::AddWindow(m_main_window);
//--- Properties
   m_main_window.XSize(400);
   m_main_window.YSize(182);
//--- Coordinates
   int x=5;
   int y=20;
   m_main_window.CaptionHeight(22);
   m_main_window.IsMovable(true);
   m_main_window.CaptionColor(m_caption_color);
   m_main_window.CaptionColorLocked(m_caption_color);
   m_main_window.CaptionColorHover(m_caption_color);
   m_main_window.BackColor(m_background_color);
   m_main_window.FontSize(m_base_font_size);
   m_main_window.Font(m_base_font);
   m_main_window.TooltipsButtonIsUsed(true);
//--- Create the form
   if(!m_main_window.CreateWindow(m_chart_id,m_subwin,CAPTION_NAME,x,y))
      return(false);
//---
   if(!CreateButton(m_main_window,m_order_button[0],MARKET_ORDER_NAME+"(M)",C'87,128,255',20,10,0))
      return(false);
   if(!CreateButton(m_main_window,m_order_button[1],PENDING_ORDER_NAME+"(P)",C'31,209,111',210,10,0))
      return(false);
   if(!CreateButton(m_main_window,m_order_button[2],MARKET_ORDERS_PROFIT_CLOSE+"(C)",C'87,128,255',20,60,0))
      return(false);
   if(!CreateButton(m_main_window,m_order_button[3],MARKET_ORDERS_LOSS_CLOSE+"(D)",C'87,128,255',20,110,0))
      return(false);
   if(!CreateButton(m_main_window,m_order_button[4],PEND_ORDERS_ALL_CLOSE+"(R)",C'31,209,111',210,60,0))
      return(false);
   return(true);
}
```

Macro substitutions are also used here, that is why add it in **Defines.mqh**.

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
#include "Program.mqh"
#define CAPTION_NAME                   (m_language==RUSSIAN ? "Быстрый трейдинг" : "Fast Trading System")
#define MARKET_ORDER_NAME              (m_language==RUSSIAN ? "Рыночный ордер" : "Marker Order")
#define PENDING_ORDER_NAME             (m_language==RUSSIAN ? "Отложенный ордер" : "Pending Order")
#define MARKET_ORDERS_PROFIT_CLOSE     (m_language==RUSSIAN ? "Закрыть все прибыльные" : "Close all profitable")
#define MARKET_ORDERS_LOSS_CLOSE       (m_language==RUSSIAN ? "Закрыть все убыточные" : "Close all losing")
#define PEND_ORDERS_ALL_CLOSE          (m_language==RUSSIAN ? "Закрыть все отложенные" : "Close all pending")
```

This is what we have when the English version is selected.

![](https://c.mql5.com/2/39/005.jpg)

Fig. 5 The main application window.

As mentioned earlier, the application should have two more windows for working with market and pending orders. Therefore, create them and link to the Market Order and Pending Order buttons: the relevant operations can be performed by button clicks or by pressing the M and P hotkeys. Therefore, add two methods, **CreateMarketOrdersWindow()** and **CreatePendingOrdersWindow()**, in the protected section of the class and implement them in **MainWindow.mqh**.

```
   bool              CreateMarketOrdersWindow(void);
   bool              CreatePendingOrdersWindow(void);
//+------------------------------------------------------------------+
//| Market order creation and editing window                         |
//+------------------------------------------------------------------+
bool CFastTrading::CreateMarketOrdersWindow(void)
{
//--- Add a window pointer to the window array
   CWndContainer::AddWindow(m_orders_windows[0]);
//--- Properties
   m_orders_windows[0].XSize(450);
   m_orders_windows[0].YSize(242+58);
//--- Coordinates
   int x=m_order_button[0].XGap();
   int y=m_order_button[0].YGap()+60;
//---
   color clrmain=C'87,128,255';
//---
   m_orders_windows[0].CaptionHeight(22);
   m_orders_windows[0].IsMovable(true);
   m_orders_windows[0].CaptionColor(clrmain);
   m_orders_windows[0].CaptionColorLocked(clrmain);
   m_orders_windows[0].CaptionColorHover(clrmain);
   m_orders_windows[0].BackColor(m_background_color);
   m_orders_windows[0].BorderColor(clrmain);
   m_orders_windows[0].FontSize(m_base_font_size);
   m_orders_windows[0].Font(m_base_font);
   m_orders_windows[0].WindowType(W_DIALOG);
//--- Create the form
   if(!m_orders_windows[0].CreateWindow(m_chart_id,m_subwin,CAPTION_M_ORD_NAME,x,y))
      return(false);
   return(true);
}
//+------------------------------------------------------------------+
//| Pending order creation and editing window                        |
//+------------------------------------------------------------------+
bool CFastTrading::CreatePendingOrdersWindow(void)
{
//--- Add a window pointer to the window array
   CWndContainer::AddWindow(m_orders_windows[1]);
//--- Properties
   m_orders_windows[1].XSize(600);
   m_orders_windows[1].YSize(580);
//--- Coordinates
   int x=m_order_button[0].XGap();
   int y=m_order_button[0].YGap()+60;
//---
   color clrmain=C'31,209,111';
//---
   m_orders_windows[1].CaptionHeight(22);
   m_orders_windows[1].IsMovable(true);
   m_orders_windows[1].CaptionColor(clrmain);
   m_orders_windows[1].CaptionColorLocked(clrmain);
   m_orders_windows[1].CaptionColorHover(clrmain);
   m_orders_windows[1].BackColor(m_background_color);
   m_orders_windows[1].BorderColor(clrmain);
   m_orders_windows[1].FontSize(m_base_font_size);
   m_orders_windows[1].Font(m_base_font);
   m_orders_windows[1].WindowType(W_DIALOG);
//--- Create the form
   if(!m_orders_windows[1].CreateWindow(m_chart_id,m_subwin,CAPTION_P_ORD_NAME,x,y))
      return(false);
   return(true);
}
```

Macro substitutions are used in these two windows, so add them to the appropriate file:

```
#define CAPTION_M_ORD_NAME             (m_language==RUSSIAN ? "Настройка: Рыночный Ордер" : "Setting: Market Order")
#define CAPTION_P_ORD_NAME             (m_language==RUSSIAN ? "Настройка: Отложенный Ордер" : "Setting: Pending Order")
```

Now, call the newly created windows in the **CreateGUI()** basic method, which created the application interface:

```
//+------------------------------------------------------------------+
//| Creates the graphical interface of the program                   |
//+------------------------------------------------------------------+
bool CFastTrading::CreateGUI(void)
{
//--- Create the main application window
   if(!CreateMainWindow())
      return(false);
   if(!CreateMarketOrdersWindow())
      return(false);
   if(!CreatePendingOrdersWindow())
      return(false);
//--- Complete GUI creation
   CWndEvents::CompletedGUI();
   return(true);
}
```

Since these created windows are dialogs, they will not be shown at application startup. We need to create a mechanism to display them. This will be opening upon a click on Market order / Pending order button or upon a hotkey press. Find the **OnEvent()** method body in the base class and write the required conditions.

```
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void CFastTrading::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
{
//--- Pressing the button event
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON)
   {
      //---
      if(lparam==m_order_button[0].Id())
         m_orders_windows[0].OpenWindow();
      //---
      if(lparam==m_order_button[1].Id())
         m_orders_windows[1].OpenWindow();
   }
//--- Key press event
   if(id==CHARTEVENT_KEYDOWN)
   {
      //--- Opening a market order window
      if(lparam==KEY_M)
      {
         if(m_orders_windows[0].IsVisible())
         {
            m_orders_windows[0].CloseDialogBox();
         }
         else
         {
            if(m_orders_windows[1].IsVisible())
            {
               m_orders_windows[1].CloseDialogBox();
            }
            //---
            m_orders_windows[0].OpenWindow();
         }
      }
      //--- Opening a pending order window
      if(lparam==KEY_P)
      {
         if(m_orders_windows[1].IsVisible())
         {
            m_orders_windows[1].CloseDialogBox();
         }
         else
         {
            if(m_orders_windows[0].IsVisible())
            {
               m_orders_windows[0].CloseDialogBox();
            }
            //---
            m_orders_windows[1].OpenWindow();
         }
      }
   }
}
```

Now compile the project and try to open dialog boxes using hotkeys. The result should be as is shown in Figure 6:

![](https://c.mql5.com/2/40/006.gif)

Fig. 6 Opening and switching windows using hotkeys.

As you can see, there is no need to close a dialog window before opening another one. It simply switches to another window — this allows saving time and switching between market and pending orders in just one click. Another convenient small touch is the ability to close dialogs using the _Esc_ key. When testing how the windows open, I was about to use this key a few times, to close the windows. So, let's add the code to the Event section:

```
      //--- Exiting the order placing window
      if(lparam==KEY_ESC)
      {
         if(m_orders_windows[0].IsVisible())
         {
            m_orders_windows[0].CloseDialogBox();
         }
         else if(m_orders_windows[1].IsVisible())
         {
            m_orders_windows[1].CloseDialogBox();
         }
      }
```

Now, let's continue working with the created windows, starting with the Market Orders window. According to Figure 2, we need to create two order management blocks - buying and selling. First, let's create two interface elements - frames - using the new **CreateFrame()** method.

```
bool              CreateFrame(CWindow &window,CFrame &frame,const int x_gap,const int y_gap,string caption,int w_number);
```

Its implementation is as follows:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateFrame(CWindow &window,CFrame &frame,const int x_gap,const int y_gap,string caption,int w_number)
{
//--- Store the pointer to the main control
   frame.MainPointer(window);
//---
   color clrmain=clrNONE;
   if(caption==BUY_ORDER)
      clrmain=C'88,212,210';
   else if(caption==SELL_ORDER)
      clrmain=C'236,85,79';
//---
   frame.YSize(110);
   frame.LabelColor(clrmain);
   frame.BorderColor(clrmain);
   frame.BackColor(m_background_color);
   frame.GetTextLabelPointer().BackColor(m_background_color);
   frame.Font(m_base_font);
   frame.FontSize(m_base_font_size);
   frame.AutoXResizeMode(true);
   frame.AutoXResizeRightOffset(10);
//--- Create a control element
   if(!frame.CreateFrame(caption,x_gap,window.CaptionHeight()+y_gap))
      return(false);
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(w_number,frame);
   return(true);
}
```

In the frame creation implementation, we have two new macro substitutions for frame headers, so add them to **Defines.mqh**:

```
#define BUY_ORDER                      (m_language==RUSSIAN ? "Buy-ордер" : "Buy-order")
#define SELL_ORDER                     (m_language==RUSSIAN ? "Sell-ордер" : "Sell-order")
```

To apply this method, go the body end of the **CreateMarketOrdersWindow()** method which creates market orders and add the following:

```
...
//--- Create the form
   if(!m_orders_windows[0].CreateWindow(m_chart_id,m_subwin,CAPTION_M_ORD_NAME,x,y))
      return(false);
//--- BUY BLOCK
   if(!CreateFrame(m_orders_windows[0],m_frame[0],10,20,BUY_ORDER,1))
      return(false);
//--- SELL BLOCK
   if(!CreateFrame(m_orders_windows[0],m_frame[1],10,160,SELL_ORDER,1))
      return(false);
   return(true);
}
```

Check the result:

![](https://c.mql5.com/2/40/007.jpg)

Fig. 7 Market order blocks.

Each of the blocks in Figure 7 will contain 4 main categories of UI elements:

- Text headers. These are Lot, Take Profit, Stop Loss.
- Toggle buttons. For a lot, this is Lot/% of Deposit; for Take Profit and Stop Loss - price or point mode.
- Input fields. Lot, Take Profit and Stop Loss.
- Action button. Sell or Buy.

Let's proceed with the step-by-step implementation of each category. Create the universal method **CreateLabel()** for text headers.

```
//+------------------------------------------------------------------+
//| Creates the text label                                           |
//+------------------------------------------------------------------+
bool CFastTrading::CreateLabel(CWindow &window,CTextLabel &text_label,const int x_gap,const int y_gap,string label_text,int w_number)
{
//--- Store the window pointer
   text_label.MainPointer(window);
//---
   text_label.Font(m_base_font);
   text_label.FontSize(m_base_font_size);
   text_label.XSize(80);
   text_label.BackColor(m_background_color);
   text_label.IsCenterText(true);
//--- Creating a label
   if(!text_label.CreateTextLabel(label_text,x_gap,window.CaptionHeight()+y_gap))
      return(false);
//--- Add a pointer to the element to the database
   CWndContainer::AddToElementsArray(w_number,text_label);
   return(true);
}
```

We have earlier supplemented the market order creation method. Now, add the following taking into account the current implementation.

```
//--- BUY BLOCK
   if(!CreateFrame(m_orders_windows[0],m_frame[0],10,20,BUY_ORDER,1))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[0],20,30,LOT,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[1],20+80+20,30,TP,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[2],20+(80+20)*2,30,SL,1))
      return(false);
//--- SELL BLOCK
   if(!CreateFrame(m_orders_windows[0],m_frame[1],10,160,SELL_ORDER,1))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[3],20,170,LOT,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[4],20+80+20,170,TP,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[5],20+(80+20)*2,170,SL,1))
      return(false);
   return(true);
}
```

Here are three new macro substitutions for Lot, Take Profit and Stop Loss. Add names in two languages:

```
#define LOT                            (m_language==RUSSIAN ? "Лот" : "Lot")
#define TP                             (m_language==RUSSIAN ? "Тейк профит" : "Take Profit")
#define SL                             (m_language==RUSSIAN ? "Стоп лосс" : "Stop Loss")
```

The next category includes toggle buttons. The specially created method **CreateSwitchButton()** will be used for them.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateSwitchButton(CWindow &window,CButton &button,string text,int x_gap,int y_gap,int w_number)
{
//--- Store the window pointer
   button.MainPointer(window);
   color baseclr=clrSlateBlue;
   color pressclr=clrIndigo;
//--- Set properties before creation
   button.XSize(80);
   button.YSize(24);
   button.Font(m_base_font);
   button.FontSize(m_base_font_size);
   button.BackColor(baseclr);
   button.BackColorHover(baseclr);
   button.BackColorPressed(pressclr);
   button.BorderColor(baseclr);
   button.BorderColorHover(baseclr);
   button.BorderColorPressed(pressclr);
   button.LabelColor(clrWhite);
   button.LabelColorPressed(clrWhite);
   button.LabelColorHover(clrWhite);
   button.IsCenterText(true);
   button.TwoState(true);
//--- Create a control element
   if(!button.CreateButton(text,x_gap,window.CaptionHeight()+y_gap))
      return(false);
//--- Add a pointer to the element to the database
   CWndContainer::AddToElementsArray(w_number,button);
   return(true);
}
```

Apply it for both blocks in the **CreateMarketWindow()** method:

```
...
//--- BUY BLOCK
   if(!CreateFrame(m_orders_windows[0],m_frame[0],10,20,BUY_ORDER,1))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[0],20,30,LOT,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[1],20+80+20,30,TP,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[2],20+(80+20)*2,30,SL,1))
      return(false);
//--- Toggle buttons
   for(int i=0; i<3; i++)
      if(!CreateSwitchButton(m_orders_windows[0],m_switch_button[i],"-",20+(80+20)*i,60,1))
         return(false);
//--- SELL BLOCK
   if(!CreateFrame(m_orders_windows[0],m_frame[1],10,160,SELL_ORDER,1))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[3],20,170,LOT,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[4],20+80+20,170,TP,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[5],20+(80+20)*2,170,SL,1))
      return(false);
//--- Toggle buttons
   for(int i=3; i<6; i++)
      if(!CreateSwitchButton(m_orders_windows[0],m_switch_button[i],"-",20+(80+20)*(i-3),170+30,1))
         return(false);
   return(true);
}
```

Special attention should be paid to the next category, related to Input Fields, as the elements need to have certain usage restrictions. For example, values for the Lot input field must be limited by the current symbol specification, as well as by the specifics of a trading account. When editing this field, the following parameters should be taken into account: minimum and maximum lot size, minimum change step. Based on this, create **CreateLotEdit()** and set its input field accordingly:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateLotEdit(CWindow &window,CTextEdit &text_edit,const int x_gap,const int y_gap,int w_number)
{
//--- Store the pointer to the main control
   text_edit.MainPointer(window);
//--- Properties
   text_edit.XSize(80);
   text_edit.YSize(24);
   text_edit.Font(m_base_font);
   text_edit.FontSize(m_base_font_size);
   text_edit.MaxValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX));
   text_edit.StepValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP));
   text_edit.MinValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN));
   text_edit.SpinEditMode(true);
   text_edit.SetDigits(2);
   text_edit.GetTextBoxPointer().XGap(1);
   text_edit.GetTextBoxPointer().XSize(80);
//--- Create a control element
   if(!text_edit.CreateTextEdit("",x_gap,window.CaptionHeight()+y_gap))
      return(false);
   text_edit.SetValue(string(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN)));
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(w_number,text_edit);
   return(true);
}
```

Also, create input fields for Stop Loss and Take Profit. All of the above should be added to the market order creation window.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateTakeProfitEdit(CWindow &window,CTextEdit &text_edit,const int x_gap,const int y_gap,int w_number)
{
//--- Store the pointer to the main control
   text_edit.MainPointer(window);
//--- Properties
   text_edit.XSize(80);
   text_edit.YSize(24);
   text_edit.Font(m_base_font);
   text_edit.FontSize(m_base_font_size);
   text_edit.MaxValue(9999);
   text_edit.StepValue(1);
   text_edit.MinValue(0);
   text_edit.SpinEditMode(true);
   text_edit.GetTextBoxPointer().XGap(1);
   text_edit.GetTextBoxPointer().XSize(80);
//--- Create a control element
   if(!text_edit.CreateTextEdit("",x_gap,window.CaptionHeight()+y_gap))
      return(false);
   text_edit.SetValue(string(150));
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(w_number,text_edit);
   return(true);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateStopLossEdit(CWindow &window,CTextEdit &text_edit,const int x_gap,const int y_gap,int w_number)
{
//--- Store the pointer to the main control
   text_edit.MainPointer(window);
//--- Properties
   text_edit.XSize(80);
   text_edit.YSize(24);
   text_edit.Font(m_base_font);
   text_edit.FontSize(m_base_font_size);
   text_edit.MaxValue(9999);
   text_edit.StepValue(1);
   text_edit.MinValue(0);
   text_edit.SpinEditMode(true);
   text_edit.GetTextBoxPointer().XGap(1);
   text_edit.GetTextBoxPointer().XSize(80);
//--- Create a control element
   if(!text_edit.CreateTextEdit("",x_gap,window.CaptionHeight()+y_gap))
      return(false);
   text_edit.SetValue(string(150));
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(w_number,text_edit);
   return(true);
}
```

And the last category of UI elements in this window contains two buttons, Buy and Sell. Button adding methods are **CreateBuyButton()** and **CreateSellButton()**.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateBuyButton(CWindow &window,CButton &button,string text,int x_gap,int y_gap,int w_number)
{
//--- Store the window pointer
   button.MainPointer(window);
   color baseclr=C'88,212,210';
//--- Set properties before creation
   button.XSize(120);
   button.YSize(40);
   button.Font(m_base_font);
   button.FontSize(m_base_font_size);
   button.BackColor(baseclr);
   button.BackColorHover(baseclr);
   button.BackColorPressed(baseclr);
   button.BorderColor(baseclr);
   button.BorderColorHover(baseclr);
   button.BorderColorPressed(baseclr);
   button.LabelColor(clrWhite);
   button.LabelColorPressed(clrWhite);
   button.LabelColorHover(clrWhite);
   button.IsCenterText(true);
//--- Create a control element
   if(!button.CreateButton(text,x_gap,y_gap))
      return(false);
//--- Add a pointer to the element to the database
   CWndContainer::AddToElementsArray(w_number,button);
   return(true);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateSellButton(CWindow &window,CButton &button,string text,int x_gap,int y_gap,int w_number)
{
//--- Store the window pointer
   button.MainPointer(window);
   color baseclr=C'236,85,79';
//--- Set properties before creation
   button.XSize(120);
   button.YSize(40);
   button.Font(m_base_font);
   button.FontSize(m_base_font_size);
   button.BackColor(baseclr);
   button.BackColorHover(baseclr);
   button.BackColorPressed(baseclr);
   button.BorderColor(baseclr);
   button.BorderColorHover(baseclr);
   button.BorderColorPressed(baseclr);
   button.LabelColor(clrWhite);
   button.LabelColorPressed(clrWhite);
   button.LabelColorHover(clrWhite);
   button.IsCenterText(true);
//--- Create a control element
   if(!button.CreateButton(text,x_gap,y_gap))
      return(false);
//--- Add a pointer to the element to the database
   CWndContainer::AddToElementsArray(w_number,button);
   return(true);
}
```

By adding two buttons, we complete the implementation of the **CreateMarketWindow()** method:

```
//+------------------------------------------------------------------+
//| Market order creation and editing window                         |
//+------------------------------------------------------------------+
bool CFastTrading::CreateMarketOrdersWindow(void)
{
//--- Add a window pointer to the window array
   CWndContainer::AddWindow(m_orders_windows[0]);
//--- Properties
   m_orders_windows[0].XSize(450);
   m_orders_windows[0].YSize(242+58);
//--- Coordinates
   int x=m_order_button[0].XGap();
   int y=m_order_button[0].YGap()+60;
//---
   color clrmain=C'87,128,255';
//---
   m_orders_windows[0].CaptionHeight(22);
   m_orders_windows[0].IsMovable(true);
   m_orders_windows[0].CaptionColor(clrmain);
   m_orders_windows[0].CaptionColorLocked(clrmain);
   m_orders_windows[0].CaptionColorHover(clrmain);
   m_orders_windows[0].BackColor(m_background_color);
   m_orders_windows[0].BorderColor(clrmain);
   m_orders_windows[0].FontSize(m_base_font_size);
   m_orders_windows[0].Font(m_base_font);
   m_orders_windows[0].WindowType(W_DIALOG);
//--- Create the form
   if(!m_orders_windows[0].CreateWindow(m_chart_id,m_subwin,CAPTION_M_ORD_NAME,x,y))
      return(false);
//--- BUY BLOCK
   if(!CreateFrame(m_orders_windows[0],m_frame[0],10,20,BUY_ORDER,1))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[0],20,30,LOT,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[1],20+80+20,30,TP,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[2],20+(80+20)*2,30,SL,1))
      return(false);
//--- Toggle buttons
   for(int i=0; i<3; i++)
      if(!CreateSwitchButton(m_orders_windows[0],m_switch_button[i],"-",20+(80+20)*i,60,1))
         return(false);
//--- Edits
   if(!CreateLotEdit(m_orders_windows[0],m_lot_edit[0],20,60+34,1))
      return(false);
   if(!CreateTakeProfitEdit(m_orders_windows[0],m_tp_edit[0],20+(80+20),60+34,1))
      return(false);
   if(!CreateStopLossEdit(m_orders_windows[0],m_sl_edit[0],20+(80+20)*2,60+34,1))
      return(false);
//--- The Buy button
   if(!CreateBuyButton(m_orders_windows[0],m_buy_execute,BUY+"(B)",m_orders_windows[0].XSize()-(120+20),103,1))
      return(false);
//--- SELL BLOCK
   if(!CreateFrame(m_orders_windows[0],m_frame[1],10,160,SELL_ORDER,1))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[3],20,170,LOT,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[4],20+80+20,170,TP,1))
      return(false);
   if(!CreateLabel(m_orders_windows[0],m_m_text_labels[5],20+(80+20)*2,170,SL,1))
      return(false);
//--- Toggle buttons
   for(int i=3; i<6; i++)
      if(!CreateSwitchButton(m_orders_windows[0],m_switch_button[i],"-",20+(80+20)*(i-3),170+30,1))
         return(false);
//--- Edits
   if(!CreateLotEdit(m_orders_windows[0],m_lot_edit[1],20,170+30+35,1))
      return(false);
   if(!CreateTakeProfitEdit(m_orders_windows[0],m_tp_edit[1],20+80+20,170+30+35,1))
      return(false);
   if(!CreateStopLossEdit(m_orders_windows[0],m_sl_edit[1],20+(80+20)*2,170+30+35,1))
      return(false);
//--- The Sell button
   if(!CreateSellButton(m_orders_windows[0],m_sell_execute,SELL+"(S)",m_orders_windows[0].XSize()-(120+20),242,1))
      return(false);
   return(true);
}
```

Do not forget to add two new macro substitutions:

```
#define BUY                            (m_language==RUSSIAN ? "Купить" : "Buy")
#define SELL                           (m_language==RUSSIAN ? "Продать" : "Sell")
```

Compile the project at this stage, and check the result:

![](https://c.mql5.com/2/40/008.jpg)

Fig. 8 The fill interface of the market order creation window.

It is just a template for now. The following tasks should be implemented in order to further develop the template:

- Enliven the toggle buttons.
- Change input field properties depending on the button states.
- Implement placing of market orders according to the switch mode and input data values.

Before starting to implement the button switching mechanism, which will change the name, it is necessary to set their default values. Figure 8 shows that now the buttons only have dashes. To set the name, add the **SetButtonParam()** method. The same method will be used later to switch the names.

```
//+------------------------------------------------------------------+
//| Setting the button text                                          |
//+------------------------------------------------------------------+
void CFastTrading::SetButtonParam(CButton &button,string text)
{
   button.LabelText(text);
   button.Update(true);
}
```

Go to the event handler and add the Event section upon the interface creation completion. There, set names for the switch buttons using the **SetButtonParam()** method.

```
//--- UI creation completion
   if(id==CHARTEVENT_CUSTOM+ON_END_CREATE_GUI)
   {
      //---
      SetButtonParam(m_switch_button[0],LOT);
      SetButtonParam(m_switch_button[1],POINTS);
      SetButtonParam(m_switch_button[2],POINTS);
      SetButtonParam(m_switch_button[3],LOT);
      SetButtonParam(m_switch_button[4],POINTS);
      SetButtonParam(m_switch_button[5],POINTS);
   }
```

Here we have one more macro substitution for the button names, add it to **Defines.mqh**:

```
#define POINTS                         (m_language==RUSSIAN ? "Пункты" : "Points")
```

Preparation for creating the switching mechanism is ready. The **ButtonSwitch()** function will track the button state (pressed/unpressed) and change the button name accordingly.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::ButtonSwitch(CButton &button,long lparam,string state1,string state2)
{
   if(lparam==button.Id())
   {
      if(!button.IsPressed())
         SetButtonParam(button,state1);
      else
         SetButtonParam(button,state2);
   }
}
```

Since the name is switched by the button click event, call the created method in the event handler section.

```
//--- Pressing the button event
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON)
   {
      //---
      if(lparam==m_order_button[0].Id())
         m_orders_windows[0].OpenWindow();
      //---
      if(lparam==m_order_button[1].Id())
         m_orders_windows[1].OpenWindow();
      //---
      ButtonSwitch(m_switch_button[0],lparam,LOT,PERC_DEPO);
      ButtonSwitch(m_switch_button[1],lparam,POINTS,PRICE);
      ButtonSwitch(m_switch_button[2],lparam,POINTS,PRICE);
      ButtonSwitch(m_switch_button[3],lparam,LOT,PERC_DEPO);
      ButtonSwitch(m_switch_button[4],lparam,POINTS,PRICE);
      ButtonSwitch(m_switch_button[5],lparam,POINTS,PRICE);
   }
```

Also describe new macro substitutions in both languages:

```
#define PERC_DEPO                      (m_language==RUSSIAN ? "% Депозит" : "% Deposit")
#define PRICE                          (m_language==RUSSIAN ? "Цена" : "Price")
```

Compile the project again: button names now change upon clicks:

![](https://c.mql5.com/2/40/009.gif)

Fig. 9 Changing the names of the toggle buttons.

Let's move on. The next task is to change the properties of the input fields depending on the state of the corresponding toggle button. To do this, add three new methods for each input field: **LotMarketSwitch(), TakeMarketSwitch(), StopMarketSwitch()**. For the first method, lot value will be switched to percent of the account balance.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::LotMarketSwitch(long lparam)
{
   for(int i=0; i<2; i++)
   {
      if(lparam==m_switch_button[i*3].Id())
      {
         if(m_switch_button[i*3].IsPressed())
         {
            m_lot_edit[i].SetDigits(0);
            m_lot_edit[i].StepValue(1);
            m_lot_edit[i].MaxValue(100);
            m_lot_edit[i].MinValue(1);
            m_lot_edit[i].SetValue(string(2));
            m_lot_edit[i].GetTextBoxPointer().Update(true);
         }
         else
         {
            m_lot_edit[i].SetDigits(2);
            m_lot_edit[i].StepValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP));
            m_lot_edit[i].MinValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN));
            m_lot_edit[i].MaxValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX));
            m_lot_edit[i].SetValue(string(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN)));
            m_lot_edit[i].GetTextBoxPointer().Update(true);
         }
      }
   }
}
```

For the other two methods, level values will be switched from points to price.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::TakeMarketSwitch(long lparam)
{
   for(int i=0; i<2; i++)
   {
      if(lparam==m_switch_button[3*i+1].Id())
      {
         if(m_switch_button[3*i+1].IsPressed())
         {
            MqlTick tick;
            if(SymbolInfoTick(Symbol(),tick))
            {
               m_tp_edit[i].SetDigits(_Digits);
               m_tp_edit[i].StepValue(_Point);
               m_tp_edit[i].SetValue(string(tick.ask));
               m_tp_edit[i].GetTextBoxPointer().Update(true);
            }
         }
         else
         {
            m_tp_edit[i].SetDigits(0);
            m_tp_edit[i].StepValue(1);
            m_tp_edit[i].SetValue(string(150));
            m_tp_edit[i].GetTextBoxPointer().Update(true);
         }
      }
   }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::StopMarketSwitch(long lparam)
{
   for(int i=0; i<2; i++)
   {
      if(lparam==m_switch_button[3*i+2].Id())
      {
         if(m_switch_button[3*i+2].IsPressed())
         {
            MqlTick tick;
            if(SymbolInfoTick(Symbol(),tick))
            {
               m_sl_edit[i].SetDigits(_Digits);
               m_sl_edit[i].StepValue(_Point);
               m_sl_edit[i].SetValue(string(tick.bid));
               m_sl_edit[i].GetTextBoxPointer().Update(true);
            }
         }
         else
         {
            m_sl_edit[i].SetDigits(0);
            m_sl_edit[i].StepValue(1);
            m_sl_edit[i].SetValue(string(150));
            m_sl_edit[i].GetTextBoxPointer().Update(true);
         }
      }
   }
}
```

Now, let's call all the three methods in the event handler, in the Button Click Event section:

```
      // --- Switch Lot/Percent of balance
      LotMarketSwitch(lparam);
      //--- Switch Take Profit Points/Price
      TakeMarketSwitch(lparam);
      //--- Switch Stop Loss Points/Price
      StopMarketSwitch(lparam);
```

Check the result:

![](https://c.mql5.com/2/40/010.gif)

Fig. 10 Switching modes of input fields in the market orders window.

The last task is to enable placing of market orders according to the settings and input field values, using a button or a hotkey. For this purpose, create three new methods:

- **OnInitTrading()** — defines and sets the trading account environment.
- **SetBuyOrder()** — opens a market buy order.
- **SetSellOrder()** — opens a market sell order.

Create them in the private section of our base class **CFastTrading**, and implement them in the same class.

```
//+------------------------------------------------------------------+
//| Trading Environment Initialization                               |
//+------------------------------------------------------------------+
void CFastTrading::OnInitTrading()
{
   string array_used_symbols[];
//--- Fill in the array of used symbols
   CreateUsedSymbolsArray(SYMBOLS_MODE_CURRENT,"",array_used_symbols);
//--- Set the type of the used symbol list in the symbol collection and fill in the list of symbol timeseries
   m_trade.SetUsedSymbols(array_used_symbols);
//--- Pass all existing collections to the trading class
   m_trade.TradingOnInit();
   m_trade.TradingSetMagic(m_magic_number);
   m_trade.TradingSetLogLevel(LOG_LEVEL_ERROR_MSG);
//--- Set synchronous passing of orders for all used symbols
   m_trade.TradingSetAsyncMode(false);
//--- Set correct order expiration and filling types to all trading objects
   m_trade.TradingSetCorrectTypeExpiration();
   m_trade.TradingSetCorrectTypeFilling();
}
```

Call this method in the class constructor:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CFastTrading::CFastTrading(void)
{
   OnInitTrading();
}
```

Before implementing trading functions, note that their execution will be carried out on two different events.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::SetBuyOrder(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_buy_execute.Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_B))
   {
      //---
      double lot;
      if(m_switch_button[0].IsPressed())
         lot=LotPercent(Symbol(),ORDER_TYPE_BUY,SymbolInfoDouble(Symbol(),SYMBOL_ASK),StringToDouble(m_lot_edit[0].GetValue()));
      else
         lot=NormalizeLot(Symbol(),StringToDouble(m_lot_edit[0].GetValue()));
      if(m_switch_button[1].IsPressed() && m_switch_button[2].IsPressed())
      {
         double tp=double(m_tp_edit[0].GetValue());
         double sl=double(m_sl_edit[0].GetValue());
         if(m_trade.OpenBuy(lot,Symbol(),m_magic_number,sl,tp))
            return(true);
      }
      else if(!m_switch_button[1].IsPressed() && !m_switch_button[2].IsPressed())
      {
         int tp=int(m_tp_edit[0].GetValue());
         int sl=int(m_sl_edit[0].GetValue());
         if(m_trade.OpenBuy(lot,Symbol(),m_magic_number,sl,tp))
            return(true);
      }
      else if(m_switch_button[1].IsPressed() && !m_switch_button[2].IsPressed())
      {
         double tp=double(m_tp_edit[0].GetValue());
         int sl=int(m_sl_edit[0].GetValue());
         if(m_trade.OpenBuy(lot,Symbol(),m_magic_number,sl,tp))
            return(true);
      }
      else if(!m_switch_button[1].IsPressed() && m_switch_button[2].IsPressed())
      {
         int tp=int(m_tp_edit[0].GetValue());
         double sl=double(m_sl_edit[0].GetValue());
         if(m_trade.OpenBuy(lot,Symbol(),m_magic_number,sl,tp))
            return(true);
      }
   }
   return(false);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::SetSellOrder(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_sell_execute.Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_S))
   {
      //---
      double lot;
      if(m_switch_button[3].IsPressed())
         lot=LotPercent(Symbol(),ORDER_TYPE_SELL,SymbolInfoDouble(Symbol(),SYMBOL_BID),StringToDouble(m_lot_edit[1].GetValue()));
      else
         lot=NormalizeLot(Symbol(),StringToDouble(m_lot_edit[1].GetValue()));
      //---
      if(m_switch_button[4].IsPressed() && m_switch_button[5].IsPressed())
      {
         double tp=double(m_tp_edit[1].GetValue());
         double sl=double(m_sl_edit[1].GetValue());
         if(m_trade.OpenSell(lot,Symbol(),m_magic_number,sl,tp))
            return(true);
      }
      else if(!m_switch_button[4].IsPressed() && !m_switch_button[5].IsPressed())
      {
         int tp=int(m_tp_edit[1].GetValue());
         int sl=int(m_sl_edit[1].GetValue());
         if(m_trade.OpenSell(lot,Symbol(),m_magic_number,sl,tp))
            return(true);
      }
      else if(!m_switch_button[4].IsPressed() && m_switch_button[5].IsPressed())
      {
         int tp=int(m_tp_edit[1].GetValue());
         double sl=double(m_sl_edit[1].GetValue());
         if(m_trade.OpenSell(lot,Symbol(),m_magic_number,sl,tp))
            return(true);
      }
      else if(m_switch_button[4].IsPressed() && !m_switch_button[5].IsPressed())
      {
         double tp=double(m_tp_edit[1].GetValue());
         int sl=int(m_sl_edit[1].GetValue());
         if(m_trade.OpenSell(lot,Symbol(),m_magic_number,sl,tp))
            return(true);
      }
   }
   return(false);
}
```

Although both trading methods will be called in the event handler, they will be outside any of the sections which identify events for the reason indicated above. The current implementation of the above functions uses the **LotPercent()** method for calculating lot size as percent of balance.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CFastTrading::LotPercent(string symbol,ENUM_ORDER_TYPE trade_type,double price,double percent)
{
   double margin=0.0;
//--- checks
   if(symbol=="" || price<=0.0 || percent<1 || percent>100)
      return(0.0);
//--- calculate margin requirements for 1 lot
   if(!OrderCalcMargin(trade_type,symbol,1.0,price,margin) || margin<0.0)
      return(0.0);
//---
   if(margin==0.0)
      return(0.0);
//--- calculate maximum volume
   double volume=NormalizeDouble(AccountInfoDouble(ACCOUNT_BALANCE)*percent/100.0/margin,2);
//--- normalize and check limits
   double stepvol=SymbolInfoDouble(symbol,SYMBOL_VOLUME_STEP);
   if(stepvol>0.0)
      volume=stepvol*MathFloor(volume/stepvol);
//---
   double minvol=SymbolInfoDouble(symbol,SYMBOL_VOLUME_MIN);
   if(volume<minvol)
      volume=0.0;
//---
   double maxvol=SymbolInfoDouble(symbol,SYMBOL_VOLUME_MAX);
   if(volume>maxvol)
      volume=maxvol;
//--- return volume
   return(volume);
}
```

The part concerning operations with market orders is ready. Now, let's move on to the window for creating pending orders. Let's create the interface following the same steps that we completed with market orders. The most of the steps are similar. I will only go deeper into detail with a few moiments.

```
//+------------------------------------------------------------------+
//| Pending order creation and editing window                        |
//+------------------------------------------------------------------+
bool CFastTrading::CreatePendingOrdersWindow(void)
{
//--- Add a window pointer to the window array
   CWndContainer::AddWindow(m_orders_windows[1]);
//--- Properties
   m_orders_windows[1].XSize(600);
   m_orders_windows[1].YSize(580);
//--- Coordinates
   int x=m_order_button[0].XGap();
   int y=m_order_button[0].YGap()+60;
//---
   color clrmain=C'31,209,111';
//---
   m_orders_windows[1].CaptionHeight(22);
   m_orders_windows[1].IsMovable(true);
   m_orders_windows[1].CaptionColor(clrmain);
   m_orders_windows[1].CaptionColorLocked(clrmain);
   m_orders_windows[1].CaptionColorHover(clrmain);
   m_orders_windows[1].BackColor(m_background_color);
   m_orders_windows[1].BorderColor(clrmain);
   m_orders_windows[1].FontSize(m_base_font_size);
   m_orders_windows[1].Font(m_base_font);
   m_orders_windows[1].WindowType(W_DIALOG);
//--- Create the form
   if(!m_orders_windows[1].CreateWindow(m_chart_id,m_subwin,CAPTION_P_ORD_NAME,x,y))
      return(false);
//---BUY-STOP BLOCK
   if(!CreateFrame(m_orders_windows[1],m_frame[2],10,20,BUYSTOP_ORDER,2))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[0],20,60,PRICE,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[1],20+80+20,30,LOT,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[2],20+(80+20)*2,30,TP,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[3],20+(80+20)*3,30,SL,2))
      return(false);
//--- Switches
   for(int i=0; i<3; i++)
      if(!CreateSwitchButton(m_orders_windows[1],m_p_switch_button[i],"-",120+(80+20)*i,60,2))
         return(false);
//--- Edits
   if(!CreatePriceEdit(m_orders_windows[1],m_pr_edit[0],20,60+35,2))
      return(false);
   if(!CreateLotEdit(m_orders_windows[1],m_lot_edit[2],20+(80+20),60+35,2))
      return(false);
   if(!CreateTakeProfitEdit(m_orders_windows[1],m_tp_edit[2],20+(80+20)*2,60+35,2))
      return(false);
   if(!CreateStopLossEdit(m_orders_windows[1],m_sl_edit[2],20+(80+20)*3,60+35,2))
      return(false);
//--- Buy Stop placing button
   if(!CreateBuyButton(m_orders_windows[1],m_buystop_execute,"Buy Stop ( 1 )",m_orders_windows[1].XSize()-(120+20),103,2))
      return(false);
//---SELL-STOP BLOCK
   if(!CreateFrame(m_orders_windows[1],m_frame[3],10,160,SELLSTOP_ORDER,2))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[4],20,170+30,PRICE,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[5],20+80+20,170,LOT,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[6],20+(80+20)*2,170,TP,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[7],20+(80+20)*3,170,SL,2))
      return(false);
//--- Switches
   for(int i=3; i<6; i++)
      if(!CreateSwitchButton(m_orders_windows[1],m_p_switch_button[i],"-",120+(80+20)*(i-3),170+30,2))
         return(false);
//--- Edits
   if(!CreatePriceEdit(m_orders_windows[1],m_pr_edit[1],20,170+30+35,2))
      return(false);
   if(!CreateLotEdit(m_orders_windows[1],m_lot_edit[3],20+(80+20),170+30+35,2))
      return(false);
   if(!CreateTakeProfitEdit(m_orders_windows[1],m_tp_edit[3],20+(80+20)*2,170+30+35,2))
      return(false);
   if(!CreateStopLossEdit(m_orders_windows[1],m_sl_edit[3],20+(80+20)*3,170+30+35,2))
      return(false);
//--- Sell Stop placing button
   if(!CreateSellButton(m_orders_windows[1],m_sellstop_execute,"Sell Stop ( 2 )",m_orders_windows[1].XSize()-(120+20),242,2))
      return(false);
//---BUY-LIMIT BLOCK
   if(!CreateFrame(m_orders_windows[1],m_frame[4],10,300,BUYLIMIT_ORDER,2))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[8],20,330,PRICE,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[9],20+80+20,310,LOT,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[10],20+(80+20)*2,310,TP,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[11],20+(80+20)*3,310,SL,2))
      return(false);
//--- Switches
   for(int i=6; i<9; i++)
      if(!CreateSwitchButton(m_orders_windows[1],m_p_switch_button[i],"-",120+(80+20)*(i-6),330,2))
         return(false);
//--- Edits
   if(!CreatePriceEdit(m_orders_windows[1],m_pr_edit[2],20,365,2))
      return(false);
   if(!CreateLotEdit(m_orders_windows[1],m_lot_edit[4],20+(80+20),365,2))
      return(false);
   if(!CreateTakeProfitEdit(m_orders_windows[1],m_tp_edit[4],20+(80+20)*2,365,2))
      return(false);
   if(!CreateStopLossEdit(m_orders_windows[1],m_sl_edit[4],20+(80+20)*3,365,2))
      return(false);
//--- Buy Limit placing button
   if(!CreateBuyButton(m_orders_windows[1],m_buylimit_execute,"Buy Limit ( 3 )",m_orders_windows[1].XSize()-(120+20),382,2))
      return(false);
//---SELL-LIMIT BLOCK
   if(!CreateFrame(m_orders_windows[1],m_frame[5],10,440,SELLLIMIT_ORDER,2))
      return(false);
//--- Headers
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[12],20,470,PRICE,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[13],20+80+20,450,LOT,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[14],20+(80+20)*2,450,TP,2))
      return(false);
   if(!CreateLabel(m_orders_windows[1],m_p_text_labels[15],20+(80+20)*3,450,SL,2))
      return(false);
//--- Switches
   for(int i=9; i<12; i++)
      if(!CreateSwitchButton(m_orders_windows[1],m_p_switch_button[i],"-",120+(80+20)*(i-9),470,2))
         return(false);
//--- Edits
   if(!CreatePriceEdit(m_orders_windows[1],m_pr_edit[3],20,505,2))
      return(false);
   if(!CreateLotEdit(m_orders_windows[1],m_lot_edit[5],20+(80+20),505,2))
      return(false);
   if(!CreateTakeProfitEdit(m_orders_windows[1],m_tp_edit[5],20+(80+20)*2,505,2))
      return(false);
   if(!CreateStopLossEdit(m_orders_windows[1],m_sl_edit[5],20+(80+20)*3,505,2))
      return(false);
//--- Sell Limit placing button
   if(!CreateSellButton(m_orders_windows[1],m_selllimit_execute,"Sell Limit ( 4 )",m_orders_windows[1].XSize()-(120+20),522,2))
      return(false);
//---
   return(true);
}
```

As you can see, the above code uses the same methods, as we used for the market orders window. However, it has one new method. **CreatePriceEdit()** is the input fields for the pending order placing price.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreatePriceEdit(CWindow &window,CTextEdit &text_edit,const int x_gap,const int y_gap,int w_number)
{
//--- Store the pointer to the main control
   text_edit.MainPointer(window);
//--- Properties
   text_edit.XSize(80);
   text_edit.YSize(24);
   text_edit.Font(m_base_font);
   text_edit.FontSize(m_base_font_size);
   text_edit.StepValue(_Point);
   text_edit.SpinEditMode(true);
   text_edit.SetDigits(_Digits);
   text_edit.GetTextBoxPointer().XGap(1);
   text_edit.GetTextBoxPointer().XSize(80);
//--- Create a control element
   if(!text_edit.CreateTextEdit("",x_gap,window.CaptionHeight()+y_gap))
      return(false);
//---
   MqlTick tick;
   if(SymbolInfoTick(Symbol(),tick))
      text_edit.SetValue(string(tick.bid));
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(w_number,text_edit);
   return(true);
}
```

The initial template is ready after the method implementation, however the interface is not yet usable. It needs to be configured. We will implement this part in the same order: configure toggle buttons, connect their state with the input field properties and create functions for placing pending orders.

Go into the **OnEvent()** handler, Interface Creation Completion section, and set names for the toggle buttons:

```
//--- UI creation completion
   if(id==CHARTEVENT_CUSTOM+ON_END_CREATE_GUI)
   {
      //---
      SetButtonParam(m_switch_button[0],LOT);
      SetButtonParam(m_switch_button[1],POINTS);
      SetButtonParam(m_switch_button[2],POINTS);
      SetButtonParam(m_switch_button[3],LOT);
      SetButtonParam(m_switch_button[4],POINTS);
      SetButtonParam(m_switch_button[5],POINTS);
      //---
      SetButtonParam(m_p_switch_button[0],LOT);
      SetButtonParam(m_p_switch_button[1],POINTS);
      SetButtonParam(m_p_switch_button[2],POINTS);
      SetButtonParam(m_p_switch_button[3],LOT);
      SetButtonParam(m_p_switch_button[4],POINTS);
      SetButtonParam(m_p_switch_button[5],POINTS);
      SetButtonParam(m_p_switch_button[6],LOT);
      SetButtonParam(m_p_switch_button[7],POINTS);
      SetButtonParam(m_p_switch_button[8],POINTS);
      SetButtonParam(m_p_switch_button[9],LOT);
      SetButtonParam(m_p_switch_button[10],POINTS);
      SetButtonParam(m_p_switch_button[11],POINTS);
   }
```

Go to 'Button Click Event' section and implement the mechanism which switches button states and changes their names:

```
      //---
      ButtonSwitch(m_p_switch_button[0],lparam,LOT,PERC_DEPO);
      ButtonSwitch(m_p_switch_button[1],lparam,POINTS,PRICE);
      ButtonSwitch(m_p_switch_button[2],lparam,POINTS,PRICE);
      ButtonSwitch(m_p_switch_button[3],lparam,LOT,PERC_DEPO);
      ButtonSwitch(m_p_switch_button[4],lparam,POINTS,PRICE);
      ButtonSwitch(m_p_switch_button[5],lparam,POINTS,PRICE);
      ButtonSwitch(m_p_switch_button[6],lparam,LOT,PERC_DEPO);
      ButtonSwitch(m_p_switch_button[7],lparam,POINTS,PRICE);
      ButtonSwitch(m_p_switch_button[8],lparam,POINTS,PRICE);
      ButtonSwitch(m_p_switch_button[9],lparam,LOT,PERC_DEPO);
      ButtonSwitch(m_p_switch_button[10],lparam,POINTS,PRICE);
      ButtonSwitch(m_p_switch_button[11],lparam,POINTS,PRICE);
```

To link the toggle buttons with the input fields and their properties, create three new methods: **LotPendingSwitch()**, **TakePendingSwitch()**, **StopPendingSwitch()**.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::LotPendingSwitch(long lparam)
{
   for(int i=0; i<4; i++)
   {
      if(lparam==m_p_switch_button[3*i].Id())
      {
         if(m_p_switch_button[3*i].IsPressed())
         {
            m_lot_edit[i+2].SetDigits(0);
            m_lot_edit[i+2].StepValue(1);
            m_lot_edit[i+2].MaxValue(100);
            m_lot_edit[i+2].MinValue(1);
            m_lot_edit[i+2].SetValue(string(2));
            m_lot_edit[i+2].GetTextBoxPointer().Update(true);
         }
         else
         {
            m_lot_edit[i+2].SetDigits(2);
            m_lot_edit[i+2].StepValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP));
            m_lot_edit[i+2].MinValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN));
            m_lot_edit[i+2].MaxValue(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX));
            m_lot_edit[i+2].SetValue(string(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN)));
            m_lot_edit[i+2].GetTextBoxPointer().Update(true);
         }
      }
   }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::TakePendingSwitch(long lparam)
{
   for(int i=0; i<4; i++)
   {
      if(lparam==m_p_switch_button[3*i+1].Id())
      {
         if(m_p_switch_button[3*i+1].IsPressed())
         {
            MqlTick tick;
            if(SymbolInfoTick(Symbol(),tick))
            {
               m_tp_edit[i+2].SetDigits(_Digits);
               m_tp_edit[i+2].StepValue(_Point);
               m_tp_edit[i+2].SetValue(string(tick.ask));
               m_tp_edit[i+2].GetTextBoxPointer().Update(true);
            }
         }
         else
         {
            m_tp_edit[i+2].SetDigits(0);
            m_tp_edit[i+2].StepValue(1);
            m_tp_edit[i+2].SetValue(string(150));
            m_tp_edit[i+2].GetTextBoxPointer().Update(true);
         }
      }
   }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::StopPendingSwitch(long lparam)
{
   for(int i=0; i<4; i++)
   {
      if(lparam==m_p_switch_button[3*i+2].Id())
      {
         if(m_p_switch_button[3*i+2].IsPressed())
         {
            MqlTick tick;
            if(SymbolInfoTick(Symbol(),tick))
            {
               m_sl_edit[i+2].SetDigits(_Digits);
               m_sl_edit[i+2].StepValue(_Point);
               m_sl_edit[i+2].SetValue(string(tick.bid));
               m_sl_edit[i+2].GetTextBoxPointer().Update(true);
            }
         }
         else
         {
            m_sl_edit[i+2].SetDigits(0);
            m_sl_edit[i+2].StepValue(1);
            m_sl_edit[i+2].SetValue(string(150));
            m_sl_edit[i+2].GetTextBoxPointer().Update(true);
         }
      }
   }
}
```

Now, let's call them in the button click event handler section:

```
      // --- Switch Lot/Percent of balance
      LotPendingSwitch(lparam);
      //--- Switch Take Profit Points/Price
      TakePendingSwitch(lparam);
      //--- Switch Stop Loss Points/Price
      StopPendingSwitch(lparam);
```

Compile the project and see the result:

![](https://c.mql5.com/2/40/011.gif)

Fig. 11 Switching input fields modes for pending orders.

The input field mode switches are ready; property changing also works. Move on to creating functions, required to place pending orders. The algorithm is also similar to market orders. Create four methods, one for each type.

```
   bool              SetBuyStopOrder(int id,long lparam);
   bool              SetSellStopOrder(int id,long lparam);
   bool              SetBuyLimitOrder(int id,long lparam);
   bool              SetSellLimitOrder(int id,long lparam);
```

Implement them. Set a hotkey for each of the pending orders, in addition to the button click.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::SetBuyStopOrder(int id,long lparam)
{
   if(!m_orders_windows[1].IsVisible())
      return(false);
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_buystop_execute.Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_1))
   {
      //---
      double lot;
      if(m_p_switch_button[0].IsPressed())
         lot=LotPercent(Symbol(),ORDER_TYPE_BUY,SymbolInfoDouble(Symbol(),SYMBOL_ASK),StringToDouble(m_lot_edit[2].GetValue()));
      else
         lot=NormalizeLot(Symbol(),StringToDouble(m_lot_edit[2].GetValue()));
      //---
      double pr=double(m_pr_edit[0].GetValue());
      //---
      if(m_p_switch_button[1].IsPressed() && m_p_switch_button[2].IsPressed())
      {
         double tp=double(m_tp_edit[2].GetValue());
         double sl=double(m_sl_edit[2].GetValue());
         if(m_trade.PlaceBuyStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(!m_p_switch_button[1].IsPressed() && !m_p_switch_button[2].IsPressed())
      {
         int tp=int(m_tp_edit[2].GetValue());
         int sl=int(m_sl_edit[2].GetValue());
         if(m_trade.PlaceBuyStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(m_p_switch_button[1].IsPressed() && !m_p_switch_button[2].IsPressed())
      {
         double tp=double(m_tp_edit[2].GetValue());
         int sl=int(m_sl_edit[2].GetValue());
         if(m_trade.PlaceBuyStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(!m_p_switch_button[1].IsPressed() && m_p_switch_button[2].IsPressed())
      {
         int tp=int(m_tp_edit[2].GetValue());
         double sl=double(m_sl_edit[2].GetValue());
         if(m_trade.PlaceBuyStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
   }
   return(false);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::SetSellStopOrder(int id,long lparam)
{
   if(!m_orders_windows[1].IsVisible())
      return(false);
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_sellstop_execute.Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_2))
   {
      //---
      double lot;
      if(m_p_switch_button[3].IsPressed())
         lot=LotPercent(Symbol(),ORDER_TYPE_SELL,SymbolInfoDouble(Symbol(),SYMBOL_BID),StringToDouble(m_lot_edit[3].GetValue()));
      else
         lot=NormalizeLot(Symbol(),StringToDouble(m_lot_edit[3].GetValue()));
      //---
      double pr=double(m_pr_edit[1].GetValue());
      //---
      if(m_p_switch_button[4].IsPressed() && m_p_switch_button[5].IsPressed())
      {
         double tp=double(m_tp_edit[3].GetValue());
         double sl=double(m_sl_edit[3].GetValue());
         if(m_trade.PlaceSellStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(!m_p_switch_button[4].IsPressed() && !m_p_switch_button[5].IsPressed())
      {
         int tp=int(m_tp_edit[3].GetValue());
         int sl=int(m_sl_edit[3].GetValue());
         if(m_trade.PlaceSellStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(m_p_switch_button[4].IsPressed() && !m_p_switch_button[5].IsPressed())
      {
         double tp=double(m_tp_edit[3].GetValue());
         int sl=int(m_sl_edit[3].GetValue());
         if(m_trade.PlaceSellStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(!m_p_switch_button[4].IsPressed() && m_p_switch_button[5].IsPressed())
      {
         int tp=int(m_tp_edit[3].GetValue());
         double sl=double(m_sl_edit[3].GetValue());
         if(m_trade.PlaceSellStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
   }
   return(false);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::SetBuyLimitOrder(int id,long lparam)
{
   if(!m_orders_windows[1].IsVisible())
      return(false);
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_buylimit_execute.Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_3))
   {
      //---
      double lot;
      if(m_p_switch_button[6].IsPressed())
         lot=LotPercent(Symbol(),ORDER_TYPE_BUY,SymbolInfoDouble(Symbol(),SYMBOL_ASK),StringToDouble(m_lot_edit[4].GetValue()));
      else
         lot=NormalizeLot(Symbol(),StringToDouble(m_lot_edit[4].GetValue()));
      //---
      double pr=double(m_pr_edit[2].GetValue());
      //---
      if(m_p_switch_button[7].IsPressed() && m_p_switch_button[8].IsPressed())
      {
         double tp=double(m_tp_edit[4].GetValue());
         double sl=double(m_sl_edit[4].GetValue());
         if(m_trade.PlaceBuyLimit(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(!m_p_switch_button[7].IsPressed() && !m_p_switch_button[8].IsPressed())
      {
         int tp=int(m_tp_edit[4].GetValue());
         int sl=int(m_sl_edit[4].GetValue());
         if(m_trade.PlaceBuyLimit(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(m_p_switch_button[7].IsPressed() && !m_p_switch_button[8].IsPressed())
      {
         double tp=double(m_tp_edit[4].GetValue());
         int sl=int(m_sl_edit[4].GetValue());
         if(m_trade.PlaceBuyLimit(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(!m_p_switch_button[7].IsPressed() && m_p_switch_button[8].IsPressed())
      {
         int tp=int(m_tp_edit[4].GetValue());
         double sl=double(m_sl_edit[4].GetValue());
         if(m_trade.PlaceBuyLimit(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
   }
   return(false);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::SetSellLimitOrder(int id,long lparam)
{
   if(!m_orders_windows[1].IsVisible())
      return(false);
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_selllimit_execute.Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_4))
   {
      //---
      double lot;
      if(m_p_switch_button[9].IsPressed())
         lot=LotPercent(Symbol(),ORDER_TYPE_SELL,SymbolInfoDouble(Symbol(),SYMBOL_BID),StringToDouble(m_lot_edit[5].GetValue()));
      else
         lot=NormalizeLot(Symbol(),StringToDouble(m_lot_edit[5].GetValue()));
      //---
      double pr=double(m_pr_edit[3].GetValue());
      //---
      if(m_p_switch_button[10].IsPressed() && m_p_switch_button[11].IsPressed())
      {
         double tp=double(m_tp_edit[5].GetValue());
         double sl=double(m_sl_edit[5].GetValue());
         if(m_trade.PlaceSellStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(!m_p_switch_button[10].IsPressed() && !m_p_switch_button[11].IsPressed())
      {
         int tp=int(m_tp_edit[5].GetValue());
         int sl=int(m_sl_edit[5].GetValue());
         if(m_trade.PlaceSellStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(m_p_switch_button[10].IsPressed() && !m_p_switch_button[11].IsPressed())
      {
         double tp=double(m_tp_edit[5].GetValue());
         int sl=int(m_sl_edit[5].GetValue());
         if(m_trade.PlaceSellStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
      else if(!m_p_switch_button[10].IsPressed() && m_p_switch_button[11].IsPressed())
      {
         int tp=int(m_tp_edit[5].GetValue());
         double sl=double(m_sl_edit[5].GetValue());
         if(m_trade.PlaceSellStop(lot,Symbol(),pr,sl,tp,m_magic_number))
            return(true);
      }
   }
   return(false);
}
```

Now, go to the event handler and add to it calls of the newly created methods:

```
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void CFastTrading::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
{
//---
   SetBuyOrder(id,lparam);
   SetSellOrder(id,lparam);
//---
   SetBuyStopOrder(id,lparam);
   SetSellStopOrder(id,lparam);
   SetBuyLimitOrder(id,lparam);
   SetSellLimitOrder(id,lparam);
```

The basic functionality for working with all order types is ready. Since the ultimate purpose if this application is to provide maximum convenience and speed in manual trading, I decided to add a small feature which will accelerate placing of market and pending orders. Its principle is as follows: when setting input field values in different modes, there should be a possibility to change the value by a specified step using up and down arrows. However, when setting, for example, a pending order price, the EURUSD price would be changed from 1.08500 to 1.85001 upon pressing an arrow. It's slow and not very convenient. Therefore, when the input field is selected for editing and we press on Up/Down hotkeys, let the value be changed by a step multiplied by 10. For this purpose we need to create the **ArrowSwitch()** method.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::ArrowSwitch(long lparam)
{
//--- Prices
   for(int i=0; i<4; i++)
   {
      if(m_pr_edit[i].GetTextBoxPointer().TextEditState())
      {
         if(lparam==KEY_UP)
         {
            //--- Get the new value
            double value=StringToDouble(m_pr_edit[i].GetValue())+m_pr_edit[i].StepValue()*10;
            //--- Increase by one step and check for exceeding the limit
            m_pr_edit[i].SetValue(DoubleToString(value),false);
         }
         else if(lparam==KEY_DOWN)
         {
            //--- Get the new value
            double value=StringToDouble(m_pr_edit[i].GetValue())-m_pr_edit[i].StepValue()*10;
            //--- Increase by one step and check for exceeding the limit
            m_pr_edit[i].SetValue(DoubleToString(value),false);
         }
      }
   }
//--- Lot
   for(int i=0; i<6; i++)
   {
      if(m_lot_edit[i].GetTextBoxPointer().TextEditState())
      {
         if(lparam==KEY_UP)
         {
            //--- Get the new value
            double value=StringToDouble(m_lot_edit[i].GetValue())+m_lot_edit[i].StepValue()*10;
            //--- Increase by one step and check for exceeding the limit
            m_lot_edit[i].SetValue(DoubleToString(value),false);
         }
         else if(lparam==KEY_DOWN)
         {
            //--- Get the new value
            double value=StringToDouble(m_lot_edit[i].GetValue())-m_lot_edit[i].StepValue()*10;
            //--- Increase by one step and check for exceeding the limit
            m_lot_edit[i].SetValue(DoubleToString(value),false);
         }
      }
   }
//--- Take Profit, Stop Loss
   for(int i=0; i<6; i++)
   {
      //---
      if(m_tp_edit[i].GetTextBoxPointer().TextEditState())
      {
         if(lparam==KEY_UP)
         {
            //--- Get the new value
            double value=StringToDouble(m_tp_edit[i].GetValue())+m_tp_edit[i].StepValue()*10;
            //--- Increase by one step and check for exceeding the limit
            m_tp_edit[i].SetValue(DoubleToString(value),false);
         }
         else if(lparam==KEY_DOWN)
         {
            //--- Get the new value
            double value=StringToDouble(m_tp_edit[i].GetValue())-m_tp_edit[i].StepValue()*10;
            //--- Increase by one step and check for exceeding the limit
            m_tp_edit[i].SetValue(DoubleToString(value),false);
         }
      }
      //---
      if(m_sl_edit[i].GetTextBoxPointer().TextEditState())
      {
         if(lparam==KEY_UP)
         {
            //--- Get the new value
            double value=StringToDouble(m_sl_edit[i].GetValue())+m_sl_edit[i].StepValue()*10;
            //--- Increase by one step and check for exceeding the limit
            m_sl_edit[i].SetValue(DoubleToString(value),false);
         }
         else if(lparam==KEY_DOWN)
         {
            //--- Get the new value
            double value=StringToDouble(m_sl_edit[i].GetValue())-m_sl_edit[i].StepValue()*10;
            //--- Increase by one step and check for exceeding the limit
            m_sl_edit[i].SetValue(DoubleToString(value),false);
         }
      }
   }
}
```

The method should be called in the event handler, within the Keypress section.

```
//--- Keypress
   if(id==CHARTEVENT_KEYDOWN)
   {
      //---
      ArrowSwitch(lparam);
....
```

Figure 12 shows the difference between the value change speed.

![](https://c.mql5.com/2/40/012.gif)

Fig. 12 Quick value change in edit fields using hotkeys.

The tools for quick placing of market and pending orders is ready. Now, let's go to the main application window and add actions relating to the existing orders (as is shown in the very first figure, at the beginning of the article). These include three actions: closing all profitable orders, closing all losing orders, and deleting all existing pending orders. These actions should only apply to the orders placed by our application, which should have a certain magic number. It should also be remembered that when placing orders with the current magic number and then changing the magic number in EA properties, the existing orders having the old number should be ignored as they will no longer be related to the current application.

Let's implement these actions. Create three new methods:

```
   void              CloseAllMarketProfit(int id,long lparam);
   void              CloseAllMarketLoss(int id,long lparam);
   void              DeleteAllPending(int id,long lparam);
```

Implement each of them.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::CloseAllMarketProfit(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_order_button[2].Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_C))
   {
      //--- declare the request and the result
      MqlTradeRequest request;
      MqlTradeResult  result;
      int total=PositionsTotal(); // the number of open positions
      //--- iterate over open positions
      for(int i=total-1; i>=0; i--)
      {
         //--- order parameters
         ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
         string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
         int    digits=(int)SymbolInfoInteger(position_symbol,SYMBOL_DIGITS);              // the number of decimal places
         ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
         double volume=PositionGetDouble(POSITION_VOLUME);                                 // position volume
         ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
         double profit=PositionGetDouble(POSITION_PROFIT);
         double swap=PositionGetDouble(POSITION_SWAP);
         //--- if MagicNumber matches
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
               if(profit+swap>0)
               {
                  //--- zeroing the request and result values
                  ZeroMemory(request);
                  ZeroMemory(result);
                  //--- set the operation parameters
                  request.action   =TRADE_ACTION_DEAL;        // trading operation type
                  request.position =position_ticket;          // position ticket
                  request.symbol   =position_symbol;          // symbol
                  request.volume   =volume;                   // position volume
                  request.deviation=5;                        // allowable price deviation
                  request.magic    =m_magic_number;           // position MagicNumber
                  //--- Set order price and type depending on the position type
                  if(type==POSITION_TYPE_BUY)
                  {
                     request.price=SymbolInfoDouble(position_symbol,SYMBOL_BID);
                     request.type =ORDER_TYPE_SELL;
                  }
                  else
                  {
                     request.price=SymbolInfoDouble(position_symbol,SYMBOL_ASK);
                     request.type =ORDER_TYPE_BUY;
                  }
                  //--- sending a request
                  if(!OrderSend(request,result))
                     PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
               }
      }
   }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::CloseAllMarketLoss(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_order_button[3].Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_D))
   {
      //--- declare the request and the result
      MqlTradeRequest request;
      MqlTradeResult  result;
      ZeroMemory(request);
      ZeroMemory(result);
      int total=PositionsTotal(); // the number of open positions
      //--- iterate over open positions
      for(int i=total-1; i>=0; i--)
      {
         //--- order parameters
         ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
         string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
         int    digits=(int)SymbolInfoInteger(position_symbol,SYMBOL_DIGITS);              // the number of decimal places
         ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
         double volume=PositionGetDouble(POSITION_VOLUME);                                 // position volume
         ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
         double profit=PositionGetDouble(POSITION_PROFIT);
         double swap=PositionGetDouble(POSITION_SWAP);
         //--- if MagicNumber matches
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
               if(profit+swap<0)
               {
                  //--- zeroing the request and result values
                  ZeroMemory(request);
                  ZeroMemory(result);
                  //--- set the operation parameters
                  request.action   =TRADE_ACTION_DEAL;        // trading operation type
                  request.position =position_ticket;          // position ticket
                  request.symbol   =position_symbol;          // symbol
                  request.volume   =volume;                   // position volume
                  request.deviation=5;                        // allowable price deviation
                  request.magic    =m_magic_number;           // position MagicNumber
                  //--- Set order price and type depending on the position type
                  if(type==POSITION_TYPE_BUY)
                  {
                     request.price=SymbolInfoDouble(position_symbol,SYMBOL_BID);
                     request.type =ORDER_TYPE_SELL;
                  }
                  else
                  {
                     request.price=SymbolInfoDouble(position_symbol,SYMBOL_ASK);
                     request.type =ORDER_TYPE_BUY;
                  }
                  //--- sending a request
                  if(!OrderSend(request,result))
                     PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
               }
      }
   }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::DeleteAllPending(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_order_button[4].Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_R))
   {
      //--- declare and initialize the trade request and result of trade request
      MqlTradeRequest request;
      MqlTradeResult  result;
      //--- iterate over all placed pending orders
      for(int i=OrdersTotal()-1; i>=0; i--)
      {
         ulong  order_ticket=OrderGetTicket(i);                   // order ticket
         ulong  magic=OrderGetInteger(ORDER_MAGIC);               // order MagicNumber
         //--- if MagicNumber matches
         if(magic==m_magic_number)
         {
            //--- zeroing the request and result values
            ZeroMemory(request);
            ZeroMemory(result);
            //--- set the operation parameters
            request.action= TRADE_ACTION_REMOVE;                  // trading operation type
            request.order = order_ticket;                         // order ticket
            //--- sending a request
            if(!OrderSend(request,result))
               PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
         }
      }
   }
}
```

Note that the created methods also stick to the rule for performing an action by two different events: by clicking a button in the main application window and by pressing a hotkey. All the three functions will be called in the event handler, outside of any sections.

```
//+------------------------------------------------------------------+
//| Chart event handler                                              |
//+------------------------------------------------------------------+
void CFastTrading::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
{
//---
   SetBuyOrder(id,lparam);
   SetSellOrder(id,lparam);
//---
   SetBuyStopOrder(id,lparam);
   SetSellStopOrder(id,lparam);
   SetBuyLimitOrder(id,lparam);
   SetSellLimitOrder(id,lparam);
//---
   CloseAllMarketProfit(id,lparam);
   CloseAllMarketLoss(id,lparam);
//---
   DeleteAllPending(id,lparam);
```

This completes the development of the basic functionality for quick manual trading toolkit. The below video demonstrates the created application.

Fast Trading - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7892)

MQL5.community

1.91K subscribers

[Fast Trading](https://www.youtube.com/watch?v=Uzepj8lsDhY)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=Uzepj8lsDhY&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7892)

0:00

0:00 / 0:35

•Live

•

### Conclusion

The attached archive contains all the listed files, which are located in the appropriate folders. For their proper operation, you only need to save the MQL5 folder into the terminal folder. To open the terminal root directory, in which the MQL5 folder is located, press the Ctrl+Shift+D key combination in the MetaTrader 5 terminal or use the context menu as shown in Fig. 13 below.

### ![](https://c.mql5.com/2/40/004__13.jpg)      Fig.13 Opening the MQL5 folder in the MetaTrader 5 terminal root

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7892](https://www.mql5.com/ru/articles/7892)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7892.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7892/mql5.zip "Download MQL5.zip")(5621.18 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [A system of voice notifications for trade events and signals](https://www.mql5.com/en/articles/8111)
- [Quick Manual Trading Toolkit: Working with open positions and pending orders](https://www.mql5.com/en/articles/7981)
- [Multicurrency monitoring of trading signals (Part 5): Composite signals](https://www.mql5.com/en/articles/7759)
- [Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)
- [Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://www.mql5.com/en/articles/7600)
- [Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://www.mql5.com/en/articles/7528)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/353053)**
(8)


![RodgFX](https://c.mql5.com/avatar/2020/5/5EB586C2-CAC3.gif)

**[RodgFX](https://www.mql5.com/en/users/rodgfx)**
\|
15 May 2020 at 17:35

You still have to know how to trade pens. It's like the multiplication table. Even if you have a calculator handy, you have to know it.

![Aleksandr Masterskikh](https://c.mql5.com/avatar/2017/5/591837C6-87CC.jpg)

**[Aleksandr Masterskikh](https://www.mql5.com/en/users/a.masterskikh)**
\|
16 May 2020 at 10:53

**RodgFX:**

You still have to know how to trade pens. It's like the multiplication table. Even if you have a calculator at hand, you need to know it.

I completely agree - nobody cancels the intuitive component of trading.

Many traders who previously trusted robots unconditionally, come to the fact

that a signal is a signal, but often the final decision to [enter the market](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "Help: Opening and closing positions in MetaTrader 5 trading terminal") must be made by yourself.

![IuriiPrugov](https://c.mql5.com/avatar/avatar_na2.png)

**[IuriiPrugov](https://www.mql5.com/en/users/iuriiprugov)**
\|
18 May 2020 at 10:14

Thank you, but what you corrected or used a new version of the library

EasyAndFastGUI - library for creating [graphical interfaces](https://www.mql5.com/en/articles/2125 "Article: Graphical Interfaces I: Preparing the Library Structure (Chapter 1) ") \- library for MetaTrader 5 Anatoli Kazharski

because its latest version is 13.03.2019 and you have some files as far back as 20 years?

![Alexander Fedosov](https://c.mql5.com/avatar/2019/5/5CE6AA22-02C3.jpg)

**[Alexander Fedosov](https://www.mql5.com/en/users/alex2356)**
\|
18 May 2020 at 14:21

**IuriiPrugov:**

Thank you, but what you corrected or used a new version of the library

EasyAndFastGUI - library for creating graphical interfaces - library for MetaTrader 5 Anatoli Kazharski

because its latest version is 13.03.2019 and you have some files as far back as 20 years?

A lot of things, for the tasks that I had.

![andy912](https://c.mql5.com/avatar/avatar_na2.png)

**[andy912](https://www.mql5.com/en/users/andy912)**
\|
13 Dec 2020 at 15:46

Hi. Nice panel. Such a question: what is the easiest way to implement its access to [global variables](https://www.mql5.com/en/docs/basis/variables/global "MQL5 Documentation: Global Variables") and take the required value from the last created one? Is it necessary to check their presence, compare with the list of previous variables and then read the value? Or is there some simpler method?

Sorry if it's rambling, it's hard to explain everything at first glance

![Calculating mathematical expressions (Part 1). Recursive descent parsers](https://c.mql5.com/2/39/MQL5-avatar-analysis.png)[Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)

The article considers the basic principles of mathematical expression parsing and calculation. We will implement recursive descent parsers operating in the interpreter and fast calculation modes, based on a pre-built syntax tree.

![Practical application of neural networks in trading. It's time to practice](https://c.mql5.com/2/39/neural_DLL.png)[Practical application of neural networks in trading. It's time to practice](https://www.mql5.com/en/articles/7370)

The article provides a description and instructions for the practical use of neural network modules on the Matlab platform. It also covers the main aspects of creation of a trading system using the neural network module. In order to be able to introduce the complex within one article, I had to modify it so as to combine several neural network module functions in one program.

![Timeseries in DoEasy library (part 45): Multi-period indicator buffers](https://c.mql5.com/2/39/MQL5-avatar-doeasy-library__1.png)[Timeseries in DoEasy library (part 45): Multi-period indicator buffers](https://www.mql5.com/en/articles/8023)

In this article, I will start the improvement of the indicator buffer objects and collection class for working in multi-period and multi-symbol modes. I am going to consider the operation of buffer objects for receiving and displaying data from any timeframe on the current symbol chart.

![Practical application of neural networks in trading](https://c.mql5.com/2/37/neural_DLL.png)[Practical application of neural networks in trading](https://www.mql5.com/en/articles/7031)

In this article, we will consider the main aspects of integration of neural networks and the trading terminal, with the purpose of creating a fully featured trading robot.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/7892&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083478151788502139)

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