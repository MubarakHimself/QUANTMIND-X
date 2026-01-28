---
title: Quick Manual Trading Toolkit: Working with open positions and pending orders
url: https://www.mql5.com/en/articles/7981
categories: Trading, Expert Advisors
relevance_score: -5
scraped_at: 2026-01-24T14:18:11.735410
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=wxgxlrvwkfiomqfspvdvpohciyzzbkyc&ssn=1769253489234378552&ssn_dr=0&ssn_sr=0&fv_date=1769253489&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7981&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Quick%20Manual%20Trading%20Toolkit%3A%20Working%20with%20open%20positions%20and%20pending%20orders%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925348933126245&fz_uniq=5083476128858905712&sv=2552)

MetaTrader 5 / Trading


### Contents

- [Introduction](https://www.mql5.com/en/articles/7981#intro)
- [Formulation of the problem](https://www.mql5.com/en/articles/7981#task)
- [Revising the application structure](https://www.mql5.com/en/articles/7981#update)
- [Conclusion](https://www.mql5.com/en/articles/7981#final)

### Introduction

[We have previously created the basic functionality](https://www.mql5.com/en/articles/7892), which is designed to assist the traders who prefer manual trading. We mainly focused on convenient work related to order placing, and thus most of the functions were related to market entries. However, any trading strategy, whether it be manual or automatic, should have three main stages when working with the markets. These include market entry rules, open position management and closing conditions. As for now, the toolkit only covers the first stage. Therefore, as further development, we can add more opportunities for working with open positions or pending order, and to expand conditions for closing deals. All calculation should be performed by the toolkit, while the decision should be made by the trader.

### Formulation of the problem

Let us begin with determining the range of tasks required to implement the new functionality. Let us determine the main stages according to which we will continue developing the current application:

- **Revising the application structure.** The application has originally been created as a main window with a set of task buttons, two of which opened modal windows with tools for setting market positions and pending orders. The other three buttons provide a simple functionality for closing positions by market and for deleting all pending orders. Now, we need to add an interface for managing current positions and for editing pending orders.
- **Expanding order operation modes.** Previously, we could close all profitable or all losing orders. Now, we need a more flexible functionality, divided by position type and featuring the possibility to set various conditions for closing all/selected orders. As for pending orders, we could only mass close the orders earlier created by the toolkit. This is not enough. We should be able to delete a separate pending order or to modify it. Also, it would be nice to have separate lists of market orders and pending orders.

Let us take a closer look at each of the stages separately.

### Revising the application structure

Let us first view what has already been implemented. Figure 1 below shows the main blocks, which execute tasks in two categories: opening/creation and closing/deletion. The third category implies manual management and modification.

![](https://c.mql5.com/2/40/001__2.jpg)

Fig. 1 The main blocks of the toolkit

So, let us create three tabs. The first one will be used for functions shown in figure 1. In the other two tabs, we will implement functions for working with positions and pending orders.

![](https://c.mql5.com/2/40/002__2.jpg)

Fig.2 New application structure

All features that were previously located in the main window will be implemented in the Trading tab. Open Positions Control will feature a table of existing positions opened via this tool. Commands for working with these positions will also be implemented there. Pending Order Control will feature the orders created through the toolkit, as well as controls for closing and modifying open orders. Let us consider the tabs in more detail.

![](https://c.mql5.com/2/40/003__2.jpg)

Fig.3 Market Positions Control tab

Figure 3 shows Market Positions Control tab. The tab contains the following elements:

- **Market Orders tab.** Displays information on all current positions which have been opened by the application. It is similar to the table in the Trade tab in MetaTrader 5.
- **Three input fields.** They correspond and are linked to the table columns: volume, Stop Loss, Take Profit.
- **Two buttons.** After a click on a table row, editable position parameters will be shown in input fields. By clicking on Edit, we can change Stop Loss and Take Profit for a position selected in the list, and even delete them. The Close button closes a position at a current price. The system additionally checks the Volume input field when closing a position. It means that you can select the lot value less than the current position lot and close the position partially.

Now, have a look at the Pending Order Control tab in figure 4.

![](https://c.mql5.com/2/40/004.jpg)

Fig. 4 Pending orders control.

It is very similar to the previous one:

- A table of pending orders. It contains a list of pending orders that have been placed via this toolkit.
- Three input fields. When a pending order is selected in the table, you can modify its current execution price, as well as edit or delete Stop Loss and Take Profit.
- Two buttons. The Edit button, in contrast to the table of market positions, accesses all three input fields for editing the selected pending order. The Close button closes the order.

Let us get back to the Trading tab and modify it for the new functionality. First, let us revise the existing tools for closing profitable and losing positions. It will be possible to close short positions in addition to long ones. This is implemented through closing mode switchers.

![](https://c.mql5.com/2/40/005__1.jpg)

Fig. 5 Expanding market position closing functionality.

As you can see in figure 5, we have 4 new buttons: Close BUY profit, Close SELL profit, Close BUY loss, Close SELL loss. There are additional switchers to the right of the buttons; let us consider them in more detail. The switchers are similar, therefore the following descriptions apply to all of them.

- **All.** Default value. Does not set any restrictions to buttons and closes all selected items.
- **>Point**. Closes all positions of the selected type in which profit/loss is higher than the specified number of points.
- **>Currency.** Closes all positions of the selected type in which profit/loss is higher than the specified amount in the deposit currency.
- **Sum>Points**. Closes all positions of the selected type having the total profit/loss higher than the specified number of points.
- **Sum>Currency**. Closes all positions of the selected type having the total profit/loss higher than the specified amount in the deposit currency.

**Important note:** when one of the last two points is selected, the total amount is calculated **only for positions opened by this application**, with the specified magic number. The amount is checked for all positions that meet two criteria: position type and magic number.

Let us consider an example of settings shown in figure 5: Close All Loss and option sum>currency. In this case, the toolkit will find all positions that it has previously opened, summarize their profit, and the loss is more than 10 units in the deposit currency, it will close all of them.

### Implementation of toolkit additions

As the basis, we will use the earlier created project from article [Quick Manual Trading Toolkit: Basic Functionality](https://www.mql5.com/en/articles/7892). First, we need to restructure the main window as is shown in fig.2. To do this, add the Tab interface element by creating the **CreateTabs()** method in the **CProgram** base class, and implement it in **MainWindow.mqh**.

```
//+------------------------------------------------------------------+
//| Create a group with tabs                                         |
//+------------------------------------------------------------------+
bool CFastTrading::CreateTabs(const int x_gap,const int y_gap)
{
//--- Store the pointer to the main control
   m_tab.MainPointer(m_main_window);
//--- Properties
   m_tab.Font(m_base_font);
   m_tab.FontSize(m_base_font_size);
   m_tab.LabelColor(clrWhite);
   m_tab.LabelColorHover(clrWhite);
   m_tab.IsCenterText(true);
   m_tab.AutoXResizeMode(true);
   m_tab.AutoYResizeMode(true);
   m_tab.AutoXResizeRightOffset(5);
   m_tab.AutoYResizeBottomOffset(5);
   m_tab.TabsYSize(27);
   m_tab.GetButtonsGroupPointer().Font(m_base_font);
   m_tab.GetButtonsGroupPointer().FontSize(m_base_font_size);
//--- Add tabs with the specified properties
   string tabs_names[3];
   tabs_names[0]=TRADING;
   tabs_names[1]=CAPTION_M_CONTROL_NAME;
   tabs_names[2]=CAPTION_P_CONTROL_NAME;
   for(int i=0; i<3; i++)
      m_tab.AddTab(tabs_names[i],180);
//--- Create a control element
   if(!m_tab.CreateTabs(x_gap,y_gap))
      return(false);
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,m_tab);
   return(true);
}
```

The created method body has three new macro substitutions, which serve as tab headers - they need to be added in two languages into the **Defines.mqh** file:

```
#define TRADING                        (m_language==RUSSIAN ? "Трейдинг" : "Trading")
#define CAPTION_M_CONTROL_NAME         (m_language==RUSSIAN ? "Контроль рыночных позиций" : "Market Control")
#define CAPTION_P_CONTROL_NAME         (m_language==RUSSIAN ? "Контроль отложенных ордеров" : "Pending Control")
```

Before applying the newly created method, we need to rebuild the methods that create buttons and link these methods to the first tab. Let us find the common method **CreateButton()** and edit it as follows:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateButton(CButton &button,string text,color baseclr,int x_gap,int y_gap)
{
//--- Store the window pointer
   button.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(0,button);
//--- Set properties before creation
   button.XSize(180);
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
   CWndContainer::AddToElementsArray(0,button);
   return(true);
}
```

Now, apply the changes in the main window creating method.

```
//+------------------------------------------------------------------+
//| Creates a form for orders                                        |
//+------------------------------------------------------------------+
bool CFastTrading::CreateMainWindow(void)
{
//--- Add a window pointer to the window array
   CWndContainer::AddWindow(m_main_window);
//--- Properties
   m_main_window.XSize(600);
   m_main_window.YSize(375);
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
//--- Tabs
   if(!CreateTabs(5,22+27))
      return(false);
//---
   if(!CreateButton(m_order_button[0],MARKET_ORDER_NAME+"(M)",C'87,128,255',20,10))
      return(false);
   if(!CreateButton(m_order_button[1],PENDING_ORDER_NAME+"(P)",C'31,209,111',210,10))
      return(false);
   if(!CreateButton(m_order_button[2],MARKET_ORDERS_PROFIT_CLOSE+"(C)",C'87,128,255',20,60))
      return(false);
   if(!CreateButton(m_order_button[3],MARKET_ORDERS_LOSS_CLOSE+"(D)",C'87,128,255',20,110))
      return(false);
   if(!CreateButton(m_order_button[4],PEND_ORDERS_ALL_CLOSE+"(R)",C'31,209,111',210,60))
      return(false);
//---
   return(true);
}
```

As can be seen here, we have first added tabs, then updated calls of methods that create buttons and resized the main window. After project compilation, all buttons will be moved to the first tab:

![](https://c.mql5.com/2/41/006.jpg)

Fig. 6 Creating tabs and moving buttons to the first tab

According to figure 5, let us create additional buttons and input fields to implement the required functionality. We will use the updated **CreateButton()** method for large buttons. However, to create input fields and switchers, we need to introduce additional methods: **CreateModeButton()** — mode switcher, **CreateModeEdit()** — input fields. Their full implementation is as follows:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateModeButton(CButton &button,string text,int x_gap,int y_gap)
{
//--- Store the window pointer
   button.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(0,button);
   color baseclr=clrDarkViolet;
//--- Set properties before creation
   button.XSize(80);
   button.YSize(20);
   button.Font(m_base_font);
   button.FontSize(9);
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
   CWndContainer::AddToElementsArray(0,button);
   return(true);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateModeEdit(CTextEdit &text_edit,int x_gap,int y_gap)
{
//--- Store the window pointer
   text_edit.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(0,text_edit);
//--- Properties
   text_edit.XSize(80);
   text_edit.YSize(20);
   text_edit.Font(m_base_font);
   text_edit.FontSize(m_base_font_size);
   text_edit.SetDigits(2);
   text_edit.MaxValue(99999);
   text_edit.StepValue(0.01);
   text_edit.MinValue(0.01);
   text_edit.SpinEditMode(true);
   text_edit.GetTextBoxPointer().XGap(1);
   text_edit.GetTextBoxPointer().XSize(80);
//--- Create a control element
   if(!text_edit.CreateTextEdit("",x_gap,y_gap))
      return(false);
   text_edit.SetValue(string(0.01));
   text_edit.IsLocked(true);
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,text_edit);
   return(true);
}
```

Below is the updated method that creates the main window using the above methods, and adds buttons from figure 5:

```
//+------------------------------------------------------------------+
//| Creates a form for orders                                        |
//+------------------------------------------------------------------+
bool CFastTrading::CreateMainWindow(void)
{
//--- Add a window pointer to the window array
   CWndContainer::AddWindow(m_main_window);
//--- Properties
   m_main_window.XSize(600);
   m_main_window.YSize(375);
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
//--- Tabs
   if(!CreateTabs(5,22+27))
      return(false);
//---
   if(!CreateButton(m_order_button[0],MARKET_ORDER_NAME+"(M)",C'87,128,255',10,10))
      return(false);
   if(!CreateButton(m_order_button[1],PENDING_ORDER_NAME+"(P)",C'31,209,111',300,10))
      return(false);
   if(!CreateButton(m_order_button[2],CLOSE_ALL_PROFIT+"(C)",C'87,128,255',10,10+45*1))
      return(false);
   if(!CreateButton(m_order_button[3],PEND_ORDERS_ALL_CLOSE+"(R)",C'31,209,111',300,10+45*1))
      return(false);
   if(!CreateButton(m_order_button[4],CLOSE_BUY_PROFIT,C'87,128,255',10,10+45*2))
      return(false);
   if(!CreateButton(m_order_button[5],CLOSE_SELL_PROFIT,C'87,128,255',10,10+45*3))
      return(false);
   if(!CreateButton(m_order_button[6],CLOSE_ALL_LOSS+"(D)",C'87,128,255',10,10+45*4))
      return(false);
   if(!CreateButton(m_order_button[7],CLOSE_BUY_LOSS,C'87,128,255',10,10+45*5))
      return(false);
   if(!CreateButton(m_order_button[8],CLOSE_SELL_LOSS,C'87,128,255',10,10+45*6))
      return(false);
//---
   for(int i=0; i<6; i++)
   {
      if(!CreateModeButton(m_mode_button[i],"all",205-10,10+45*(i+1)))
         return(false);
      if(!CreateModeEdit(m_mode_edit[i],204-10,30+45*(i+1)))
         return(false);
   }
//---
   return(true);
}
```

New micro substitutions are used here, so the appropriate values should be added to **Defines.mqh**:

```
#define CLOSE_BUY_PROFIT               (m_language==RUSSIAN ? "Закрыть BUY прибыльные" : "Close BUY Profit")
#define CLOSE_SELL_PROFIT              (m_language==RUSSIAN ? "Закрыть SELL прибыльные" : "Close SELL Profit")
#define CLOSE_ALL_PROFIT               (m_language==RUSSIAN ? "Закрыть ВСЕ прибыльные" : "Close ALL Profit")
#define CLOSE_BUY_LOSS                 (m_language==RUSSIAN ? "Закрыть BUY убыточные" : "Close BUY Losing")
#define CLOSE_SELL_LOSS                (m_language==RUSSIAN ? "Закрыть SELL убыточные" : "Close SELL Losing")
#define CLOSE_ALL_LOSS                 (m_language==RUSSIAN ? "Закрыть ВСЕ убыточные" : "Close ALL Losing")
```

Compile the project and check the intermediate result:

![](https://c.mql5.com/2/41/007.jpg)

Fig. 7 Adding buttons and mode switches

But this is only a visual implementation. The next step is to assign a logical task for each of the added elements. Let us set up the switch mechanism, because all subsequent elements will refer to their values and states. Create a new method **ModeButtonSwitch()** for button objects. It will switch modes when a button is pressed.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::ModeButtonSwitch(CButton &button,long lparam,string &states[])
{
   if(lparam==button.Id())
   {
      int size=ArraySize(states);
      for(int i=0; i<size; i++)
      {
         if(button.LabelText()==states[i])
         {
            if(i==size-1)
            {
               SetButtonParam(button,states[0]);
               break;
            }
            else
            {
               SetButtonParam(button,states[i+1]);
               break;
            }
         }
      }
   }
}
```

Another new method **ModeEditSwitch()** provides the correspondence of input field settings to the selected mode. For example, points are integer numbers, and when we use deposit currency, values should have 2 decimal places.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::ModeEditSwitch(long lparam,string &states[])
{
   for(int i=0; i<6; i++)
   {
      if(lparam==m_mode_button[i].Id())
      {
         if(m_mode_button[i].LabelText()==states[1])
         {
            m_mode_edit[i].IsLocked(false);
            m_mode_edit[i].SetDigits(0);
            m_mode_edit[i].StepValue(1);
            m_mode_edit[i].MaxValue(9999);
            m_mode_edit[i].MinValue(1);
            m_mode_edit[i].SetValue(string(20));
            m_mode_edit[i].GetTextBoxPointer().Update(true);
            m_current_mode[i]=1;
         }
         else if(m_mode_button[i].LabelText()==states[2])
         {
            m_mode_edit[i].IsLocked(false);
            m_mode_edit[i].SetDigits(2);
            m_mode_edit[i].StepValue(0.01);
            m_mode_edit[i].MaxValue(99999);
            m_mode_edit[i].MinValue(0.01);
            m_mode_edit[i].SetValue(string(0.1));
            m_mode_edit[i].GetTextBoxPointer().Update(true);
            m_current_mode[i]=2;
         }
         else if(m_mode_button[i].LabelText()==states[3])
         {
            m_mode_edit[i].IsLocked(false);
            m_mode_edit[i].SetDigits(0);
            m_mode_edit[i].StepValue(1);
            m_mode_edit[i].MaxValue(9999);
            m_mode_edit[i].MinValue(1);
            m_mode_edit[i].SetValue(string(20));
            m_mode_edit[i].GetTextBoxPointer().Update(true);
            m_current_mode[i]=3;
         }
         else if(m_mode_button[i].LabelText()==states[4])
         {
            m_mode_edit[i].IsLocked(false);
            m_mode_edit[i].SetDigits(2);
            m_mode_edit[i].StepValue(0.01);
            m_mode_edit[i].MaxValue(99999);
            m_mode_edit[i].MinValue(0.01);
            m_mode_edit[i].SetValue(string(0.1));
            m_mode_edit[i].GetTextBoxPointer().Update(true);
            m_current_mode[i]=4;
         }
         else
         {
            m_mode_edit[i].IsLocked(true);
            m_current_mode[i]=0;
         }
      }
   }
}
```

The current implementation has a static array m\_current\_mode — its size corresponds to the number of mode switchers, i.e. 6. Modes of each closing button, selected by the user, are written to this array. To activate the newly added methods, open **OnEvent()** event handler and add the following code in the button click event:

```
      //---
      string states[5]= {"all",">points",">currency","sum>points","sum>currency"};
      for(int i=0; i<6; i++)
         ModeButtonSwitch(m_mode_button[i],lparam,states);
      //---
      ModeEditSwitch(lparam,states);
```

Compile the project. Now you can see that mode switching changes the properties of input fields. This is shown in Fig. 8.

![](https://c.mql5.com/2/41/008.gif)

Fig. 8 Switching the market position closing modes

The next step is to implement the action logic according to button descriptions, which should also be linked to the earlier added modes. We already have two actions: "Close all profitable" and "Close all losing". Now, they should be expanded in accordance with the new closing modes. These actions are performed by methods **CloseAllMarketProfit()** and **CloseAllMarketLoss()**.

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
      int sum_pp=0;
      double sum_cur=0.0;
      //--- Calculate profit for modes 4 and 5
      if(m_current_mode[0]>2)
      {
         for(int i=total-1; i>=0; i--)
         {
            //--- order parameters
            ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
            string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
            ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
            ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
            double profit_cur=PositionGetDouble(POSITION_PROFIT);
            double swap=PositionGetDouble(POSITION_SWAP);
            //---
            int profit_pp;
            if(type==POSITION_TYPE_BUY)
               profit_pp=int((SymbolInfoDouble(position_symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN))/_Point);
            else
               profit_pp=int((PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(position_symbol,SYMBOL_ASK))/_Point);
            //---
            if(magic==m_magic_number)
               if(position_symbol==Symbol())
               {
                  switch(m_current_mode[0])
                  {
                  case  3:
                     sum_pp+=profit_pp;
                     break;
                  case  4:
                     sum_cur+=profit_cur+swap;
                     break;
                  }
               }
         }
      }
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
         double profit_cur=PositionGetDouble(POSITION_PROFIT);
         double swap=PositionGetDouble(POSITION_SWAP);
         int profit_pp;
         //---
         if(type==POSITION_TYPE_BUY)
            profit_pp=int((SymbolInfoDouble(position_symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN))/_Point);
         else
            profit_pp=int((PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(position_symbol,SYMBOL_ASK))/_Point);
         //--- if MagicNumber matches
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
            {
               if(   (m_current_mode[0]==0 && profit_cur+swap>0) ||                                   // Close all profitable positions
                     (m_current_mode[0]==1 && profit_pp>int(m_mode_edit[0].GetValue())) ||            // Close all positions having profit more than N points
                     (m_current_mode[0]==2 && profit_cur+swap>double(m_mode_edit[0].GetValue())) ||   // Close all positions having profit more than N deposit currency units
                     (m_current_mode[0]==3 && sum_pp>int(m_mode_edit[0].GetValue())) ||               // Close all positions with the total profit more than N points
                     (m_current_mode[0]==4 && sum_cur>double(m_mode_edit[0].GetValue()))              // Close all positions with the total profit more than N deposit currency units
                 )
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
                  bool res=true;
                  for(int j=0; j<5; j++)
                  {
                     res=OrderSend(request,result);
                     if(res && result.retcode==TRADE_RETCODE_DONE)
                        break;
                     else
                        PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
                  }
               }
            }
      }
   }
}
```

This is a modification of the method closing all market positions. Earlier, we have introduced the m\_current\_mode static array which tracks the mode selection for each of the actions on the button. Therefore, calculation is made for modes 4 and 5, in which all positions are closed by total profit in points or in deposit currency. After that, we select positions that belong to our toolkit and, depending on the selected closing mode, select conditions, upon which all market positions created by the toolkit should be closed.

Similarly, change the second method that closes all losing positions:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::CloseAllMarketLoss(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_order_button[6].Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_D))
   {
      //--- declare the request and the result
      MqlTradeRequest request;
      MqlTradeResult  result;
      ZeroMemory(request);
      ZeroMemory(result);
      int total=PositionsTotal(); // the number of open positions
      int sum_pp=0;
      double sum_cur=0.0;
      //--- Calculate losses
      if(m_current_mode[3]>2)
      {
         for(int i=total-1; i>=0; i--)
         {
            //--- order parameters
            ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
            string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
            ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
            ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
            double loss_cur=PositionGetDouble(POSITION_PROFIT);
            double swap=PositionGetDouble(POSITION_SWAP);
            int loss_pp;
            //---
            if(type==POSITION_TYPE_BUY)
               loss_pp=int((SymbolInfoDouble(position_symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN))/_Point);
            else
               loss_pp=int((PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(position_symbol,SYMBOL_ASK))/_Point);
            //---
            if(magic==m_magic_number)
               if(position_symbol==Symbol())
               {
                  switch(m_current_mode[3])
                  {
                  case  3:
                     sum_pp+=loss_pp;
                     break;
                  case  4:
                     sum_cur+=loss_cur+swap;
                     break;
                  }
               }
         }
      }
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
         double loss_cur=PositionGetDouble(POSITION_PROFIT);
         double swap=PositionGetDouble(POSITION_SWAP);
         int loss_pp;
         if(type==POSITION_TYPE_BUY)
            loss_pp=int((SymbolInfoDouble(position_symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN))/_Point);
         else
            loss_pp=int((PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(position_symbol,SYMBOL_ASK))/_Point);
         //--- if MagicNumber matches
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
            {
               if(   (m_current_mode[3]==0 && loss_cur+swap<0) ||
                     (m_current_mode[3]==1 && loss_pp<-int(m_mode_edit[3].GetValue())) ||
                     (m_current_mode[3]==2 && loss_cur+swap<-double(m_mode_edit[3].GetValue())) ||
                     (m_current_mode[3]==3 && sum_pp<-int(m_mode_edit[3].GetValue())) ||
                     (m_current_mode[3]==4 && sum_cur<-double(m_mode_edit[3].GetValue()))
                 )
               {
                  //--- zeroing the request and result values
                  ZeroMemory(request);
                  ZeroMemory(result);
                  //--- set the operation parameters
                  request.action=TRADE_ACTION_DEAL;         // trading operation type
                  request.position=position_ticket;         // position ticket
                  request.symbol=position_symbol;           // symbol
                  request.volume=volume;                    // position volume
                  request.deviation=5;                      // allowable price deviation
                  request.magic=m_magic_number;             // position MagicNumber
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
                  bool res=true;
                  for(int j=0; j<5; j++)
                  {
                     res=OrderSend(request,result);
                     if(res && result.retcode==TRADE_RETCODE_DONE)
                        break;
                     else
                        PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
                  }
               }
            }
      }
   }
}
```

Also, calculation is performed for modes, in which the total loss of all open positions in points or in deposit currency is checked and all market positions created by the toolkit are closed if the appropriate loss conditions are met.

Now let us move on to new actions with open positions: Closing Buy/Sell positions, either profitable or losing. In fact, these are special cases of the two methods described above. Thus, we will add filtering by position type. First, create methods performing the specified actions:

- **CloseBuyMarketProfit()** — closes all profitable Buy positions.
- **CloseSellMarketProfit()** — closes all profitable Sell positions.
- **CloseBuyMarketLoss()** — closes all losing Buy positions.
- **CloseSellMarketLoss()** — close all losing Sell positions.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::CloseBuyMarketProfit(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_order_button[4].Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_U))
   {
      //--- declare the request and the result
      MqlTradeRequest request;
      MqlTradeResult  result;
      int total=PositionsTotal(); // the number of open positions
      int sum_pp=0;
      double sum_cur=0.0;
      //--- Calculate profit for modes 4 and 5
      if(m_current_mode[1]>2)
      {
         for(int i=total-1; i>=0; i--)
         {
            //--- order parameters
            ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
            string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
            ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
            ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
            double profit_cur=PositionGetDouble(POSITION_PROFIT);
            double swap=PositionGetDouble(POSITION_SWAP);
            //---
            if(magic==m_magic_number)
               if(position_symbol==Symbol())
                  if(type==POSITION_TYPE_BUY)
                  {
                     int profit_pp=int((SymbolInfoDouble(position_symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN))/_Point);
                     switch(m_current_mode[1])
                     {
                     case  3:
                        sum_pp+=profit_pp;
                        break;
                     case  4:
                        sum_cur+=profit_cur+swap;
                        break;
                     }
                  }
         }
      }
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
         double profit_cur=PositionGetDouble(POSITION_PROFIT);
         double swap=PositionGetDouble(POSITION_SWAP);
         //---
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
               if(type==POSITION_TYPE_BUY)
               {
                  int profit_pp=int((SymbolInfoDouble(position_symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN))/_Point);
                  if(   (m_current_mode[1]==0 && profit_cur+swap>0) ||                                   // Close all profitable positions
                        (m_current_mode[1]==1 && profit_pp>int(m_mode_edit[1].GetValue())) ||            // Close all positions having profit more than N points
                        (m_current_mode[1]==2 && profit_cur+swap>double(m_mode_edit[1].GetValue())) ||   // Close all positions having profit more than N deposit currency units
                        (m_current_mode[1]==3 && sum_pp>int(m_mode_edit[1].GetValue())) ||               // Close all positions with the total profit more than N points
                        (m_current_mode[1]==4 && sum_cur>double(m_mode_edit[1].GetValue()))              // Close all positions with the total profit more than N deposit currency units
                    )
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
                     request.price=SymbolInfoDouble(position_symbol,SYMBOL_BID);
                     request.type =ORDER_TYPE_SELL;
                     //--- sending a request
                     bool res=true;
                     for(int j=0; j<5; j++)
                     {
                        res=OrderSend(request,result);
                        if(res && result.retcode==TRADE_RETCODE_DONE)
                           break;
                        else
                           PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
                     }
                  }
               }
      }
   }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::CloseBuyMarketLoss(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_order_button[7].Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_H))
   {
      //--- declare the request and the result
      MqlTradeRequest request;
      MqlTradeResult  result;
      int total=PositionsTotal(); // the number of open positions
      int sum_pp=0;
      double sum_cur=0.0;
      //--- Calculate profit for modes 4 and 5
      if(m_current_mode[4]>2)
      {
         for(int i=total-1; i>=0; i--)
         {
            //--- order parameters
            ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
            string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
            ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
            ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
            double profit_cur=PositionGetDouble(POSITION_PROFIT);
            double swap=PositionGetDouble(POSITION_SWAP);
            //---
            if(magic==m_magic_number)
               if(position_symbol==Symbol())
                  if(type==POSITION_TYPE_BUY)
                  {
                     int profit_pp=int((SymbolInfoDouble(position_symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN))/_Point);
                     switch(m_current_mode[1])
                     {
                     case  3:
                        sum_pp+=profit_pp;
                        break;
                     case  4:
                        sum_cur+=profit_cur+swap;
                        break;
                     }
                  }
         }
      }
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
         double profit_cur=PositionGetDouble(POSITION_PROFIT);
         double swap=PositionGetDouble(POSITION_SWAP);
         //---
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
               if(type==POSITION_TYPE_BUY)
               {
                  int profit_pp=int((SymbolInfoDouble(position_symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN))/_Point);
                  if(   (m_current_mode[4]==0 && profit_cur+swap<0) ||                                   // Close all profitable positions
                        (m_current_mode[4]==1 && profit_pp<-int(m_mode_edit[4].GetValue())) ||            // Close all positions having profit more than N points
                        (m_current_mode[4]==2 && profit_cur+swap<-double(m_mode_edit[4].GetValue())) ||   // Close all positions having profit more than N deposit currency units
                        (m_current_mode[4]==3 && sum_pp<-int(m_mode_edit[4].GetValue())) ||               // Close all positions with the total profit more than N points
                        (m_current_mode[4]==4 && sum_cur<-double(m_mode_edit[4].GetValue()))              // Close all positions with the total profit more than N deposit currency units
                    )
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
                     request.price=SymbolInfoDouble(position_symbol,SYMBOL_BID);
                     request.type =ORDER_TYPE_SELL;
                     //--- sending a request
                     bool res=true;
                     for(int j=0; j<5; j++)
                     {
                        res=OrderSend(request,result);
                        if(res && result.retcode==TRADE_RETCODE_DONE)
                           break;
                        else
                           PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
                     }
                  }
               }
      }
   }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::CloseSellMarketProfit(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_order_button[5].Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_J))
   {
      //--- declare the request and the result
      MqlTradeRequest request;
      MqlTradeResult  result;
      int total=PositionsTotal(); // the number of open positions
      int sum_pp=0;
      double sum_cur=0.0;
      //--- Calculate profit for modes 4 and 5
      if(m_current_mode[2]>2)
      {
         for(int i=total-1; i>=0; i--)
         {
            //--- order parameters
            ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
            string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
            ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
            ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
            double profit_cur=PositionGetDouble(POSITION_PROFIT);
            double swap=PositionGetDouble(POSITION_SWAP);
            //---
            if(magic==m_magic_number)
               if(position_symbol==Symbol())
                  if(type==POSITION_TYPE_SELL)
                  {
                     int profit_pp=int((PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(position_symbol,SYMBOL_ASK))/_Point);
                     switch(m_current_mode[2])
                     {
                     case  3:
                        sum_pp+=profit_pp;
                        break;
                     case  4:
                        sum_cur+=profit_cur+swap;
                        break;
                     }
                  }
         }
      }
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
         double profit_cur=PositionGetDouble(POSITION_PROFIT);
         double swap=PositionGetDouble(POSITION_SWAP);
         //---
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
               if(type==POSITION_TYPE_SELL)
               {
                  int profit_pp=int((PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(position_symbol,SYMBOL_ASK))/_Point);
                  if(   (m_current_mode[2]==0 && profit_cur+swap>0) ||                                   // Close all profitable positions
                        (m_current_mode[2]==1 && profit_pp>int(m_mode_edit[2].GetValue())) ||            // Close all positions having profit more than N points
                        (m_current_mode[2]==2 && profit_cur+swap>double(m_mode_edit[2].GetValue())) ||   // Close all positions having profit more than N deposit currency units
                        (m_current_mode[2]==3 && sum_pp>int(m_mode_edit[2].GetValue())) ||               // Close all positions with the total profit more than N points
                        (m_current_mode[2]==4 && sum_cur>double(m_mode_edit[2].GetValue()))              // Close all positions with the total profit more than N deposit currency units
                    )
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
                     request.price=SymbolInfoDouble(position_symbol,SYMBOL_ASK);
                     request.type =ORDER_TYPE_BUY;
                     //--- sending a request
                     bool res=true;
                     for(int j=0; j<5; j++)
                     {
                        res=OrderSend(request,result);
                        if(res && result.retcode==TRADE_RETCODE_DONE)
                           break;
                        else
                           PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
                     }
                  }
               }
      }
   }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::CloseSellMarketLoss(int id,long lparam)
{
   if((id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON && lparam==m_order_button[8].Id()) ||
         (id==CHARTEVENT_KEYDOWN && lparam==KEY_L))
   {
      //--- declare the request and the result
      MqlTradeRequest request;
      MqlTradeResult  result;
      int total=PositionsTotal(); // the number of open positions
      int sum_pp=0;
      double sum_cur=0.0;
      //--- Calculate profit for modes 4 and 5
      if(m_current_mode[5]>2)
      {
         for(int i=total-1; i>=0; i--)
         {
            //--- order parameters
            ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
            string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
            ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
            ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
            double profit_cur=PositionGetDouble(POSITION_PROFIT);
            double swap=PositionGetDouble(POSITION_SWAP);
            //---
            if(magic==m_magic_number)
               if(position_symbol==Symbol())
                  if(type==POSITION_TYPE_SELL)
                  {
                     int profit_pp=int((PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(position_symbol,SYMBOL_ASK))/_Point);
                     switch(m_current_mode[1])
                     {
                     case  3:
                        sum_pp+=profit_pp;
                        break;
                     case  4:
                        sum_cur+=profit_cur+swap;
                        break;
                     }
                  }
         }
      }
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
         double profit_cur=PositionGetDouble(POSITION_PROFIT);
         double swap=PositionGetDouble(POSITION_SWAP);
         //---
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
               if(type==POSITION_TYPE_SELL)
               {
                  int profit_pp=int((PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(position_symbol,SYMBOL_ASK))/_Point);
                  if(   (m_current_mode[5]==0 && profit_cur+swap<0) ||                                   // Close all profitable positions
                        (m_current_mode[5]==1 && profit_pp<-int(m_mode_edit[5].GetValue())) ||            // Close all positions having profit more than N points
                        (m_current_mode[5]==2 && profit_cur+swap<-double(m_mode_edit[5].GetValue())) ||   // Close all positions having profit more than N deposit currency units
                        (m_current_mode[5]==3 && sum_pp<-int(m_mode_edit[5].GetValue())) ||               // Close all positions with the total profit more than N points
                        (m_current_mode[5]==4 && sum_cur<-double(m_mode_edit[5].GetValue()))              // Close all positions with the total profit more than N deposit currency units
                    )
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
                     request.price=SymbolInfoDouble(position_symbol,SYMBOL_ASK);
                     request.type =ORDER_TYPE_BUY;
                     //--- sending a request
                     bool res=true;
                     for(int j=0; j<5; j++)
                     {
                        res=OrderSend(request,result);
                        if(res && result.retcode==TRADE_RETCODE_DONE)
                           break;
                        else
                           PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
                     }
                  }
               }
      }
   }
}
```

After creation and implementation, call the methods in the **OnEvent()** event handler body, without any sections.

```
//---
   CloseBuyMarketProfit(id,lparam);
   CloseSellMarketProfit(id,lparam);
   CloseBuyMarketLoss(id,lparam);
   CloseSellMarketLoss(id,lparam);
```

Each of the actions can be performed not only by a button click, but also by keypress events. You can reassign hotkeys in the code. For convenience, let us display their values next to action names in buttons.

```
//---
   if(!CreateButton(m_order_button[0],MARKET_ORDER_NAME+"(M)",C'87,128,255',10,10))
      return(false);
   if(!CreateButton(m_order_button[1],PENDING_ORDER_NAME+"(P)",C'31,209,111',300,10))
      return(false);
   if(!CreateButton(m_order_button[2],CLOSE_ALL_PROFIT+"(C)",C'87,128,255',10,10+45*1))
      return(false);
   if(!CreateButton(m_order_button[3],PEND_ORDERS_ALL_CLOSE+"(R)",C'31,209,111',300,10+45*1))
      return(false);
   if(!CreateButton(m_order_button[4],CLOSE_BUY_PROFIT+"(U)",C'87,128,255',10,10+45*2))
      return(false);
   if(!CreateButton(m_order_button[5],CLOSE_SELL_PROFIT+"(J)",C'87,128,255',10,10+45*3))
      return(false);
   if(!CreateButton(m_order_button[6],CLOSE_ALL_LOSS+"(D)",C'87,128,255',10,10+45*4))
      return(false);
   if(!CreateButton(m_order_button[7],CLOSE_BUY_LOSS+"(H)",C'87,128,255',10,10+45*5))
      return(false);
   if(!CreateButton(m_order_button[8],CLOSE_SELL_LOSS+"(L)",C'87,128,255',10,10+45*6))
      return(false);
```

As a result, we have the following values:

![](https://c.mql5.com/2/41/009.jpg)

Fig. 9 Assigning hotkeys to added actions

We have implemented all the functionality of the Trading tab. Now, let us move on to the next stage: creating a table of positions opened by the toolkit and adding the possibility to manage positions on the tab "Market Positions Control". Figure 3 at the beginning of the article contains a visual scheme for creating interface elements, which consists of three input fields, two buttons and the table. Let us start creating the tab. First, we will create three input fields for editing the lot, Stop loss and Take profit of open positions. These is done in methods **CreateLotControl(), CreateStopLossControl() and CreateTakeProfitControl().**

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateLotControl(CTextEdit &text_edit,const int x_gap,const int y_gap,int tab)
{
//--- Store the pointer to the main control
   text_edit.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(tab,text_edit);
//--- Properties
   text_edit.XSize(80);
   text_edit.YSize(24);
   text_edit.Font(m_base_font);
   text_edit.FontSize(m_base_font_size);
   text_edit.MaxValue(9999);
   text_edit.StepValue(_Point);
   text_edit.SetDigits(_Digits);
   text_edit.SpinEditMode(true);
   text_edit.GetTextBoxPointer().XGap(1);
   text_edit.GetTextBoxPointer().XSize(80);
//--- Create a control element
   if(!text_edit.CreateTextEdit("",x_gap,y_gap))
      return(false);
   text_edit.SetValue(string(0));
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,text_edit);
   return(true);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateTakeProfitControl(CTextEdit &text_edit,const int x_gap,const int y_gap,int tab)
{
//--- Store the pointer to the main control
   text_edit.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(tab,text_edit);
//--- Properties
   text_edit.XSize(80);
   text_edit.YSize(24);
   text_edit.Font(m_base_font);
   text_edit.FontSize(m_base_font_size);
   text_edit.MaxValue(9999);
   text_edit.StepValue(_Point);
   text_edit.SetDigits(_Digits);
   text_edit.SpinEditMode(true);
   text_edit.GetTextBoxPointer().XGap(1);
   text_edit.GetTextBoxPointer().XSize(80);
//--- Create a control element
   if(!text_edit.CreateTextEdit("",x_gap,y_gap))
      return(false);
   text_edit.SetValue(string(0));
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,text_edit);
   return(true);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateStopLossControl(CTextEdit &text_edit,const int x_gap,const int y_gap,int tab)
{
//--- Store the pointer to the main control
   text_edit.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(tab,text_edit);
//--- Properties
   text_edit.XSize(80);
   text_edit.YSize(24);
   text_edit.Font(m_base_font);
   text_edit.FontSize(m_base_font_size);
   text_edit.MaxValue(9999);
   text_edit.StepValue(_Point);
   text_edit.SetDigits(_Digits);
   text_edit.MinValue(0);
   text_edit.SpinEditMode(true);
   text_edit.GetTextBoxPointer().XGap(1);
   text_edit.GetTextBoxPointer().XSize(80);
//--- Create a control element
   if(!text_edit.CreateTextEdit("",x_gap,y_gap))
      return(false);
   text_edit.SetValue(string(0));
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,text_edit);
   return(true);
}
```

Add new methods in the **CreateMainWindow()** body.

```
//--- Input field for editing open positions
   if(!CreateLotControl(m_lot_edit[6],375,3,1))
      return(false);
   if(!CreateStopLossControl(m_sl_edit[6],375+80,3,1))
      return(false);
   if(!CreateTakeProfitControl(m_tp_edit[6],375+83*2,3,1))
      return(false);
```

For the 'Edit' and 'Close' buttons, we need to create two new methods that implement them: **CreateModifyButton()** and **CreateCloseButton()**.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateModifyButton(CButton &button,string text,int x_gap,int y_gap,int tab)
{
//--- Store the window pointer
   button.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(tab,button);
   color baseclr=clrDarkOrange;
   color pressclr=clrOrange;
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
//--- Create a control element
   if(!button.CreateButton(text,x_gap,y_gap))
      return(false);
//--- Add a pointer to the element to the database
   CWndContainer::AddToElementsArray(0,button);
   return(true);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::CreateCloseButton(CButton &button,string text,int x_gap,int y_gap,int tab)
{
//--- Store the window pointer
   button.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(tab,button);
   color baseclr=clrCrimson;
   color pressclr=clrFireBrick;
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
//--- Create a control element
   if(!button.CreateButton(text,x_gap,y_gap))
      return(false);
//--- Add a pointer to the element to the database
   CWndContainer::AddToElementsArray(0,button);
   return(true);
}
```

Add them to the main window creation method:

```
//--- Position editing/closing buttons
   if(!CreateModifyButton(m_small_button[0],MODIFY,622,3,1))
      return(false);
   if(!CreateCloseButton(m_small_button[1],CLOSE,622+80,3,1))
      return(false);
```

Here we have two new macro substitutions for UI localization, so open **Defines.mqh** and add the appropriate values:

```
#define MODIFY                         (m_language==RUSSIAN ? "Изменить" : "Modify")
#define CLOSE                          (m_language==RUSSIAN ? "Закрыть" : "Close")
```

Now, let us move on to the table. First, create the **CreatePositionsTable()** method, implement it and add to the main window method.

```
//+------------------------------------------------------------------+
//| Create a table of positions                                      |
//+------------------------------------------------------------------+
bool CFastTrading::CreatePositionsTable(CTable &table,const int x_gap,const int y_gap)
{
#define COLUMNS2_TOTAL 9
//--- Store the pointer to the main control
   table.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(1,table);
//--- Array of column widths
   int width[COLUMNS2_TOTAL];
   ::ArrayInitialize(width,80);
   width[0]=100;
   width[1]=110;
   width[2]=100;
   width[3]=60;
   width[6]=90;
//--- Array of text alignment in columns
   ENUM_ALIGN_MODE align[COLUMNS2_TOTAL];
   ::ArrayInitialize(align,ALIGN_CENTER);
//--- Array of text offset along the X axis in the columns
   int text_x_offset[COLUMNS2_TOTAL];
   ::ArrayInitialize(text_x_offset,7);
//--- Array of column image offsets along the X axis
   int image_x_offset[COLUMNS2_TOTAL];
   ::ArrayInitialize(image_x_offset,3);
//--- Array of column image offsets along the Y axis
   int image_y_offset[COLUMNS2_TOTAL];
   ::ArrayInitialize(image_y_offset,2);
//--- Properties
   table.Font(m_base_font);
   table.FontSize(m_base_font_size);
   table.XSize(782);
   table.CellYSize(24);
   table.TableSize(COLUMNS2_TOTAL,1);
   table.TextAlign(align);
   table.ColumnsWidth(width);
   table.TextXOffset(text_x_offset);
   table.ImageXOffset(image_x_offset);
   table.ImageYOffset(image_y_offset);
   table.ShowHeaders(true);
   table.HeadersColor(C'87,128,255');
   table.HeadersColorHover(clrCornflowerBlue);
   table.HeadersTextColor(clrWhite);
   table.IsSortMode(false);
   table.LightsHover(true);
   table.SelectableRow(true);
   table.IsZebraFormatRows(clrWhiteSmoke);
   table.DataType(0,TYPE_LONG);
   table.AutoYResizeMode(true);
   table.AutoYResizeBottomOffset(5);
//--- Create a control element
   if(!table.CreateTable(x_gap,y_gap))
      return(false);
//--- Set the header titles
   table.SetHeaderText(0,TICKET);
   table.SetHeaderText(1,SYMBOL);
   table.SetHeaderText(2,TYPE_POS);
   table.SetHeaderText(3,PRICE);
   table.SetHeaderText(4,VOLUME);
   table.SetHeaderText(5,SL);
   table.SetHeaderText(6,TP);
   table.SetHeaderText(7,SWAP);
   table.SetHeaderText(8,PROFIT);
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,table);
   return(true);
}
```

The table will have 9 columns. Let us adjust the column width, as column names are different in length. Set column names for the two available languages using macro substitutions in the **Defines.mqh** file.

```
#define SYMBOL                         (m_language==RUSSIAN ? "Символ" : "Symbol")
#define VOLUME                         (m_language==RUSSIAN ? "Объем" : "Volume")
#define TYPE_POS                       (m_language==RUSSIAN ? "Тип позиции" : "Position Type")
#define SWAP                           (m_language==RUSSIAN ? "Своп" : "Swap")
#define PROFIT                         (m_language==RUSSIAN ? "Прибыль" : "Profit")
```

However, if you try to compile the project now, you will see that the table, buttons and fields go beyond the right edge of the window. Therefore, we need to add a mechanism that adjusts the main window width in accordance with its contents. This can be done by the **WindowRezise()** method.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::WindowResize(int x_size,int y_size)
{
   m_main_window.GetCloseButtonPointer().Hide();
   m_main_window.ChangeWindowWidth(x_size);
   m_main_window.ChangeWindowHeight(y_size);
   m_main_window.GetCloseButtonPointer().Show();
}
```

In the event handler, create a new tab click event. Add the main window width for each tab in this event.

```
//--- Tab switching event
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_TAB)
   {
      if(m_tab.SelectedTab()==0)
         WindowResize(600,m_main_window.YSize());
      if(m_tab.SelectedTab()==1)
         WindowResize(782+10,m_main_window.YSize());
      if(m_tab.SelectedTab()==2)
         WindowResize(682+10,m_main_window.YSize());
   }
```

Now the information will be displayed correctly, as is shown in figure 3. The next step is to receive data on positions opened by the application and to display this information in the created table. We have three steps in preparing and displaying data in the table:

- Initialization. Here we determine the number of positions that belong to our toolkit and rebuild the table for this data.
- Adding data. Add parameters of each open position to the table, according to the description in the column headers.
- Update the table with the filled data.

For the first initialization step, create the **InitializePositionsTable()** function and make a selection of all open positions by the current symbol and magic number. Thus, we obtain the number of positions which meet our conditions. Add the same number of rows to the table.

```
//+------------------------------------------------------------------+
//| Initializing the table of positions                              |
//+------------------------------------------------------------------+
void CFastTrading::InitializePositionsTable(void)
{
//--- Get symbols of open positions
   int total=PositionsTotal(); // the number of open positions
   int cnt=0;
//--- Delete all rows
   m_table_positions.DeleteAllRows();
//--- Set the number of rows equal to the number of positions
   for(int i=0; i<total; i++)
   {
      //--- order parameters
      ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
      string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
      ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
      //--- if MagicNumber matches
      if(magic==m_magic_number)
         if(position_symbol==Symbol())
         {
            m_table_positions.AddRow(cnt);
            cnt++;
         }
   }
//--- If there are positions
   if(cnt>0)
   {
      //--- Set the values in the table
      SetValuesToPositionsTable();
      //--- Update the table
      UpdatePositionsTable();
   }
}
```

Then check if there is at least one position opened by the toolkit. If there are any such positions, set the information about open positions in the newly created rows using the **SetValuePositionTable()** method.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::SetValuesToPositionsTable(void)
{
//---
   int cnt=0;
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      //--- order parameters
      ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
      string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
      int    digits=(int)SymbolInfoInteger(position_symbol,SYMBOL_DIGITS);              // the number of decimal places
      ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
      double volume=PositionGetDouble(POSITION_VOLUME);                                 // position volume
      ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
      double stoploss=PositionGetDouble(POSITION_SL);
      double takeprofit=PositionGetDouble(POSITION_TP);
      double swap=PositionGetDouble(POSITION_SWAP);
      double profit=PositionGetDouble(POSITION_PROFIT);
      double openprice=PositionGetDouble(POSITION_PRICE_OPEN);
      string pos=(type==POSITION_TYPE_BUY)?"BUY":"SELL";
      profit+=swap;
      //--- if MagicNumber matches
      if(magic==m_magic_number)
         if(position_symbol==Symbol())
         {
            m_table_positions.SetValue(0,i,string(position_ticket));
            m_table_positions.SetValue(1,i,string(position_symbol));
            m_table_positions.SetValue(2,i,pos);
            m_table_positions.SetValue(3,i,string(openprice));
            m_table_positions.SetValue(4,i,string(volume));
            m_table_positions.SetValue(5,i,string(stoploss));
            m_table_positions.SetValue(6,i,string(takeprofit));
            m_table_positions.SetValue(7,i,string(swap));
            m_table_positions.SetValue(8,i,DoubleToString(profit,2));
            //---
            m_table_positions.TextColor(2,i,(pos=="BUY")? clrForestGreen : clrCrimson);
            m_table_positions.TextColor(8,i,(profit!=0)? (profit>0)? clrForestGreen : clrCrimson : clrSilver);
            cnt++;
         }
   }
}
```

After filling the data, update the table using **UpdatePositionsTable()**:

```
//+------------------------------------------------------------------+
//| Update the table of positions                                    |
//+------------------------------------------------------------------+
void CFastTrading::UpdatePositionsTable(void)
{
//--- Update the table
   m_table_positions.Update(true);
   m_table_positions.GetScrollVPointer().Update(true);
}
```

To activate the changes in our product, we should properly configure them. Open the SimpleTrading.mq5 file, find the **OnInit()** function and add the call of the method initializing the application class:

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
   program.OnInitEvent();
//---
   return(INIT_SUCCEEDED);
}
```

This must be done strictly after creating the application GUI **CreateGUI()**. Now, go to **OnInitEvent()** body and call the table initialization in it.

```
//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
void CFastTrading::OnInitEvent(void)
{
   InitializePositionsTable();
}
```

Now the table properly displays information about open positions. However, this data is only relevant at the application launch time, and it should therefore be constantly updated. This should be done in the **OnTrade()** trade event handler and in the **OnTick()** function. In trading events, we will track the number of current open positions and their parameters. Information about the current profit of each order will be updated in OnTick.

Create the **OnTradeEvent()** method in the public section of the base class and call the table initialization in its body.

```
//+------------------------------------------------------------------+
//| Trade event                                                      |
//+------------------------------------------------------------------+
void CFastTrading::OnTradeEvent(void)
{
//--- If a new trade
      InitializePositionsTable();
}
```

The new method is called in the trade event handler:

```
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade(void)
{
   program.OnTradeEvent();
}
```

The above actions set the relevance of displayed open positions. To update the profit, create the **UpdatePositionProfit()** method in the public section of the **CFastTrading** class:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::UpdatePositionProfit(void)
{
//---
   int cnt=0;
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      //--- order parameters
      ulong  position_ticket=PositionGetTicket(i);                                      // position ticket
      string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
      ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // position MagicNumber
      double swap=PositionGetDouble(POSITION_SWAP);
      double profit=PositionGetDouble(POSITION_PROFIT);
      profit+=swap;
      //--- if MagicNumber matches
      if(magic==m_magic_number)
         if(position_symbol==Symbol())
         {
            m_table_positions.SetValue(8,i,DoubleToString(profit,2));
            //---
            m_table_positions.TextColor(8,i,(profit!=0)? (profit>0)? clrForestGreen : clrCrimson : clrSilver);
            cnt++;
         }
   }
//---
   if(cnt>0)
      UpdatePositionsTable();
}
```

Call it in **OnTick()**:

```
void OnTick()
{
   program.UpdatePositionProfit();
}
```

This completes implementation of the table. Now, let us create the possibility to edit and close open positions available in the current list. We should make so that upon a click on a table row, input fields will display the lot, Take Profit and Stop Loss of a selected position.

```
//--- Event of clicking on a table row
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_LIST_ITEM)
   {
      //--- Check the element ID
      if(lparam==m_table_positions.Id())
      {
         //---
         int row=m_table_positions.SelectedItem();
         m_lot_edit[6].SetValue(string(m_table_positions.GetValue(4,row)));
         m_sl_edit[6].SetValue(string(m_table_positions.GetValue(5,row)));
         m_tp_edit[6].SetValue(string(m_table_positions.GetValue(6,row)));
         m_lot_edit[6].GetTextBoxPointer().Update(true);
         m_sl_edit[6].GetTextBoxPointer().Update(true);
         m_tp_edit[6].GetTextBoxPointer().Update(true);
      }
   }
```

Compile the project and receive a result as is shown in figure 10: values from a selected market position shown in input fields.

![](https://c.mql5.com/2/41/010.jpg)

Fig.10 Selecting an open position for further editing.

Then create two methods **ModifyPosition()** and **ClosePosition()**. Upon a click on Modify and Close buttons, the method will apply the appropriate actions to the selected open position.

```
//+------------------------------------------------------------------+
//| Modifying a selected open position                               |
//+------------------------------------------------------------------+
bool CFastTrading::ModifyPosition(long lparam)
{
//--- Check the element ID
   if(lparam==m_small_button[0].Id())
   {
//--- Get index and symbol
      if(m_table_positions.SelectedItem()==WRONG_VALUE)
         return(false);
      int row=m_table_positions.SelectedItem();
      ulong ticket=(ulong)m_table_positions.GetValue(0,row);
//---
      if(PositionSelectByTicket(ticket))
      {
         //--- declare the request and the result
         MqlTradeRequest request;
         MqlTradeResult  result;
         //--- calculation and rounding of the Stop Loss and Take Profit values
         double sl=NormalizeDouble((double)m_sl_edit[6].GetValue(),_Digits);
         double tp=NormalizeDouble((double)m_tp_edit[6].GetValue(),_Digits);
         //--- zeroing the request and result values
         ZeroMemory(request);
         ZeroMemory(result);
         //--- set the operation parameters
         request.action  =TRADE_ACTION_SLTP; // trading operation type
         request.position=ticket;            // position ticket
         request.symbol=Symbol();            // symbol
         request.sl      =sl;                // position Stop Loss
         request.tp      =tp;                // position Take Profit
         request.magic=m_magic_number;       // position MagicNumber
         //--- sending a request
         bool res=true;
         for(int j=0; j<5; j++)
         {
            res=OrderSend(request,result);
            if(res && result.retcode==TRADE_RETCODE_DONE)
               return(true);
            else
               PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
         }
      }
   }
//---
   return(false);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::ClosePosition(long lparam)
{
//--- Check the element ID
   if(lparam==m_small_button[1].Id())
   {
      //--- Get index and symbol
      if(m_table_positions.SelectedItem()==WRONG_VALUE)
         return(false);
      int row=m_table_positions.SelectedItem();
      ulong ticket=(ulong)m_table_positions.GetValue(0,row);
//---
      if(PositionSelectByTicket(ticket))
      {
         ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // position type
         string position_symbol=PositionGetString(POSITION_SYMBOL);                        // symbol
         //--- declare the request and the result
         MqlTradeRequest request;
         MqlTradeResult  result;
         //--- zeroing the request and result values
         ZeroMemory(request);
         ZeroMemory(result);
         //--- Set order price and type depending on the position type
         if(type==POSITION_TYPE_BUY)
         {
            request.price=SymbolInfoDouble(position_symbol,SYMBOL_BID);
            request.type =ORDER_TYPE_SELL;
         }
         else if(type==POSITION_TYPE_SELL)
         {
            request.price=SymbolInfoDouble(position_symbol,SYMBOL_ASK);
            request.type =ORDER_TYPE_BUY;
         }
         //--- check volume
         double position_volume=PositionGetDouble(POSITION_VOLUME);
         double closing_volume=(double)m_lot_edit[6].GetValue();
         if(closing_volume>position_volume)
            closing_volume=position_volume;
         //--- setting request
         request.action   =TRADE_ACTION_DEAL;
         request.position =ticket;
         request.symbol   =Symbol();
         request.volume   =NormalizeLot(Symbol(),closing_volume);
         request.magic    =m_magic_number;
         request.deviation=5;
         //--- sending a request
         bool res=true;
         for(int j=0; j<5; j++)
         {
            res=OrderSend(request,result);
            if(res && result.retcode==TRADE_RETCODE_DONE)
               return(true);
            else
               PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
         }
      }
   }
//---
   return(false);
}
```

This completes the implementation of tasks in the Market Positions Control tab. Let us move on to developing the Pending Orders Control tab. At the end of the **CreateMainWindow()** main window method body, add the code that adds the same functionality that we implemented in the previous tab: a table of pending orders, buttons and input fields for editing the orders.

```
//--- Create a table of pending orders
   if(!CreateOrdersTable(m_table_orders,0,22+5))
      return(false);
//--- Input fields for editing pending orders
   if(!CreateLotControl(m_pr_edit[4],360,3,2))
      return(false);
   if(!CreateStopLossControl(m_sl_edit[7],360+80,3,2))
      return(false);
   if(!CreateTakeProfitControl(m_tp_edit[7],360+80*2,3,2))
      return(false);
//--- Pending order modifying/deleting orders
   if(!CreateModifyButton(m_small_button[2],MODIFY,361+80*3,3,2))
      return(false);
   if(!CreateCloseButton(m_small_button[3],REMOVE,361+80*3,3+24,2))
      return(false);
```

This addition has a new method that adds a table for pending orders, and a new macro substitution for the order deleting button. Let us take a closer look at how a table is created:

```
//+------------------------------------------------------------------+
//| Creates a table of pending orders                                |
//+------------------------------------------------------------------+
//---
bool CFastTrading::CreateOrdersTable(CTable &table,const int x_gap,const int y_gap)
{
#define COLUMNS1_TOTAL 7
//--- Store the pointer to the main control
   table.MainPointer(m_tab);
//--- Attach to tab
   m_tab.AddToElementsArray(2,table);
//--- Array of column widths
   int width[COLUMNS1_TOTAL];
   ::ArrayInitialize(width,80);
   width[0]=100;
   width[2]=100;
//--- Array of text offset along the X axis in the columns
   int text_x_offset[COLUMNS1_TOTAL];
   ::ArrayInitialize(text_x_offset,7);
//--- Array of column image offsets along the X axis
   int image_x_offset[COLUMNS1_TOTAL];
   ::ArrayInitialize(image_x_offset,3);
//--- Array of column image offsets along the Y axis
   int image_y_offset[COLUMNS1_TOTAL];
   ::ArrayInitialize(image_y_offset,2);
//--- Array of text alignment in columns
   ENUM_ALIGN_MODE align[COLUMNS1_TOTAL];
   ::ArrayInitialize(align,ALIGN_CENTER);
   align[6]=ALIGN_LEFT;
//--- Properties
   table.Font(m_base_font);
   table.FontSize(m_base_font_size);
   table.XSize(602);
   table.CellYSize(24);
   table.TableSize(COLUMNS1_TOTAL,1);
   table.TextAlign(align);
   table.ColumnsWidth(width);
   table.TextXOffset(text_x_offset);
   table.ImageXOffset(image_x_offset);
   table.ImageYOffset(image_x_offset);
   table.ShowHeaders(true);
   table.HeadersColor(C'87,128,255');
   table.HeadersColorHover(clrCornflowerBlue);
   table.HeadersTextColor(clrWhite);
   table.IsSortMode(false);
   table.LightsHover(true);
   table.SelectableRow(true);
   table.IsZebraFormatRows(clrWhiteSmoke);
   table.AutoYResizeMode(true);
   table.AutoYResizeBottomOffset(5);
   table.DataType(0,TYPE_LONG);
   table.DataType(1,TYPE_STRING);
   table.DataType(2,TYPE_STRING);
   table.DataType(3,TYPE_DOUBLE);
   table.DataType(4,TYPE_DOUBLE);
   table.DataType(5,TYPE_DOUBLE);
//--- Create a control element
   if(!table.CreateTable(x_gap,y_gap))
      return(false);
//--- Set the header titles
   table.SetHeaderText(0,TICKET);
   table.SetHeaderText(1,SYMBOL);
   table.SetHeaderText(2,TYPE_POS);
   table.SetHeaderText(3,VOLUME);
   table.SetHeaderText(4,PRICE);
   table.SetHeaderText(5,SL);
   table.SetHeaderText(6,TP);
//--- Add the object to the common array of object groups
   CWndContainer::AddToElementsArray(0,table);
   return(true);
}
```

This table has a different number of columns, which are arranged in a different order. When editing open market positions, we can modify the lot, as well as the Take Profit and Stop Loss values. As for pending orders, here we modify the open price instead of the lot. Compile the project and check the result in the Pending Orders Control tab:

![](https://c.mql5.com/2/41/011.jpg)

Fig.11 Interface for working with pending orders.

Further development is similar to what we have done with the previous tab. First, find all orders belonging to the toolkit, using **InitializeOrdersTable()**:

```
//+------------------------------------------------------------------+
//| Initializing the table of positions                              |
//+------------------------------------------------------------------+
void CFastTrading::InitializeOrdersTable(void)
{
//---
   int total=OrdersTotal();
   int cnt=0;
//--- Delete all rows
   m_table_orders.DeleteAllRows();
//--- Set the number of rows equal to the number of positions
   for(int i=0; i<total; i++)
   {
      //--- order parameters
      ulong  order_ticket=OrderGetTicket(i);                                   // order ticket
      string order_symbol=OrderGetString(ORDER_SYMBOL);                        // symbol
      ulong  magic=OrderGetInteger(ORDER_MAGIC);                               // position MagicNumber
      //--- if MagicNumber matches
      if(magic==m_magic_number)
         if(order_symbol==Symbol())
         {
            m_table_orders.AddRow(cnt);
            cnt++;
         }
   }
//--- If there are positions
   if(cnt>0)
   {
      //--- Set the values in the table
      SetValuesToOrderTable();
      //--- Update the table
      UpdateOrdersTable();
   }
}
```

If pending orders are found, add the relevant information to the table using the **SetValuesToOrderTable()** method:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::SetValuesToOrderTable(void)
{
//---
   int cnt=0;
   ulong    ticket;
   for(int i=0; i<OrdersTotal(); i++)
   {
      //--- order parameters
      if((ticket=OrderGetTicket(i))>0)
      {
         string position_symbol=OrderGetString(ORDER_SYMBOL);                          // symbol
         ulong  magic=OrderGetInteger(ORDER_MAGIC);                                    // order MagicNumber
         double volume=OrderGetDouble(ORDER_VOLUME_INITIAL);                           // order volume
         ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);            // order type
         double price=OrderGetDouble(ORDER_PRICE_OPEN);
         double stoploss=OrderGetDouble(ORDER_SL);
         double takeprofit=OrderGetDouble(ORDER_TP);
         string pos="";
         if(type==ORDER_TYPE_BUY_LIMIT)
            pos="Buy Limit";
         else if(type==ORDER_TYPE_SELL_LIMIT)
            pos="Sell Limit";
         else if(type==ORDER_TYPE_BUY_STOP)
            pos="Buy Stop";
         else if(type==ORDER_TYPE_SELL_STOP)
            pos="Sell Stop";
         //---
         if(magic==m_magic_number)
            if(position_symbol==Symbol())
            {
               m_table_orders.SetValue(0,i,string(ticket));
               m_table_orders.SetValue(1,i,string(position_symbol));
               m_table_orders.SetValue(2,i,pos);
               m_table_orders.SetValue(3,i,string(volume));
               m_table_orders.SetValue(4,i,string(price));
               m_table_orders.SetValue(5,i,string(stoploss));
               m_table_orders.SetValue(6,i,string(takeprofit));
               cnt++;
            }
      }
   }
}
```

Update the added data using the **UpdateOrdersTable()** method:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFastTrading::UpdateOrdersTable(void)
{
//--- Update the table
   m_table_orders.Update(true);
   m_table_orders.GetScrollVPointer().Update(true);
}
```

To connect this functionality to the application, do the same what we have done in the previous tab. Namely, initialize the table with pending orders:

```
//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
void CFastTrading::OnInitEvent(void)
{
   InitializeOrdersTable();
   InitializePositionsTable();
}
```

Repeat actions in the trade event handler:

```
//+------------------------------------------------------------------+
//| Trade event                                                      |
//+------------------------------------------------------------------+
void CFastTrading::OnTradeEvent(void)
{
//---
   InitializePositionsTable();
   InitializeOrdersTable();
}
```

To enable the display of relevant data in input fields upon table row click, add the following code in the appropriate section of the event handler:

```
//--- Event of clicking on a table row
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_LIST_ITEM)
   {
      //--- Check the element ID
      if(lparam==m_table_positions.Id())
      {
         //---
         int row=m_table_positions.SelectedItem();
         m_lot_edit[6].SetValue(string(m_table_positions.GetValue(4,row)));
         m_sl_edit[6].SetValue(string(m_table_positions.GetValue(5,row)));
         m_tp_edit[6].SetValue(string(m_table_positions.GetValue(6,row)));
         m_lot_edit[6].GetTextBoxPointer().Update(true);
         m_sl_edit[6].GetTextBoxPointer().Update(true);
         m_tp_edit[6].GetTextBoxPointer().Update(true);
      }
      //--- Check the element ID
      if(lparam==m_table_orders.Id())
      {
         //---
         int row=m_table_orders.SelectedItem();
         m_pr_edit[4].SetValue(string(m_table_orders.GetValue(4,row)));
         m_sl_edit[7].SetValue(string(m_table_orders.GetValue(5,row)));
         m_tp_edit[7].SetValue(string(m_table_orders.GetValue(6,row)));
         m_pr_edit[4].GetTextBoxPointer().Update(true);
         m_sl_edit[7].GetTextBoxPointer().Update(true);
         m_tp_edit[7].GetTextBoxPointer().Update(true);
      }
   }
```

Now, let us assign the relevant actions to buttons Modify and Delete. Create methods **ModifyOrder()** and **RemoveOrder()**.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::ModifyOrder(long lparam)
{
//--- Check the element ID
   if(lparam==m_small_button[2].Id())
   {
      //--- Get index and symbol
      if(m_table_orders.SelectedItem()==WRONG_VALUE)
         return(false);
      int row=m_table_orders.SelectedItem();
      ulong ticket=(ulong)m_table_orders.GetValue(0,row);
      //---
      if(OrderSelect(ticket))
      {
         string position_symbol=OrderGetString(ORDER_SYMBOL);                          // symbol
         ulong  magic=OrderGetInteger(ORDER_MAGIC);                                    // order MagicNumber
         ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);            // order type
         //--- calculation and rounding of the Stop Loss and Take Profit values
         double sl=NormalizeDouble((double)m_sl_edit[7].GetValue(),_Digits);
         double tp=NormalizeDouble((double)m_tp_edit[7].GetValue(),_Digits);
         double price=NormalizeDouble((double)m_pr_edit[4].GetValue(),_Digits);
         //--- declare the request and the result
         MqlTradeRequest request;
         MqlTradeResult  result;
         //--- zeroing the request and result values
         ZeroMemory(request);
         ZeroMemory(result);
         //--- set the operation parameters
         request.action=TRADE_ACTION_MODIFY; // trading operation type
         request.order = ticket;             // order ticket
         request.symbol=Symbol();            // symbol
         request.sl      =sl;                // position Stop Loss
         request.tp      =tp;                // position Take Profit
         request.price=price;                // new price
         request.magic=m_magic_number;       // position MagicNumber
         //--- sending a request
         bool res=true;
         for(int j=0; j<5; j++)
         {
            res=OrderSend(request,result);
            if(res && result.retcode==TRADE_RETCODE_DONE)
               return(true);
            else
               PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
         }
      }
   }
//---
   return(false);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFastTrading::RemoveOrder(long lparam)
{
//--- Check the element ID
   if(lparam==m_small_button[3].Id())
   {
      //--- Get index and symbol
      if(m_table_orders.SelectedItem()==WRONG_VALUE)
         return(false);
      int row=m_table_orders.SelectedItem();
      ulong ticket=(ulong)m_table_orders.GetValue(0,row);
      //---
      if(OrderSelect(ticket))
      {
         string position_symbol=OrderGetString(ORDER_SYMBOL);                          // symbol
         ulong  magic=OrderGetInteger(ORDER_MAGIC);                                    // order MagicNumber
         ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);            // order type
         //--- declare the request and the result
         MqlTradeRequest request;
         MqlTradeResult  result;
         //--- zeroing the request and result values
         ZeroMemory(request);
         ZeroMemory(result);
         //--- set the operation parameters
         request.action=TRADE_ACTION_REMOVE;             // trading operation type
         request.order = ticket;                         // order ticket
         //--- sending a request
         bool res=true;
         for(int j=0; j<5; j++)
         {
            res=OrderSend(request,result);
            if(res && result.retcode==TRADE_RETCODE_DONE)
               return(true);
            else
               PrintFormat("OrderSend error %d",GetLastError());  // if unable to send the request, output the error code
         }
      }
   }
//---
   return(false);
}
```

Call them in the event handler, outside any sections:

```
      //---
      if(ModifyOrder(lparam))
         return;
      if(RemoveOrder(lparam))
         return;
```

I would like to add another convenient feature for working with pending orders. It provides the ability to set an open price of the previously selected ending order by clicking on the chart. This is shown in figure 12:

![](https://c.mql5.com/2/39/Screenshot_3.jpg)

Fig.12 Setting a pending order open price.

It works as follows:

- A click on the input field activates the value editing.
- Then move the mouse pointer to any place on the chart — the appropriate values are shown along the axis. For convenience, you can click on the mouse wheel and select the desired price using crosshairs.
- Stop the mouse pointer at the desired price and click on the chart. This price value will be set in the input field.

Using this method, you can quickly and conveniently set the price for a pending order, especially when it is easier to do from a visual analysis. If you need to set the price in a more precise way, enter it in the input field using the keyboard. The implementation is very simple. In the base class handler, create a section with the event of clicking on the chart and add the following code:

```
//--- The event of clicking on the chart
   if(id==CHARTEVENT_CLICK)
   {
      for(int i=0; i<4; i++)
      {
         if(m_pr_edit[i].GetTextBoxPointer().TextEditState())
         {
            m_last_index=i;
            break;
         }
         else
         {
            if(m_last_index>=0)
            {
               //---
               datetime dt    =0;
               int      window=0;
               //--- convert X and Y coordinates to date/time
               if(ChartXYToTimePrice(0,(int)lparam,(int)dparam,window,dt,m_xy_price))
               {
                  m_pr_edit[m_last_index].SetValue(DoubleToString(m_xy_price));
                  m_pr_edit[m_last_index].GetTextBoxPointer().Update(true);
                  m_last_index=-1;
               }
            }
         }
      }
   }
```

Determine here which of the fields is being edited, remember it, receive the value by clicking on the chart and insert this value in the input field. The main features and innovations are shown in the video below.

Fast Trading in MetaTrader 5 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7981)

MQL5.community

1.91K subscribers

[Fast Trading in MetaTrader 5](https://www.youtube.com/watch?v=Q5UFVEKwIq0)

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

[Watch on](https://www.youtube.com/watch?v=Q5UFVEKwIq0&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7981)

0:00

0:00 / 1:50

•Live

•

### Conclusion

The attached archive contains all the listed files, which are located in the appropriate folders. For their proper operation, you only need to save the MQL5 folder into the terminal folder. To open the terminal root directory, in which the MQL5 folder is located, press the Ctrl+Shift+D key combination in the MetaTrader 5 terminal or use the context menu as shown in Fig. 13 below.

![](https://c.mql5.com/2/41/013.jpg)

Fig.13 Opening the MQL5 folder in the MetaTrader 5 terminal root

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7981](https://www.mql5.com/ru/articles/7981)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7981.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7981/mql5.zip "Download MQL5.zip")(5952.86 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [A system of voice notifications for trade events and signals](https://www.mql5.com/en/articles/8111)
- [Quick Manual Trading Toolkit: Basic Functionality](https://www.mql5.com/en/articles/7892)
- [Multicurrency monitoring of trading signals (Part 5): Composite signals](https://www.mql5.com/en/articles/7759)
- [Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)
- [Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://www.mql5.com/en/articles/7600)
- [Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://www.mql5.com/en/articles/7528)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/354328)**
(10)


![Alexander Fedosov](https://c.mql5.com/avatar/2019/5/5CE6AA22-02C3.jpg)

**[Alexander Fedosov](https://www.mql5.com/en/users/alex2356)**
\|
17 Jun 2020 at 11:43

**Vitalii Vilnyi:**

Interesting article for informative purposes and can be further developed

I would like to have one for MT4 as well

Of course, it can be customised. There can be any number of different actions when working on the markets in [manual trading](https://www.mql5.com/en/articles/5268 "Article: Reversal: formalizing the entry point and writing a manual trading algorithm ").

And for MT4, anyone can order one for me in freelancing, for example.

![Sergei Lebedev](https://c.mql5.com/avatar/2019/7/5D3E1392-839A.jpg)

**[Sergei Lebedev](https://www.mql5.com/en/users/salebedev)**
\|
28 Aug 2020 at 22:02

I would like to see an implementation on Easy&FastGUI - there is no strength to integrate another gnraphic self-design.


![bankova.elizabet](https://c.mql5.com/avatar/2020/5/5EB59C5D-122D.gif)

**[bankova.elizabet](https://www.mql5.com/en/users/bankova.elizabet)**
\|
8 Sep 2020 at 16:43

One thing we can agree on is that you can't get anywhere without being able to trade hands.


![Fabiano Luiz Roberto](https://c.mql5.com/avatar/2022/7/62D98351-35AE.jpg)

**[Fabiano Luiz Roberto](https://www.mql5.com/en/users/fabianolr)**
\|
13 Dec 2020 at 02:28

Excellent article, thanks for posting it.

As a reminder, when compiling, the message below appears:

"deprecated behaviour, hidden method calling will be disabled in a future MQL compiler version ListView.mqh"

Any suggestions?

Thank you.

* * *

Excellent article, gratitude for the publication.

As information, when compiling, the message below appears:

"deprecated behaviour, hidden method calling will be disabled in a future MQL compiler version ListView.mqh"

Any suggestions?

Thanks.

![BeverlyHillsCop](https://c.mql5.com/avatar/avatar_na2.png)

**[BeverlyHillsCop](https://www.mql5.com/en/users/beverlyhillscop)**
\|
5 Mar 2021 at 06:40

So educational!  Thank you for a great and clear article!


![Probability theory and mathematical statistics with examples (part I): Fundamentals and elementary theory](https://c.mql5.com/2/39/Probability_theory_1.png)[Probability theory and mathematical statistics with examples (part I): Fundamentals and elementary theory](https://www.mql5.com/en/articles/8038)

Trading is always about making decisions in the face of uncertainty. This means that the results of the decisions are not quite obvious at the time these decisions are made. This entails the importance of theoretical approaches to the construction of mathematical models allowing us to describe such cases in meaningful manner.

![Timeseries in DoEasy library (part 46): Multi-period multi-symbol indicator buffers](https://c.mql5.com/2/39/MQL5-avatar-doeasy-library__2.png)[Timeseries in DoEasy library (part 46): Multi-period multi-symbol indicator buffers](https://www.mql5.com/en/articles/8115)

In this article, I am going to improve the classes of indicator buffer objects to work in the multi-symbol mode. This will pave the way for creating multi-symbol multi-period indicators in custom programs. I will add the missing functionality to the calculated buffer objects allowing us to create multi-symbol multi-period standard indicators.

![On Methods to Detect Overbought/Oversold Zones. Part I](https://c.mql5.com/2/39/logo_200x200.png)[On Methods to Detect Overbought/Oversold Zones. Part I](https://www.mql5.com/en/articles/7782)

Overbought/oversold zones characterize a certain state of the market, differentiating through weaker changes in the prices of securities. This adverse change in the synamics is pronounced most at the final stage in the development of trends of any scales. Since the profit value in trading depends directly on the capability of covering as large trend amplitude as possible, the accuracy of detecting such zones is a key task in trading with any securities whatsoever.

![Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://c.mql5.com/2/39/MQL5-avatar-analysis__1.png)[Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://www.mql5.com/en/articles/8028)

In this article, we consider the principles of mathematical expression parsing and evaluation using parsers based on operator precedence. We will implement Pratt and shunting-yard parser, byte-code generation and calculations by this code, as well as view how to use indicators as functions in expressions and how to set up trading signals in Expert Advisors based on these indicators.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/7981&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083476128858905712)

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