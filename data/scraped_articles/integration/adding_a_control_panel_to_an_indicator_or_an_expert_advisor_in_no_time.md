---
title: Adding a control panel to an indicator or an Expert Advisor in no time
url: https://www.mql5.com/en/articles/2171
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:26:29.834731
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=humicdbngmasukcmryzjusujwtbrvpbw&ssn=1769178388451769882&ssn_dr=0&ssn_sr=0&fv_date=1769178388&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2171&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Adding%20a%20control%20panel%20to%20an%20indicator%20or%20an%20Expert%20Advisor%20in%20no%20time%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917838831885746&fz_uniq=6478625108831172372&sv=2552)

MetaTrader 5 / Examples


### Using Graphical Panels

Your MQL4/MQL5 indicator or Expert Advisor may be the most efficient in the world but there is always a room for improvements. In most cases, you need to enter the program's settings to change its inputs. However, this step can be avoided.

Develop your own control panel based on [Standard Library classes](https://www.mql5.com/en/docs/standardlibrary/controls). This will allow you to change the settings without restarting a program. Besides, this will make your program more attractive allowing it to stand out from the competitors. You can browse through multiple [graphical](https://www.mql5.com/en/market/mt5/utility) panels in the Market.

In this article, I will show you how to add a simple panel to your MQL4/MQL5 program. You will also find out how to teach a program to read the inputs and react to changes of their values.

### 1\. Combining the indicator with the panel

**1.1. Indicator**

The **NewBar.mq5** indicator performs a single action. It prints a message in the terminal's Experts log when a new bar arrives. The indicator code is provided below:

```
//+------------------------------------------------------------------+
//|                                                       NewBar.mq5 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "The indicator identifies a new bar"
#property indicator_chart_window
#property indicator_plots 0
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   static datetime prev_time;
//--- revert access to array time[] - do it like in timeseries
   ArraySetAsSeries(time,true);
//--- first calculation or number of bars was changed
   if(prev_calculated==0)// first calculation
     {
      prev_time=time[0];
      return(rates_total);
     }
//---
   if(time[0]>prev_time)
      Print("New bar!");
//---
   prev_time=time[0];
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

Now, let's delve into some details of **NewBar.mq5** operation.

The **prev\_time** [static](https://www.mql5.com/en/docs/basis/variables/static) variable is declared in the OnCalculate() function. This variable stores the **time\[0\]** open time. During the next pass, the **time\[0\]** open time is compared with the **prev\_time** variable. In other words, the current tick's **time\[0\]** open time is compared to the previous tick's one. If the following condition is met:

```
if(time[0]>prev_time)
```

a new bar is considered to be detected.

The next example shows in details how **NewBar.mq5** detects a new bar:

![New bar](https://c.mql5.com/2/22/new_bar__1.png)

Fig. 1. Detecting a new bar in the indicator

Let's consider 10 ticks on a very quiet market.

Ticks 1-3: open time of a bar with the index 0 ( **_time\[0\]_**) is equal to the time stored in the **_prev\_time_** static variable meaning that there is no new bar.

Tick 4: the tick arrived on a new bar. When entering the OnCalculate() function, **_time\[0\]_** has the bar open time (2015.12.01 00: **02**:00), while the _**prev\_time**_ variable still stores the previous tick's time (2015.12.01 00: **01**:00). Therefore, we detect the new bar when checking the **_time\[0\]>prev\_time_** condition. Before exiting OnCalculate(), the **_prev\_time_** variable obtains the time from **_time\[0\]_** (2015.12.01 00: **02**:00).

Ticks 5-8: open time of a bar with the index 0 ( **_time\[0\]_**) is equal to the time stored in the **_prev\_time_** static variable meaning that there is no new bar.

Tick 9: the tick arrived on a new bar. When entering the OnCalculate() function, **_time\[0\]_** has the bar open time (2015.12.01 00: **03**:00), while the _**prev\_time**_ variable still stores the previous tick's time (2015.12.01 00: **02**:00). Therefore, we detect the new bar when checking the **_time\[0\]>prev\_time_** condition. Before exiting OnCalculate(), the **_prev\_time_** variable obtains the time from **_time\[0\]_** (2015.12.01 00: **03**:00).

Tick 10: open time of a bar with the index 0 ( **_time\[0\]_**) is equal to the time stored in the **_prev\_time_** static variable meaning that there is no new bar.

**1.2. Panel**

All panel plotting parameters (amount, size, and coordinates of control elements) are gathered in a single [include](https://www.mql5.com/en/docs/basis/preprosessor/include) file **PanelDialog.mqh**, which serves as a panel implementation class.

The panel looks as follows:

![Panel](https://c.mql5.com/2/21/panel.png)

Fig. 2. Panel

The code of the **PanelDialog.mqh** include file is presented below:

```
//+------------------------------------------------------------------+
//|                                                  PanelDialog.mqh |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#include <Controls\Dialog.mqh>
#include <Controls\CheckGroup.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
//--- indents and gaps
#define INDENT_LEFT                         (11)      // indent from left (with allowance for border width)
#define INDENT_TOP                          (11)      // indent from top (with allowance for border width)
#define INDENT_BOTTOM                       (11)      // indent from bottom (with allowance for border width)
//--- for buttons
#define BUTTON_WIDTH                        (100)     // size by X coordinate
//+------------------------------------------------------------------+
//| Class CControlsDialog                                            |
//| Usage: main dialog of the Controls application                   |
//+------------------------------------------------------------------+
class CControlsDialog : public CAppDialog
  {
private:
   CCheckGroup       m_check_group;                   // CCheckGroup object

public:
                     CControlsDialog(void);
                    ~CControlsDialog(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- chart event handler
   virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);

protected:
   //--- create dependent controls
   bool              CreateCheckGroup(void);
   //--- handlers of the dependent controls events
   void              OnChangeCheckGroup(void);
  };
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CControlsDialog)
ON_EVENT(ON_CHANGE,m_check_group,OnChangeCheckGroup)
EVENT_MAP_END(CAppDialog)
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CControlsDialog::CControlsDialog(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CControlsDialog::~CControlsDialog(void)
  {
  }
//+------------------------------------------------------------------+
//| Create                                                           |
//+------------------------------------------------------------------+
bool CControlsDialog::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
   if(!CAppDialog::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create dependent controls
   if(!CreateCheckGroup())
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the "CheckGroup" element                                  |
//+------------------------------------------------------------------+
bool CControlsDialog::CreateCheckGroup(void)
  {
//--- coordinates
   int x1=INDENT_LEFT;
   int y1=INDENT_TOP;
   int x2=x1+BUTTON_WIDTH;
   int y2=ClientAreaHeight()-INDENT_BOTTOM;
//--- create
   if(!m_check_group.Create(m_chart_id,m_name+"CheckGroup",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!Add(m_check_group))
      return(false);
   m_check_group.Alignment(WND_ALIGN_HEIGHT,0,y1,0,INDENT_BOTTOM);
//--- fill out with strings
   if(!m_check_group.AddItem("Mail",1<<0))
      return(false);
   if(!m_check_group.AddItem("Push",1<<1))
      return(false);
   if(!m_check_group.AddItem("Alert",1<<2))
      return(false);
   Comment(__FUNCTION__+" : Value="+IntegerToString(m_check_group.Value()));
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CControlsDialog::OnChangeCheckGroup(void)
  {
   Comment(__FUNCTION__+" : Value="+IntegerToString(m_check_group.Value()));
  }
//+------------------------------------------------------------------+
```

As you can see, the class of our panel does not contain the methods for setting and reading the status of switches with independent fixing.

**Our objective** is to make the **NewBar.mq5** the main file and add the inputs, for example, the ability to choose new bar arrival alert methods ( **Mail**, **Push**, or **Alert**). Besides, the **PanelDialog.mqh** include file should contain the methods for setting and reading the status of **Mail**, **Push**, and **Alert** switches with independent fixing.

**1.3. Changing the indicator**

Note: all implemented changes are marked with color.

First, we should implement the **PanelDialog.mqh** include file:

```
#property indicator_chart_window
#property indicator_plots 0
#include "PanelDialog.mqh"
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
```

Then, add the inputs:

```
#property indicator_chart_window
#property indicator_plots 0
#include "PanelDialog.mqh"
//--- input parameters
input bool     bln_mail=false;      // Notify by email
input bool     bln_push=false;      // Notify by push
input bool     bln_alert=true;      // Notify by alert
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
```

Compile the indicator (F7 in MetaEditor) and make sure the input parameters are displayed correctly in the terminal:

![Input parameters](https://c.mql5.com/2/22/2015-12-28_15h04_26.png)

Fig. 3. Indicator input parameters

**1.4. Changing the panel**

Now, we should add **Mail**, **Push**, and **Alert** methods for setting and reading the status of switches with independent fixing to the panel.

Let's add the new methods to the panel class:

```
class CControlsDialog : public CAppDialog
  {
private:
   CCheckGroup       m_check_group;                   // CCheckGroup object

public:
                     CControlsDialog(void);
                    ~CControlsDialog(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- chart event handler
   virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
   //--- set check for element
   virtual bool      SetCheck(const int idx,const int value);
   //--- get check for element
   virtual int       GetCheck(const int idx) const;

protected:
   //--- create dependent controls
   bool              CreateCheckGroup(void);
```

Implementing the methods:

```
//+------------------------------------------------------------------+
//| Set check for element                                            |
//+------------------------------------------------------------------+
bool CControlsDialog::SetCheck(const int idx,const bool check)
  {
   return(m_check_group.Check(idx,check));
  }
//+------------------------------------------------------------------+
//| Get check for element                                            |
//+------------------------------------------------------------------+
int CControlsDialog::GetCheck(const int idx)
  {
   return(m_check_group.Check(idx));
  }
```

**1.5. The final stage of combining the indicator with the panel**

Declare the variable of our panel class in the block of global variables declaration of the **NewBar.mq5** indicator:

```
#property indicator_chart_window
#property indicator_plots 0
#include "PanelDialog.mqh"
//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CControlsDialog ExtDialog;
//--- input parameters
input bool     bln_mail=false;      // Notify by email
input bool     bln_push=false;      // Notify by push
input bool     bln_alert=true;      // Notify by alert
```

and add the OnChartEvent() function at the very end:

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   ExtDialog.ChartEvent(id,lparam,dparam,sparam);
  }
```

Create the panel in the OnInit() function of the **NewBar.mq5** indicator and click the check boxes programmatically according to the input parameters:

```
int OnInit()
  {
//--- indicator buffers mapping
//--- create application dialog
   if(!ExtDialog.Create(0,"Notification",0,50,50,180,160))
      return(INIT_FAILED);
//--- run application
   if(!ExtDialog.Run())
      return(INIT_FAILED);
//---
   ExtDialog.SetCheck(0,bln_mail);
   ExtDialog.SetCheck(1,bln_push);
   ExtDialog.SetCheck(2,bln_alert);
//---
   return(INIT_SUCCEEDED);
  }
```

Thus, we have combined the indicator with the panel. We have implemented the method for determining a check box status – pressed/released ( **SetCheck**), as well as the method for receiving it ( **GetCheck**).

### 2\. Combining the Expert Advisor with the panel

**2.1. Expert Advisor**

Let's use the EA from the standard delivery set ...\\MQL5\\Experts\\Examples\\MACD\ **MACD Sample.mq5** as a basis.

**2.2. Panel**

The finalized **PanelDialog2.mqh** panel looks as follows:

![Panel number two](https://c.mql5.com/2/22/2016-01-19_14h11_20.png)

Fig. 4. Panel number two

What are the benefits of combining the **MACD Sample.mq5** EA with the **PanelDialog2.mqh** panel? This allows us to quickly change the EA parameters ( **Lots**, **Trailing Stop Level (in pips)**, and others), as well as trade event notification settings ( **Mail**, **Push**, and **Alert**) on the current timeframe the EA is to be launched at.

Changed EA parameters ( **Lots**, **Trailing Stop Level (in pips)**, and others) are applied after clicking the **Apply changes** button. Changes of the trade event notification settings ( **Mail**, **Push**, and **Alert**) are applied automatically. There is no need to press the **Apply changes** button.

**2.3. The EA and the panel should have means of communication**

![Communication between the EA and the panel](https://c.mql5.com/2/22/communication.png)

Fig. 5. Communication between the EA and the panel

After the launch, the EA should send its parameters to the panel. After clicking the **Apply changes** button and changing the parameters, the panel should return the altered parameters to the EA for its initialization with the new parameters.

**2.4. Step one. Changing the EA**

Take the EA from the standard delivery set ...\\MQL5\\Experts\\Examples\\MACD\ **MACD Sample.mq5** and copy it to your folder. For example, you can create the **Notification** folder and copy the EA there:

![Creating a new folder](https://c.mql5.com/2/22/new_folder.png)

Fig. 6. Creating a new folder

In the area of the [EA's global variables](https://www.mql5.com/en/docs/basis/variables/global) (not to be confused with the [terminal's ones](https://www.mql5.com/en/docs/globals)), declare the new variables defining a method of sending notifications of the EA's trading activity. Please note that these variables have the **Inp** prefix just like other external variables:

```
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
//--- input parameters
input bool     InpMail=false;          // Notify by email
input bool     InpPush=false;          // Notify by push
input bool     InpAlert=true;          // Notify by alert
//---
input double InpLots          =0.1; // Lots
input int    InpTakeProfit    =50;  // Take Profit (in pips)
```

Add the duplicate copies of all the EA's external variables just below. The duplicate copies have the **Ext** prefix:

```
input int    InpMACDCloseLevel=2;   // MACD close level (in pips)
input int    InpMATrendPeriod =26;  // MA trend period
//--- ext variables
bool           ExtMail;
bool           ExtPush;
bool           ExtAlert;

double         ExtLots;
int            ExtTakeProfit;
int            ExtTrailingStop;
int            ExtMACDOpenLevel;
int            ExtMACDCloseLevel;
int            ExtMATrendPeriod;
//---
int ExtTimeOut=10; // time out in seconds between trade operations
//+------------------------------------------------------------------+
//| MACD Sample expert class                                         |
//+------------------------------------------------------------------+
```

Use OnInit() to set copying the values from external variables to duplicate variable values:

```
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   ExtMail=InpMail;
   ExtPush=InpPush;
   ExtAlert=InpAlert;

   ExtLots=InpLots;
   ExtTakeProfit=InpTakeProfit;
   ExtTrailingStop=InpTrailingStop;
   ExtMACDOpenLevel=InpMACDOpenLevel;
   ExtMACDCloseLevel=InpMACDCloseLevel;
   ExtMATrendPeriod=InpMATrendPeriod;
//--- create all necessary objects
   if(!ExtExpert.Init())
```

At this stage, the EA's external variables with the **Inp** prefix are used in the EA's **CSampleExpert::InitIndicators**, **CSampleExpert::InitCheckParameters**, and **CSampleExpert::Init** functions. We need to replace the external variables in these functions with their duplicate copies (having the **Ext** prefix). I suggest quite an unconventional solution here:

Adding a control panel to an indicator or an Expert - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2171)

MQL5.community

1.91K subscribers

[Adding a control panel to an indicator or an Expert](https://www.youtube.com/watch?v=ChxS6nXuAGM)

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

[Watch on](https://www.youtube.com/watch?v=ChxS6nXuAGM&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2171)

0:00

0:00 / 4:31

•Live

•

After the replacement is performed, compile the file to make sure that all has been done correctly. There should be no errors.

**2.5. Step two. Changing the panel**

The panel shown in Fig. 4 is a blank. It has neither the function for "communicating" with the EA, nor the function for processing the input data yet. Copy the panel blank file **PanelDialog2Original.mqh** to the **Notification** folder as well.

Add the internal variables to the panel class. They will be used to store the status of the entire input data. Note the **mModification** variable. I will provide more details on it in p. 2.7.

```
private:
   //--- get check for element
   virtual int       GetCheck(const int idx);
   //---
   bool              mMail;
   bool              mPush;
   bool              mAlert_;
   double            mLots;               // Lots
   int               mTakeProfit;         // Take Profit (in pips)
   int               mTrailingStop;       // Trailing Stop Level (in pips)
   int               mMACDOpenLevel;      // MACD open level (in pips)
   int               mMACDCloseLevel;     // MACD close level (in pips)
   int               mMATrendPeriod;      // MA trend period
   //---
   bool              mModification;       // Values have changed
  };
//+------------------------------------------------------------------+
//| Event Handling                                                   |
```

Initialize the internal variables in the panel class constructor just below:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CControlsDialog::CControlsDialog(void) : mMail(false),
                                         mPush(false),
                                         mAlert_(true),
                                         mLots(0.1),
                                         mTakeProfit(50),
                                         mTrailingStop(30),
                                         mMACDOpenLevel(3),
                                         mMACDCloseLevel(2),
                                         mMATrendPeriod(26),
                                         mModification(false)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
```

Add installing the groups of switcher elements according to the internal variables to the CControlsDialog::Create function:

```
if(!CreateButtonOK())
      return(false);

//---
   SetCheck(0,mMail);
   SetCheck(1,mPush);
   SetCheck(2,mAlert_);

//--- succeed
   return(true);
  }
```

**2.6.** **Step three.** **Changing the** **EA**

Until now, the EA and the panel were two separate files independent of each other. Let's connect them and declare the **ExtDialog** variable of our panel class:

```
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include "PanelDialog2Original.mqh"
//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CControlsDialog ExtDialog;
//--- input parameters
input bool     InpMail=false;          // Notify by email
input bool     InpPush=false;          // Notify by push
```

In order to make the panel operational and visible, it should be created and launched. Also, make sure to add the OnChartEvent() (for handling the ChartEvent) and OnDeinit() functions. OnInit() in the EA looks as follows:

```
int OnInit(void)
  {
   ExtMail=InpMail;
   ExtPush=InpPush;
   ExtAlert=InpAlert;

   ExtLots=InpLots;
   ExtTakeProfit=InpTakeProfit;
   ExtTrailingStop=InpTrailingStop;
   ExtMACDOpenLevel=InpMACDOpenLevel;
   ExtMACDCloseLevel=InpMACDCloseLevel;
   ExtMATrendPeriod=InpMATrendPeriod;
//--- create all necessary objects
   if(!ExtExpert.Init())
      return(INIT_FAILED);
//--- create application dialog
   if(!ExtDialog.Create(0,"Notification",0,100,100,360,380))
      return(INIT_FAILED);
//--- run application
   if(!ExtDialog.Run())
      return(INIT_FAILED);
//--- succeed
   return(INIT_SUCCEEDED);
  }
```

Let's destroy our panel in OnDeinit() and set the OnDeinit() function right after OnInit():

```
//--- succeed
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Comment("");
//--- destroy dialog
   ExtDialog.Destroy(reason);
  }
//+------------------------------------------------------------------+
//| Expert new tick handling function                                |
//+------------------------------------------------------------------+
void OnTick(void)
```

Add the OnChartEvent() function to the very end of the EA (after the OnTick function):

```
//--- change limit time by timeout in seconds if processed
         if(ExtExpert.Processing())
            limit_time=TimeCurrent()+ExtTimeOut;
        }
     }
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   ExtDialog.ChartEvent(id,lparam,dparam,sparam);
  }
//+------------------------------------------------------------------+
```

Now, the EA can be compiled and checked on the chart. The EA is launched with the panel:

![EA and panel](https://c.mql5.com/2/22/panel3step3__1.png)

Fig. 7. The EA and the panel

**2.7.** **Step four.** **Changing the panel. Big integration**

The EA is launched first. Then, its inputs are defined by a user. Only after that, the panel is launched. Therefore, the panel should have the functions for exchanging data with the EA.

Let's add the **Initialization()** method which accepts the parameters and uses them to initialize the panel's internal variables. Declaration:

```
virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
      //--- initialization
   virtual bool      Initialization(const bool Mail,const bool Push,const bool Alert_,
                                    const double Lots,const int TakeProfit,
                                    const int  TrailingStop,const int MACDOpenLevel,
                                    const int  MACDCloseLevel,const int MATrendPeriod);

protected:
   //--- create dependent controls
   bool              CreateCheckGroup(void);
```

The method's body (insert it before CControlsDialog::GetCheck):

```
//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
bool CControlsDialog::Initialization(const bool Mail,const bool Push,const bool Alert_,
                                     const double Lots,const int TakeProfit,
                                     const int  TrailingStop,const int MACDOpenLevel,
                                     const int  MACDCloseLevel,const int MATrendPeriod)
  {
   mMail=Mail;
   mPush=Push;
   mAlert_=Alert_;

   mLots=Lots;
   mTakeProfit=TakeProfit;
   mTrailingStop=TrailingStop;
   mMACDOpenLevel=MACDOpenLevel;
   mMACDCloseLevel=MACDCloseLevel;
   mMATrendPeriod=MATrendPeriod;
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Get check for element                                            |
//+------------------------------------------------------------------+
int CControlsDialog::GetCheck(const int idx)
```

Since the panel's internal variables have been initialized by the data, we need to fill in the panel's control elements (entry fields) correctly. Since we have six entry fields, I will provide an example based on _m\_edit1_. The string the text was assigned at looked as follows:

```
...
   if(!m_edit1.Text("Edit1"))
...
```

But now it looks differently:

```
...
   if(!m_edit1.Text(DoubleToString(mLots,2)))
...
```

Thus, each entry field corresponds to a certain internal variable.

The next method **named GetValues()** returns the values of internal variables:

```
virtual bool      Initialization(const bool Mail,const bool Push,const bool Alert_,
                                    const double Lots,const int TakeProfit,
                                    const int  TrailingStop,const int MACDOpenLevel,
                                    const int  MACDCloseLevel,const int MATrendPeriod);
   //--- get values
   virtual void      GetValues(bool &Mail,bool &Push,bool &Alert_,
                               double &Lots,int &TakeProfit,
                               int &TrailingStop,int &MACDOpenLevel,
                               int &MACDCloseLevel,int &MATrendPeriod);

protected:
   //--- create dependent controls
   bool              CreateCheckGroup(void);
```

Insert its body after CControlsDialog::Initialization()):

```
//+------------------------------------------------------------------+
//| Get values                                                       |
//+------------------------------------------------------------------+
void CControlsDialog::GetValues(bool &Mail,bool &Push,bool &Alert_,
                                double &Lots,int &TakeProfit,
                                int &TrailingStop,int &MACDOpenLevel,
                                int &MACDCloseLevel,int &MATrendPeriod)
  {
   Mail=mMail;
   Push=mPush;
   Alert_=mAlert_;

   Lots=mLots;
   TakeProfit=mTakeProfit;
   TrailingStop=mTrailingStop;
   MACDOpenLevel=mMACDOpenLevel;
   MACDCloseLevel=mMACDCloseLevel;
   MATrendPeriod=mMATrendPeriod;
  }
//+------------------------------------------------------------------+
//| Get check for element                                            |
//+------------------------------------------------------------------+
int CControlsDialog::GetCheck(const int idx)
```

Since the panel is to send a notification in response to any trading action performed by the EA, it should have a special method responsible for that. Let's declare it:

```
virtual void      GetValues(bool &Mail,bool &Push,bool &Alert_,
                               double &Lots,int &TakeProfit,
                               int &TrailingStop,int &MACDOpenLevel,
                               int &MACDCloseLevel,int &MATrendPeriod);   //--- send notifications
   virtual void      Notifications(const string text);

protected:
   //--- create dependent controls
   bool              CreateCheckGroup(void);
```

Insert its body after CControlsDialog::GetValues()):

```
//+------------------------------------------------------------------+
//|  Send notifications                                              |
//+------------------------------------------------------------------+
void CControlsDialog::Notifications(const string text)
  {
   int i=m_check_group.ControlsTotal();
   if(GetCheck(0))
      SendMail(" ",text);
   if(GetCheck(1))
      SendNotification(text);
   if(GetCheck(2))
      Alert(text);
  }
//+------------------------------------------------------------------+
//| Get check for element                                            |
//+------------------------------------------------------------------+
int CControlsDialog::GetCheck(const int idx)
```

The **mModification** flag (mentioned in p. 2.5) is used in order to remember, whether the parameters in the panel were changed.

```
virtual void      Notifications(const string text);
   //---
   virtual bool      Modification(void) const { return(mModification);          }
   virtual void      Modification(bool value) { mModification=value;            }

protected:
   //--- create dependent controls
   bool              CreateCheckGroup(void);
```

The changes are to be controlled in CControlsDialog::OnClickButtonOK, which handles the event of pressing the **Apply changes** button:

```
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CControlsDialog::OnClickButtonOK(void)
  {
//--- verifying changes
   if(m_check_group.Check(0)!=mMail)
      mModification=true;
   if(m_check_group.Check(1)!=mPush)
      mModification=true;
   if(m_check_group.Check(2)!=mAlert_)
      mModification=true;

   if(StringToDouble(m_edit1.Text())!=mLots)
     {
      mLots=StringToDouble(m_edit1.Text());
      mModification=true;
     }
   if(StringToInteger(m_edit2.Text())!=mTakeProfit)
     {
      mTakeProfit=(int)StringToDouble(m_edit2.Text());
      mModification=true;
     }
   if(StringToInteger(m_edit3.Text())!=mTrailingStop)
     {
      mTrailingStop=(int)StringToDouble(m_edit3.Text());
      mModification=true;
     }
   if(StringToInteger(m_edit4.Text())!=mMACDOpenLevel)
     {
      mMACDOpenLevel=(int)StringToDouble(m_edit4.Text());
      mModification=true;
     }
   if(StringToInteger(m_edit5.Text())!=mMACDCloseLevel)
     {
      mMACDCloseLevel=(int)StringToDouble(m_edit5.Text());
      mModification=true;
     }
   if(StringToInteger(m_edit6.Text())!=mMATrendPeriod)
     {
      mMATrendPeriod=(int)StringToDouble(m_edit6.Text());
      mModification=true;
     }
  }
```

Also, the panel checks the input data in the handlers:

```
void              OnChangeCheckGroup(void);
   void              OnChangeEdit1(void);
   void              OnChangeEdit2(void);
   void              OnChangeEdit3(void);
   void              OnChangeEdit4(void);
   void              OnChangeEdit5(void);
   void              OnChangeEdit6(void);
   void              OnClickButtonOK(void);
```

I will skip their description.

**2.8.** **Step five.** **Changing the EA. Last edits**

At the moment, the panel does not work in the strategy tester, therefore we need to implement the protection and introduce the internal variable – the **bool\_tester** flag.

```
//---
int ExtTimeOut=10; // time out in seconds between trade operations
bool           bool_tester=false;      // true - mode tester
//+------------------------------------------------------------------+
//| MACD Sample expert class                                         |
//+------------------------------------------------------------------+
class CSampleExpert
```

Insert changes to OnInit() – protect from launching in the strategy tester. Also, initialize the panel's parameters before visualizing it:

```
//--- create all necessary objects
   if(!ExtExpert.Init())
      return(INIT_FAILED);
//---
   if(!MQLInfoInteger(MQL_TESTER))
     {
      bool_tester=false;
      //---
      ExtDialog.Initialization(ExtMail,ExtPush,ExtAlert,
                               ExtLots,ExtTakeProfit,ExtTrailingStop,
                               ExtMACDOpenLevel,ExtMACDCloseLevel,ExtMATrendPeriod);
      //--- create application dialog
      if(!ExtDialog.Create(0,"Notification",0,100,100,360,380))
         return(INIT_FAILED);
      //--- run application
      if(!ExtDialog.Run())
         return(INIT_FAILED);
     }
   else
      bool_tester=true;
//--- secceed
   return(INIT_SUCCEEDED);
  }
```

Check if the parameters in the panel were changed in OnChartEvent(). If yes, the EA should be initialized with the new parameters:

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   ExtDialog.ChartEvent(id,lparam,dparam,sparam);
// Ask the bool variable in the panel if the parameters were changed
// If yes, ask the panel parameters and call
// CSampleExpert::Init(void)
   if(ExtDialog.Modification())
     {
      ExtDialog.GetValues(ExtMail,ExtPush,ExtAlert,
                          ExtLots,ExtTakeProfit,ExtTrailingStop,
                          ExtMACDOpenLevel,ExtMACDCloseLevel,ExtMATrendPeriod);
      if(ExtExpert.Init())
        {
         ExtDialog.Modification(false);
         Print("Parameters changed, ",ExtLots,", ",ExtTakeProfit,", ",ExtTrailingStop,", ",
               ExtMACDOpenLevel,", ",ExtMACDCloseLevel,", ",ExtMATrendPeriod);
        }
      else
        {
         ExtDialog.Modification(false);
         Print("Parameter change error");
        }
     }
  }
//+------------------------------------------------------------------+
```

### Conclusion

Combining the panel with the indicator has turned out to be easy enough. To achieve this, we have implemented the entire functionality (control elements' size and location, response to events) in the panel class, as well as declared the variable of our panel class and added the OnChartEvent() function in the indicator.

Combining the EA with the more complex panel has been more challenging, mainly due to the need to arrange "communication" between the EA and the panel. The complexity of the problem mostly depends on whether the panel is ready for connection. In other words, if the panel initially has a decent amount of functions and possibilities for integration with other programs, it will be much easier to combine it with another application (indicator or EA).

The following files are attached to the article:

- **NewBarOriginal**.mq5 — initial indicator file.
- **PanelDialogOriginal**.mqh — initial panel file.
- **NewBar**.mq5 — changed indicator file.
- **PanelDialog**.mqh — changed panel file.
- **PanelDialog2Original**.mqh — initial second panel file.
- **PanelDialog2**.mqh — changed second panel file.
- **MACD Sample**.mq5 — changed EA file.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2171](https://www.mql5.com/ru/articles/2171)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2171.zip "Download all attachments in the single ZIP archive")

[newbar.mq5](https://www.mql5.com/en/articles/download/2171/newbar.mq5 "Download newbar.mq5")(3.24 KB)

[paneldialog.mqh](https://www.mql5.com/en/articles/download/2171/paneldialog.mqh "Download paneldialog.mqh")(6.39 KB)

[newbaroriginal.mq5](https://www.mql5.com/en/articles/download/2171/newbaroriginal.mq5 "Download newbaroriginal.mq5")(2.05 KB)

[paneldialogoriginal.mqh](https://www.mql5.com/en/articles/download/2171/paneldialogoriginal.mqh "Download paneldialogoriginal.mqh")(4.86 KB)

[macd\_sample.mq5](https://www.mql5.com/en/articles/download/2171/macd_sample.mq5 "Download macd_sample.mq5")(24.12 KB)

[paneldialog2.mqh](https://www.mql5.com/en/articles/download/2171/paneldialog2.mqh "Download paneldialog2.mqh")(28.48 KB)

[paneldialog2original.mqh](https://www.mql5.com/en/articles/download/2171/paneldialog2original.mqh "Download paneldialog2original.mqh")(21.33 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717)
- [Gap - a profitable strategy or 50/50?](https://www.mql5.com/en/articles/5220)
- [Elder-Ray (Bulls Power and Bears Power)](https://www.mql5.com/en/articles/5014)
- [Improving Panels: Adding transparency, changing background color and inheriting from CAppDialog/CWndClient](https://www.mql5.com/en/articles/4575)
- [How to create a graphical panel of any complexity level](https://www.mql5.com/en/articles/4503)
- [Comparing speeds of self-caching indicators](https://www.mql5.com/en/articles/4388)
- [LifeHack for traders: Blending ForEach with defines (#define)](https://www.mql5.com/en/articles/4332)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/74560)**
(39)


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
12 Jan 2017 at 13:25

**Vitor Hervatin:**

I know this, but maybe you can help me!

I won't be able to help you - I don't support the old terminal long ago.


![Vitor Hervatin](https://c.mql5.com/avatar/2016/2/56CE8342-033E.jpeg)

**[Vitor Hervatin](https://www.mql5.com/en/users/vhervatin)**
\|
12 Jan 2017 at 13:26

**Vladimir Karputov:**

I won't be able to help you - I don't support the old terminal long ago.

Ok ok, no problem! Thanks a lot


![Yuriy Zaytsev](https://c.mql5.com/avatar/2011/11/4ECD90EB-D242.jpg)

**[Yuriy Zaytsev](https://www.mql5.com/en/users/yuraz)**
\|
28 Feb 2017 at 08:37

**Vladimir Karputov:**

1. In the nearest [update the standard library](https://www.mql5.com/en/articles/741 "MQL5 standard library extension and code reuse") will be restored - accordingly the Defines.mqh file will be restored.
2. It is not good to edit the standard library.

In theory - you can Defines.mqh - just inside the project, i.e. do not access the file

#include <Controls\\Label.mqh>

#include <Controls\\Panel.mqh>

#include <Controls\\Edit.mqh>

// #include <Controls\\Defines.mqh>

#include <Controls\\Button.mqh>

But the method below is more beautiful.

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/ru/forum)

[Why does the panel move away when updating Expert Advisor settings?](https://www.mql5.com/ru/forum/74605/page4#comment_2324750)

[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter), 2016.03.10 13:17

Approximately like this:

```
// BEFORE connecting Dialog.mqh!
#include <Controls\Defines.mqh>

//--- Forget the old colours
#undef   CONTROLS_DIALOG_COLOR_BORDER_LIGHT
#undef   CONTROLS_DIALOG_COLOR_BORDER_DARK
#undef   CONTROLS_DIALOG_COLOR_BG
#undef   CONTROLS_DIALOG_COLOR_CAPTION_TEXT
#undef   CONTROLS_DIALOG_COLOR_CLIENT_BG
#undef   CONTROLS_DIALOG_COLOR_CLIENT_BORDER

//--- Set new colours
#define  CONTROLS_DIALOG_COLOR_BORDER_LIGHT  clrWhite            // Dialog border colour (outside)
#define  CONTROLS_DIALOG_COLOR_BORDER_DARK   C'0xB6,0xB6,0xB6'   // Dialog border colour (inside)
#define  CONTROLS_DIALOG_COLOR_BG            clrLightGreen       // Dialog background (under the caption and around the client area)
#define  CONTROLS_DIALOG_COLOR_CAPTION_TEXT  C'0x28,0x29,0x3B'   // Dialog caption text colour
#define  CONTROLS_DIALOG_COLOR_CLIENT_BG     clrAliceBlue        // Client area background colour
#define  CONTROLS_DIALOG_COLOR_CLIENT_BORDER C'0xC8,0xC8,0xC8'   // Client area colour

// Now connect
#include <Controls\Dialog.mqh>
```

![Zi Feng Ding](https://c.mql5.com/avatar/2018/5/5B0AAA83-8E04.jpg)

**[Zi Feng Ding](https://www.mql5.com/en/users/freeharbor)**
\|
14 Sep 2018 at 10:10

**"prev\_time** [static](https://www.mql5.com/en/docs/basis/variables/static) today variable declared in OnCalculate() function"

Typo, it's not a today variable it's a static variable.

![Sergei Kiriakov](https://c.mql5.com/avatar/2020/8/5F4162EE-7B61.png)

**[Sergei Kiriakov](https://www.mql5.com/en/users/cergoo)**
\|
11 Dec 2023 at 07:37

I have buttons on the panel spontaneously pressed when I just move the mouse over them without clicking, it's weird.


![Graphical Interfaces II: the Menu Item Element (Chapter 1)](https://c.mql5.com/2/22/Graphic-interface-part2.png)[Graphical Interfaces II: the Menu Item Element (Chapter 1)](https://www.mql5.com/en/articles/2200)

In the second part of the series, we will show in detail the development of such interface elements as main menu and context menu. We will also mention drawing elements and create a special class for it. We will discuss in depth such question as managing program events including custom ones.

![Graphical Interfaces I: Testing Library in Programs of Different Types and in the MetaTrader 4 Terminal (Chapter 5)](https://c.mql5.com/2/21/Graphic-interface__4.png)[Graphical Interfaces I: Testing Library in Programs of Different Types and in the MetaTrader 4 Terminal (Chapter 5)](https://www.mql5.com/en/articles/2129)

In the previous chapter of the first part of the series about graphical interfaces, the form class was enriched by methods which allowed managing the form by pressing its controls. In this article, we will test our work in different types of MQL program such as indicators and scripts. As the library was designed to be cross-platform so it could be used in all MetaTrader platforms, we will also test it in MetaTrader 4.

![Fuzzy logic to create manual trading strategies](https://c.mql5.com/2/22/2195.png)[Fuzzy logic to create manual trading strategies](https://www.mql5.com/en/articles/2195)

This article suggests the ways of improving manual trading strategy by applying fuzzy set theory. As an example we have provided a step-by-step description of the strategy search and the selection of its parameters, followed by fuzzy logic application to blur overly formal criteria for the market entry. This way, after strategy modification we obtain flexible conditions for opening a position that has a reasonable reaction to a market situation.

![Trading signals module using the system by Bill Williams](https://c.mql5.com/2/20/MQL5_wizard_ru.png)[Trading signals module using the system by Bill Williams](https://www.mql5.com/en/articles/2049)

The article describes the rules of the trading system by Bill Williams, the procedure of application for a developed MQL5 module to search and mark patterns of this system on the chart, automated trading with found patterns, and also presents the results of testing on various trading instruments.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hkkbysgicsdaqwevnrcobfdmoyhkonmp&ssn=1769178388451769882&ssn_dr=0&ssn_sr=0&fv_date=1769178388&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2171&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Adding%20a%20control%20panel%20to%20an%20indicator%20or%20an%20Expert%20Advisor%20in%20no%20time%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691783883181641&fz_uniq=6478625108831172372&sv=2552)

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