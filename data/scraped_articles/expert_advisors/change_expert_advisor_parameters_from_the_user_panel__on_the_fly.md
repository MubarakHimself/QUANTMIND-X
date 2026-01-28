---
title: Change Expert Advisor Parameters From the User Panel "On the Fly"
url: https://www.mql5.com/en/articles/572
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:29:34.885495
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=dtjmwdjdiqgoplhzejifptvnzbvkqoog&ssn=1769192973679797232&ssn_dr=0&ssn_sr=0&fv_date=1769192973&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F572&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Change%20Expert%20Advisor%20Parameters%20From%20the%20User%20Panel%20%22On%20the%20Fly%22%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919297358180211&fz_uniq=5071872269686353933&sv=2552)

MetaTrader 5 / Examples


### Contents

[Introduction](https://www.mql5.com/en/articles/572#para1)

[1\. Issues in Focus](https://www.mql5.com/en/articles/572#para2)

[2\. Structure of the Expert Advisor](https://www.mql5.com/en/articles/572#para3)

[3\. Interaction with the User Panel](https://www.mql5.com/en/articles/572#para4)

[Conclusion](https://www.mql5.com/en/articles/572#para5)

### Introduction

When developing complex Expert Advisors, the number of external parameters can be very large. And settings very often need to be changed manually, making the whole process very time-consuming, given a massive list of parameters. One can, of course, prepare sets in advance and have them saved, yet it may not be exactly what is required in some cases. This is where MQL5 comes in handy, as it always does.

Let us try to create a user panel that will allow us to change parameters of an Expert Advisor "on the fly" while trading. This may be relevant to those who trade manually or in semi-automatic mode. Upon any change made, parameters will be written to a file from which they will then be read by the Expert Advisor to be further displayed on the panel.

### 1\. Issues in Focus

For illustration, we will develop a simple EA that opens a position in the direction of the [JMA](https://www.mql5.com/en/code/427 "Adaptive JMA") indicator. The EA will work on completed bars on the current symbol and time frame. External parameters will include Indicator Period, Stop Loss, Take Profit, Reverse and Lot. These options will be quite sufficient in our example.

Let us add two additional parameters to be able to turn the panel on/off (On/Off Info Panel) and enable/disable the Expert Advisor parameter setting mode ("On The Fly" Setting). Where the number of parameters is large, It is always more convenient to place additional options at the very beginning or end of the list for an easy and quick access.

![Fig. 1. Info Panel with parameters of the Expert Advisor](https://c.mql5.com/2/5/article_img_001_fixed.png)

Fig. 1. Info Panel with parameters of the Expert Advisor

"On The Fly" setting mode is disabled by default. When you enable this mode for the first time, the Expert Advisor creates a file in order to save all parameters it currently has. The same will happen if the file is accidentally deleted. The Expert Advisor will detect the deletion and recreate the file. With the "On The Fly" setting mode being disabled, the Expert Advisor will be guided by external parameters.

If this mode is enabled, the Expert Advisor will read the parameters from the file and, by simply clicking on any parameter on the info panel, you will be able to either select the required value or enter a new value in the popping up dialog window. The file data will be updated every time a new value is selected.

### 2\. Structure of the Expert Advisor

Although the program is small and all functions could easily fit in one file, it is still much more convenient to navigate all project information when it is properly categorized. Therefore, it is best to categorize the functions by type and have them in different files from the very beginning to later include them in the master file. The figure below shows a shared project folder with the OnTheFly Expert Advisor and all include files. The include files are placed in a separate folder (Include).

![Fig. 2. Project files in the Navigator window of MetaEditor](https://c.mql5.com/2/5/article_img_02_en.png)

Fig. 2. Project files in the Navigator window of MetaEditor

When the include files are in the same folder with the master file, the code is as follows:

```
//+------------------------------------------------------------------+
//| CUSTOM LIBRARIES                                                 |
//+------------------------------------------------------------------+
#include "Include/!OnChartEvent.mqh"
#include "Include/CREATE_PANEL.mqh"
#include "Include/FILE_OPERATIONS.mqh"
#include "Include/ERRORS.mqh"
#include "Include/ARRAYS.mqh"
#include "Include/TRADE_SIGNALS.mqh"
#include "Include/TRADE_FUNCTIONS.mqh"
#include "Include/GET_STRING.mqh"
#include "Include/GET_COLOR.mqh"
#include "Include/ADD_FUNCTIONS.mqh"
```

More information on how to include files can be found in [MQL5 Reference](https://www.mql5.com/en/docs/basis/preprosessor/include "Including files (#include)").

We will need global variables - copies of external parameters. Their values will either be assigned from the external parameters or the file, depending on the Expert Advisor's mode. These variables are used throughout the entire program code, e.g. in displaying values on the info panel, in trading functions, etc.

```
// COPY OF EXTERNAL PARAMETERS
int    gPeriod_Ind = 0;
double gTakeProfit = 0.0;
double gStopLoss   = 0.0;
bool   gReverse    = false;
double gLot        = 0.0;
```

As in all other Expert Advisors, we will have the main functions: [OnInit](https://www.mql5.com/en/docs/basis/function/events#oninit), [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick) and [OnDeinit](https://www.mql5.com/en/docs/basis/function/events#ondeinit). And there will also be the [OnTimer](https://www.mql5.com/en/docs/basis/function/events#ontimer) function. Every second it will check the existence of the parameter file and restore it in case it was accidentally deleted. Since we need to interact with the user panel, the [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events#onchartevent) function will be used, too. This function together with other related functions has been placed in a separate file (!OnChartEvent.mqh).

The core code of the master file is as follows:

```
#define szArrIP 5 // Size of the parameter array
#define NAME_EXPERT MQL5InfoString(MQL5_PROGRAM_NAME) // Name of EA
#define TRM_DP TerminalInfoString(TERMINAL_DATA_PATH) // Folder that contains the terminal data
//+------------------------------------------------------------------+
//| STANDARD LIBRARIES                                               |
//+------------------------------------------------------------------+
#include <Trade/SymbolInfo.mqh>
#include <Trade/Trade.mqh>
//+------------------------------------------------------------------+
//| CUSTOM LIBRARIES                                                 |
//+------------------------------------------------------------------+
#include "Include/!OnChartEvent.mqh"
#include "Include/CREATE_PANEL.mqh"
#include "Include/FILE_OPERATIONS.mqh"
#include "Include/ERRORS.mqh"
#include "Include/ARRAYS.mqh"
#include "Include/TRADE_SIGNALS.mqh"
#include "Include/TRADE_FUNCTIONS.mqh"
#include "Include/GET_STRING.mqh"
#include "Include/GET_COLOR.mqh"
#include "Include/ADD_FUNCTIONS.mqh"
//+------------------------------------------------------------------+
//| CREATING CLASS INSTANCES                                         |
//+------------------------------------------------------------------+
CSymbolInfo mysymbol; // CSymbolInfo class object
CTrade      mytrade;  // CTrade class object
//+------------------------------------------------------------------+
//| EXTERNAL PARAMETERS                                              |
//+------------------------------------------------------------------+
input int    Period_Ind      = 10;    // Indicator Period
input double TakeProfit      = 100;   // Take Profit (p)
input double StopLoss        = 30;    // Stop Loss (p)
input bool   Reverse         = false; // Reverse Position
input double Lot             = 0.1;   // Lot
//---
input string slash="";       // * * * * * * * * * * * * * * * * * * *
sinput bool  InfoPanel       = true;  // On/Off Info Panel
sinput bool  SettingOnTheFly = false; // "On The Fly" Setting
//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                 |
//+------------------------------------------------------------------+
int hdlSI=INVALID_HANDLE; // Signal indicator handle
double lcheck=0;  // For the check of parameter values
bool isPos=false; // Position availability
//--- COPY OF EXTERNAL PARAMETERS
int    gPeriod_Ind = 0;
double gTakeProfit = 0.0;
double gStopLoss   = 0.0;
bool   gReverse    = false;
double gLot        = 0.0;
//+------------------------------------------------------------------+
//| EXPERT ADVISOR INITIALIZATION                                    |
//+------------------------------------------------------------------+
void OnInit()
  {
   if(NotTest()) { EventSetTimer(1); } // If it's not the tester, set the timer
//---
   Init_arr_vparams(); // Initialization of the array of parameter values
   SetParameters(); // Set the parameters
   GetIndicatorsHandles(); // Get indicator handles
   NewBar(); // New bar initialization
   SetInfoPanel(); // Info panel
  }
//+------------------------------------------------------------------+
//| CURRENT SYMBOL TICKS                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// If the bar is not new, exit
   if(!NewBar()) { return; }
   else
     { TradingBlock(); }
  }
//+------------------------------------------------------------------+
//| TIMER                                                            |
//+------------------------------------------------------------------+
void OnTimer()
  {
   SetParameters(); SetInfoPanel();
  }
//+------------------------------------------------------------------+
//| EXPERT ADVISOR DEINITIALIZATION                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// Get the deinitialization reason code
   if(NotTest()) {
     { Print(getUnitReasonText(reason)); }
//---
// When deleting from the chart
   if(reason==REASON_REMOVE)
     {
      // Delete all objects created by the Expert Advisor
      DeleteAllExpertObjects();
      //---
      if(NotTest()) { EventKillTimer(); } // Stop the timer
      IndicatorRelease(hdlSI); // Delete the indicator handle
     }
  }
```

I have also included a few more functions in the master file:

- **GetIndicatorsHandles** – gets the indicator handle.
- **NewBar** – determines the new bar event.
- **SetParameters** – sets parameters depending on the mode.
- **iZeroMemory** – zeroes out some variables and arrays.

Source codes for these functions can be found in the files attached to the article. Here, we will only review the SetParameters function (explanatory comments are provided in the code):

```
//+------------------------------------------------------------------+
//| SETTING PARAMETERS IN TWO MODES                                  |
//+------------------------------------------------------------------+
// If this variable is set to false, the parameters in the file are read from the array
// where they are saved for quick access after they have been read for the first time.
// The variable is zeroed out upon the change in the value on the panel.
bool flgRead=false;
double arrParamIP[]; // Array where the parameters from the file are saved
//---
void SetParameters()
  {
// If currently in the tester or
// in real time but with the "On The Fly" Setting mode disabled
   if(!NotTest() || (NotTest() && !SettingOnTheFly))
     {
      // Zero out the variable and parameter array
      flgRead=false;
      ArrayResize(arrParamIP,0);
      //---
      // Check the Indicator Period for correctness
      if(Period_Ind<=0)
        { lcheck=10; }
      else { lcheck=Period_Ind; }
      gPeriod_Ind=(int)lcheck;
      //---
      gStopLoss=StopLoss;
      gTakeProfit=TakeProfit;
      gReverse=Reverse;
      //---
      // Check the Lot for correctness
      if(Lot<=0)
        { lcheck=0.1; }
      else { lcheck=Lot; }
      gLot=lcheck;
     }
   else // If "On The Fly" Setting mode is enabled
     {
      // Check whether there is a file to write/read parameters to/from the file
      string lpath="";
      //---
      // If the folder exists
      if((lpath=CheckCreateGetPath())!="")
        {
         // Write or read the file
         WriteReadParameters(lpath);
        }
     }
  }
```

The source code for the SetParameters function is simple and straightforward. Let us have a closer look at the WriteReadParameters function. Everything is pretty simple here. First we check if the file with parameters exists. If it does, we read the file and write parameter values to an array using the GetValuesParamsFromFile function. If the file does not exist, it will be created, with current external parameters written to it.

Below is the code with more detailed comments for the implementation of the actions described above:

```
//+------------------------------------------------------------------+
//| WRITE DATA TO FILE                                               |
//+------------------------------------------------------------------+
void WriteReadParameters(string pth)
  {
   string nm_fl=pth+"ParametersOnTheFly.ini"; // File name and path
//---
// Get the file handle to read the file
   int hFl=FileOpen(nm_fl,FILE_READ|FILE_ANSI);
//---
   if(hFl!=INVALID_HANDLE) // If the handle has been obtained, the file exists
     {
      // Get parameters from the file
      if(!flgRead)
        {
         // Set the array size
         ArrayResize(arrParamIP,szArrIP);
         //---
         // Fill the array with values from the file
         flgRead=GetValuesParamsFromFile(hFl,arrParamIP);
        }
      //---
      // If the array size is correct,...
      if(ArraySize(arrParamIP)==szArrIP)
        {
         // ...set the parameters to the variables
         //---
         // Check the Indicator Period for correctness
         if((int)arrParamIP[0]<=0) { lcheck=10; }
         else { lcheck=(int)arrParamIP[0]; }
         gPeriod_Ind=(int)lcheck;
         //---
         gTakeProfit=arrParamIP[1];
         gStopLoss=arrParamIP[2];
         gReverse=arrParamIP[3];
         //---
         // Check the Lot for correctness
         if(arrParamIP[4]<=0)
           { lcheck=0.1; }
         else { lcheck=arrParamIP[4]; }
         gLot=lcheck;
        }
     }
   else // If the file does not exist
     {
      iZeroMemory(); // Zero out variables
      //---
      // When creating the file, write current parameters of the Expert Advisor
      //---
      // Get the file handle to write to the file
      int hFl2=FileOpen(nm_fl,FILE_WRITE|FILE_CSV|FILE_ANSI,"");
      //---
      if(hFl2!=INVALID_HANDLE) // If the handle has been obtained
        {
         string sep="=";
         //---
         // Parameter names and values are obtained from arrays in the ARRAYS.mqh file
         for(int i=0; i<szArrIP; i++)
           { FileWrite(hFl2,arr_nmparams[i],sep,arr_vparams[i]); }
         //---
         FileClose(hFl2); // Close the file
         //---
         Print("File with parameters of the "+NAME_EXPERT+" Expert Advisor created successfully.");
        }
     }
//---
   FileClose(hFl); // Close the file
  }
```

The WriteReadParameters and GetValuesParamsFromFile functions can be found in the FILE\_OPERATIONS.mqh file.

Some of the functions have already been described in my previous article ["How to Prepare MetaTrader 5 Quotes for Other Applications"](https://www.mql5.com/en/articles/502 "How to Prepare MetaTrader 5 Quotes for Other Applications"), therefore we will not dwell on them here. You should not experience any difficulties with trading functions either, as they are very straightforward and commented out extensively. What we are going to focus on is the main subject of the article.

### 3\. Interaction with the User Panel

The !OnChartEvent.mqh file contains functions for interaction with the user panel. Variables and arrays that are used in many functions are declared in the global scope at the very beginning:

```
// Current value on the panel or
// entered in the input box
string currVal="";
bool flgDialogWin=false; // Flag for panel existence
int
szArrList=0,// Size of the option list array
number=-1; // Parameter number in the panel list
string
nmMsgBx="",  // Name of the dialog window
nmValObj=""; // Name of the selected object
//---
// Option list arrays in the dialog window
string lenum[],lenmObj[];
//---
// colors of the dialog window elements
color
clrBrdBtn=clrWhite,
clrBrdFonMsg=clrDimGray,clrFonMsg=C'15,15,15',
clrChoice=clrWhiteSmoke,clrHdrBtn=clrBlack,
clrFonHdrBtn=clrGainsboro,clrFonStr=C'22,39,38';
```

This is followed by the main function that handles events. In our example, we will need to handle two events:

- The **CHARTEVENT\_OBJECT\_CLICK** event – left-click on the graphical object.
- The **CHARTEVENT\_OBJECT\_EDIT** event – end of text editing in the Edit graphical object.

You can read more on other MQL5 events in [MQL5 Reference](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents "Types of chart events").

Let us first set a check for handling the events in real time only, always provided that the "On The Fly" setting mode is enabled (SettingOnTheFly). The handling of events will be dealt with by separate functions: ChartEvent\_ObjectClick and ChartEvent\_ObjectEndEdit.

```
//+------------------------------------------------------------------+
//| USER EVENTS                                                      |
//+------------------------------------------------------------------+
void OnChartEvent(const int     id,
                  const long   &lparam,
                  const double &dparam,
                  const string &sparam)
  {
// If the event is real time and the "On The Fly" setting mode is enabled
   if(NotTest() && SettingOnTheFly)
     {
      //+------------------------------------------------------------------+
      //| THE CHARTEVENT_OBJECT_CLICK EVENT                                |
      //+------------------------------------------------------------------+
      if(ChartEvent_ObjectClick(id,lparam,dparam,sparam)) { return; }
      //---
      //+------------------------------------------------------------------+
      //| THE CHARTEVENT_OBJECT_ENDEDIT EVENT                              |
      //+------------------------------------------------------------------+
      if(ChartEvent_ObjectEndEdit(id,lparam,dparam,sparam)) { return; }
     }
//---
   return;
  }
```

When you click on the object that belongs to the list, a dialog window will appear on the info panel allowing you to select another value or enter a new value in the input box.

![Fig. 3. Dialog window for modifications of the value of the selected parameter](https://c.mql5.com/2/5/article_img_003_fixed.png)

Fig. 3. Dialog window for modifications of the value of the selected parameter

Let us have a closer look at how it works. When a graphical object is clicked, the program first uses the ChartEvent\_ObjectClick function to check by the event identifier whether there really was a click on a graphical object.

If you want the dialog window to open in the middle of the chart, you need to know the chart size. It can be obtained by indicating the CHART\_WIDTH\_IN\_PIXELS and CHART\_HEIGHT\_IN\_PIXELS properties in the [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) function. The program then switches over to the DialogWindowInfoPanel. You can familiarize yourself with all chart properties in [MQL5 Reference](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer "Chart properties: ENUM_CHART_PROPERTY_INTEGER").

Below is the code for the implementation of the above actions:

```
//+------------------------------------------------------------------+
//| THE CHARTEVENT_OBJECT_CLICK EVENT                                |
//+------------------------------------------------------------------+
bool ChartEvent_ObjectClick(int id,long lparam,double dparam,string sparam)
  {
   // If there was an event of clicking on a graphical object
   if(id==CHARTEVENT_OBJECT_CLICK) // id==1
     {
      Get_STV(); // Get all data on the symbol
      //---
      string clickedChartObject=sparam; // Name of the clicked object
      //---
      // Get the chart size
      width_chart=(int)ChartGetInteger(0,CHART_WIDTH_IN_PIXELS,0);
      height_chart=(int)ChartGetInteger(0,CHART_HEIGHT_IN_PIXELS,0);
      //---
      DialogWindowInfoPanel(clickedChartObject);
     }
//---
   return(false);
  }
```

Using the DialogWindowInfoPanel function, we first check whether the dialog window is currently open. If the window is not found, the GetNumberClickedObjIP function checks whether the click was in relation to an object from the list on the info panel. If the clicked object is the object from the list, the function will return the relevant element number from the array of objects. Using that number, the InitArraysAndDefault function then determines the list array size in the dialog window and default values. If all actions are successful, the dialog window will appear.

If the DialogWindowInfoPanel function determines that the dialog window is already open, the program will check whether there was a click on an object in the dialog window. For example, upon opening the dialog window, the line whose value is currently displayed on the panel will appear as selected. If you click on another option in the list, the program will use the SelectionOptionInDialogWindow function that selects the dialog window list option clicked.

If you click on the list option that is currently selected, this object will be identified as an object to be edited and an input box will appear so that a new value can be entered when you click on the box. The SetEditObjInDialogWindow function is responsible for setting the input box.

And finally, if the Apply button was clicked, the program will check if the value has been modified. If it has, the new value will appear on the panel and will be written to the file.

The code of the main function of the dialog window is provided below:

```
//+------------------------------------------------------------------+
//| DIALOG WINDOW OF THE INFO PANEL                                  |
//+------------------------------------------------------------------+
void DialogWindowInfoPanel(string clickObj)
  {
// If there is currently no dialog window
   if(!flgDialogWin)
     {
      // Get the object number in the array
      // Exit if none of the parameters displayed on the panel has been clicked
      if((number=GetNumberClickedObjIP(clickObj))==-1) { return; }
      //---
      // Initialization of default values
      //and determination of the list array size
      if(!InitArraysAndDefault()) { return; }
      //---
      // Set the dialog window
      SetDialogWindow();
      //---
      flgDialogWin=true; // Mark the dialog window as open
      ChartRedraw();
     }
   else // If the dialog window is open
     {
      // Set the input box for the modification of the value
      SetEditObjInDialogWindow(clickObj);
      //---
      // If one of the buttons in the dialog box is clicked
      if(clickObj=="btnApply" || clickObj=="btnCancel")
        {
         // If the Apply button is clicked
         if(clickObj=="btnApply")
           {
            // Compare values on the panel with the ones on the list
            // If the value on the list is different from the one that is currently displayed on the panel
            // (which means it is different from the one in the file),
            // ...change the value on the panel and update the file
            if(currVal!=ObjectGetString(0,nmValObj,OBJPROP_TEXT))
              {
               // Update the value on the panel
               ObjectSetString(0,nmValObj,OBJPROP_TEXT,currVal); ChartRedraw();
               //---
               // Read all data on the panel and write it to the file
               WriteNewData();
              }
           }
         //---
         DelDialogWindow(lenmObj); // Delete the dialog window
         iZeroMemory(); // Zero out the variables
         //---
         // Update the data
         SetParameters();
         GetHandlesIndicators();
         SetInfoPanel();
         //---
         ChartRedraw();
        }
      else // If neither Apply nor Cancel has been clicked
        {
         // Selection of the dialog window list option
         SelectionOptionInDialogWindow(clickObj);
         //---
         ChartRedraw();
        }
     }
  }
```

Every time a new value is entered in the input box, the CHARTEVENT\_OBJECT\_EDIT event is generated and the program switches over to the ChartEvent\_ObjectEndEdit function. If the value from the dialog window has been modified, the entered value will be stored, checked for correctness and assigned to the object in the list. You can see it in more detail in the code below:

```
//+------------------------------------------------------------------+
//| THE CHARTEVENT_OBJECT_ENDEDIT EVENT                              |
//+------------------------------------------------------------------+
bool ChartEvent_ObjectEndEdit(int id,long lparam,double dparam,string sparam)
  {
   if(id==CHARTEVENT_OBJECT_ENDEDIT) // id==3
     {
      string editObject=sparam; // Name of the edited object
      //---
      // If the value has been entered in the input box in the dialog window
      if(editObject=="editValIP")
        {
         // Get the entered value
         currVal=ObjectGetString(0,"editValIP",OBJPROP_TEXT);
         //---
         // (0) Period Indicator
         if(number==0)
           {
            // Correct the value if it is wrong
            if(currVal=="0" || currVal=="" || SD(currVal)<=0) { currVal="1"; }
            //---
            // Set the entered value
            ObjectSetString(0,"enumMB0",OBJPROP_TEXT,currVal);
           }
         //---
         // (4) Lot
         if(number==4)
           {
            // Correct the value if it is wrong
            if(currVal=="0" || currVal=="" || SD(currVal)<=0) { currVal=DS(SS.vol_min,2); }
            //---
            // Set the entered value
            ObjectSetString(0,"enumMB0",OBJPROP_TEXT,DS2(SD(currVal)));
           }
         //---
         // (1) Take Profit (p)
         // (2) Stop Loss (p)
         if(number==1 || number==2)
           {
            // Correct the value if it is wrong
            if(currVal=="0" || currVal=="" || SD(currVal)<=0) { currVal="1"; }
            //---
            // Set the entered value
            ObjectSetString(0,"enumMB1",OBJPROP_TEXT,currVal);
           }
         //---
         DelObjbyName("editValIP"); ChartRedraw();
        }
     }
//---
   return(false);
  }
```

The Expert Advisor in action can be seen in the video below:

YouTube

### Conclusion

Zipped files attached at the end of the article can be downloaded for closer study.

I hope this article will help those of you who only start learning MQL5 to find quick answers to many questions using the simple examples given. I intentionally left out some checks from the provided code snippets.

For example, if you change the chart height/width when the dialog window is open, the dialog window will not be automatically centered. And if you top it up with selecting another option from the list, the object that serves to select the relevant line will be considerably shifted. Let this be your homework. It is very important to practice programming and the more you practice, the better.

Good luck!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/572](https://www.mql5.com/ru/articles/572)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/572.zip "Download all attachments in the single ZIP archive")

[onthefly\_en.zip](https://www.mql5.com/en/articles/download/572/onthefly_en.zip "Download onthefly_en.zip")(26.38 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Magic of time trading intervals with Frames Analyzer tool](https://www.mql5.com/en/articles/11667)
- [The power of ZigZag (part II). Examples of receiving, processing and displaying data](https://www.mql5.com/en/articles/5544)
- [The power of ZigZag (part I). Developing the base class of the indicator](https://www.mql5.com/en/articles/5543)
- [Universal RSI indicator for working in two directions simultaneously](https://www.mql5.com/en/articles/4828)
- [Expert Advisor featuring GUI: Adding functionality (part II)](https://www.mql5.com/en/articles/4727)
- [Expert Advisor featuring GUI: Creating the panel (part I)](https://www.mql5.com/en/articles/4715)
- [Visualizing optimization results using a selected criterion](https://www.mql5.com/en/articles/4636)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/10616)**
(52)


![Rustamzhan Salidzhanov](https://c.mql5.com/avatar/2014/7/53BC95D7-C166.jpg)

**[Rustamzhan Salidzhanov](https://www.mql5.com/en/users/faq)**
\|
23 Nov 2012 at 18:12

**komposter:**

That's not gonna fit in a half a bite. And it won't be free in the base.

But now it's a different task.

Well, it didn't cost that much.


![Denis Lazarev](https://c.mql5.com/avatar/avatar_na2.png)

**[Denis Lazarev](https://www.mql5.com/en/users/lazarev-d-m)**
\|
23 Nov 2012 at 18:16

**FAQ:**

No, I'm trying to get the dialogue going in the direction of developing a really intuitive and understandable interface.

Ideally it would be nice if the tester was intuitive, but to start with, we could first move the idea to the beginning of implementation, and when something is ready to improve towards clarity and "understandability".


![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
15 Feb 2013 at 12:45

Please help. I want set period parameters of custom-indicators "on the fly", but not work, why?

```
ENUM_TIMEFRAMES tf0,tf,tf1;
void OnInit()
  {
.................
tf0 = PERIOD_CURRENT;
  switch (PERIOD_CURRENT)
  {
     case PERIOD_M1:   tf =PERIOD_M5;tf1 =PERIOD_M15; break;
      case PERIOD_M5:   tf =PERIOD_M15; tf1 =PERIOD_H1;break;
      case PERIOD_M15:   tf =PERIOD_H1;tf1 =PERIOD_H4;break;
      case PERIOD_M30:   tf =PERIOD_H2;tf1 =PERIOD_H8;  break;
      case PERIOD_H1:   tf =PERIOD_H4; tf1 =PERIOD_H12;break;
      case PERIOD_H4:   tf =PERIOD_H12;  tf1 =PERIOD_D1; break;
      case PERIOD_D1:   tf =PERIOD_D1;  tf1 =PERIOD_W1;  break;
  }
//--- get MA's handles
   Ext1Handle=iCustom(NULL,PERIOD_CURRENT,"xxxx",SlowEMA1);//work ok
   Ext2Handle=iCustom(NULL,PERIOD_M5,"xxxx",SlowEMA1);//work ok
   Ext3Handle=iCustom(NULL,PERIOD_M15,"xxxx",SlowEMA1);//work ok
//below worked error!
// Ext1Handle=iCustom(NULL,PERIOD_CURRENT,"xxxx",SlowEMA1);
// Ext2Handle=iCustom(NULL,tf,"xxxx",SlowEMA1);
// Ext3Handle=iCustom(NULL,tf1,"xxxx",SlowEMA1);
```

not work for below code: so can not [change period](https://www.mql5.com/en/docs/check/period "MQL5 documentation: Period function") parameter on the fly!!!

Ext1Handle=iCustom(NULL,PERIOD\_CURRENT,"xxxx",SlowEMA1);

Ext2Handle=iCustom(NULL,tf,"xxxx",SlowEMA1);

Ext3Handle=iCustom(NULL,tf1,"xxxx",SlowEMA1);

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
23 May 2014 at 16:09

Good Afternoon !

Any idea why this is happening ?

Error in red @ the bottom left corner

Thank you

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
23 May 2014 at 16:21

Found the problem:

Line 172 is incorrect.

Correct Value is: GetIndicatorsHandles();

File is !OnChartEvent.mqh

![MetaTrader 4 Expert Advisor exchanges information with the outside world](https://c.mql5.com/2/13/1062_113.jpg)[MetaTrader 4 Expert Advisor exchanges information with the outside world](https://www.mql5.com/en/articles/1361)

A simple, universal and reliable solution of information exchange between МetaТrader 4 Expert Advisor and the outside world. Suppliers and consumers of the information can be located on different computers, the connection is performed through the global IP addresses.

![Fast Testing of Trading Ideas on the Chart](https://c.mql5.com/2/0/avatar__23.png)[Fast Testing of Trading Ideas on the Chart](https://www.mql5.com/en/articles/505)

The article describes the method of fast visual testing of trading ideas. The method is based on the combination of a price chart, a signal indicator and a balance calculation indicator. I would like to share my method of searching for trading ideas, as well as the method I use for fast testing of these ideas.

![Calculation of Integral Characteristics of Indicator Emissions](https://c.mql5.com/2/0/avatar__22.png)[Calculation of Integral Characteristics of Indicator Emissions](https://www.mql5.com/en/articles/610)

Indicator emissions are a little-studied area of market research. Primarily, this is due to the difficulty of analysis that is caused by the processing of very large arrays of time-varying data. Existing graphical analysis is too resource intensive and has therefore triggered the development of a parsimonious algorithm that uses time series of emissions. This article demonstrates how visual (intuitive image) analysis can be replaced with the study of integral characteristics of emissions. It can be of interest to both traders and developers of automated trading systems.

![MQL5 Market Turns One Year Old](https://c.mql5.com/2/0/mql5-market-1year-avatar.png)[MQL5 Market Turns One Year Old](https://www.mql5.com/en/articles/632)

One year has passed since the launch of sales in MQL5 Market. It was a year of hard work, which turned the new service into the largest store of trading robots and technical indicators for MetaTrader 5 platform.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/572&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071872269686353933)

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