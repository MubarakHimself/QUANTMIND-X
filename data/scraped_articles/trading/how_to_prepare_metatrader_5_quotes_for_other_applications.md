---
title: How to Prepare MetaTrader 5 Quotes for Other Applications
url: https://www.mql5.com/en/articles/502
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:21:37.996978
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ymmpyynpgekyiqcwjgjqohplkqaqaznk&ssn=1769181696514947131&ssn_dr=0&ssn_sr=0&fv_date=1769181696&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F502&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Prepare%20MetaTrader%205%20Quotes%20for%20Other%20Applications%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918169612189302&fz_uniq=5069373187130590207&sv=2552)

MetaTrader 5 / Examples


### Contents

[Introduction](https://www.mql5.com/en/articles/502#01)

[1\. Covered Topics](https://www.mql5.com/en/articles/502#02)

[2\. Data Format](https://www.mql5.com/en/articles/502#para3)

[3\. Program's External Parameters](https://www.mql5.com/en/articles/502#para4)

[4\. Checking Parameters Entered by a User](https://www.mql5.com/en/articles/502#para5)

[5\. Global Variables](https://www.mql5.com/en/articles/502#para6)

[6\. Information Panel](https://www.mql5.com/en/articles/502#para7)

[7\. Application's Main Block](https://www.mql5.com/en/articles/502#para8)

[8\. Creating Folders and Filing the Data](https://www.mql5.com/en/articles/502#para9)

[Conclusion](https://www.mql5.com/en/articles/502#para10)

### Introduction

Before I started studying MQL5, I tried many other applications for development of trading systems. I can't say that I wasted my time. Some of them contain a few useful tools allowing users to save time, deal with many issues, destroy some myths and quickly select some further direction for development without the knowledge of programming languages.

These applications need historical data. Due to the absence of some certain standard data format, they often had to be edited before they could be used (for example, in Excel) to comply with the format applicable to the necessary program. Even if you are able to figure out all necessary details, many things should still be done manually. Users can find different versions of scripts designed to copy the quotes from MetaTrader 4 to the necessary format. If there is such a demand, we can also develop the version of the script for MQL5.

### 1\. Covered Topics

The article deals with the following topics:

- Working with the symbol list in [Market Watch](https://www.metatrader5.com/en/terminal/help/trading/market_watch "https://www.metatrader5.com/en/terminal/help/trading/market_watch") window and the common symbol list on the server.
- Checking available data depth and downloading missing amount if necessary with correct handling of various situations.
- Displaying information about the requested data on the custom panel chart and the [journal](https://www.metatrader5.com/en/terminal/help/startworking/interface "https://www.metatrader5.com/en/terminal/help/startworking/interface").
- Preparing data for filing in a user-defined format.
- Creating file directories.
- Data filing.

### 2\. Data Format

I will give an example of preparing the data meant to be used in NeuroShell DayTrader Professional (NSDT). I have tried NSDT versions 5 and 6 and found out that they have different requirements to the data format. NSDT version 5 date and time data should be in different columns. The first line in the file should have the following look:

> "Date" "Time" "Open" "High" "Low" "Close" "Volume"

The header line in NSDT version 6 should have a different look to allow the application to accept a file. It means that date and time should be in the same column:

> Date,Open,High,Low,Close,Volume

MetaTrader 5 allows users to save quotes in \*.csv files. The data in a file look as follows:

![Fig. 1. Data saved by MetaTrader 5 terminal](https://c.mql5.com/2/4/omyj08_guebqq57ish_jxtp1uenmn_MetaTrader_5__1.png)

Fig. 1. Data saved by MetaTrader 5 terminal

However, we cannot just edit the header line as the date should have another format. For NSDT v.5:

> dd.mm.yyyy,hh:mm,Open,High,Low,Close,Volume

For NSDT v.6:

> dd/mm/yyyy hh:mm,Open,High,Low,Close,Volume

Drop-down lists will be used in the script external parameters where users are able to select the necessary format. Apart from selecting header and date formats, we will grant users the possibility to select the number of the symbols, data on which they want to write to files. To do this, we will prepare three versions:

- Write the data on the current symbol only, on the chart of which (ONLY CURRENT SYMBOL) script has been launched.
- Write the data on the symbols located in Market Watch window (MARKETWATCH SYMBOLS).
- Write the data on all the symbols available on the server (ALL LIST SYMBOLS).

Let's enter the following code before the external parameters in the script code to create such lists:

```
//_________________________________
// HEADER_FORMATS_ENUMERATION
enum FORMAT_HEADERS
  {
   NSDT_5 = 0, // "Date" "Time" "Open" "High" "Low" "Close" "Volume"
   NSDT_6 = 1  // Date,Open,High,Low,Close,Volume
  };
//---
//___________________________
// ENUMERATION_OF_DATA_FORMATS
enum FORMAT_DATETIME
  {
   SEP_POINT1 = 0, // dd.mm.yyyy hh:mm
   SEP_POINT2 = 1, // dd.mm.yyyy, hh:mm
   SEP_SLASH1 = 2, // dd/mm/yyyy hh:mm
   SEP_SLASH2 = 3  // dd/mm/yyyy, hh:mm
  };
//---
//____________________________
// ENUMERATION_OF_FILING_MODES
enum CURRENT_MARKETWATCH
  {
   CURRENT          = 0, // ONLY CURRENT SYMBOLS
   MARKETWATCH      = 1, // MARKETWATCH SYMBOLS
   ALL_LIST_SYMBOLS = 2  // ALL LIST SYMBOLS
  };
```

More on enumerations can be found in [MQL5 Reference](https://www.mql5.com/en/docs/basis/types/integer/enumeration).

### 3\. Program's External Parameters

Now, we can create the entire list of all external parameters of the script:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| EXTERNAL_PARAMETERS                                              |
//+------------------------------------------------------------------+
input datetime            start_date     = D'01.01.2011'; // Start Date
input datetime            end_date       = D'18.09.2012'; // End Date
input FORMAT_HEADERS      format_headers = NSDT_5;     // Format Headers
input FORMAT_DATETIME     format_date    = SEP_POINT2; // Format Datetime
input CURRENT_MARKETWATCH curr_mwatch    = CURRENT;    // Mode Write Symbols
input bool                clear_mwatch   = true;        // Clear Market Watch
input bool                show_progress  = true;        // Show Progress (%)
```

External parameters are used for the following purposes:

- Users can specify a date interval using Start Date (start\_date) and End Date (end\_date) parameters.
- Format Headers (format\_headers) drop-down list allows users to choose a header format.
- Format Datetime (format\_date) drop-down list allows users to choose date and time formats.
- Mode Write Symbols (curr\_mwatch) drop-down list allows users to select the number of symbols for filing.
- If Clear Market Watch (clear\_mwatch) parameter is true, that allows users to delete all symbols from Market Watch window after the filing. This concerns only the symbols with the charts that are not active at the moment.
- Show Progress (%) (show\_progress) parameter displays the filing progress in data panel. The filing will be performed faster if this parameter is disabled.

Below is how the external parameters will look during the launch:

![Fig. 2. Application's external parameters](https://c.mql5.com/2/4/Image_2.png)

Fig. 2. Application's external parameters

### 4\. Checking Parameters Entered by a User

Let's create the function for checking the parameters entered by users before the base code. For example, the start date in Start Date parameter should be earlier than the one in End Date. Format of headers should match date and time formats. If a user has made some errors when setting the parameters, the following warning message is displayed and the program is stopped.

Sample warning message:

![Fig. 3. Warning of incorrectly specified values](https://c.mql5.com/2/4/Image_4.png)

Fig. 3. Sample error message on incorrectly specified parameters

ValidationParameters() function:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| CHECKING_CORRECTNESS_OF_PARAMETERS                               |
//+------------------------------------------------------------------+
bool ValidationParameters()
  {
   if(start_date>=end_date)
     {
      MessageBox("The start date should be earlier than the ending one!\n\n"
                 "Application cannot continue. Please retry.",
                 //---
                 "Parameter error!",MB_ICONERROR);
      //---
      return(true);
     }
//---
   if(format_headers==NSDT_5 &&
      (format_date==SEP_POINT1 || format_date==SEP_SLASH1))
     {
      MessageBox("For the headers of the following format:\n\n"
                 "\"Date\" ""\"Time\" ""\"Open\" ""\"High\" ""\"Low\" ""\"Close\" ""\"Volume\"\n\n"
                 "Date/time format can be selected out of two versions:\n\n"
                 "dd.mm.yyyy, hh:mm\n"
                 "dd/mm/yyyy, hh:mm\n\n"
                 "Application cannot continue. Please retry.",
                 //---
                 "Header and date/time formats do not match!",MB_ICONERROR);
      //---
      return(true);
     }
//---
   if(format_headers==NSDT_6 &&
      (format_date==SEP_POINT2 || format_date==SEP_SLASH2))
     {
      MessageBox("For the headers of the following format:\n\n"
                 "Date,Open,High,Low,Close,Volume\n\n"
                 "Date/time format can be selected out of two versions:\n\n"
                 "dd.mm.yyyy hh:mm\n"
                 "dd/mm/yyyy hh:mm\n\n"
                 "Application cannot continue. Please retry.",
                 //---
                 "Header and date/time formats do not match!",MB_ICONERROR);
      //---
      return(true);
     }
//---
   return(false);
  }
```

### 5\. Global Variables

Next, we should determine all [global variables](https://www.mql5.com/en/docs/basis/variables/global) and arrays that will be used in the script:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| GLOBAL_VARIABLES_AND_ARRAYS                                      |
//+------------------------------------------------------------------+
MqlRates rates[]; // Array for copying data
//---
string symbols[]; // Symbol array
//---
// Array of graphic object names
string arr_nmobj[22]=
  {
   "fon","hd01",
   "nm01","nm02","nm03","nm04","nm05","nm06","nm07","nm08","nm09","nm10",
   "nm11","nm12","nm13","nm14","nm15","nm16","nm17","nm18","nm19","nm20"
  };
//---
// Array of displayed text containing graphic objects
string arr_txtobj[21];
//---
string path="";         // File path
int cnt_symb=0;         // Number of symbols
int sz_arr_symb=0;      // Symbol array size
int bars=0;             // Number of bars according to the specified TF
int copied_bars=0;      // Number of bars copied for writing
double pgs_pcnt=0;      // Writing progress
int hFl=INVALID_HANDLE;  // File handle
//---
string   // Variables for data formatting
sdt="",  // Date line
dd="",   // Day
mm="",   // Month
yyyy="", // Year
tm="",   // Time
sep="";  // Separator
//---
int max_bars=0; // Maximum number of bars in the terminal settings
//---
datetime
first_date=0,        // First available data in a specified period
first_termnl_date=0, // First available data in the terminal's database
first_server_date=0, // First available data in the server's database
check_start_date=0;  // Checked correct date value
```

### 6\. Information Panel

Now, we should deal with the elements that are to be displayed at the information panel. Three types of graphic objects can be used as a background:

- The most simple and obvious one – "Rectangle label" ( [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object)).
- Those who want their interface to have a unique look can use "Bitmap" object ( [OBJ\_BITMAP](https://www.mql5.com/en/docs/constants/objectconstants/enum_object)).
- "Edit" object ( [OBJ\_EDIT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object)) can also be used as a background. Set "read only" property to remove the possibility to enter a text. There is also another advantage of using "Edit" object. If you have created an information panel in an Expert Advisor and you want it to have the same look during the tests in visualization mode, the last type is the only method to achieve that so far. Neither OBJ\_RECTANGLE\_LABEL, nor OBJ\_BITMAP are displayed during the tests in visualization mode.

Though only a script instead of an Expert Advisor is developed in our case, the background with OBJ\_EDIT object will be made as an example. The result is shown on the figure below:

![Fig. 4. Information panel](https://c.mql5.com/2/4/9wa9w6z_um6_1u5khu202161j6_crtbim.png)

Fig. 4. Information panel

Let's describe all the data shown on the panel:

- **Symbol** (current/total) – symbol, the data on which is downloaded/copied/written at the moment. The left number in brackets shows the current symbol number. The right number shows common number of symbols the script will work with.
- **Path Symbol** – symbol path or category it belongs to. If you bring up the context menu by right-clicking in Market Watch window and select "Symbols…", the window with the list of all symbols will appear. You can find more about that in the terminal's User Guide.
- **Timeframe** – period (timeframe). The timeframe, at which the script will be launched, is to be used.
- **Input Start Date** – data start date specified by a user in the script parameters.
- **First Date** **(H1)** – the first available data date (bar) of the current timeframe.
- **First Terminal Date (M1)** – the first available date of M1 timeframe in already existing terminal data.
- **First Server Date (M1)** – the first available date of M1 timeframe on the server.
- **Max. Bars In Options Terminal** – maximum number of bars to be displayed on the chart specified in the terminal settings.
- **Copied Bars** – number of copied bars for writing.
- **Progress Value Current Symbol** – percentage value of written data of the current symbol.

Below is the code of such information panel:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| INFORMATION_PANEL                                                |
//|------------------------------------------------------------------+
void InfoTable(int s)
  {
   int fnt_sz=8;            // Font size
   string fnt="Calibri";     // Header font
   color clr=clrWhiteSmoke;  // Color
//---
   int xH=300;
   int height_pnl=0;
   int yV1=1,yV2=12,xV1=165,xV2=335,xV3=1;
//---
   string sf="",stf="",ssf="";
   bool flg_sf=false,flg_stf=false,flg_ssf=false;
//---
   if(show_progress) { height_pnl=138; } else { height_pnl=126; }
//---
   flg_sf=SeriesInfoInteger(symbols[s],_Period,SERIES_FIRSTDATE,first_date);
   flg_stf=SeriesInfoInteger(symbols[s],PERIOD_M1,SERIES_TERMINAL_FIRSTDATE,first_termnl_date);
   flg_ssf=SeriesInfoInteger(symbols[s],PERIOD_M1,SERIES_SERVER_FIRSTDATE,first_server_date);
//---
   if(flg_sf) { sf=TSdm(first_date); } else { sf="?"; }
   if(flg_stf) { stf=TSdm(first_termnl_date); } else { stf="?"; }
   if(flg_ssf) { ssf=TSdm(first_server_date); } else { ssf="?"; }
//---
   if(cnt_symb==0) { cnt_symb=1; }
//---
   int anchor1=ANCHOR_LEFT_UPPER,anchor2=ANCHOR_RIGHT_UPPER,corner=CORNER_LEFT_UPPER;
//---
   string path_symbol=SymbolInfoString(symbols[s],SYMBOL_PATH);
   path_symbol=StringSubstr(path_symbol,0,StringLen(path_symbol)-StringLen(symbols[s]));
//---
   arr_txtobj[0]="INFO TABLE";
   arr_txtobj[1]="Symbol (current / total) : ";
   arr_txtobj[2]=""+symbols[s]+" ("+IS(s+1)+"/"+IS(cnt_symb)+")";
   arr_txtobj[3]="Path Symbol : ";
   arr_txtobj[4]=path_symbol;
   arr_txtobj[5]="Timeframe : ";
   arr_txtobj[6]=gStrTF(_Period);
   arr_txtobj[7]="Input Start Date : ";
   arr_txtobj[8]=TSdm(start_date);
   arr_txtobj[9]="First Date (H1) : ";
   arr_txtobj[10]=sf;
   arr_txtobj[11]="First Terminal Date (M1) : ";
   arr_txtobj[12]=stf;
   arr_txtobj[13]="First Server Date (M1) : ";
   arr_txtobj[14]=ssf;
   arr_txtobj[15]="Max. Bars In Options Terminal : ";
   arr_txtobj[16]=IS(max_bars);
   arr_txtobj[17]="Copied Bars : ";
   arr_txtobj[18]=IS(copied_bars);
   arr_txtobj[19]="Progress Value Current Symbol : ";
   arr_txtobj[20]=DS(pgs_pcnt,2)+"%";
//---
   Create_Edit(0,0,arr_nmobj[0],"",corner,fnt,fnt_sz,clrDimGray,clrDimGray,345,height_pnl,xV3,yV1,2,C'15,15,15');
//---
   Create_Edit(0,0,arr_nmobj[1],arr_txtobj[0],corner,fnt,8,clrWhite,C'64,0,0',345,12,xV3,yV1,2,clrFireBrick);
//---
   Create_Label(0,arr_nmobj[2],arr_txtobj[1],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2,0);
   Create_Label(0,arr_nmobj[3],arr_txtobj[2],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2,0);
//---
   Create_Label(0,arr_nmobj[4],arr_txtobj[3],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*2,0);
   Create_Label(0,arr_nmobj[5],arr_txtobj[4],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*2,0);
//---
   Create_Label(0,arr_nmobj[6],arr_txtobj[5],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*3,0);
   Create_Label(0,arr_nmobj[7],arr_txtobj[6],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*3,0);
//---
   Create_Label(0,arr_nmobj[8],arr_txtobj[7],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*4,0);
   Create_Label(0,arr_nmobj[9],arr_txtobj[8],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*4,0);
//---
   Create_Label(0,arr_nmobj[10],arr_txtobj[9],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*5,0);
   Create_Label(0,arr_nmobj[11],arr_txtobj[10],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*5,0);
//---
   Create_Label(0,arr_nmobj[12],arr_txtobj[11],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*6,0);
   Create_Label(0,arr_nmobj[13],arr_txtobj[12],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*6,0);
//---
   Create_Label(0,arr_nmobj[14],arr_txtobj[13],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*7,0);
   Create_Label(0,arr_nmobj[15],arr_txtobj[14],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*7,0);
//---
   Create_Label(0,arr_nmobj[16],arr_txtobj[15],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*8,0);
   Create_Label(0,arr_nmobj[17],arr_txtobj[16],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*8,0);
//---
   Create_Label(0,arr_nmobj[18],arr_txtobj[17],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*9,0);
   Create_Label(0,arr_nmobj[19],arr_txtobj[18],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*9,0);
//---
   if(show_progress)
     {
      Create_Label(0,arr_nmobj[20],arr_txtobj[19],anchor2,corner,fnt,fnt_sz,clr,xV1,yV1+yV2*10,0);
      Create_Label(0,arr_nmobj[21],arr_txtobj[20],anchor2,corner,fnt,fnt_sz,clr,xV2,yV1+yV2*10,0);
     }
  }
//____________________________________________________________________
//+------------------------------------------------------------------+
//| CREATING_LABEL_OBJECT                                            |
//+------------------------------------------------------------------+
void Create_Label(long   chrt_id,   // chart id
                  string lable_nm,  // object name
                  string rename,    // displayed name
                  long   anchor,    // anchor point
                  long   corner,    // attachment corner
                  string font_bsc,  // font
                  int    font_size, // font size
                  color  font_clr,  // font color
                  int    x_dist,    // X scale coordinate
                  int    y_dist,    // Y scale coordinate
                  long   zorder)    // priority
  {
   if(ObjectCreate(chrt_id,lable_nm,OBJ_LABEL,0,0,0)) // creating object
     {
      ObjectSetString(chrt_id,lable_nm,OBJPROP_TEXT,rename);          // set name
      ObjectSetString(chrt_id,lable_nm,OBJPROP_FONT,font_bsc);        // set font
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_COLOR,font_clr);      // set font color
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_ANCHOR,anchor);       // set anchor point
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_CORNER,corner);       // set attachment corner
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_FONTSIZE,font_size);  // set font size
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_XDISTANCE,x_dist);    // set X coordinates
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_YDISTANCE,y_dist);    // set Y coordinates
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_SELECTABLE,false);     // unable to highlight the object, if FALSE
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_ZORDER,zorder);       // Higher/lower priority
      ObjectSetString(chrt_id,lable_nm,OBJPROP_TOOLTIP,"\n");         // no tooltip, if "\n"
     }
  }
//____________________________________________________________________
//+------------------------------------------------------------------+
//| CREATING_EDIT_OBJECT                                             |
//+------------------------------------------------------------------+
void Create_Edit(long   chrt_id,       // chart id
                 int    nmb_win,       // window (subwindow) index
                 string lable_nm,      // object name
                 string text,          // displayed text
                 long   corner,        // attachment corner
                 string font_bsc,      // font
                 int    font_size,     // font size
                 color  font_clr,      // font color
                 color  font_clr_brd,  // font color
                 int    xsize,         // width
                 int    ysize,         // height
                 int    x_dist,        // X scale coordinate
                 int    y_dist,        // Y scale coordinate
                 long   zorder,        // priority
                 color  clr)           // background color
  {
   if(ObjectCreate(chrt_id,lable_nm,OBJ_EDIT,nmb_win,0,0)) // creating object
     {
      ObjectSetString(chrt_id,lable_nm,OBJPROP_TEXT,text);                     // set name
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_CORNER,corner);                // set attachment corner
      ObjectSetString(chrt_id,lable_nm,OBJPROP_FONT,font_bsc);                 // set font
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_ALIGN,ALIGN_CENTER);         // center alignment
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_FONTSIZE,font_size);           // set font size
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_COLOR,font_clr);               // font color
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_BORDER_COLOR,font_clr_brd);    // background color
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_BGCOLOR,clr);                  // background color
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_XSIZE,xsize);                  // width
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_YSIZE,ysize);                  // height
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_XDISTANCE,x_dist);             // set X coordinate
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_YDISTANCE,y_dist);             // set Y coordinate
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_SELECTABLE,false);              // unable to highlight the object, if FALSE
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_ZORDER,zorder);                // Higher/lower priority
      ObjectSetInteger(chrt_id,lable_nm,OBJPROP_READONLY,true);                // Read only
      ObjectSetString(chrt_id,lable_nm,OBJPROP_TOOLTIP,"\n");                  // no tooltip if "\n"
     }
  }
```

After the script operation is complete or the script is deleted by a user ahead of time, all graphic objects created by the script should be deleted. The following functions will be used for that:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| DELETE_ALL_GRAPHICAL_OBJECTS_CREATED_BY_THE_SCRIPT               |
//+------------------------------------------------------------------+
void DelAllScriptObjects()
  {
// Receive the size of graphical object names array
   int sz_arr1=ArraySize(arr_nmobj);
//---
// Delete all objects
   for(int i=0; i<sz_arr1; i++)
     { DelObjbyName(arr_nmobj[i]);  }
  }
//____________________________________________________________________
//+------------------------------------------------------------------+
//| DELETE_OBJECTS_BY_NAME                                           |
//+------------------------------------------------------------------+
int DelObjbyName(string Name)
  {
   int nm_obj=0;
   bool res=false;
//---
   nm_obj=ObjectFind(ChartID(),Name);
//---
   if(nm_obj>=0)
     {
      res=ObjectDelete(ChartID(),Name);
      //---
      if(!res) { Print("Object deletion error: - "+ErrorDesc(Error())+""); return(false); }
     }
//---
   return(res);
  }
```

### 7\. Application's Main Block

The scripts' main function is [OnStart()](https://www.mql5.com/en/docs/basis/function/events#onstart). This is the function used for calling all other functions for execution. The details of the program's operation are shown in the code:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| SCRIPT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> |
//+------------------------------------------------------------------+
void OnStart()
  {
// If user-defined parameters are incorrect,
// error message is shown and the program is closed
   if(ValidationParameters()) { return; }
//---
   max_bars=TerminalInfoInteger(TERMINAL_MAXBARS); // Receive available number of bars in the window
//---
   GetSymbolsToArray();           // Filling symbol array with names
   sz_arr_symb=ArraySize(symbols); // Receive symbol array size
//---
   SetSeparateForFormatDate();    // Set a separator for date format
//---
// Revise all symbols and write their data to file
   for(int s=0; s<=sz_arr_symb-1; s++)
     {
      copied_bars=0; // Reset copied bars variable to zero for writing
      pgs_pcnt=0.0;  // Reset variable of the symbol data writing progress
      //---
      InfoTable(s); ChartRedraw();
      //---
      // Receive current symbol data
      int res=GetDataCurrentSymbol(s);
      //---
      if(res==0) { BC } // If zero, break the loop or start the next iteration
      //---
      if(res==2)        // Program operation interrupted by user
        {
         DelAllScriptObjects(); // Deleted objects created by the script from the chart
         //---
         Print("------\nUser deleted the script!"); break;
        }
      //---
      // Receive the path for creating the file and create directories for them
      // If the string is empty, break the loop or start the next iteration
      if((path=CheckCreateGetPath(s))=="") { BC }
      //---
      WriteDataToFile(s); // Write data to file
     }
//---
// Delete symbols from Market Watch window if necessary
   DelSymbolsFromMarketWatch();
//---
// Delete objects created by the script from the chart
   Sleep(1000); DelAllScriptObjects();
  }
```

Let's examine the functions where the key activities take place.

The symbol array (symbols\[\]) is filled with symbol names in GetSymbolsToArray() function. The array size, as well as the number of symbols in it depend on the variant chosen by a user in Mode Write Symbols (curr\_mwatch) parameter.

If a user should have data only from one symbol, the array size is equal to 1.

```
ArrayResize(symbols,1); // Set the array size to be equal to 1
symbols[0]=_Symbol;     // Specify the current symbol's name
```

If a user wants to receive the data on all symbols from Market Watch window or all available symbols, the array size will be defined by the following function:

```
int SymbolsTotal(
   bool selected   // true – only MarketWatch symbols
);
```

To avoid creation of two blocks for two variants with almost identical code, we will make MWatchOrAllList() pointer function, which will return true or false. This value defines where the symbol list should be taken from - only from Market Watch window (true) or from the common list of available symbols (false).

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| POINTER_TO_MARKET_WATCH_WINDOW_OR_TO_COMMON_LIST                 |
//+------------------------------------------------------------------+
bool MWatchOrAllList()
  {
   if(curr_mwatch==MARKETWATCH) { return(true); }
   if(curr_mwatch==ALL_LIST_SYMBOLS) { return(false); }
//---
   return(true);
  }
```

After obtaining the number of symbols in the loop, we should go through the entire list and put the symbol name into the array at each iteration increasing the array size by one. The symbol name, in its turn, is obtained by the index number using [SymbolName()](https://www.mql5.com/en/docs/marketinformation/symbolname) function.

```
int SymbolName(
   int pos,        // list index number
   bool selected   // true – only MarketWatch symbols
);
```

MWatchOrAllList() pointer function is also used in SymbolName() function to select the symbol list. GetSymbolsToArray() function complete code:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| FILLING_SYMBOL_ARRAY_WITH_NAMES                                  |
//+------------------------------------------------------------------+
void GetSymbolsToArray()
  {
// If only the current symbol data is required
   if(curr_mwatch==CURRENT)
     { ArrayResize(symbols,1); symbols[0]=_Symbol; }
//---
// If data on all symbols  from Market Watch window or
// or the entire symbol list is required
   if(curr_mwatch==MARKETWATCH || curr_mwatch==ALL_LIST_SYMBOLS)
     {
      // Receive the number of symbols in Market Watch window
      cnt_symb=SymbolsTotal(MWatchOrAllList());
      //---
      for(int i=0; i<=cnt_symb-1; i++)
        {
         string nm_symb="";
         //---
         ArrayResize(symbols,i+1); // Increase the array size by one once again
         //---
         // Receive a name of a symbol from Market Watch window
         nm_symb=SymbolName(i,MWatchOrAllList());
         symbols[i]=nm_symb; // Put the symbol name into the array
        }
     }
  }
```

SetSeparateForFormatDate() function is very simple. it is used to define what kind of separator will be used in the date depending on the user's choice in the drop-down list of Format Date (format\_date) parameter.

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| DEFINING_SEPARATOR_FOR_DATE_FORMAT                               |
//+------------------------------------------------------------------+
void SetSeparateForFormatDate()
  {
   switch(format_date)
     {
      case SEP_POINT1 : case SEP_POINT2 : sep="."; break; // Full point as a separator
      case SEP_SLASH1 : case SEP_SLASH2 : sep="/"; break; // Slash as a separator
     }
  }
```

The basic loop with various checks comes next. If all the checks are successful, the data is written to the file. Otherwise, the loop is broken, all objects are deleted from the chart and the script is removed (in case of one symbol) or the next iteration starts (in case of more than one symbol). Each symbol of symbols\[\] array is consistently called in the loop. The index number is sent to each function of the loop. Thus, the accurate sequence in all functions is preserved.

The data on the current symbol in the loop is received at each iteration at the very beginning of the loop body. GetDataCurrentSymbol() function is used for that. Let's see what happens in this function.

Data availability is checked using CheckLoadHistory() function before copying symbol data to rate\[\] array. This function is provided by the developers as an example. Its initial version can be found in [MQL5 Reference](https://www.mql5.com/en/docs/series/timeseries_access). I have only made slight corrections for using in this script. The Reference contains a detailed description (it would be a good idea to study it as well), therefore, I am not going to display my version here as it is almost the same. Besides, it can be found in the code with detailed comments.

The only thing that can be mentioned now is that CheckLoadHistory() function returns the error or successful execution code, according to which the appropriate message from [switch](https://www.mql5.com/en/docs/basis/operators/switch) operator block is saved in the journal. According to the received code, GetDataCurrentSymbol() function either goes on with its operation or returns its code.

If all goes well, history data is copied using [CopyRates()](https://www.mql5.com/en/docs/series/copyrates) function. The array size is saved in the global variable. Then, the exit from the function accompanied by returning of code 1 is performed. If something goes wrong, the function stops its operation in switch operator and returns code 0 or 2.

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| RECEIVE_SYMBOL_DATA                                              |
//+------------------------------------------------------------------+
int GetDataCurrentSymbol(int s)
  {
   Print("------\n№"+IS(s+1)+" >>>"); // Save a symbol number in the journal
//---
// Check and download the necessary amount of requested data
   int res=CheckLoadHistory(s,_Period);
//---
   InfoTable(s); ChartRedraw(); // Update the data in the data table
//---
   switch(res)
     {
      case -1 : Print("Unknown symbol "+symbols[s]+" (code: -1)!");                        return(0);
      case -2 :
         Print("Number of requested bars exceeds the maximum number that can be displayed on a chart (code: -2)!...\n"
               "...The available amount of data will be used for writing.");                break;
      //---
      case -3 : Print("Execution interrupted by user (code: -3)!");                         return(2);
      case -4 : Print("Download failed (code: -4)!");                                      return(0);
      case  0 : Print("All symbol data downloaded (code: 0).");                             break;
      case  1 : Print("Time series data is sufficient (code: 1).");                          break;
      case  2 : Print("Time series created based on existing terminal data (code: 2).");      break;
      //---
      default : Print("Execution result is not defined!");
     }
//---
// Copy data to the array
   if(CopyRates(symbols[s],_Period,check_start_date,end_date,rates)<=0)
     { Print("Error when copying symbol data "+symbols[s]+" - ",ErrorDesc(Error())+""); return(0); }
   else
     {
      copied_bars=ArraySize(rates); // Receive array size
      //---
      Print("Symbol: ",symbols[s],"; Timeframe: ",gStrTF(_Period),"; Copied bars: ",copied_bars);
     }
//---
   return(1); // Return 1, if all is well
  }
```

After that, the program is located in the main loop's body of [OnStart()](https://www.mql5.com/en/docs/basis/function/events#onstart) function again. The code is assigned by _res_ local variable and the check is performed according to its value. Zero value stands for an error. It means that the data of the current symbol in the loop cannot be written. Error explanation has been saved in the journal and the decision is made if the loop should be broken (break) or the next iteration should be started (continue).

```
if(res==0) { BC } // If zero, the loop is interrupted or the next iteration starts
```

The code line above shows that this selection is performed by some mysterious BC characters. This is a macro expansion. More information about it can be found in [MQL5 Reference](https://www.mql5.com/en/docs/basis/preprosessor/constant). The only thing that should be mentioned here is that the entire expressions (typed in one line) can be pasted in a short entry as shown in the above example (BC). In some cases, this method can be even more convenient and compact than a function. In the current case, this looks as follows:

```
// Macro expansion with further action selection
#define BC if(curr_mwatch==CURRENT) { break; } if(curr_mwatch==MARKETWATCH || curr_mwatch==ALL_LIST_SYMBOLS) { continue; }
```

Below are other examples of macro expansions used in this script:

```
#define nmf __FUNCTION__+": " // Macro expansion of the function name before sending the message to the journal
//---
#define TRM_DP TerminalInfoString(TERMINAL_DATA_PATH) // Folder for storing the terminal data
```

If GetDataCurrentSymbol() returns 2, the program has been deleted by a user. MQL5 has [IsStopped()](https://www.mql5.com/en/docs/check/isstopped) function to identify this event. This function can be very helpful in loops to stop the program's operation correctly and in time. If the function returns true, then there are about three seconds to perform all actions before the program is forcibly deleted. In our case, all graphical objects are removed and the message is sent to the journal:

```
if(res==2) // Program execution interrupted by user
   {
    DelAllScriptObjects(); // Delete all objects created by the script from the chart
    //---
    Print("------\nUser deleted the script!"); break;
   }
```

### 8\. Creating Folders and Filing the Data

CheckCreateGetPath() function checks the presence of the root data folder. Let's call it DATA\_OHLC and place it to C:\\Metatrader 5\\MQL5\\Files. It will contain folders with the symbol names. The files for writing the data will be created there.

If the root folder or the folder for the current symbol in the loop does not exist, the function creates it. If all goes well, the function returns a string containing the path for creating a file. The function returns an empty string in case of an error or an attempt to delete the program from the chart performed by a user.

The code below contains detailed comments making it easy to understand:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| CHECK_DIRECTORY_AND_CREATE_NECESSARY_DATA_FOLDERS                |
//+------------------------------------------------------------------+
string CheckCreateGetPath(int s)
  {
   int i=1;
   long search=-1;
   string ffname="",lpath="";
   string file="*.csv",folder="*";
   string
   root="DATA_OHLC\\",         // Root data folder
   fSmb=symbols[s]+"\\",     // Symbol name
   fTF=gStrTF(_Period)+"\\"; // Symbol time frame
//---
   bool flgROOT=false,flgSYMBOL=false;
//---
//+------------------------------------------------------------------+
//| SEARCHING_FOR_DATA_OHLC_ROOT_FOLDER                              |
//+------------------------------------------------------------------+
   lpath=folder;
   search=FileFindFirst(lpath,ffname); // Set search handle in Metatrader 5\MQL5\Files
//---
   Print("Directory: ",TRM_DP+"\\MQL5\\Files\\");
//---
// Set the flag if the first folder is a root one
   if(ffname==root)
     { flgROOT=true; Print("Root folder "+root+" present"); }
//---
   if(search!=INVALID_HANDLE) // If search handle received
     {
      if(!flgROOT) // If the first folder is not a root one
        {
         // Sort out all files searching for the root folder
         while(FileFindNext(search,ffname))
           {
            if(IsStopped()) // Execution interrupted by user
              {
               // Delete objects created by the script from the chart
               DelAllScriptObjects();
               //---
               Print("------\nUser deleted the script!"); return("");
              }
            //---
            if(ffname==root) // Set the flag if found
              { flgROOT=true; Print("Root folder "+root+" present"); break; }
           }
        }
      //---
      FileFindClose(search); search=-1; // Close root folder search handle
     }
   else { Print("Error when receiving the search handle or directory "+TRM_DP+" is empty: ",ErrorDesc(Error())); }
//---
//+------------------------------------------------------------------+
//| SEARCHING_SYMBOL_FOLDER                                          |
//+------------------------------------------------------------------+
   lpath=root+folder;
//---
// Set search handle in the root folder ..\Files\DATA OHLC\
   search=FileFindFirst(lpath,ffname);
//---
// Set the flag if the first folder of the current symbol
   if(ffname==fSmb) { flgSYMBOL=true; Print("Symbol folder "+fSmb+" present"); }
//---
   if(search!=INVALID_HANDLE) // If search handle is received
     {
      if(!flgSYMBOL) // If the first folder is not of the current symbol
        {
         // Sort out all the files in the root folder searching the symbol folder
         while(FileFindNext(search,ffname))
           {
            if(IsStopped()) // Execution interrupted by user
              {
               // Delete objects created by the script from the chart
               DelAllScriptObjects();
               //---
               Print("------\nUser deleted the script!"); return("");
              }
            //---
            if(ffname==fSmb) // Set the flag if found
              { flgSYMBOL=true; Print("Symbol folder"+fSmb+" present"); break; }
           }
        }
      //---
      FileFindClose(search); search=-1; // Close symbol folder search handle
     }
   else { Print("Error when receiving search handle or the directory "+path+" is empty"); }
//---
//+------------------------------------------------------------------+
//| CREATE_NECESSARY_DIRECTORIES_ACCORDING_TO_CHECK_RESULTS          |
//+------------------------------------------------------------------+
   if(!flgROOT) // If there is no DATA_OHLC... root folder
     {
      if(FolderCreate("DATA_OHLC")) // ...we should create it
        { Print("..\DATA_OHLC\\ root folder created"); }
      else
        { Print("Error when creating DATA_OHLC: root folder",ErrorDesc(Error())); return(""); }
     }
//---
   if(!flgSYMBOL) // If there is no folder of the symbol, the values of which should be received...
     {
      if(FolderCreate(root+symbols[s])) // ...we should create it
        {
         Print("..\DATA_OHLC\\" symbol folder created+fSmb+"");
         //---
         return(root+symbols[s]+"\\"); // Return the path for creating the file for writing
        }
      else
        { Print("Error when creating ..\DATA_OHLC\\ symbol folder"+fSmb+"\: ",ErrorDesc(Error())); return(""); }
     }
//---
   if(flgROOT && flgSYMBOL)
     {
      return(root+symbols[s]+"\\"); // Return the path for creating the file for writing
     }
//---
   return("");
  }
```

If CheckCreateGetPath() function returns an empty line, the loop is interrupted or the next iteration starts using the already familiar macro expansion (BC):

```
// Receive the path for creating a file and create directories for them
// If the line is empty, the loop is interrupted or the next iteration starts
if((path=CheckCreateGetPath(s))=="") { BC }
```

If you have reached this stage, it means that the data has been successfully copied and path string variable contains the path for creating the file for writing the data of the current symbol in the loop.

Create WriteDataToFile() function to write data to the file. \[Path\]+\[file name\] is generated at the beginning of the function. A file name consists of a symbol name and the current time frame. For example, EURUSD\_H1.csv. If the file with such a name is already present, it is just opened for writing. Previously written data will be deleted. New data will be written in it instead. If the file is created/opened successfully, [FileOpen()](https://www.mql5.com/en/docs/files/fileopen) function returns the handle that will be used for accessing the file.

Checking for the handle. If it is present, the header line is written. The appropriate line will be written depending on what headers have been selected by a user. The main loop of writing the history data begins afterwards.

Before writing the next line, it should be converted to the format specified by the user. To do that, we should receive the bar open time and sort out day, month, year and time separately by variables using [StringSubstr()](https://www.mql5.com/en/docs/strings/stringsubstr) function. Then, we should define if the date and time will be located in a single or separate columns depending on the format specified by the user. Then all parts are joined together into one line using [StringConcatenate()](https://www.mql5.com/en/docs/strings/stringconcatenate) function. After all the lines are written, the file is closed by [FileClose()](https://www.mql5.com/en/docs/files/fileclose) function.

The entire WriteDataToFile() function code is shown below:

```
//____________________________________________________________________
//+------------------------------------------------------------------+
//| WRITE_DATA_TO_FILE                                               |
//+------------------------------------------------------------------+
void WriteDataToFile(int s)
  {
// Number of decimal places in the symbol price
   int dgt=(int)SymbolInfoInteger(symbols[s],SYMBOL_DIGITS);
//---
   string nm_fl=path+symbols[s]+"_"+gStrTF(_Period)+".csv"; // File name
//---
// Receive file handle for writing
   hFl=FileOpen(nm_fl,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
//---
   if(hFl>0) // If the handle is received
     {
      // Write the headers
      if(format_headers==NSDT_5)
        { FileWrite(hFl,"\"Date\" ""\"Time\" ""\"Open\" ""\"High\" ""\"Low\" ""\"Close\" ""\"Volume\""); }
      //---
      if(format_headers==NSDT_6)
        { FileWrite(hFl,"Date","Open","High","Low","Close","Volume"); }
      //---
      // Write the data
      for(int i=0; i<=copied_bars-1; i++)
        {
         if(IsStopped()) // If program execution interrupted by a user
           {
            DelAllScriptObjects(); // Delete objects created by the script from the chart
            //---
            Print("------\nUser deleted the script!"); break;
           }
         //---
         sdt=TSdm(rates[i].time); // Bar open time
         //---
         // Divide the date by year, month and time
         yyyy=StringSubstr(sdt,0,4);
         mm=StringSubstr(sdt,5,2);
         dd=StringSubstr(sdt,8,2);
         tm=StringSubstr(sdt,11);
         //---
         string sep_dt_tm=""; // Separator of Date and Time columns
         //---
         // Join the data with the separator in the necessary order
         if(format_date==SEP_POINT1 || format_date==SEP_SLASH1) { sep_dt_tm=" "; }
         if(format_date==SEP_POINT2 || format_date==SEP_SLASH2) { sep_dt_tm=","; }
         //---
         // Join everything in one line
         StringConcatenate(sdt,dd,sep,mm,sep,yyyy,sep_dt_tm,tm);
         //---
         FileWrite(hFl,
                   sdt,// Date-time
                   DS_dgt(rates[i].open,dgt),      // Open price
                   DS_dgt(rates[i].high,dgt),      // High price
                   DS_dgt(rates[i].low,dgt),       // Low price
                   DS_dgt(rates[i].close,dgt),     // Close price
                   IS((int)rates[i].tick_volume)); // Tick volume price
         //---
         // Update writing progress value for the current symbol
         pgs_pcnt=((double)(i+1)/copied_bars)*100;
         //---
         // Update data in the table
         InfoTable(s); if(show_progress) { ChartRedraw(); }
        }
      //---
      FileClose(hFl); // Close the file
     }
   else { Print("Error when creating/opening file!"); }
  }
```

This was the last function in OnStart() function's basic loop. If it was not the last symbol, then everything is repeated for the next one. Otherwise, the loop is broken. If the user specified to clear the list of symbols in Market Watch window in the script parameters, the symbols with currently non-active charts will be deleted by DelSymbolsFromMarketWatch() function. After that, all graphical objects created by the script are deleted and the program stops. The data is ready for use.

The details on how to download data to NeuroShell DayTrader Professional can be found in my [blog](https://www.mql5.com/go?link=http://tol64.blogspot.com/2012/09/zagruzka-dannyh-v-nsdt.html "http://tol64.blogspot.com/2012/09/zagruzka-dannyh-v-nsdt.html"). Below is the video showing the script operation:

Скрипт для записи котировок в файл из Metatrader 5 - YouTube

[Photo image of Anatoli Kazharski](https://www.youtube.com/channel/UC6taRQOV118GaAi8A8_f3_g?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F502)

Anatoli Kazharski

54 subscribers

[Скрипт для записи котировок в файл из Metatrader 5](https://www.youtube.com/watch?v=pxZI81lEAsA)

Anatoli Kazharski

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

[Watch on](https://www.youtube.com/watch?v=pxZI81lEAsA&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F502)

0:00

0:00 / 0:52

•Live

•

### Conclusion

Whatever program I used for developing trading strategies, I always reached some limitations preventing me from further development of my ideas. Eventually, I realized that programming is essential here. MQL5 is the best solution for those who really want to succeed. However, other programs for data analysis and development of trading strategies can also be useful when searching for new ideas. It would have taken me much longer to find them, if I had used only one tool.

Good luck!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/502](https://www.mql5.com/ru/articles/502)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/502.zip "Download all attachments in the single ZIP archive")

[writedatatofile.mq5](https://www.mql5.com/en/articles/download/502/writedatatofile.mq5 "Download writedatatofile.mq5")(68.96 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/9815)**
(6)


![Anatoli Kazharski](https://c.mql5.com/avatar/2022/1/61D72F6B-7C12.jpg)

**[Anatoli Kazharski](https://www.mql5.com/en/users/tol64)**
\|
5 Oct 2012 at 07:40

**komposter:**

Thanks for the article, it was a pleasure to read it.

Question about the script functionality: is it impossible to get a history longer than "Max bars in the window" programmatically?

And a wish for future versions: add other data formats (for different programmes). And put the code in the database and update it as you improve it.

Thanks again!

Thank you. ))

I think it is possible to get the history more than set in the limitation. It's me who has already played with it. In the original version from the help this check was there, because the example was given for displaying data on the chart. But for this script it is still unnecessary. But I'd better check it again and write about it later (I've switched to another task). It is probably not worth changing the article. I'll put it in the database later, and it will be an incentive to add some more data formats. ))

![Serhiy Dotsenko](https://c.mql5.com/avatar/avatar_na2.png)

**[Serhiy Dotsenko](https://www.mql5.com/en/users/thejobber)**
\|
17 Dec 2014 at 22:49

How about the reverse task, from finam, for example, to insert quotes (for lukoil, for example) into MT?


![Anatoli Kazharski](https://c.mql5.com/avatar/2022/1/61D72F6B-7C12.jpg)

**[Anatoli Kazharski](https://www.mql5.com/en/users/tol64)**
\|
18 Dec 2014 at 10:02

**thejobber:**

What about the reverse task, for example, to insert quotes from Finam (for Lukoil, for example) into MT?

In MetaTrader 5? You can, if only as an indicator. It is easier to open an account with a broker, where the required tool is available.

If the broker does not provide this platform, you should ask "when will you finally give us the opportunity to trade via MetaTrader 5". ))

![Serhiy Dotsenko](https://c.mql5.com/avatar/avatar_na2.png)

**[Serhiy Dotsenko](https://www.mql5.com/en/users/thejobber)**
\|
18 Dec 2014 at 11:44

**tol64:**

In MetaTrader 5? It is possible, if only as an indicator. It's easier to open an account with a broker, where the necessary tool is available.

not an option, I need it to bring everything to one platform, so that it would not be necessary to test stocks in Tradmatic or Welslab, forex in MT.

and so there is one normal (understandable and most importantly free, quality product with normal classical language), where you could put any quotes and test them..... such logic ))

and there is no need to splurge on QPILE, Lua (Quik), C# [(ctrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade "Standard library: CTrade class"), Tradematic, Wealth-Lab), S# (many things), java (jforex) etc...

![Mikhail Khlestov](https://c.mql5.com/avatar/avatar_na2.png)

**[Mikhail Khlestov](https://www.mql5.com/en/users/exploits)**
\|
22 Mar 2019 at 09:14

Required view <DATE>,<TIME>,<BID>,<OFFER>

20170102,0,1.07139,1.07149

20170102,500,1.07139,1.07159

20170102,1000,1.07139,1.07169

20170102,1500,1.07174,1.07194

20170102,2000,1.07197,1.07217

20170102,2500,1.07174,1.07194

Can you help to implement such a thing?

![Interview with Alexey Masterov (ATC 2012)](https://c.mql5.com/2/0/avatar_reinhardf17.png)[Interview with Alexey Masterov (ATC 2012)](https://www.mql5.com/en/articles/624)

We do our best to introduce all the leading Championship Participants to our audience in reasonable time. To achieve that, we closely monitor the most promising contestants in our TOP-10 and arrange interviews with them. However, the sharp rise of Alexey Masterov (reinhard) up to the third place has become a real surprise!

![Interview with Sergey Pankratyev (ATC 2012)](https://c.mql5.com/2/0/s75-avatar31t.png)[Interview with Sergey Pankratyev (ATC 2012)](https://www.mql5.com/en/articles/623)

The Championship is coming to an end leaving us with vivid impressions of many unusual trading strategies. However, the trading robot of Sergey Pankratyev (s75) is showing really peculiar things - it is trading all 12 currency pairs opening only long positions. It is not an error but just a response to some certain market conditions.

![General information on Trading Signals for MetaTrader 4 and MetaTrader 5](https://c.mql5.com/2/0/signal_mt4_mt5__1.png)[General information on Trading Signals for MetaTrader 4 and MetaTrader 5](https://www.mql5.com/en/articles/618)

MetaTrader 4 / MetaTrader 5 Trading Signals is a service allowing traders to copy trading operations of a Signals Provider. Our goal was to develop the new massively used service protecting Subscribers and relieving them of unnecessary costs.

![Interview with Enbo Lu (ATC 2012)](https://c.mql5.com/2/0/luenbo_avatar614.png)[Interview with Enbo Lu (ATC 2012)](https://www.mql5.com/en/articles/622)

"Be sure to participate in the Automated Trading Championships, where you can get a truly invaluable experience!" - this is the motto of contestant Enbo Lu (luenbo) from China. He appeared in the TOP-10 of Automated Trading Championship 2012 last week and is now consistently trying to reach the podium.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ldxbybctvbdohlvfpjswszwtkicfqclb&ssn=1769181696514947131&ssn_dr=0&ssn_sr=0&fv_date=1769181696&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F502&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Prepare%20MetaTrader%205%20Quotes%20for%20Other%20Applications%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918169612055763&fz_uniq=5069373187130590207&sv=2552)

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