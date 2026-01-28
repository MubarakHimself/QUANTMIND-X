---
title: Creating and Publishing of Trade Reports and SMS Notification
url: https://www.mql5.com/en/articles/61
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:21:47.628736
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/61&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071776393131404857)

MetaTrader 5 / Examples


### Introduction

This article describes how to generate a report of trade results (using Expert Advisor, Indicator or Script) as HTML-file and upload it via FTP to WWW-server. We will also consider sending notification of trade events as SMS to mobile phone.

To be more comfortable with material described in this article, the reader is advised to be familiar with the HTML (HyperText Markup Language).

To implement the upload reports we need a WWW-server (it can be any computer), that can accept data via FTP. To implement the possibility of receiving notifications about trade events as SMS, we need an EMAIL-SMS gateway (this service is provided by most of mobile operators and third-party organizations).

### 1\. Creating a Report and Sending it via FTP

Let's create a MQL5-program, which generates a trade report and sends it via FTP-protocol. First we make it as a Script. In future we can use it as a finished block, that can be inserted into Expert Advisors and Indicators. For example, in Expert Advisers you can use this block as the [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) or [Timer](https://www.mql5.com/en/docs/runtime/event_fire#timer) event handler, to run this block after trade request, or to set some actions for the [ChartEvent](https://www.mql5.com/en/docs/runtime/event_fire#chartevent) event. In Indicators you can included this block to the [Timer](https://www.mql5.com/en/docs/runtime/event_fire#timer) or [ChartEvent](https://www.mql5.com/en/docs/runtime/event_fire#chartevent) event handlers.

The example of report, created by program, is shown on Figures 1, 2 and 3. Or you can download this report by link located at the end of the article.

![Figure 1. Example of Report - Table of Deals and Positions.](https://c.mql5.com/2/1/picture01m1.gif)

Figure 1. Example of Report - Table of Deals and Positions.

![Figure 2. Example of Report - Balance Chart.](https://c.mql5.com/2/1/picture02m1.gif)

Figure 2. Example of Report - Balance Chart.

![Figure 3. Example of Report - Price Chart on Current Instrument.](https://c.mql5.com/2/1/fig3__2.gif)

Figure 3. Example of Report - Price Chart on Current Instrument.

In table of deals and positions (Figure 1) all deals for convenience are divided into positions. The left side of the table shows the volume, time and price to enter the market (of opening positions and additions). The right side of the table shows the same parameters to exit the market (partial or complete closure of position). On in/out the deal is broken into two parts - the closure of one position and the opening of next.

Under the table of deals and positions shown the balance chart (horizontal axis - time), and at the bottom - the price chart on the current instrument.

The program creates the "report.html", "picture1.gif" and "picture2.gif" files (html-file of report, image files of balance chart and price chart) in the _MetaTarder5\_istall\_dir\\MQL5\\Files_ folder. And FTP publishing is enabled in terminal settings - it sends these three files to specified server. In addition, we will need another two files - images with arrows pointing the direction of open position - Buy or Sell ("buy.gif" and "sell.gif"). You can take these images (download link at the end of the article) or draw them yourself in any graphics editor. These two files should be placed in the same folder of WWW-server with "report.html" file.

As input parameters the program accepts the start and end time of period, for which the report is generated. In our example, the end of the report period is the current time, and user selects the variant of report period: entire period, last day, last week, last month or last year.

A few words about how the report is created. Trade server is requested for all available history of deals. Obtained deals are processed one after another. The deal\_status\[\] array stores information about whether the deal is processed or not. The element indexes of this array are the deals numbers, received from the trade server list of deals. And the elements values are interpreted as follows: 0 - deal has not been yet processed, 1 - deal has been already partially processed (in/out), 127 - deal has been already processed (other values are not used and reserved for future use).

The symb\_list\[\] array contains the list of financial instruments names, by which the trade was conducted, and the lots\_list\[\] array - volumes of open positions for each instrument at the time of deal processing. Positive values of volume correspond to long positions, negative - to short. If volume is equal to zero, it means that this tool has no open positions. If during deals processing a financial instrument, that is not in the list (in the symb\_list\[\] array), is encountered - it is added there, and the number of financial instruments (the symb\_total variable) is incremented by 1.

On each deal processing every subsequent deal is analyzed by the same financial instrument, until position is closed or until in/out. Only those deals are analyzed, for which the value of the deal\_status\[\] array is less than 127. After deal processing, the corresponding element of the deal\_status\[\] array is assigned with value 127, and if the deal is the in/out of position - with value 1. If the time, when the position has been opened, matches the report period (defined by the StartTime and EndTime variables) - this position is logged to report (all inputs and outputs).

In addition to table of deals, a new chart for the current financial instrument is opened. For this chart all the necessary properties are provided, and using the [ChartScreenShot()](https://www.mql5.com/en/docs/chart_operations) function a screenshot is made - so we get an image file with price chart for the current instrument. Next, on this chart the price chart is masked and the chart of balance changes is drawn, and then another screenshot is created.

When two image files with charts and HTML-file with report are created, ability to send files via FTP is checked. If it is allowed - "report.html", "picture1.gif" and "picture2.gif" files are sent using the [SendFTP()](https://www.mql5.com/en/docs/network/sendftp) function, according to settings specified in MetaTrader 5.

Launch the MetaQuotes Language Editor and start creating a script. Define constants - the timeout of chart refresh (in seconds), the width and height of the price chart, and the maximum width of the balance chart. The period of chart, which will display the curve of balance change, is chosen depending on duration of the report period and on maximum width of the chart. The width of chart is adjusted to the size, needed for the balance chart.

The height of chart is automatically calculated as half of the width. Also, we will specify the width of vertical axis as constant - it is the number of pixels by which the graphics area is decreased comparing to the width of picture because of the vertical axis.

```
#define timeout 10           // chart refresh timeout
#define Picture1_width 800   // max width of chart in report
#define Picture2_width 800   // width of price chart in report
#define Picture2_height 600  // height of price chart in report
#define Axis_Width 59        // width of vertical axis (in pixels)
```

Specify that input parameters will be requested from user.

```
// request input parameters
#property script_show_inputs
```

Create enumeration of report periods.

```
// enumeration of report periods
enum report_periods
  {
   All_periods,
   Last_day,
   Last_week,
   Last_month,
   Last_year
  };
```

Ask user for report period (by default it is the entire period).

```
// ask for report period
input report_periods ReportPeriod=0;
```

Write the body of the [OnStart()](https://www.mql5.com/en/docs/basis/function/events#onstart) function.

```
void OnStart()
  {
```

Determine the beginning and the end of report period.

```
  datetime StartTime=0;           // beginning of report period
  datetime EndTime=TimeCurrent(); // end of report period

  // calculating the beginning of report period
  switch(ReportPeriod)
    {
     case 1:
        StartTime=EndTime-86400;    // day
        break;
     case 2:
        StartTime=EndTime-604800;   // week
        break;
     case 3:
        StartTime=EndTime-2592000;  // month
        break;
     case 4:
        StartTime=EndTime-31536000; // year
        break;
    }
  // if none of the options is executed, then StartTime=0 (entire period)
```

Declare variables that will be used in the program. The purpose of variables is described in comments.

```
   int total_deals_number;  // number of deals for history data
   int file_handle;         // file handle
   int i,j;                 // loop counters
   int symb_total;          // number of instruments, that were traded
   int symb_pointer;        // pointer to current instrument
   char deal_status[];      // state of deal (processed/not processed)
   ulong ticket;            // ticket of deal
   long hChart;             // chart id

   double balance;           // current balance value
   double balance_prev;      // previous balance value
   double lot_current;       // volume of current deal
   double lots_list[];       // list of open volumes by instruments
   double current_swap;      // swap of current deal
   double current_profit;    // profit of current deal
   double max_val,min_val;   // maximal and minimal value

   string symb_list[];       // list of instruments, that were traded
   string in_table_volume;   // volume of entering position
   string in_table_time;     // time of entering position
   string in_table_price;    // price of entering position
   string out_table_volume;  // volume of exiting position
   string out_table_time;    // time of exiting position
   string out_table_price;   // price of exiting position
   string out_table_swap;    // swap of exiting position
   string out_table_profit;  // profit of exiting position

   bool symb_flag;           // flag that instrument is in the list

   datetime time_prev;           // previous value of time
   datetime time_curr;           // current value of time
   datetime position_StartTime;  // time of first enter to position
   datetime position_EndTime;    // time of last exit from position

   ENUM_TIMEFRAMES Picture1_period;  // period of balance chart
```

Open a new chart and set its properties - this is the price chart, that will be output at the bottom of report.

```
 // open a new chart and set its properties
hChart=ChartOpen(Symbol(),0);
ChartSetInteger(hChart,CHART_MODE,CHART_BARS);            // bars chart
ChartSetInteger(hChart,CHART_AUTOSCROLL,true);            // autoscroll enabled
ChartSetInteger(hChart,CHART_COLOR_BACKGROUND,White);     // white background
ChartSetInteger(hChart,CHART_COLOR_FOREGROUND,Black);     // axes and labels are black
ChartSetInteger(hChart,CHART_SHOW_OHLC,false);            // OHLC are not shown
ChartSetInteger(hChart,CHART_SHOW_BID_LINE,true);         // show BID line
ChartSetInteger(hChart,CHART_SHOW_ASK_LINE,false);        // hide ASK line
ChartSetInteger(hChart,CHART_SHOW_LAST_LINE,false);       // hide LAST line
ChartSetInteger(hChart,CHART_SHOW_GRID,true);             // show grid
ChartSetInteger(hChart,CHART_SHOW_PERIOD_SEP,true);       // show period separators
ChartSetInteger(hChart,CHART_COLOR_GRID,LightGray);       // grid is light-gray
ChartSetInteger(hChart,CHART_COLOR_CHART_LINE,Black);     // chart lines are black
ChartSetInteger(hChart,CHART_COLOR_CHART_UP,Black);       // up bars are black
ChartSetInteger(hChart,CHART_COLOR_CHART_DOWN,Black);     // down bars are black
ChartSetInteger(hChart,CHART_COLOR_BID,Gray);             // BID line is gray
ChartSetInteger(hChart,CHART_COLOR_VOLUME,Green);         // volumes and orders levels are green
ChartSetInteger(hChart,CHART_COLOR_STOP_LEVEL,Red);       // SL and TP levels are red
ChartSetString(hChart,CHART_COMMENT,ChartSymbol(hChart)); // comment contains instrument <end segm
```

Screen shoot a chart and save it as "picture2.gif".

```
// save chart as image file
ChartScreenShot(hChart,"picture2.gif",Picture2_width,Picture2_height);
```

Request deals history for entire time of account existence.

```
// request deals history for entire period
HistorySelect(0,TimeCurrent());
```

Open the "report.html" file, in which we will write HTML page with report (ANSI encoding).

```
// open report file
file_handle=FileOpen("report.html",FILE_WRITE|FILE_ANSI);
```

Write the beginning part of HTML-document:

- start of html-document (<html>)
- title that will be displayed at the top of your browser window (<head><title>Expert Trade Report</title></head>)
- beginning of the main part of html-document with background color (<body bgcolor='#EFEFEF'>)
- center alignment (<center>)
- title of the deals and positions table (<h2>Trade Report</h2>)
- beginning of the deals and positions table with alignment, border width, background color, border color, cell spacing and cell padding (<table align='center' border='1' bgcolor='#FFFFFF' bordercolor='#7F7FFF' cellspacing='0' cellpadding='0'>)
- table heading


```
// write the beginning of HTML
   FileWrite(file_handle,"<html>"+
                           "<head>"+
                              "<title>Expert Trade Report</title>"+
                           "</head>"+
                              "<body bgcolor='#EFEFEF'>"+
                              "<center>"+
                              "<h2>Trade Report</h2>"+
                              "<table align='center' border='1' bgcolor='#FFFFFF' bordercolor='#7F7FFF' cellspacing='0' cellpadding='0'>"+
                                 "<tr>"+
                                    "<th rowspan=2>SYMBOL</th>"+
                                    "<th rowspan=2>Direction</th>"+
                                    "<th colspan=3>Open</th>"+
                                    "<th colspan=3>Close</th>"+
                                    "<th rowspan=2>Swap</th>"+
                                    "<th rowspan=2>Profit</th>"+
                                 "</tr>"+
                                 "<tr>"+
                                    "<th>Volume</th>"+
                                    "<th>Time</th>"+
                                    "<th>Price</th>"+
                                    "<th>Volume</th>"+
                                    "<th>Time</th>"+
                                    "<th>Price</th>"+
                                 "</tr>");
```

Getting number of deals in the list.

```
// number of deals in history
total_deals_number=HistoryDealsTotal();
```

Setting dimensions for the symb\_list\[\], lots\_list\[\] and deal\_status\[\] arrays.

```
// setting dimensions for the instruments list, the volumes list and the deals state arrays
ArrayResize(symb_list,total_deals_number);
ArrayResize(lots_list,total_deals_number);
ArrayResize(deal_status,total_deals_number);
```

Initializing all elements of the deal\_status\[\] array with value 0 - all deals are not processed.

```
// setting all elements of array with value 0 - deals are not processed
ArrayInitialize(deal_status,0);
```

Setting the initial values of balance and variable, used to store previous value of balance.

```
balance=0;       // initial balance
balance_prev=0;  // previous balance
```

Setting the initial value of variable, used to store the number of financial instruments in the list.

```
// number of instruments in the list
symb_total=0;
```

Create a loop, that sequentially processes each deal in the list.

```
// processing all deals in history
for(i=0;i<total_deals_number;i++)
  {
```

Select current deal and get its ticket.

```
//select deal, get ticket
ticket=HistoryDealGetTicket(i);
```

Changing the balance by the amount of profit in the current deal.

```
// changing balance
balance+=HistoryDealGetDouble(ticket,DEAL_PROFIT);
```

Getting the time of deal - it will be used frequently further.

```
// reading the time of deal
time_curr=HistoryDealGetInteger(ticket,DEAL_TIME);
```

If it is the first deal in the list - we need to adjust the boundaries of report period and select the period for balance chart, depending on the duration of report period and the width of region, in which the chart will be plotted. Setting the initial values of maximal and minimal balances (these variables will be used to set maximum and minimum of the chart.)

```
// if this is the first deal
if(i==0)
  {
   // if the report period starts before the first deal,
   // then the report period will start from the first deal
   if(StartTime<time_curr) StartTime=time_curr;
   // if report period ends before the current time,
   // then the end of report period corresponds to the current time
   if(EndTime>TimeCurrent()) EndTime=TimeCurrent();
   // initial values of maximal and minimal balances
   // are equal to the current balance
   max_val=balance;
   min_val=balance;
   // calculating the period of balance chart depending on the duration of
   // report period
   Picture1_period=PERIOD_M1;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)) Picture1_period=PERIOD_M2;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*120) Picture1_period=PERIOD_M3;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*180) Picture1_period=PERIOD_M4;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*240) Picture1_period=PERIOD_M5;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*300) Picture1_period=PERIOD_M6;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*360) Picture1_period=PERIOD_M10;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*600) Picture1_period=PERIOD_M12;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*720) Picture1_period=PERIOD_M15;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*900) Picture1_period=PERIOD_M20;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*1200) Picture1_period=PERIOD_M30;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*1800) Picture1_period=PERIOD_H1;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*3600) Picture1_period=PERIOD_H2;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*7200) Picture1_period=PERIOD_H3;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*10800) Picture1_period=PERIOD_H4;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*14400) Picture1_period=PERIOD_H6;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*21600) Picture1_period=PERIOD_H8;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*28800) Picture1_period=PERIOD_H12;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*43200) Picture1_period=PERIOD_D1;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*86400) Picture1_period=PERIOD_W1;
   if(EndTime-StartTime>(Picture1_width-Axis_Width)*604800) Picture1_period=PERIOD_MN1;
   // changing the period of opened chart
   ChartSetSymbolPeriod(hChart,Symbol(),Picture1_period);
  }
```

If this deal is not the first - create the "line" object, using which the chart of balance change is plotted. The line is plotted only if at least one its end is in the report period. If both ends are in the report period - the line will be "thick". The color of balance line is green. If the balance is beyond the range of minimal and maximal balance - this range is adjusted.

```
else
  // if this is not the first deal
  {
   // plotting the balance line, if the deal is in the report period,
   // and setting properties of the balance line
   if(time_curr>=StartTime && time_prev<=EndTime)
     {
      ObjectCreate(hChart,IntegerToString(i),OBJ_TREND,0,time_prev,balance_prev,time_curr,balance);
      ObjectSetInteger(hChart,IntegerToString(i),OBJPROP_COLOR,Green);
      // if both ends of line are in the report period,
      // it will be "thick"
      if(time_prev>=StartTime && time_curr<=EndTime)
        ObjectSetInteger(hChart,IntegerToString(i),OBJPROP_WIDTH,2);
     }
   // if new value of balance exceeds the range
   // of minimal and maximal values, it must be adjusted
   if(balance<min_val) min_val=balance;
   if(balance>max_val) max_val=balance;
  }
```

Assign the previous value of time to corresponding variable.

```
// changing the previous time value
time_prev=time_curr;
```

If the deal has not been processed yet - process it.

```
// if the deal has not been processed yet
if(deal_status[i]<127)
  {
```

If this deal is balance charge and it's in the report period - the corresponding string is written to report. The deal is marked as processed.

```
// If this deal is balance charge
if(HistoryDealGetInteger(ticket,DEAL_TYPE)==DEAL_TYPE_BALANCE)
  {
   // if it's in the report period - write the corresponding string to report.
   if(time_curr>=StartTime && time_curr<=EndTime)
     FileWrite(file_handle,"<tr><td colspan='9'>Balance:</td><td align='right'>",HistoryDealGetDouble(ticket,DEAL_PROFIT),
     "</td></tr>");
   // mark deal as processed
   deal_status[i]=127;
  }
```

If this deal is Buy or Sell, check if this instrument is in the list (symb\_list\[\] array). If not - put it there. The symb\_pointer variable points to element of the symb\_list\[\] array, which contains instrument name of the current deal.

```
// if this deal is buy or sell
if(HistoryDealGetInteger(ticket,DEAL_TYPE)==DEAL_TYPE_BUY || HistoryDealGetInteger(ticket,DEAL_TYPE)==DEAL_TYPE_SELL)
  {
   // check if there is instrument of this deal in the list
   symb_flag=false;
   for(j=0;j<symb_total;j++)
     {
      if(symb_list[j]==HistoryDealGetString(ticket,DEAL_SYMBOL))
        {
         symb_flag=true;
         symb_pointer=j;
        }
     }
   // if there is no instrument of this deal in the list
   if(symb_flag==false)
     {
      symb_list[symb_total]=HistoryDealGetString(ticket,DEAL_SYMBOL);
      lots_list[symb_total]=0;
      symb_pointer=symb_total;
      symb_total++;
     }
```

Set the initial values of the position\_StartTime and position\_EndTime variables, that store the initial and final position lifetime.

```
// set the initial value for the beginning time of deal
position_StartTime=time_curr;
// set the initial value for the end time of deal
position_EndTime=time_curr;
```

The in\_table\_volume, in\_table\_time, in\_table\_price, out\_table\_volume, out\_table\_time, out\_table\_price, out\_table\_swap and out\_table\_profit variables will store tables, that will be inside the cells of larger table: volume, time and price of entering the market; volume, time, price, swap and profit of exiting the market. The in\_table\_volume variable will also store name of financial instrument and link to an image, that corresponds the direction of open position. Assign all these these variables with initial values.

```
// creating the string in report - instrument, position direction, beginning of table for volumes to enter the market
if(HistoryDealGetInteger(ticket,DEAL_TYPE)==DEAL_TYPE_BUY)
   StringConcatenate(in_table_volume,"<tr><td align='left'>",symb_list[symb_pointer],
   "</td><td align='center'><img src='buy.gif'></td><td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>");

if(HistoryDealGetInteger(ticket,DEAL_TYPE)==DEAL_TYPE_SELL)
   StringConcatenate(in_table_volume,"<tr><td align='left'>",symb_list[symb_pointer],
   "</td><td align='center'><img src='sell.gif'></td><td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>");
// creating the beginning of time table to enter the market
in_table_time="<td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>";
// creating the beginning of price table to enter the market
in_table_price="<td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>";
// creating the beginning of volume table to exit the market
out_table_volume="<td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>";
// creating the beginning of time table to exit the market
out_table_time="<td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>";
// creating the beginning of price table to exit the market
out_table_price="<td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>";
// creating the beginning of swap table to exit the market
out_table_swap="<td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>";
// creating the beginning of profit table to exit the market
out_table_profit="<td><table border='1' width='100%' bgcolor='#FFFFFF' bordercolor='#DFDFFF'>";
```

Process all deals starting with the current until position is closed. Process all of them, if they weren't processed earlier.

```
// process all deals for this position starting with the current(until position is closed)
for(j=i;j<total_deals_number;j++)
  {
   // if the deal has not been processed yet - process it
   if(deal_status[j]<127)
     {
```

Select deal and get its ticket.

```
// select deal, get ticket
ticket=HistoryDealGetTicket(j);
```

If the deal is on the same instrument as the open position - process it. Get the deal time. If the deal time goes beyond the range of position time - extend the range. Get the deal volume.

```
// if the instrument of deal matches the instrument of position, that is processed
if(symb_list[symb_pointer]==HistoryDealGetString(ticket,DEAL_SYMBOL))
  {
   // get the deal time
   time_curr=HistoryDealGetInteger(ticket,DEAL_TIME);
   // If the deal time goes beyond the range of position time
   // - extend position time
   if(time_curr<position_StartTime) position_StartTime=time_curr;
   if(time_curr>position_EndTime) position_EndTime=time_curr;
   // get the volume of deal
   lot_current=HistoryDealGetDouble(ticket,DEAL_VOLUME);
```

Buy and Sell deals are processed separately. Begin with the Buy deals.

```
// if this deal is buy
if(HistoryDealGetInteger(ticket,DEAL_TYPE)==DEAL_TYPE_BUY)
  {
```

If you have already opened position for Sell - this deal for Buy will exit the market. And if the volume of deal will be greater than the volume of opened short position - this will be the in/out. Assign string variables with required values. Assign the deal\_status\[\] array with value 127, if the deal is fully processed, or with value 1 if it's the in/out, and this deal must be analyzed for other position.

```
// if position is opened for sell - this will be exit from market
if(NormalizeDouble(lots_list[symb_pointer],2)<0)
  {
   // if buy volume is greater than volume of opened short position - then this is in/out
   if(NormalizeDouble(lot_current+lots_list[symb_pointer],2)>0)
     {
      // creating table of volumes to exit the market - indicate only volume of opened short position
      StringConcatenate(out_table_volume,out_table_volume,"<tr><td align='right'>",DoubleToString(-lots_list[symb_pointer],2),"</td></tr>");
      // mark position as partially processed
      deal_status[j]=1;
     }
   else
     {
      // if buy volume is equal or less than volume of opened short position - then this is partial or full close
      // creating the volume table to exit the market
      StringConcatenate(out_table_volume,out_table_volume,"<tr><td align='right'>",DoubleToString(lot_current,2),"</td></tr>");
      // mark deal as processed
      deal_status[j]=127;
     }

   // creating the time table to exit the market
   StringConcatenate(out_table_time,out_table_time,"<tr><td align='center'>",TimeToString(time_curr,TIME_DATE|TIME_SECONDS),"</td></tr>");

   // creating the price table to exit the market
   StringConcatenate(out_table_price,out_table_price,"<tr><td align='center'>",DoubleToString(HistoryDealGetDouble(ticket,DEAL_PRICE),
   (int)SymbolInfoInteger(symb_list[symb_pointer],SYMBOL_DIGITS)),"</td></tr>");

   // get the swap of current deal
   current_swap=HistoryDealGetDouble(ticket,DEAL_SWAP);

   // if swap is equal to zero - create empty string of the swap table to exit the market
   if(NormalizeDouble(current_swap,2)==0) StringConcatenate(out_table_swap,out_table_swap,"<tr></tr>");
   // else create the swap string in the swap table to exit the market
   else StringConcatenate(out_table_swap,out_table_swap,"<tr><td align='right'>",DoubleToString(current_swap,2),"</td></tr>");

   // get the profit of current deal
   current_profit=HistoryDealGetDouble(ticket,DEAL_PROFIT);

   // if profit is negative (loss) - it is displayed as red in the profit table to exit the market
   if(NormalizeDouble(current_profit,2)<0) StringConcatenate(out_table_profit,out_table_profit,"<tr><td align=right><SPAN style='COLOR: #EF0000'>",
   DoubleToString(current_profit,2),"</SPAN></td></tr>");
   // else - it is displayed as green
   else StringConcatenate(out_table_profit,out_table_profit,"<tr><td align='right'><SPAN style='COLOR: #00EF00'>",
        DoubleToString(current_profit,2),"</SPAN></td></tr>");
  }
```

If you have already opened long position - the buy in this deal will be the enter to the market (the first or addition). If the deal\_status\[\] array element, that corresponds to this deal, has the value 1 - it means that in/out was made. Assign string variables with required values and mark deal as processed (assign the corresponding element of the deal\_status\[\] array with value 127).

```
else
   // if position is opened for buy - this will be the enter to the market
   {
    // if this deal has been already partially processed (in/out)
    if(deal_status[j]==1)
      {
       // create the volume table of entering the market (volume, formed after in/out, is put here)
       StringConcatenate(in_table_volume,in_table_volume,"<tr><td align='right'>",DoubleToString(lots_list[symb_pointer],2),"</td></tr>");
       // indemnity of volume change, which will be produced (the volume of this deal is already taken into account)
       lots_list[symb_pointer]-=lot_current;
      }
    // if this deal has not been processed yet, create the volume table to enter the market
    else StringConcatenate(in_table_volume,in_table_volume,"<tr><td align='right'>",DoubleToString(lot_current,2),"</td></tr>");

    // creating the time table of entering the market
    StringConcatenate(in_table_time,in_table_time,"<tr><td align center>",TimeToString(time_curr,TIME_DATE|TIME_SECONDS),"</td></tr>");

    // creating the price table of entering the market
    StringConcatenate(in_table_price,in_table_price,"<tr><td align='center'>",DoubleToString(HistoryDealGetDouble(ticket,DEAL_PRICE),
    (int)SymbolInfoInteger(symb_list[symb_pointer],SYMBOL_DIGITS)),"</td></tr>");

    // mark deal as processed
    deal_status[j]=127;
   }
```

Change the volume of position to the volume of current deal. If position is closed (the volume is equal to zero) - stop process this position (exit the loop with the j variable) and look for the next unprocessed deal (in the loop with the i variable).

```
 // change of position volume by the current instrument, taking into account the volume of current deal
 lots_list[symb_pointer]+=lot_current;
 // if the volume of opened position by the current instrument became equal to zero - position is closed
 if(NormalizeDouble(lots_list[symb_pointer],2)==0 || deal_status[j]==1) break;
}
```

The sell deals are processed similarly, and then we exit the loop with the j variable.

```
       // if this deal is sell
       if(HistoryDealGetInteger(ticket,DEAL_TYPE)==DEAL_TYPE_SELL)
         {
          // if position has been already opened for buy - this will be the exit from market
          if(NormalizeDouble(lots_list[symb_pointer],2)>0)
            {
             // if sell volume is greater than volume of opened long position - then this is in/out
             if(NormalizeDouble(lot_current-lots_list[symb_pointer],2)>0)
               {
                // creating table of volumes to exit the market - indicate only volume of opened long position
                StringConcatenate(out_table_volume,out_table_volume,"<tr><td align='right'>",DoubleToString(lots_list[symb_pointer],2),"</td></tr>");
                // mark position as partially processed
                deal_status[j]=1;
               }
             else
               {
                // if sell volume is equal or greater than volume of opened short position - then this is partial or full close
                // creating the volume table to exit the market
                StringConcatenate(out_table_volume,out_table_volume,"<tr><td align='right'>",DoubleToString(lot_current,2),"</td></tr>");
                // mark deal as processed
                deal_status[j]=127;
               }

             // creating the time table to exit the market
             StringConcatenate(out_table_time,out_table_time,"<tr><td align='center'>",TimeToString(time_curr,TIME_DATE|TIME_SECONDS),"</td></tr>");

             // creating the price table to exit the market
             StringConcatenate(out_table_price,out_table_price,"<tr><td align='center'>",DoubleToString(HistoryDealGetDouble(ticket,DEAL_PRICE),
             (int)SymbolInfoInteger(symb_list[symb_pointer],SYMBOL_DIGITS)),"</td></tr>");

             // get the swap of current deal
             current_swap=HistoryDealGetDouble(ticket,DEAL_SWAP);

             // if swap is equal to zero - create empty string of the swap table to exit the market
             if(NormalizeDouble(current_swap,2)==0) StringConcatenate(out_table_swap,out_table_swap,"<tr></tr>");
             // else create the swap string in the swap table to exit the market
             else StringConcatenate(out_table_swap,out_table_swap,"<tr><td align='right'>",DoubleToString(current_swap,2),"</td></tr>");

             // get the profit of current deal
             current_profit=HistoryDealGetDouble(ticket,DEAL_PROFIT);

             // if profit is negative (loss) - it is displayed as red in the profit table to exit the market
             if(NormalizeDouble(current_profit,2)<0) StringConcatenate(out_table_profit,out_table_profit,"<tr><td align='right'>
             <SPAN style='COLOR: #EF0000'>",DoubleToString(current_profit,2),"</SPAN></td></tr>");
             // else - it is displayed as green
             else StringConcatenate(out_table_profit,out_table_profit,"<tr><td align='right'><SPAN style='COLOR: #00EF00'>",
                  DoubleToString(current_profit,2),"</SPAN></td></tr>");
            }
          else
            // if position is opened for sell - this will be the enter to the market
            {
             // if this deal has been already partially processed (in/out)
             if(deal_status[j]==1)
               {
                // create the volume table of entering the market (volume, formed after in/out, is put here)
                StringConcatenate(in_table_volume,in_table_volume,"<tr><td align='right'>",DoubleToString(-lots_list[symb_pointer],2),"</td></tr>");

                // indemnity of volume change, which will be produced (the volume of this deal is already taken into account)
                lots_list[symb_pointer]+=lot_current;
               }
             // if this deal has not been processed yet, create the volume table to enter the market
             else StringConcatenate(in_table_volume,in_table_volume,"<tr><td align='right'>",DoubleToString(lot_current,2),"</td></tr>");

             // creating the time table of entering the market
             StringConcatenate(in_table_time,in_table_time,"<tr><td align='center'>",TimeToString(time_curr,TIME_DATE|TIME_SECONDS),"</td></tr>");

             // creating the price table of entering the market
             StringConcatenate(in_table_price,in_table_price,"<tr><td align='center'>",DoubleToString(HistoryDealGetDouble(ticket,DEAL_PRICE),
             (int)SymbolInfoInteger(symb_list[symb_pointer],SYMBOL_DIGITS)),"</td></tr>");

             // mark deal as processed
             deal_status[j]=127;
            }
          // change of position volume by the current instrument, taking into account the volume of current deal
          lots_list[symb_pointer]-=lot_current;
          // if the volume of opened position by the current instrument became equal to zero - position is closed
          if(NormalizeDouble(lots_list[symb_pointer],2)==0 || deal_status[j]==1) break;
         }
      }
   }
}
```

If the time, when position has been opened, is in the report period (at least partially) - the corresponding entry is output to the "report.html" file.

```
// if the position period is in the the report period - the position is printed to report
if(position_EndTime>=StartTime && position_StartTime<=EndTime) FileWrite(file_handle,
in_table_volume,"</table></td>",
in_table_time,"</table></td>",
in_table_price,"</table></td>",
out_table_volume,"</table></td>",
out_table_time,"</table></td>",
out_table_price,"</table></td>",
out_table_swap,"</table></td>",
out_table_profit,"</table></td></tr>");
```

Assign the balance\_prev variable with the balance value. Exit the loop with the i variable.

```
   }
 // changing balance
 balance_prev=balance;
}
```

Write the end of HTML-file (links to images, the end of center alignment, the end of main part, the end HTML-document). Close the "report.html" file.

```
// create the end of html-file
   FileWrite(file_handle,
         "</table><br><br>"+
            "<h2>Balance Chart</h2><img src='picture1.gif'><br><br><br>"+
            "<h2>Price Chart</h2><img src='picture2.gif'>"+
         "</center>"+
         "</body>"+
   "</html>");
// close file
   FileClose(file_handle);
```

Waiting for chart update no longer than time specified in the timeout constant.

```
// get current time
time_curr=TimeCurrent();
// waiting for chart update
while(SeriesInfoInteger(Symbol(),Picture1_period,SERIES_BARS_COUNT)==0 && TimeCurrent()-time_curr<timeout) Sleep(1000);
```

Setting the fixed maximum and minimum of chart.

```
// setting maximal and minimal values for the balance chart (10% indent from upper and lower boundaries)
ChartSetDouble(hChart,CHART_FIXED_MAX,max_val+(max_val-min_val)/10);
ChartSetDouble(hChart,CHART_FIXED_MIN,min_val-(max_val-min_val)/10);
```

Setting properties of the balance chart.

```
// setting properties of the balance chart
ChartSetInteger(hChart,CHART_MODE,CHART_LINE);                // chart as line
ChartSetInteger(hChart,CHART_FOREGROUND,false);               // chart on foreground
ChartSetInteger(hChart,CHART_SHOW_BID_LINE,false);            // hide BID line
ChartSetInteger(hChart,CHART_COLOR_VOLUME,White);             // volumes and orders levels are white
ChartSetInteger(hChart,CHART_COLOR_STOP_LEVEL,White);         // SL and TP levels are white
ChartSetInteger(hChart,CHART_SHOW_GRID,true);                 // show grid
ChartSetInteger(hChart,CHART_COLOR_GRID,LightGray);           // grid is light-gray
ChartSetInteger(hChart,CHART_SHOW_PERIOD_SEP,false);          // hide period separators
ChartSetInteger(hChart,CHART_SHOW_VOLUMES,CHART_VOLUME_HIDE); // hide volumes
ChartSetInteger(hChart,CHART_COLOR_CHART_LINE,White);         // chart is white
ChartSetInteger(hChart,CHART_SCALE,0);                        // minimal scale
ChartSetInteger(hChart,CHART_SCALEFIX,true);                  // fixed scale on vertical axis
ChartSetInteger(hChart,CHART_SHIFT,false);                    // no chart shift
ChartSetInteger(hChart,CHART_AUTOSCROLL,true);                // autoscroll enabled
ChartSetString(hChart,CHART_COMMENT,"BALANCE");               // comment on chart
```

Redrawing the balance chart.

```
// redraw the balance chart
ChartRedraw(hChart);
Sleep(8000);
```

Screen shoot the chart (save the "picture1.gif" image). The width of chart adjusts to the width of the report period (but because of the holidays the inaccuracies often happen and the chart becomes wider than the curve of balance change), the height is calculated as half of the width.

```
// screen shooting the balance chart
ChartScreenShot(hChart,"picture1.gif",(int)(EndTime-StartTime)/PeriodSeconds(Picture1_period),
(int)(EndTime-StartTime)/PeriodSeconds(Picture1_period)/2,ALIGN_RIGHT);
```

Delete all objects from the chart and close it.

```
// delete all objects from the balance chart
ObjectsDeleteAll(hChart);
// close chart
ChartClose(hChart);
```

If sending files via FTP is allowed, send three files: "report.html", picture1.gif "and" picture2.gif ".

```
// if report publication is enabled - send via FTP
// HTML-file and two images - price chart and balance chart
if(TerminalInfoInteger(TERMINAL_FTP_ENABLED))
   {
    SendFTP("report.html");
    SendFTP("picture1.gif");
    SendFTP("picture2.gif");
   }
}
```

For now program description is complete. To send files via FTP you must adjust MetaTrader 5 settings - go to Tools menu, then Options and open Publisher tab (Figure 4).

![Figure 4. Options of publishing report via FTP.](https://c.mql5.com/2/1/fig4.png)

Figure 4. Options of publishing report via FTP.

In the Options dialog box you need to check the "Enable" option, specify account number, FTP address, path, login and password for access. Refresh periodicity does not matter.

Now you can run the script. After running the balance chart appears on the screen for a few seconds and then disappears. In the Journal you can find possible error and see if files were sent via FTP. If everything works fine, then three new file will appear on the server, in specified folder. If you place there two files with arrows images, and if WWW-server is configured and working - you can open a report via web browser.

### 2\. Sending notifications as SMS to mobile phone

There are times when you are away from your computer and other electronic devices, and you have only a mobile phone at hand. But you want to control the trade on your account or monitor quotes for financial instrument. In this case, you can set sending notifications via SMS to mobile phone. Many mobile operators (and third parties) provide EMAIL-SMS service, which allows you to receive messages as letters, sent to a specific email address.

For this you must have an e-mail box (particularly, you must know your SMTP server). Adjust your MetaTrader 5 settings - go to Tools menu, then Options and open Email tab (Figure 5).

![Figure 5. Setting up of sending notifications via email](https://c.mql5.com/2/1/fig5__1.png)

Figure 5. Setting up of sending notifications via email

Check the "Enable" option, specify SMTP server address, login and password, sender address (your e-mail) and recipient address - the email address used to send messages as SMS (check with your mobile operator). If everything is correct, then when you click "Test" button a verification message will be sent (see additional information in the Journal).

The easiest way to be notified when the price reaches a certain level - is to create an alert. To do this open the appropriate "Toolbox" tab, right click and select "Create" (Figure 6).

![Figure 6. Creating alert.](https://c.mql5.com/2/1/fig6.png)

Figure 6. Creating alert.

In this window check the "Enable" option, select the "Mail" action, select financial instrument, condition, enter value for condition and write the text of message. In the "Maximum iterations" enter 1 if you do not want the message to came repeatedly. When all fields are filled out, click OK.

If we send a message from a MQL5-program, we will have more possibilities. We will use the [SendMail()](https://www.mql5.com/en/docs/network/sendmail) function. It has two parameters. First - the title, second - the body of message.

You can call the SendMail() function after trade request ( [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function) or in the [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) event handler. So we will have notifications of trade events - entering the market, placing orders, closing positions. Or you can place [SendMail()](https://www.mql5.com/en/docs/network/sendmail) inside the [OnTimer()](https://www.mql5.com/en/docs/basis/function/events#ontimer) function - we will receive periodic notifications about current quotes. You can arrange sending of notifications when certain trade signals appear - when indicator lines intersect, when price reaches some lines and levels, etc.

Let's consider some examples.

If in Expert Advisor or in Script you replace

```
OrderSend(request,result};
```

with the following

```
string msg_subj,msg_text;
if(OrderSend(request,result))
  {
   switch(request.action)
     {
      case TRADE_ACTION_DEAL:
         switch(request.type)
           {
            case ORDER_TYPE_BUY:
               StringConcatenate(msg_text,"Buy ",result.volume," ",request.symbol," at price ",result.price,", SL=",request.sl,", TP=",request.tp);
               break;
            case ORDER_TYPE_SELL:
               StringConcatenate(msg_text,"Sell ",result.volume," ",request.symbol," at price ",result.price,", SL=",request.sl,", TP=",request.tp);
               break;
           }
         break;
      case TRADE_ACTION_PENDING:
         switch(request.type)
           {
            case ORDER_TYPE_BUY_LIMIT:
               StringConcatenate(msg_text,"Set BuyLimit ",result.volume," ",request.symbol," at price ",request.price,", SL=",request.sl,", TP=",request.tp);
               break;
            case ORDER_TYPE_SELL_LIMIT:
               StringConcatenate(msg_text,"Set SellLimit ",result.volume," ",request.symbol," at price ",request.price,", SL=",request.sl,", TP=",request.tp);
               break;
            case ORDER_TYPE_BUY_STOP:
               StringConcatenate(msg_text,"Set BuyStop ",result.volume," ",request.symbol," at price ",request.price,", SL=",request.sl,", TP=",request.tp);
               break;
            case ORDER_TYPE_SELL_STOP:
               StringConcatenate(msg_text,"Set SellStop ",result.volume," ",request.symbol," at price ",request.price,", SL=",request.sl,", TP=",request.tp);
               break;
            case ORDER_TYPE_BUY_STOP_LIMIT:
               StringConcatenate(msg_text,"Set BuyStopLimit ",result.volume," ",request.symbol," at price ",request.price,", stoplimit=",request.stoplimit,
               ", SL=",request.sl,", TP=",request.tp);
               break;
            case ORDER_TYPE_SELL_STOP_LIMIT:
               StringConcatenate(msg_text,"Set SellStop ",result.volume," ",request.symbol," at price ",request.price,", stoplimit=",request.stoplimit,
               ", SL=",request.sl,", TP=",request.tp);
               break;
           }
         break;
       case TRADE_ACTION_SLTP:
          StringConcatenate(msg_text,"Modify SL&TP. SL=",request.sl,", TP=",request.tp);
          break;
       case TRADE_ACTION_MODIFY:
          StringConcatenate(msg_text,"Modify Order",result.price,", SL=",request.sl,", TP=",request.tp);
          break;
       case TRADE_ACTION_REMOVE:
          msg_text="Delete Order";
          break;
     }
  }
  else msg_text="Error!";
StringConcatenate(msg_subj,AccountInfoInteger(ACCOUNT_LOGIN),"-",AccountInfoString(ACCOUNT_COMPANY));
SendMail(msg_subj,msg_text);
```

then after trade request the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function will send a message using the [SendMail()](https://www.mql5.com/en/docs/network/sendmail) function. It will include information about trade account number, name of a broker and actions made (buy, sell, placing pending order, modification or deletion of order), like the following:

```
59181-MetaQuotes Software Corp. Buy 0.1 EURUSD at price 1.23809, SL=1.2345, TP=1.2415
```

And if in any Expert Advisor or Indicator inside the body of [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) you'll start the timer using the [EventSetTimer()](https://www.mql5.com/en/docs/eventfunctions/eventsettimer) function (it has only one parameter - the period of timer in seconds):

```
void OnInit()
  {
   EventSetTimer(3600);
  }
```

in the [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit) don't forget to turn it off using the [EventKillTimer()](https://www.mql5.com/en/docs/eventfunctions/eventkilltimer):

```
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
```

and in the [OnTimer()](https://www.mql5.com/en/docs/basis/function/events#ontimer) to send messages using the [SendMail()](https://www.mql5.com/en/docs/network/sendmail):

```
void OnTimer()
  {
   SendMail(Symbol(),DoubleToString(SymbolInfoDouble(Symbol(),SYMBOL_BID),_Digits));
  }
```

then you will receive messages about price of current financial instrument with specified period.

### Conclusion

This article describes how to use the MQL5 program to create a HTML and image files and how to upload them to WWW-server via FTP. it also describes how to configure sending notifications to your mobile phone as SMS.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/61](https://www.mql5.com/ru/articles/61)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/61.zip "Download all attachments in the single ZIP archive")

[report.zip](https://www.mql5.com/en/articles/download/61/report.zip "Download report.zip")(33.08 KB)

[sendreport\_en.mq5](https://www.mql5.com/en/articles/download/61/sendreport_en.mq5 "Download sendreport_en.mq5")(32.06 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Creating Tick Indicators in MQL5](https://www.mql5.com/en/articles/60)
- [Introduction to MQL5: How to write simple Expert Advisor and Custom Indicator](https://www.mql5.com/en/articles/35)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1563)**
(4)


![Iurii Tokman](https://c.mql5.com/avatar/2012/9/5049EA35-743B.jpg)

**[Iurii Tokman](https://www.mql5.com/en/users/satop)**
\|
2 Jun 2010 at 14:43

There are no pictures at the beginning, just titles.


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
2 Jun 2010 at 15:27

**satop:**

There are no pictures at the beginning, only titles.

Please attach a screenshot. And tell me your browser version. I can see everything.


![Denis Zyatkevich](https://c.mql5.com/avatar/2010/2/4B83E65A-F9FB.jpg)

**[Denis Zyatkevich](https://www.mql5.com/en/users/zdd)**
\|
2 Jun 2010 at 15:46

**satop:**

No pictures at the beginning, just titles.

Try refreshing the page. Are any of the figures in the article visible? Are the figures in other articles visible?


![Marcelo Plaza](https://c.mql5.com/avatar/2016/7/578B3192-571D.png)

**[Marcelo Plaza](https://www.mql5.com/en/users/mcptrader)**
\|
18 Oct 2015 at 07:41

Denis, thank you for sharing and teaching us how to use the history of deals, and also how to create a report in html.

Very nice work.

I'm trying to integrate it with my EA, but I still have some work to do or maybe just use the modded script whenever I need it.

Thanks again !

Marcelo

![Creating Multi-Colored Indicators in MQL5](https://c.mql5.com/2/0/paint_indicator_MQL5__1.png)[Creating Multi-Colored Indicators in MQL5](https://www.mql5.com/en/articles/135)

In this article, we will consider how to create multi-colored indicators or convert the existing ones to multi-color. MQL5 allows to represent the information in the convenient form. Now it isn't necessary to look at a dozen of charts with indicators and perform analyses of the RSI or Stochastic levels, it's better just to paint the candles with different colors depending on the values of the indicators.

![Transferring Indicators from MQL4 to MQL5](https://c.mql5.com/2/0/migrate_indicators_mql4_to_MQL5__1.png)[Transferring Indicators from MQL4 to MQL5](https://www.mql5.com/en/articles/66)

This article is dedicated to peculiarities of transferring price constructions written in MQL4 to MQL5. To make the process of transferring indicator calculations from MQL4 to MQL5 easier, the mql4\_2\_mql5.mqh library of functions is suggested. Its usage is described on the basis of transferring of the MACD, Stochastic and RSI indicators.

![Limitations and Verifications in Expert Advisors](https://c.mql5.com/2/0/restrictions_in_Experts_MQL5__1.png)[Limitations and Verifications in Expert Advisors](https://www.mql5.com/en/articles/22)

Is it allowed to trade this symbol on Monday? Is there enough money to open position? How big is the loss if Stop Loss triggers? How to limit the number of pending orders? Was the trade operation executed at the current bar or at the previous one? If a trade robot cannot perform this kind of verifications, then any trade strategy can turn into a losing one. This article shows the examples of verifications that are useful in any Expert Advisor.

![The Optimal Method for Calculation of Total Position Volume by Specified Magic Number](https://c.mql5.com/2/0/calculate_volume_MQL5__1.png)[The Optimal Method for Calculation of Total Position Volume by Specified Magic Number](https://www.mql5.com/en/articles/125)

The problem of calculation of the total position volume of the specified symbol and magic number is considered in this article. The proposed method requests only the minimum necessary part of the history of deals, finds the closest time when the total position was equal to zero, and performs the calculations with the recent deals. Working with global variables of the client terminal is also considered.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/61&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071776393131404857)

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