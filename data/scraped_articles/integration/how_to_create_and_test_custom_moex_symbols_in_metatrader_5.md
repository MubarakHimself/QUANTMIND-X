---
title: How to create and test custom MOEX symbols in MetaTrader 5
url: https://www.mql5.com/en/articles/5303
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:15:39.699027
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/5303&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071699831044385994)

MetaTrader 5 / Integration


### Introduction

The two basic types of financial markets include exchange and over-the-counter markets. We can enjoy the OTC Forex trading using modern
MetaTrader and MetaEditor tools, which are constantly being further improved. In addition to trading automation, these tools enable
comprehensive testing of trading algorithms using historical data.

What about using our own ideas for exchange trading? Some exchange trading terminals have built-in programming languages. For example, the
popular Transaq terminal features the ATF (Advanced Trading Facility) programming language. But, of course, it cannot be compared with
MQL5. Moreover, it does not have any strategy testing functionality. A good solution is to obtain exchange data and optimize trading
algorithms in the MetaTrader strategy tester.

This can be done through the creation of custom symbols. The process of custom symbol creation is described in detail in the article [Creating \\
and testing custom symbols in MetaTrader 5](https://www.mql5.com/en/articles/3540). All that is needed is to obtain data in the CSV (TXT) format and import the price history
following the steps described in the article.

This would be easy if not for the difference in data formats. For example, let us consider the popular exchange related web resource finam.ru.
Quotes can be downloaded here:

![](https://c.mql5.com/2/34/Finam.gif)

Exporting quotes from Moscow Exchange

What data formats are provided by Finam:

![](https://c.mql5.com/2/34/u6er.png)

Available date formats: "yyyymmdd", "yymmdd", "ddmmyy", "dd/mm/yy", "mm/dd/yy". Our format:

![](https://c.mql5.com/2/34/7zcwthtsxs3x.png)

The format that we need "yyyy.mm.dd" is not available. So, finam.ru provides a large variety of formats, bot does not have the one we
need.

Furthermore, there a lot of other exchange resources. Formats provided by other sites may also be inappropriate. We need a certain order of data.
However, quotes can be stored in a different order, for example, Open, Close, High, Low.

Therefore, our task is to convert data provided in a random order and different formats into the required format. This will provide an opportunity to
receive data for MetaTrader 5 from any resources. Then we will create a custom symbol based on the received data using the MQL5 tools, which
will enable us to perform tests.

There are a couple of difficulties connected with the import of quotes.

The exchange supports spread, Ask and Bid. However, all these values exist only at the "moment", in the Market Depth. After that only the deal
price is written regardless of its execution price, i.e. ask or bid. We need the spread value for the terminal. Here a fixed spread is added,
because it is impossible to restore the Market Depth spread. If the spread is essential, you can simulate it somehow. One of the methods is
described in the article

[Modeling time series using custom symbols according to specified distribution laws](https://www.mql5.com/en/articles/4566) _._ Alternatively,
you can write a simple function presenting the dependence of spread on volatility

**Spread = f(High-Low).**

When working with timeframes, use of a fixed spread is quite acceptable. The error will be insignificant on large periods. However, spread
modeling is important for ticks. Exchange tick format:

![](https://c.mql5.com/2/35/Ticks.png)

Our format:

![](https://c.mql5.com/2/35/Symbols__1.png)

Here we need to set ASK and BID in addition to LAST. Data is sorted with millisecond precision. The exchange provides only a stream of prices.
Data in the first page are more like dividing a large lot into pieces. There are no ticks, in Forex terms. This can be Bid, Ask or both Bid and Ask
at the same time. In addition, we need to artificially rank deals by time and add milliseconds.

Thus, the article deals not with the data import, but with data modeling, just like the article mentioned above. Therefore, not to mislead you, I
decided not to publish tick importing application operating according to the principle of ASK=BID(+spread)=LAST. Spread is significant
when working with milliseconds, so in testing we need to choose an appropriate modeling method.

Amending of the tick importing code after that will take a couple of minutes. It is only necessary to replace the MqlRates structure with MqlTick.
The

[CustomRatesUpdate(](https://www.mql5.com/en/docs/customsymbols/customratesupdate)) function needs to be replaced with [CustomTicksAdd()](https://www.mql5.com/en/docs/customsymbols/customticksadd).

The next point is connected with the inability to consider all possible data formats. For example, numbers can be written with separators 1
000 000 or with a comma used instead of a point for decimal separator like 3,14. Or even worse - when both the data separator and the decimal
separator are a dot or a comma (how do you distinguish between them). Only the most common formats are considered here. If you need to deal with
a non-standard format, you will have to process it yourself.

In addition, there is no tick history on the exchange — it only provides deal volumes. Therefore, in this article we use exchange volume
=VOL=TICKVOL.

The article is divided into two parts. Part one features the code description. It allows you to familiarize with the code, so that later you
will be able to edit it for operations with non-standard data formats. Part two contains a step by step guide (user manual). It is intended for
those, who are not interested in programming but only need to use the implemented functionality. If you operate with standard data formats
(in particular, use the finam.ru website as a source), you can immediately proceed to part 2.

### Part 1. Code Description

Only part of code is provided here. The full code is available in the attached file.

First, let us enter the required parameters, such as position of data in the string, file parameters, symbol name, etc.

```
input int SkipString        =1;                               // The number of strings to skip
input string mark1          ="Time position and format";      // Time
input DATE indate           =yyyymmdd;                        // Source date format
input TIME intime           =hhdmmdss;                        // Source time format
input int DatePosition      =1;                               // Date position
input int TimePosition      =2;                               // Time position
//------------------------------------------------------------------+
input string mark2          ="Price data position";           // Price
input int OpenPosition      =3;                               // Open price position
input int HighPosition      =4;                               // High price position
input int LowPosiotion      =5;                               // Low price position
input int ClosePosition     =6;                               // Close price position
input int VolumePosition    =7;                               // Volume position
input string mark3          ="File parameters";               // File
//-------------------------------------------------------------------+
input string InFileName     ="sb";                            // Source file name
input DELIMITER Delimiter   =comma;                           // Separator
input CODE StrType          =ansi;                            // String type
input string mark4          ="Other parameters";              // Other
//-------------------------------------------------------------------+
input string spread         ="2";                             // Fixed spread in points
input string Name           ="SberFX";                        // The name of the symbol you are creating
```

Enumerations are created for some data. For example, for the date and time format:

```
enum DATE
{
yyyycmmcdd, // yyyy.mm.dd
yyyymmdd,   // yyyymmdd
yymmdd,     // yymmdd
ddmmyy,     // ddmmyy
ddslmmslyy, // dd/mm/yy
mmslddslyy  // mm/dd/yy
// Additional formats should be added here
};

enum TIME
{
hhmmss,     // hhmmss
hhmm,       // hhmm
hhdmmdss,   // hh:mm:ss
hhdmm       // hh:mm
// Additional formats should be added here
};
```

If the required format is not available, add it.

Then open the source file. For the convenient editing of formatted data, I suggest saving them in a CSV file. At the same time, the data
should be written to the

[MqlRates](https://www.mql5.com/en/docs/constants/structures/mqlrates) structure to enable the automatic creation
of the custom symbol.

```
// Open in file

  int out =FileOpen(InFileName,FILE_READ|StrType|FILE_TXT);
  if(out==INVALID_HANDLE)
  {
   Alert("Failed to open the file for reading");
   return;
  }
// Open out file
  int in =FileOpen(Name+"(f).csv",FILE_WRITE|FILE_ANSI|FILE_CSV);
  if(in==INVALID_HANDLE)
  {
   Alert("Failed to open the file for writing");
   return;
  }
  //---Insert caption string
  string Caption ="<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>";
  FileWrite(in,Caption);
//-----------------------------------------------------------
string fdate="",ftime="",open="";
string high="",low="",close="",vol="";
int left=0,right=0;

string str="",temp="";
for(int i=0;i<SkipString;i++)
   {
   str =FileReadString(out);
   i++;
   }
MqlRates Rs[];
ArrayResize(Rs,43200,43200);  //43200 minutes in a month
datetime time =0;
```

The source file must be saved to the MQL5/Files directory. The SkipString external variable presents the number of lines in the file
header to skip. To be able to use spaces and tabs as separators, we open the file with the

[flag](https://www.mql5.com/en/docs/constants/io_constants/fileflags) FILE\_TXT.

Then we need to extract data from the string. The location is specified in input parameters. Numbering starts from 1. Let us use Sberbank
shares quotes as an example.

![](https://c.mql5.com/2/35/SBER.png)

Here the date position is 1, time is 2, etc. SkipString=1.

To parse the string, we could use the [StringSplit()](https://www.mql5.com/en/docs/strings/stringsplit)
function. But it is better to develop our own functions, for a more convenient monitoring of errors in the source file. Data analysis can
be added to such functions. Although, using the StringSplit() code would be easier. The first function which finds data boundaries
receives the string, the separator and the position. Boundaries are written to the a and b variables which are passed by reference.

```
//---Search for data position boundaries-----------------------------+
bool SearchBorders(string str,int pos,int &a,int &b,DELIMITER delim)
{
// Auxiliary variables
int left=0,right=0;
int count=0;
int start=0;
string delimiter="";
//-------------------------------------------------------------------+

switch(delim)
{
case comma : delimiter =",";
   break;
case tab : delimiter ="/t";
   break;
case space : delimiter =" ";
   break;
case semicolon : delimiter =";";
   break;
}

while(count!=pos||right!=-1)
   {
   right =StringFind(str,delimiter,start);

   if(right==-1&&count==0){Print("Wrong date");return false;} //Incorrect data

   if(right==-1)
      {
      right =StringLen(str)-1;
      a =left;
      b =right;
      break;
      }

   count++;
      if(count==pos)
      {
      a =left;
      b =right-1;
      return true;
      }
   left =right+1;
   start =left;
   }

return true;
}
```

Now, let us obtain appropriate data using [StringSubstr()](https://www.mql5.com/en/docs/strings/stringsubstr).
The received values must be converted to the desired format. For that purpose, let's write date and time conversion functions. For
example, here is the date conversion function:

```
//---Date formatting-------------------------------------------------+
//2017.01.02
string DateFormat(string str,DATE date)
{

string res="";
string yy="";

switch(date)
  {
   case yyyycmmcdd :  //Our format
      res =str;
      if(StringLen(res)!=10)res=""; // Checking the date format
   case yyyymmdd :
      res =StringSubstr(str,0,4)+"."+StringSubstr(str,4,2)+"."+StringSubstr(str,6,2);
      if(StringLen(res)!=10)res=""; // Checking the date format
      break;
   case yymmdd :
      yy =StringSubstr(str,0,2);
      if(StringToInteger(yy)>=70)
         yy ="19"+yy;
      else
         yy ="20"+yy;
      res =yy+"."+StringSubstr(str,2,2)+"."+StringSubstr(str,4,2);
      if(StringLen(res)!=10)res=""; // Checking the date format
      break;
//---Other formats (full code is in the file)-------------
//Add parsing of other formats if necessary
   default :
      break;

  }

return res;
}
```

If the required format is not available (for example a date like 01 January 18), it should be added. Here a check is performed of whether
the received data corresponds to the required format (in case of an error in the source file)

if(StringLen(res)!=10)
res="";. I
understand that this is not a thorough check. But data analysis is not an easy task, so a separate program would be needed for a more
detailed analysis. In case of an error, the function returns res =,"" and the appropriate line is then skipped.



The following conversion is provided for formats of type ddmmyy, in which the year is written as two digits. Values >=70 are converted
to 19yy, values less than that are converted to 20yy.

After format conversion, we write data to appropriate variables and compile a final string.

```
while(!FileIsEnding(out))
   {
   str =FileReadString(out);
   count++;
//---fdate-----------------------------
   if(SearchBorders(str,DatePosition,left,right,Delimiter))
   {
   temp =StringSubstr(str,left,right-left+1);
   fdate =DateFormat(temp,indate);
   if(fdate==""){Print("Error in string   ",count);continue;}
   }
   else {Print("Error in string   ",count);continue;}
//---Other data are handled similarly
```

If an error is found in functions SearchBorders, DateFormat or TimeFormat, the string is skipped and its sequence number is written
using the Print() function. All enumerations and format conversion functions are located in a separate include file,
FormatFunctions.mqh.

Then the resulting string is formed and written. Data are assigned to appropriate elements of the MqlRates structure.

```
//-------------------------------------------------------------------+
   str =fdate+","+ftime+","+open+","+high+","+low+","+close+","+vol+","+vol+","+Spread;
   FileWrite(in,str);
//---Filling MqlRates -----------------------------------------------+
   Rs[i].time               =time;
   Rs[i].open               =StringToDouble(open);
   Rs[i].high               =StringToDouble(high);
   Rs[i].low                =StringToDouble(low);
   Rs[i].close              =StringToDouble(close);
   Rs[i].real_volume        =StringToInteger(vol);
   Rs[i].tick_volume        =StringToInteger(vol);
   Rs[i].spread             =int(StringToInteger(Spread));
   i++;
//-------------------------------------------------------------------+
   }
```

After reading all strings, the dynamic array gets its final size and the files are closed:

```
   ArrayResize(Rs,i);
   FileClose(out);
   FileClose(in);
```

Now everything is ready for creating a custom symbol. In addition, we have a CSV file, which can be easily edited directly in MetaEditor.
Based on the CSV file, we can create custom symbols using standard methods in the MetaTrader 5 terminal.

![](https://c.mql5.com/2/35/SBERuf9.png)

### Creating a custom symbol using MQL5

Now that all the data have been prepared, we only need to add the custom symbol.

```
   CustomSymbolCreate(Name);
   CustomRatesUpdate(Name,Rs);
```

Quotes are imported using the [CustomRatesUpdate()](https://www.mql5.com/en/docs/customsymbols/customratesupdate)
function, which means that the program can be used not only for symbol creation, but also for the addition of new data. If the symbol already
exists,

[CustomSymbolCreate()](https://www.mql5.com/en/docs/customsymbols/customsymbolcreate) will
return -1 (minus one) and program execution will continue, so quotes will be updated through the

CustomRatesUpdate() function. The symbol is displayed in the [MarketWatch](https://www.metatrader5.com/en/terminal/help/trading/market_watch "https://www.metatrader5.com/en/terminal/help/trading/market_watch") window
and is highlighted in green.

![](https://c.mql5.com/2/35/MW.png)

Now we can open the chart to make sure that everything works correctly:

![](https://c.mql5.com/2/35/yqnnzz_mqmlydrl7.png)

### The EURUSD chart

### Setting specifications (symbol properties)

When testing a symbol, we may need to configure its characteristics (specifications). I have written a separate Specification include file,
which enables the convenient editing of symbol properties. In this file, symbol properties are set in the SetSpecifications() function.
All symbol properties from the

ENUM\_SYMBOL\_INFO\_INTEGER, ENUM\_SYMBOL\_INFO\_DOUBLE , ENUM\_SYMBOL\_INFO\_STRING enumerations ate collected here.

```
void SetSpecifications(string Name)
   {
//---Integer Properties-------------------------------------
//   CustomSymbolSetInteger(Name,SYMBOL_CUSTOM,true);                       // bool An indication that the symbol is custom
//   CustomSymbolSetInteger(Name,SYMBOL_BACKGROUND_COLOR,clrGreen);         // color The background color used for the symbol in Market Watch
// Other Integer properties
//---Double Properties ---------------------------------------------------
//   CustomSymbolSetDouble(Name,SYMBOL_BID,0);                              // Bid, the best price at which a symbol can be sold
//  CustomSymbolSetDouble(Name,SYMBOL_BIDHIGH,0);                           // Highest Bid per day
//   CustomSymbolSetDouble(Name,SYMBOL_BIDLOW,0);                           // Lowest Bid per day
// Other Double properties
//---String Properties-----------------------------------------------+
//   CustomSymbolSetString(Name,SYMBOL_BASIS,"");                           // The name of the underlaying asset for the custom symbol
//   CustomSymbolSetString(Name,SYMBOL_CURRENCY_BASE,"");                   // Base currency of the symbol
//   CustomSymbolSetString(Name,SYMBOL_CURRENCY_PROFIT,"");                 // Profit currency
// Other String properties
}
```

This function is executed after the CustomSymbolCreate function. It is not known in advance which type of symbol this is, futures, stock or
option, most properties are not required and are commented out. Only some of the lines are uncommented in the source code:

```
   CustomSymbolSetInteger(Name,SYMBOL_CUSTOM,true);                       // bool An indication that the symbol is custom
   CustomSymbolSetInteger(Name,SYMBOL_BACKGROUND_COLOR,clrGreen);         // color The background color used for the symbol in Market Watch
   CustomSymbolSetInteger(Name,SYMBOL_SELECT,true);                       // bool An indication that the symbol is selected in Market Watch
   CustomSymbolSetInteger(Name,SYMBOL_VISIBLE,true);                      // bool An indication that the symbol is displayed in Market Watch
```

The following parameters are uncommented for testing purposes: minimum volume, volume step, price step, point size, which are the most
necessary characteristics. This characteristics are typical of Sberbank stocks. The set of properties and their characteristics differ
for different symbols.

```
   CustomSymbolSetDouble(name,SYMBOL_POINT,0.01);               // The value of one point
   CustomSymbolSetDouble(name,SYMBOL_VOLUME_MIN,1);             // Minimum volume for a deal
   CustomSymbolSetDouble(name,SYMBOL_VOLUME_STEP,1);            // Minimum volume change step
   CustomSymbolSetInteger(name,SYMBOL_DIGITS,2);                // int Number of decimal places
   CustomSymbolSetInteger(name,SYMBOL_SPREAD,2);                // int Spread value in points
   CustomSymbolSetInteger(name,SYMBOL_SPREAD_FLOAT,false);      // bool An indication of floating spread
   CustomSymbolSetDouble(name,SYMBOL_TRADE_TICK_SIZE,0.01);	// Minimum price change
```

This would be a good approach, if we didn't have to recompile the code each time we need to set the desired properties. It would be more convenient
if this could be done just by entering desired parameters. Therefore I had to change the approach. Symbol properties will be provided in a
plain text file Specifications.txt, which can be edited manually for each new symbol. This does not require the recompilation of the source
code.

It is more convenient to edit the text file in MetaEditor. Mainly because MetaEditor provides highlighting of parameters and data. The
properties are written in the following format:

[![](https://c.mql5.com/2/35/Specs.png)](https://c.mql5.com/2/35/Specifications.png "https://c.mql5.com/2/35/Specifications.png")

[https://c.mql5.com/2/35/Specifications.png](https://c.mql5.com/2/35/Specifications.png "https://c.mql5.com/2/35/Specifications.png")

Data are separated by commas. The strings are parsed as follows:

```
   while(!FileIsEnding(handle))
     {
     str =FileReadString(handle);
//--- Skipping lines -----------------------+
     if(str=="") continue;
     if(StringFind(str,"//")<10) continue;
//------------------------------------------+
     sub =StringSplit(str,u_sep,split);
     if(sub<2) continue;
     SetProperties(SName,split[0],split[1]);
     }
```

A line is skipped if it is empty or there is the comment symbol "//" at the beginning (position<10). Then the string is divided into
substrings using the

StringSplit() function. After that the
strings are passed to the SetProperties() function, where symbol properties are set. The function code structure:

```
void SetProperties(string name,string str1,string str2)
   {
   int n =StringTrimLeft(str1);
       n =StringTrimRight(str1);
       n =StringTrimLeft(str2);
       n =StringTrimRight(str2);

   if(str1=="SYMBOL_CUSTOM")
      {
      if(str2=="0"||str2=="false"){CustomSymbolSetInteger(name,SYMBOL_CUSTOM,false);}
      else {CustomSymbolSetInteger(name,SYMBOL_CUSTOM,true);}
      return;
      }
   if(str1=="SYMBOL_BACKGROUND_COLOR")
      {
      CustomSymbolSetInteger(name,SYMBOL_BACKGROUND_COLOR,StringToInteger(str2));
      return;
      }
   if(str1=="SYMBOL_CHART_MODE")
      {
      if(str2=="SYMBOL_CHART_MODE_BID"){CustomSymbolSetInteger(name,SYMBOL_CHART_MODE,SYMBOL_CHART_MODE_BID);}
      if(str2=="SYMBOL_CHART_MODE_LAST"){CustomSymbolSetInteger(name,SYMBOL_CHART_MODE,SYMBOL_CHART_MODE_LAST);}
      return;
      }
//--- Other symbol properties
}
```

Two more functions are added for the cases if the user leaves spaces or tabs when editing, [StringTrimLeft()](https://www.mql5.com/en/docs/strings/stringtrimleft)and[StringTrimRight()](https://www.mql5.com/en/docs/strings/stringtrimright).

The full code is available in the include file PropertiesSet.mqh.

Now all the symbol properties are set through the attached text file, while no additional re-compilation is needed. You may check both code
variants, which are attached below. The first variant which requires the property setting via an include file is commented out.

### Interface

For code editing convenience, settings are specified using [input \\
parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables). If there is nothing to edit, we can think about the interface. For the final version, I have developed the inputs panel:

![](https://c.mql5.com/2/37/0kglt2_en.png)

About the panel code. [The standard set of controls](https://www.mql5.com/en/docs/standardlibrary/controls)
from the following include files is used here:

```
#include <Controls\Dialog.mqh>
#include <Controls\Label.mqh>
#include <Controls\Button.mqh>
#include <Controls\ComboBox.mqh>
```

An event handler has been created for the OK button.

```
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CFormatPanel)
ON_EVENT(ON_CLICK,BOK,OnClickButton)
EVENT_MAP_END(CAppDialog)

void CFormatPanel::OnClickButton(void)
  {
// Program described above
  }
```

Now almost the entire code of the program described above is moved to this event handler. External parameters become [local \\
variables](https://www.mql5.com/en/docs/basis/variables/local).

```
long SkipString        =1;                              // The number of strings to skip
DATE indate           =yyyymmdd;                        // Source date format
TIME intime           =hhdmmdss;                        // Source time format
int DatePosition      =1;                               // Date position
int TimePosition      =2;                               // Time position
// Other parameters
```

The Create() function is written for each control, so appropriate values are added to the list of controls after its execution. For example,
the following is done for the date format:

```
//-----------ComboBox Date Format------------------------------------+
    if(!CreateComboBox(CDateFormat,"ComDateFormat",x0,y0+h+1,x0+w,y0+2*h+1))
     {
      return false;
     }
   CDateFormat.ListViewItems(6);
   CDateFormat.AddItem(" yyyy.mm.dd",0);
   CDateFormat.AddItem(" yyyymmdd",1);
   CDateFormat.AddItem(" yymmdd",2);
   CDateFormat.AddItem(" ddmmyy",3);
   CDateFormat.AddItem(" dd/mm/yy",4);
   CDateFormat.AddItem(" mm/dd/yy",5);
   CDateFormat.Select(1);
     }
```

These values are then returned from the input fields to the corresponding variables:

```
long sw;
SkipString =StringToInteger(ESkip.Text());

sw =CDateFormat.Value();
switch(int(sw))
{
   case 0 :indate =yyyycmmcdd;
      break;
   case 1 :indate =yyyymmdd;
      break;
   case 2 :indate =yymmdd;
      break;
   case 3 :indate =ddmmyy;
      break;
   case 4 :indate =ddslmmslyy;
      break;
   case 5 :indate =mmslddslyy;
      break;
}
//Other variables
```

This version has a larger implementation, so if you need to edit the code, you should work with the input version.

### Part 2. Step-by-Step Guide

This part features a step-by-step description of actions required for the creation of a custom exchange symbol. This guide can be used
when available quotes have any of standard formats and you do not need to edit the code. For example, if the quotes are obtained from the
site finam.ru site. If the quotes are in some non-standard formats, then you should edit the code described in Part 1.

So, we have a source file with exchange quotes of a financial instrument. Suppose, we have obtained it from the Finam site as described at
the article beginning. Do not forget that we need the quotes of the one-minute timeframe.

Two data import options are described in the article. You may use either the CreateCustomSymbol script and the
CreateSymbolPanel Expert Advisor, which has the inputs panel. Both EAs perform exactly the same. For example, let us consider
operation with the inputs panel. In the examples provided here, we use Sberbank shares quotes from

[Moscow Exchange](https://www.mql5.com/go?link=https://www.moex.com/ "https://www.moex.com/").
The quotes are attached below, in the sb.csv file.

1\. **Arrangement of files**

First of all, we need to save the quotes file to MQL5/Files. This is connected with the MQL5 programming concept, due to which operations
with files are strictly controlled for security reasons. The easiest method to find the desired directory is to open it from
MetaTrader. In the Navigator window, right-click in the Files folder and select "Open folder" from the context menu.

![](https://c.mql5.com/2/35/MEMW.png)

The source data file should be saved to this folder (the location of program files is described in the Files chapter below). The file can now be
opened in MetaEditor.

![](https://c.mql5.com/2/34/wf8kj87e6.png)

Add Specifications.txt to the same folder. It sets symbol properties.

2\. **Inputs**

The next step is to determine data format and position, select file properties and set the name for our custom symbol. The example of how the
fields can be filled is shown below:

![](https://c.mql5.com/2/35/DateFormat.png)

The data should be transferred to the panel. Fixed spread in points is used in this version, so floating spread is not modeled. Therefore, you
should enter an appropriate spread value here.

![](https://c.mql5.com/2/37/a72nmh_en.png)

Write the full file name including the extension.

Now, before you click "OK", specify the necessary symbol specifications. They are available in the Specifications.txt file which was
earlier placed in MQL5/Files.

It is very convenient to edit the text file in MetaEditor. The main reason is data highlighting supported in MetaEditor. If you cannot
understand any of the properties, hover the cursor over it and press F1.

![](https://c.mql5.com/2/35/Specs.png)

Properties are highlighted in red and values are shown in green. The commented properties (//) which are not used are shown in gray. Note that commas
are used for data separation. Do not delete properties while editing.

_**In order to avoid errors, you should keep the existing format.**_

To edit properties, uncomment the desired ones (remove "//") and then set the appropriate value. A minimum set of properties given set in the
attached file: price step, point value, minimum lot, etc.

All these characteristics (in the source file) are required for Sberbank stocks on Moscow Exchange. **Different**
**characteristics are required for other financial instruments, therefore you need to edit the properties.**

Theminimum required set of properties is located at the very beginning of the file.

Usually, stock prices have 2 decimal places (SYMBOL\_DIGITS), while the point
value is equal to 0.01 rubles. The number of decimal places in stock futures prices is 0 and point value is 1 ruble. See specifications at

[moex.com](https://www.mql5.com/go?link=https://www.moex.com/ "https://www.moex.com/").

Once you have set all the required properties, click OK. The created custom symbol will appear in the navigator window. In my example, it is
highlighted in green.

![](https://c.mql5.com/2/35/qt037cpz32.png)

Open the chart to check:

![](https://c.mql5.com/2/35/bl8s9xen_nmlv6f.png)

Everything is fine, so the custom symbol can now be tested in the [Strategy \\
Tester](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing").

Custom symbol settings is performed similarly to a standard symbol. An important point here is to properly configure symbol
specifications.

As an example, let us test any standard Expert Advisor available in the terminal (here the Moving Average) using our data:

![](https://c.mql5.com/2/35/ED.png)

Everything works as expected. If you need to add new quotes or change the properties, simply repeat the described actions for the already existing
symbol. If the specifications have not changed, click OK without editing the properties.

### Files

The attached files are located in folders the way they should be saved to your computer:

- CreateCustomSymbol script and code: MQL5\\Scripts
- CreateSymbolPanel Expert Advisor and code: MQL5\\Experts
- Include files FormatFunctions, PropertiesSet, Specification: MQL5\\Include
- Text file with symbol settings: in MQL5\\Files

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5303](https://www.mql5.com/ru/articles/5303)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5303.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/5303/mql5.zip "Download MQL5.zip")(526.52 KB)

[CreateCustomSymbol.mq5](https://www.mql5.com/en/articles/download/5303/createcustomsymbol.mq5 "Download CreateCustomSymbol.mq5")(13.49 KB)

[CreateSymbolPanel.mq5](https://www.mql5.com/en/articles/download/5303/createsymbolpanel.mq5 "Download CreateSymbolPanel.mq5")(47.11 KB)

[PropertiesSet.mqh](https://www.mql5.com/en/articles/download/5303/propertiesset.mqh "Download PropertiesSet.mqh")(35.55 KB)

[FormatFunctions.mqh](https://www.mql5.com/en/articles/download/5303/formatfunctions.mqh "Download FormatFunctions.mqh")(9.14 KB)

[Specification.mqh](https://www.mql5.com/en/articles/download/5303/specification.mqh "Download Specification.mqh")(21.21 KB)

[Specifications.txt](https://www.mql5.com/en/articles/download/5303/specifications.txt "Download Specifications.txt")(8.04 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Money Management by Vince. Implementation as a module for MQL5 Wizard](https://www.mql5.com/en/articles/4162)
- [The NRTR indicator and trading modules based on NRTR for the MQL5 Wizard](https://www.mql5.com/en/articles/3690)
- [Sorting methods and their visualization using MQL5](https://www.mql5.com/en/articles/3118)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/302157)**
(5)


![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
6 Dec 2018 at 19:54

Thanks to the author for an interesting approach. It's a pity that not ticks [were imported](https://www.mql5.com/en/economic-calendar/united-states/imports "US Economic Calendar: Imports (Imports)"), from which you can build minutes on the fly. It's also a pity that there is no OOP... but that's already a bias....

Personally, I liked the solution of date and time format.

A small remark. The files for the examples in the article are in the MQL5.zip archive. The unzipped sources are also located here....

![Dmitrii Troshin](https://c.mql5.com/avatar/2020/3/5E5D0467-98B7.png)

**[Dmitrii Troshin](https://www.mql5.com/en/users/orangetree)**
\|
6 Dec 2018 at 23:24

**Denis Kirichenko:**

Thanks to the author for an interesting approach. It's a pity that not ticks were imported, from which you can build minutes on the fly. It's also a pity that there is no OOP... but that's already an addiction....

Personally, I liked the solution of date and time format.

A small remark. The files for the examples in the article are in the MQL5.zip archive. And there are unzipped sources here as well....

When I started writing, it seemed to be two lines of code - redo date, redo time. That's why without OOP :)

![Vasiliy Smirnov](https://c.mql5.com/avatar/2009/11/4B14587B-1412.jpg)

**[Vasiliy Smirnov](https://www.mql5.com/en/users/zfs)**
\|
7 Dec 2018 at 08:28

Something and don't need to import, some of the mmvb shares are in the terminal at brokers and metaquot.


![Aleksandr Masterskikh](https://c.mql5.com/avatar/2017/5/591837C6-87CC.jpg)

**[Aleksandr Masterskikh](https://www.mql5.com/en/users/a.masterskikh)**
\|
9 Dec 2018 at 09:33

Although it is easier to use financial instruments - analogues of exchange shares (in the form of cfd), which many brokers have in the Metatrader platform,

This is very interesting information for understanding the specifics of the stock exchange, and understanding the differences from the forex market.

Thanks to the author for the article!

![Roman Vasilchenko](https://c.mql5.com/avatar/avatar_na2.png)

**[Roman Vasilchenko](https://www.mql5.com/en/users/vrs42)**
\|
12 Jan 2019 at 05:28

I don't know whether to open a secret or not.... but MICEX brokers have MetaTrader in their available terminals. Just open a [demo account](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 documentation: Account Properties") if you don't have a real one and test)

![Selection and navigation utility in MQL5 and MQL4: Adding "homework" tabs and saving graphical objects](https://c.mql5.com/2/35/Select_Symbols_Utility_MQL5.png)[Selection and navigation utility in MQL5 and MQL4: Adding "homework" tabs and saving graphical objects](https://www.mql5.com/en/articles/5417)

In this article, we are going to expand the capabilities of the previously created utility by adding tabs for selecting the symbols we need. We will also learn how to save graphical objects we have created on the specific symbol chart, so that we do not have to constantly create them again. Besides, we will find out how to work only with symbols that have been preliminarily selected using a specific website.

![Applying the probability theory to trading gaps](https://c.mql5.com/2/34/Gap_Probability.png)[Applying the probability theory to trading gaps](https://www.mql5.com/en/articles/5373)

In this article, we will apply the probability theory and mathematical statistics methods to creating and testing trading strategies. We will also look for optimal trading risk using the differences between the price and the random walk. It is proved that if prices behave like a zero-drift random walk (with no directional trend), then profitable trading is impossible.

![Separate optimization of a strategy on trend and flat conditions](https://c.mql5.com/2/35/Frame_2.png)[Separate optimization of a strategy on trend and flat conditions](https://www.mql5.com/en/articles/5427)

The article considers applying the separate optimization method during various market conditions. Separate optimization means defining trading system's optimal parameters by optimizing for an uptrend and downtrend separately. To reduce the effect of false signals and improve profitability, the systems are made flexible, meaning they have some specific set of settings or input data, which is justified because the market behavior is constantly changing.

![Developing the symbol selection and navigation utility in MQL5 and MQL4](https://c.mql5.com/2/34/Select_Symbols_Utility_MQL5.png)[Developing the symbol selection and navigation utility in MQL5 and MQL4](https://www.mql5.com/en/articles/5348)

Experienced traders are well aware of the fact that most time-consuming things in trading are not opening and tracking positions but selecting symbols and looking for entry points. In this article, we will develop an EA simplifying the search for entry points on trading instruments provided by your broker.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ezemlcvxnwqrkzybtwyesngpszmuostn&ssn=1769192138579186586&ssn_dr=0&ssn_sr=0&fv_date=1769192138&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5303&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20create%20and%20test%20custom%20MOEX%20symbols%20in%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919213839095474&fz_uniq=5071699831044385994&sv=2552)

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