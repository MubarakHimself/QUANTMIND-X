---
title: Continuous Walk-Forward Optimization (Part 3): Adapting a Robot to Auto Optimizer
url: https://www.mql5.com/en/articles/7490
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:14:47.679513
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/7490&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071687525963082901)

MetaTrader 5 / Examples


### Introduction

This is the third article from a series devoted to the creation of an auto optimizer for the continuous walk-forward optimization. The
previous two articles are available at the following links:

1. [Continuous Walk-Forward Optimization (Part 1): Working with optimization reports](https://www.mql5.com/en/articles/7290)
2. [Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://www.mql5.com/en/articles/7452)

The first article in this series is devoted to the creation of a mechanism for working and forming trading report files, which are needed for
the auto optimizer to operate. The second article presented the key objects which implement trading history downloading and create
trading reports based on the downloaded data. The current article serves as a bridge between the previous two parts: it describes the
mechanism of interaction with the DLL considered in the first article and the objects for report downloading, which were described in the
second article.

We will analyze the process of wrapper creation for a class which is imported from DLL and which forms an XML file with the trading history. We
will also consider a method for interacting with this wrapper. The article also contains descriptions of two functions which download
detailed and generalized trading history for further analysis. In the conclusion, I will present a ready-to-use template which can work
with the auto optimizer. I will also show an example of a standard algorithm from the default experts set, which demonstrates how any
existing algorithm can be amended to interact with the auto optimizer.

### Download accumulated trading history

Sometimes we need to download trading history to a file for further history analysis and for other purposes. Unfortunately, such an interface is not
yet available in the terminal, however, this task can be implemented using the classes described in the previous article. The directory
where the files of the described classes are located, has two other files, "ShortReport.mqh" and "DealsHistory.mqh", which download
generalized and detailed data.

Let's start with the ShortReport.mqh file. This file contains functions and macros, the main one of which is the SaveReportToFile function.
First of all, let's consider the 'write' function which writes data to a file.

```
//+------------------------------------------------------------------+
//| File writer                                                      |
//+------------------------------------------------------------------+
void writer(string fileName,string headder,string row)
  {
   bool isFile=FileIsExist(fileName,FILE_COMMON); // Flag of whether the file exists
   int file_handle=FileOpen(fileName,FILE_READ|FILE_WRITE|FILE_CSV|FILE_COMMON|FILE_SHARE_WRITE|FILE_SHARE_READ); // Open file
   if(file_handle) // If the file has opened
     {
      FileSeek(file_handle,0,SEEK_END); // Move cursor to the file end
      if(!isFile) // If it is a newly created file, write a header
         FileWrite(file_handle,headder);
      FileWrite(file_handle,row); // Write message
      FileClose(file_handle); // Close the file
     }
  }
```

Data is written to the file sandbox Terminal/Common/Files. The idea of the function is to write data to a file by adding rows to it, that is why we
open the file, get its handle and move to the file end. If the file has
just been created, write the passed headers, otherwise ignore this
parameter.

As for the macro, it is only intended for the easy adding of robot parameters to the file.

```
#define WRITE_BOT_PARAM(fileName,param) writer(fileName,"",#param+";"+(string)param);
```

In this macro, we use the advantage of macros and form a string containing
the macro name and its value. A parameter variable is input into the function. Further details will be shown in the macro use example.

The main method SaveReportToFile is quite lengthy, that is why I only provide some code parts. The method creates an instance of the
CDealHistoryGetter class and receives an array with the accumulated trading history, in which one row indicates one deal.

```
DealDetales history[];
CDealHistoryGetter dealGetter(_comission_manager);
dealGetter.getDealsDetales(history,0,TimeCurrent());
```

Then it verifies that the history is not empty, creates the CReportCreator class instance and receives structures with the main
coefficients:

```
if(ArraySize(history)==0)
   return;

CReportCreator reportCreator(_comission_manager);
reportCreator.Create(history,0);

TotalResult totalResult;
reportCreator.GetTotalResult(totalResult);
PL_detales pl_detales;
reportCreator.GetPL_detales(pl_detales);
```

Then it saves historical data in a loop, by using the 'writer' function. At the end of the cycle, fields with the following coefficients and
values are added:

- PL
- Total trades
- Consecutive wins
- Consecutive Drawdowns
- Recovery factor
- Profit factor
- Payoff
- Drawdown by pl

```
writer(fileName,"","==========================================================================================================");
writer(fileName,"","PL;"+DoubleToString(totalResult.total.PL)+";");
int total_trades=pl_detales.total.profit.orders+pl_detales.total.drawdown.orders;
writer(fileName,"","Total trdes;"+IntegerToString(total_trades));
writer(fileName,"","Consecutive wins;"+IntegerToString(pl_detales.total.profit.dealsInARow));
writer(fileName,"","Consecutive DD;"+IntegerToString(pl_detales.total.drawdown.dealsInARow));
writer(fileName,"","Recovery factor;"+DoubleToString(totalResult.total.recoveryFactor)+";");
writer(fileName,"","Profit factor;"+DoubleToString(totalResult.total.profitFactor)+";");
double payoff=MathAbs(totalResult.total.averageProfit/totalResult.total.averageDD);
writer(fileName,"","Payoff;"+DoubleToString(payoff)+";");
writer(fileName,"","Drawdown by pl;"+DoubleToString(totalResult.total.maxDrawdown.byPL)+";");
```

The method operation completes here. Now let's consider an easy way to download history: let's add this feature to an Expert Advisor from the
standard delivery package, Experts/Examples/Moving Average/Moving Average.mq5. Firstly, we need to connect our file:

```
#include <History manager/ShortReport.mqh>
```

Then, add to inputs the variables which set custom commission and slippage:

```
input double custom_comission = 0; // Custom commission;
input int custom_shift = 0; // Custom shift;
```

If we want our commission and slippage to be set directively rather than conditionally (please see the CDealHistoryGetter class
description in the previous article), then before connecting the file we need to determine the ONLY\_CUSTOM\_COMISSION parameter as it is
shown below:

```
#define ONLY_CUSTOM_COMISSION
#include <History manager/ShortReport.mqh>
```

Then create the CCCM class sample, in the OnInit method add commission and slippage to this class instance, which stores commissions.

```
CCCM _comission_manager_;

...

int OnInit(void)
  {
   _comission_manager_.add(_Symbol,custom_comission,custom_shift);

...

  }
```

Then add the following code lines in the OnDeinit method:

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(MQLInfoInteger(MQL_TESTER)==1)
     {
      string file_name = __FILE__+" Report.csv";
      SaveReportToFile(file_name,&_comission_manager_);

      WRITE_BOT_PARAM(file_name,MaximumRisk);      // Maximum Risk in percentage
      WRITE_BOT_PARAM(file_name,DecreaseFactor);   // Decrease factor
      WRITE_BOT_PARAM(file_name,MovingPeriod);     // Moving Average period
      WRITE_BOT_PARAM(file_name,MovingShift);      // Moving Average shift
      WRITE_BOT_PARAM(file_name,custom_comission); // Custom commission
      WRITE_BOT_PARAM(file_name,custom_shift);     // Custom shift
     }
  }
```

This code performs a check after robot instance deletion: the condition
of whether it is running in the tester is checked. If the robot is running in the tester, a function
saving the robot trading history to a file entitled "Compiled\_file\_name Report.csv" will be called. After all the data that will be written to the
file, add 6 more lines for the input parameters of this file. Every time after launching the Expert Advisor in the Strategy Tester in testing
mode, we will receive a file with the description of deals performed by the EA. This file will be overwritten every time we start a new test. The
file will be stored in the file sandbox, under the Common/Files directory.

### Downloading trading history split by deals

Now let us view how to download a detailed trading report, i.e. a report in which all deals are grouped into positions. For this purpose, we will
use the DealsHistory.mqh file which already contains the ShortReport.mqh file connection. It means that by connecting the only
DealsHistory.mqh file we can use both methods at a time.

The file contains two functions. The first one is an ordinary function, which enables a nice summation:

```
void AddRow(string item, string &str)
  {
   str += (item + ";");
  }
```

The second function writes data to a file using the earlier considered 'writer' function. Its full implementation is shown below.

```
void WriteDetalesReport(string fileName,CCCM *_comission_manager)
  {

   if(FileIsExist(fileName,FILE_COMMON))
     {
      FileDelete(fileName,FILE_COMMON);
     }

   CDealHistoryGetter dealGetter(_comission_manager);

   DealKeeper deals[];
   dealGetter.getHistory(deals,0,TimeCurrent());

   int total= ArraySize(deals);

   string headder = "Asset;From;To;Deal DT (Unix seconds); Deal DT (Unix miliseconds);"+
                    "ENUM_DEAL_TYPE;ENUM_DEAL_ENTRY;ENUM_DEAL_REASON;Volume;Price;Comission;"+
                    "Profit;Symbol;Comment";

   for(int i=0; i<total; i++)
     {
      DealKeeper selected = deals[i];
      string asset = selected.symbol;
      datetime from = selected.DT_min;
      datetime to = selected.DT_max;

      for(int j=0; j<ArraySize(selected.deals); j++)
        {
         string row;
         AddRow(asset,row);
         AddRow((string)from,row);
         AddRow((string)to,row);

         AddRow((string)selected.deals[j].DT,row);
         AddRow((string)selected.deals[j].DT_msc,row);
         AddRow(EnumToString(selected.deals[j].type),row);
         AddRow(EnumToString(selected.deals[j].entry),row);
         AddRow(EnumToString(selected.deals[j].reason),row);
         AddRow((string)selected.deals[j].volume,row);
         AddRow((string)selected.deals[j].price,row);
         AddRow((string)selected.deals[j].comission,row);
         AddRow((string)selected.deals[j].profit,row);
         AddRow(selected.deals[j].symbol,row);
         AddRow(selected.deals[j].comment,row);

         writer(fileName,headder,row);

        }

      writer(fileName,headder,"");
     }

  }
```

After receiving trading data and creating the header, move on to writing the detailed trading report. For this purpose, implement a loop with a
nested loop: the main loop works with positions, and the nested
one loops through deals within these positions. After writing each new position (i.e. the series of deals which constitute the
position), the positions are separated using a space. This ensures
efficient reading of the resulting file. We don't need to make dramatic changes in order to add this feature into the robot, while we only need
to perform a call in OnDeinit:

```
void OnDeinit(const int reason)
  {
   if(MQLInfoInteger(MQL_TESTER)==1)
     {
      string file_name = __FILE__+" Report.csv";
      SaveReportToFile(file_name,&_comission_manager_);

      WRITE_BOT_PARAM(file_name,MaximumRisk);      // Maximum Risk in percentage
      WRITE_BOT_PARAM(file_name,DecreaseFactor);   // Descrease factor
      WRITE_BOT_PARAM(file_name,MovingPeriod);     // Moving Average period
      WRITE_BOT_PARAM(file_name,MovingShift);      // Moving Average shift
      WRITE_BOT_PARAM(file_name,custom_comission); // Custom commission;
      WRITE_BOT_PARAM(file_name,custom_shift);     // Custom shift;

      WriteDetalesReport(__FILE__+" Deals Report.csv", &_comission_manager_);
     }
  }
```

To demonstrate in detail how the data downloading is performed, here is an empty EA template with the added methods for downloading the
report:

```
//+------------------------------------------------------------------+
//|                                                         Test.mq5 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#define ONLY_CUSTOM_COMISSION
#include <History manager/DealsHistory.mqh>

input double custom_comission   = 0;       // Custom commission;
input int    custom_shift       = 0;       // Custom shift;

CCCM _comission_manager_;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   _comission_manager_.add(_Symbol,custom_comission,custom_shift);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(MQLInfoInteger(MQL_TESTER)==1)
     {
      string arr[];
      StringSplit(__FILE__,'.',arr);
      string file_name = arr[0]+" Report.csv";
      SaveReportToFile(file_name,&_comission_manager_);
      WRITE_BOT_PARAM(file_name,custom_comission); // Custom commission;
      WRITE_BOT_PARAM(file_name,custom_shift);     // Custom shift;

      WriteDetalesReport(arr[0]+" Deals Report.csv", &_comission_manager_);
     }
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

By adding any desired logic to the above template, we can enable the generation of a trading report after EA testing completion in the
Strategy Tester.

### Wrapper for DLL creating the accumulated trading history

The first [article](https://www.mql5.com/en/articles/7290) in this series was devoted to the creation of a DLL in the C#
language for working with optimization reports. Since the most convenient format for the continuous walk-forward optimization is XML, we
have created a DLL which can read, write and sort the resulting report. In the Expert Advisor we only need data writing functionality.
However, since operations with pure functions are less convenient than objects, a wrapper class for the data downloading function has been
created. This object is located in the XmlHistoryWriter.mqh file, it is called СXmlHistoryWriter. In addition to the object, it defines
the structure of the EA parameters. This structure will be used when passing the list of EA parameters to the object. Let's consider all the
implementation details.

To enable the creation of the optimization report, connect the ReportCreator.mqh file. In order to use the static methods of the class from
the DLL described in the first article, let's import it (the library must be already available under the MQL5/Libraries directory).

```
#include "ReportCreator.mqh"
#import "ReportManager.dll"
```

After adding the required links, ensure the convenient adding of robot parameters to the parameter collection, which is then passed to the
target class.

```
struct BotParams
  {
   string            name,value;
  };

#define ADD_TO_ARR(arr, value) \
{\
   int s = ArraySize(arr);\
   ArrayResize(arr,s+1,s+1);\
   arr[s] = value;\
}

#define APPEND_BOT_PARAM(Var,BotParamArr) \
{\
   BotParams param;\
   param.name = #Var;\
   param.value = (string)Var;\
   \
   ADD_TO_ARR(BotParamArr,param);\
}
```

Since we are going to operate with a collection of objects, it will be easier to work with dynamic arrays. The ADD\_TO\_ARR
macro was created for the convenient addition of elements to the dynamic array. The macro changes the collection size and then adds the passed item
to it. The macro provides a universal solution. And thus, now we can quickly add values of any type to the dynamic array.

The next macro directly works with the parameters. This
macro creates a BotParams structure instance and adds it to the array by inputting only the array, to which the parameter
description should be added and a variable storing this parameter. The macro will assign an appropriate name
to the parameter based on the variable name, and it will assign the
parameter value converted to a string format.

The string format ensures the proper correspondence of settings in \*.set files and of data which are being saved to the \*.xml file. As already
described in previous articles, set files store EA input parameters in the key-value format, in which the variable name as in the code is
accepted as the key, and the value assigned to the input parameter is accepted as the value. All enumerations (num) must be specified as an int
type, and not as the result of the EnumToString() function. The described macro converts all parameters to the required string type, while
all enumerations are first converted to int and then to the required string format.

We have also declared a function, which allows copying the array of robot parameters to another array.

```
void CopyBotParams(BotParams &dest[], const BotParams &src[])
  {
   int total = ArraySize(src);
   for(int i=0; i<total; i++)
     {
      ADD_TO_ARR(dest,src[i]);
     }
  }
```

We need this because the standard ArrayCopy function does not work with the array of structures.

The wrapper class is declared as follows:

```
class CXmlHistoryWriter
  {
private:
   const string      _path_to_file,_mutex_name;
   CReportCreator    _report_manager;

   string            get_path_to_expert();//

   void              append_bot_params(const BotParams  &params[]);//
   void              append_main_coef(PL_detales &pl_detales,
                                      TotalResult &totalResult);//
   double            get_average_coef(CoefChartType type);
   void              insert_day(PLDrawdown &day,ENUM_DAY_OF_WEEK day);//
   void              append_days_pl();//

public:
                     CXmlHistoryWriter(string file_name,string mutex_name,
                     CCCM *_comission_manager);//
                     CXmlHistoryWriter(string mutex_name,CCCM *_comission_manager);
                    ~CXmlHistoryWriter(void) {_report_manager.Clear();} //

   void              Write(const BotParams &params[],datetime start_test,datetime end_test);//
  };
```

Two constant string fields are declared for writing to a file:

- \_path\_to\_file
- \_mutex\_name

The first field contains path to a file, to which data will be written. The second field contains the name of the used mutex. Implementation of
the named mutex is provided in C# DLL. We need this mutex because the optimization process will be implemented in different threads, on
different cores and in different processes (one launch of the robot = one process). Therefore, a situation can occur in which two
optimizations have completed, and two or more processes simultaneously try to write the result to the same file, which is unacceptable. To
eliminate this situation, we use a synchronization object based on the operating system core, i.e. a named mutex.

The CReportCreator class instance is needed as a field because other functions will access this object, and thus it would be illogical to
create it each time anew. Now let's view the implementation of each method.

Let's begin with the class constructor.

```
CXmlHistoryWriter::CXmlHistoryWriter(string file_name,
                                     string mutex_name,
                                     CCCM *_comission_manager) : _mutex_name(mutex_name),
   _path_to_file(TerminalInfoString(TERMINAL_COMMONDATA_PATH)+"\\"+file_name),
   _report_manager(_comission_manager)
  {
  }
CXmlHistoryWriter::CXmlHistoryWriter(string mutex_name,
                                     CCCM *_comission_manager) : _mutex_name(mutex_name),
   _path_to_file(TerminalInfoString(TERMINAL_COMMONDATA_PATH)+"\\"+MQLInfoString(MQL_PROGRAM_NAME)+"_"+"Report.xml"),
   _report_manager(_comission_manager)
  {
  }
```

The class contains two constructors. Pay attention to the second constructor, which sets
the name of the file in which the optimizations report is stored. In the auto optimizer, which will be considered in the next article,
it will be possible to set custom optimization managers. The default hard coded manager already has an implemented agreement on the naming
of report files generated by the robot. Thus, the second constructor sets this agreement. According to it, the file name must start with the
EA name, followed by the underscore and the postfix "\_Report.xml". Although a DLL can write a report file anywhere on the PC, to preserve the
information that the file belongs to the terminal operation, we will always store it under the
Common directory from the MetaTrader 5 sandbox.

The method receiving the path to the Expert Advisor:

```
string CXmlHistoryWriter::get_path_to_expert(void)
  {
   string arr[];
   StringSplit(MQLInfoString(MQL_PROGRAM_PATH),'\\',arr);
   string relative_dir=NULL;

   int total= ArraySize(arr);
   bool save= false;
   for(int i=0; i<total; i++)
     {
      if(save)
        {
         if(relative_dir== NULL)
            relative_dir=arr[i];
         else
            relative_dir+="\\"+arr[i];
        }

      if(StringCompare("Experts",arr[i])==0)
         save=true;
     }

   return relative_dir;
  }
```

The EA path is needed for the automated launch of the selected Expert Advisor. For this purpose, its path should be specified in the ini file
which is passed at terminal start. The path should be specified relative to the Experts folder and not the full path, which is obtained by the function which
receives the path to the current EA. Therefore, we first need to split the obtained path into components, in which the separator is a slash.
Then, in a loop, search for the Experts directory starting with the very
first directory. Once it is found, form path to the robot starting
with the next directory (or the EA file if it is located directly in the root of the desired directory).

The append\_bot\_params method:

This method is a wrapper for an imported method with the same name. Its implementation is as follows:

```
void CXmlHistoryWriter::append_bot_params(const BotParams &params[])
  {

   int total= ArraySize(params);
   for(int i=0; i<total; i++)
     {
      ReportWriter::AppendBotParam(params[i].name,params[i].value);
     }
  }
```

The earlier mentioned array of EA parameters is input into this method. Then, in a loop, call the imported method from our DLL for each of the EA
parameters.

The implementation of the append\_main\_coef method is easy, so we will not consider it here. It accepts structures from the CReportCreator
class as input parameters.

The get\_average\_coef method is intended for calculating average values of coefficients by a simple MA method based on the passed coefficient charts. It is used
for calculating the average profit factor and the average recovery factor.

The insert\_day method is an easily called wrapper for the imported ReportWriter::AppendDay method. The append\_days\_pl method already
uses the earlier mentioned wrapper.

Among all these wrapper methods, there is one public method which acts as the main one — it is the 'Write' method, which triggers the whole
mechanism for saving the data.

```
void CXmlHistoryWriter::Write(const BotParams &params[],datetime start_test,datetime end_test)
  {
   if(!_report_manager.Create())
     {
      Print("##################################");
      Print("Can`t create report:");
      Print("###################################");
      return;
     }
   TotalResult totalResult;
   _report_manager.GetTotalResult(totalResult);
   PL_detales pl_detales;
   _report_manager.GetPL_detales(pl_detales);

   append_bot_params(params);
   append_main_coef(pl_detales,totalResult);

   ReportWriter::AppendVaR(totalResult.total.VaR_absolute.VAR_90,
                           totalResult.total.VaR_absolute.VAR_95,
                           totalResult.total.VaR_absolute.VAR_99,
                           totalResult.total.VaR_absolute.Mx,
                           totalResult.total.VaR_absolute.Std);

   ReportWriter::AppendMaxPLDD(pl_detales.total.profit.totalResult,
                               pl_detales.total.drawdown.totalResult,
                               pl_detales.total.profit.orders,
                               pl_detales.total.drawdown.orders,
                               pl_detales.total.profit.dealsInARow,
                               pl_detales.total.drawdown.dealsInARow);
   append_days_pl();

   string error_msg=ReportWriter::MutexWriter(_mutex_name,get_path_to_expert(),AccountInfoString(ACCOUNT_CURRENCY),
                    _report_manager.GetBalance(),
                    (int)AccountInfoInteger(ACCOUNT_LEVERAGE),
                    _path_to_file,
                    _Symbol,(int)Period(),
                    start_test,
                    end_test);
   if(StringCompare(error_msg,"")!=0)
     {
      Print("##################################");
      Print("Error while creating (*.xml) report file:");
      Print("_________________________________________");
      Print(error_msg);
      Print("###################################");
     }
  }
```

If an attempt to create a report fails, an appropriate entry is added to logs. If the report is successfully created, we move on to receive the
desired coefficients and then call the above mentioned methods one by one. According to the first article, these methods add the requested
parameters to a C# class. Then a method writing data to a file is
called. If writing fails, error\_msg will contain the text of the error,
which is written to tester logs.

The resulting class can generate the trading report, as well as write data to a file upon the call of the 'Write' method. However, I want to
simplify the process further, so that we only deal with input parameters and nothing else. The following class has been created for this
purpose.

The CAutoUpLoader class automatically generates a trading report upon testing completion. It is located in the AutoLoader.mqh file. For
the class operation, we should add a link to the previously described class, which generates the report in XML format.

```
#include <History manager/XmlHistoryWriter.mqh>
```

The class signature is simple:

```
class CAutoUploader
  {
private:

   datetime          From,Till;
   CCCM              *comission_manager;
   BotParams         params[];
   string            mutexName;

public:
                     CAutoUploader(CCCM *comission_manager, string mutexName, BotParams &params[]);
   virtual          ~CAutoUploader(void);

   virtual void      OnTick();

  };
```

The class has an overload OnTick method, as well as a virtual destructor. This ensures that the class can be applied both using aggregation and
using inheritance. Here is what I mean. The purpose of this class is to overwrite the testing completion time at each tick, as well as to
remember the testing start time. These two parameters are required for using the previous described object. There are several approaches
to its application: we can either instantiate this object somewhere in the robot class (if the robot is developed using OOP), or in the global
scope — this solution can be used for the C-like programming.

After that, the OnTick() method of the class instance is called in this function. After destroying the class object, the trading report will be
unloaded in its destructor. The second way to apply the class is to inherit an EA class from it. The virtual destructor and the OnTick()
overload method are created for this purpose. As a result of applying the second method, we will work directly with the Expert Advisor. The
implementation of this class is simple, as it delegates operation to the CXmlHistoryWriter class:

```
void CAutoUploader::OnTick(void)
  {
   if(MQLInfoInteger(MQL_OPTIMIZATION)==1 ||
      MQLInfoInteger(MQL_TESTER)==1)
     {
      if(From == 0)
         From = iTime(_Symbol,PERIOD_M1,0);
      Till=iTime(_Symbol,PERIOD_M1,0);
     }
  }
CAutoUploader::CAutoUploader(CCCM *_comission_manager,string _mutexName,BotParams &_params[]) : comission_manager(_comission_manager),
   mutexName(_mutexName)
  {
   CopyBotParams(params,_params);
  }
CAutoUploader::~CAutoUploader(void)
  {
   if(MQLInfoInteger(MQL_OPTIMIZATION)==1 ||
      MQLInfoInteger(MQL_TESTER)==1)
     {
      CXmlHistoryWriter historyWriter(mutexName,
                                      comission_manager);

      historyWriter.Write(params,From,Till);
     }
  }
```

Let's add the described functionality to our EA template:

```
//+------------------------------------------------------------------+
//|                                                         Test.mq5 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#define ONLY_CUSTOM_COMISSION
#include <History manager/DealsHistory.mqh>
#include <History manager/AutoLoader.mqh>

class CRobot;

input double custom_comission   = 0;       // Custom commission;
input int    custom_shift       = 0;       // Custom shift;

CCCM _comission_manager_;
CRobot *bot;
const string my_mutex = "My Mutex Name for this expert";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   _comission_manager_.add(_Symbol,custom_comission,custom_shift);

   BotParams params[];

   APPEND_BOT_PARAM(custom_comission,params);
   APPEND_BOT_PARAM(custom_shift,params);

   bot = new CRobot(&_comission_manager_,my_mutex,params);

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(MQLInfoInteger(MQL_TESTER)==1)
     {
      string arr[];
      StringSplit(__FILE__,'.',arr);
      string file_name = arr[0]+" Report.csv";
      SaveReportToFile(file_name,&_comission_manager_);
      WRITE_BOT_PARAM(file_name,custom_comission); // Custom commission;
      WRITE_BOT_PARAM(file_name,custom_shift);     // Custom shift;

      WriteDetalesReport(arr[0]+" Deals Report.csv", &_comission_manager_);
     }

   delete bot;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   bot.OnTick();
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Main Robot class                                                 |
//+------------------------------------------------------------------+
class CRobot : CAutoUploader
  {
public:
                     CRobot(CCCM *_comission_manager, string _mutexName, BotParams &_params[]) : CAutoUploader(_comission_manager,_mutexName,_params)
     {}

   void              OnTick() override;
  };

//+------------------------------------------------------------------+
//| Robot logic triggering method                                    |
//+------------------------------------------------------------------+
void CRobot::OnTick(void)
  {
   CAutoUploader::OnTick();

   Print("This should be the robot logic start");
  }
//+------------------------------------------------------------------+
```

Thus, the first thing to do is to add a reference to a
file where the wrapper class for automatic report downloading to XML is stored. The
robot class is predetermined, because it is easier to implement and describe it at the end of the project. Usually I create algorithms
as MQL5 projects, while this is much more convenient than the one-page approach, because the class with the robot and related classes are
divided into files. However, for the convenience, everything was placed in one file.

Then the
class is described. In this example, it is an empty class with one overloaded OnTick method. Thus, the second CAutoUploader
application way, i.e. the inheritance is used. Please note that in the overloaded OnTick method, it is necessary to explicitly
call the OnTick method of the base class so that the calculation of dates does not stop. This is essential for the entire operation of
the auto optimizer.

The next step is to create a pointer to the class with the robot,
because it is more convenient to populate it from the OnInit method rather than from the global scope. Also, a
variable storing the mutex name is created.

The robot is instantiated in the OnInit method, and it is deleted
in OnDeinit. To ensure passing of a new tick arrival callback to the robot, call
the overloaded OnTick() method at the robot pointer. Once this has been done, write the robot in the CRobot class.

The variants of report downloading by aggregation or by creating a CAutoUpLoader instance in global scope are similar. Should you have any
questions, please feel free to contact me.

Thus, by using this robot template or by adding the appropriate calls to your existing algorithms, you can use them together with the auto
optimizer, which will be discussed in the next article.

### Conclusion

In the first article, the mechanism of operation with XML report files and the creation of the file structure was analyzed.
Creation of reports was considered in the second article. The report generation mechanism was examined, starting with the history
downloading object and ending with the objects generating the report. When studying the objects which are involved in the report creation
process, the calculation part was analyzed in detail. The article also contained the main coefficient formulas, as well as the description
of possible calculation issues.

As it was mentioned in the introduction to this article, the objects described in this part serve as a bridge between the data downloading
mechanism and the report generation mechanism. In addition to the functions which save trading report files, the article contains the
description of classes participating in XML report unloading, as well as the description of robot templates that can automatically use
these features. The article also described how to add the created features to an existing algorithm. It means that auto optimizer users can
optimize both old and new algorithms.

Two folders are available in the attached archive. Unzip both of them to MQL/Include directory. The ReportManager.dll library must be added
to MQL5/Libraries. You can downloaded from the previous [article](https://www.mql5.com/en/articles/7290).

The following files are included in the attachment:

1. CustomGeneric

   - GenericSorter.mqh
   - ICustomComparer.mqh

3. History manager
   - AutoLoader.mqh
   - CustomComissionManager.mqh
   - DealHistoryGetter.mqh
   - DealsHistory.mqh
   - ReportCreator.mqh
   - ShortReport.mqh
   - XmlHistoryWriter

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7490](https://www.mql5.com/ru/articles/7490)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7490.zip "Download all attachments in the single ZIP archive")

[Include.zip](https://www.mql5.com/en/articles/download/7490/include.zip "Download Include.zip")(29.82 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Continuous walk-forward optimization (Part 8): Program improvements and fixes](https://www.mql5.com/en/articles/7891)
- [Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://www.mql5.com/en/articles/7747)
- [Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://www.mql5.com/en/articles/7718)
- [Continuous Walk-Forward Optimization (Part 5): Auto Optimizer project overview and creation of a GUI](https://www.mql5.com/en/articles/7583)
- [Continuous Walk-Forward Optimization (Part 4): Optimization Manager (Auto Optimizer)](https://www.mql5.com/en/articles/7538)
- [Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://www.mql5.com/en/articles/7452)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/336152)**
(11)


![Andrey Azatskiy](https://c.mql5.com/avatar/2018/6/5B127D58-708F.jpg)

**[Andrey Azatskiy](https://www.mql5.com/en/users/andreykrivcov)**
\|
2 Feb 2020 at 22:25

**Kristian Kafarov:**

It didn't even come to robots, I just try to call your panel from the first two articles by running the OptimisationManagerExtention Expert Advisor. After that the terminal crashes.

On purpose now recompiled and ran the old [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it's not accurate ") from scratch. Everything worked for me. So I cannot reproduce the error.

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
2 Feb 2020 at 23:50

**Kristian Kafarov:**

Most of all, I want to implement cross-validation to beat the story into K parts, each of them dumped in turn, optimise on the remaining ones, then check on the dumped one, and so K times.

Selected will not work in general case. You need to set two input parameters in your TS that define a non-trading (thrown out) interval. Then it is real.

For the general case you can [create a custom symbol](https://www.mql5.com/en/docs/customsymbols/customsymbolcreate "MQL5 documentation: CustomSymbolCreate function"), which is obtained from the original one by throwing out the interval.

![Reset_index](https://c.mql5.com/avatar/avatar_na2.png)

**[Reset\_index](https://www.mql5.com/en/users/xaba3abr)**
\|
3 Feb 2020 at 12:08

**fxsaber:**

The highlighted one will not work in the general case. You need to set two input parameters in your TS that define a non-trading (thrown out) interval. Then it is realistic.

For the general case you can create a custom symbol, which is obtained from the original one by throwing out the interval.

That's exactly how I was going to do it. Only one parameter is enough, because the division goes into equal parts. The parameter specifies the number of the segment to be discarded. Well, you can also add the parameter "number of parts".

With Andrew's tools, you can give the master terminal a task to perform k optimisations, each of which will have its own parameter "number of validation section". Then, however, you will have to write an add-on to bring the statistics together.

Everything would be a hundred times simpler if the tester had a possibility to forcibly enumerate some parameters completely during genetics. Then the opta results can be analysed by dividing them by the parameter "plot number".

Another option is the [OnTesterDeinit() function](https://www.mql5.com/en/docs/basis/function/events#oninit "MQL5 Documentation: Event Handling Functions"). I have already implemented a full-fledged WFO in it, and you can easily do cross-validation by any criterion there. But it will be "correct" only in case of a full search, because it is done by enumerating the frames of the whole testing section. A full enumeration is unrealistic in most cases. And if we run genetics, the set of frames will be unfair, because in the process of opta it selects the results also by the sections we want to make test sections. Although how much real damage this will do is a question. If the ratio of the length of the test site to the total length is small, genetics should still have a sufficient number of variants where the test site turns out to suck. And after all such a common variant, it is possible to leave one more site, which did not participate in it, and check the result on it.

![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
3 Feb 2020 at 21:33

**Kristian Kafarov:**

That's exactly what I was going for. Only one parameter is enough, because it is divided into equal parts. The parameter specifies the number of the section to be discarded. You can also add the parameter "number of parts".

With Andrew's tools, you can give the master terminal a task to perform k optimisations, each of which will have its own parameter "number of validation section". Then, however, you will have to write an add-on to bring the statistics together.

Everything would be a hundred times simpler if the tester had a possibility to forcibly enumerate some parameters completely during genetics. Then the opta results could be analysed by dividing them by the parameter "plot number".

There is also [fxsaber tool](https://www.mql5.com/en/code/27755), it will help with the rest.

![Reset_index](https://c.mql5.com/avatar/avatar_na2.png)

**[Reset\_index](https://www.mql5.com/en/users/xaba3abr)**
\|
4 Feb 2020 at 09:25

**Andrey Khatimlianskii:**

There is also a [fxsaber tool](https://www.mql5.com/en/code/27755), it will help with the rest.

Awesome, fxsaber did exactly what I needed. Thanks for the link!

![Econometric approach to finding market patterns: Autocorrelation, Heat Maps and Scatter Plots](https://c.mql5.com/2/37/jlp_0d3zw11j.png)[Econometric approach to finding market patterns: Autocorrelation, Heat Maps and Scatter Plots](https://www.mql5.com/en/articles/5451)

The article presents an extended study of seasonal characteristics: autocorrelation heat maps and scatter plots. The purpose of the article is to show that "market memory" is of seasonal nature, which is expressed through maximized correlation of increments of arbitrary order.

![Library for easy and quick development of MetaTrader programs (part XXX): Pending trading requests - managing request objects](https://c.mql5.com/2/37/MQL5-avatar-doeasy__18.png)[Library for easy and quick development of MetaTrader programs (part XXX): Pending trading requests - managing request objects](https://www.mql5.com/en/articles/7481)

In the previous article, we have created the classes of pending request objects corresponding to the general concept of library objects. This time, we are going to deal with the class allowing the management of pending request objects.

![Multicurrency monitoring of trading signals (Part 1): Developing the application structure](https://c.mql5.com/2/37/Article_Logo__2.png)[Multicurrency monitoring of trading signals (Part 1): Developing the application structure](https://www.mql5.com/en/articles/7417)

In this article, we will discuss the idea of creating a multicurrency monitor of trading signals and will develop a future application structure along with its prototype, as well as create its framework for further operation. The article presents a step-by-step creation of a flexible multicurrency application which will enable the generation of trading signals and which will assist traders in finding the desired signals.

![Library for easy and quick development of MetaTrader programs (part XXIX): Pending trading requests - request object classes](https://c.mql5.com/2/37/MQL5-avatar-doeasy__17.png)[Library for easy and quick development of MetaTrader programs (part XXIX): Pending trading requests - request object classes](https://www.mql5.com/en/articles/7454)

In the previous articles, we checked the concept of pending trading requests. A pending request is, in fact, a common trading order executed by a certain condition. In this article, we are going to create full-fledged classes of pending request objects — a base request object and its descendants.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/7490&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071687525963082901)

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