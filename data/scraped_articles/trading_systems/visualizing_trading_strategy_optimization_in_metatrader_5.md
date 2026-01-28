---
title: Visualizing trading strategy optimization in MetaTrader 5
url: https://www.mql5.com/en/articles/4395
categories: Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:38:35.088774
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/4395&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049273191156787303)

MetaTrader 5 / Tester


### Contents

- [Introduction](https://www.mql5.com/en/articles/4395#para1)
- [Developing the graphical interface](https://www.mql5.com/en/articles/4395#para2)
- [Developing the class for working with frame data](https://www.mql5.com/en/articles/4395#para3)
- [Working with optimization data in the application class](https://www.mql5.com/en/articles/4395#para4)
- [Displaying the obtained results](https://www.mql5.com/en/articles/4395#para5)
- [Conclusion](https://www.mql5.com/en/articles/4395#para6)

### Introduction

When developing trading algorithms, it is useful to view test results while optimizing parameters. However, a single graph on the **Optimization Graph** tab may be insufficient for assessing a trading strategy efficiency. It would be much better to view balance curves of multiple tests simultaneously being able to analyze them even after the optimization. We have already examined such an application in the article ["Visualize a strategy in the MetaTrader 5 tester"](https://www.mql5.com/en/articles/403). However, many new opportunities have appeared since then. Therefore, it is now possible to implement a similar but much more powerful application.

The article implements an MQL application with a graphical interface for extended visualization of the optimization process. The graphical interface applies [the last version of EasyAndFast library](https://www.mql5.com/en/code/19703). Many MQL community users may ask why they need graphical interfaces in MQL applications. This article shows their potential uses. It also may be useful for those applying the library in their work.

### Developing the graphical interface

Here I will briefly describe developing the graphical interface. If you have already mastered **EasyAndFast** library, you will be able to quickly understand how to use it and evaluate how easy it is to develop the graphical interface for your MQL application.

First, let's describe the general structure of the developed application. **Program.mqh** file is to contain **CProgram** application class. This base class should be connected to the graphical library engine.

```
//+------------------------------------------------------------------+
//|                                                      Program.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
//--- Library class for creating the graphical interface
#include <EasyAndFastGUI\WndEvents.mqh>
//+------------------------------------------------------------------+
//| Class for developing the application                             |
//+------------------------------------------------------------------+
class CProgram : public CWndEvents
  {
  };
```

**EasyAndFast** library is displayed in a single block (Library GUI) in order not to clutter up the image. You can see it in full on the [library page](https://www.mql5.com/en/code/19703).

![Fig. 1. Including the library for creating GUI](https://c.mql5.com/2/31/001.png)

Fig. 1. Including the library for creating GUI

Similar methods should be created in **CProgram** class to connect with the MQL program's main functions. We will need the methods from **OnTesterXXX**() category to work with frames.

```
class CProgram : public CWndEvents
  {
public:
   //--- Initialization/deinitialization
   bool              OnInitEvent(void);
   void              OnDeinitEvent(const int reason);
   //--- "New tick" event handler
   void              OnTickEvent(void);
   //--- Trading event handler
   void              OnTradeEvent(void);
   //--- Timer
   void              OnTimerEvent(void);
   //--- Tester
   double            OnTesterEvent(void);
   void              OnTesterPassEvent(void);
   void              OnTesterInitEvent(void);
   void              OnTesterDeinitEvent(void);
  };
```

In this case, the methods should be called the following way in the application's main file:

```
//--- Include application class
#include "Program.mqh"
CProgram program;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//--- Initialize program
   if(!program.OnInitEvent())
     {
      ::Print(__FUNCTION__," > Failed to initialize!");
      return(INIT_FAILED);
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) { program.OnDeinitEvent(reason); }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(void) { program.OnTickEvent(); }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer(void) { program.OnTimerEvent(); }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
   { program.ChartEvent(id,lparam,dparam,sparam); }
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester(void) { return(program.OnTesterEvent()); }
//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
void OnTesterInit(void) { program.OnTesterInitEvent(); }
//+------------------------------------------------------------------+
//| TesterPass function                                              |
//+------------------------------------------------------------------+
void OnTesterPass(void) { program.OnTesterPassEvent(); }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit(void) { program.OnTesterDeinitEvent(); }
//+------------------------------------------------------------------+
```

Thus, the application workpiece is ready for developing the graphical interface. The main work is conducted in the **CProgram** class. All files necessary for work are included to **Program.mqh.**

Now let's define the contents of the graphical interface. List all the elements to be created.

- Form for controls.
- Field for specifying the amount of balances to be displayed on the graph.
- Field for adjusting the speed of repeated display of optimization results.
- Button for launching a repeated display.
- Result statistics table.
- Table for displaying the EA's external parameters.
- Balance curve graph.
- Optimization results graph.
- Status bar for displaying additional summary information.
- Progress bar showing a percentage of displayed results from the total amount when re-scrolling.

Below are declarations of control element class instances and their creation methods (see the code listing below). The codes of the methods are put into a separate file — **CreateFrameModeGUI.mqh**, which is associated with **CProgram** class file. As the code of the developed application grows, the method of distribution by individual files becomes more relevant making it easier to navigate the project.

```
class CProgram : public CWndEvents
  {
private:
   //--- Window
   CWindow           m_window1;
   //--- Status bar
   CStatusBar        m_status_bar;
   //--- Input fields
   CTextEdit         m_curves_total;
   CTextEdit         m_sleep_ms;
   //--- Buttons
   CButton           m_reply_frames;
   //--- Tables
   CTable            m_table_stat;
   CTable            m_table_param;
   //--- Graphs
   CGraph            m_graph1;
   CGraph            m_graph2;
   //--- Progress bar
   CProgressBar      m_progress_bar;
   //---
public:
   //--- Create the graphical interface for working with frames in optimization mode
   bool              CreateFrameModeGUI(void);
   //---
private:
   //--- Form
   bool              CreateWindow(const string text);
   //--- Status bar
   bool              CreateStatusBar(const int x_gap,const int y_gap);
   //--- Tables
   bool              CreateTableStat(const int x_gap,const int y_gap);
   bool              CreateTableParam(const int x_gap,const int y_gap);
   //--- Input fields
   bool              CreateCurvesTotal(const int x_gap,const int y_gap,const string text);
   bool              CreateSleep(const int x_gap,const int y_gap,const string text);
   //--- Buttons
   bool              CreateReplyFrames(const int x_gap,const int y_gap,const string text);
   //--- Graphs
   bool              CreateGraph1(const int x_gap,const int y_gap);
   bool              CreateGraph2(const int x_gap,const int y_gap);
   //--- Progress bar
   bool              CreateProgressBar(const int x_gap,const int y_gap,const string text);
  };
//+------------------------------------------------------------------+
//| Methods for creating control elements                            |
//+------------------------------------------------------------------+
#include "CreateFrameModeGUI.mqh"
//+------------------------------------------------------------------+
```

Let's enable including the file to be connected with in **CreateFrameModeGUI.mqh** as well. We will show here only one main method for creating the app's graphical interface as an example:

```
//+------------------------------------------------------------------+
//|                                           CreateFrameModeGUI.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#include "Program.mqh"
//+------------------------------------------------------------------+
//| Create the graphical interface                                   |
//| for analyzing optimization results and working with frames       |
//+------------------------------------------------------------------+
bool CProgram::CreateFrameModeGUI(void)
  {
//--- Create the interface only in the mode for working with optimization frames
   if(!::MQLInfoInteger(MQL_FRAME_MODE))
      return(false);
//--- Create the form for control elements
   if(!CreateWindow("Frame mode"))
      return(false);
//--- Create control elements
   if(!CreateStatusBar(1,23))
      return(false);
   if(!CreateCurvesTotal(7,25,"Curves total:"))
      return(false);
   if(!CreateSleep(145,25,"Sleep:"))
      return(false);
   if(!CreateReplyFrames(255,25,"Replay frames"))
      return(false);
   if(!CreateTableStat(2,50))
      return(false);
   if(!CreateTableParam(2,212))
      return(false);
   if(!CreateGraph1(200,50))
      return(false);
   if(!CreateGraph2(200,159))
      return(false);
//--- Progress bar
   if(!CreateProgressBar(2,3,"Processing..."))
      return(false);
//--- Complete GUI creation
   CWndEvents::CompletedGUI();
   return(true);
  }
...
```

Connection between the files belonging to one class is shown as the two-sided yellow arrow:

![Fig. 2. Dividing the project into several files](https://c.mql5.com/2/31/002.png)

Fig. 2. Dividing the project into several files

### Developing the class for working with frame data

Let's write a separate class **CFrameGenerator** to work with frames. The class is to be contained in **FrameGenerator.mqh** that should be included to **Program.mqh**. As an example, I will demonstrate two options for receiving these frames for display in graphical interface elements.

- In the first case, in order to display frames on graph objects, pointers to these objects are passed to class methods.
- In the second case, we receive frame data for filling in the tables from other categories using special methods.

You decide, which of these options is to be left as the main one.

**EasyAndFast** library applies **CGraphic** class from the standard library to visualize data. Let's include it to **FrameGenerator.mqh** to access its methods.

```
//+------------------------------------------------------------------+
//|                                               FrameGenerator.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#include <Graphics\Graphic.mqh>
//+------------------------------------------------------------------+
//| Class for receiving optimization results                         |
//+------------------------------------------------------------------+
class CFrameGenerator
  {
  };
```

The program arrangement now looks as follows:

![Fig. 3. Connecting to class projects for work](https://c.mql5.com/2/31/003.png)

Fig. 3. Connecting to class projects for work

Now, let's see how **CFrameGenerator** class is organized. It also needs methods for processing strategy tester events (see the code listing below). They are to be called in similar class methods of the program we develop — **CProgram**. Pointers to graph objects the current optimization process is displayed at are passed to **CFrameGenerator::OnTesterInitEvent**() method.

- The first graph ( **graph\_balance**) displays the specified number of the last series of the optimization result balances.
- The second graph ( **graph\_result**) displays the overall optimization results.

```
class CFrameGenerator
  {
private:
   //--- Graph pointers for data visualization
   CGraphic         *m_graph_balance;
   CGraphic         *m_graph_results;
   //---
public:
   //--- Strategy tester event handlers
   void              OnTesterEvent(const double on_tester_value);
   void              OnTesterInitEvent(CGraphic *graph_balance,CGraphic *graph_result);
   void              OnTesterDeinitEvent(void);
   bool              OnTesterPassEvent(void);
  };
//+------------------------------------------------------------------+
//| Should be called in OnTesterInit() handler                       |
//+------------------------------------------------------------------+
void CFrameGenerator::OnTesterInitEvent(CGraphic *graph_balance,CGraphic *graph_results)
  {
   m_graph_balance =graph_balance;
   m_graph_results =graph_results;
  }
```

On both graphs, positive results are displayed in green, while negative ones are shown in red.

In **CFrameGenerator::OnTesterEvent**() method, we receive the test result balance and statistical parameters. These data are passed to a frame using **CFrameGenerator::GetBalanceData**() and **CFrameGenerator::GetStatData**() methods. **CFrameGenerator::GetBalanceData**() method receives the entire test history and sums up all **in**-/ **inout** trades. The obtained result is saved to **m\_balance**\[\] array step by step. In turn, this array is a member of **CFrameGenerator** class.

The dynamic array to be sent to a frame is passed to **CFrameGenerator::GetStatData**() method. Its size is to match the size of the array for the previously received result balance. Besides, a number of elements we receive statistical parameters to is added.

```
//--- Number of statistical parameters
#define STAT_TOTAL 7
//+------------------------------------------------------------------+
//| Class for working with optimization results                      |
//+------------------------------------------------------------------+
class CFrameGenerator
  {
private:
   //--- Result balance
   double            m_balance[];
   //---
private:
   //--- Receive balance data
   int               GetBalanceData(void);
   //--- Receive statistical data
   void              GetStatData(double &dst_array[],double on_tester_value);
  };
//+------------------------------------------------------------------+
//| Get balance data                                                 |
//+------------------------------------------------------------------+
int CFrameGenerator::GetBalanceData(void)
  {
   int    data_count      =0;
   double balance_current =0;
//--- Request all trading history
   ::HistorySelect(0,LONG_MAX);
   uint deals_total=::HistoryDealsTotal();
//--- Gather data on trades
   for(uint i=0; i<deals_total; i++)
     {
      //--- Receive a ticket
      ulong ticket=::HistoryDealGetTicket(i);
      if(ticket<1)
         continue;
      //--- If a starting balance or out-/inout trade
      long entry=::HistoryDealGetInteger(ticket,DEAL_ENTRY);
      if(i==0 || entry==DEAL_ENTRY_OUT || entry==DEAL_ENTRY_INOUT)
        {
         double swap      =::HistoryDealGetDouble(ticket,DEAL_SWAP);
         double profit    =::HistoryDealGetDouble(ticket,DEAL_PROFIT);
         double commision =::HistoryDealGetDouble(ticket,DEAL_COMMISSION);
         //--- Calculate balance
         balance_current+=(profit+swap+commision);
         //--- Save to array
         data_count++;
         ::ArrayResize(m_balance,data_count,100000);
         m_balance[data_count-1]=balance_current;
        }
     }
//--- Get amount of data
   return(data_count);
  }
//+------------------------------------------------------------------+
//| Receive statistical data                                         |
//+------------------------------------------------------------------+
void CFrameGenerator::GetStatData(double &dst_array[],double on_tester_value)
  {
   ::ArrayResize(dst_array,::ArraySize(m_balance)+STAT_TOTAL);
   ::ArrayCopy(dst_array,m_balance,STAT_TOTAL,0);
//--- Fill in the first array values (STAT_TOTAL) with test results
   dst_array[0] =::TesterStatistics(STAT_PROFIT);               // net profit
   dst_array[1] =::TesterStatistics(STAT_PROFIT_FACTOR);        // profitability factor
   dst_array[2] =::TesterStatistics(STAT_RECOVERY_FACTOR);      // recovery factor
   dst_array[3] =::TesterStatistics(STAT_TRADES);               // number of trades
   dst_array[4] =::TesterStatistics(STAT_DEALS);                // number of deals
   dst_array[5] =::TesterStatistics(STAT_EQUITY_DDREL_PERCENT); // maximum funds drawdown in %
   dst_array[6] =on_tester_value;                               // custom optimization criterion value
  }
```

**CFrameGenerator::GetBalanceData**() and **CFrameGenerator::GetStatData**() methods are called in the test completion event handler — **CFrameGenerator::OnTesterEvent**(). Data received. Send them to the terminal in a frame.

```
//+------------------------------------------------------------------+
//| Prepare the array of balance values and send it in a frame       |
//| The function should be called in the EA's OnTester() handler     |
//+------------------------------------------------------------------+
void CFrameGenerator::OnTesterEvent(const double on_tester_value)
  {
//--- Get balance data
   int data_count=GetBalanceData();
//--- Array for sending data to a frame
   double stat_data[];
   GetStatData(stat_data,on_tester_value);
//--- Create a frame with data and send it to the terminal
   if(!::FrameAdd(::MQLInfoString(MQL_PROGRAM_NAME),1,data_count,stat_data))
      ::Print(__FUNCTION__," > Frame add error: ",::GetLastError());
   else
      ::Print(__FUNCTION__," > Frame added, Ok");
  }
```

Now let's consider the methods to be used in the frame arrival event handler during optimization — **CFrameGenerator::OnTesterPassEvent**(). We will need the variables for working with frames: name, ID, pass number, accepted value and accepted data array. All these data are sent to the frame using [FrameAdd()](https://www.mql5.com/en/docs/optimization_frames/frameadd) function displayed above.

```
class CFrameGenerator
  {
private:
   //--- Variables for working with frames
   string            m_name;
   ulong             m_pass;
   long              m_id;
   double            m_value;
   double            m_data[];
  };
```

**CFrameGenerator::SaveStatData**() method from the array we accepted in the frame is used to take statistical parameters and save them to a separate string array. There the data are to contain the indicator name and its value. '=' symbol is used as a separator.

```
class CFrameGenerator
  {
private:
   //--- Array with statistical parameters
   string            m_stat_data[];
   //---
private:
   //--- Save statistical data
   void              SaveStatData(void);
  };
//+------------------------------------------------------------------+
//| Save the result statistical parameters to the array              |
//+------------------------------------------------------------------+
void CFrameGenerator::SaveStatData(void)
  {
//--- Array for accepting frame statistical parameters
   double stat[];
   ::ArrayCopy(stat,m_data,0,0,STAT_TOTAL);
   ::ArrayResize(m_stat_data,STAT_TOTAL);
//--- Fill in the array with test results
   m_stat_data[0] ="Net profit="+::StringFormat("%.2f",stat[0]);
   m_stat_data[1] ="Profit Factor="+::StringFormat("%.2f",stat[1]);
   m_stat_data[2] ="Factor Recovery="+::StringFormat("%.2f",stat[2]);
   m_stat_data[3] ="Trades="+::StringFormat("%G",stat[3]);
   m_stat_data[4] ="Deals="+::StringFormat("%G",stat[4]);
   m_stat_data[5] ="Equity DD="+::StringFormat("%.2f%%",stat[5]);
   m_stat_data[6] ="OnTester()="+::StringFormat("%G",stat[6]);
  }
```

Statistical data should be saved in a separate array, so that they can be retrieved in the application ( **CProgram**) class for filling in the table. **CFrameGenerator::CopyStatData**() public method is called to receive them after passing the array for copying.

```
class CFrameGenerator
  {
public:
   //--- Get statistical parameters to the passed array
   int               CopyStatData(string &dst_array[]) { return(::ArrayCopy(dst_array,m_stat_data)); }
  };
```

To update result graphs during optimization, we will need auxiliary methods responsible for adding positive and negative results to arrays. Please note that the result is added to the current frame counter value by X axis. As a result, the formed voids are not reflected on the graph as zero values.

```
//--- Stand-by size for arrays
#define RESERVE_FRAMES 1000000
//+------------------------------------------------------------------+
//| Class for working with optimization results                      |
//+------------------------------------------------------------------+
class CFrameGenerator
  {
private:
   //--- Frame counter
   ulong             m_frames_counter;
   //--- Data on positive and negative results
   double            m_loss_x[];
   double            m_loss_y[];
   double            m_profit_x[];
   double            m_profit_y[];
   //---
private:
   //--- Add (1) negative and (2) positive result to arrays
   void              AddLoss(const double loss);
   void              AddProfit(const double profit);
  };
//+------------------------------------------------------------------+
//| Add negative result to array                                     |
//+------------------------------------------------------------------+
void CFrameGenerator::AddLoss(const double loss)
  {
   int size=::ArraySize(m_loss_y);
   ::ArrayResize(m_loss_y,size+1,RESERVE_FRAMES);
   ::ArrayResize(m_loss_x,size+1,RESERVE_FRAMES);
   m_loss_y[size] =loss;
   m_loss_x[size] =(double)m_frames_counter;
  }
//+------------------------------------------------------------------+
//| Add positive result to array                                     |
//+------------------------------------------------------------------+
void CFrameGenerator::AddProfit(const double profit)
  {
   int size=::ArraySize(m_profit_y);
   ::ArrayResize(m_profit_y,size+1,RESERVE_FRAMES);
   ::ArrayResize(m_profit_x,size+1,RESERVE_FRAMES);
   m_profit_y[size] =profit;
   m_profit_x[size] =(double)m_frames_counter;
  }
```

The main methods for updating graphs here are **CFrameGenerator::UpdateResultsGraph**() and **CFrameGenerator::UpdateBalanceGraph**():

```
class CFrameGenerator
  {
private:
   //--- Update results graph
   void              UpdateResultsGraph(void);
   //--- Update balance graph
   void              UpdateBalanceGraph(void);
  };
```

In **CFrameGenerator::UpdateResultsGraph**() method, the test results (positive/negative profit) are added to the arrays. Then, these data are displayed on an appropriate graph. The names of the graph series display the current number of positive and negative results.

```
//+------------------------------------------------------------------+
//| Update results graph                                             |
//+------------------------------------------------------------------+
void CFrameGenerator::UpdateResultsGraph(void)
  {
//--- Negative result
   if(m_data[0]<0)
      AddLoss(m_data[0]);
//--- Positive result
   else
      AddProfit(m_data[0]);
//--- Update series on the optimization results graph
   CCurve *curve=m_graph_results.CurveGetByIndex(0);
   curve.Name("P: "+(string)ProfitsTotal());
   curve.Update(m_profit_x,m_profit_y);
//---
   curve=m_graph_results.CurveGetByIndex(1);
   curve.Name("L: "+(string)LossesTotal());
   curve.Update(m_loss_x,m_loss_y);
//--- Horizontal axis properties
   CAxis *x_axis=m_graph_results.XAxis();
   x_axis.Min(0);
   x_axis.Max(m_frames_counter);
   x_axis.DefaultStep((int)(m_frames_counter/8.0));
//--- Update graph
   m_graph_results.CalculateMaxMinValues();
   m_graph_results.CurvePlotAll();
   m_graph_results.Update();
  }
```

At the very start of **CFrameGenerator::UpdateBalanceGraph**() method, the data related to the balance is retrieved from the array of data passed in the frame. Since several series can be displayed on the graph, we should make the series update consistent. To achieve this, we will use a separate series counter. To configure the number of simultaneously displayed balance series on the graph, we need **CFrameGenerator::SetCurvesTotal**() public method. As soon as the series counter in it reaches the established limit, the count starts from the beginning. The frame counter acts as the series names. The series color also depends on the result: green stands for a positive result, red — for a negative one.

Since the number of trades in each result is different, we should define the largest series and set the maximum by X axis to fit all necessary series on the graph.

```
class CFrameGenerator
  {
private:
   //--- Number of series
   uint              m_curves_total;
   //--- Index of the current series on the graph
   uint              m_last_serie_index;
   //--- To define the maximum series
   double            m_curve_max[];
   //---
public:
   //--- Set the number of series to display on the graph
   void              SetCurvesTotal(const uint total);
  };
//+------------------------------------------------------------------+
//| Set the number of series for display on the graph                |
//+------------------------------------------------------------------+
void CFrameGenerator::SetCurvesTotal(const uint total)
  {
   m_curves_total=total;
   ::ArrayResize(m_curve_max,total);
   ::ArrayInitialize(m_curve_max,0);
  }
//+------------------------------------------------------------------+
//| Update the balance graph                                         |
//+------------------------------------------------------------------+
void CFrameGenerator::UpdateBalanceGraph(void)
  {
//--- Array for accepting balance values of the current frame
   double serie[];
   ::ArrayCopy(serie,m_data,0,STAT_TOTAL,::ArraySize(m_data)-STAT_TOTAL);
//--- Send the array for displaying on the balance graph
   CCurve *curve=m_graph_balance.CurveGetByIndex(m_last_serie_index);
   curve.Name((string)m_frames_counter);
   curve.Color((m_data[0]>=0)? ::ColorToARGB(clrLimeGreen) : ::ColorToARGB(clrRed));
   curve.Update(serie);
//--- Get the series size
   int serie_size=::ArraySize(serie);
   m_curve_max[m_last_serie_index]=serie_size;
//--- Define the series with the maximum number of elements
   double x_max=0;
   for(uint i=0; i<m_curves_total; i++)
      x_max=::fmax(x_max,m_curve_max[i]);
//--- Horizontal axis properties
   CAxis *x_axis=m_graph_balance.XAxis();
   x_axis.Min(0);
   x_axis.Max(x_max);
   x_axis.DefaultStep((int)(x_max/8.0));
//--- Update the graph
   m_graph_balance.CalculateMaxMinValues();
   m_graph_balance.CurvePlotAll();
   m_graph_balance.Update();
//--- Increase the series counter
   m_last_serie_index++;
//--- If the limit is reached, set the series counter to zero
   if(m_last_serie_index>=m_curves_total)
      m_last_serie_index=0;
  }
```

We considered the methods needed to organize the work in the frame handler. Now let's have a closer look at **CFrameGenerator::OnTesterPassEvent**() method handler itself. It returns **true**, while optimization is underway and [FrameNext()](https://www.mql5.com/en/docs/optimization_frames/framenext) function gets frame data. After completing the optimization, the method returns **false**.

In the EA list of parameters that can be obtained using [FrameInputs()](https://www.mql5.com/en/docs/optimization_frames/frameinputs) function, the parameters set for optimization go first followed by the ones that do not participate in optimization.

If frame data is obtained, [FrameInputs()](https://www.mql5.com/en/docs/optimization_frames/frameinputs) function allows us to obtain EA parameters during the current optimization pass. Then we save the statistics, update the graphs and increase the frame counter. After that, **CFrameGenerator::OnTesterPassEvent**() method returns **true** till the next call.

```
class CFrameGenerator
  {
private:
   //--- EA parameters
   string            m_param_data[];
   uint              m_par_count;
  };
//+------------------------------------------------------------------+
//| Receive frame with data during optimization and display the graph|
//+------------------------------------------------------------------+
bool CFrameGenerator::OnTesterPassEvent(void)
  {
//--- After getting a new frame, try to retrieve data from it
   if(::FrameNext(m_pass,m_name,m_id,m_value,m_data))
     {
      //--- Get input parameters of the EA the frame is formed for
      ::FrameInputs(m_pass,m_param_data,m_par_count);
      //--- Save result statistical parameters to the array
      SaveStatData();
      //--- Update the result and balance graph
      UpdateResultsGraph();
      UpdateBalanceGraph();
      //--- Increase the processed frames counter
      m_frames_counter++;
      return(true);
     }
//---
   return(false);
  }
```

After optimization is complete, [TesterDeinit](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) event is generated and **CFrameGenerator::OnTesterDeinitEvent**() method is called in the frame processing mode. At the moment, not all frames can be processed during the optimization, therefore the results visualization graph will be incomplete. To see the full picture, you need to cycle through all the frames using **CFrameGenerator::FinalRecalculateFrames**() method and reload the graph right after the optimization.

To do this, relocate the pointer to the start of the frame list, then set result arrays and frame counter to zero. Then, cycle through the full list of frames, fill in the arrays by positive and negative results and eventually update the graph.

```
class CFrameGenerator
  {
private:
   //--- Free the arrays
   void              ArraysFree(void);
   //--- Final data re-calculation from all frames after optimization
   void              FinalRecalculateFrames(void);
  };
//+------------------------------------------------------------------+
//| Free the arrays                                                  |
//+------------------------------------------------------------------+
void CFrameGenerator::ArraysFree(void)
  {
   ::ArrayFree(m_loss_y);
   ::ArrayFree(m_loss_x);
   ::ArrayFree(m_profit_y);
   ::ArrayFree(m_profit_x);
  }
//+------------------------------------------------------------------+
//| Final data re-calculation from all frames after optimization     |
//+------------------------------------------------------------------+
void CFrameGenerator::FinalRecalculateFrames(void)
  {
//--- Set the frame pointer to the start
   ::FrameFirst();
//--- Reset the counter and the arrays
   ArraysFree();
   m_frames_counter=0;
//--- Launch cycling through frames
   while(::FrameNext(m_pass,m_name,m_id,m_value,m_data))
     {
      //--- Negative result
      if(m_data[0]<0)
         AddLoss(m_data[0]);
      //--- Positive result
      else
         AddProfit(m_data[0]);
      //--- Increase the counter of processed frames
      m_frames_counter++;
     }
//--- Update series on the graph
   CCurve *curve=m_graph_results.CurveGetByIndex(0);
   curve.Name("P: "+(string)ProfitsTotal());
   curve.Update(m_profit_x,m_profit_y);
//---
   curve=m_graph_results.CurveGetByIndex(1);
   curve.Name("L: "+(string)LossesTotal());
   curve.Update(m_loss_x,m_loss_y);
//--- Horizontal axis properties
   CAxis *x_axis=m_graph_results.XAxis();
   x_axis.Min(0);
   x_axis.Max(m_frames_counter);
   x_axis.DefaultStep((int)(m_frames_counter/8.0));
//--- Update the graph
   m_graph_results.CalculateMaxMinValues();
   m_graph_results.CurvePlotAll();
   m_graph_results.Update();
  }
```

In this case, **CFrameGenerator::OnTesterDeinitEvent**() method code looks as in the listing below. Here we also remember the total number of frames and set the counter to zero.

```
//+------------------------------------------------------------------+
//| Should be called in OnTesterDeinit() handler                     |
//+------------------------------------------------------------------+
void CFrameGenerator::OnTesterDeinitEvent(void)
  {
//--- Final re-calculation of data from all frames after optimization
   FinalRecalculateFrames();
//--- Remember the total number of frames and set the counters to zero
   m_frames_total     =m_frames_counter;
   m_frames_counter   =0;
   m_last_serie_index =0;
  }
```

Next, let's have a look at using **CFrameGenerator** class methods in the application class.

### Working with optimization data in the application class

The graphical interface is created in **CProgram::OnTesterInitEvent**() test initialization method. After that, the graphical interface should be made inaccessible. To do this, we need additional methods **CProgram::IsAvailableGUI**() and **CProgram::IsLockedGUI**() that will be used in other **CProgram** class methods.

Let's initialize the frame generator: pass pointers to the graphs to be used to visualize optimization results.

```
class CProgram : public CWndEvents
  {
private:
   //--- Interface availability
   void              IsAvailableGUI(const bool state);
   void              IsLockedGUI(const bool state);
  }
//+------------------------------------------------------------------+
//| Optimization start event                                         |
//+------------------------------------------------------------------+
void CProgram::OnTesterInitEvent(void)
  {
//--- Create the graphical interface
   if(!CreateFrameModeGUI())
     {
      ::Print(__FUNCTION__," > Could not create the GUI!");
      return;
     }
//--- Make the interface inaccessible
   IsLockedGUI(false);
//--- Initialize the frames generator
   m_frame_gen.OnTesterInitEvent(m_graph1.GetGraphicPointer(),m_graph2.GetGraphicPointer());
  }
//+------------------------------------------------------------------+
//| Interface availability                                           |
//+------------------------------------------------------------------+
void CProgram::IsAvailableGUI(const bool state)
  {
   m_window1.IsAvailable(state);
   m_sleep_ms.IsAvailable(state);
   m_curves_total.IsAvailable(state);
   m_reply_frames.IsAvailable(state);
  }
//+------------------------------------------------------------------+
//| Block the interface                                              |
//+------------------------------------------------------------------+
void CProgram::IsLockedGUI(const bool state)
  {
   m_window1.IsAvailable(state);
   m_sleep_ms.IsLocked(!state);
   m_curves_total.IsLocked(!state);
   m_reply_frames.IsLocked(!state);
  }
```

We have already mentioned that the data in the tables is to be updated in the application class using **CProgram::UpdateStatTable**() and **CProgram::UpdateParamTable**() methods. The code of both tables is identical, so we will give an example of only one of them. Parameter names and values in the same line are displayed using '=' as a separator. Therefore, we pass through them in a loop and split into a separate array dividing into two elements. Then, we enter these values to table cells.

```
class CProgram : public CWndEvents
  {
private:
   //--- Update statistic table
   void              UpdateStatTable(void);
   //--- Update parameter table
   void              UpdateParamTable(void);
  }
//+------------------------------------------------------------------+
//| Update statistic table                                           |
//+------------------------------------------------------------------+
void CProgram::UpdateStatTable(void)
  {
//--- Get data array for statistic table
   string stat_data[];
   int total=m_frame_gen.CopyStatData(stat_data);
   for(int i=0; i<total; i++)
     {
      //--- Split into two lines and enter to the table
      string array[];
      if(::StringSplit(stat_data[i],'=',array)==2)
        {
         if(m_frame_gen.CurrentFrame()>1)
            m_table_stat.SetValue(1,i,array[1],0,true);
         else
           {
            m_table_stat.SetValue(0,i,array[0],0,true);
            m_table_stat.SetValue(1,i,array[1],0,true);
           }
        }
     }
//--- Update the table
   m_table_stat.Update();
  }
```

Both methods for updating data in the tables are called in **CProgram::OnTesterPassEvent**() method by a positive answer from the method of the same name **CFrameGenerator::OnTesterPassEvent**():

```
//+------------------------------------------------------------------+
//| Optimization pass processing event                               |
//+------------------------------------------------------------------+
void CProgram::OnTesterPassEvent(void)
  {
//--- Process obtained test results and display the graph
   if(m_frame_gen.OnTesterPassEvent())
     {
      UpdateStatTable();
      UpdateParamTable();
     }
  }
```

After completing optimization, **CProgram::CalculateProfitsAndLosses**() method calculates the percentage ratio of positive and negative results and displays the data in the status bar:

```
class CProgram : public CWndEvents
  {
private:
   //--- Calculate the ratio of positive and negative results
   void              CalculateProfitsAndLosses(void);
  }
//+------------------------------------------------------------------+
//| Calculate the ratio of positive and negative results             |
//+------------------------------------------------------------------+
void CProgram::CalculateProfitsAndLosses(void)
  {
//--- Exit if there are no frames
   if(m_frame_gen.FramesTotal()<1)
      return;
//--- Number of negative and positive results
   int losses  =m_frame_gen.LossesTotal();
   int profits =m_frame_gen.ProfitsTotal();
//--- Percentage ratio
   string pl =::DoubleToString(((double)losses/(double)m_frame_gen.FramesTotal())*100,2);
   string pp =::DoubleToString(((double)profits/(double)m_frame_gen.FramesTotal())*100,2);;
//--- Display in the status bar
   m_status_bar.SetValue(1,"Profits: "+(string)profits+" ("+pp+"%)"+" / Losses: "+(string)losses+" ("+pl+"%)");
   m_status_bar.GetItemPointer(1).Update(true);
  }
```

The code of the method for processing [TesterDeinit](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) event is displayed below. Initializing the graphics core means that the movement of the mouse cursor is to be tracked and the timer is to be turned on. Unfortunately, in the current **MetaTrader 5** version the timer does not turn on when optimization is complete. Let's hope this opportunity will appear in the future.

```
//+------------------------------------------------------------------+
//| Optimization completion event                                    |
//+------------------------------------------------------------------+
void CProgram::OnTesterDeinitEvent(void)
  {
//--- Optimization completion
   m_frame_gen.OnTesterDeinitEvent();
//--- Make the interface accessible
   IsLockedGUI(true);
//--- Calculate the ratio of positive and negative results
   CalculateProfitsAndLosses();
//--- initialize GUI core
   CWndEvents::InitializeCore();
  }
```

Now we can also work with frame data after optimization is complete. The EA is placed to the terminal chart, and the frames can be accessed to analyze results. The graphical interface makes it all intuitive. In **CProgram::OnEvent**() event handler method, we track:

- changes in the input field for setting the number of displayed balance series on the graph;
- launching viewing the optimization results.

**CProgram::UpdateBalanceGraph**() method is used for updating the graph after changing the number of series. Here we set the number of series for working in the frame generator and then reserve this number on the graph.

```
class CProgram : public CWndEvents
  {
private:
   //--- Update the graph
   void              UpdateBalanceGraph(void);
  };
//+------------------------------------------------------------------+
//| Update the graph                                                 |
//+------------------------------------------------------------------+
void CProgram::UpdateBalanceGraph(void)
  {
//--- Set the number of series for work
   int curves_total=(int)m_curves_total.GetValue();
   m_frame_gen.SetCurvesTotal(curves_total);
//--- Delete the series
   CGraphic *graph=m_graph1.GetGraphicPointer();
   int total=graph.CurvesTotal();
   for(int i=total-1; i>=0; i--)
      graph.CurveRemoveByIndex(i);
//--- Add the series
   double data[];
   for(int i=0; i<curves_total; i++)
      graph.CurveAdd(data,CURVE_LINES,"");
//--- Update the graph
   graph.CurvePlotAll();
   graph.Update();
  }
```

In the event handler, **CProgram::UpdateBalanceGraph**() method is called when toggling the buttons in the input field ( **ON\_CLICK\_BUTTON**) and when the value is entered in the field from keyboard ( **ON\_END\_EDIT**):

```
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CProgram::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
//--- Button pressing events
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON)
     {
      //--- Change the number of series on the graph
      if(lparam==m_curves_total.Id())
        {
         UpdateBalanceGraph();
         return;
        }
      return;
     }
//--- Event of entering the value in the input field
   if(id==CHARTEVENT_CUSTOM+ON_END_EDIT)
     {
      //--- Change the number of series on the graph
      if(lparam==m_curves_total.Id())
        {
         UpdateBalanceGraph();
         return;
        }
      return;
     }
  }
```

To view the results after optimization in **CFrameGenerator** class, **CFrameGenerator::ReplayFrames**() public method is implemented. Here, at the very beginning, we define the following by the frames counter: if the process has just started, the arrays are set to zero, and the frames pointer is moved to the very beginning of the list. Afterwards, the frames are cycled through and the same actions as in previously described **CFrameGenerator::OnTesterPassEvent**() method are performed. If a frame is received, the method returns **true**. Upon completion, the frame and series counters are set to zero and the method returns **false**.

```
class CFrameGenerator
  {
public:
   //--- Cycle through frames
   bool              ReplayFrames(void);
  };
//+------------------------------------------------------------------+
//| Re-play frames after optimization is complete                    |
//+------------------------------------------------------------------+
bool CFrameGenerator::ReplayFrames(void)
  {
//--- Set the frame pointer to beginning
   if(m_frames_counter<1)
     {
      ArraysFree();
      ::FrameFirst();
     }
//--- Launch cycling through frames
   if(::FrameNext(m_pass,m_name,m_id,m_value,m_data))
     {
      //--- Get EA inputs, for which a frame has been formed
      ::FrameInputs(m_pass,m_param_data,m_par_count);
      //--- Save statistical result parameters to array
      SaveStatData();
      //--- Update the result and balance graph
      UpdateResultsGraph();
      UpdateBalanceGraph();
      //--- Increase the counter of processed frames
      m_frames_counter++;
      return(true);
     }
//--- Complete cycling
   m_frames_counter   =0;
   m_last_serie_index =0;
   return(false);
  }
```

**CFrameGenerator::ReplayFrames**() method is called in **CProgram** class from **ViewOptimizationResults**() method. Before launching the frames, the graphical interface becomes unavailable. Scrolling speed can be adjusted by specifying a pause in **Sleep** input field. Meanwhile, the status bar displays the progress bar showing time before the end of the process.

```
class CFrameGenerator
  {
private:
   //--- View optimization results
   void              ViewOptimizationResults(void);
  };
//+------------------------------------------------------------------+
//| View optimization results                                        |
//+------------------------------------------------------------------+
void CProgram::ViewOptimizationResults(void)
  {
//--- Make the interface unavailable
   IsAvailableGUI(false);
//--- Pause
   int pause=(int)m_sleep_ms.GetValue();
//--- Play the frames
   while(m_frame_gen.ReplayFrames() && !::IsStopped())
     {
      //--- Update the tables
      UpdateStatTable();
      UpdateParamTable();
      //--- Update the progress bar
      m_progress_bar.Show();
      m_progress_bar.LabelText("Replay frames: "+string(m_frame_gen.CurrentFrame())+"/"+string(m_frame_gen.FramesTotal()));
      m_progress_bar.Update((int)m_frame_gen.CurrentFrame(),(int)m_frame_gen.FramesTotal());
      //--- Pause
      ::Sleep(pause);
     }
//--- Calculate the ratio of positive and negative results
   CalculateProfitsAndLosses();
//--- Hide the progress bar
   m_progress_bar.Hide();
//--- Make the interface available
   IsAvailableGUI(true);
   m_reply_frames.MouseFocus(false);
   m_reply_frames.Update(true);
  }
```

**CProgram::ViewOptimizationResults**() method is called by pressing **Replay frames** button on the application graphical interface. **ON\_CLICK\_BUTTON** event is generated.

```
//+------------------------------------------------------------------+
//| Events handler                                                   |
//+------------------------------------------------------------------+
void CProgram::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
//--- Event of pressing the buttons
   if(id==CHARTEVENT_CUSTOM+ON_CLICK_BUTTON)
     {
      //--- View optimization results
      if(lparam==m_reply_frames.Id())
        {
         ViewOptimizationResults();
         return;
        }
      //---
      ...
      return;
     }
  }
```

Now it is time to view the results and define what a user actually sees on the graph during optimization when working with frames.

### Displaying the obtained results

For tests, we will use the trade algorithm from the standard delivery — **Moving Average**. We will implement it as a class ("as is") with no additions and corrections. All files of the developed application are to be located in the same folder. The strategy file is included to **Program.mqh** file.

**FormatString.mqh** is included here as an addition with functions for lines formatting. They are not yet part of any class, so let's mark the arrow with black color. The resulting application structure looks as follows:

![Fig. 4. Including the trading strategy class and file with additional functions](https://c.mql5.com/2/31/004.png)

Fig. 4. Including the trading strategy class and file with additional functions

Let's try to optimize the parameters and see how it looks on the terminal chart. Tester settings: EURUSD H1, time range 2017.01.01 – 2018.01.01.

![Fig. 5. Showing Moving Average EA result from the standard delivery](https://c.mql5.com/2/31/005.png)

Fig. 5. Showing Moving Average EA result from the standard delivery

As we can see, it turned out to be quite informative. Almost all results for this trading algorithm are negative (95.23%). If we increase the time range, they become even worse. However, when developing a trading system, we should make sure that most results are positive. Otherwise, the algorithm is loss-making and should not be used. It is necessary to optimize the parameters on more data and ensure there are as many trades as possible.

Let's try to test another trading algorithm from the standard delivery — **MACD Sample.mq5**. It is already implemented as a class. After minor improvements, we can simply connect it to our application, like the previous one. We should test it on the same symbol and timeframe. Although we should increase the time range for more trades in the tests (2010.01.01 – 2018.01.01). Below is the optimization result of a trading EA:

![Fig. 6. Showing MACD Sample result from the standard delivery](https://c.mql5.com/2/31/006.png)

Fig. 6. Showing MACD Sample optimization result

Here we see a very different result: 90.89% of positive outcomes.

Optimization of parameters can take a very long time depending on the amount of data used. You do not need to sit in front of your PC during the entire process. After optimization, you can launch the repeated view of the results in accelerated mode by pressing **Replay frames**. Let's start playing frames with the display limit of 25 series. Here is how it looks:

![Fig. 7. Show MACD Sample EA result after optimization](https://c.mql5.com/2/31/007.gif)

Fig. 7. Show MACD Sample EA result after optimization

### Conclusion

In this article, we presented the modern version of the program for receiving and analyzing optimization frames. The data is visualized in the graphical interface environment developed on the basis of **EasyAndFast** library.

A drawback of this solution is that upon completing the optimization in frames processing mode, it is impossible to launch the timer. This imposes some limitations on working with the same graphical interface. The second issue is that deinitialization in [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit) function is not triggered when removing the EA from the chart. This interferes with the correct event processing. Perhaps, these issues will be solved in one of the future **MetaTrader 5** builds.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/4395](https://www.mql5.com/ru/articles/4395)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4395.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/4395/mql5.zip "Download MQL5.zip")(33.34 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/239284)**
(69)


![Singgih Wasito Adhi](https://c.mql5.com/avatar/2018/4/5AD65D36-7E97.png)

**[Singgih Wasito Adhi](https://www.mql5.com/en/users/ovvn)**
\|
20 Apr 2018 at 17:29

Good job


![Cid Ougaske](https://c.mql5.com/avatar/2019/12/5DE95532-14B5.jpg)

**[Cid Ougaske](https://www.mql5.com/en/users/ougaske)**
\|
4 May 2018 at 20:19

EasyAndFast is the best! Thank you Anatoli.

![Luis Antonio Da Silva Junior](https://c.mql5.com/avatar/2023/2/63dff513-cb40.png)

**[Luis Antonio Da Silva Junior](https://www.mql5.com/en/users/luisjuniorj)**
\|
10 Jul 2018 at 01:31

How to do that in real time while optimizating?


![behzadmuller](https://c.mql5.com/avatar/2021/5/60AE217E-9952.jpg)

**[behzadmuller](https://www.mql5.com/en/users/behzadmuller)**
\|
15 May 2021 at 18:11

Thanks


![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
7 Oct 2021 at 04:11

This is the error I'm facing:

[![](https://c.mql5.com/3/368/6033439098375__1.png)](https://c.mql5.com/3/368/6033439098375.png "https://c.mql5.com/3/368/6033439098375.png")

![How to create a graphical panel of any complexity level](https://c.mql5.com/2/31/graph_panel.png)[How to create a graphical panel of any complexity level](https://www.mql5.com/en/articles/4503)

The article features a detailed explanation of how to create a panel on the basis of the CAppDialog class and how to add controls to the panel. It provides the description of the panel structure and a scheme, which shows the inheritance of objects. From this article, you will also learn how events are handled and how they are delivered to dependent controls. Additional examples show how to edit panel parameters, such as the size and the background color.

![How to create Requirements Specification for ordering an indicator](https://c.mql5.com/2/31/Spec_Indicator.png)[How to create Requirements Specification for ordering an indicator](https://www.mql5.com/en/articles/4304)

Most often the first step in the development of a trading system is the creation of a technical indicator, which can identify favorable market behavior patterns. A professionally developed indicator can be ordered from the Freelance service. From this article you will learn how to create a proper Requirements Specification, which will help you to obtain the desired indicator faster.

![Comparing speeds of self-caching indicators](https://c.mql5.com/2/31/ioba2pczxv_grzmti38_0ew8fnzw9enkgmrv_6f1dur6dvwg.png)[Comparing speeds of self-caching indicators](https://www.mql5.com/en/articles/4388)

The article compares the classic MQL5 access to indicators with alternative MQL4-style methods. Several varieties of MQL4-style access to indicators are considered: with and without the indicator handles caching. Considering the indicator handles inside the MQL5 core is analyzed as well.

![Money Management by Vince. Implementation as a module for MQL5 Wizard](https://c.mql5.com/2/30/MQL5-avatar-capital-001.png)[Money Management by Vince. Implementation as a module for MQL5 Wizard](https://www.mql5.com/en/articles/4162)

The article is based on 'The Mathematics of Money Management' by Ralph Vince. It provides the description of empirical and parametric methods used for finding the optimal size of a trading lot. Also the article features implementation of trading modules for the MQL5 Wizard based on these methods.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/4395&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049273191156787303)

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