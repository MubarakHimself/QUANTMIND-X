---
title: Creating Neural Network EAs Using MQL5 Wizard and Hlaiman EA Generator
url: https://www.mql5.com/en/articles/706
categories: Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:43:51.009081
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/706&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049336563399240101)

MetaTrader 5 / Expert Advisors


### Introduction

Virtually every trader knows about the existence of neural networks. But for the majority of them it remains a black box, with the only things known being the ability of neural networks to recognize patterns, produce associative search for solutions and learn, as well as the fact that they can be effective in predicting the market behavior and in automated trading. Many sources of information that focus on the application of neural networks often speak about their difficulty, emphasizing that one has to devote a great deal of time and effort to learn this subject well and be able to use neural networks in the future.

This article aims at refuting these arguments and proving that advanced automation methods enable traders to have an easy start with neural networks and avoid the lengthy learning process. There is nothing difficult in getting your own experience with neural networks. It is certainly easier than technical analysis.

With this in view, we will describe a method of automatic generation of neural network EAs for MetaTrader 5 using the [MQL5 Wizard](https://www.metatrader5.com/en/automated-trading/mql5wizard) and [Hlaiman EA Generator](https://www.mql5.com/go?link=http://hlaiman.com/commercial-products/demo-versiya "http://hlaiman.com/commercial-products/demo-versiya").

The choice of tools to solve the task at hand is far from being random:

1. [MQL5 Wizard](https://www.metatrader5.com/en/automated-trading/mql5wizard) is an efficient and the fastest mechanism of automatic MQL5 code generation to date that allows you to scale the generated code using additional modules.
2. [Hlaiman EA Generator](https://www.mql5.com/go?link=http://hlaiman.com/commercial-products/demo-versiya "http://hlaiman.com/commercial-products/demo-versiya") is a neural network engine with a flexible mechanism of object integration, programmable directly in the MQL5 code of an Expert Advisor.

The abbreviation 'EA' has been added to the name of the Expert Advisor intentionally as humanlike properties associated with recognition and learning are predominant in a neural network EA unlike in other cases where the use of 'EA' in a name is often misleading and does not reflect the true nature of things.

### General Description

Due to the reason outlined in the article's objective, you will not find here any theoretical information, classifications and structure of neural networks or research data related to financial markets. That information is available in other sources. In this article, we will deliberately confine ourselves to the concept of a neural network as a black box capable of associative thinking and predicting market entries based on recognition of graphical price patterns. For the same reason, we will use the simplest notion of the pattern being a continuous sequence of bars in a trading instrument chart that precedes a profitable price movement.

A few words about problem-solving tools. Unlike [Hlaiman](https://www.mql5.com/go?link=http://hlaiman.com/ "http://hlaiman.com/"), MQL5 Wizard has often been the subject of various articles and documentation and needs no introduction just as [MetaTrader 5](https://www.metatrader5.com/). The socially-oriented Hlaiman project is intended for the development and promotion of multi-purpose software in the form of plug-ins, with [EA Generator](https://www.mql5.com/go?link=http://hlaiman.com/commercial-products/demo-versiya "http://hlaiman.com/commercial-products/demo-versiya") being one of them. As mentioned earlier, functionally EA Generator represents a neural network engine with the integration mechanism.

[Hlaiman EA Generator](https://www.mql5.com/go?link=http://hlaiman.com/commercial-products/demo-versiya "http://hlaiman.com/commercial-products/demo-versiya") includes <hlaim.exe> - a shell that represents a Windows GUI application with multi-document interface and plug-ins in the form of dynamically loadable component libraries. The system provides a wide range of manual and algorithmic adjustment and component control methods that can be both standard and loadable as part of plug-ins. In the course of system operation, you can create complex hierarchies of objects and have flexible control over their methods and properties, using Object Inspector and automation software tools, e.g. scripts.

The integration of Hlaiman EA Generator with MQL5 involves the Object Pascal script interpreter, while the source code is passed via [Named Pipes](https://www.mql5.com/en/articles/503). A multilayer perceptron (MLP) is used as the main neural network component.

Hlaiman EA Generator is integrated with the MQL5 Wizard using a signal library module - **SignalHNN.mqh**. Once generated automatically, the Expert Advisor can then be taught to trade on any number of instruments and time frames. In the МetaТrader 5 terminal, arrows indicating signals can be either drawn in the price chart manually using the graphical objects for arrows or automatically using the **TeachHNN.mq5** script that at the same time initiates the process of teaching the Expert Advisor.

This concludes the theoretical description. We now proceed to the practical part that is divided into two sections - Operating Principles and Implementation.

The second section is aimed at software developers and is provided here primarily out of respect to this website. It is therefore optional, especially for traders who have little or no interest in gaining programming skills but are interested in creating neural network EAs and assessing their efficiency or uselessness in terms of trading.

### Operating Principles

In [MQL5.community](https://www.mql5.com/), it would probably be unnecessary to mention that you need the MetaТrader 5 terminal in order to proceed. If you do not have it, [download](https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe) and install it. You should also download and install a [demo version of Hlaiman EA Generator](https://www.mql5.com/go?link=http://hlaiman.com/commercial-products/demo-versiya "http://hlaiman.com/commercial-products/demo-versiya").

Launch the МetaТrader 5 terminal and start MetaEditor. Open the MQL5 Wizard. You can do it using the 'New' option in the Standard toolbar or under the File menu, as well as using the 'Ctrl+N' hotkey.

In the MQL5 Wizard window, select 'Expert Advisor (generate)' and click 'Next'.

![Fig. 1. Creating an Expert Advisor in the MQL5 Wizard](https://c.mql5.com/2/6/en_001.png)

Fig. 1. Creating an Expert Advisor in the MQL5 Wizard

Specify the location and the name of the Expert Advisor, e.g. **'Experts\\SampleHNN'** and click 'Next'.

![Fig. 2. General Properties of the Expert Advisor](https://c.mql5.com/2/6/en_002.png)

Fig. 2. General Properties of the Expert Advisor

Click the 'Add' button. You will see the window of 'Parameters of Signal Module' where you need to select **'Signals of patterns Hlaiman Neural Network EA generator'** from the drop-down list and click 'OK'.

![Fig. 3. Selecting the trading signal module of Hlaiman Neural Network EA generator](https://c.mql5.com/2/6/en_003.png)

Fig. 3. Selecting the trading signal module of Hlaiman Neural Network EA generator

In case of a very basic implementation, you can click 'Next' at all the remaining steps of the MQL5 Wizard. If necessary, the Expert Advisor can be enhanced by selecting additional options.

Upon completion of the code generation, click 'Compile' and close the MetaEditor window. The generated Expert Advisor will appear in the Navigator panel of the MetaTrader 5 terminal under 'Expert Advisors'.

![Fig. 4. The SampleHNN Expert Advisor](https://c.mql5.com/2/6/en_004.png)

Fig. 4. The SampleHNN Expert Advisor

Before we proceed to teaching the generated Expert Advisor, we need to open a chart with the required symbol and time frame in the terminal. The [Hlaiman EA Generator](https://www.mql5.com/go?link=http://hlaiman.com/commercial-products/demo-versiya "http://hlaiman.com/commercial-products/demo-versiya") application must be up and running.

![Fig. 5. Neural network teaching preparation](https://c.mql5.com/2/6/en_005.png)

Fig. 5. Neural network teaching preparation

To teach the Expert Advisor, select 'TeachHNN' under 'Scripts' in the Navigator panel of the terminal and activate it for the specified chart.

Prior to running the 'TeachHNN' script, you should make sure it has all the appropriate settings. It has the following parameters:

- **Document name** \- name of the Expert Advisor for teaching;
- **Neural layers** \- number of neural network layers;
- **Middle neurons** \- number of neurons;
- **Teaching epochs** \- number of teaching epochs;
- **Pattern bars** \- number of bars in a pattern;
- **Teaching a net?** \- start the neural network teaching process (or simply the signal generation);
- **SignalsCreate** \- to automatically create graphical images of signals;
- **SignalsBarPoints** \- signal generation threshold expressed in points;
- **SignalsBarsCount** \- number of bars for the calculation of points;
- **SignalsStartTime**, **SignalsEndTime** \- start and end time of the period for the signal generation;
- **SignalsClear** \- to automatically delete signal images upon completion of teaching.

![Fig. 6. The TeachHNN script parameters](https://c.mql5.com/2/6/en_006.png)

Fig. 6. The TeachHNN script parameters

If everything is ready, click 'OK' to start the process of teaching the Expert Advisor. This will initiate the automatic generation of graphical patterns for each of the available signals in the chart.

The relevant information is displayed in the 'Experts' tab on the 'Toolbox' panel of the terminal, while the corresponding objects appear in the [Hlaiman EA Generator](https://www.mql5.com/go?link=http://hlaiman.com/commercial-products/demo-versiya "http://hlaiman.com/commercial-products/demo-versiya") window.

Upon completion of the pattern generation, the teaching process starts. It is displayed in the teaching progress bar that appears on the screen.

![Teaching Hlaiman EA Generator](https://c.mql5.com/2/6/image013.png)

Fig. 7. Teaching progress panel

Wait until the process is completed. The teaching process can be terminated before it is completed by right-clicking on the teaching progress bar and selecting the appropriate option in the context menu.

Upon completion of the teaching process and script operation, the relevant message will be added to the log in the 'Experts' tab, e.g. 'Neural net create success! On 431 patterns' indicates that the teaching of the Expert Advisor was successfully completed using 431 signals.

These messages show how many patterns were involved in the teaching process and find out the numbers of those patterns. The BUY and SELL, in particular, are determined using the messages of the following type: 'Sell signal detected at pattern #211'.

![Fig. 8. The TeachHNN script messages in the course of teaching](https://c.mql5.com/2/6/en_008.png)

Fig. 8. The TeachHNN script messages in the course of teaching

The reasons why the process of teaching the Expert Advisor may start with an error are as follows:

1. The [Hlaiman](https://www.mql5.com/go?link=http://hlaiman.com/ "http://hlaiman.com/") application was not up and running prior to the start, as required. In this case the error message will be as follows **"CSignalHNN::InitHNN: Error! initializing pipe server (possible reason: HLAIMAN APPLICATION IS NOT RUNNING!)"**.
2. The absence of arrows indicating signals in the chart upon disabled automatic generation of signals (the SignalsCreate variable = false). In this case the error message to be displayed is as follows: **"OnStart: error, orders arrow not found!"** If the automatic generation of signals is enabled (the SignalsCreate variable = true), an error may be caused by the presence of other graphical objects in the chart, since custom markings are not supposed to be messed with in the program. It is therefore recommended to open all charts separately for the purpose of automatic generation of signals.

When the teaching of the Expert Advisor is completed, you can view the relevant results by switching to GUI Hlaiman and selecting the appropriate objects and visualization panels.

![Fig. 9. The 'Text' tab of the Hlaiman application](https://c.mql5.com/2/6/Figg9bHlaiman_GUI_Text.png)

Fig. 9. The 'Text' tab of the Hlaiman application

![Fig. 10. The 'Graph' tab of the Hlaiman application](https://c.mql5.com/2/6/Figo10xHlaiman_GUI_Graph.png)

Fig. 10. The 'Graph' tab of the Hlaiman application

After successfully teaching the Expert Advisor on at least one trading instrument, we can proceed to testing and/or optimization.

To do this, select the name of the trained Expert Advisor, symbol, time frame, interval and other testing parameters in the Strategy Tester. Set the external variables, if necessary, and run the test.

![Fig. 11. Settings of the SampleHNN Expert Advisor for backtesting](https://c.mql5.com/2/6/en_011.png)

Fig. 11. Settings of the SampleHNN Expert Advisor for backtesting

![Fig. 12. External variables of the SampleHNN Expert Advisor can be modified](https://c.mql5.com/2/6/en_012.png)

Fig. 12. External variables of the SampleHNN Expert Advisor can be modified

Below is an example of the Expert Advisor operation report in the Strategy Tester. The Expert Advisor has been taught using automatically generated signals, with all the external parameters of the teaching script being set by default. The teaching period: 01.01.2010-01.07.2013, instrument: EURUSD H4.

### Strategy Tester Report

| Expert Advisor: | **SampleHNN** |
| --- | --- |
| Symbol: | **EURUSD** |
| Period: | **H4 (2010.01.01-2013.07.12)** |
| Currency: | **USD** |
| Initial Deposit: | **10,000.00** |
| Leverage: | **0,111111111** |
| **Backtesting** |  |
| History Quality: | **100%** |
| Bars: | **5497** |
| Net Profit: | **9,159.58** |
| Gross Profit: | **29,735.97** |
| Gross Loss: | **-20,576.39** |
| Profit Factor: | **1.45** |
| Recovery Factor: | **12.81** |
| AHPR: | **1.0005 (0.05%)** |
| GHPR: | **1.0005 (0.05%)** |
| Total Trades: | **1417** |
| Total Deals: | **2246** |
| Ticks: | **60211228** |
| Balance Drawdown Absolute: | **0.00** |
| Balance Drawdown Maximal: | **679.98 (3.81%)** |
| Balance Drawdown Relative: | **4.00% (715.08)** |
| Expected Payoff: | **6.46** |
| Sharpe Ratio: | **0.16** |
| LR Correlation: | **0.98** |
| LR Standard Error: | **595.06** |
| Short Trades (won %): | **703 (56.61%)** |
| Profit Trades (% of total): | **793 (55.96%)** |
| Largest Profit Trade: | **53.00** |
| Average Profit Trade: | **37.50** |
| Maximum consecutive wins: | **9 (450.38)** |
| Maximum consecutive profit: | **450.38 (9)** |
| Average consecutive wins: | **2** |
| Symbols: | **1** |
| Equity Drawdown Absolute: | **6.60** |
| Equity Drawdown Maximal: | **715.08 (4.00%)** |
| Equity Drawdown Relative: | **4.00% (715.08)** |
| Margin Level: | **6929.24%** |
| Z-Account: | **-1.24 (78.50%)** |
| OnTester Result: | **0** |
| Long Trades (won %): | **714 (55.32%)** |
| Loss Trades (% of total): | **624 (44.04%)** |
| Largest Loss Trade: | **-53.30** |
| Average Loss Trade: | **-32.97** |
| Maximum consecutive losses: | **9 (-234.00)** |
| Maximum consecutive loss: | **-276.67 (7)** |
| Average consecutive losses: | **2** |

![Fig. 13. The SampleHNN Expert Advisor backtesting results](https://c.mql5.com/2/6/en_013.png)

Fig. 13. The SampleHNN Expert Advisor backtesting results

![Fig. 14. The SampleHNN Expert Advisor market entry statistics](https://c.mql5.com/2/6/en_014.png)

Fig. 14. The SampleHNN Expert Advisor market entry statistics

![Fig. 15. Correlation between profit and MFE/MAE of the SampleHNN Expert Advisor](https://c.mql5.com/2/6/Figy15ySampleHNN_Test_Results_Statistics_Correlation.png)

Fig. 15. Correlation between profit and MFE/MAE of the SampleHNN Expert Advisor

![Fig. 16. The SampleHNN Expert Advisor position holding time statistics](https://c.mql5.com/2/6/Figt16qSampleHNN_Test_Results_Statistics_Position_Time.png)

Fig. 16. The SampleHNN Expert Advisor position holding time statistics

### Implementation

The main MQL5 implementation component is the CSignalHNN class described in the SignalHNN.mqh signal module. The class is inherited from the CExpertSignal base class and includes all the necessary data fields and methods for the operation and integration of Hlaiman, as well as for working with Expert Advisors generated using the MQL5 Wizard.

The class template is as follows:

```
//+------------------------------------------------------------------+
//| Class CSignalHNN.                                                |
//| Purpose: Class of generator of trade signals based on            |
//|          the 'Hlaiman EA Generator Neural Net' indicator.        |
//| Is derived from the CExpertSignal class.                         |
//+------------------------------------------------------------------+
class CSignalHNN :public CExpertSignal
  {
protected:
   //--- variables
   int               m_hnn;                   // handle of HNN connect
   string            hnn_path;                // MT5 Terminal data path
   string            hnn_fil;                 // HNN file w neural net
   string            hnn_nam;                 // Expert name
   string            hnn_sym;                 // Symbol name
   string            hnn_per;                 // Period name
   ENUM_TIMEFRAMES   hnn_period;              // Period timeframe
   int               hnn_index;               // Index ext multinet
   int               hnn_bar;                 // index of last bar
   int               hnn_in;                  // input layer
   int               hnn_out;                 // output layer
   int               hnn_layers;              // layers count
   int               hnn_neurons;             // neurons count
   int               hnn_epoch;               // learn epoch
   double            hnn_signal;              // value of last signal
   double            pattern[];               // values of the pattern
   bool              hnn_norm;                // normalize pattern

public:
                     CSignalHNN(void);        // class constructor
                    ~CSignalHNN(void);        // class destructor
   //--- methods of setting adjustable parameters
   void              PatternBarsCount(int value) { hnn_in = value; ArrayResize(pattern, value + 1);  }
   void              LayersCount(int value)      { hnn_layers = value;  }
   void              NeuronsCount(int value)     { hnn_neurons = value;  }
   void              EpochCount(int value)       { hnn_epoch = value;  }
   void              Normalize(bool value)       { hnn_norm = value;  }
   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking conditions of entering the market
   virtual double    Direction(void);

   bool              FillPattern(datetime tim = 0);      // prepare pattern
   bool              AddPattern(string name, int ptype);  // add new pattern
   bool              TeachHNN(void);                     // learn neural net
   bool              SaveFileHNN(void);                  // neural net file
   double            CalculateHNN(void);                 // calc neural signal

                                                        //protected:
   //--- method of initialization of the Hlaiman Application
   bool              InitHNN(bool openn);                // Hlaiman App Init
   void              FreeHNN(void)
     {                     // Hlaiman App Deinit
      if(m_hnn!=0 && m_hnn!=INVALID_HANDLE)
        {
         FileClose(m_hnn);
         m_hnn=0;
        }
     };
  };
```

Following the creation of the class instance using the constructor, this object can work in two main modes:

1. Teaching mode: this mode is associated with collection of market patterns and teaching of the neural network.
2. Indicator mode: in this mode, the neural network signal is calculated using the current pattern.

The mode is identified upon calling the InitHNN initialization mode using the Boolean parameter openn. The true value of this parameter initiates the search and opening of, as well as loading and operation of, the data file of the taught neural network in the indicator mode (2). This mode is considered to be the operating mode and is used in the Expert Advisor for trading.

Unlike the teaching mode (1) that is initialized when calling the InitHNN method with openn=false, the indicator mode is preparatory for the Expert Advisor and is used for the operation of the teaching script.

The initialization method is implemented as follows:

```
//+------------------------------------------------------------------+
//| Initialize HNN                                                   |
//+------------------------------------------------------------------+
bool CSignalHNN::InitHNN(bool openn)
  {
//--- initialize Hlaiman Application
   int num=0;
   ulong res=0;
   if(m_symbol!=NULL)
     {
      hnn_sym=m_symbol.Name();
      hnn_period=m_period;
        } else {
      hnn_sym=_Symbol;
      hnn_period=_Period;
     }
   hnn_per = string(PeriodSeconds(hnn_period) / 60);
   hnn_fil = hnn_nam + NAME_DELIM + hnn_sym + hnn_per + NAME_DELIM + string(hnn_index) + TYPE_NEURO;
   if(m_hnn== 0|| m_hnn == INVALID_HANDLE)
      m_hnn=FileOpen(HLAIMAN_PIPE,FILE_READ|FILE_WRITE|FILE_BIN);
   if(m_hnn!=0 && m_hnn!=INVALID_HANDLE)
     {
      string source,result="";
      if(openn==true)
        {
         result=CON_OPENN+CON_TRUE;
         if(!FileIsExist(hnn_fil,FILE_READ))
           {
            if(FileIsExist(hnn_fil,FILE_READ|FILE_COMMON))
               hnn_fil=TerminalInfoString(TERMINAL_COMMONDATA_PATH)+PATH_FILES+hnn_fil;
            else
              {
               //              hnn_fil = hnn_path + PATH_MQL5 + PATH_FILES + hnn_fil;
               hnn_fil=TerminalInfoString(TERMINAL_DATA_PATH)+PATH_MQL5+PATH_FILES+hnn_fil;
              }
           }
         else hnn_fil=TerminalInfoString(TERMINAL_DATA_PATH)+PATH_MQL5+PATH_FILES+hnn_fil;
           } else {
         result=CON_OPENN+CON_FALSE;
         hnn_fil=TerminalInfoString(TERMINAL_DATA_PATH)+PATH_MQL5+PATH_FILES+hnn_fil;
        }
      source="unit InitHNN; Interface "+result+" var libr, term, exp, sym: TObject;"
             " Implementation function main: integer;\n\r" // Line #1
             " begin"
             " Result := 0;"
             " libr := Open('mt45.dll');\n\r" // Line #2
             " if (libr <> nil) then"
             " begin"
             " term := Open('"+hnn_path+"');\n\r" // Line #3
             " if (term <> nil) then"
             " begin"
             " exp := term.ObjectOfName('"+hnn_nam+"');"
             " if (exp = nil) then exp := term.AddObject('TMT45Expert');\n\r" // Line #5
             " if (exp <> nil) then"
             " begin"
             " if (exp.Name <> '"+hnn_nam+"') then exp.Name := '"+hnn_nam+"';\n\r" // Line #6
             " sym := exp.ObjectOfName('"+hnn_sym+hnn_per+"');"
             " if (sym = nil) then sym := exp.AddObject('TMT45Symbol');"
             " if (sym <> nil) then"
             " begin"
             " sym.Log.Add('"+hnn_sym+hnn_per+"');\n\r"
             " if (sym.Name <> '"+hnn_sym+hnn_per+"') then sym.Name := '"+hnn_sym+hnn_per+"';"
             " if (sym.Period <> "+hnn_per+") then sym.Period := "+hnn_per+";"
             " if (openn = true) then"
             " begin"
             //                   " sym.Log.Add('" + hnn_fil + "');"
             " if (sym.Open('"+hnn_fil+"')) then Result := sym.TeachInput;\n\r" // ret input Line #8
             " end else"
             " begin"
             " sym.TeachInput := "+IntegerToString(hnn_in)+";"
             " sym.TeachOutput := "+IntegerToString(hnn_out)+";"
             " sym.TeachLayer := "+IntegerToString(hnn_layers)+";"
             " sym.TeachNeurons := "+IntegerToString(hnn_neurons)+";"
             " sym.TeachEpoch := "+IntegerToString(hnn_epoch)+";"
             " sym.FileName := '"+hnn_fil+"';"
             " Result := sym.TeachInput;\n\r" // ret input Line #9
             " end;"
             " end;"
             " end;"
             " end;"
             " end;"
             " end; end.";
      FileWriteString(m_hnn,source,StringLen(source));
      FileFlush(m_hnn);
      while(res<=0 && (MQL5InfoInteger(MQL5_TESTER) || num<WAIT_TIMES))
        {
         Sleep(SLEEP_TIM);
         res=FileSize(m_hnn);
         num++;
        }
      if(res>0)
        {
         result=FileReadString(m_hnn,int(res/2));
         res=StringToInteger(result);
         if(res<=RES_OK)
            printf(__FUNCTION__+": Error! Initialization data(possible reason: FILE NOT EXIST OR CORRUPTED "+hnn_fil);
         else
           {
            printf(__FUNCTION__+": Initialization successful! NEURAL PATTERN "+string(res));
            ArrayResize(pattern,int(res+1));
            return(true);
           }
        }
      else
         printf(__FUNCTION__+": Error! pipe server not responding(possible elimination: RESTART HLAIMAN APPLICATION)");
     }
   else
      printf(__FUNCTION__+": Error! initializing pipe server (possible reason: HLAIMAN APPLICATION IS NOT RUNNING!)");
//--- ok
   return(false);
  }
```

As can be seen from the code, the first initialization step covers an attempt to open a named pipe for connectivity with the Hlaiman application. If this attempt fails (e.g., when <hlaim.exe> is not running), the exit is performed with a negative status. At the second step (upon the successful completion of the first step and operating indicator mode), local and common folders of the terminal are searched for the required file name with the neural network data. The third step deals with the preparation of the code in ObjectPascal (Delphi) for the initialization directly in the Hlaiman application.

The text of the code is then moved to the source string. For convenience of formatting, it is broken down into substrings using '\\n\\r' and contains invocations of Hlaiman object properties and methods (see comments). As defined in the text, the object-based environment of the MetaTrader 5 Hlaiman plug-in represents tree architecture, with the object of the plug-in lying at the root.

The МetaТrader 5 terminal object is at the next level followed by Expert Advisor and symbol objects. In case of successful translation and execution of the source code passed via the named pipe, the returned Result value will contain the number of elements of the neural network input vector. As the code suggests, this value is used to initialize the pattern array and the method execution is completed with a positive status.

The other key methods of the CSignalHNN class are CalculateHNN, AddPattern and TeachHNN. The first one returns the result of the neural network calculation in the indicator mode. The other two methods are used in the teaching mode when collecting patterns and initiating the neural network teaching process, respectively.

The implementation of these methods in <SignalHNN.mqh> is as follows:

```
//+------------------------------------------------------------------+
//| Calculate HNN signal                                             |
//+------------------------------------------------------------------+
double CSignalHNN::CalculateHNN(void)
  {
   if(m_hnn==0 || m_hnn==INVALID_HANDLE) return(0.0);
   int num = 0;
   ulong siz = 0;
   double res=0.0;
   string source,result="";
   if(FillPattern(0)==true)
     {
      result=CON_START;
      for(int i=1; i<(ArraySize(pattern)-1); i++)
         result= result+DoubleToString(pattern[i])+CON_ADD;
      result = result + DoubleToString(pattern[ArraySize(pattern) - 1]) + CON_END;
      source = "unit CalcHNN; Interface " + result + " var i: integer; libr, term, exp, sym, lst: TObject;"
              " Implementation function main: double;\n\r" // Line #1
              " begin"
              " Result := 0.0;"
              " libr := Open('mt45.dll');\n\r" // Line #2
              " if (libr <> nil) then"
              " begin"
              " term := Open('"+hnn_path+"');\n\r" // Line #3
              " if (term <> nil) then"
              " begin"
              " exp := term.ObjectOfName('"+hnn_nam+"');\n\r" // Line #4
              " if (exp <> nil) then"
              " begin"
              " sym := exp.ObjectOfName('"+hnn_sym+hnn_per+"');\n\r" // Line #5
              " if (sym <> nil) then"
              " begin"
              " lst := TStringList.Create;"
              " if (lst <> nil) then"
              " begin"
              " lst.Text := cons;"
              " if (lst.Count >= sym.NetInputs.Count) then"
              " begin"
              " for i := 0 to sym.NetInputs.Count - 1 do"
              " begin"
              " sym.NetInputs.Objects[i].NetValue := StrToFloat(lst[i]);\n\r" // Line #6
              //                    " sym.Log.Add('Input ' + IntToStr(i) + ' = ' + lst[i]);"
              " end;"
              " sym.Computed := true;"
              " Result := sym.NetOutputs.Objects[0].NetValue;\n\r" // ret input Line #7
              " end;"
              " lst.Free;"
              " end;"
              " end;"
              " end;"
              " end;"
              " end;"
              " end; end.";
      FileWriteString(m_hnn,source,StringLen(source));
      FileFlush(m_hnn);
      while(siz<=0 && (MQL5InfoInteger(MQL5_TESTER) || num<WAIT_TIMES))
        {
         Sleep(SLEEP_TIM);
         siz=FileSize(m_hnn);
         num++;
        }
      if(siz>0)
        {
         result=FileReadString(m_hnn,int(siz/2));
         res=StringToDouble(result);
        }
     } //else Print("fill pattern error!");
   return(res);
  }
//+------------------------------------------------------------------+
//| AddPattern                                                       |
//+------------------------------------------------------------------+
bool CSignalHNN::AddPattern(string name,int ptype)
  {
   int num=0;
   long res=0;
   ulong siz=0;
   string result,source,nam=name;
   if(m_hnn!=0 || m_hnn!=INVALID_HANDLE)
     {
      pattern[0]=ptype;
      result=CON_START;
      for(int i=0; i<(ArraySize(pattern)-1); i++)
         result= result+DoubleToString(pattern[i])+CON_ADD;
      result = result + DoubleToString(pattern[ArraySize(pattern) - 1]) + CON_END;
      source = "unit AddPatternHNN; Interface " + result + " Implementation function main: integer;"
              " var i: integer; out: double; onam: string;"
              " libr, term, exp, sym, ord, tck, lst: TObject;\n\r" // Line #1
              " begin"
              " Result := 0;"
              " libr := Open('mt45.dll');\n\r" // Line #2
              " if (libr <> nil) then"
              " begin"
              " term := Open('"+hnn_path+"');\n\r" // Line #3
              " if (term <> nil) then"
              " begin"
              " exp := term.ObjectOfName('"+hnn_nam+"');\n\r" // Line #4
              " if (exp <> nil) then"
              " begin"
              " sym := exp.ObjectOfName('"+hnn_sym+hnn_per+"');\n\r" // Line #5
              " if (sym <> nil) then"
              " begin"
              " lst := TStringList.Create;"
              " if (lst <> nil) then"
              " begin"
              " lst.Text := cons;"
              " if (lst.Count >= (sym.TeachInput + sym.TeachOutput)) then"
              " begin"
              " out := StrToFloat(lst[0]);"
              " if(out >= 0) then onam := 'BUY-"+nam+"'"
              " else onam := 'SELL-"+nam+"';"
              " ord := sym.ObjectOfName(onam);"
              " if (ord = nil) then ord := sym.AddObject('TMT45Order');\n\r" // Line #6
              " if (ord <> nil) then"
              " begin"
              " if (ord.Name <> onam) then ord.Name := onam;\n\r" // Line #7
              " if (out >= 0) then ord.OrderType := 0 else ord.OrderType := 1;"
              " if (ord.NetOutput <> out) then ord.NetOutput := out;\n\r" // Line #8
              " for i := 1 to sym.TeachInput do"
              " begin"
              " if(i <= ord.Count) then tck := ord.Items[i - 1] else"
              " tck := ord.AddObject('TMT45Tick');\n\r" // Line #10
              " if (tck <> nil) then"
              " begin"
              " tck.x := i;"
              " tck.y := StrToFloat(lst[i]);\n\r" // Line #11
              " end;"
              " end;"
              " end;"
              " Result := sym.Count;\n\r" // ret input Line #12
              " end;"
              " lst.Free;"
              " end;"
              " end;"
              " end;"
              " end;"
              " end;"
              " end; end.";
      FileWriteString(m_hnn,source,StringLen(source));
      FileFlush(m_hnn);
      while(siz<=0 && (MQL5InfoInteger(MQL5_TESTER) || num<WAIT_TIMES))
        {
         Sleep(SLEEP_TIM);
         siz=FileSize(m_hnn);
         num++;
        }
      if(siz>0)
        {
         result=FileReadString(m_hnn,int(siz/2));
         res=StringToInteger(result);
        }
     }
   return(res>0);
  }
//+------------------------------------------------------------------+
//| TeachHNN                                                         |
//+------------------------------------------------------------------+
bool CSignalHNN::TeachHNN(void)
  {
   int num=0;
   long res=0;
   ulong siz=0;
   string result,source;
   if(m_hnn!=0 || m_hnn!=INVALID_HANDLE)
     {
      source="unit TeachHNN; Interface const WAIT_TIM = 100; WAIT_CNT = 100;"
             "  var i: integer; libr, term, exp, sym: TObject;"
             " Implementation function main: integer;\n\r" // Line #1
             " begin"
             " Result := 0;"
             " libr := Open('mt45.dll');\n\r" // Line #2
             " if (libr <> nil) then"
             " begin"
             " term := Open('"+hnn_path+"');\n\r" // Line #3
             " if (term <> nil) then"
             " begin"
             " exp := term.ObjectOfName('"+hnn_nam+"');\n\r" // Line #4
             " if (exp <> nil) then"
             " begin"
             " sym := exp.ObjectOfName('"+hnn_sym+hnn_per+"');\n\r" // Line #5
             " if (sym <> nil) then"
             " begin"
             " if (sym.Teached) then sym.Teached := false;\n\r" // Line #6
             " sym.Teached := true;\n\r" // Line #7
             " Result := sym.Count;\n\r" // ret input Line #8
             " end;"
             " end;"
             " end;"
             " end;"
             " end; end.";
      FileWriteString(m_hnn,source,StringLen(source));
      FileFlush(m_hnn);
      while(siz<=0)
        {// && (MQL5InfoInteger(MQL5_TESTER) || num < WAIT_TIMES)) {
         Sleep(SLEEP_TIM);
         siz=FileSize(m_hnn);
         num++;
        }
      if(siz>0)
        {
         result=FileReadString(m_hnn,int(siz/2));
         res=StringToInteger(result);
        }
     }
   return(res>0);
  }
```

As can be seen from the code, the method body primarily consists of the source lines, whose text is arranged similarly to the texts considered above in the InitHNN method description. The only difference being that the object-based hierarchy of the plug-in has two more levels for the pattern representation - order and tick. Furthermore, the code contains additional object properties and methods. E.g. the start of the neural network calculation is flagged by the Computed flag of the 'symbol' object, while the Teached flag is used when initiating the teaching process.

The CalculateHNN method is also different from other methods in that the type of the 'main' value returned by the function in this case is 'double'. This value is the neural network output - the signal, whereby the BUY level lies in the range 0..1 and the SELL level is in the range 0..-1. This signal is used by the Expert Advisor in taking decisions regarding opening or closing of trading positions and is controlled by the Direction method. This method performs recalculation in case of the new bar and returns its value expressed as percentage.

```
//+------------------------------------------------------------------+
//| Check conditions for trading signals.                            |
//+------------------------------------------------------------------+
double CSignalHNN::Direction(void)
  {
   if( m_hnn == 0 || m_hnn == INVALID_HANDLE) return(EMPTY_VALUE);
//--- check new bar condition
   int cur_bar = Bars(hnn_sym, hnn_period);
   if (hnn_bar != cur_bar) {
//--- condition OK
      hnn_signal = CalculateHNN() * 100;
      hnn_bar = cur_bar;
   }
   return(hnn_signal);
  }
```

To set the signal response threshold of the Expert Advisor in respect of signals for opening and closing trading positions, you can use the following external variables:

- input int    Signal\_ThresholdOpen =10;      // Signal threshold value to open \[0...100\]
- input int    Signal\_ThresholdClose=10;      // Signal threshold value to close \[0...100\]

In practice, signal levels depend on the neural network teaching quality and intensity which, as a rule, can be assessed visually by monitoring the decrease dynamics of the computational error displayed in the indicator in the course of teaching.

### Conclusions

Hlaiman EA Generator provides components and a transparent controlled object-based environment for the integration with MQL5, whereby:

1. The MQL5 Wizard interface gets an additional type based on signal and pattern recognition, as well as the possibility of generating neural network EAs.
2. In addition to the ability to quickly generate neural network EAs, you can also quickly adapt them to the changing market behavior and repeatedly teach them on different trading instruments and time frames.
3. Due to the fact that the MQL5 Wizard can enable multiple signal modules, you can create complex multi-currency neural network EAs and/or compound indicator-based neural network EAs. They can also be combined with various additional filters, e.g. time filters.
4. And finally, the neural network module in itself can be used as an additional filter to increase the efficiency of a ready-made Expert Advisor. This is possible due to the ability of a neural network to be taught using the visualization charts of test results of the original Expert Advisor.

The use of the script interpreter causing the integrated computing system to appear not very high performing can be considered as one of the disadvantages of the implementation provided above. First off, however, it should be noted that the script code interpretation, as well as the operation of the Hlaiman plug-in is asynchronous with EX5, i.e. we deal with task parallelization. Second, to increase the speed of time-consuming calculations, e.g. when dealing with large neural networks, MetaTrader 5 and Hlaiman can be run on different computers connected via named network pipes. Launching a trading terminal on a separate computer, you will not only gain in performance but you may also increase its security.

When put in perspective, we can look into development of Expert Advisors that can self-learn in the course of trading. At this point the easiest way to do this is by combining the code of the Expert Advisor with the teaching script code as both of them use the same CSignalHNN class that provides the required functionality. This material can become the subject of the follow-through article or form the basis of a completely new one, if it appears to be of interest.

The demo version of Hlaiman EA Generator can be downloaded [here](https://www.mql5.com/go?link=http://hlaiman.com/commercial-products/demo-versiya "http://hlaiman.com/commercial-products/demo-versiya").

**Attachments:**

- **SignalHNN.mqh** \- signal module "MQL5\\Include\\Expert\\Signal\\".
- **TeachHNN.mq5** \- teaching script "MQL5\\Scripts\\".
- **SampleHNN.mq5** \- Expert Advisor based on the trading signal module 'Signals of patterns Hlaiman Neural Network EA generator' and generated using the MQL5 Wizard.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/706](https://www.mql5.com/ru/articles/706)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/706.zip "Download all attachments in the single ZIP archive")

[signalhnn.mqh](https://www.mql5.com/en/articles/download/706/signalhnn.mqh "Download signalhnn.mqh")(23.88 KB)

[teachhnn.mq5](https://www.mql5.com/en/articles/download/706/teachhnn.mq5 "Download teachhnn.mq5")(9.67 KB)

[samplehnn.mq5](https://www.mql5.com/en/articles/download/706/samplehnn.mq5 "Download samplehnn.mq5")(6.35 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/14072)**
(108)


![Ivan Negreshniy](https://c.mql5.com/avatar/2013/7/51E51A58-4224.jpg)

**[Ivan Negreshniy](https://www.mql5.com/en/users/hlaiman)**
\|
16 Jan 2019 at 07:58

**Otto Pauser:**

Metaquotes is probably involved in the sales of the Hlaiman EA Generator!

Are you just interested in counting someone else's money, or would you like to know how it [works](https://www.mql5.com/en/articles/180 "Article: Averaging of price series without additional buffers for intermediate calculations")?

For example, how the MetaQuotes have built a system in which the company's revenue can be shared in proportion to the participation in companies, traders and very small programmers :)

![Alina](https://c.mql5.com/avatar/avatar_na2.png)

**[Alina](https://www.mql5.com/en/users/elka97)**
\|
1 Aug 2020 at 00:29

couldn't find the demo version

where to download

I can't find it on the website or I can't see it

registration does not work

![Ivan_Invanov](https://c.mql5.com/avatar/avatar_na2.png)

**[Ivan\_Invanov](https://www.mql5.com/en/users/ivan_invanov)**
\|
1 Aug 2020 at 14:02

Why are there no bidding statistics?


![Vasiliy Pototskiy](https://c.mql5.com/avatar/2020/4/5EA07245-6F8F.jpg)

**[Vasiliy Pototskiy](https://www.mql5.com/en/users/mrharvester)**
\|
28 Jul 2021 at 15:16

How do I activate the demo?


![Vitaliy Kostrubko](https://c.mql5.com/avatar/2016/8/579E94F7-83FB.png)

**[Vitaliy Kostrubko](https://www.mql5.com/en/users/bbk30)**
\|
3 Nov 2024 at 18:14

... when starting the tester, an error occurred:

[![](https://c.mql5.com/3/447/5871773259356__1.png)](https://c.mql5.com/3/447/5871773259356.png "https://c.mql5.com/3/447/5871773259356.png")

How can I fix it?

![How to Make Money from MetaTrader AppStore and Trading Signals Services If You Are Not a Seller or a Provider](https://c.mql5.com/2/0/mql5_share_avatar.png)[How to Make Money from MetaTrader AppStore and Trading Signals Services If You Are Not a Seller or a Provider](https://www.mql5.com/en/articles/756)

It is possible to start making money on MQL5.com right now without having to be a seller of Market applications or a profitable signals provider. Select the products you like and post links to them on various web resources. Attract potential customers and the profit is yours!

![Extending MQL5 Standard Library and Reusing Code](https://c.mql5.com/2/0/regular-polyhedra-02.png)[Extending MQL5 Standard Library and Reusing Code](https://www.mql5.com/en/articles/741)

MQL5 Standard Library makes your life as a developer easier. Nevertheless, it does not implement all the needs of all developers in the world, so if you feel that you need some more custom stuff you can take a step further and extend. This article walks you through integrating MetaQuotes' Zig-Zag technical indicator into the Standard Library. We get inspired by MetaQuotes' design philosophy to achieve our goal.

![EA Status SMS Notifications](https://c.mql5.com/2/17/831_34.png)[EA Status SMS Notifications](https://www.mql5.com/en/articles/1376)

Developing a system of SMS notifications that informs you of the status of your EA so that you are always aware of any critical situation, wherever you may be.

![Expert Advisor for Trading in the Channel](https://c.mql5.com/2/17/834_22.gif)[Expert Advisor for Trading in the Channel](https://www.mql5.com/en/articles/1375)

The Expert Advisor plots the channel lines. The upper and lower channel lines act as support and resistance levels. The Expert Advisor marks datum points, provides sound notification every time the price reaches or crosses the channel lines and draws the relevant marks. Upon fractal formation, the corresponding arrows appear on the last bars. Line breakouts may suggest the possibility of a growing trend. The Expert Advisor is extensively commented throughout.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/706&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049336563399240101)

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