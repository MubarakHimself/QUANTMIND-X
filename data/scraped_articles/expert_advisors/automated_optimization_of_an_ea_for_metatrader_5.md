---
title: Automated Optimization of an EA for MetaTrader 5
url: https://www.mql5.com/en/articles/4917
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:30:15.249254
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/4917&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068314679555586037)

MetaTrader 5 / Tester


### Introduction

Our [BuddyIlan](https://www.mql5.com/en/market/product/28759) EA uses 4 main parameters that we wanted to optimize automatically each week to best match market variations.

These parameters are:

- SL,
- TP,
- STOFilter,
- STOTimeFrameFilter.

It is unrealistic to launch this type of process each week manually, so we looked for an existing mechanism to perform repetitive tasks but without success (for MetaTrader 5) so we developed this one.

Thanks to Igor Malcev who write the article " [Automated Optimization of a Trading Robot in Real Trading](https://www.mql5.com/en/articles/1467)" for MetaTrader 4.

### Principle

![Optimization process](https://c.mql5.com/2/33/Optimisation_Process__2.png)

The first MetaTrader 5 instance is running 24/7, this instance hosts the BuddyIlan EA and the EA on which we will work today (Optimizer EA) and which will launch the optimization processes on the second MetaTrader 5 instance.

At the end of the process, the Optimizer EA will set optimized values in Global variables which will be read by the running Buddy Ilan EA.

The optimization is scheduled every Saturday without any manual intervention.

### Copying data

As we said above, we need two MetaTrader 5 instances.

The first MetaTrader 5 instance is responsible for copying configuration, parameters and report files between the two instances.

For security reasons, access to files outside a sandbox is not possible under MetaTrader 5, so we will use the DOS command "xcopy" to copy the data between the 2 environments.

To do this we will need to use a Windows based DLL that we will declare as follows:

```
#import  "shell32.dll"
int ShellExecuteW(int hwnd,string Operation,string
                  File,string Parameters,string Directory,int ShowCmd);
#import
```

The call to this function will be done as follows:

```
string PathIniFile = sTerminalTesterDataPath + "\\config\\common.ini";
string PathTester=TerminalInfoString(TERMINAL_DATA_PATH)+"\\MQL5\\Files\\Optimiser\\";

int ret=ShellExecuteW(0,"Open","xcopy","\""+PathIniFile+"\" \""+PathTester+"\" /y","",0);
```

This function will also be called to start optimization processes, e.g.:

```
int start = ShellExecuteW(0, "Open", sTerminalTesterPath + "\\terminal64.exe", "/config:" + TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\Optimiser\\optimise.ini", "", 0);
if(start<32)
  {
   Print("Failed starting Tester");
   return false;
  }
```

DLL import must be authorized for this EA:

![DLL Import](https://c.mql5.com/2/33/dll.PNG)

### Automated Optimization

MetaTrader 5 can be launched via online commands (see: [How to Start the Trading Platform)](https://www.metatrader5.com/en/terminal/help/start_advanced/start "https://www.metatrader5.com/en/terminal/help/start_advanced/start"), automatic tasks can also be launched in this way.

For example, you can add a stanza "\[Tester\]" in the default configuration file (common.ini) to start an automatic optimization at MetaTrader 5 startup.

That's what we're going to do.

### Implementation

The Optimizer EA needs to know the paths of the MetaTrader 5 Tester instance, they will be entered as parameters.

```
input string sTerminalTesterPath = "C:\\Program Files\\ForexTime MT5";
input string sTerminalTesterDataPath="C:\\Users\\BPA\\AppData\\Roaming\\MetaQuotes\\Terminal\\5405B7A2ED87FF45712A041DEF45780";
```

We have defined a working directory: "MQL5\\Files\\Optimiser" in the first MetaTrader 5 instance.

Below the function "CopyAndMoveCommonIni()" copies the default configuration file "common.ini" of the MetaTrader 5 Tester instance in our working directory and renames it as "optimise.ini".

```
bool CopyAndMoveCommonIni()
  {
   string PathIniFile= sTerminalTesterDataPath+"\\config\\common.ini";
   string PathTester = TerminalInfoString(TERMINAL_DATA_PATH)+"\\MQL5\\Files\\Optimiser\\";

   int ret=ShellExecuteW(0,"Open","xcopy","\""+PathIniFile+"\" \""+PathTester+"\" /y","",0);

// wait until the file is copied
   Sleep(2500);
   if(ret<32)
     {
      Print("Failed copying ini file");
      return false;
     }

// We are working now in the sand box, we can use usual MetaTrader 5 File commands
   string IniFileName="Optimiser\\common.ini";
   string CopyTo="Optimiser\\optimise.ini";

   return FileMove( IniFileName, 0, CopyTo, 0 );
  }
```

For those interested, more information about the "ShellExecuteW" function can be found at this address: [ShellExecuteW](https://www.mql5.com/go?link=https://docs.microsoft.com/fr-fr/windows/desktop/api/shellapi/nf-shellapi-shellexecutea "https://docs.microsoft.com/fr-fr/windows/desktop/api/shellapi/nf-shellapi-shellexecutea"). This function does not wait for the execution of the DOS command to return, hence use the delay (Sleep 2500).

We now add the stanza "Tester" in this file:

```
bool AddTesterStanza()
  {
   int filehandle=FileOpen("Optimiser\\Optimise.ini",FILE_READ|FILE_WRITE|FILE_TXT);

   if(filehandle!=INVALID_HANDLE)
     {
      FileSeek(filehandle,0,SEEK_END);

      FileWrite(filehandle,"[Tester]\n",
                "Expert=BuddyIlan\\BuddyIlan\n",
                "ExpertParameters=BuddyIlanTester.set\n",
                "Symbol="+_Symbol+"\n",
                "Period=M15\n",
                "Login=\n",
                "Model=4\n",
                "ExecutionMode=0\n",
                "Optimization=2\n",
                "OptimizationCriterion=0\n",
                "FromDate="+TimeToString(TimeGMT()-InpTesterPeriod*86400,TIME_DATE)+"\n",
                "ToDate="+TimeToString(TimeGMT(),TIME_DATE)+"\n",
                "ForwardMode=0\n",
                "Report=MQL5\\Files\\Reports\\BuddyIlanReport\n",
                "ReplaceReport=1\n",
                "ShutdownTerminal=1\n",
                "Deposit=10000\n",
                "Currency=EURUSD\n",
                "Leverage=1:100\n",
                "UseLocal=1\n",
                "UseRemote=0\n",
                "UseCloud=0\n",
                "Visual=1\n");

      FileClose(filehandle);
     }
   else
     {
      Print("FileOpen, error ",GetLastError());
      return false;
     }
   return true;
  }
```

In this code block, we define the Expert we want to optimize ("BuddyIlan") - this EA must be present in the second environment - and the ExpertParameters file as "BuddyIlanTester.set" (be careful not to use the same name as your EA), we set the period (FromDate - ToDate) and all the parameters needed for the optimization.

We set "ShutdownTerminal=1" which means that the terminal will shut down at the end of the optimization.

The Report will be generated in the file "Files\\Reports\\BuddyIlanReport" - ".xlm" extension will be added by the platform.

If your running EAs are hosted on a Virtual Server with low CPU resources, you can use some remote or cloud agents (see "UseRemote" or "UseCloud") for the optimization process.

### Parameter file

Then we have to create the parameter file we defined above (BuddyIlanTester.set) which includes the values of each parameter of the EA (BuddyIlan) that we want to optimize.

The default values of those parameters are set by the user (defined as parameters):

```
input _TradingMode TradingMode = Dynamic;             // Fixed or Dynamic volume
input double  InpIlanFixedVolume = 0.1;               // Fixed volume size (if set)

input int InpNCurrencies=1;                           // Number of Buddy Ilan instances on this account

input double  LotExponent = 1.4;
input bool    DynamicPips = true;
input int     DefaultPips = 15;

input int Glubina=24;                                 // Number of last bars for calculation of volatility
input int DEL=3;

input int TakeProfit = 40.0;                          // Take Profit (Point)
input int Stoploss = 1000.0;                          // Stop Loss (Point)

input bool InpIlanTrailingStop = true;                // Enable Trailing Stop
input int InpIlanDistanceTS = 5;                      // Trailing Stop distance (Point)

input int MaxTrades=10;
input int InpDeviation=10;                            // Max allowed price deviation (Points)

input bool bSTOFilter = true;                         // Dynamic Trend Filter
input bool bSTOTimeFrameFilter = false;               // Dynamic TimeFrame Filter
input int InpMaxTf = 60;                              // Max TimeFrame
```

The function below accepts 8 arguments, the first 4 correspond to the parameters to be optimized (SL, TP, STOFilter and STOTimeFrameFilter), if true, a "Y" will be positioned at the end of the corresponding parameter line. The following 4 arguments correspond to the already optimized values that we want to take into account during the next optimization.

As its name indicates, this function also copies the parameter file in the ad hoc directory (MQL5\\Profiles\\Tester) of the MetaTrader 5 Tester instance.

```
bool CreateAndCopyParametersFile( bool SL, bool TP, bool STOFilter, bool STOTimeFrameFilter, int SLValue, int TPValue, bool STOFilterValue, bool STOTimeFrameFilterValue )
  {
   int filehandle=FileOpen("Optimiser\\BuddyIlanTester.set",FILE_WRITE|FILE_TXT);

   if(filehandle!=INVALID_HANDLE)
     {
      FileWrite(filehandle,
                "_EA_IDENTIFIER=Buddy Ilan\n",
                "_EA_MAGIC_NUMBER=1111||0||1||10||N\n",
                StringFormat("TradingMode=%d||0||0||0||N\n",TradingMode),
                StringFormat("InpIlanFixedVolume=%lf||0.0||0.000000||0.000000||N\n",InpIlanFixedVolume),
                StringFormat("InpNCurrencies=%d||0||1||10||N\n",InpNCurrencies),
                StringFormat("LotExponent=%lf||0.0||0.000000||0.000000||N\n",LotExponent),
                StringFormat("DynamicPips=%s||false||0||true||N\n",(DynamicPips==true)?"true":"false"),
                StringFormat("DefaultPips=%d||0||1||10||N\n",DefaultPips),
                StringFormat("Glubina=%d||0||1||10||N\n",Glubina),
                StringFormat("DEL=%d||0||1||10||N\n",DEL),

                StringFormat("TakeProfit=%d||30||10||70||%s\n",(TPValue==0)?30:TPValue,(TP==true)?"Y":"N"),
                StringFormat("Stoploss=%d||500||250||1500||%s\n",(SLValue==0)?1000:SLValue,(SL==true)?"Y":"N"),

                StringFormat("InpIlanTrailingStop=%s||false||0||true||N\n",(InpIlanTrailingStop==true)?"true":"false"),
                StringFormat("InpIlanDistanceTS=%d||0||1||10||N\n",InpIlanDistanceTS),
                StringFormat("MaxTrades=%d||0||1||10||N\n",MaxTrades),
                StringFormat("InpDeviation=%d||0||1||10||N\n",InpDeviation),

                StringFormat("bSTOFilter=%s||false||0||true||%s\n",(STOFilterValue==true)?"true":"false",(STOFilter==true)?"Y":"N"),
                StringFormat("bSTOTimeFrameFilter=%s||false||0||true||%s\n",(STOTimeFrameFilterValue==true)?"true":"false",(STOTimeFrameFilter==true)?"Y":"N"),
                StringFormat("InpMaxTf=%d||0||1||10||N\n",InpMaxTf));

      FileClose(filehandle);
     }
   else
     {
      Print("FileOpen BuddyIlanTester.set, error ",GetLastError());
      return false;
     }

   Sleep(1500);

   string PathTester=TerminalInfoString(TERMINAL_DATA_PATH)+"\\MQL5\\Files\\Optimiser\\BuddyIlanTester.set";
   string PathProfile=sTerminalTesterDataPath+"\\MQL5\\Profiles\\Tester\\";

// copy the ini file into the tester folder
   int ret=ShellExecuteW(0,"Open","xcopy","\""+PathTester+"\" \""+PathProfile+"\" /y","",0);

// wait until the file is copied
   Sleep(2500);
   if(ret<32)
     {
      Print("Failed copying parameters file");
      return false;
     }
   return true;
  }
```

### Starting the Optimization

The function below launches the MetaTrader 5 Tester instance, the optimization will be automatically launched using the parameters we have specified. This second instance will generate the result file and then it will shut down.

```
bool StartOptimizer()
  {
// Delete previous Report
   FileDelete("Optimiser\\BuddyIlanReport.xml");

// Delete previous Report (second MetaTrader 5 instance)
   string PathReport=sTerminalTesterDataPath+"\\MQL5\\Files\\Reports\\BuddyIlanReport.xml";

   ShellExecuteW(0,"Open","cmd.exe"," /C del "+PathReport,"",0);

   Sleep(2500);

   string sTerminalPath=TerminalInfoString(TERMINAL_PATH);

// Start Optimization process
   int start=ShellExecuteW(0,"Open",sTerminalTesterPath+"\\terminal64.exe","/config:"+TerminalInfoString(TERMINAL_DATA_PATH)+"\\MQL5\\Files\\Optimiser\\optimise.ini","",0);
   if(start<32)
     {
      Print("Failed starting Tester");
      return false;
     }
   Sleep(15000);
   return true;
  }
```

From the first MetaTrader 5 instance, the easiest way to find out if optimization is complete is to check if the report file is present.

When the report file is generated, we copy it into our working directory.

```
bool CopyReport()
  {
   int nTry=0;

// Waiting and copy Report file

   while(nTry++<500) // Timeout : 2 hours
     {
      string PathReport = sTerminalTesterDataPath + "\\MQL5\\Files\\Reports\\BuddyIlanReport.xml";
      string PathTarget = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\Optimiser\\";

      int ret=ShellExecuteW(0,"Open","xcopy","\""+PathReport+"\" \""+PathTarget+"\" /y","",0);

      if(ret<32)
        {
         PrintFormat("Waiting generation report (%d) ...",nTry);
         Sleep(15000);
        }
      else
        {
         if(FileIsExist("Optimiser\\BuddyIlanReport.xml")==true)
           {
            PrintFormat("Report found (ret=%d) ...",ret);
            Sleep(2500);
            return true;
           }
         else
           {
            PrintFormat("Waiting report (%d) ...",nTry);
            Sleep(15000);
           }
        }
     }
   return false;
  }
```

### Reading the results

Report file is in XML format. Fortunately, Paul van Hemmen wrote a library for MetaTrader 5 to access this type of data, this library is available at this address: [https://www.mql5.com/en/code/1998](https://www.mql5.com/en/code/1998) \- many thanks to him.

We add this library in our EA as follows:

```
#include <EasyXML\EasyXml.mqh>
```

In fact, we added the function below and modified a few little things in this library to adapt it to our report files (see attached files).

```
//+------------------------------------------------------------------+
//| Load XML by given file                                           |
//+------------------------------------------------------------------+
bool CEasyXml::loadXmlFromFullPathFile(string pFilename)
  {
   string sStream;
   int    iStringSize;

   Print("Loading XML File ",pFilename);
   int hFile=FileOpen(pFilename,FILE_ANSI|FILE_READ,0,CP_UTF8);
   if(hFile==INVALID_HANDLE)
     {
      Err=EASYXML_ERR_CONNECTION_FILEOPEN;
      PrintFormat("[%s] Err=%d",pFilename,GetLastError());
      return(Error());
     }

   while(!FileIsEnding(hFile))
     {
      iStringSize = FileReadInteger(hFile, INT_VALUE);
      sStream    += FileReadString(hFile, iStringSize);
     }

   FileClose(hFile);

   return(loadXmlFromString(sStream));
  }
```

Access to the data is quite simple, several functions allow us to parse the results and read the data that interests us.

```
bool LoadResults( OptimisationType eType )
  {
// Init variable
   BetterProfit=0.0;

// Load Results
   CEasyXml EasyXmlDocument;
   EasyXmlDocument.setDebugging(false);

   if(EasyXmlDocument.loadXmlFromFullPathFile("Optimiser\\BuddyIlanReport.xml")==true)
     {
      str="";
      CEasyXmlNode *RootNode=EasyXmlDocument.getDocumentRoot();
      for(int j=0; j<RootNode.Children().Total(); j++)
        {
         CEasyXmlNode *ChildNode=RootNode.Children().At(j);
         for(int i=0; i<ChildNode.Children().Total(); i++)
           {
            CEasyXmlNode *cNode=ChildNode.Children().At(i);
            if(cNode.getName() == "Worksheet" )
              {
               switch(eType)
                 {
                  case _SL :
                     DisplayNodesSL(cNode);
                     PrintFormat("-> SL=%d (Profit=%.2lf)",BetterSL,BetterProfit);
                     break;

                  case _TP :
                     DisplayNodesTP(cNode);
                     PrintFormat("-> TP=%d (Profit=%.2lf DD=%lf)",BetterTP,BetterProfit,BetterDD);
                     break;

                  case _STO :
                     DisplayNodesSTO(cNode);
                     PrintFormat("-> STOFilter=%s STOTimeFrameFilter=%s (Profit=%.2lf)",(BetterSTOFilter==true)?"true":"false",(BetterSTOTimeFrameFilter==true)?"true":"false",BetterProfit);
                     break;
                 }
               break;
              }
           }
        }
     }
   else
      PrintFormat("Error found");
   return true;
  }
```

Since we want to optimize several parameters, we will analyze the results in different ways, so we need a specific function for each optimization. (SL, TP and STO parameters). These functions are recursive.

Below, the one used to analyze the results of the SL optimization:

```
void DisplayNodesSL( CEasyXmlNode *Node )
  {
   for(int i=0; i<Node.Children().Total(); i++)
     {
      CEasyXmlNode *ChildNode=Node.Children().At(i);

      if(ChildNode.Children().Total()==0)
        {
         str+=ChildNode.getValue()+",";
        }
      else
        {
         DisplayNodesSL(ChildNode);

         if(Node.getName()=="Table" && ChildNode.getName()=="Row")
           {
            string res[];
            StringSplit(str,',',res);

            // Bypass columns titles
            if(StringCompare(res[0],"Pass",true)!=0)
              {
               double profit=StringToDouble(res[2]);
               int sl=(int) StringToInteger(res[10]);

               PrintFormat("[%s]  Profit=%.2lf StopLoss=%d DD=%s",str,profit,sl,res[8]);

               if(profit>BetterProfit || (profit==BetterProfit && sl<BetterSL))
                 {
                  BetterProfit=profit;
                  BetterSL=sl;
                 }
              }
           }
         if(Node.getName()=="Table")
            str="";
        }
     }
  }
```

This function is called on each row and cell.

If a node doesn't have any child, that means that it contains data, we store these data in a string that we split at the end of the line.

```
if( ChildNode.Children().Total() == 0 )
  {
   str+=ChildNode.getValue()+",";
  }
```

So, values for each column are available in the array "res\[\]" and we select the results of our choice.

### EA Body

We now have all the necessary bricks to optimize our 4 parameters, deduce the best possible parameter setting and set the value of the corresponding global variables which will be read by the running BuddyIlan EA.

```
void OnTimer()
  {
   MqlDateTime dt;

   datetime now=TimeLocal(dt);

// On Saturday
   if(dt.day_of_week!=6)
     {
      bOptimisationDone=false;
      return;
     }

// At 6:00 am
   if(dt.hour<6)
      return;

// Already done ?
   if(bOptimisationDone==true)
      return;

// Remove previous "optimise.ini"
   FileDelete("Optimiser\\Optimise.ini");

// Create the EA config file and copy it to \MQL5\Profiles\Test (Tester Instance)
   if(CreateAndCopyParametersFile(true,false,false,false,0,0,true,false)==false)
      return;

// Copy common.ini -> optimise.ini
   if(CopyAndMoveCommonIni()==false)
      return;

// Add [Tester] stanza in optimise.ini - https://www.metatrader5.com/en/terminal/help/start_advanced/start
   if(AddTesterStanza()==false)
      return;

   Print("=======================\nOptimization SL-1");

// Start first optimization SL
   StartOptimizer();

// Copying the report file to the working directory
   if(CopyReport()==false)
      return;

// Analyse reports
   if(LoadResults(_SL)==false)
      return;

   Print("=======================\nOptimization STO");

// Create parameter file for STO optimization (the 2 parameters will be optimized at the same time)
   if(CreateAndCopyParametersFile(false,false,true,true,BetterSL,0,true,false)==false)
      return;

// Start optimizer STO
   StartOptimizer();

// Copying the report file to the working directory
   if(CopyReport()==false)
      return;

   if(LoadResults(_STO)==false)
      return;

   Print("=======================\nOptimization SL-2");

// Create parameter file for second SL optimization (recalculation with new STO parameter values)
   if(CreateAndCopyParametersFile(true,false,false,false,0,0,BetterSTOFilter,BetterSTOTimeFrameFilter)==false)
      return;

// Start optimizer
   StartOptimizer();

   if(CopyReport()==false)
      return;

   if(LoadResults(_SL)==false)
      return;

   Print("=======================\nOptimization TP");

// Create parameter file for TP optimization
   if(CreateAndCopyParametersFile(false,true,false,false,BetterSL,0,BetterSTOFilter,BetterSTOTimeFrameFilter)==false)
      return;

// Start optimizer
   StartOptimizer();

   if(CopyReport()==false)
      return;

   if(LoadResults(_TP)==false)
      return;

// Conclusion

   PrintFormat("=======================\nSL=%d TP=%d STOFilter=%s STOTimeFrameFilter=%s (Profit=%.2lf DD=%lf)\n=======================",
               BetterSL,BetterTP,(BetterSTOFilter==true)?"true":"false",(BetterSTOTimeFrameFilter==true)?"true":"false",BetterProfit,BetterDD);

// Set Global variables - The running BuddyIlan EA will read and use these new values

// If the Draw Down found is over 50%, the EA Stop trading
   if(BetterDD>50.0 && GlobalVariableSet(gVarStop,1.0)==false)
     {
      PrintFormat("Error setting Global Variable [%s]",gVarStop);
     }

   if(GlobalVariableSet(gVarSL,BetterSL)==false)
     {
      PrintFormat("Error setting Global Variable [%s]=%d",gVarSL,BetterSL);
     }

   if(GlobalVariableSet(gVarTP,BetterTP)==false)
     {
      PrintFormat("Error setting Global Variable [%s]=%d",gVarTP,BetterTP);
     }

   if(GlobalVariableSet(gVarSTOFilter,(BetterSTOFilter==true)?1.0:0.0)==false)
     {
      PrintFormat("Error setting Global Variable [%s]=%.1lf",gVarSTOFilter,(BetterSTOFilter==true)?1.0:0.0);
     }

   if(GlobalVariableSet(gVarSTOTimeFrameFilter,(BetterSTOTimeFrameFilter==true)?1.0:0.0)==false)
     {
      PrintFormat("Error setting Global Variable [%s]=%.1lf",gVarSTOTimeFrameFilter,(BetterSTOTimeFrameFilter==true)?1.0:0.0);
     }

   bOptimisationDone=true;
  }
```

Global variable names are built in the OnInit() function:

```
int OnInit()
  {
// Global variables

   gVarStop="BuddyIlan."+_Symbol+".Stop";
   gVarSL = "BuddyIlan." + _Symbol + ".SL";
   gVarTP = "BuddyIlan." + _Symbol + ".TP";
   gVarSTOFilter="BuddyIlan."+_Symbol+".STOFilter";
   gVarSTOTimeFrameFilter="BuddyIlan."+_Symbol+".STOTimeFrameFilter";
```

Below is the full optimization process:

```
2018.07.07 13:20:15.978 BuddyIlanOptimizer (EURGBP,M15) TERMINAL_PATH = C:\Program Files\MetaTrader 5 - ActivTrades
2018.07.07 13:20:15.978 BuddyIlanOptimizer (EURGBP,M15) TERMINAL_DATA_PATH = C:\Users\BPA\AppData\Roaming\MetaQuotes\Terminal\FE0E65DDB0B7B40DE125080872C34D61
2018.07.07 13:20:15.978 BuddyIlanOptimizer (EURGBP,M15) TERMINAL_COMMONDATA_PATH = C:\Users\BPA\AppData\Roaming\MetaQuotes\Terminal\Common
2018.07.07 13:20:32.586 BuddyIlanOptimizer (EURGBP,M15) =======================
2018.07.07 13:20:32.586 BuddyIlanOptimizer (EURGBP,M15) Optimization SL-1
2018.07.07 13:20:50.439 BuddyIlanOptimizer (EURGBP,M15) Waiting report (1) ...
2018.07.07 13:21:05.699 BuddyIlanOptimizer (EURGBP,M15) Waiting report (2) ...
2018.07.07 13:21:20.859 BuddyIlanOptimizer (EURGBP,M15) Waiting report (3) ...
2018.07.07 13:21:35.952 BuddyIlanOptimizer (EURGBP,M15) Report found (ret=42) ...
2018.07.07 13:21:38.471 BuddyIlanOptimizer (EURGBP,M15) Loading XML File Optimiser\BuddyIlanReport.xml
2018.07.07 13:21:38.486 BuddyIlanOptimizer (EURGBP,M15) [0,11032.2600,1032.2600,3.3406,1.7096,1.5083,0.1558,0,6.2173,309,500,]  Profit=1032.26 StopLoss=500 DD=6.2173
2018.07.07 13:21:38.487 BuddyIlanOptimizer (EURGBP,M15) [2,11463.8000,1463.8000,4.7837,2.0386,0.8454,0.1540,0,15.4222,306,1000,]  Profit=1463.80 StopLoss=1000 DD=15.4222
2018.07.07 13:21:38.487 BuddyIlanOptimizer (EURGBP,M15) [4,11444.1000,1444.1000,4.7348,2.0340,0.8340,0.1529,0,15.4493,305,1500,]  Profit=1444.10 StopLoss=1500 DD=15.4493
2018.07.07 13:21:38.487 BuddyIlanOptimizer (EURGBP,M15) [1,11297.1900,1297.1900,4.2392,1.8414,0.8180,0.1400,0,14.1420,306,750,]  Profit=1297.19 StopLoss=750 DD=14.1420
2018.07.07 13:21:38.487 BuddyIlanOptimizer (EURGBP,M15) [3,11514.0800,1514.0800,4.9158,2.3170,1.4576,0.2055,0,9.3136,308,1250,]  Profit=1514.08 StopLoss=1250 DD=9.3136
2018.07.07 13:21:38.487 BuddyIlanOptimizer (EURGBP,M15) -> SL=1250 (Profit=1514.08)
2018.07.07 13:21:38.487 BuddyIlanOptimizer (EURGBP,M15) =======================
2018.07.07 13:21:38.487 BuddyIlanOptimizer (EURGBP,M15) Optimization STO
2018.07.07 13:22:02.660 BuddyIlanOptimizer (EURGBP,M15) Waiting report (1) ...
2018.07.07 13:22:17.768 BuddyIlanOptimizer (EURGBP,M15) Waiting report (2) ...
2018.07.07 13:22:32.856 BuddyIlanOptimizer (EURGBP,M15) Waiting report (3) ...
2018.07.07 13:22:47.918 BuddyIlanOptimizer (EURGBP,M15) Waiting report (4) ...
2018.07.07 13:23:02.982 BuddyIlanOptimizer (EURGBP,M15) Report found (ret=42) ...
2018.07.07 13:23:05.485 BuddyIlanOptimizer (EURGBP,M15) Loading XML File Optimiser\BuddyIlanReport.xml
2018.07.07 13:23:05.499 BuddyIlanOptimizer (EURGBP,M15) [0,11463.5000,1463.5000,4.4483,2.0614,0.8452,0.1540,0,15.4267,329,false,false,]  Profit=1463.50 false false  DD=15.4267
2018.07.07 13:23:05.499 BuddyIlanOptimizer (EURGBP,M15) [1,11444.1000,1444.1000,4.7348,2.0340,0.8340,0.1529,0,15.4493,305,true,false,]  Profit=1444.10 true false  DD=15.4493
2018.07.07 13:23:05.499 BuddyIlanOptimizer (EURGBP,M15) [2,11430.5300,1430.5300,5.1090,2.1548,0.8917,0.1717,0,14.4493,280,false,true,]  Profit=1430.53 false true  DD=14.4493
2018.07.07 13:23:05.499 BuddyIlanOptimizer (EURGBP,M15) [3,11470.7100,1470.7100,6.2851,1.8978,0.8146,0.1288,0,17.3805,234,true,true,]  Profit=1470.71 true true  DD=17.3805
2018.07.07 13:23:05.499 BuddyIlanOptimizer (EURGBP,M15) -> STOFilter=true STOTimeFrameFilter=true (Profit=1470.71)
2018.07.07 13:23:05.500 BuddyIlanOptimizer (EURGBP,M15) =======================
2018.07.07 13:23:05.500 BuddyIlanOptimizer (EURGBP,M15) Optimization SL-2
2018.07.07 13:23:29.921 BuddyIlanOptimizer (EURGBP,M15) Waiting report (1) ...
2018.07.07 13:23:45.043 BuddyIlanOptimizer (EURGBP,M15) Waiting report (2) ...
2018.07.07 13:24:00.170 BuddyIlanOptimizer (EURGBP,M15) Waiting report (3) ...
2018.07.07 13:24:15.268 BuddyIlanOptimizer (EURGBP,M15) Waiting report (4) ...
2018.07.07 13:24:30.340 BuddyIlanOptimizer (EURGBP,M15) Report found (ret=42) ...
2018.07.07 13:24:32.854 BuddyIlanOptimizer (EURGBP,M15) Loading XML File Optimiser\BuddyIlanReport.xml
2018.07.07 13:24:32.872 BuddyIlanOptimizer (EURGBP,M15) [0,9269.9000,-730.1000,-2.7760,0.7328,-0.3644,-0.0532,0,19.4241,263,500,]  Profit=-730.10 StopLoss=500 DD=19.4241
2018.07.07 13:24:32.872 BuddyIlanOptimizer (EURGBP,M15) [4,11470.7100,1470.7100,6.2851,1.8978,0.8146,0.1288,0,17.3805,234,1500,]  Profit=1470.71 StopLoss=1500 DD=17.3805
2018.07.07 13:24:32.872 BuddyIlanOptimizer (EURGBP,M15) [3,11475.9500,1475.9500,6.2806,1.8995,0.8175,0.1290,0,17.3718,235,1250,]  Profit=1475.95 StopLoss=1250 DD=17.3718
2018.07.07 13:24:32.872 BuddyIlanOptimizer (EURGBP,M15) [2,11400.7500,1400.7500,5.8609,1.8442,0.7759,0.1292,0,17.3805,239,1000,]  Profit=1400.75 StopLoss=1000 DD=17.3805
2018.07.07 13:24:32.872 BuddyIlanOptimizer (EURGBP,M15) [1,10662.5500,662.5500,2.8807,1.3618,0.3815,0.0862,0,16.7178,230,750,]  Profit=662.55 StopLoss=750 DD=16.7178
2018.07.07 13:24:32.873 BuddyIlanOptimizer (EURGBP,M15) -> SL=1250 (Profit=1475.95)
2018.07.07 13:24:32.873 BuddyIlanOptimizer (EURGBP,M15) =======================
2018.07.07 13:24:32.873 BuddyIlanOptimizer (EURGBP,M15) Optimization TP
2018.07.07 13:24:57.175 BuddyIlanOptimizer (EURGBP,M15) Waiting report (1) ...
2018.07.07 13:25:12.311 BuddyIlanOptimizer (EURGBP,M15) Waiting report (2) ...
2018.07.07 13:25:27.491 BuddyIlanOptimizer (EURGBP,M15) Waiting report (3) ...
2018.07.07 13:25:42.613 BuddyIlanOptimizer (EURGBP,M15) Waiting report (4) ...
2018.07.07 13:25:57.690 BuddyIlanOptimizer (EURGBP,M15) Report found (ret=42) ...
2018.07.07 13:26:00.202 BuddyIlanOptimizer (EURGBP,M15) Loading XML File Optimiser\BuddyIlanReport.xml
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) [1,11768.5700,1768.5700,8.2259,2.4484,1.1024,0.2233,0,14.1173,215,40,]  Profit=1768.57 TakeProfit=40 DD=14.117300
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) [4,12343.5200,2343.5200,13.5464,2.5709,1.3349,0.2519,0,15.0389,173,70,]  Profit=2343.52 TakeProfit=70 DD=15.038900
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) [0,11243.4600,1243.4600,5.2913,1.6399,0.6887,0.1039,0,17.3805,235,30,]  Profit=1243.46 TakeProfit=30 DD=17.380500
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) [3,12292.3500,2292.3500,11.8162,2.5837,0.9257,0.2538,0,20.4354,194,60,]  Profit=2292.35 TakeProfit=60 DD=20.435400
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) [2,12146.3900,2146.3900,11.0639,2.4416,1.2226,0.2292,0,15.0772,194,50,]  Profit=2146.39 TakeProfit=50 DD=15.077200
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) -> TP=70 (Profit=2343.52 DD=15.038900)
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) =======================
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) SL=1250 TP=70 STOFilter=true STOTimeFrameFilter=true (Profit=2343.52 DD=15.038900)
2018.07.07 13:26:00.219 BuddyIlanOptimizer (EURGBP,M15) =======================
```

### Conclusion

The implementation of this process requires a minimum knowledge of MetaTrader 5, its optimization mechanisms and programming.

Attached are the sources of this EA, XML Parser files from Paul van Hemmen and the modified file "EasyXml.mqh".

Hope this helps.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4917.zip "Download all attachments in the single ZIP archive")

[Files.zip](https://www.mql5.com/en/articles/download/4917/files.zip "Download Files.zip")(14.39 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/282288)**
(24)


![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
5 Jun 2023 at 23:16

**Gamuchirai Ndawana [#](https://www.mql5.com/en/forum/282288/page2#comment_47323204):** Can someone explain to me why this can't be done using the MetaTrader5 module for Python?

Because Python code runs externally to MetaTrader 5 via an API. It does not run as a compiled MQL5 executable.


![atesz5870](https://c.mql5.com/avatar/avatar_na2.png)

**[atesz5870](https://www.mql5.com/en/users/atesz5870)**
\|
2 Aug 2023 at 03:05

That's awesome! I have an "optimization strategy" for my ea, can I automatizate my optimization like this? (first gen opt, choose favorable set, then opt other inputs, then again opt other inputs, etc..)


![Luandre Ezra](https://c.mql5.com/avatar/2020/3/5E660C07-11A4.png)

**[Luandre Ezra](https://www.mql5.com/en/users/ezraluandre)**
\|
4 Dec 2023 at 14:41

Hi [@Bruno Paulet](https://www.mql5.com/en/users/bpasoftware)

I want to ask about 2 things

First is about this statement,

we define the Expert we want to optimize ("BuddyIlan") - this EA must be present in the second environment

Do I need to put my EA at specified location or not?

Second is that whenever EA tester launch it will immediately close, how can I fix this?

![Gamuchirai Zororo Ndawana](https://c.mql5.com/avatar/2024/3/6607f08d-4cae.jpg)

**[Gamuchirai Zororo Ndawana](https://www.mql5.com/en/users/gamuchiraindawa)**
\|
13 Jan 2024 at 10:54

**Fernando Carreiro [#](https://www.mql5.com/en/forum/282288/page2#comment_47323219):** Because Python code runs externally to MetaTrader 5 via an API. It does not run as a compiled MQL5 executable.

What if I converted the Python script to a DLL?

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
13 Jan 2024 at 11:40

**Gamuchirai Zororo Ndawana [#](https://www.mql5.com/en/forum/282288/page2#comment_51690548):** What if I converted the Python script to a DLL?

Then you might as well just write directly it in MQL.


![MQL5 Cookbook: Getting properties of an open hedge position](https://c.mql5.com/2/34/position.png)[MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)

MetaTrader 5 is a multi-asset platform. Moreover, it supports different position management systems. Such opportunities provide significantly expanded options for the implementation and formalization of trading ideas. In this article, we discuss methods of handling and accounting of position properties in the hedging mode. The article features a derived class, as well as examples showing how to get and process the properties of a hedge position.

![Elder-Ray (Bulls Power and Bears Power)](https://c.mql5.com/2/33/Elder-Ray-las1su67-2niearv.png)[Elder-Ray (Bulls Power and Bears Power)](https://www.mql5.com/en/articles/5014)

The article dwells on Elder-Ray trading system based on Bulls Power, Bears Power and Moving Average indicators (EMA — exponential averaging). This system was described by Alexander Elder in his book "Trading for a Living".

![Using indicators for optimizing Expert Advisors in real time](https://c.mql5.com/2/34/indicator_RealTime_optimaze.png)[Using indicators for optimizing Expert Advisors in real time](https://www.mql5.com/en/articles/5061)

Efficiency of any trading robot depends on the correct selection of its parameters (optimization). However, parameters that are considered optimal for one time interval may not retain their effectiveness in another period of trading history. Besides, EAs showing profit during tests turn out to be loss-making in real time. The issue of continuous optimization comes to the fore here. When facing plenty of routine work, humans always look for ways to automate it. In this article, I propose a non-standard approach to solving this issue.

![50,000 completed orders in the MQL5.com Freelance service](https://c.mql5.com/2/34/freelance-icon.png)[50,000 completed orders in the MQL5.com Freelance service](https://www.mql5.com/en/articles/5226)

Members of the official MetaTrader Freelance service have completed more than 50,000 orders as at October 2018. This is the world's largest Freelance site for MQL programmers: more than a thousand developers, dozens of new orders daily and 7 languages localization.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=cdqwclcywrnnwnlhaivwpcbmuzrvimnj&ssn=1769178613943390255&ssn_dr=0&ssn_sr=0&fv_date=1769178613&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F4917&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automated%20Optimization%20of%20an%20EA%20for%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917861397644423&fz_uniq=5068314679555586037&sv=2552)

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