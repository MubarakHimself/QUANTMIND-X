---
title: MQL5 Cookbook: Saving Optimization Results of an Expert Advisor Based on Specified Criteria
url: https://www.mql5.com/en/articles/746
categories: Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:43:41.486884
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=lyigzdrbpkpvacqrheysreniggnocyxb&ssn=1769093020967697924&ssn_dr=0&ssn_sr=0&fv_date=1769093020&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F746&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Cookbook%3A%20Saving%20Optimization%20Results%20of%20an%20Expert%20Advisor%20Based%20on%20Specified%20Criteria%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909302029959475&fz_uniq=5049334540469643675&sv=2552)

MetaTrader 5 / Tester


### Introduction

We continue the series of articles on MQL5 programming. This time we will see how to get the results of each optimization pass during the Expert Advisor parameter optimization. The implementation will be done so as to ensure that if a certain condition specified in the external parameters is met, the corresponding pass values will be written to a file. In addition to test values, we will also save the parameters that brought about such results.

### Development

To implement the idea, we are going to use the ready-made Expert Advisor with a simple trading algorithm described in the article ["MQL5 Cookbook: How to Avoid Errors When Setting/Modifying Trade Levels"](https://www.mql5.com/en/articles/643) and just add to it all the necessary functions. The source code has been prepared using the approach employed in the most recent articles of the series. So, all the functions are arranged into different files and included in the main project file. You can see how files can be included in the project in the article ["MQL5 Cookbook: Using Indicators to Set Trading Conditions in Expert Advisors"](https://www.mql5.com/en/articles/645).

To gain access to data in the course of optimization, you can use special MQL5 functions: [OnTesterInit()](https://www.mql5.com/en/docs/basis/function/events#ontesterinit), [OnTester()](https://www.mql5.com/en/docs/basis/function/events#ontester), [OnTesterPass()](https://www.mql5.com/en/docs/basis/function/events#ontesterpass) and [OnTesterDeinit()](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit). Let's have a quick look at each of them:

- [OnTesterInit()](https://www.mql5.com/en/docs/basis/function/events#ontesterinit) \- this function is used to determine the optimization start.
- [OnTester()](https://www.mql5.com/en/docs/basis/function/events#ontester) \- this function is responsible for adding so-called frames after every optimization pass. The definition of frames will be given further below.
- [OnTesterPass()](https://www.mql5.com/en/docs/basis/function/events#ontesterpass) \- this function gets frames after every optimization pass.
- [OnTesterDeinit()](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) \- this function generates the event of the end of the Expert Advisor parameter optimization.

Now we should define a frame. Frame is some sort of a data structure of a single optimization pass. During optimization, frames are saved to the **\*.mqd** archive created in the **MetaTrader 5/MQL5/Files/Tester** folder. Data (frames) of this archive can be accessed both during the optimization "on the fly" and after its completion. For example, the article ["Visualize a Strategy in the MetaTrader 5 Tester"](https://www.mql5.com/en/articles/403) illustrates how we can visualize the process of the optimization "on the fly" and then view the results following the optimization.

In this article, we will use the following functions for working with frames:

- [FrameAdd()](https://www.mql5.com/en/docs/optimization_frames/frameadd) \- adds data from a file or array.
- [FrameNext()](https://www.mql5.com/en/docs/optimization_frames/framenext) \- a call to get a single numerical value or the entire frame data.
- [FrameInputs()](https://www.mql5.com/en/docs/optimization_frames/frameinputs) \- gets input parameters based on which a given frame with the specified pass number is formed.

Further information about the above-listed functions can be found in the [MQL5 Reference](https://www.mql5.com/en/docs). As usual, we start with external parameters. Below you can see what parameters should be added to the already existing ones:

```
//--- External parameters of the Expert Advisor
input  int              NumberOfBars           = 2;              // Number of one-direction bars
sinput double           Lot                    = 0.1;            // Lot
input  double           TakeProfit             = 100;            // Take Profit
input  double           StopLoss               = 50;             // Stop Loss
input  double           TrailingStop           = 10;             // Trailing Stop
input  bool             Reverse                = true;           // Position reversal
sinput string           delimeter=""; // --------------------------------
sinput bool             LogOptimizationReport  = true;           // Writing results to a file
sinput CRITERION_RULE   CriterionSelectionRule = RULE_AND;       // Condition for writing
sinput ENUM_STATS       Criterion_01           = C_NO_CRITERION; // 01 - Criterion name
sinput double           CriterionValue_01      = 0;              // ---- Criterion value
sinput ENUM_STATS       Criterion_02           = C_NO_CRITERION; // 02 - Criterion name
sinput double           CriterionValue_02      = 0;              // ---- Criterion value
sinput ENUM_STATS       Criterion_03           = C_NO_CRITERION; // 03 - Criterion name
sinput double           CriterionValue_03      = 0;              // ---- Criterion value
```

The **LogOptimizationReport** parameter will be used to indicate whether the results and parameters should or should not be written to a file during the optimization.

In this example, we will implement the possibility of specifying up to three criteria based on which the results will be selected to be written to a file. We will also add a rule ( **CriterionSelection** **Rule** parameter) where you can specify whether the results will be written if either all the given conditions are satisfied ( **AND**) or if at least one of them ( **OR**) is met. For this purpose, we create an enumeration in the **Enums.mqh** file:

```
//--- Rules for checking criteria
enum CRITERION_RULE
  {
   RULE_AND = 0, // AND
   RULE_OR  = 1  // OR
  };
```

The main test parameters will be used as criteria. Here, we need another enumeration:

```
//--- Statistical parameters
enum ENUM_STATS
  {
   C_NO_CRITERION              = 0, // No criterion
   C_STAT_PROFIT               = 1, // Profit
   C_STAT_DEALS                = 2, // Total deals
   C_STAT_PROFIT_FACTOR        = 3, // Profit factor
   C_STAT_EXPECTED_PAYOFF      = 4, // Expected payoff
   C_STAT_EQUITY_DDREL_PERCENT = 5, // Max. equity drawdown, %
   C_STAT_RECOVERY_FACTOR      = 6, // Recovery factor
   C_STAT_SHARPE_RATIO         = 7  // Sharpe ratio
  };
```

Each parameter will be checked for exceeding the value specified in the external parameters, with the exception of max. equity drawdown as the selection must be done based on the min. drawdown.

We also need to add a few global variables (see the code below):

```
//--- Global variables
int    AllowedNumberOfBars=0;       // For checking the NumberOfBars external parameter value
string OptimizationResultsPath="";  // Location of the folder for saving folders and files
int    UsedCriteriaCount=0;         // Number of used criteria
int    OptimizationFileHandle=-1;   // Handle of the file of optimization results
```

Furthermore, the following arrays are required:

```
int    criteria[3];                    // Criteria for optimization report generation
double criteria_values[3];             // Values of criteria
double stat_values[STAT_VALUES_COUNT]; // Array for testing parameters
```

The main file of the Expert Advisor needs to be enhanced with functions for handling Strategy Tester events described at the beginning of the article:

```
//+--------------------------------------------------------------------+
//| Optimization start                                                 |
//+--------------------------------------------------------------------+
void OnTesterInit()
  {
   Print(__FUNCTION__,"(): Start Optimization \n-----------");
  }
//+--------------------------------------------------------------------+
//| Test completion event handler                                      |
//+--------------------------------------------------------------------+
double OnTester()
  {
//--- If writing of optimization results is enabled
   if(LogOptimizationReport)
      //---
//---
   return(0.0);
  }
//+--------------------------------------------------------------------+
//| Next optimization pass                                             |
//+--------------------------------------------------------------------+
void OnTesterPass()
  {
//--- If writing of optimization results is enabled
   if(LogOptimizationReport)
      //---
  }
//+--------------------------------------------------------------------+
//| End of optimization                                                |
//+--------------------------------------------------------------------+
void OnTesterDeinit()
  {
   Print("-----------\n",__FUNCTION__,"(): End Optimization");
//--- If writing of optimization results is enabled
   if(LogOptimizationReport)
      //---
  }
```

If we start the optimization now, the chart with the symbol and time frame on which the Expert Advisor is running will appear in the terminal. Messages from the functions used in the above code will be printed to the journal of the terminal instead of the journal of the Strategy Tester. A message from the [OnTesterInit()](https://www.mql5.com/en/docs/basis/function/events#ontesterinit) function will be printed at the very beginning of the optimization. But during the optimization and upon its completion, you will not be able to see any messages in the journal. If after the optimization you delete the chart opened by the Strategy Tester, a message from the [OnTesterDeinit()](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) function will be printed to the journal. Why is that?

The thing is that in order to ensure the correct operation, the [OnTester()](https://www.mql5.com/en/docs/basis/function/events#ontester) function needs to use the [FrameAdd()](https://www.mql5.com/en/docs/optimization_frames/frameadd) function to add a frame, as shown below.

```
//+--------------------------------------------------------------------+
//| Test completion event handler                                      |
//+--------------------------------------------------------------------+
double OnTester()
  {
//--- If writing of optimization results is enabled
   if(LogOptimizationReport)
     {
      //--- Create a frame
      FrameAdd("Statistics",1,0,stat_values);
     }
//---
   return(0.0);
  }
```

Now, during the optimization, a message from the [OnTesterPass()](https://www.mql5.com/en/docs/basis/function/events#ontesterpass) function will be printed to the journal after each optimization pass and the the message regarding the optimization completion will be added after the end of optimization by the [OnTesterDeinit()](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) function. The optimization completion message will also be generated if the optimization is stopped manually.

![Fig.1 - Messages from testing and optimization functions printed to the journal](https://c.mql5.com/2/6/en_01.png)

Fig.1 - Messages from testing and optimization functions printed to the journal

Everything is now ready to proceed to functions responsible for creating folders and files, determining optimization parameters specified and writing the results that satisfy the conditions.

Let's create a file, **FileFunctions.mqh**, and include it in the project. At the very beginning of this file, we write the **GetTestStatistics**() function that will by reference get an array for filling each optimization pass with values.

```
//+--------------------------------------------------------------------+
//| Filling the array with test results                                |
//+--------------------------------------------------------------------+
void GetTestStatistics(double &stat_array[])
  {
//--- Auxiliary variables for value adjustment
   double profit_factor=0,sharpe_ratio=0;
//---
   stat_array[0]=TesterStatistics(STAT_PROFIT);                // Net profit upon completion of testing
   stat_array[1]=TesterStatistics(STAT_DEALS);                 // Number of executed deals
//---
   profit_factor=TesterStatistics(STAT_PROFIT_FACTOR);         // Profit factor – the STAT_GROSS_PROFIT/STAT_GROSS_LOSS ratio
   stat_array[2]=(profit_factor==DBL_MAX) ? 0 : profit_factor; // adjust if necessary
//---
   stat_array[3]=TesterStatistics(STAT_EXPECTED_PAYOFF);       // Expected payoff
   stat_array[4]=TesterStatistics(STAT_EQUITY_DDREL_PERCENT);  // Max. equity drawdown, %
   stat_array[5]=TesterStatistics(STAT_RECOVERY_FACTOR);       // Recovery factor – the STAT_PROFIT/STAT_BALANCE_DD ratio
//---
   sharpe_ratio=TesterStatistics(STAT_SHARPE_RATIO);           // Sharpe ratio - investment portfolio (asset) efficiency index
   stat_array[6]=(sharpe_ratio==DBL_MAX) ? 0 : sharpe_ratio;   // adjust if necessary
  }
```

The **GetTestStatistics**() function must be inserted before adding a frame:

```
//+--------------------------------------------------------------------+
//| Test completion event handler                                      |
//+--------------------------------------------------------------------+
double OnTester()
  {
//--- If writing of optimization results is enabled
   if(LogOptimizationReport)
     {
      //--- Fill the array with test values
      GetTestStatistics(stat_values);
      //--- Create a frame
      FrameAdd("Statistics",1,0,stat_values);
     }
//---
   return(0.0);
  }
```

The filled array is passed to the [FrameAdd()](https://www.mql5.com/en/docs/optimization_frames/frameadd) function as the last argument. You can even pass a data file, if necessary.

In the [OnTesterPass()](https://www.mql5.com/en/docs/basis/function/events#ontesterpass) function, we can now check the obtained data. To see how it works, we will for now simply display the profit for each result in the terminal journal. Use [FrameNext()](https://www.mql5.com/en/docs/optimization_frames/framenext) to get the current frame values. Please see the below example:

```
//+--------------------------------------------------------------------+
//| Next optimization pass                                             |
//+--------------------------------------------------------------------+
void OnTesterPass()
  {
//--- If writing of optimization results is enabled
   if(LogOptimizationReport)
     {
      string name ="";  // Public name/frame label
      ulong  pass =0;   // Number of the optimization pass at which the frame is added
      long   id   =0;   // Public id of the frame
      double val  =0.0; // Single numerical value of the frame
      //---
      FrameNext(pass,name,id,val,stat_values);
      //---
      Print(__FUNCTION__,"(): pass: "+IntegerToString(pass)+"; STAT_PROFIT: ",DoubleToString(stat_values[0],2));
     }
  }
```

If you do not use the [FrameNext()](https://www.mql5.com/en/docs/optimization_frames/framenext) function, the values in the **stat\_values** array will be zero. If, however, everything is done correctly, we will get the result as shown in screenshot below:

![Fig. 2 - Messages from the OnTesterPass() function printed to the journal](https://c.mql5.com/2/6/en_02.png)

Fig. 2 - Messages from the OnTesterPass() function printed to the journal

By the way, if the optimization is run without modifying the external parameters, the results will be loaded to the Strategy Tester from cache, bypassing the [OnTesterPass()](https://www.mql5.com/en/docs/basis/function/events#ontesterpass) and [OnTesterDeinit()](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) functions. You should bear this in mind not to think that there is an error.

Further, in **FileFunctions.mqh** we create a **CreateOptimizationReport**() function. The key activity will be performed within this function. The function code is provided below:

```
//+--------------------------------------------------------------------+
//| Generating and writing report on optimization results              |
//+--------------------------------------------------------------------+
void CreateOptimizationReport()
  {
   static int passes_count=0;                // Pass counter
   int        parameters_count=0;            // Number of Expert Advisor parameters
   int        optimized_parameters_count=0;  // Counter of optimized parameters
   string     string_to_write="";            // String for writing
   bool       include_criteria_list=false;   // For determining the start of the list of parameters/criteria
   int        equality_sign_index=0;         // The '=' sign index in the string
   string     name            ="";           // Public name/frame label
   ulong      pass            =0;            // Number of the optimization pass at which the frame is added
   long       id              =0;            // Public id of the frame
   double     value           =0.0;          // Single numerical value of the frame
   string     parameters_list[];             // List of the Expert Advisor parameters of the "parameterN=valueN" form
   string     parameter_names[];             // Array of parameter names
   string     parameter_values[];            // Array of parameter values
//--- Increase the pass counter
   passes_count++;
//--- Place statistical values into the array
   FrameNext(pass,name,id,value,stat_values);
//--- Get the pass number, list of parameters, number of parameters
   FrameInputs(pass,parameters_list,parameters_count);
//--- Iterate over the list of parameters in a loop (starting from the upper one on the list)
//    The list starts with the parameters that are flagged for optimization
   for(int i=0; i<parameters_count; i++)
     {
      //--- Get the criteria for selection of results at the first pass
      if(passes_count==1)
        {
         string current_value="";      // Current parameter value
         static int c=0,v=0,trigger=0; // Counters and trigger
         //--- Set a flag if you reached the list of criteria
         if(StringFind(parameters_list[i],"CriterionSelectionRule",0)>=0)
           {
            include_criteria_list=true;
            continue;
           }
         //--- At the last parameter, count the used criteria,
         //    if the AND mode is selected
         if(CriterionSelectionRule==RULE_AND && i==parameters_count-1)
            CalculateUsedCriteria();
         //--- If you reached criteria in the parameter list
         if(include_criteria_list)
           {
            //--- Determine names of criteria
            if(trigger==0)
              {
               equality_sign_index=StringFind(parameters_list[i],"=",0)+1;          // Determine the '=' sign position in the string
               current_value =StringSubstr(parameters_list[i],equality_sign_index); // Get the parameter value
               //---
               criteria[c]=(int)StringToInteger(current_value);
               trigger=1; // Next parameter will be a value
               c++;
               continue;
              }
            //--- Determine values of criteria
            if(trigger==1)
              {
               equality_sign_index=StringFind(parameters_list[i],"=",0)+1;          // Determine the '=' sign position in the string
               current_value=StringSubstr(parameters_list[i],equality_sign_index);  // Get the parameter value
               //---
               criteria_values[v]=StringToDouble(current_value);
               trigger=0; // Next parameter will be a criterion
               v++;
               continue;
              }
           }
        }
      //--- If the parameter is enabled for optimization
      if(ParameterEnabledForOptimization(parameters_list[i]))
        {
         //--- Increase the counter of the optimized parameters
         optimized_parameters_count++;
         //--- Write the names of the optimized parameters to the array
         //    only at the first pass (for headers)
         if(passes_count==1)
           {
            //--- Increase the size of the array of parameter values
            ArrayResize(parameter_names,optimized_parameters_count);
            //--- Determine the '=' sign position
            equality_sign_index=StringFind(parameters_list[i],"=",0);
            //--- Take the parameter name
            parameter_names[i]=StringSubstr(parameters_list[i],0,equality_sign_index);
           }
         //--- Increase the size of the array of parameter values
         ArrayResize(parameter_values,optimized_parameters_count);
         //--- Determine the '=' sign position
         equality_sign_index=StringFind(parameters_list[i],"=",0)+1;
         //--- Take the parameter value
         parameter_values[i]=StringSubstr(parameters_list[i],equality_sign_index);
        }
     }
//--- Generate a string of values to the optimized parameters
   for(int i=0; i<STAT_VALUES_COUNT; i++)
      StringAdd(string_to_write,DoubleToString(stat_values[i],2)+",");
//--- Add values of the optimized parameters to the string of values
   for(int i=0; i<optimized_parameters_count; i++)
     {
      //--- If it is the last value in the string, do not use the separator
      if(i==optimized_parameters_count-1)
        {
         StringAdd(string_to_write,parameter_values[i]);
         break;
        }
      //--- Otherwise use the separator
      else
         StringAdd(string_to_write,parameter_values[i]+",");
     }
//--- At the first pass, generate the optimization report file with headers
   if(passes_count==1)
      WriteOptimizationReport(parameter_names);
//--- Write data to the file of optimization results
   WriteOptimizationResults(string_to_write);
  }
```

We have got a quite large function. Let's have a closer look at it. At the very beginning, right after declaring the variables and arrays, we get the frame data using the [FrameNext()](https://www.mql5.com/en/docs/optimization_frames/framenext) function as demonstrated in the examples given above. Then, using the [FrameInputs()](https://www.mql5.com/en/docs/optimization_frames/frameinputs) function, we get the list of parameters to the **parameters\_list\[\]** string array, along with the total number of parameters that is passed to the **parameters\_count** variable.

The optimized parameters (flagged in the Strategy Tester) in the parameter list received from the [FrameInputs()](https://www.mql5.com/en/docs/optimization_frames/frameinputs) function are located at the very beginning, irrespective of their order in the list of external parameters of the Expert Advisor.

This is followed by the loop that iterates over the list of parameters. The array of criteria **criteria\[\]** and the array of values of criteria **criteria\_values\[\]** are filled at the very first pass. The criteria used are counted in the **CalculateUsedCriteria()** function, provided that the **AND** mode is enabled and the current parameter is the last one:

```
//+--------------------------------------------------------------------+
//| Counting the number of used criteria                               |
//+--------------------------------------------------------------------+
void CalculateUsedCriteria()
  {
   UsedCriteriaCount=0; // Zeroing out
//--- Iterate over the list of criteria in a loop
   for(int i=0; i<ArraySize(criteria); i++)
     {
      //--- count the used criteria
      if(criteria[i]!=C_NO_CRITERION)
         UsedCriteriaCount++;
     }
  }
```

In the same loop we further check if any given parameter is selected for optimization. The check is performed at every pass and is done using the **ParameterEnabledForOptimization()** function to which the current external parameter is passed for checking. If the function returns true, the parameter will be optimized.

```
//+---------------------------------------------------------------------+
//| Checking whether the external parameter is enabled for optimization |
//+---------------------------------------------------------------------+
bool ParameterEnabledForOptimization(string parameter_string)
  {
   bool enable;
   long value,start,step,stop;
//--- Determine the '=' sign position in the string
   int equality_sign_index=StringFind(parameter_string,"=",0);
//--- Get the parameter values
   ParameterGetRange(StringSubstr(parameter_string,0,equality_sign_index),
                     enable,value,start,step,stop);
//--- Return the parameter status
   return(enable);
  }
```

In this case, the arrays for names **parameter\_names** and parameter values **parameter\_values** are filled. The array for optimized parameter names is only filled at the first pass.

Then, using two loops, we generate the string of test and parameter values for writing to a file. Following that the file for writing is generated using the **WriteOptimizationReport()** function at the first pass.

```
//+--------------------------------------------------------------------+
//| Generating the optimization report file                            |
//+--------------------------------------------------------------------+
void WriteOptimizationReport(string &parameter_names[])
  {
   int files_count     =1; // Counter of optimization files
//--- Generate a header to the optimized parameters
   string headers="#,PROFIT,TOTAL DEALS,PROFIT FACTOR,EXPECTED PAYOFF,EQUITY DD MAX REL%,RECOVERY FACTOR,SHARPE RATIO,";
//--- Add the optimized parameters to the header
   for(int i=0; i<ArraySize(parameter_names); i++)
     {
      if(i==ArraySize(parameter_names)-1)
         StringAdd(headers,parameter_names[i]);
      else
         StringAdd(headers,parameter_names[i]+",");
     }
//--- Get the location for the optimization file and the number of files for the index number
   OptimizationResultsPath=CreateOptimizationResultsFolder(files_count);
//--- If there is an error when getting the folder, exit
   if(OptimizationResultsPath=="")
     {
      Print("Empty path: ",OptimizationResultsPath);
      return;
     }
   else
     {
      OptimizationFileHandle=FileOpen(OptimizationResultsPath+"\optimization_results"+IntegerToString(files_count)+".csv",
                                      FILE_CSV|FILE_READ|FILE_WRITE|FILE_ANSI|FILE_COMMON,",");
      //---
      if(OptimizationFileHandle!=INVALID_HANDLE)
         FileWrite(OptimizationFileHandle,headers);
     }
  }
```

The purpose of the **WriteOptimizationReport()** function is to generate headers, create folders, if necessary, in the common folder of the terminal, as well as to create a file for writing. That is, files associated with previous optimizations are not removed and the function every time creates a new file with the index number. Headers are saved in a newly created file. The file itself remains open until the end of optimization.

The above code contains the string with the **CreateOptimizationResultsFolder()** function, where folders for saving files with optimization results are created:

```
//+--------------------------------------------------------------------+
//| Creating folders for optimization results                          |
//+--------------------------------------------------------------------+
string CreateOptimizationResultsFolder(int &files_count)
  {
   long   search_handle       =INVALID_HANDLE;        // Search handle
   string returned_filename   ="";                    // Name of the found object (file/folder)
   string path                ="";                    // File/folder search location
   string search_filter       ="*";                   // Search filter (* - check all files/folders)
   string root_folder         ="OPTIMIZATION_DATA\\"; // Root folder
   string expert_folder       =EXPERT_NAME+"\\";      // Folder of the Expert Advisor
   bool   root_folder_exists  =false;                 // Flag of existence of the root folder
   bool   expert_folder_exists=false;                 // Flag of existence of the Expert Advisor folder
//--- Search for the OPTIMIZATION_DATA root folder in the common folder of the terminal
   path=search_filter;
//--- Set the search handle in the common folder of all client terminals \Files
   search_handle=FileFindFirst(path,returned_filename,FILE_COMMON);
//--- Print the location of the common folder of the terminal to the journal
   Print("TERMINAL_COMMONDATA_PATH: ",COMMONDATA_PATH);
//--- If the first folder is the root folder, flag it
   if(returned_filename==root_folder)
     {
      root_folder_exists=true;
      Print("The "+root_folder+" root folder exists.");
     }
//--- If the search handle has been obtained
   if(search_handle!=INVALID_HANDLE)
     {
      //--- If the first folder is not the root folder
      if(!root_folder_exists)
        {
         //--- Iterate over all files to find the root folder
         while(FileFindNext(search_handle,returned_filename))
           {
            //--- If it is found, flag it
            if(returned_filename==root_folder)
              {
               root_folder_exists=true;
               Print("The "+root_folder+" root folder exists.");
               break;
              }
           }
        }
      //--- Close the root folder search handle
      FileFindClose(search_handle);
     }
   else
     {
      Print("Error when getting the search handle "
            "or the "+COMMONDATA_PATH+" folder is empty: ",ErrorDescription(GetLastError()));
     }
//--- Search for the Expert Advisor folder in the OPTIMIZATION_DATA folder
   path=root_folder+search_filter;
//--- Set the search handle in the ..\Files\OPTIMIZATION_DATA\ folder
   search_handle=FileFindFirst(path,returned_filename,FILE_COMMON);
//--- If the first folder is the folder of the Expert Advisor
   if(returned_filename==expert_folder)
     {
      expert_folder_exists=true; // Remember this
      Print("The "+expert_folder+" Expert Advisor folder exists.");
     }
//--- If the search handle has been obtained
   if(search_handle!=INVALID_HANDLE)
     {
      //--- If the first folder is not the folder of the Expert Advisor
      if(!expert_folder_exists)
        {
         //--- Iterate over all files in the DATA_OPTIMIZATION folder to find the folder of the Expert Advisor
         while(FileFindNext(search_handle,returned_filename))
           {
            //--- If it is found, flag it
            if(returned_filename==expert_folder)
              {
               expert_folder_exists=true;
               Print("The "+expert_folder+" Expert Advisor folder exists.");
               break;
              }
           }
        }
      //--- Close the root folder search handle
      FileFindClose(search_handle);
     }
   else
      Print("Error when getting the search handle or the "+path+" folder is empty.");
//--- Generate the path to count the files
   path=root_folder+expert_folder+search_filter;
//--- Set the search handle in the ..\Files\OPTIMIZATION_DATA\ folder of optimization results
   search_handle=FileFindFirst(path,returned_filename,FILE_COMMON);
//--- If the folder is not empty, start the count
   if(StringFind(returned_filename,"optimization_results",0)>=0)
      files_count++;
//--- If the search handle has been obtained
   if(search_handle!=INVALID_HANDLE)
     {
      //--- Count all files in the Expert Advisor folder
      while(FileFindNext(search_handle,returned_filename))
         files_count++;
      //---
      Print("Total files: ",files_count);
      //--- Close the Expert Advisor folder search handle
      FileFindClose(search_handle);
     }
   else
      Print("Error when getting the search handle or the "+path+" folder is empty");
//--- Create the necessary folders based on the check results
//    If there is no OPTIMIZATION_DATA root folder
   if(!root_folder_exists)
     {
      if(FolderCreate("OPTIMIZATION_DATA",FILE_COMMON))
        {
         root_folder_exists=true;
         Print("The root folder ..\Files\OPTIMIZATION_DATA\\ has been created");
        }
      else
        {
         Print("Error when creating the OPTIMIZATION_DATA root folder: ",
               ErrorDescription(GetLastError()));
         return("");
        }
     }
//--- If there is no Expert Advisor folder
   if(!expert_folder_exists)
     {
      if(FolderCreate(root_folder+EXPERT_NAME,FILE_COMMON))
        {
         expert_folder_exists=true;
         Print("The Expert Advisor folder ..\Files\OPTIMIZATION_DATA\\ has been created"+expert_folder);
        }
      else
        {
         Print("Error when creating the Expert Advisor folder ..\Files\\"+expert_folder+"\: ",
               ErrorDescription(GetLastError()));
         return("");
        }
     }
//--- If the necessary folders exist
   if(root_folder_exists && expert_folder_exists)
     {
      //--- Return the location for creating the file of optimization results
      return(root_folder+EXPERT_NAME);
     }
//---
   return("");
  }
```

The above code is provided with the detailed comments so you should not face any difficulty in understanding it. Let's just outline the key points.

First, we check for the **OPTIMIZATION\_DATA** root folder containing the results of the optimization. If the folder exists, this is marked in the **root\_folder\_exists** variable. The search handle is then set in the **OPTIMIZATION\_DATA** folder where we check for the Expert Advisor folder.

We further count the files that the Expert Advisor folder contains. Finally, based on the check results, where necessary (if the folders could not be found), the required folders are created and the location for the new file with the index number is returned. If an error has occurred, an empty string will be returned.

Now, we only need to consider the **WriteOptimizationResults**() function where we check the conditions for writing data to the file and write the data if the condition is met. The code of this function is provided below:

```
//+--------------------------------------------------------------------+
//| Writing the results of the optimization by criteria                |
//+--------------------------------------------------------------------+
void WriteOptimizationResults(string string_to_write)
  {
   bool condition=false; // To check the condition
//--- If at least one criterion is satisfied
   if(CriterionSelectionRule==RULE_OR)
      condition=AccessCriterionOR();
//--- If all criteria are satisfied
   if(CriterionSelectionRule==RULE_AND)
      condition=AccessCriterionAND();
//--- If the conditions for criteria are satisfied
   if(condition)
     {
      //--- If the file of optimization results is opened
      if(OptimizationFileHandle!=INVALID_HANDLE)
        {
         int strings_count=0; // String counter
         //--- Get the number of strings in the file and move the pointer to the end
         strings_count=GetStringsCount();
         //--- Write the string with criteria
         FileWrite(OptimizationFileHandle,IntegerToString(strings_count),string_to_write);
        }
      else
         Print("Invalid optimization file handle!");
     }
  }
```

Let's take a look at the strings that contain the functions highlighted in the code. The choice of the function used depends on the rule selected for checking the criteria. If all the specified criteria need to be satisfied, we use the **AccessCriterionAND**() function:

```
//+--------------------------------------------------------------------+
//| Checking multiple conditions for writing to the file               |
//+--------------------------------------------------------------------+
bool AccessCriterionAND()
  {
   int count=0; // Criterion counter
//--- Iterate over the array of criteria in a loop and see
//    if all the conditions for writing parameters to the file are met
   for(int i=0; i<ArraySize(criteria); i++)
     {
      //--- Move to the next iteration, if the criterion is not determined
      if(criteria[i]==C_NO_CRITERION)
         continue;
      //--- PROFIT
      if(criteria[i]==C_STAT_PROFIT)
        {
         if(stat_values[0]>criteria_values[i])
           {
            count++;
            if(count==UsedCriteriaCount)
               return(true);
           }
        }
      //--- TOTAL DEALS
      if(criteria[i]==C_STAT_DEALS)
        {
         if(stat_values[1]>criteria_values[i])
           {
            count++;
            if(count==UsedCriteriaCount)
               return(true);
           }
        }
      //--- PROFIT FACTOR
      if(criteria[i]==C_STAT_PROFIT_FACTOR)
        {
         if(stat_values[2]>criteria_values[i])
           {
            count++;
            if(count==UsedCriteriaCount)
               return(true);
           }
        }
      //--- EXPECTED PAYOFF
      if(criteria[i]==C_STAT_EXPECTED_PAYOFF)
        {
         if(stat_values[3]>criteria_values[i])
           {
            count++;
            if(count==UsedCriteriaCount)
               return(true);
           }
        }
      //--- EQUITY DD REL PERC
      if(criteria[i]==C_STAT_EQUITY_DDREL_PERCENT)
        {
         if(stat_values[4]<criteria_values[i])
           {
            count++;
            if(count==UsedCriteriaCount)
               return(true);
           }
        }
      //--- RECOVERY FACTOR
      if(criteria[i]==C_STAT_RECOVERY_FACTOR)
        {
         if(stat_values[5]>criteria_values[i])
           {
            count++;
            if(count==UsedCriteriaCount)
               return(true);
           }
        }
      //--- SHARPE RATIO
      if(criteria[i]==C_STAT_SHARPE_RATIO)
        {
         if(stat_values[6]>criteria_values[i])
           {
            count++;
            if(count==UsedCriteriaCount)
               return(true);
           }
        }
     }
//--- Conditions for writing are not met
   return(false);
  }
```

If you need at least one of the specified criteria to be satisfied, use the **AccessCriterionOR**() function:

```
//+--------------------------------------------------------------------+
//| Checking for meeting one of the conditions for writing to the file |
//+--------------------------------------------------------------------+
bool AccessCriterionOR()
  {
//--- Iterate over the array of criteria in a loop and see
//    if all the conditions for writing parameters to the file are met
   for(int i=0; i<ArraySize(criteria); i++)
     {
     //---
      if(criteria[i]==C_NO_CRITERION)
         continue;
      //--- PROFIT
      if(criteria[i]==C_STAT_PROFIT)
        {
         if(stat_values[0]>criteria_values[i])
            return(true);
        }
      //--- TOTAL DEALS
      if(criteria[i]==C_STAT_DEALS)
        {
         if(stat_values[1]>criteria_values[i])
            return(true);
        }
      //--- PROFIT FACTOR
      if(criteria[i]==C_STAT_PROFIT_FACTOR)
        {
         if(stat_values[2]>criteria_values[i])
            return(true);
        }
      //--- EXPECTED PAYOFF
      if(criteria[i]==C_STAT_EXPECTED_PAYOFF)
        {
         if(stat_values[3]>criteria_values[i])
            return(true);
        }
      //--- EQUITY DD REL PERC
      if(criteria[i]==C_STAT_EQUITY_DDREL_PERCENT)
        {
         if(stat_values[4]<criteria_values[i])
            return(true);
        }
      //--- RECOVERY FACTOR
      if(criteria[i]==C_STAT_RECOVERY_FACTOR)
        {
         if(stat_values[5]>criteria_values[i])
            return(true);
        }
      //--- SHARPE RATIO
      if(criteria[i]==C_STAT_SHARPE_RATIO)
        {
         if(stat_values[6]>criteria_values[i])
            return(true);
        }
     }
//--- Conditions for writing are not met
   return(false);
  }
```

The **GetStringsCount()** function moves the pointer to the end of the file and returns the number of strings in the file:

```
//+--------------------------------------------------------------------+
//| Counting the number of strings in the file                         |
//+--------------------------------------------------------------------+
int GetStringsCount()
  {
   int   strings_count =0;  // String counter
   ulong offset        =0;  // Offset for determining the position of the file pointer
//--- Move the file pointer to the beginning
   FileSeek(OptimizationFileHandle,0,SEEK_SET);
//--- Read until the current position of the file pointer reaches the end of the file
   while(!FileIsEnding(OptimizationFileHandle) || !IsStopped())
     {
      //--- Read the whole string
      while(!FileIsLineEnding(OptimizationFileHandle) || !IsStopped())
        {
         //--- Read the string
         FileReadString(OptimizationFileHandle);
         //--- Get the position of the pointer
         offset=FileTell(OptimizationFileHandle);
         //--- If it's the end of the string
         if(FileIsLineEnding(OptimizationFileHandle))
           {
            //--- Move to the next string
            //    if it's not the end of the file, increase the pointer counter
            if(!FileIsEnding(OptimizationFileHandle))
               offset++;
            //--- Move the pointer
            FileSeek(OptimizationFileHandle,offset,SEEK_SET);
            //--- Increase the string counter
            strings_count++;
            break;
           }
        }
      //--- If it's the end of the file, exit the loop
      if(FileIsEnding(OptimizationFileHandle))
         break;
     }
//--- Move the pointer to the end of the file for writing
   FileSeek(OptimizationFileHandle,0,SEEK_END);
//--- Return the number of strings
   return(strings_count);
  }
```

Everything is set and ready now. Now we need to insert the **CreateOptimizationReport()** function to the [OnTesterPass()](https://www.mql5.com/en/docs/basis/function/events#ontesterpass) function body and close the optimization file handle in the [OnTesterDeinit()](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) function.

Let's now test the Expert Advisor. Its parameters will be optimized using the [MQL5 Cloud Network](https://cloud.mql5.com/) of distributed computing. The Strategy Tester needs to be set as shown in screenshot below:

![Fig. 3 - Strategy Tester settings](https://c.mql5.com/2/6/en_03.png)

Fig. 3 - Strategy Tester settings

We will optimize all parameters of the Expert Advisor and set the parameters of the criteria so that only the results where **Profit Factor** is greater than 1 and **Recovery Factor** is greater than 2 are written to the file (see the screenshot below):

![Fig. 4 - The Expert Advisor settings for parameter optimization](https://c.mql5.com/2/6/en_04.png)

Fig. 4 - The Expert Advisor settings for parameter optimization

The [MQL5 Cloud Network](https://cloud.mql5.com/en) of distributed computing has processed 101,000 passes in just ~5 minutes! If I hadn't used the network resources, the optimization would have taken several days to complete. That is a great opportunity for all who know the value of time.

The resulting file can now be opened in Excel. 719 results have been selected out of 101,000 passes to be written to the file. In the screenshot below, I highlighted the columns with the parameters based on which the results were selected:

![Fig. 5 - Optimization results in Excel](https://c.mql5.com/2/6/en_05.png)

Fig. 5 - Optimization results in Excel

### Conclusion

It is time to draw a line under this article. The subject of analysis of optimization results is in fact far from being fully exhausted and we will certainly get back to it in the future articles. Attached to the article is the downloadable archive with the files of the Expert Advisor for your consideration.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/746](https://www.mql5.com/ru/articles/746)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/746.zip "Download all attachments in the single ZIP archive")

[writeoptimizationresults.zip](https://www.mql5.com/en/articles/download/746/writeoptimizationresults.zip "Download writeoptimizationresults.zip")(21.18 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/14592)**
(25)


![Thyago Sousa Mendes](https://c.mql5.com/avatar/2018/3/5AB8686B-E220.jpg)

**[Thyago Sousa Mendes](https://www.mql5.com/en/users/thyagomendes)**
\|
1 May 2019 at 20:13

Please, Is it possible to save the equity curve for each [parameter](https://www.mql5.com/en/docs/directx/dxinputset "MQL5 Documentation: DXInputSet function") set?


![Daniel Molnar](https://c.mql5.com/avatar/avatar_na2.png)

**[Daniel Molnar](https://www.mql5.com/en/users/pcdeni)**
\|
25 May 2019 at 21:59

Is it possible to save only the used parameters for the top 10 passes from the optimization results?

With higher pass numbers the logged results file (starting from terminal) can be tens of gigabytes, which is unnecessary. So I was wondering if in the [OnTesterDeinit()](https://www.mql5.com/en/docs/basis/function/events#ontesterdeinit) function, results can be accessed and only a small part of it saved into a file?

![Nextor](https://c.mql5.com/avatar/avatar_na2.png)

**[Nextor](https://www.mql5.com/en/users/nexxtor)**
\|
24 Jul 2021 at 21:54

Doesn't work, saves less than half of the results when optimised in the cloud.


![SidhuKid](https://c.mql5.com/avatar/avatar_na2.png)

**[SidhuKid](https://www.mql5.com/en/users/sidhukid)**
\|
5 May 2022 at 10:33

Thanks this article helped me to solve the issue I had with reports.


![Saad Janah](https://c.mql5.com/avatar/2023/2/63e3bc3f-5677.png)

**[Saad Janah](https://www.mql5.com/en/users/jukerone)**
\|
30 Mar 2024 at 18:49

Great useful article on optimization reposts but, when compiled, there are a lot of errors and warnings displayed, maybe due to the changes made to the MQL5 syntax.

I fixed most of those errors and warnings, but the reports are not generated when an optimization is executed.


![Marvel Your MQL5 Customers with a Usable Cocktail of Technologies!](https://c.mql5.com/2/0/cocktails.png)[Marvel Your MQL5 Customers with a Usable Cocktail of Technologies!](https://www.mql5.com/en/articles/728)

MQL5 provides programmers with a very complete set of functions and object-oriented API thanks to which they can do everything they want within the MetaTrader environment. However, Web Technology is an extremely versatile tool nowadays that may come to the rescue in some situations when you need to do something very specific, want to marvel your customers with something different or simply you do not have enough time to master a specific part of MT5 Standard Library. Today's exercise walks you through a practical example about how you can manage your development time at the same time as you also create an amazing tech cocktail.

![Lite_EXPERT2.mqh: Functional Kit for Developers of Expert Advisors](https://c.mql5.com/2/17/812_123.gif)[Lite\_EXPERT2.mqh: Functional Kit for Developers of Expert Advisors](https://www.mql5.com/en/articles/1380)

This article continues the series of articles "Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization". It familiarizes the readers with a more universal function library of the Lite\_EXPERT2.mqh file.

![Technical Indicators and Digital Filters](https://c.mql5.com/2/0/Indicators_as_digital_filters_MQL5.png)[Technical Indicators and Digital Filters](https://www.mql5.com/en/articles/736)

In this article, technical indicators are treated as digital filters. Operation principles and basic characteristics of digital filters are explained. Also, some practical ways of receiving the filter kernel in MetaTrader 5 terminal and integration with a ready-made spectrum analyzer proposed in the article "Building a Spectrum Analyzer" are considered. Pulse and spectrum characteristics of the typical digital filters are used as examples.

![MetaTrader AppStore Results for Q3 2013](https://c.mql5.com/2/0/avatar3.png)[MetaTrader AppStore Results for Q3 2013](https://www.mql5.com/en/articles/769)

Another quarter of the year has passed and we have decided to sum up its results for MetaTrader AppStore - the largest store of trading robots and technical indicators for MetaTrader platforms. More than 500 developers have placed over 1 200 products in the Market by the end of the reported quarter.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=bfgyxzhxqpiumnwcqonkioabhvsvtgtl&ssn=1769093020967697924&ssn_dr=0&ssn_sr=0&fv_date=1769093020&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F746&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Cookbook%3A%20Saving%20Optimization%20Results%20of%20an%20Expert%20Advisor%20Based%20on%20Specified%20Criteria%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909302029880982&fz_uniq=5049334540469643675&sv=2552)

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