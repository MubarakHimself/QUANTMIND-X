---
title: Using Discriminant Analysis to Develop Trading Systems
url: https://www.mql5.com/en/articles/335
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:01:36.933736
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/335&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071516040803854718)

MetaTrader 5 / Trading systems


### Introduction

One of the major tasks of technical analysis is to determine the direction in which the market will move in the near future. From a statistical standpoint, it boils down to selecting indicators and determining their values based on which it would be possible to divide the future market situation into two categories: 1) upward move, 2) downward move.

Discriminant analysis offers one of the ways to decide what kind of indicators and what values allow for better discrimination between these categories. In other words, discriminant analysis enables to build a model that will predict the market direction based on the data received from indicators.

Such analysis is however rather complicated requiring a great amount of data at the input. Therefore it is quite time-consuming to use it manually for analysis of the market situation. Fortunately, the emergence of the [MQL5](https://www.mql5.com/en/docs) language and statistical software has enabled us to automate data selection and preparation and application of the discriminant analysis.

This article gives an example of developing an EA for market data collection. It serves as a tutorial for application of the discriminant analysis for building prognostic model for the FOREX market in [Statistica](https://www.mql5.com/go?link=http://statistica.io/ "http://www.statsoft.com/") software.

### 1\. What Is the Discriminant Analysis?

The discriminant analysis (hereinafter "DA") is one of the pattern recognition methods. Neural networks can be considered a special case of DA. DA is used in the majority of successful defense systems that are based on pattern recognition.

It allows to determine what variables divide (discriminate) the incoming data flow into groups and see the mechanism of such discrimination.

Let us have a look at a simplified example of using DA for the FOREX market. We have data values from [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI), [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") and [Relative Vigor Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rvi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rvi") (RVI) indicators and we need to predict the price direction. As a result of DA, we can get the following.

a. RVI indicator does not contribute to the forecast. So let us exclude it from the analysis.

b. DA has produced two discriminant equations:

1. G1 = a1\*RSI+b1\*MACD+с1, the equation for cases where the price went up;
2. G2 = a2\*RSI+b2\*MACD+с2, the equation for cases where the price went down.

Calculating G1 and G2 at the beginning of each bar, we predict that if G1 > G2, then the price will go up; whereas if G1 < G2, the price will go down.

DA may prove to be useful for initial acquaintance with neural networks. When using DA, we get equations similar to the ones calculated for operation of neural networks. This helps to better understand their structure and preliminarily determine whether it is worth using neural networks in your strategies.

### 2\. Stages of the Discriminant Analysis

The analysis can be divided into several stages.

1. Data preparation;
2. Selection of the best variables from the prepared data;
3. Analysis and testing of the resulting model using test data;
4. Building of the model on the basis of discriminant equations.

Discriminant analysis is a part of almost all modern software packages designed for statistical data analysis. The most popular are [Statistica](https://www.mql5.com/go?link=http://statistica.io/ "http://www.statsoft.com/") (by StatSoft Inc.) and [SPSS](https://www.mql5.com/go?link=http://www-01.ibm.com/software/analytics/spss/products/statistics/ "http://www-01.ibm.com/software/analytics/spss/products/statistics/") (by IBM Corporation). We will further consider the application of the discriminant analysis using Statistica software. The screenshots provided are obtained from Statistica version 8.0. These would look more or less the same in the earlier versions of the software. It should be noted that Statistica offers many other useful tools for the trader including neural networks.

### 2.1. Data Preparation

Data collection depends on a certain task at hand. Let us define the task as follows: using indicators, to predict the direction of the price chart on the bar following the bar with known values of indicators. An EA will be developed for data collection to save indicator values and price data into a file.

This shall be a CSV file with a following structure. Variables shall be arranged in columns where every column corresponds to a certain indicator. The rows shall contain consecutive measurements (cases), i.e. values of indicators for certain bars. In other words, the horizontal table headers contain indicators, the vertical table headers contain consecutive bars.

The table shall have a variable based on which the grouping will be made (the grouping variable). In our case, such variable will be based on the price change on the bar following the bar whose indicator values were obtained. The grouping variable shall contain the number of the group whose data is displayed in the same line. For example, number 1 for cases where the price went up and number 2 for cases where the price went down.

We will need values of the following indicators:

- [Accelerator Oscillator](https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/ao "https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/ao");
- [Bears Power](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears");
- [Bulls Power](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bulls "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bulls");
- [Awesome Oscillator](https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/awesome "https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/awesome");
- [Commodity Channel Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/cci "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/cci");
- [DeMarker](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/demarker "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/demarker");
- [Fractal Adaptive Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/fama "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/fama");
- [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd");
- [Relative Strength Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi");
- [Relative Vigor Index](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rvi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rvi");
- [Stochastic](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so");
- [Williams Percent Range](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/wpr "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/wpr").

The [OnInit()](https://www.mql5.com/en/docs/basis/function/events#ontick) function creates the indicators (obtains indicator handles) and MasterData.csv file where it saves the column data header:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- initialization of the indicators
   h_AC=iAC(Symbol(),Period());
   h_BearsPower=iBearsPower(Symbol(),Period(),BearsPower_PeriodBears);
   h_BullsPower=iBullsPower(Symbol(),Period(),BullsPower_PeriodBulls);
   h_AO=iAO(Symbol(),Period());
   h_CCI=iCCI(Symbol(),Period(),CCI_PeriodCCI,CCI_Applied);
   h_DeMarker=iDeMarker(Symbol(),Period(),DeM_PeriodDeM);
   h_FrAMA=iFrAMA(Symbol(),Period(),FraMA_PeriodMA,FraMA_Shift,FraMA_Applied);
   h_MACD=iMACD(Symbol(),Period(),MACD_PeriodFast,MACD_PeriodSlow,MACD_PeriodSignal,MACD_Applied);
   h_RSI=iRSI(Symbol(),Period(),RSI_PeriodRSI,RSI_Applied);
   h_RVI=iRVI(Symbol(),Period(),RVI_PeriodRVI);
   h_Stoch=iStochastic(Symbol(),Period(),Stoch_PeriodK,Stoch_PeriodD,Stoch_PeriodSlow,MODE_SMA,Stoch_Applied);
   h_WPR=iWPR(Symbol(),Period(),WPR_PeriodWPR);

   if(h_AC==INVALID_HANDLE || h_BearsPower==INVALID_HANDLE ||
      h_BullsPower==INVALID_HANDLE || h_AO==INVALID_HANDLE ||
      h_CCI==INVALID_HANDLE || h_DeMarker==INVALID_HANDLE ||
      h_FrAMA==INVALID_HANDLE || h_MACD==INVALID_HANDLE ||
      h_RSI==INVALID_HANDLE || h_RVI==INVALID_HANDLE ||
      h_Stoch==INVALID_HANDLE || h_WPR==INVALID_HANDLE)
     {
      Print("Error creating indicators");
      return(1);
     }

   ArraySetAsSeries(buf_AC,true);
   ArraySetAsSeries(buf_BearsPower,true);
   ArraySetAsSeries(buf_BullsPower,true);
   ArraySetAsSeries(buf_AO,true);
   ArraySetAsSeries(buf_CCI,true);
   ArraySetAsSeries(buf_DeMarker,true);
   ArraySetAsSeries(buf_FrAMA,true);
   ArraySetAsSeries(buf_MACD_m,true);
   ArraySetAsSeries(buf_MACD_s,true);
   ArraySetAsSeries(buf_RSI,true);
   ArraySetAsSeries(buf_RVI_m,true);
   ArraySetAsSeries(buf_RVI_s,true);
   ArraySetAsSeries(buf_Stoch_m,true);
   ArraySetAsSeries(buf_Stoch_s,true);
   ArraySetAsSeries(buf_WPR,true);

   FileHandle=FileOpen("MasterData2.csv",FILE_ANSI|FILE_WRITE|FILE_CSV|FILE_SHARE_READ,';');
   if(FileHandle!=INVALID_HANDLE)
     {
      Print("FileOpen OK");
      //--- saving names of the variables in the first line of the file for convenience of working with it
      FileWrite(FileHandle,"Time","Hour","Price","AC","dAC","Bears","dBears","Bulls","dBulls",
                "AO","dAO","CCI","dCCI","DeMarker","dDeMarker","FrAMA","dFrAMA","MACDm","dMACDm",
                "MACDs","dMACDs","MACDms","dMACDms","RSI","dRSI","RVIm","dRVIm","RVIs","dRVIs",
                "RVIms","dRVIms","Stoch_m","dStoch_m","Stoch_s","dStoch_s","Stoch_ms","dStoch_ms",
                "WPR","dWPR");
     }
   else
     {
      Print("FileOpen action failed. Error",GetLastError());
      ExpertRemove();
     }
//---
   return(0);
  }
```

The [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) event handler identifies new bars and saves data in the file.

The price behavior will be determined by the last completed bar and the values of the indicators will be obtained from the bar preceding the last completed bar. Apart from the absolute indicator value, we need to save the difference between the absolute and the preceding value in order to see the direction of the change. The names of such variables in the example provided will have prefix "d".

For signal line indicators, it is necessary to save the difference between the main and signal line as well as its dynamics. In addition, save the time of the new bar and the relevant hour value. This may come in handy for filtering the data by time.

Thus, we will take into account 37 indicators to build a forecasting model to estimate the price movement.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//| Monitoring the market situation and saving values                |
//| of the indicators into the file at the beginning of every new bar|
//+------------------------------------------------------------------+
void OnTick()
  {
//--- declaring a static variable of datetime type
   static datetime Prev_time;

//--- it will be used to store prices, volumes and spread of each bar
   MqlRates mrate[];
   MqlTick tickdata;

   ArraySetAsSeries(mrate,true);

//--- obtaining the recent quotes
   if(!SymbolInfoTick(_Symbol,tickdata))
     {
      Alert("Quote update error - error: ",GetLastError(),"!!");
      return;
     }
///--- copying data of the last 4 bars
   if(CopyRates(_Symbol,_Period,0,4,mrate)<0)
     {
      Alert("Historical quote copy error - error: ",GetLastError(),"!!");
      return;
     }
//--- if both time values are equal, there is no new bar
   if(Prev_time==mrate[0].time) return;
//--- saving the time in the static variable
   Prev_time=mrate[0].time;

//--- filling the arrays with values of the indicators
   bool copy_result=true;
   copy_result=copy_result && FillArrayFromBuffer1(buf_AC,h_AC,4);
   copy_result=copy_result && FillArrayFromBuffer1(buf_BearsPower,h_BearsPower,4);
   copy_result=copy_result && FillArrayFromBuffer1(buf_BullsPower,h_BullsPower,4);
   copy_result=copy_result && FillArrayFromBuffer1(buf_AO,h_AO,4);
   copy_result=copy_result && FillArrayFromBuffer1(buf_CCI,h_CCI,4);
   copy_result=copy_result && FillArrayFromBuffer1(buf_DeMarker,h_DeMarker,4);
   copy_result=copy_result && FillArrayFromBuffer1(buf_FrAMA,h_FrAMA,4);
   copy_result=copy_result && FillArraysFromBuffers2(buf_MACD_m,buf_MACD_s,h_MACD,4);
   copy_result=copy_result && FillArrayFromBuffer1(buf_RSI,h_RSI,4);
   copy_result=copy_result && FillArraysFromBuffers2(buf_RVI_m,buf_RVI_s,h_RVI,4);
   copy_result=copy_result && FillArraysFromBuffers2(buf_Stoch_m,buf_Stoch_s,h_Stoch,4);
   copy_result=copy_result && FillArrayFromBuffer1(buf_WPR,h_WPR,4);

//--- checking the accuracy of copying the data
   if(!copy_result==true)
     {
      Print("Data copy error");
      return;
     }

//--- saving to the file the price movement within the last two bars
//--- and the preceding values of the indicators
   if(FileHandle!=INVALID_HANDLE)
     {
      MqlDateTime tm;
      TimeCurrent(tm);
      uint Result=0;
      Result=FileWrite(FileHandle,TimeToString(TimeCurrent()),tm.hour, // time of the bar
                       (mrate[1].close-mrate[2].close)/_Point,       // difference between the closing prices of the last two bars
                       buf_AC[2],buf_AC[2]-buf_AC[3],                // value of the indicator on the preceding bar and its dynamics
                       buf_BearsPower[2],buf_BearsPower[2]-buf_BearsPower[3],
                       buf_BullsPower[2],buf_BullsPower[2]-buf_BullsPower[3],
                       buf_AO[2],buf_AO[2]-buf_AO[3],
                       buf_CCI[2],buf_CCI[2]-buf_CCI[3],
                       buf_DeMarker[2],buf_DeMarker[2]-buf_DeMarker[3],
                       buf_FrAMA[2],buf_FrAMA[2]-buf_FrAMA[3],
                       buf_MACD_m[2],buf_MACD_m[2]-buf_MACD_m[3],
                       buf_MACD_s[2],buf_MACD_s[2]-buf_MACD_s[3],
                       buf_MACD_m[2]-buf_MACD_s[2],buf_MACD_m[2]-buf_MACD_s[2]-buf_MACD_m[3]+buf_MACD_s[3],
                       buf_RSI[2],buf_RSI[2]-buf_RSI[3],
                       buf_RVI_m[2],buf_RVI_m[2]-buf_RVI_m[3],
                       buf_RVI_s[2],buf_RVI_s[2]-buf_RVI_s[3],
                       buf_RVI_m[2]-buf_RVI_s[2],buf_RVI_m[2]-buf_RVI_s[2]-buf_RVI_m[3]+buf_RVI_s[3],
                       buf_Stoch_m[2],buf_Stoch_m[2]-buf_Stoch_m[3],
                       buf_Stoch_s[2],buf_Stoch_s[2]-buf_Stoch_s[3],
                       buf_Stoch_m[2]-buf_Stoch_s[2],buf_Stoch_m[2]-buf_Stoch_s[2]-buf_Stoch_m[3]+buf_Stoch_s[3],
                       buf_WPR[2],buf_WPR[2]-buf_WPR[3]);

      if(Result==0)
        {
         Print("FileWrite action error ",GetLastError());
         ExpertRemove();
        }
     }

  }
```

After starting the EA, the MasterData.CSV file will be created in terminal\_data\_directory/MQL5/Files. When starting the EA in the tester, it will be located in terminal\_data\_directory/tester/Agent-127.0.0.1-3000/MQL5/Files. The file as obtained can already be used in Statistica.

An example of such file can be found in MasterData.CSV. The data was collected for EURUSD H1 from August 1, 2011 to October 1, 2011.

In order to open the file in Statistica, do as follows.

- In Statistica, go to menu File > Open, select the file type: Data files and open your file.

- Leave Delimited in the Text File Import Type window and click OK.
- Enable the underlined items in the opened window.

- Bear in mind to put the decimal point in the Decimal separator character field regardless of whether it is already there or not.

[![Fig. 1. Importing the file into Statistica](https://c.mql5.com/2/3/Import-Stat_n__1.png)](https://c.mql5.com/2/3/Import-Stat_n.png "https://c.mql5.com/2/3/Import-Stat_n.png")

Fig. 1. Importing the file into Statistica

Click OK and the table containing our data is ready.

[![Fig. 2. Database in Statistica](https://c.mql5.com/2/3/MasterDataTable_n__1.png)](https://c.mql5.com/2/3/MasterDataTable_n.png "https://c.mql5.com/2/3/MasterDataTable_n.png")

Fig. 2. Database in Statistica

Now create the grouping variable on the basis of the Price variable.

We will single out four groups depending on the price behavior:

1. Over 200 points downwards;
2. Less than 200 points downwards;
3. Less than 200 points upwards;
4. Over 200 points upwards.

In order to add a new variable, right-click on the AC column header and select the Add Variable option.

![Fig. 3. Adding a new variable](https://c.mql5.com/2/3/AddVariable_n.png)

Fig. 3. Adding a new variable

Specify the name "Group" for the new variable in the opened window and add the formula for conversion of the Price variable to the number of groups.

The formula is as follows:

=iif(v3<=-200;1;0)+iif(v3<0 and v3>-200;2;0)+iif(v3>0 and v3<200;3;0)+iif(v3>=200;4;0)

![Fig. 4. Description of the variable](https://c.mql5.com/2/3/Figure4_AddVariable2.png)

Fig. 4. Description of the variable

The file is ready for the discriminant analysis. An example of this file can be found in MasterData.STA.

### 2.2. Selection of the Best Variables

Run the discriminant analysis (Statistics->Multivariate Exploratory Techniques->Discriminant Analysis).

![Fig. 5. Running the discriminant analysis](https://c.mql5.com/2/3/Figure_5_GetStarted.png)

[https://c.mql5.com/2/3/GetStarted__2.png](https://c.mql5.com/2/3/GetStarted__2.png "https://c.mql5.com/2/3/GetStarted__2.png")

Fig. 5. Running the discriminant analysis

Click Variables in the opened window.

Select the grouping variable in the first field and all the variables based on which the grouping will be done - in the second field.

In our case, the Group variable is specified in the first field and all the variables obtained from the indicators as well as the additional variable Hour (the hour of receiving the data) - in the second field.

![Fig. 6. Selection of variables](https://c.mql5.com/2/3/Figure_6_VariablesSelecting2.png)

Fig. 6. Selection of variables

Click the Select Cases button (Figure 8). A window will open for selection of cases (data rows) which will be used in the discriminant analysis. Enable items as shown in the screenshot below (Figure 7).

Only the first 700 cases will be used for the analysis. The remaining ones will afterwards be used for testing of the resulting prognostic model. The numbers of cases are set via the variable V0. By specifying the cases in this manner, we set a sample of the training data for DA.

Then click OK.

![Fig. 7. Defining the training sample](https://c.mql5.com/2/3/Figure7_CaseSelection2.png)

Fig. 7. Defining the training sample

Now let us select the groups for which our prognostic model will be built.

There is one issue that requires our attention. One of the weak points of DA is sensitivity to data outliers. Rare yet powerful events - in our case, price spikes- can distort the model. For example, following the unexpected news, the market responded with substantial movements lasting for a few hours. The values of technical indicators were in this case of little importance in the forecast yet they will be considered highly significant in DA as there was a marked price change. It is therefore advisable to check the data for outliers before running DA.

In order to exclude outliers from our example we will only analyze groups 2 and 3. Since there was a substantial price change in groups 1 and 4, there may be outliers in the indicator values.

So, click on Codes for grouping variable (Figure 8). And specify the numbers of groups for the analysis.

![Fig. 8. Selection of groups for the analysis ](https://c.mql5.com/2/3/Figure8_GroupSelecting3.png)

Fig. 8. Selection of groups for the analysis

Enable Advanced options. It will allow for the stepwise analysis that will be required at a later stage.

To run DA, click OK.

A message as below may pop up. This means that one of the selected variables is excessive and is substantially conditional on other variables, e.g. it is the sum of two other variables.

This is quite possible for the data flow obtained from the indicators. The presence of such variables affects the quality of the analysis. And they shall be removed. In order to do this, go back to the window for selection of variables for DA and identify the excessive variables by adding them one by one and running DA again and again.

![Fig. 9. Low tolerance value message](https://c.mql5.com/2/3/Figure9_Message.png)

Fig. 9. Low tolerance value message

Then a window for selection of the DA method will open (Figure 10). Select Forward Stepwise in the drop-down list. Since the values of the indicators have little prognostic importance, the use of the stepwise analysis is preferred. And the model of group discrimination will automatically be built stepwise.

Specifically, at each step all variables will be reviewed and evaluated to determine which one will contribute most to the discrimination between the groups. That variable will then be included in the model and the process will start over again. All the variables that best discriminate between the data sample will be selected in the specified manner step by step.

![Fig. 10. Method selection](https://c.mql5.com/2/3/Figure10_ModelDefinition.png)

Fig. 10. Method selection

Click OK and a window will open informing you that DA was successfully completed.

![Fig. 11. Window of DA results](https://c.mql5.com/2/3/Figure11_DA3.png)

Fig. 11. Window of DA results

Click Summary: Variables in the model to see the list of variables included in the model following the stepwise analysis. These variables best discriminate between our groups. Note that the variables producing the accuracy of discrimination of over 95% (p<0.05) are displayed in red. The accuracy of discrimination with regard to other variables is lower. The model shall only include the variables producing the accuracy of discrimination of at least 95%.

However according to the "golden rule" of statistics, only the variables producing the accuracy of over 95% shall be used. We will therefore exclude from the analysis all variables that are not displayed in red. These are dBulls, Bulls, FrAMA, Hour. To exclude these variables, go back to the window where the stepwise analysis was selected and specify them in the window which will open after clicking Variables.

Repeat the analysis. By clicking the Summary: Variables in the model, we will again see that yet three other variables now appear as insignificant. These are DeMarker, Stoch\_s, AO. We will also exclude them from the analysis.

As a result, we will have a model that includes the variables producing accurate discrimination between the groups (p<0.01).

![Fig. 12. Variables included in the model](https://c.mql5.com/2/3/RightVariables_n.png)

Fig. 12. Variables included in the model

Thus, only seven out of 37 variables were left in our example as being the most significant for the forecast.

This approach allows to select the key indicators on the basis of technical analysis for further development of custom trading systems, including the ones that utilize neural networks.

### 2.3. Analysis and Testing of the Resulting Model Using Test Data

Upon completion of DA, we obtained the prognostic model and the results of its application to training data.

To see the model and group discrimination results, open the Classification tab.

![Fig. 13. Classification tab](https://c.mql5.com/2/3/Figure13_DA4.png)

Fig. 13. Classification tab

Click Classification matrix to see the table containing the results of application of the model to training data.

The rows show the observed classifications. The columns contain the predicted classifications according to the model calculated. The cells that contain accurate predictions are marked in green and inaccurate predictions appear in red.

The first column displays the accuracy of prediction in %.

![Fig. 14. Training data classification](https://c.mql5.com/2/3/matrix1_n.png)

Fig. 14. Training data classification

The accuracy of prediction (Total) using training data turned out to be 60%.

Let us test the model using test data. To do this, click Select (Figure 13) and specify v0>700 following which the model will be checked within the range of data that was not used for building the model.

We will have the following:

![Fig. 15. Test data classification](https://c.mql5.com/2/3/matrix2_n.png)

Fig. 15. Test data classification

The overall accuracy of prediction using the test sample turned out to be roughly at the same level reaching 55%. This is a fairly good level for the FOREX market.

### 2.4. Developing a Trading System

The prognostic model in DA is based on the system of linear equations according to which values of the indicators are classified into one group or the other.

In order to see the descriptions of these functions, go to the Classification tab in the window of DA results (Figure 13) and click Classification functions. You will see a window with a table containing the coefficients of discriminant equations.

![Fig. 16. Discriminant equations](https://c.mql5.com/2/3/Functions_n.png)

Fig. 16. Discriminant equations

Let us develop a system of two equations on the basis of the table data:

Group2 = 157.17\*AC - 465.64\*Bears + 82.24\*dBears - 0.006\*dCCI + 761.06\*dFrAMA + 2418.79\*dMACDm + 0.01\*dStoch\_ms - 1.035

Group3 = 527.11\*AC - 641.97\*Bears + 271.21\*dBears - 0.002\*dCCI + 1483.47\*dFrAMA - 726.16\*dMACDm - 0.034\*dStoch\_ms - 1.353

In order to use this model, insert the indicator values into the equations and calculate the Group value.

The forecast will concern the group whose Group value is higher. According to our example, if the Group2 value is bigger than that of Group3, it is predicted that within the next hour the price chart will most probably be moving downwards. The forecast will turn out to be quite the opposite in the case where the Group3 value is bigger than that of Group2.

It should be noted that the values of the indicators and period of analysis in our example were selected rather randomly. But even this amount of data was sufficient to demonstrate the potential and power of DA.

### Conclusion

The discriminant analysis is a useful tool as applied to the FOREX market. It can be used to search and check the optimal set of variables allowing to classify the observed indicator values into different forecasts. It can also be utilized for building prognostic models.

The models built as a result of the discriminant analysis can easily be integrated into EAs which does not require a considerable developing experience. The discriminant analysis in itself is also relatively easy to use. The above step-by-step tutorial would suffice to analyze your own data.

More on the discriminant analysis can be found in the relevant section of the [electronic textbook](https://www.mql5.com/go?link=http://www.statsoft.com/Textbook "http://www.statsoft.com/textbook/").

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/335](https://www.mql5.com/ru/articles/335)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/335.zip "Download all attachments in the single ZIP archive")

[masterdata.zip](https://www.mql5.com/en/articles/download/335/masterdata.zip "Download masterdata.zip")(662.43 KB)

[da\_demo.mq5](https://www.mql5.com/en/articles/download/335/da_demo.mq5 "Download da_demo.mq5")(12.36 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Multiple Regression Analysis. Strategy Generator and Tester in One](https://www.mql5.com/en/articles/349)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6167)**
(21)


![ArtemGaleev](https://c.mql5.com/avatar/2011/11/4EB2FE09-7030.jpg)

**[ArtemGaleev](https://www.mql5.com/en/users/artemgaleev)**
\|
16 Nov 2011 at 19:33

**Virty:**

Agreed. Discriminant analysis, neural networks, self-learning programs, Kohonen maps, etc. are just analysis tools, but we need a forecast. A market model is needed for forecasting. Until there is no market model, it is inefficient to use analysis tools.

Thus, in the discussed article, the market model was sewed into several indicators taken for analysis. This model describes the market poorly, which manifested itself in the high probability of coefficients equality to zero.

There is no sense to propose and discuss instruments without a market model.

The probability of zero equality mentioned in the comments has no relation to this model :-)

In general, this is a holy war argument - whether technical analysis works or not. Someone successfully trades using only technical analysis, others prefer fundamental analysis, and some prefer both.

According to the systems theory: it is not necessary to know how a system works in order to predict its behaviour. It is doubtful that a market model for forex is even possible. The appearance of any model will immediately be taken into account in the price according to technical analysis :-)

![Гребенев Вячеслав](https://c.mql5.com/avatar/avatar_na2.png)

**[Гребенев Вячеслав](https://www.mql5.com/en/users/virty)**
\|
17 Nov 2011 at 14:02

**ArtemGaleev:**

The probability of zero mentioned in the comments has nothing to do with this model :-)

In general, this is a holy war argument - whether technical analysis works or not. Someone successfully trades using only technical analysis, others prefer fundamental analysis, and some prefer both.

According to the systems theory: it is not necessary to know how a system works in order to predict its behaviour. It is doubtful that a market model for forex is even possible. The appearance of any model will immediately be factored into the price according to technical analysis :-)

There are many forex market models. Here are some of them: 1. The price wanders randomly. 2. Price is a smooth function with added noise. 3. There are sometimes trends in price movement. 4. A lot of models from economics and fundamental analysis.

To predict the behaviour of a system, you don't even need to know its structure. It is possible, for example, to transfer the history of the system to the future. But the transfer rule will be the model of the system.

![Eddy Taylor](https://c.mql5.com/avatar/avatar_na2.png)

**[Eddy Taylor](https://www.mql5.com/en/users/eddy.taylor)**
\|
14 Feb 2012 at 23:22

Hi,

Thank you very much for producing this article! I am ever so grateful. I was wondering where to find a software like this! So what if I like a certain setup already but I want to improve my win rate? How do I use the DA Statistica for this?

I have used some [neural network](https://www.mql5.com/en/articles/497 "Article: Neural networks - from theory to practice ") and Genetic Programming software. They are able to build strategies towards a user's goals - win rate, Net Profit, Frequency of trades, E(r), Profit Factor, etc. Is it possible to do that with this software? Can you show how if you can?

![ArtemGaleev](https://c.mql5.com/avatar/2011/11/4EB2FE09-7030.jpg)

**[ArtemGaleev](https://www.mql5.com/en/users/artemgaleev)**
\|
15 Feb 2012 at 04:25

StatSoft Statistica was used for the DA analysis.

I believe, other goals can be reached too. However you should think out how they can be included in your DA model. Just as an idea, if you trade on H1 time frame, you may try to forecast your win rate by H4 or D1 indicator data.

The DA for H4 timeframe is carried out in the same way as for H1.

![Gerard William G J B M Dinh Sy](https://c.mql5.com/avatar/2026/1/69609d33-0703.png)

**[Gerard William G J B M Dinh Sy](https://www.mql5.com/en/users/william210)**
\|
2 Jun 2025 at 09:31

It's an interesting idea. I like the option of exporting to a [spreadsheet](https://www.mql5.com/en/articles/8699 "Article: Using Spreadsheets to Build Trading Strategies ") for validation.


![The All or Nothing Forex Strategy](https://c.mql5.com/2/0/allVSzero.png)[The All or Nothing Forex Strategy](https://www.mql5.com/en/articles/336)

The purpose of this article is to create the most simple trading strategy that implements the "All or Nothing" gaming principle. We don't want to create a profitable Expert Advisor - the goal is to increase the initial deposit several times with the highest possible probability. Is it possible to hit the jackpot on ForEx or lose everything without knowing anything about technical analysis and without using any indicators?

![Speed Up Calculations with the MQL5 Cloud Network](https://c.mql5.com/2/0/speed_network.png)[Speed Up Calculations with the MQL5 Cloud Network](https://www.mql5.com/en/articles/341)

How many cores do you have on your home computer? How many computers can you use to optimize a trading strategy? We show here how to use the MQL5 Cloud Network to accelerate calculations by receiving the computing power across the globe with the click of a mouse. The phrase "Time is money" becomes even more topical with each passing year, and we cannot afford to wait for important computations for tens of hours or even days.

![Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://c.mql5.com/2/0/MQL5_protection_methods.png)[Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)

Most developers need to have their code secured. This article will present a few different ways to protect MQL5 software - it presents methods to provide licensing capabilities to MQL5 Scripts, Expert Advisors and Indicators. It covers password protection, key generators, account license, time-limit evaluation and remote protection using MQL5-RPC calls.

![Create Your Own Graphical Panels in MQL5](https://c.mql5.com/2/0/graph_pannels_MQL5.png)[Create Your Own Graphical Panels in MQL5](https://www.mql5.com/en/articles/345)

The MQL5 program usability is determined by both its rich functionality and an elaborate graphical user interface. Visual perception is sometimes more important than fast and stable operation. Here is a step-by-step guide to creating display panels on the basis of the Standard Library classes on your own.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/335&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071516040803854718)

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