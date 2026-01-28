---
title: Features of Experts Advisors
url: https://www.mql5.com/en/articles/1494
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:13:53.172940
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/1494&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083422854084565872)

MetaTrader 4 / Examples


Creation of expert advisors in the MetaTrader trading system has a number of features.

- Before opening a position, you should check if there is money available on the account.
If there is not enough money on the account, the operation of opening a position
will not be successful. The "FreeMargin" value must not be less than
1000 only during tests because the price of one lot is 1000 during tests.




```
if(AccountFreeMargin() < 1000) return(0);    // not enough money
```

- You can access history data by using the predefined arrays Time, Open, Low, High,
Close, Volume. Due to historical reasons, index in these arrays increases from
the end to the beginning. It means that the latest data have index 0. Index 1 indicates
data shifted one period backwards, index 2 means data shifted two periods backwards,
3 is three periods backwards, etc.




```
// if Close on the previous bar is less than
// Close on the bar before previous
if(Close[1] < Close[2]) return(0);
```

- It is also possible to access historical data using other time intervals and even
using other currency pairs. To get such data, it is necessary to define a one-dimensional
array first and perform a copy operation using the "ArrayCopySeries"
function. Mind that, while calling the function, it is possible to pass a less
number of parameters and do not specify default parameters.




```
double eur_close_m1[];
int number_copied = ArrayCopySeries(eur_close_m1, MODE_CLOSE, "EURUSD", PERIOD_M1);
```

- In the process of writing an expert advisor, as well as any other software, it is
sometimes necessary to get some additional debugging information. The [MQL4](https://docs.mql4.com/) language provides several methods for getting such information.

  - The "Alert" function displays a dialog box with some data defined by the
    user.




    ```
    Alert("FreeMargin grows to ", AccountFreeMargin(), "!");
    ```

  - The "Comment" function displays data defined by the user in the upper-left
    corner of the chart. The character sequence "\\n" is used to start a new
    line.




    ```
    Comment("FreeMargin is ", AccountFreeMargin(), ".");
    ```

  - The "Print" function saves data defined by the user to the system log.





    ```
    Print("FreeMargin is ", AccountFreeMargin(), ".");
    ```
- To get the information about errors in the program, the "GetLastError"
function is very useful. For example, an operation with an order always returns
the ticket number. If the ticket number equals 0 (some error has occurred in the
process of performing the operation), it is necessary to call the "GetLastError"
function to get additional information about the error:




```
int iTickNum = 0;
int iLastError = 0;
...
iTickNum = OrderSet (OP_BUY, g_Lots, Ask, 3, 0, Ask + g_TakeProfit * g_Points, Red);
if (iTickNum <= 0)
     {
       iLastError = GetLastError();
       if (iLastError != ERROR_SUCCESS) Alert("Some Message");
     }
```


You should remember that calling the "GetLastError" function displays
the code of the last error and resets its value. That is why calling this function
once again in a row will always return value 0.

- How to define the beginning of a new bar? (It can be necessary to find out that
the previous bar has just been finished.) There are several methods to do it.



The first method is based on checking the number of bars:




```
int prevbars = 0;
...
if(prevbars == Bars) return(0);
prevbars = Bars;
...
```


This method can fail to work while loading the history. That is, the number of bars
is changed while the "previous" one has not been finished yet. In this
case you can make checking more complicated by introducing a check for difference
between the values equal to one.




The next method is based on the fact that the "Volume" value is generated
depending on the number of ticks that have come for each bar and the first tick
means that the "Volume" value of a new bar equals 1:




```
if( Volume > 1) return(0);
...
```


This method can fail to work when there are a lot of incoming price ticks. The matter
is that incoming price tricks are processed in a separate thread. And if this thread
is busy when the next tick comes, this new incoming tick is not processed to avoid
overloading the processor! In this case you can also make checking more complicated
by saving the previous "Volume" value.




The third method is based on the time a bar is opened:




```
datetime prevtime=0;
...
if(prevtime == Time[0]) return(0);
prevtime = Time[0];
...
```


It is the most reliable method. It works in all cases.

- An example of working with a file of the "CSV" type:




```
int h1;
h1 = FileOpen("my_data.csv", MODE_CSV | MODE_WRITE, ";");
if(h1 < 0)
    {
     Print("Unable to open file my_data.csv");
     return(false);
    }
FileWrite(h1, High[1], Low[1], Close[1], Volume[1]);
FileClose(h1);
```


Some explanations to the code. The file of the "CSV" format is opened
first. In case there occurs an error while opening the file, the program is exited.
In case the file is successfully opened, its content gets cleared, data are saved
to the file and the file is closed. If you need to keep the content of the file
being opened, you should use the MODE\_READ opening mode:




```
int h1;
h1 = FileOpen("my_data.csv", MODE_CSV | MODE_WRITE | MODE_READ, ";");
if(h1 < 0)
    {
     Print("Unable to open file my_data.csv");
     return(false);
    }
FileSeek(h1, 0, SEEK_END);
FileWrite(h1, High[1], Low[1], Close[1], Volume[1]);
FileClose(h1);
```


In this example, data are added to the end of the file. To do it, we used the "FileSeek"
function right after it was opened.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1494](https://www.mql5.com/ru/articles/1494)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39335)**
(3)


![Sumit Dutta](https://c.mql5.com/avatar/2018/4/5AE0D924-07B8.jpg)

**[Sumit Dutta](https://www.mql5.com/en/users/loveyourockg)**
\|
21 May 2018 at 19:45

```
//+------------------------------------------------------------------+
//|                                       GoldminerAbsoluteEntry.mq4 |
//|                                             Copyright © 2009, Me |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2009, Me"

#property indicator_chart_window
#property indicator_buffers 2
#property indicator_color1 Magenta
#property indicator_color2 Yellow

extern string Remark1 = "-- Goldminer1 Settings --";
extern int RISK = 4;
extern int VerifyShift = 1;

//---- buffers
double ExtMapBuffer1[];
double ExtMapBuffer2[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int init()
  {
//---- indicators
   SetIndexStyle(0,DRAW_ARROW);
   SetIndexArrow(0,234);
   SetIndexBuffer(0,ExtMapBuffer1);
   SetIndexEmptyValue(0,0.0);
   SetIndexStyle(1,DRAW_ARROW);
   SetIndexArrow(1,233);
   SetIndexBuffer(1,ExtMapBuffer2);
   SetIndexEmptyValue(1,0.0);
//----
   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
int deinit()
  {

   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int start()
  {

   //+------------------------------------------------------------------+
   //| Goldminer Begin                                                  |
   //+------------------------------------------------------------------+

   double GN1,GN2;
   string GoldMiner;

   for(int a=0;a<Bars;a++){//For Loop
   GN1=iCustom(0,0,"goldminer1",RISK,Bars,0,a+VerifyShift);
   GN2=iCustom(0,0,"goldminer1",RISK,Bars,1,a+VerifyShift);

   if(GN2>GN1){GoldMiner="Up";
      ExtMapBuffer2[a]=Low[a];}
   else if(GN1>GN2){GoldMiner="Down";
      ExtMapBuffer1[a]=High[a];}
   else {GoldMiner="None";}

   }//End Of For Loop

   //+---------------------------------------------------------------+
   return(0);
  }
//+------------------------------------------------------------------+
```

![Sumit Dutta](https://c.mql5.com/avatar/2018/4/5AE0D924-07B8.jpg)

**[Sumit Dutta](https://www.mql5.com/en/users/loveyourockg)**
\|
21 May 2018 at 19:47

**Sumit Dutta:**

any one help i want my indicator alart up when back singel candel is green or alart down when back sigel candel is reed filter signal


![Paweł Wilski](https://c.mql5.com/avatar/2024/8/66acbf09-d8f5.jpg)

**[Paweł Wilski](https://www.mql5.com/en/users/mqlzone)**
\|
2 Aug 2024 at 11:37

//+------------------------------------------------------------------+

//\|                                       GoldminerAbsoluteEntry.mq4 \|

//\|                                             Copyright © 2009, Me \|

//+------------------------------------------------------------------+

#property copyright "Copyright © 2009, Me"

[#property indicator\_chart\_window](https://www.mql5.com/en/docs/basis/preprosessor/compilation "MQL5 Documentation: Program Properties (#property)")

#property indicator\_buffers 2

#property indicator\_color1 Magenta

#property indicator\_color2 Yellow

extern string Remark1 = "-- Goldminer1 Settings --";

extern int RISK = 4;

extern int VerifyShift = 1;

//\-\-\-\- buffers

double ExtMapBuffer1\[\];

double ExtMapBuffer2\[\];

//+------------------------------------------------------------------+

//\| Custom indicator initialization function                         \|

//+------------------------------------------------------------------+

int init()

{

    //\-\-\-\- indicators

    SetIndexStyle(0, DRAW\_ARROW);

    SetIndexArrow(0, 234);

    SetIndexBuffer(0, ExtMapBuffer1);

    SetIndexEmptyValue(0, 0.0);

    SetIndexStyle(1, DRAW\_ARROW);

    SetIndexArrow(1, 233);

    SetIndexBuffer(1, ExtMapBuffer2);

    SetIndexEmptyValue(1, 0.0);

    //----

    return(0);

}

//+------------------------------------------------------------------+

//\| Custom indicator deinitialization function                       \|

//+------------------------------------------------------------------+

int deinit()

{

    return(0);

}

//+------------------------------------------------------------------+

//\| Custom indicator iteration function                              \|

//+------------------------------------------------------------------+

int start()

{

    //+------------------------------------------------------------------+

    //\| Goldminer Begin                                                  \|

    //+------------------------------------------------------------------+

    double GN1, GN2;

    string GoldMiner;

    // Ensure there are enough bars

    if (Bars <= VerifyShift) return(0);

    int counted\_bars = IndicatorCounted();

    int limit = Bars - counted\_bars;

    // Clear buffers

    ArraySetAsSeries(ExtMapBuffer1, true);

    ArraySetAsSeries(ExtMapBuffer2, true);

    for (int a = limit - 1; a >= 0; a--)

    {

        GN1 = iCustom(NULL, 0, "goldminer1", RISK, Bars, 0, a + VerifyShift);

        GN2 = iCustom(NULL, 0, "goldminer1", RISK, Bars, 1, a + VerifyShift);

        if (GN2 > GN1)

        {

            GoldMiner = "Up";

            ExtMapBuffer2\[a\] = Low\[a\];

            // Check if previous candle is green (bullish)

            if (Close\[1\] > Open\[1\])

            {

                Alert("GoldMiner Up signal with previous candle green.");

            }

        }

        else if (GN1 > GN2)

        {

            GoldMiner = "Down";

            ExtMapBuffer1\[a\] = High\[a\];

            // Check if previous candle is red (bearish)

            if (Close\[1\] < Open\[1\])

            {

                Alert("GoldMiner Down signal with previous candle red.");

            }

        }

        else

        {

            GoldMiner = "None";

        }

    }

    //+---------------------------------------------------------------+

    return(0);

}

//+------------------------------------------------------------------+


![Features of Custom Indicators Creation](https://c.mql5.com/2/16/77_1.gif)[Features of Custom Indicators Creation](https://www.mql5.com/en/articles/1497)

Creation of Custom Indicators in the MetaTrader trading system has a number of features.

![Statistical Arbitrage Through Cointegrated Stocks (Part 10): Detecting Structural Breaks](https://c.mql5.com/2/191/20946-statistical-arbitrage-through-logo__1.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 10): Detecting Structural Breaks](https://www.mql5.com/en/articles/20946)

This article presents the Chow test for detecting structural breaks in pair relationships and the application of the Cumulative Sum of Squares - CUSUM - for structural breaks monitoring and early detection. The article uses the Nvidia/Intel partnership announcement and the US Gov foreign trade tariff announcement as examples of slope inversion and intercept shift, respectively. Python scripts for all the tests are provided.

![Strategy Tester: Modes of Modeling during Testing](https://c.mql5.com/2/17/78_7.gif)[Strategy Tester: Modes of Modeling during Testing](https://www.mql5.com/en/articles/1511)

Many programs of technical analysis allow to test trading strategies on history data. In the most cases, the testing is conducted on already completed data without any attempts to model the trends within a price bar. It was made quickly, but not precisely

![Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://c.mql5.com/2/120/Hybrid_Sequence_Graph_Models___LOGO.png)[Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)

Hybrid graph sequence models (GSM++) combine the advantages of different architectures to provide high-fidelity data analysis and optimized computational costs. These models adapt effectively to dynamic market data, improving the presentation and processing of financial information.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=orqdlmfruhgiwbzyhkrnndmmjhczlvfb&ssn=1769253232621141172&ssn_dr=0&ssn_sr=0&fv_date=1769253232&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1494&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Features%20of%20Experts%20Advisors%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925323221071354&fz_uniq=5083422854084565872&sv=2552)

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