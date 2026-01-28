---
title: Expert Advisor Sample
url: https://www.mql5.com/en/articles/1510
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:30:15.277050
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/1510&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071881233283100725)

MetaTrader 4 / Examples


The principles of MQL4-programs development are shown on sample of creating a simple Expert Advisor system based on the standard
MACD indicator. In this Expert Advisor, we will also see examples of implementing such features as setting take profit levels with the support of trailing stop as well as the most
means ensuring safe work. In our example, trading is done through opening and managing a single position.

Trading principles:

- **Long (BUY) entry** – MACD indicator is below zero, goes upwards and is crossed by the Signal Line going downwards.

![](https://c.mql5.com/2/17/macd_down.gif)

- **Short (SELL) entry** – MACD indicator is above zero, goes downwards and is crossed by the Signal Line going upwards.

![](https://c.mql5.com/2/17/macd_up.gif)

- **Long exit** – by execution of the take profit limit, by execution of the trailing stop or when MACD crosses its Signal Line (MACD is above zero, goes
downwards and is crossed by the Signal Line going upwards).

- **Short exit** – by execution of the take profit limit, by execution of the trailing stop or when MACD crosses its Signal Line (MACD is below zero, goes
upwards and is crossed by the Signal Line going downwards).


**Important notice:** to exclude insignificant changes of the MACD indicator (small 'hillocks' on the chart) from our analysis, we introduce an additional measure
of controlling the size of the plotted 'hillocks' as follows: the indicator size should be at least 5 units of the minimum price (5\*Point, which for USD/CHF = 0.0005 and for USD/JPY = 0.05).

![](https://c.mql5.com/2/17/sample.gif)

**Step 1 – Writing the Expert Advisor description**

|     |     |
| --- | --- |
| ![](https://c.mql5.com/2/17/exp_add2.gif) | Point the mouse cursor at the Expert Advisors section of the Navigator window, press the right button of the mouse, and select "Create a new Expert" <br>command in the appearing menu. The Initializing Wizard of the Expert Advisor will ask you for entering certain data. In the appearing window, write the name (Name) of the Expert Advisor - MACD Sample, the author (Author) - indicate your name, the link (Link) - a link to your website, in the notes (Notes) - Test example of an MACD-based Expert Advisor. |

**Step 2 – Creating the primary structure of the program**

Source code of the test Expert Advisor will only occupy several pages, but even such volume is often difficult to grasp, especially regarding that we are not professional programmers
\- otherwise, we would not need this description at all, would we? :)

To get some idea of the structure of a standard Expert Advisor, let us take a look at the description given below:

1. Initializing variables

2. Initial data checks

   - check the chart, number of bars on the chart

   - check the values of external variables: Lots, S/L, T/P, T/S
3. Setting the internal variables for quick data access

4. Checking the trading terminal – is it void? If yes, then:

   - checks: availability of funds on the account etc...

   - is it possible to take a long position (BUY)?

     - open a long position and exit
5. is it possible to take a short position (SELL)?

   - open a short position and exit

exiting the Expert Advisor...

- Control of the positions previously opened in the cycle

  - if it is a long position

    - should it be closed?

    - should the trailing stop be reset?
  - if it is a short position

    - should it be closed?

    - should the trailing stop be reset?

It turns out to be quite simple, only 4 main blocks.

Now let us try to generate pieces of code step by step for each section of the structural scheme:

1. **Initializing variables**


     All variables to be used in the expert program must be defined according to the syntax of [MetaQuotes \\
     Language 4](https://docs.mql4.com/) first. That is why we insert the block for initializing variables at the beginning of the program


     ```
     extern double TakeProfit = 50;
     extern double Lots = 0.1;
     extern double TrailingStop = 30;
     extern double MACDOpenLevel=3;
     extern double MACDCloseLevel=2;
     extern double MATrendPeriod=26;
     ```


     MetaQuotes Language 4 is supplemented by "external variables" term. External variables can be set from the outside without modifying
     the source code of the expert program. It provides additional flexibility. In our program, the MATrendPeriod variable is defined as extern variable. We insert the definition of
     this variable at the beginning of the program.


     ```
     extern double MATrendPeriod=26;
     ```

2. **Initial data checks**


     This part of code is usually used in any expert with minor modifications because it is a virtually standard check block:



     ```
     // initial data checks
     // it is important to make sure that the expert works with a normal
     // chart and the user did not make any mistakes setting external
     // variables (Lots, StopLoss, TakeProfit,
     // TrailingStop) in our case, we check TakeProfit
     // on a chart of less than 100 bars
        if(Bars<100)
          {
           Print("bars less than 100");
           return(0);
          }
        if(TakeProfit<10)
          {
           Print("TakeProfit less than 10");
           return(0);  // check TakeProfit
          }
     ```

3. **Setting internal variables for quick access to data**


     In the source code it is very often necessary to access the indicator values or handle the calculated values. To simplify the coding and speed up the access, data are put into internal variables.


     ```
     int start()
       {
        double MacdCurrent, MacdPrevious, SignalCurrent;
        double SignalPrevious, MaCurrent, MaPrevious;
        int cnt, ticket, total;

     // to simplify the coding and speed up access
     // data are put into internal variables
        MacdCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,0);
        MacdPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,1);
        SignalCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,0);
        SignalPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,1);
        MaCurrent=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,0);
        MaPrevious=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,1);
     ```




     Now, instead of the monstrous notation of **iMACD(NULL,0,12,26,9,PRICE\_CLOSE,MODE\_MAIN,0)**, you can use **MacdCurrent** in the source code.

4. **Checking the trading terminal – is it empty? If it is, then:**


     In our Expert Advisor, we use only those positions which are opened with market orders
     and do not handle the pending orders. However, to be on the safe side, let us introduce a check of the trading terminal for previously placed orders:


     ```
      total=OrdersTotal();
        if(total<1)
          {
     ```


     - **checks: availability of funds on the account etc...**


       Before analyzing the market situation it is advisable to check the status of your account to
       make sure that there are free funds on it for opening a position.


       ```
             if(AccountFreeMargin()<(1000*Lots))
               {
                Print("We have no money. Free Margin = ", AccountFreeMargin());
                return(0);
               }
       ```

     - **is it possible to take a long position (BUY)?**


       Condition of entry into the long position: MACD is below zero, goes upwards and is crossed by the Signal Line going downwards. This is how we describe it
       in MQL4 (note that we operate on the indicator values which were previously saved in the variables):


       ```
             // check for long position (BUY) possibility
             if(MacdCurrent<0 && MacdCurrent>SignalCurrent &&
                MacdPrevious<SignalPrevious &&
                MathAbs(MacdCurrent)>(MACDOpenLevel*Point) &&
                MaCurrent>MaPrevious)
               {
                ticket=OrderSend(Symbol(),OP_BUY,Lots,Ask,3,0,Ask+TakeProfit*Point,
                                 "macd sample",16384,0,Green);
                if(ticket>0)
                  {
                   if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES))
                      Print("BUY order opened : ",OrderOpenPrice());
                  }
                else Print("Error opening BUY order : ",GetLastError());
                return(0);
               }
       ```


       Additional control over the size of the 'hillocks' being drawn was already mentioned above. MACDOpenLevel variable is a user-defined variable which may
       be changed without interfering with the program text, to ensure greater flexibility. In the beginning of the program we insert a description of this variable (as well as
       the description of the variable used below).

     - **is it possible to take a short position (SELL)?**


       Condition of entry of a short position: MACD is above zero, goes downwards and is crossed by the Signal Line going
       upwards. The notation is as follows:


       ```
                 // check for short position (SELL) possibility
                 if(MacdCurrent>0 && MacdCurrentSignalPrevious &&
                    MacdCurrent>(MACDOpenLevel*Point) &&
                    MaCurrent<MaPrevious)
                   {
                    ticket=OrderSend(Symbol(),OP_SELL,Lots,Bid,3,0,Bid-TakeProfit*Point,
                                     "macd sample",16384,0,Red);
                    if(ticket>0)
                      {
                       if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES))
                          Print("SELL order opened : ",OrderOpenPrice());
                      }
                    else Print("Error opening SELL order : ",GetLastError());
                    return(0);
                   }

         return(0);
        }
       ```
5. **Control of the positions previously opened in the cycle**


     ```
     // it is important to enter the market correctly,
     // but it is more important to exit it correctly...
     for(cnt=0;cnt
       {
        OrderSelect(cnt, SELECT_BY_POS, MODE_TRADES);
        if(OrderType()<=OP_SELL &&   // check for opened position
           OrderSymbol()==Symbol())  // check for symbol
          {
     ```


     "cnt" – " is a cycle variable that must be defined at the beginning of the program as follows:


     ```
      int cnt = 0;
     ```




     - if it is a long position


       ```
       if(OrderType()==OP_BUY)   // long position is opened
         {
       ```



       - **should it be closed?**


         Condition for exiting a long position: MACD is crossed by its Signal Line, MACD being above zero, going downwards and being
         crossed by the Signal Line going upwards.


         ```
         if(MacdCurrent>0 && MacdCurrent<SignalPrevious &&
            MacPrevious>SignalPrevious &&
            MacdCurrent>(MACDCloseLevel*Point))
           {
            OrderClose(OrderTicket(),OrderLots(),Bid,3,Violet); // close position
            return(0); // exit
           }
         ```

       - **should the trailing stop be reset?**


         We set the trailing stop only in case the position already has a profit exceeding the trailing stop level in points, and in
         case the new level of the stop is better than the previous.


         ```
         // check for trailing stop
         if(TrailingStop>0)
           {
            if(Bid-OrderOpenPrice()>Point*TrailingStop)
              {
               if(OrderStopLoss()
                 {
                  OrderModify(OrderTicket(),OrderOpenPrice(),Bid-Point*TrailingStop,
                              OrderTakeProfit(),0,Green);
                  return(0);
                 }
              }
           }
         ```


We close the brace of the operator.

```
   }
```
6. if it is a short position


     ```
     else // go to short position
       {
     ```



     - **should it be closed?**


       Condition for exiting a short position: MACD is crossed by its Signal Line, MACD being below zero, going upwards and being crossed
       by the Signal Line going downwards.


       ```
       if(MacdCurrent<0 && MacdCurrent>SignalCurrent &&
          MacdPrevious<SignalPrevious &&
          MathAbs(MacdCurrent)>(MACDCloseLevel*Point))
         {
          OrderClose(OrderTicket(),OrderLots(),Ask,3,Violet); // close position
          return(0); // exit
         }
       ```

     - **should the trailing stop be reset?**


       We set the trailing stop only in case the position already has a profit exceeding the trailing stop level in points, and in case the new level of the stop is better than the previous.


       ```
       // check for trailing stop
       if(TrailingStop>0)
         {
          if((OrderOpenPrice()-Ask)>(Point*TrailingStop))
            {
             if((OrderStopLoss()>(Ask+Point*TrailingStop)) || (OrderStopLoss()==0))
               {
                OrderModify(OrderTicket(),OrderOpenPrice(),Ask+Point*TrailingStop,
                            OrderTakeProfit(),0,Red);
                return(0);
               }
            }
         }
       ```


Closing all the curly bracket which remain open.

```
         }
      }
   }
 return(0);
}
```

So, following this step-by-step procedure, we have written our Expert Advisor...

**Step 3 – Assembling the resulting code of the programme**

Let's open the Expert Advisor settings (using a button or a line in the "Properties..." menu). We are offered a window in which we have to define the external
settings of the working parameters:

![](https://c.mql5.com/2/17/properties.gif)

Let's assemble all the code from the previous section:

```
//+------------------------------------------------------------------+
//|                                                  MACD Sample.mq4 |
//|                      Copyright © 2005, MetaQuotes Software Corp. |
//|                                      https://www.metaquotes.net/ |
//+------------------------------------------------------------------+
extern double TakeProfit = 50;
extern double Lots = 0.1;
extern double TrailingStop = 30;
extern double MACDOpenLevel=3;
extern double MACDCloseLevel=2;
extern double MATrendPeriod=26;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int start()
  {
   double MacdCurrent, MacdPrevious, SignalCurrent;
   double SignalPrevious, MaCurrent, MaPrevious;
   int cnt, ticket, total;
// initial data checks
// it is important to make sure that the expert works with a normal
// chart and the user did not make any mistakes setting external
// variables (Lots, StopLoss, TakeProfit,
// TrailingStop) in our case, we check TakeProfit
// on a chart of less than 100 bars
   if(Bars<100)
     {
      Print("bars less than 100");
      return(0);
     }
   if(TakeProfit<10)
     {
      Print("TakeProfit less than 10");
      return(0);  // check TakeProfit
     }
// to simplify the coding and speed up access
// data are put into internal variables
   MacdCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,0);
   MacdPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_MAIN,1);
   SignalCurrent=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,0);
   SignalPrevious=iMACD(NULL,0,12,26,9,PRICE_CLOSE,MODE_SIGNAL,1);
   MaCurrent=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,0);
   MaPrevious=iMA(NULL,0,MATrendPeriod,0,MODE_EMA,PRICE_CLOSE,1);
   total=OrdersTotal();
   if(total<1)
     {
      // no opened orders identified
      if(AccountFreeMargin()<(1000*Lots))
        {
         Print("We have no money. Free Margin = ", AccountFreeMargin());
         return(0);
        }
      // check for long position (BUY) possibility
      if(MacdCurrent<0 && MacdCurrent>SignalCurrent && MacdPrevious<SignalPrevious &&
         MathAbs(MacdCurrent)>(MACDOpenLevel*Point) && MaCurrent>MaPrevious)
        {
         ticket=OrderSend(Symbol(),OP_BUY,Lots,Ask,3,0,Ask+TakeProfit*Point,"macd sample",16384,0,Green);
         if(ticket>0)
           {
            if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES)) Print("BUY order opened : ",OrderOpenPrice());
           }
         else Print("Error opening BUY order : ",GetLastError());
         return(0);
        }
      // check for short position (SELL) possibility
      if(MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious &&
         MacdCurrent>(MACDOpenLevel*Point) && MaCurrent<MaPrevious)
        {
         ticket=OrderSend(Symbol(),OP_SELL,Lots,Bid,3,0,Bid-TakeProfit*Point,"macd sample",16384,0,Red);
         if(ticket>0)
           {
            if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES)) Print("SELL order opened : ",OrderOpenPrice());
           }
         else Print("Error opening SELL order : ",GetLastError());
         return(0);
        }
      return(0);
     }
   // it is important to enter the market correctly,
   // but it is more important to exit it correctly...
   for(cnt=0;cnt<total;cnt++)
     {
      OrderSelect(cnt, SELECT_BY_POS, MODE_TRADES);
      if(OrderType()<=OP_SELL &&   // check for opened position
         OrderSymbol()==Symbol())  // check for symbol
        {
         if(OrderType()==OP_BUY)   // long position is opened
           {
            // should it be closed?
            if(MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious &&
               MacdCurrent>(MACDCloseLevel*Point))
                {
                 OrderClose(OrderTicket(),OrderLots(),Bid,3,Violet); // close position
                 return(0); // exit
                }
            // check for trailing stop
            if(TrailingStop>0)
              {
               if(Bid-OrderOpenPrice()>Point*TrailingStop)
                 {
                  if(OrderStopLoss()<Bid-Point*TrailingStop)
                    {
                     OrderModify(OrderTicket(),OrderOpenPrice(),Bid-Point*TrailingStop,OrderTakeProfit(),0,Green);
                     return(0);
                    }
                 }
              }
           }
         else // go to short position
           {
            // should it be closed?
            if(MacdCurrent<0 && MacdCurrent>SignalCurrent &&
               MacdPrevious<SignalPrevious && MathAbs(MacdCurrent)>(MACDCloseLevel*Point))
              {
               OrderClose(OrderTicket(),OrderLots(),Ask,3,Violet); // close position
               return(0); // exit
              }
            // check for trailing stop
            if(TrailingStop>0)
              {
               if((OrderOpenPrice()-Ask)>(Point*TrailingStop))
                 {
                  if((OrderStopLoss()>(Ask+Point*TrailingStop)) || (OrderStopLoss()==0))
                    {
                     OrderModify(OrderTicket(),OrderOpenPrice(),Ask+Point*TrailingStop,OrderTakeProfit(),0,Red);
                     return(0);
                    }
                 }
              }
           }
        }
     }
   return(0);
  }
// the end.
```

For the final configuration of our expert advisor, just specify the values of external variables "Lots = 1", "Stop Loss (S/L) = 0" (not used),
"Take Profit (T/P) = 120" (appropriate for one-hour intervals), "Trailing Stop (T/S) = 30". Of course, you can set your own values. Press "Compile" button
and, if there isn't any error message (by the way, you can copy the text from the listing above into the MetaEditor), press "Save" button to save the Expert Advisor.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1510](https://www.mql5.com/ru/articles/1510)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**[Go to discussion](https://www.mql5.com/en/forum/39562)**

![Requirements Applicable to Articles Offered for Publishing at MQL4.com](https://c.mql5.com/2/17/99_6.gif)[Requirements Applicable to Articles Offered for Publishing at MQL4.com](https://www.mql5.com/en/articles/1402)

Requirements Applicable to Articles Offered for Publishing at MQL4.com

![What the Numbers in the Expert Testing Report Mean](https://c.mql5.com/2/17/81_1.gif)[What the Numbers in the Expert Testing Report Mean](https://www.mql5.com/en/articles/1486)

Article explains how to read testing reports and to interpret the obtained results properly.

![How to Evaluate the Expert Testing Results](https://c.mql5.com/2/13/125_2.gif)[How to Evaluate the Expert Testing Results](https://www.mql5.com/en/articles/1403)

The article gives formulas and the calculation order for data shown in the Tester report.

![Testing Features and Limits in MetaTrader 4](https://c.mql5.com/2/17/80_1.gif)[Testing Features and Limits in MetaTrader 4](https://www.mql5.com/en/articles/1512)

This article allows to find out more about features and limits of Strategy Tester in MetaTrader 4.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/1510&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071881233283100725)

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