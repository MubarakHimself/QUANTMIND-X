---
title: Evaluating the effectiveness of trading systems by analyzing their components
url: https://www.mql5.com/en/articles/1924
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:45:05.327769
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=vdfxuxzalserqqdfqwvtthldaukfgxml&ssn=1769186704232648943&ssn_dr=0&ssn_sr=0&fv_date=1769186704&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1924&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Evaluating%20the%20effectiveness%20of%20trading%20systems%20by%20analyzing%20their%20components%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176918670431159677&fz_uniq=5070531419551242128&sv=2552)

MetaTrader 4 / Tester


### Introduction

Anyone who starts trading in financial markets very quickly realizes, that in this line of business success is only possible with a certain systematic approach. Spontaneous trading can be a matter of luck, hit-and-miss, emotional and normally doesn't help to achieve positive results and if it does, the success is brief and the consequences can be dramatic. Therefore, any analysis, whether graphic, based on indicators, or any other, is one of the key components of successful trading in financial markets. This article is to some extent a research of few simple and independent trading systems for analyzing their effectiveness and usefulness of the joint application.

### Setting criteria for evaluation of the trading system's effectiveness

Usually, evaluation of the effectiveness of any trading system is a certain set of defined and specified parameters, as well as some of the resulting values ​​of the system. Periods of indicators, Stop Loss or Take Profit sizes, or a more complex set of system's coefficients that effect the entry and exit from the market, can be used as set parameters. The resulting values in turn are net profit, drawdown, percentage of successful transactions or the average value of profit trades in the currency deposit.

Any trading system over time has its effectiveness reduced, since the nature of currency markets is constantly changing, as it becomes evident from the resulting indicators. Therefore, the set system parameters have to be changed and adjusted to the changing conditions. In this article we aim to define the concept of a complex trading system.

_Complex trading system_ is a set of separate blocks that have their own algorithms and parameters and work in conjunction with each other. The operation of any of these blocks included in the system can be evaluated by a given set of criteria. We will consider the evaluation scheme of system's effectiveness shown in Figure 1. Complex system has three constituent blocks A, B, C and each of them has its personal operating parameters. Operation and effectiveness of this system can be evaluated by three dimensions: 1, 2, 3. Over time the market conditions, where this system trades, change, and the parameters 1-3 are changed towards unsatisfactory side, which signals that it is time to reconfigure the system. This can be done using the following methods:

1. Start optimization of all nine parameters in order to adapt the system to the current market realities. This method, however, contains redundancy - why optimize all parameters, if it is possible to find out which block has started operating worse?
2. Accordingly, the second method. A common set of evaluation criteria C1, C2, C3 for all three system blocks is created, which allows to evaluate and compare the operation efficiency for each of the blocks separately.

Advantages of this method:

- Redundancy avoidance. There is no need in optimization, if blocks operate smoothly.
- Monitoring system. The parameters C1-C3 can be measured after some periods of time, thereby clarifying the behavior of the system and its units together and separately at different times of the market (trading sessions, news release, etc.).
- Vulnerability analysis. It allows to determine, which blocks drag the entire system down, in order to optimize, replace or improve solely the particular block, rather than the whole system.

![](https://c.mql5.com/2/23/_image_en.png)

Fig.1. Complex trading system

### Example of evaluating the effectiveness of trading system by analyzing its components

In order to test the efficiency of the entire trading system, we need to check how do the system blocks operate separately from each other and how they work together. The testing system will consist of two blocks:

1. Block A. It is based on the signals of the standard indicator Parabolic SAR.
2. Block B. It is based on the signals of the standard indicator Accelerator Oscillator (AC).

Immediately we will define a set of criteria for evaluating these blocks:

- Criteria C1 — net profit.
- Criteria C2 — maximum drawdown.
- Criteria C3 — profit trades (% of all).

For complete clarity we will choose the same indicators as the parameters of overall efficiency:

- Parameter 1 — net profit.
- Parameter 2 — maximum drawdown.
- Parameter 3 — profitable trades (% of all).

Examples of market entry signals for these indicators are located in the Docs of MQL5 Standard Library: [Parabolic SAR signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_sar) and [Accelerator Oscillator signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ac) (first conditions). For each of them we will describe the entry conditions and find out the conditions under which they are effective at a predetermined interval testing.

All the parameters of the test Expert Advisor are written the following way:

![](https://c.mql5.com/2/19/param__2.png)

Fig.2. Parameters of Expert Advisor

```
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                Alexander Fedosov |
//|                           https://www.mql5.com/ru/users/alex2356 |
//+------------------------------------------------------------------+
#property copyright "Alexander Fedosov"
#property link      "https://www.mql5.com/ru/users/alex2356"
#property version   "1.00"
#property strict

#include "trading.mqh"

input int            tm = 1;                       //Test mode 1,2 or 3
input int            SL = 40;                      //Stop-loss
input int            TP = 70;                      //Take-profit
input bool           lot_const = false;            //Lot of balance?
input double         lt=0.01;                      //Lot if Lot of balance=false
input double         Risk=2;                       //The risk in the lot of the balance, %
input int            Slippage= 5;                  //Slippage
input int            magic = 2356;                 //Magic number
input ENUM_TIMEFRAMES tf1 = PERIOD_H1;             //Timeframe for the calculation module1
input ENUM_TIMEFRAMES tf2 = PERIOD_M5;             //Timeframe for the calculation module2
input double         Step = 0.02;                  //Step PSAR
input double         Mxm = 0.2;                    //Maximum PSAR

CTrading tr(magic,Slippage,lt,lot_const,Risk,5);
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(tm<1 || tm>3)
      return;
//--- Market entry conditions for Parabolic SAR module
   if(tm==1 && !tr.isOpened(magic))
     {
      double psar[],prc[];
      ArrayResize(psar,3);
      ArrayResize(prc,3);
      for(int i=0; i<3; i++)
        {
         psar[i]=iSAR(_Symbol,tf1,Step,Mxm,i);
         prc[i]=iClose(_Symbol,tf1,i);
        }

      if(psar[2]>prc[2] && psar[1]<prc[1] && psar[0]<prc[0])
         tr.OpnOrd(OP_BUY,lt,TP,SL);
      if(psar[2]<prc[2] && psar[1]>prc[1] && psar[0]>prc[0])
         tr.OpnOrd(OP_SELL,lt,TP,SL);

     }
//--- Market entry conditions for AC module
   if(tm==2 && !tr.isOpened(magic))
     {
      double ac[];
      ArrayResize(ac,3);
      for(int i=0; i<3; i++)
         ac[i]=iAC(Symbol(),tf2,i);

      if(ac[2]>0 && ac[1]>0 && ac[1]>ac[2])
         tr.OpnOrd(OP_BUY,lt,TP,SL);
      if(ac[2]<0 && ac[1]<0 && ac[1]<ac[2])
         tr.OpnOrd(OP_SELL,lt,TP,SL);

     }
//--- Market entry conditions when two modules operate together
   if(tm==3 && !tr.isOpened(magic))
     {
      double psar[],prc[],ac[];
      ArrayResize(psar,3);
      ArrayResize(prc,3);
      ArrayResize(ac,3);
      for(int i=0; i<3; i++)
        {
         psar[i]=iSAR(_Symbol,tf1,Step,Mxm,i);
         prc[i]=iClose(_Symbol,tf1,i);
         ac[i]=iAC(Symbol(),tf2,i);
        }
      if((psar[2]>prc[2] && psar[1]<prc[1] && psar[0]<prc[0]) || (ac[2]>0 && ac[1]>0 && ac[1]>ac[2]))
         tr.OpnOrd(OP_BUY,lt,TP,SL);
      if((psar[2]<prc[2] && psar[1]>prc[1] && psar[0]>prc[0]) || (ac[2]<0 && ac[1]<0 && ac[1]<ac[2]))
         tr.OpnOrd(OP_SELL,lt,TP,SL);

     }

  }
//+------------------------------------------------------------------+
```

The first parameter of EA is **tm**(Test mode) and it can hold values 1, 2 or 3. These values ​​correspond to the three modes of operation:

- tm = 1. Only the market entry conditions with Parabolic SAR indicator are used. Operation mode only for A block.
- tm = 2. Only the market entry conditions with Accelerator Oscillator indicator are used. Operation mode only for B block.
- tm = 3. Joint operation of both blocks. Whole system operates.

By changing the Test Mode parameter we can discover the parameters of our interest: for А and B blocks - C1-C3, for the whole system — parameters 1-3. The following are the block's testing results on Parabolic SAR (Fig.3), AC (Fig. 4) and their joint operation (Fig. 5).

![](https://c.mql5.com/2/23/Im1.png)

Fig.3. Testing Parabolic SAR module

![](https://c.mql5.com/2/23/Im2.png)

Fig.4. Testing AC modules

![](https://c.mql5.com/2/23/Im3.png)

Fig.5. Joint operation of modules Parabolic SAR and AC

Based on the results of three different operating modes the comparative table will be provided for clarification purposes:

| Test Mode | Net Profit | Profit trades,% | Maximum drawdown |
| --- | --- | --- | --- |
| 1 | 27,56 | 37,91 | 32,71 |
| 2 | 106,98 | 39,19 | 38,94 |
| 3 | 167,16 | 40,64 | 18,62 |

### Simulation of the system deterioration

Let's change the operational parameters of the block A (Test Mode = 1) towards the side of its efficiency reduction. This will give us the answer to the question of what will happen to the system, in case one of its blocks changes for the worse.

To simulate the decrease of the system's efficiency we will change one parameter related to the operation of block A - **Timeframe for the calculation module1 -** the following way as shown in Fig.6. This will change the period of its calculation based on Parabolic SAR which will affect the market entry points, and hence change the efficiency of the entire system.

![](https://c.mql5.com/2/19/param-1__1.png)

Fig.6. Simulation of decreasing the system's efficiency

![](https://c.mql5.com/2/23/Im4.png)

Fig.7. The result of changing the module's parameter based on Parabolic SAR

For illustrative purposes we will compare the observed performance of the block A with two different values ​​of its parameters **Timeframe for the calculation module1:**

| **Timeframe for the calculation module1** | Net profit | Profit trades,% | Maximum drawdown |
| --- | --- | --- | --- |
| 1 Hour | 27.56 | 37.91 | 32.71 |
| 4 Hours | 2.54 | 36.73 | 43.21 |

This is evident, that the calculation of 4-hour period gave us poorer results of the module operation for all three evaluation criteria. Testing the entire system with deliberately degraded values of one of its components has also reduced the effectiveness of the parameters 1-3, as shown in Figure 8.

![](https://c.mql5.com/2/23/Im5.png)

Fig.8. Joint operation of Parabolic SAR (deteriorated) and AC

Now we will put all test results together to determine how the changes in the block A have effected the overall efficiency of the system.

| **Timeframe for the calculation module1** | Net profit (module 1) | Profit trades,%<br>(module 1) | Maximum drawdown,% <br>(module 1) | Net profit<br>(system) | Profit trades,%<br> (system) | Maximum drawdown,%<br>(system) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 Hours | 2.54 | 36.73 | 43.21 | 139.34 | 40.00 | 24.9 |
| 1 Hour | 27.56 | 37.91 | 32.71 | 167.16 | 40.64 | 18.62 |

The results showed, that researching each block of the complex system separately in order to improve the efficiency has had a positive effect on the evaluation parameters of the entire system. The theses in the beginning of this article have also got confirmed that this method is less redundant than the optimization of the entire system as a single object; moreover, this method provides the additional abilities to monitor the system.

### Conclusion

This article has considered the method of evaluating the effectiveness of trading systems by analyzing their components. On the basis of testing and researching the component of the block system and its modules, the following conclusions can be drawn:

- This method of evaluation is less redundant than the optimization of all system parameters by identifying only those components that are required to be optimized and improved.
- The creation of trading systems in form of the components using blocks that can be measured by set criteria, allowing to manage them better, identify weaknesses and upgrade with more flexibility.

A separate folder has to be created for correct testing, where both files should be placed, for example MQL4\\Experts\\article1924.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1924](https://www.mql5.com/ru/articles/1924)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1924.zip "Download all attachments in the single ZIP archive")

[trading.mqh](https://www.mql5.com/en/articles/download/1924/trading.mqh "Download trading.mqh")(30.03 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [A system of voice notifications for trade events and signals](https://www.mql5.com/en/articles/8111)
- [Quick Manual Trading Toolkit: Working with open positions and pending orders](https://www.mql5.com/en/articles/7981)
- [Quick Manual Trading Toolkit: Basic Functionality](https://www.mql5.com/en/articles/7892)
- [Multicurrency monitoring of trading signals (Part 5): Composite signals](https://www.mql5.com/en/articles/7759)
- [Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)
- [Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://www.mql5.com/en/articles/7600)
- [Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://www.mql5.com/en/articles/7528)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/84429)**
(1)


![Aida Chavez](https://c.mql5.com/avatar/2017/3/58C1FA6D-13DC.jpg)

**[Aida Chavez](https://www.mql5.com/en/users/gastor)**
\|
11 Aug 2016 at 18:13

Very good


![Graphical Interfaces V: The Vertical and Horizontal Scrollbar (Chapter 1)](https://c.mql5.com/2/22/v-avatar__2.png)[Graphical Interfaces V: The Vertical and Horizontal Scrollbar (Chapter 1)](https://www.mql5.com/en/articles/2379)

We are still discussing the development of the library for creating graphical interfaces in the MetaTrader environment. In the first article of the fifth part of the series, we will write classes for creating vertical and horizontal scrollbars.

![Calculator of signals](https://c.mql5.com/2/22/calculator_signal.png)[Calculator of signals](https://www.mql5.com/en/articles/2329)

The calculator of signals operates directly from the MetaTrader 5 terminal, which is a serious advantage, since the terminal provides a preliminary selection and sorts out signals. This way, users can see in the terminal only the signals that ensure a maximum compatibility with their trading accounts.

![Graphical Interfaces V: The List View Element (Chapter 2)](https://c.mql5.com/2/22/v-avatar.png)[Graphical Interfaces V: The List View Element (Chapter 2)](https://www.mql5.com/en/articles/2380)

In the previous chapter, we wrote classes for creating vertical and horizontal scrollbars. In this chapter, we will implement them. We will write a class for creating the list view element, a compound part of which will be a vertical scrollbar.

![Applying fuzzy logic in trading by means of MQL4](https://c.mql5.com/2/20/fuzzy-logic1.png)[Applying fuzzy logic in trading by means of MQL4](https://www.mql5.com/en/articles/2032)

The article deals with examples of applying fuzzy set theory in trading by means of MQL4. The use of FuzzyNet library for MQL4 in the development of an indicator and an Expert Advisor is described as well.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lhcauujhbuunfhyifltdgpjumpysztud&ssn=1769186704232648943&ssn_dr=0&ssn_sr=0&fv_date=1769186704&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1924&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Evaluating%20the%20effectiveness%20of%20trading%20systems%20by%20analyzing%20their%20components%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176918670431016932&fz_uniq=5070531419551242128&sv=2552)

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