---
title: How to Quickly Create an Expert Advisor for Automated Trading Championship 2010
url: https://www.mql5.com/en/articles/148
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:52:33.139635
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ebttrbukobenmmwkrivxuhfdjzficqnd&ssn=1769158352648029897&ssn_dr=0&ssn_sr=0&fv_date=1769158352&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F148&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Quickly%20Create%20an%20Expert%20Advisor%20for%20Automated%20Trading%20Championship%202010%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915835224433096&fz_uniq=5062783281369360591&sv=2552)

MetaTrader 5 / Examples


### Introduction

In order to develop an expert to participate in [Automated Trading Championship 2010](https://championship.mql5.com/2010/en), let's use a template of ready expert advisor from [The Prototype of Trade Robot](https://www.mql5.com/en/articles/132) article. Even novice MQL5 programmer will be capable of this task, because for your strategies the  basic classes, functions, templates are already developed. It's enough to write a minimal amount of code to implement your trading idea.


What we will need to prepare:


- Selection of strategy

- Writing an Expert Advisor

- Testing

- Optimization in Strategy Tester

- Optimization of the strategy

- Testing on different intervals


### 1\. Selection of Strategy

It is believed that trading with trend is more profitable than trading in a range, and the bounce from the intraday levels occurs more frequently than the breakdown of channel borders.

Based on these assumptions, we will open position towards the current trend on the bounce from the channel boundaries (Envelopes). We'll close position on a signal to close position or when the Stop Loss or Take Profit levels will be reached.



As the trend signal we'll use MACD growth or dwindling on the daily chart, and we will trade on the bounce from the channel boundaries on the hour timeframe.


![Figure 1. MACD Indicator on EURUSD Daily Chart](https://c.mql5.com/2/2/Fig1.png)

Figure 1. MACD Indicator on EURUSD Daily Chart

If MACD indicator grows on two bars in succession - this is the Buy signal. If it dwindles on two bars in succession - this is the Sell signal.


![Figure 2. Price Bounce from the Envelopes Boundaries](https://c.mql5.com/2/2/Fig.gif)

Figure 2. Price Bounce from the Envelopes Boundaries

### 2\. Writing an Expert Advisor

**2.1. Included Modules**

The expert will use the ExpertAdvisor class from the ExpertAdvisor.mqh module.


```
#include <ExpertAdvisor.mqh>
```

**2.2. Input Variables**

```
input int    SL        =  50; // Stop Loss distance
input int    TP        = 100; // Take Profit distance
input int    TS        =  50; // Trailing Stop distance
input int    FastEMA   =  15; // Fast EMA
input int    SlowEMA   =  26; // Slow EMA
input int    MACD_SMA  =   1; // MACD signal line
input int    EnvelPer  =  20; // Envelopes period
input double EnvelDev  = 0.4; // Envelopes deviation
input double Risk      = 0.1; // Risk
```

**2.3. Create a Class Inherited From CExpertAdvisor**

```
class CMyEA : public CExpertAdvisor
  {
protected:
   double            m_risk;          // size of risk
   int               m_sl;            // Stop Loss
   int               m_tp;            // Take Profit
   int               m_ts;            // Trailing Stop
   int               m_pFastEMA;      // Fast EMA
   int               m_pSlowEMA;      // Slow EMA
   int               m_pMACD_SMA;     // MACD signal line
   int               m_EnvelPer;      // Envelopes period
   double            m_EnvelDev;      // Envelopes deviation
   int               m_hmacd;         // MACD indicator handle
   int               m_henvel;        // Envelopes indicator handle
public:
   void              CMyEA();
   void             ~CMyEA();
   virtual bool      Init(string smb,ENUM_TIMEFRAMES tf); // initialization
   virtual bool      Main();                              // main function
   virtual void      OpenPosition(long dir);              // open position on signal
   virtual void      ClosePosition(long dir);             // close position on signal
   virtual long      CheckSignal(bool bEntry);            // check signal
  };
//------------------------------------------------------------------
```

**2.4. Delete Indicators**

```
//------------------------------------------------------------------
void CMyEA::~CMyEA()
  {
   IndicatorRelease(m_hmacd);  // delete MACD indicator
   IndicatorRelease(m_henvel); // delete Envelopes indicator
  }
//------------------------------------------------------------------
```

**2.5. Initialize Variables**

```
//------------------------------------------------------------------    Init
bool CMyEA::Init(string smb,ENUM_TIMEFRAMES tf)
  {
   if(!CExpertAdvisor::Init(0,smb,tf)) return(false);    // initialize parent class
   // copy parameters
    m_risk=Risk;
   m_tp=TP;
   m_sl=SL;
   m_ts=TS;
   m_pFastEMA=FastEMA;
   m_pSlowEMA=SlowEMA;
   m_pMACD_SMA=MACD_SMA;
   m_EnvelPer = EnvelPer;
   m_EnvelDev = EnvelDev;
   m_hmacd=iMACD(m_smb,PERIOD_D1,m_pFastEMA,m_pSlowEMA,m_pMACD_SMA,PRICE_CLOSE);      // create MACD indicator
   m_henvel=iEnvelopes(m_smb,PERIOD_H1,m_EnvelPer,0,MODE_SMA,PRICE_CLOSE,m_EnvelDev); // create Envelopes indicator
   if(m_hmacd==INVALID_HANDLE ||m_henvel==INVALID_HANDLE ) return(false);             // if there is an error, then exit
   m_bInit=true;
   return(true);                                                                      // trade allowed
  }
```

**2.6. Trade Function**

```
//------------------------------------------------------------------    CheckSignal
long CMyEA::CheckSignal(bool bEntry)
  {
   double macd[4],   // Array of MACD indicator values
         env1[3],    // Array of Envelopes' upper border values
         env2[3];    // Array of Bollinger Bands' lower border values
   MqlRates rt[3];   // Array of price values of last 3 bars

   if(CopyRates(m_smb,m_tf,0,3,rt)!=3) // Copy price values of last 3 bars to array
     {
       Print("CopyRates ",m_smb," history is not loaded");
        return(WRONG_VALUE);
     }
   // Copy indicator values to array
   if(CopyBuffer(m_hmacd,0,0,4,macd)<4 || CopyBuffer(m_henvel,0,0,2,env1)<2 ||CopyBuffer(m_henvel,1,0,2,env2)<2)
     {
        Print("CopyBuffer - no data");
       return(WRONG_VALUE);
     }
   // Buy if MACD is growing and if there is a bounce from the Evelopes' lower border
   if(rt[1].open<env2[1] && rt[1].close>env2[1] && macd[1]<macd[2] &&  macd[2]<macd[3])
      return(bEntry ? ORDER_TYPE_BUY:ORDER_TYPE_SELL); // condition for buy
   // Sell if MACD is dwindling and if there is a bounce from the Evelopes' upper border
   if(rt[1].open>env1[1] && rt[2].close<env1[1]&& macd[1]>macd[2] &&  macd[2]>macd[3])
      return(bEntry ? ORDER_TYPE_SELL:ORDER_TYPE_BUY); // condition for sell

   return(WRONG_VALUE); // if there is no signal
  }

CMyEA ea; // class instance
```

And so, after writing the code, send the resulting expert to Strategy Tester.


### 3\. Testing

In Strategy Tester for the "Last year" period on EURUSD we get the following chart:


![Figure 3. Results of Testing the Trading System with Initial Parameters](https://c.mql5.com/2/2/Fig3.png)

Figure 3. Results of Testing the Trading System with Initial Parameters

The results are not impressive, so let's start to optimize the Stop Loss and Take Profit levels.


### 4\. Optimization in Strategy Tester

We will optimize the Stop Loss and Take Profit parameters at the interval 10-500 with step 50.

Best results: Stop Loss = 160, Take Profit = 310. After Stop Loss and Take Profit optimization we received 67% of profitable trades against the previous 36% and net profit of $1522.97. Thus, by simple manipulations, we've upgraded our system to break-even and even got some profit.


![Figure 4. Results of Testing the Trading System with Optimized Stop Loss and Take Profit](https://c.mql5.com/2/2/Fig4.png)

Figure 4. Results of Testing the Trading System with Optimized Stop Loss and Take Profit

Next let's optimize the Envelopes period and deviation.

Envelopes period will change from 10 to 40 with step 4, and deviation - from 0.1 to 1 with step 0.1.

The best optimization results are: Envelopes period = 22, Envelopes deviation = 0.3. Even now we've got $14418.92 net profit and 79% of profitable trades.


![Figure 5. Results of Testing the Trading System with Optimized Envelopes Period and Deviation](https://c.mql5.com/2/2/Fig5.png)

Figure 5. Results of Testing the Trading System with Optimized Envelopes Period and Deviation

If we increase the risk to 0.8, we'll get $77330.95 net profit.


![Figure 6. The Results of Testing the Trading System with Optimized Risk](https://c.mql5.com/2/2/Fig6.png)

Figure 6. The Results of Testing the Trading System with Optimized Risk

### 5\. Optimization of the Strategy

Optimization of the strategy may consist of the following steps:


- Change trend indicator

- Select another envelope

- Select another timeframe

- Change trade conditions


**5.1. Change Trend Indicator**

As we can see from the [Several Ways of Finding a Trend in MQL5](https://www.mql5.com/en/articles/136) article, the best trend indicators are moving average and a "fan" of moving averages.


Let's replace the MACD indicator with simple moving average. The expert's code can be found in the attached Macena.mq5 file.


**5.2. Select Another Envelope**

Besides the **Envelopes** you can also select another envelope at our disposal. For example, **Price Channel**, **Bollinger Bands** or an envelope based on moving averages.


An example of expert, that uses MA and Bollinger Bands, can be found in the attached Maboll.mq5 file.


**5.3. Select Another Timeframe**

Let's change the timeframe to bigger or lesser. As a bigger timeframe - take H4, as the lesser - M15, and then test and optimize your system.

To do this, replace only one line in the code:


```
m_henvel=iEnvelopes(m_smb,PERIOD_H1,m_EnvelPer,0,MODE_SMA,PRICE_CLOSE,m_EnvelDev);  // create Envelopes indicator
```

In the case of H4 timeframe:


```
m_henvel=iEnvelopes(m_smb,PERIOD_H4,m_EnvelPer,0,MODE_SMA,PRICE_CLOSE,m_EnvelDev);  // create Envelopes indicator
```

For the M15 timeframe:


```
m_henvel=iEnvelopes(m_smb,PERIOD_M15,m_EnvelPer,0,MODE_SMA,PRICE_CLOSE,m_EnvelDev);  // create Envelopes indicator
```

**5.4. Change Trade Conditions**

As an experiment, also let's change trade conditions.


1. Make the system able to reverse. We will buy on the bounce from the Envelope's lower boundary, and sell on the bounce from the Envelopes upper boundary.

2. Check the system without following the day trend. This is done simply by inserting the following code into trade block:



```
//------------------------------------------------------------------ CheckSignal
long CMyEA::CheckSignal(bool bEntry)
     {
      double env1[3],   // Array of Envelopes' upper border values
            env2[3];    // Array of Bollinger Bands' lower border values
      MqlRates rt[3];   // Array of price values of last 3 bars

      if(CopyRates(m_smb,m_tf,0,3,rt)!=3) // Copy price values of last 3 bars to array
        {
         Print("CopyRates ",m_smb," history is not loaded");
         return(WRONG_VALUE);
        }
// Copy indicator values to array
      if(CopyBuffer(m_henvel,0,0,2,env1)<2 || CopyBuffer(m_henvel,1,0,2,env2)<2)
        {
         Print("CopyBuffer - no data");
         return(WRONG_VALUE);
        }
// Buy if there is a bounce from the Evelopes' lower border
      if(rt[1].open<env2[1] && rt[1].close>env2[1])
         return(bEntry ? ORDER_TYPE_BUY:ORDER_TYPE_SELL); // condition for buy
// Sell if there is a bounce from the Evelopes' upper border
      if(rt[1].open>env1[1] && rt[2].close<env1[1])
         return(bEntry ? ORDER_TYPE_SELL:ORDER_TYPE_BUY); // condition for sell

      return(WRONG_VALUE); // if there is no signal
     }

CMyEA ea; // class instance
//------------------------------------------------------------------    OnInit
```


      3\. We will close short position when the price did not go far down, but turned and went up.


      4\. We will close long position when the price did not go far up, but turned and went down.


You can invent many other ways to optimize a trading strategy, some of them are described in corresponding literature.

Further researches are up to you.


### 6\. Testing on Different Intervals

Test our Expert Advisor on equal intervals of time with a shift of 1 month. Let's take the "Last year" as a testing period. Period of time - 3 months.


| Testing interval | Profit, USD | Profitable trades |
| --- | --- | --- |
| 1.01.2010 - 30.03.2010 | 7239.50 | 76.92% |
| 1.02.2010 - 30.04.2010 | -6577.50 | 0% |
| 1.03.2010 - 30.05.2010 | -8378.50 | 50% |
| 1.04.2010 - 30.06.2010 | -6608.00 | 0% |
| 1.05.2010 - 30.07.2010 | 41599.50 | 80% |
| 1.06.2010 - 30.08.2010 | 69835.50 | 85% |

**Summary**: It's not desirable to use Expert Advisor with such an aggressive money management. Reduce the risk.


### Conclusion

Brief conclusion: on the basis on this template you can quite quickly implement your trading idea with minimum of time and effort.

Optimization of system parameters and trade criteria is also makes no problems.

To create a more stable working trading system, it is desirable to optimize all parameters over longer time intervals.


**List of used sources:**

1. [20 Trade Signals in MQL5](https://www.mql5.com/en/articles/130) article.

2. [The Prototype of Trade Robot](https://www.mql5.com/en/articles/132) article.

3. [Several Ways of Detecting a Trend in MQL5](https://www.mql5.com/en/articles/136) article.

4. [Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization](https://www.mql5.com/en/articles/1517) article.

5. [Limitations and Verifications in Expert Advisors](https://www.mql5.com/en/articles/22) article.

6. [Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://www.mql5.com/en/articles/116) article.

7. [Functions for Money Management in an Expert Advisor](https://www.mql5.com/en/articles/113) article.



Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/148](https://www.mql5.com/ru/articles/148)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/148.zip "Download all attachments in the single ZIP archive")

[expertadvisor\_\_1.mqh](https://www.mql5.com/en/articles/download/148/expertadvisor__1.mqh "Download expertadvisor__1.mqh")(17.92 KB)

[mabol.mq5](https://www.mql5.com/en/articles/download/148/mabol.mq5 "Download mabol.mq5")(6.65 KB)

[macena.mq5](https://www.mql5.com/en/articles/download/148/macena.mq5 "Download macena.mq5")(6.94 KB)

[maena.mq5](https://www.mql5.com/en/articles/download/148/maena.mq5 "Download maena.mq5")(6.58 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1936)**
(7)


![Vladislav Andruschenko](https://c.mql5.com/avatar/2026/1/695fc0e9-27f5.png)

**[Vladislav Andruschenko](https://www.mql5.com/en/users/vladon)**
\|
28 Aug 2010 at 15:47

Hahaha. Sorry for the off-topic. Before writing this article I did exactly that. and I used 2 articles, prototype and [20 trading signals](https://www.mql5.com/en/articles/130 "Article: 20 Trading Signals in MQL5") and found my strategy, changed the code a bit, added a couple of functions and the results were excellent.


![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
8 Sep 2010 at 10:24

Please give some sample code for folloe thr rules of Championship 2010.

It is not easy to code since using position, order, and deal to manage order in mql5, and some functions like ordertotal() not always work right.

for example, iinsdie Ontrade() function,   ordertotal()  always return  0 at [test mode](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info#enum_mql5_info_integer "MQL5 documentation: Running MQL5 Program Properties").

![koko](https://c.mql5.com/avatar/avatar_na2.png)

**[koko](https://www.mql5.com/en/users/kmtm)**
\|
23 Sep 2010 at 23:06

Where all the files go? I put them in" C:\\Program Files\\MetaTrader 5\\MQL5\\Experts " and try to compile them and compiler shows a bunch of errors...?!?

Although I've managed to write few indicators in MQL5 I dont like the way MQL is going... too complicated.

I'm on Win7 x64. I cant believe I  cant use already written code...!!!

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
24 Sep 2010 at 03:00

Read client terminal help ( **F1**)

[![](https://c.mql5.com/3/2/24.09__1.png)](https://c.mql5.com/3/2/24.09.png "https://c.mql5.com/3/2/24.09.png")

![KjLNi](https://c.mql5.com/avatar/avatar_na2.png)

**[KjLNi](https://www.mql5.com/en/users/kjlni)**
\|
22 Jan 2020 at 15:50

Hello, thanks for this article.

A little question: at the beginning of the programm, it talks about the include "ExpertAdvisor.mqh".

When I tried to code this, it told me that this include does not exist.

Has there been a name change? or am I doing something wrong?

thanks in advance for your help.

![Using the TesterWithdrawal() Function for Modeling the Withdrawals of Profit](https://c.mql5.com/2/0/TesterWithdrawal_MQL5.png)[Using the TesterWithdrawal() Function for Modeling the Withdrawals of Profit](https://www.mql5.com/en/articles/131)

This article describes the usage of the TesterWithDrawal() function for estimating risks in trade systems which imply the withdrawing of a certain part of assets during their operation. In addition, it describes the effect of this function on the algorithm of calculation of the drawdown of equity in the strategy tester. This function is useful when optimizing parameter of your Expert Advisors.

![20 Trade Signals in MQL5](https://c.mql5.com/2/0/20_Trading_Signals_MQL5__1.png)[20 Trade Signals in MQL5](https://www.mql5.com/en/articles/130)

This article will teach you how to receive trade signals that are necessary for a trade system to work. The examples of forming 20 trade signals are given here as separate custom functions that can be used while developing Expert Advisors. For your convenience, all the functions used in the article are combined in a single mqh include file that can be easily connected to a future Expert Advisor.

![Interview with Alexander Topchylo (ATC 2010)](https://c.mql5.com/2/0/35.png)[Interview with Alexander Topchylo (ATC 2010)](https://www.mql5.com/en/articles/527)

Alexander Topchylo (Better) is the winner of the Automated Trading Championship 2007. Alexander is an expert in neural networks - his Expert Advisor based on a neural network was on top of best EAs of year 2007. In this interview Alexander tells us about his life after the Championships, his own business and new algorithms for trading systems.

![The Prototype of a Trading Robot](https://c.mql5.com/2/0/Prototype_Expert_Advisor_MQL5.png)[The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

This article summarizes and systematizes the principles of creating algorithms and elements of trading systems. The article considers designing of expert algorithm. As an example the CExpertAdvisor class is considered, which can be used for quick and easy development of trading systems.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lvtubvdxrbhqwnmcitbwsiaooioobdet&ssn=1769158352648029897&ssn_dr=0&ssn_sr=0&fv_date=1769158352&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F148&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Quickly%20Create%20an%20Expert%20Advisor%20for%20Automated%20Trading%20Championship%202010%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915835224493637&fz_uniq=5062783281369360591&sv=2552)

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