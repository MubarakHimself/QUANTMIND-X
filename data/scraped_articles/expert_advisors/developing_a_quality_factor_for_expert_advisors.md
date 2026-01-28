---
title: Developing a quality factor for Expert Advisors
url: https://www.mql5.com/en/articles/11373
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:29:45.405278
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11373&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068305754613544919)

MetaTrader 5 / Examples


### Introduction

In this article, we will see how to develop a quality score that your Expert Advisor can display in the strategy tester. In Figure 1 below, you can see that the "OnTester result" value was 1.0639375, which shows an example of the quality of the system that was executed. In this article, we will learn two possible approaches to measuring system quality, and will see how to log both values, since we can only return one of them.

![](https://c.mql5.com/2/54/707135741355.png)

**Figure 1**: Highlighted "OnTester result" field.

### Starting a trading model and building an EA

Before addressing the system quality factor, it is necessary to create a basic system that will be used in tests. We have chosen a simple system: we will choose a random number and, if it is even, we will open a buy position; otherwise we will enter a sell position since the number is odd.

To hold the draw, we will use the _MathRand()_ function, which will provide a number between 0 (zero) and 32767. In addition, to make the system more balanced, we will add two complementary rules. With these three rules we will try to ensure a more reliable system. Here they are:

- When we not in a position, generate a random number

  - If the number is 0 or 32767, do nothing

  - If the number is even, buy the asset in the volume equal to the minimum lot size

  - If the number is odd, sell the asset in the volume equal to the minimum lot size

- When we are in a position, move the stop level in the direction of each new candlestick which is higher than the previous one in the direction of the movement

  - The stop level used will be based on the 1 period ATR indicator normalized with an 8 EMA. In addition, it will be located at the farthest end from the two candlesticks used for analysis


- If the time is outside the range from 11:00 to 16:00, we will not be allowed to open a position, and at 16:30 the position must be closed.


Below is the code used to develop these rules.

```
//--- Indicator ATR(1) with EMA(8) used for the stop level...
int ind_atr = iATR(_Symbol, PERIOD_CURRENT, 1);
int ind_ema = iMA(_Symbol, PERIOD_CURRENT, 8, 0, MODE_EMA, ind_atr);
//--- Define a variable that indicates that we have a deal...
bool tem_tick = false;
//--- An auxiliary variable for opening a position
#include<Trade/Trade.mqh>
#include<Trade/SymbolInfo.mqh>
CTrade negocios;
CSymbolInfo info;
//--- Define in OnInit() the use of the timer every second
//--- and start CTrade
int OnInit()
  {
//--- Set the fill type to keep a pending order
//--- until it is fully filled
   negocios.SetTypeFilling(ORDER_FILLING_RETURN);
//--- Leave the fixed deviation at it is not used on B3 exchange
   negocios.SetDeviationInPoints(5);
//--- Define the symbol in CSymbolInfo...
   info.Name(_Symbol);
//--- Set the timer...
   EventSetTimer(1);
//--- Set the base of the random number to have equal tests...
   MathSrand(0xDEAD);
   return(INIT_SUCCEEDED);
  }
//--- Since we set a timer, we need to destroy it in OnDeInit().
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
//--- The OnTick function only informs us that we have a new deal
void OnTick()
  {
   tem_tick = true;
  }
//+------------------------------------------------------------------+
//| Expert Advisor main function                                     |
//+------------------------------------------------------------------+
void OnTimer()
  {
   MqlRates cotacao[];
   bool fechar_tudo = false;
   bool negocios_autorizados = false;
//--- Do we have a new trade?
   if(tem_tick == false)
      return ;
//--- To check, return information of the last 3 candlesticks....
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 3, cotacao) != 3)
      return ;
//--- Is there a new candlestick since the last check?
   if(tem_vela_nova(cotacao[2]) == false)
      return ;
//--- Get data from the trade window and closing...
   negocios_autorizados = esta_na_janela_de_negocios(cotacao[2], fechar_tudo);
//--- If we are going to close everything and if there is a position, close it...
   if(fechar_tudo)
     {
      negocios.PositionClose(_Symbol);
      return ;
     }
//--- if we are not closing everything, move stop level if there is a position...
   if(arruma_stop_em_posicoes(cotacao))
      return ;
   if (negocios_autorizados == false) // are we outside the trading window?
      return ;
//--- We are in the trading window, try to open a new position!
   int sorteio = MathRand();
//--- Entry rule 1.1
   if(sorteio == 0 || sorteio == 32767)
      return ;
   if(MathMod(sorteio, 2) == 0)  // Draw rule 1.2 -- even number - Buy
     {
     negocios.Buy(info.LotsMin(), _Symbol);
     }
   else // Draw rule 1.3 -- odd number - Sell
     {
     negocios.Sell(info.LotsMin(), _Symbol);
     }
  }
//--- Check if we have a new candlestick...
bool tem_vela_nova(const MqlRates &rate)
  {
   static datetime vela_anterior = 0;
   datetime vela_atual = rate.time;
   if(vela_atual != vela_anterior) // is time different from the saved one?
     {
      vela_anterior = vela_atual;
      return true;
     }
   return false;
  }
//--- Check if the time is n the trade period to close positions...
bool esta_na_janela_de_negocios(const MqlRates &rate, bool &close_positions)
  {
   MqlDateTime mdt;
   bool ret = false;
   close_positions = true;
   if(TimeToStruct(rate.time, mdt))
     {
      if(mdt.hour >= 11 && mdt.hour < 16)
        {
         ret = true;
         close_positions = false;
        }
      else
        {
         if(mdt.hour == 16)
            close_positions = (mdt.min >= 30);
        }
     }
   return ret;
  }
//---
bool arruma_stop_em_posicoes(const MqlRates &cotacoes[])
  {
   if(PositionsTotal()) // Is there a position?
     {
      double offset[1] = { 0 };
      if(CopyBuffer(ind_ema, 0, 1, 1, offset) == 1 // EMA successfully copied?
         && PositionSelect(_Symbol))  // Select the existing position!
        {
         ENUM_POSITION_TYPE tipo = (ENUM_POSITION_TYPE) PositionGetInteger(POSITION_TYPE);
         double SL = PositionGetDouble(POSITION_SL);
         double TP = info.NormalizePrice(PositionGetDouble(POSITION_TP));
         if(tipo == POSITION_TYPE_BUY)
           {
            if (cotacoes[1].high > cotacoes[0].high)
               {
                  double sl = MathMin(cotacoes[0].low, cotacoes[1].low) - offset[0];
                  info.NormalizePrice(sl);
                  if (sl > SL)
                     {
                        negocios.PositionModify(_Symbol, sl, TP);
                     }
               }
           }
         else // tipo == POSITION_TYPE_SELL
           {
           if (cotacoes[1].low < cotacoes[0].low)
               {
                  double sl = MathMax(cotacoes[0].high, cotacoes[1].high) + offset[0];
                  info.NormalizePrice(sl);
                  if (SL == 0 || (sl > 0 && sl < SL))
                     {
                        negocios.PositionModify(_Symbol, sl, TP);
                     }
               }
           }
        }
      return true;
     }
   // there was no position
   return false;
  }
```

Let's briefly consider the above code. We will use the average calculated from the ATR to determine the size of stops that will be placed at the candlestick boundaries when we find a candlestick that exceeds the previous one. This is done in the function _arruma\_stop\_em\_posicoes_. Whenever true is returned, there is a position and we should not advance in the main code in _OnTimer_. I use this function instead of _OnTick_ because I do not need a large function running for each trade performed. The function ishould be performed with each new candlestick of the defined period. In _OnTick_, the value set to true indicates a previous trade. This is necessary; otherwise, during periods when the market is closed, the strategy tester would introduce pauses as it would execute the function even without a previous trade.

Up to this point, everything has strictly followed the defined plan, including the two specified windows. The first is the window for opening trades, which is between 11:00 and 4:00. The second is the management window which allows the algorithm to manage the open trade by moving the stops until 16:30 – at this point it should close all trades for the day.

Please note that if we trade this EA now, the "OnTester result" will be zero, as seen in Figure 2, since we have not provided a calculation function for this value.

![](https://c.mql5.com/2/54/6164466945497.png)

**Figure 2:** EA executed on USDJPY, H1 in OHLC mode 1 minute in the period from 2023-01-01 to 2023-05-19

### About the quality factor

To enable the display of the "OnTester result" value, we need to define the _OnTester_ function which returns a _double_ value. simple as that! Here, using the code below, we get the result shown in Figure 3.

![EA executed on USDJPY, H1 in OHLC mode 1 minute in the period from 2023-01-01 to 2023-05-19.](https://c.mql5.com/2/55/4858630882261.png)

**Figure 3:** EA executed on USDJPY, H1 in OHLC mode 1 minute in the period from 2023-01-01 to 2023-05-19.

The following code should be placed at the end of the previous code. Here we calculate the average risk-reward ratio of trades: this ratio is usually expressed as the return received, since the risk is assumed to be constant, equal to 1. Thus, we can interpret the risk-return ratio as 1:1.23 or simply 1.23, and another example could be 0.43. In the first example, for every 1 dollar risked, we gain 1.23 dollars, while in the second example, for every 1 dollar risked, we lose 0.43. Therefore, when the return is 1 or close to it, it means we break even, and above it means we are winning.

Since the statistics do not provide the average amount gained or lost, we will use the gross value normalized by the quantity of trades on each side (buy or sell). When returning the values of executed trades for use, 1 is added. This way, if there are no profitable trades or no losing trades, the program will not terminate due to division by zero during the calculation. Additionally, to avoid displaying a large number of digits as seen in Figure 1 previously in which we had more than 5 digits, we used _NormalizeDouble_ to display the result with only two digits.

```
double OnTester()
  {
//--- Average profit
   double lucro_medio=TesterStatistics(STAT_GROSS_PROFIT)/(TesterStatistics(STAT_PROFIT_TRADES)+1);
//--- Average loss
   double prejuizo_medio=-TesterStatistics(STAT_GROSS_LOSS)/(TesterStatistics(STAT_LOSS_TRADES)+1);
//--- Risk calculation: profitability to be returned
   double rr_medio = lucro_medio / prejuizo_medio;
//---
   return NormalizeDouble(rr_medio, 2);
  }
```

The _OnTester_ function must exist in each Expert Advisor to ensure the value is displayed in the report. To minimize the work related to copying several lines of code, we will move the function into a separate file. This way we will only need to copy a single line every time. This is done as follows:

```
#include "ARTICLE_METRICS.mq5"
```

This way we have a concise code! In the specified file, the function will be defined by a definition. In case we want to use _include_, this would allow us to easily change the name of the function to be included, avoiding possible duplication errors if the OnTester function is already defined. So we can think of this as a mechanism to give preference to OnTester which will be inserted directly into the EA code. If we want to use it with _include_, we will simply comment out the OnTester function in the EA code and comment out the corresponding macro definition. We will get back to this a little later.

Initially, the ARTICLE\_METRICS.mq5 file will look like this:

```
//--- Risk calculation: average return on operation
double rr_medio()
  {
//--- Average profit
   double lucro_medio=TesterStatistics(STAT_GROSS_PROFIT)/(TesterStatistics(STAT_PROFIT_TRADES)+1);
//--- Average loss
   double prejuizo_medio=-TesterStatistics(STAT_GROSS_LOSS)/(TesterStatistics(STAT_LOSS_TRADES)+1);
//--- Risk calculation: profitability to be returned
   double rr_medio = lucro_medio / prejuizo_medio;
//---
   return NormalizeDouble(rr_medio, 2);
  }

//+------------------------------------------------------------------+
//| OnTester                                                         |
//+------------------------------------------------------------------+
#ifndef SQN_TESTER_ON_TESTER
#define SQN_TESTER_ON_TESTER OnTester
#endif
double SQN_TESTER_ON_TESTER()
  {
   return rr_medio();
  }
```

Please note that the correct file name must have the extension "mqh". However, since I plan to save the file in the Experts directory, I intentionally left the code extension. It is at your discretion.

### First version of quality calculation

Our first version of the quality calculation is based on the approach created by _Sunny Harris_, entitled _CPC Index_. This approach uses three metrics that are multiplied among them: risk:average return, success rate and profit ratio. However, we will modify it to not use the profit factor – instead, it will use the lowest value between the profit factor and the recovery factor. Although, if we consider the difference between the two, we should opt for the recovery factor, I preferred to leave it as is because it already leads to an improvement in the calculation.

The following code implements the approach mentioned in the previous paragraph. We just need to call it in _OnTester._ Please note that we have not added 1 to the number of trades here because the value provided is generic and we expect there to be at least 1 trade to evaluate.

```
//--- Calculating CPC Index by Sunny Harris
double CPCIndex()
  {
   double taxa_acerto=TesterStatistics(STAT_PROFIT_TRADES)/TesterStatistics(STAT_TRADES);
   double fator=MathMin(TesterStatistics(STAT_PROFIT_FACTOR), TesterStatistics(STAT_RECOVERY_FACTOR));
   return NormalizeDouble(fator * taxa_acerto * rr_medio(), 5);
  }
```

### Second version of quality calculation

The second quality factor we'll talk about is called _System Quality Index_ (SQN). It was created by Van Tharp. We will calculate the trades executed each month and obtain a simple average of all months in the simulation. SQN differs from the approach described in the previous section because it seeks to emphasize the stability of the trading system.

An important feature of SQN is that it penalizes systems that have pronounced spikes. Thus, if the system has a series of small trades and one large one, the latter will be penalized. This means that if we have a system with small losses and one large profit,that profit will be penalized. The opposite (small profits and one large loss) will also be penalized. The latter is worst for those who trade.

Remember that trading is a long-term race and always focus on following your system! Not on the money you will make at the end of the month as it can have a large variability.

```
//--- standard deviation of executed trades based on results in money
double dp_por_negocio(uint primeiro_negocio, uint ultimo_negocio,
                      double media_dos_resultados, double quantidade_negocios)
  {
   ulong ticket=0;
   double dp=0.0;
   for(uint i=primeiro_negocio; i < ultimo_negocio; i++)
     {
      //--- try to get deals ticket
      if((ticket=HistoryDealGetTicket(i))>0)
        {
         //--- get deals properties
         double profit=HistoryDealGetDouble(ticket,DEAL_PROFIT);
         //--- create price object
         if(profit!=0)
           {
            dp += MathPow(profit - media_dos_resultados, 2.0);
           }
        }
     }
   return MathSqrt(dp / quantidade_negocios);
  }

//--- Calculation of System Quality Number, SQN, by Van Tharp
double sqn(uint primeiro_negocio, uint ultimo_negocio,
           double lucro_acumulado, double quantidade_negocios)
  {
   double lucro_medio = lucro_acumulado / quantidade_negocios;
   double dp = dp_por_negocio(primeiro_negocio, ultimo_negocio,
                              lucro_medio, quantidade_negocios);
   if(dp == 0.0)
     {
      // Because the standard deviation returned a value of zero, which we didn't expect
      // we change it to average_benefit, since there is no deviation, which
      // brings the system closer to result 1.
      dp = lucro_medio;
     }
//--- The number of trades here will be limited to 100, so that the result will not be
//--- maximized due to the large number of trades.
   double res = (lucro_medio / dp) * MathSqrt(MathMin(100, quantidade_negocios));
   return NormalizeDouble(res, 2);
  }

//--- returns if a new month is found
bool eh_um_novo_mes(datetime timestamp, int &mes_anterior)
  {
   MqlDateTime mdt;
   TimeToStruct(timestamp, mdt);
   if(mes_anterior < 0)
     {
      mes_anterior=mdt.mon;
     }
   if(mes_anterior != mdt.mon)
     {
      mes_anterior = mdt.mon;
      return true;
     }
   return false;
  }

//--- Monthly SQN
double sqn_mes(void)
  {
   double sqn_acumulado = 0.0;
   double lucro_acumulado = 0.0;
   double quantidade_negocios = 0.0;
   int sqn_n = 0;
   int mes = -1;
   uint primeiro_negocio = 0;
   uint total_negocios;
//--- request the history of trades
   if(HistorySelect(0,TimeCurrent()) == false)
      return 0.0;
   total_negocios = HistoryDealsTotal();
//--- the average for each month is calculated for each trade
   for(uint i=primeiro_negocio; i < total_negocios; i++)
     {
      ulong    ticket=0;
      //--- Select the required ticket to pick up data
      if((ticket=HistoryDealGetTicket(i))>0)
        {
         datetime time = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
         double   lucro = HistoryDealGetDouble(ticket,DEAL_PROFIT);
         if(lucro == 0)
           {
            //--- If there is no result, move on to the next trade.
            continue;
           }
         if(eh_um_novo_mes(time, mes))
           {
            //--- If we have trades, then we calculate sqn, otherwise it will be equal to zero...
            if(quantidade_negocios>0)
              {
               sqn_acumulado += sqn(primeiro_negocio, i, lucro_acumulado,
                                    quantidade_negocios);
              }
            //--- The calculated amount sqns is always updated!
            sqn_n++;
            primeiro_negocio=i;
            lucro_acumulado = 0.0;
            quantidade_negocios = 0;
           }
         lucro_acumulado += lucro;
         quantidade_negocios++;
        }
     }
//--- when exiting "for", we can have undesired result
   if(quantidade_negocios>0)
     {
      sqn_acumulado += sqn(primeiro_negocio, total_negocios,
                           lucro_acumulado, quantidade_negocios);
      sqn_n++;
     }
//--- take the simple average of sqns
   return NormalizeDouble(sqn_acumulado / sqn_n, 2);
  }
```

Let's go through the code:

The first function calculates the standard deviation of a set of trades. Here we follow the recommendation of Van Tharp and include all trades in the standard deviation calculation. However, in the final formula (in the function below) we limit the number of trades to 100. This is done so that the result is not distorted due to the number of trades, which makes it more practical and meaningful.

Finally we have the _sqn\_mes_ function, which checks if it is a new month and accumulates some data needed for the above functions. At the end of this function, the average monthly SQN is calculated for the period in which the simulation was run. This brief explanation is intended to provide an overview of the code and the purpose of each function. By following this approach, you can better understand the SQN calculation.

The OnTester function can print all three values and can be queried in the tester tab or saved to a file, or we can even return one value multiplied by the other so that it appears in the report, as can be seen below.

```
double SQN_TESTER_ON_TESTER()
  {
   PrintFormat("%G,%G,%G", rr_medio(), CPCIndex(), sqn_mes());
   return NormalizeDouble(sqn_mes() * CPCIndex(), 5);
  }
```

### Before we finish

Before concluding this article, let's return to the topic with _include_ to see how to avoid duplication errors. Let's assume that we have an Expert advisor code with the _OnTester_ function and we want to put the _include_ of the specified file. It will look something like below (ignore the _OnTester_ content in this example).

```
//+------------------------------------------------------------------+
double OnTester()
  {
   return __LINE__;
  }
//+------------------------------------------------------------------+
#include "ARTICLE_METRICS.mq5"
```

This code will result in a function duplication error, since both the EA in our code and the include file have a function with the same name _OnTester_. However, we can use two definitions to rename one of them and simulate a mechanism for enabling or disabling which function should be used. See the example below.

```
//+------------------------------------------------------------------+
#define OnTester disable
//#define SQN_TESTER_ON_TESTER disable
double OnTester()
  {
   return __LINE__;
  }
#undef OnTester
//+------------------------------------------------------------------+
#include "ARTICLE_METRICS.mq5"
```

In this new format, we will not have a duplicate function error since the definition will change the name of the function in the EA code from _OnTester_ to _disable_. Now if we comment out the first definition and uncomment the second, the function inside the ARTICLE\_METRICS file will be renamed to _disable_, and the function in the Expert Advisor file will still be called _OnTester_.

This approach seems to be a fairly simple way to switch between both functions without having to comment out several lines of code. Even though it is a bit more invasive, I believe it can be considered by the user. Another thing for the user to consider is whether there is a need to keep the function within the EA, given that there is already one in the included file, which could become confusing.

### Conclusion

We have reached the end of this article, in which we presented a model of an Expert Advisor that operates randomly. We used it as an example in demonstrating quality factor calculations. We considered two possible calculations: Van Tharp and Sunny Harris. In addition, an introductory factor using the risk-return relationship was presented. We also demonstrated how the use of 'includes' can make it easier to switch between different available functions.

If you have any questions or found an error, please comment on the article. The discussed codes for both the Expert Advisor and the metrics file are in the attached zip for your study.

Do you use another quality metric? Share by commenting here! Thank you very much for reading this article.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11373](https://www.mql5.com/pt/articles/11373)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11373.zip "Download all attachments in the single ZIP archive")

[ARTICLE.zip](https://www.mql5.com/en/articles/download/11373/article.zip "Download ARTICLE.zip")(4.73 KB)

[ARTICLE\_METRICS.mq5](https://www.mql5.com/en/articles/download/11373/article_metrics.mq5 "Download ARTICLE_METRICS.mq5")(10.33 KB)

[ARTICLE\_MT5.mq5](https://www.mql5.com/en/articles/download/11373/article_mt5.mq5 "Download ARTICLE_MT5.mq5")(11.87 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/457841)**
(1)


![Feng Chen](https://c.mql5.com/avatar/2023/10/651980c7-c6df.jpg)

**[Feng Chen](https://www.mql5.com/en/users/fengbank)**
\|
10 Jan 2024 at 13:18

Great, just what I needed for this tutorial!


![Design Patterns in software development and MQL5 (Part 2): Structural Patterns](https://c.mql5.com/2/61/Design_Patterns_2Part_2i_Structural_Patterns_Logo.png)[Design Patterns in software development and MQL5 (Part 2): Structural Patterns](https://www.mql5.com/en/articles/13724)

In this article, we will continue our articles about Design Patterns after learning how much this topic is more important for us as developers to develop extendable, reliable applications not only by the MQL5 programming language but others as well. We will learn about another type of Design Patterns which is the structural one to learn how to design systems by using what we have as classes to form larger structures.

![Combinatorially Symmetric Cross Validation In MQL5](https://c.mql5.com/2/60/aticleicon.png)[Combinatorially Symmetric Cross Validation In MQL5](https://www.mql5.com/en/articles/13743)

In this article we present the implementation of Combinatorially Symmetric Cross Validation in pure MQL5, to measure the degree to which a overfitting may occure after optimizing a strategy using the slow complete algorithm of the Strategy Tester.

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://c.mql5.com/2/60/rj-article-image-60x60.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://www.mql5.com/en/articles/13770)

The Multi-Currency Expert Advisor in this article is Expert Advisor or trading robot that can trade (open orders, close orders and manage orders for example: Trailing Stop Loss and Trailing Profit) for more than one symbol pair only from one symbol chart. This time we will use only 1 indicator, namely Triangular moving average in multi-timeframes or single timeframe.

![Developing a Replay System — Market simulation (Part 13): Birth of the SIMULATOR (III)](https://c.mql5.com/2/54/replay-p13-avatar.png)[Developing a Replay System — Market simulation (Part 13): Birth of the SIMULATOR (III)](https://www.mql5.com/en/articles/11034)

Here we will simplify a few elements related to the work in the next article. I'll also explain how you can visualize what the simulator generates in terms of randomness.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/11373&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068305754613544919)

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