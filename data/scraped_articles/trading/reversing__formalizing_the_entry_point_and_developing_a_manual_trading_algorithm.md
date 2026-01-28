---
title: Reversing: Formalizing the entry point and developing a manual trading algorithm
url: https://www.mql5.com/en/articles/5268
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 5
scraped_at: 2026-01-23T17:32:43.280361
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=axblfznhmkqvvlkphqcjxfzvfrrlcpkh&ssn=1769178761836393114&ssn_dr=1&ssn_sr=0&fv_date=1769178761&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5268&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Reversing%3A%20Formalizing%20the%20entry%20point%20and%20developing%20a%20manual%20trading%20algorithm%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917876202287698&fz_uniq=5068361026547677321&sv=2552)

MetaTrader 5 / Trading


### Introduction

For already two articles ( [Reversing: The holy grail or a dangerous delusion?](https://www.mql5.com/en/articles/5008) and [Reversing: Reducing maximum drawdown and testing other markets](https://www.mql5.com/en/articles/5111)), we have been studying the Reversing trading strategy. We considered the use of the trading strategy in different markets, found the most suitable markets and formalized the basic rules for proper reversing. The subject seems to be fully discussed. What else can be written about the reversing technique? However, there is a problem which we mentioned earlier, but the solution to which we never approached.

The problem is connected with the entry point, which is not formalized in our strategy. It means a deal entry can be performed at any moment. The result of this can be unpredictable. Someone may have a deal closed by Take Profit. Another trader may enter 5 minutes later and get the entire chain of deals closed by Stop Loss.

That is why, all tests performed in previous articles can be considered reliable only for the cases, when you decide to enter exactly on the same historical day, hour and minute, as was done in the Strategy Tester. The same profit growth cannot be guaranteed, if you enter a trade a minute earlier or later.

So, we need some rules to determine "when to enter a deal" and "the entry direction". Such rules should not worsen our profit charts much. Let us try to find such rules.

### Tested symbols

Testing will be performed on the symbols, which showed the best performance in the [previous article](https://www.mql5.com/en/articles/5111). Also, the symbols need to have a sufficient amount of history, to avoid randomness of obtained results.

As in the previous article, we will test the strategy with several brokers. Symbols from different markets will be used for testing, except for Forex pairs. This is because we could not achieve better or comparatively similar results using standard indicators. So, we found out in [the first article](https://www.mql5.com/en/articles/5008) that none of the tested indicators could produce results similar to a random-time entry in the Forex market.

In this article, we will test the following securities:

**Broker 1, Stock Market:** TripAdvisor, Sberbank, Nintendo\_US, Tencent, Michael\_Kors, Starbucks, Gazprom, Petrobras, Snap, SQM.

**Broker 2, Stock Market**: ORCL.m, INTC.m, FOXA.m, SBUX.m, NKE.m, HPE.m, MSFT.m, KO.m, ATVI.m.

**Broker 1, indices**: YM.

**Broker 2, commodity:** BRENT.

### What's new in RevertEA

The following changes have been implemented in RevertEA compared to the previous article:

- the new _Close_ button allows closing any open position;
- added possibility to display current position profit on the chart (next to the _Close_ button);
- added possibility to set stop loss as a percentage of the current price;
- added parameters _Use constant trailing after N profit points_ and _Constant trading in points_;
- added possibility to use the ORDER\_FILLING\_IOC filling mode.


### What's new in RevertManualEA

The following changes have been made in RevertManualEA:

- the new _Close all_ button allows closing all open positions managed by the EA;
- the total profit of all positions managed by the EA is displayed near the _Close all_ button;
- the buttons of all positions managed by the EA are displayed in the chart;
- added possibility to use the ORDER\_FILLING\_IOC filling mode.

**Buttons for positions managed by the EA**. A click on a symbol button opens the chart of the corresponding symbol. Also, these buttons display related information, such as the symbol name, the current step, as well as the current position profit. If the button color has changed, it means the position has been fully closed (either by a take profit or upon reaching the maximum number of the chain steps).

Also pay attention to the new parameter _Show the sum of last losses by a symbol button click_. It is set to _true_ by default. This means that the amount of last losses will be displayed in a comment to the chart, which opens at a click on the appropriate symbol button. This enables the fast access to data about the minimum profit required to cover all losses in the chain.

### Formalizing the entry point

In order to eliminate the dependence of trading system results on the entry time, we need to find some rules, which would allow to enter under certain conditions, instead of entering whenever there are no open positions for that symbol.

In my opinion, the Moving Average is the most suitable tool for determining such an entry point.

First, it allows the determining of a global trend and therefore the entry direction: if the moving average grows, enter a Long position, otherwise go Short.

Second, an additional rule allows limiting the entry. For example, after the previous bar crossed the Moving Average upwards or downwards.

These rules will be used as entry conditions. Possibilities for the use of such rules were implemented in the RevertEA Expert Advisor long ago. The only thing we need to do to enable the use of the Moving Average is to change the following parameters:

- _Open short positions_ — set to _true_, otherwise the EA will only open Long positions, while ignoring Short signals;
- _Period_ — set the desired period in this parameter under the _Indicator MA #1_ block;
- Configure all other settings under the _Indicator MA #1_ block as you need.

An archive attached below contains all SET files with proper settings for specific symbols.

in this article, we will use a simple Moving Average. If you prefer any other MA type, you may select the desired one in the appropriate parameter under the _Indicator MA #1_ block in the EA settings.

**Stock market**. In the previous article, we found out that the Stock market is the most suitable one for the reversing strategy. Let us start with this market.

First, we will compare results with various symbols for Broker 1. This broker offers individual swaps for each symbol (you may check the swap from a table published in the previous article), but in any case, the swaps for Long positions are at least twice as large as those offered by Broker 2. However, the broker also has advantages: swap is positive for Short positions, i.e. swap is paid to you for holding short positions.

Testing results for all markets will be presented in appropriate tables. Each table row first presents information on testing results without the use of any indicators. I.e. a position is opened immediately after the previous chain is closed by Take Profit or Stop Loss. Then, a new line in the same table row presents testing results for the same symbol with the same SL and TP values, but with the use of the _Simple Moving Average_ indicator.

Since RevertEA also supports trading without reversing, we will also check the profitability of classical trading, using the Moving Average. The best results of these tests will be presented in the third line of each table row.

Trading strategy testing results with reversing, without indicators, differ from results published in the previous article. This is because more than a month has passed since its publication. So, we can check what has changed since that time.

Before proceeding to testing results, let us recall the purpose of some table columns:

- _Annual %_ — the value is calculated according to the following formula: _((profit/max. drawdown)\*100)/number of years for which the historic symbol data are available_, it is an approximate number and is also provided for the purpose of comparative analysis;
- _Max. losses_ — the maximum number of consecutive losses, i.e. how deep into the chain we had to go before reaching Take Profit;
- _Trades (year/total)_ — the average number of positions which will be opened for that symbol per year (the total number of trades is divided by the number of years, for which historical data is available).


Broker 1, Stock Market:

| Symbol | Trades (year/total) | Profit factor | Max. drawdown | Profit Column | Annual % | Max. losses | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TripAdvisor, revert<br> TripAdvisor, revert + MA<br> TripAdvisor, no revert | 98 / 246<br> 46 / 116<br> 65 / 164 | 1.36<br> 1.6<br> 1.69 | 98<br> 53<br> 6 | 231<br> 156<br> 49 | 94 %<br> 117 %<br> 326% | 6<br> 4<br> 9 | 155<br> 155<br> 60 | 195<br> 195<br> 155 |
| Sberbank, revert<br> Sberbank, revert + MA<br> Sberbank, no revert | 150 / 225<br> 118 / 178<br> 231 / 347 | 1.34<br> 1.31<br> 0.79 | 45<br> 17<br> 43 | 86<br> 49<br> -37 | 127 %<br> 192 %<br> - | 5<br> 5<br> 9 | 420<br> 420<br> 155 | 510<br> 510<br> 360 |
| Nintendo\_US, revert<br> Nintendo\_US, revert + MA<br> Nintendo\_US, no revert | 169 / 339<br> 32 / 64<br> 25 / 51 | 1.49<br> 1.72<br> 1.12 | 18<br> 16<br> 7 | 104<br> 59<br> 4 | 288 %<br> 184 %<br> 28 % | 6<br> 4<br> 5 | 55<br> 55<br> 35 | 80<br> 80<br> 55 |
| Tencent, revert<br> Tencent, revert + MA<br> Tencent, no revert | 74 / 223<br> 26 / 80<br> 18 / 54 | 2.54<br> 2.48<br> 1.47 | 43<br> 69<br> 6 | 527<br> 381<br> 20 | 408 %<br> 184 %<br> 111 % | 5<br> 5<br> 3 | 450<br> 450<br> 530 | 1500<br> 1500<br> 880 |
| Michael\_Kors, revert<br> Michael\_Kors, revert + MA<br> Michael\_Kors, no revert | 36 / 109<br> 23 / 70<br> 15 / 46 | 1.51<br> 1.57<br> 1 | 134<br> 71<br> 15 | 240<br> 140<br> 0 | 59 %<br> 65 %<br> - | 5<br> 4<br> 4 | 190<br> 190<br> 200 | 330<br> 330<br> 400 |
| Starbucks, revert<br> Starbucks, revert + MA<br> Starbucks, no revert | 15 / 231<br> 11 / 171<br> 20 / 300 | 1.69<br> 1.64<br> 1.07 | 38<br> 36<br> 13 | 251<br> 174<br> 11 | 45 %<br> 33 %<br> 5 % | 4<br> 4<br> 6 | 160<br> 160<br> 95 | 195<br> 195<br> 105 |
| Gazprom, revert<br> Gazprom, revert + MA<br> Gazprom, no revert | 265 / 398<br> 101 / 152<br> 36 / 54 | 1.33<br> 1.42<br> 1.03 | 59<br> 18<br> 20 | 142<br> 55<br> 2 | 160 %<br> 203 %<br> 6 % | 6<br> 4<br> 5 | 150<br> 150<br> 240 | 180<br> 180<br> 480 |
| Petrobras, revert<br> Petrobras, revert + MA<br> Petrobras, no revert | 23 / 337<br> 15 / 219<br> 11 / 162 | 1.61<br> 1.64<br> 1.16 | 86<br> 160<br> 28 | 623<br> 388<br> 39 | 49 %<br> 16 %<br> 9 % | 5<br> 5<br> 7 | 240<br> 240<br> 240 | 300<br> 300<br> 410 |
| Snap, revert<br> Snap, revert + MA<br> Snap, no revert | 62 / 93<br> 24 / 36<br> 24 / 37 | 1.97<br> 3.47<br> 1.49 | 44<br> 12<br> 7 | 227<br> 95<br> 12 | 343 %<br> 527 %<br> 114 % | 4<br> 2<br> 6 | 75<br> 75<br> 45 | 135<br> 135<br> 120 |
| SQM, revert<br> SQM, revert + MA<br> SQM, no revert | 64 / 162<br> 29 / 74<br> 22 / 57 | 1.81<br> 1.89<br> 1.02 | 55<br> 68<br> 12 | 288<br> 142<br> 1 | 209 %<br> 83 %<br> 3 % | 5<br> 5<br> 5 | 125<br> 125<br> 150 | 240<br> 240<br> 185 |

Let's not draw conclusions now. First, view the corresponding charts. Conclusions will be made at the end of this section for all markets.

As for the charts, the following abbreviations are used on them:

- PLAIN — the chart of a trading strategy without any indicator (a Long entry is performed immediately, once there are no positions for the symbol);
- MA — entry based on Moving Average signals;
- NOREVERT — entry based on Moving Average signals without using the reversing technique.

**TripAdvisor**:

![TripAdvisor](https://c.mql5.com/2/34/TripAdvisor.png)

**Sberbank**:

![Sberbank](https://c.mql5.com/2/34/Sberbank.png)

**Nintendo\_US**:

![Nintendo_US](https://c.mql5.com/2/34/Nintendo_US.png)

**Tencent**:

![Tencent](https://c.mql5.com/2/34/Tencent.png)

**Michael\_Kors**:

![Michael_Kors](https://c.mql5.com/2/34/Michael_Kors.png)

**Starbucks**:

![Starbucks](https://c.mql5.com/2/34/Starbucks.png)

**Gazprom**:

![Gazprom](https://c.mql5.com/2/34/Gazprom.png)

**Petrobras**:

![Petrobras](https://c.mql5.com/2/34/Petrobras.png)

**Snap**:

![Snap](https://c.mql5.com/2/34/Snap.png)

**SQM**:

![SQM](https://c.mql5.com/2/34/SQM.png)

Now let us view results for Broker 2. This broker provides negative swap regardless of the position direction. Another difference from Broker 1: Broker 2 does not pay dividends for your Long positions. However, the advantage is that this broker does not charge dividends for Short positions.

Broker 2, Stock Market:

| Symbol | Trades (year/total) | Profit factor | Max. drawdown | Profit Column | Annual % | Max. losses | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ORCL.m, revert<br> ORCL.m, revert + MA<br> ORCL.m, no revert | 44 / 246<br> 26 / 147<br> 13 / 76 | 1.07<br> 1.64<br> 1.51 | 275<br> 65<br> 9 | 46<br> 281<br> 24 | 3 %<br> 78 %<br> 48 % | 8<br> 6<br> 7 | 90<br> 90<br> 100 | 150<br> 150<br> 235 |
| INTC.m, revert<br> INTC.m, revert + MA<br> INTC.m, no revert | 33 / 182<br> 19 / 109<br> 29 / 162 | 1.92<br> 1.94<br> 1.31 | 23<br> 24<br> 6 | 219<br> 116<br> 22 | 173 %<br> 87 %<br> 66 % | 4<br> 4<br> 7 | 130<br> 130<br> 70 | 180<br> 180<br> 145 |
| FOXA.m, revert<br> FOXA.m, revert + MA<br> FOXA.m, no revert | 53 / 268<br> 41 / 207<br> 28 / 144 | 1.6<br> 1.66<br> 1.07 | 47<br> 28<br> 8 | 201<br> 150<br> 6 | 85 %<br> 107 %<br> 15 % | 5<br> 4<br> 5 | 90<br> 90<br> 115 | 120<br> 120<br> 135 |
| SBUX.m, revert<br> SBUX.m, revert + MA<br> SBUX.m, no revert | 49 / 270<br> 28 / 154<br> 31 / 173 | 1.5<br> 1.86<br> 1.11 | 58<br> 22<br> 11 | 217<br> 143<br> 9 | 68 %<br> 118 %<br> 13 % | 5<br> 4<br> 8 | 130<br> 130<br> 70 | 160<br> 160<br> 150 |
| NKE.m, revert<br> NKE.m, revert + MA<br> NKE.m, no revert | 35 / 197<br> 22 / 123<br> 26 / 146 | 2.26<br> 2.32<br> 1.29 | 140<br> 75<br> 11 | 947<br> 422<br> 28 | 122 %<br> 102 %<br> 46 % | 6<br> 5<br> 8 | 135<br> 135<br> 105 | 275<br> 275<br> 205 |
| HPE.m, revert<br> HPE.m, revert + MA<br> HPE.m, no revert | 33 / 99<br> 16 / 48<br> 12 / 38 | 1.89<br> 3.55<br> 2.05 | 27<br> 8<br> 7 | 104<br> 51<br> 19 | 128 %<br> 212 %<br> 90 % | 4<br> 2<br> 5 | 55<br> 55<br> 55 | 70<br> 70<br> 85 |
| MSFT.m, revert<br> MSFT.m, revert + MA<br> MSFT.m, no revert | 36 / 199<br> 28 / 158<br> 37 / 206 | 2.11<br> 2.06<br> 1.23 | 70<br> 73<br> 12 | 508<br> 478<br> 31 | 131 %<br> 119 %<br> 46 % | 7<br> 5<br> 9 | 150<br> 150<br> 100 | 275<br> 275<br> 205 |
| KO.m, revert<br> KO.m, revert + MA<br> KO.m, no revert | 20 / 110<br> 16 / 93<br> 14 / 82 | 2.02<br> 1.77<br> 1.75 | 29<br> 51<br> 7 | 144<br> 147<br> 31 | 90 %<br> 52 %<br> 80 % | 4<br> 5<br> 3 | 100<br> 100<br> 120 | 160<br> 160<br> 150 |
| ATVI.m, revert<br> ATVI.m, revert + MA<br> ATVI.m, no revert | 77 / 425<br> 24 / 135<br> 19 / 109 | 1.34<br> 1.99<br> 1.52 | 54<br> 15<br> 7 | 235<br> 109<br> 33 | 79 %<br> 132 %<br> 85 % | 5<br> 3<br> 4 | 135<br> 135<br> 115 | 140<br> 140<br> 155 |

Of all the analyzed symbols, only ORCL.m shows disappointing results during the past month. It managed to reach the maximum 8th chain step. In that case, the entire chain was closed with a loss. That was a strategy without indicators. Due to this, annual percentage of profit was very small. The loss could be avoided if we traded using the indicator.

**ORCL.m**:

![ORCL.m](https://c.mql5.com/2/34/ORCL.png)

**INTC.m**:

![INTC.m](https://c.mql5.com/2/34/INTC.png)

**FOXA.m**:

![FOXA.m](https://c.mql5.com/2/34/FOXA.png)

**SBUX.m**:

![SBUX.m](https://c.mql5.com/2/34/SBUX.png)

**NKE.m**:

![NKE.m](https://c.mql5.com/2/34/NKE.png)

**HPE.m**:

![HPE.m](https://c.mql5.com/2/34/HPE.png)

**MSFT.m**:

![MSFT.m](https://c.mql5.com/2/34/MSFT.png)

**KO.m**:

![KO.m](https://c.mql5.com/2/34/KO.png)

**ATVI.m**:

![ATVI.m](https://c.mql5.com/2/34/ATVI.png)

**Indices**. We will consider only one index — Dow Jones (YM) with Broker 1. Its profit chart could be called perfect... To be more precise, the profit chart from the previous article was perfect. But in new tests, the profitable chain was unexpectedly interrupted in 2009. But the addition of a Moving Average helped to avoid the loss of the entire chain, as shown in the below chart.

Broker 1, indices

| Symbol | Trades (year/total) | Profit factor | Max. drawdown | Profit Column | Annual % | Max. losses | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YM, revert<br> YM, revert + MA<br> YM, no revert | 106 / 1 287<br> 76 / 958<br> 54 / 678 | 1.46<br> 1.51<br> 0.9 | 2 797<br> 1 343<br> 820 | 20 089<br> 13 216<br> -636 | 57 %<br> 78 %<br> - | 8<br> 6<br> 11 | 155<br> 155<br> 170 | 210<br> 210<br> 200 |

![YM](https://c.mql5.com/2/34/ym.png)

**Commodities**. Here we also consider only one security — Brent oil with Broker 2. Only this symbol showed good profit and balance chart in the previous article. Although the use of the Moving Average helped in reducing the maximum number of consecutive losses, the total results became much worse.

Broker 2, commodity:

| Symbol | Trades (year/total) | Profit factor | Max. drawdown | Profit Column | Annual % | Max. losses | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Brent, revert<br> Brent, revert + MA<br> Brent, no revert | 146 / 733<br> 137 / 685<br> 136 / 682 | 1.5<br> 1.26<br> 1.1 | 8 721<br> 8 962<br> 154 | 31 144<br> 18 624<br> 338 | 71 %<br> 41 %<br> 43 % | 21<br> 19<br> 17 | 70<br> 70<br> 60 | 275<br> 275<br> 215 |

![Brent](https://c.mql5.com/2/34/BRENT.png)

### Let's sum up.

So, what do we have? We have succeeded in formalizing the entry point. So now, the testing start date does not affect testing results so much.

In addition, this increased the profitability for most of the securities, as well as reduced the maximum number of consecutive losses by 1-2 steps.

However, the net profit of all symbols decreased due to a lower number of executed trades (the total number reduced by 1.5-2 times). Here it is up to you to decide: you may choose either larger profit or lower risks.

### Testing Stop Loss percentage of the price and ATR

In comments to previous articles, users mentioned that it was not right to use fixed Stop Loss and Take Profit values, and it would be better to use percentage of ATR. There is some logic in this statement. Indeed, we used the same Stop Loss and Take Profit level for 15 years of testing. But a financial symbol may have different volatility in different time periods. Moreover, symbol price can change greatly within 15 years, so that a 15 year old Stop Loss may no longer be suitable.

Therefore, RevertEA has been modified to enable Stop Loss specification in points, as percentage of the current price or of ATR. The desired type can be selected using the _Stop Loss type_ parameter in the Expert Advisor settings.

ATR in this case is determined according to Gerchik's method. That is, ATR will be calculated on a daily chart, without taking into account too small and too large bars. If the number of bars is not enough for the calculation, the standard method will be used.

If this ATR determining function does not suit you, you can easily re-write the _getATR_ function for your own needs. This function is available in the _strategy\_base.mqh_ file of the _Strategies_ folder. Its original source code is presented below:

```
double getATR(string name){
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   double atr=0;
   double pre_atr=0;
   int count=0;

   // Get bar data for the last 7 days
   //(this is more than needed, because we will calculate ATR for 5 days)
   int copied=CopyRates(name, PERIOD_D1, 0, 7, rates);

   // Determine ATR for the last 7 days, excluding
   // too large and too small bars
   if(copied>=5){
      for(int j=1; j<copied; j++){
         pre_atr+=rates[j].high-rates[j].low;
      }
      pre_atr/=(copied-1);
   }

   // Determine ATR for the last 5 days or less, including
   // too large and too small bars
   // i.e. the bars having the size over 1.5 7-day ATR
   // and bars less than 0.3 of 7-day ATR
   // are discarded
   if(pre_atr>0){
      for(int j=1; j<copied; j++){
         if( rates[j].high-rates[j].low > pre_atr*1.5 ) continue;
         if( rates[j].high-rates[j].low < pre_atr*0.3 ) continue;

         if( ++count > 5 ){
            count=5;
            break;
         }

         atr+=rates[j].high-rates[j].low;
      }
      // if there are not enough medium bars
      // calculate a normal ATR 5
      if(count<2){
         count=0;
         for(int j=1; j<=5; j++){
            ++count;
            atr+=rates[j].high-rates[j].low;
         }
      }
      atr= NormalizeDouble(atr/count, _Digits);
   }

   return atr;
}
```

As for the Take Profit level, it can now be specified not only in points, but also as a multiple of Stop Loss (the _Take type_ parameter in EA settings. For example, you can set Take Profit at a distance of three Stop Losses from the open price.

Comparative results of testing with different Stop Loss levels is provided below. Testing was performed without the use of a Moving Average, because this way we get more entry points and have more accurate results for comparative analysis.

Each table row consists of three lines:

- Stop Loss and Take Profit in points;
- Stop Loss as percentage of the price, Take Profit as SL multiplier;
- Stop Loss as percentage of ATR, Take Profit as SL multiplier.

As for Snap, the broker stopped this symbol trading by the time I started testing ATR-based Stop Loss. It was only possible to closed deals. So I was not able to perform Snap testing.

| Symbol | Trades (year/total) | Profit factor | Max. drawdown | Profit Column | Annual % | Max. losses | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TripAdvisor<br> TripAdvisor, percent<br> TripAdvisor, atr | 98 / 246<br> 532 / 1 331<br> 56 / 141 | 1.36<br> 1.41<br> 1.42 | 98<br> 310<br> 386 | 231<br> 1 161<br> 366 | 94 %<br> 149 %<br> 37 % | 6<br> 11<br> 8 | 155<br> 1<br> 96 | 195<br> 1.7<br> 2.7 |
| Sberbank<br> Sberbank, percent<br> Sberbank, atr | 150 / 225<br> 178 / 267<br> 196 / 295 | 1.34<br> 1.63<br> 1.78 | 45<br> 223<br> 67 | 86<br> 713<br> 257 | 127 %<br> 213 %<br> 255 % | 5<br> 11<br> 6 | 420<br> 1.2<br> 73 | 510<br> 2.6<br> 1.7 |
| Nintendo\_US<br> Nintendo\_US, percent<br> Nintendo\_US, atr | 169 / 339<br> 182 / 365<br> 144 / 288 | 1.49<br> 1.39<br> 1.65 | 18<br> 31<br> 35 | 104<br> 100<br> 182 | 288 %<br> 161 %<br> 260 % | 6<br> 5<br> 9 | 55<br> 1.2<br> 82 | 80<br> 1.4<br> 4.1 |
| Tencent<br> Tencent, percent<br> Tencent, atr | 74 / 223<br> 25 / 76<br> 213 / 640 | 2.54<br> 2.18<br> 1.38 | 43<br> 24<br> 42 | 527<br> 168<br> 177 | 408 %<br> 233 %<br> 140 % | 5<br> 4<br> 6 | 450<br> 3.6<br> 76 | 1500<br> 2<br> 1.4 |
| Michael\_Kors<br> Michael\_Kors, percent<br> Michael\_Kors, atr | 36 / 109<br> 46 / 140<br> 84 / 252 | 1.51<br> 2.6<br> 2.18 | 134<br> 281<br> 266 | 240<br> 1 452<br> 1 352 | 59 %<br> 172 %<br> 169 % | 5<br> 7<br> 7 | 190<br> 3<br> 77 | 330<br> 2<br> 2.3 |
| Starbucks<br> Starbucks, percent<br> Starbucks, atr | 15 / 231<br> 26 / 388<br> 56 / 816 | 1.69<br> 1.26<br> 1.2 | 38<br> 743<br> 426 | 251<br> 531<br> 740 | 45 %<br> 4 %<br> 11 % | 4<br> 30<br> 17 | 160<br> 2.8<br> 75 | 195<br> 3.5<br> 4.1 |
| Gazprom<br> Gazprom, percent<br> Gazprom, atr | 265 / 398<br> 64 / 97<br> 76 / 114 | 1.33<br> 1.99<br> 3.02 | 59<br> 102<br> 115 | 142<br> 294<br> 404 | 160 %<br> 192 %<br> 234 % | 6<br> 11<br> 9 | 150<br> 1.6<br> 69 | 180<br> 2.8<br> 3.7 |
| Petrobras<br> Petrobras, percent<br> Petrobras, atr | 23 / 337<br> 25 / 375<br> 55 / 803 | 1.61<br> 1.45<br> 1.92 | 86<br> 391<br> 650 | 623<br> 456<br> 3 743 | 49 %<br> 8 %<br> 39 % | 5<br> 12<br> 20 | 240<br> 1.6<br> 83 | 300<br> 2.8<br> 4 |
| Snap<br> Snap, percent<br> Snap, atr | 62 / 93<br> 55 / 83<br> - | 1.97<br> 2.85<br> - | 44<br> 83<br> - | 227<br> 411<br> - | 343 %<br> 330 %<br> - | 4<br> 5<br> - | 75<br> 4<br> - | 135<br> 2.7<br> - |
| SQM<br> SQM, percent<br> SQM, atr | 64 / 162<br> 58 / 145<br> 170 / 426 | 1.81<br> 1.79<br> 1.32 | 55<br> 128<br> 249 | 288<br> 337<br> 314 | 209 %<br> 105 %<br> 50 % | 5<br> 6<br> 9 | 125<br> 3.4<br> 67 | 240<br> 1.7<br> 1.9 |
| ORCL.m<br> ORCL.m, percent<br> ORCL.m, atr | 44 / 246<br> 16 / 88<br>  215 / 1 186 | 1.07<br> 1.64<br> 1.24 | 275<br> 65<br> 261 | 46<br> 146<br> 351 | 3 %<br> 40 %<br> 24 % | 8<br> 5<br> 8 | 90<br> 4<br> 70 | 150<br> 1.4<br> 1.4 |
| INTC.m<br> INTC.m, percent<br> INTC.m, atr | 33 / 182<br> 25 / 142<br> 116 / 641 | 1.92<br> 3.36<br> 1.62 | 23<br> 133<br> 385 | 219<br> 1 323<br> 1 358 | 173 %<br> 180 %<br> 64 % | 4<br> 6<br> 14 | 130<br> 2.4<br> 56 | 180<br> 3.2<br> 4.1 |
| FOXA.m<br> FOXA.m, percent<br> FOXA.m, atr | 53 / 268<br> 93 / 467<br> 234 / 1 171 | 1.6<br> 1.57<br> 1.28 | 47<br> 133<br> 165 | 201<br> 407<br> 305 | 85 %<br> 61 %<br> 36 % | 5<br> 7<br> 8 | 90<br> 2<br> 71 | 120<br> 1.4<br> 1.4 |
| SBUX.m<br> SBUX.m, percent<br> SBUX.m, atr | 49 / 270<br> 62 / 345<br> 154 / 849 | 1.5<br> 1.89<br> 1.5 | 58<br> 100<br> 269 | 217<br> 694<br> 1 379 | 68 %<br> 126 %<br> 93 % | 5<br> 7<br> 15 | 130<br> 1.8<br> 57 | 160<br> 1.7<br> 3.2 |
| NKE.m<br> NKE.m, percent<br> NKE.m, atr | 35 / 197<br> 20 / 112<br> 407 / 2 240 | 2.26<br> 2.18<br> 1.17 | 140<br> 41<br> 174 | 947<br> 257<br> 384 | 122 %<br> 113 %<br> 40 % | 6<br> 3<br> 12 | 135<br> 4<br> 47 | 275<br> 1.4<br> 1.7 |
| HPE.m<br> HPE.m, percent<br> HPE.m, atr | 33 / 99<br> 66 / 198<br> 60 / 180 | 1.89<br> 1.76<br> 2.62 | 27<br> 52<br> 110 | 104<br> 143<br> 674 | 128 %<br> 91 %<br> 204 % | 4<br> 5<br> 14 | 55<br> 2.8<br> 76 | 70<br> 1.4<br> 3.8 |
| MSFT.m<br> MSFT.m, percent<br> MSFT.m, atr | 36 / 199<br> 28 / 154<br> 126 / 693 | 2.11<br> 1.39<br> 1.42 | 70<br> 340<br> 434 | 508<br> 288<br> 1 042 | 131 %<br> 15 %<br> 43 % | 7<br> 6<br> 16 | 150<br> 3.2<br> 69 | 275<br> 1.4<br> 2.9 |
| KO.m<br> KO.m, percent<br> KO.m, atr | 20 / 110<br> 22 / 121<br> 125 / 688 | 2.02<br> 2.12<br> 1.35 | 29<br> 163<br> 244 | 144<br> 397<br> 2 272 | 90 %<br> 44 %<br> 169 % | 4<br> 6<br> 14 | 100<br> 2<br> 64 | 160<br> 2<br> 2.9 |
| ATVI.m<br> ATVI.m, percent<br> ATVI.m, atr | 77 / 425<br> 15 / 83<br> 194 / 1 072 | 1.34<br> 2.86<br> 1.46 | 54<br> 302<br> 235 | 235<br> 1 313<br> 1 198 | 79 %<br> 79 %<br> 92 % | 5<br> 6<br> 9 | 135<br> 4<br> 60 | 140<br> 3.2<br> 2.3 |
| YM<br> YM, percent<br> YM, atr | 106 / 1 287<br> 157 / 1 969<br> 234 / 2 927 | 1.46<br> 1.29<br> 1.02 | 2 797<br> 13 288<br> 19 984 | 20 089<br> 41 936<br> 2 669 | 57 %<br> 25 %<br> 1 % | 8<br> 27<br> 17 | 155<br> 0.6<br> 51 | 210<br> 2.9<br> 1.7 |
| Brent<br> Brent, percent<br> Brent, atr | 146 / 733<br> 114 / 574<br> 81 / 407 | 1.5<br> 1.21<br> 1.34 | 8 721<br> 13 883<br> 9 368 | 31 144<br> 12 988<br> 16 587 | 71 %<br> 18 %<br> 35 % | 21<br> 19<br> 13 | 70<br> 1.4<br> 70 | 275<br> 3.8<br> 3.2 |

None of the Stop Loss calculation methods was the best for all tested symbols. In some cases, Stop Loss in points was better. With other symbols price or ATR percentage Stop Loss showed greater profit. However, there is one regularity. The maximum number of consecutive losing trades significantly increases almost with all deals, if we calculate SL as percentage of price or as percentage of ATR. The maximum drawdown increases accordingly.

In general, it is impossible to say for sure which of the Stop Loss and Take Profit calculation methods is better. But in my opinion, Stop Loss and Take Profit in points shows better results, since this way we have a smaller drawdown.

### Testing the Trailing Stop

Many traders prefer to use a floating Trailing Stop, while they believe that it can protect profits and thus increase the Expert Advisor profitability. Let's check this option now.

The EA has two parameters:

- _Use constant trailing after N profit points_ — this parameter sets the number if points which the price should move in the Take Profit direction, after which the Trailing Stop mechanism will be activated (this limitation protects a chain from being closed with a loss on further reversing steps, especially if lot increase is not used at every step);
- _Constant trailing in points_ — number of points from the current price, by which position Stop Loss will be moved once the condition of _Use constant trailing after N profit points_ is met.

The EA determines the need to trail Stop Loss at each bar beginning. I.e. the check is performed every 15 minutes, because we test the strategy on the M15 timeframe.

If trailing is needed, then in addition to moving Stop Loss, the EA cancels Take Profit. Now, Stop Loss remains the only method for closing the position. That is, in addition to the protection from possible price direction change, the Trailing Stop allows you to increase the profit amount by canceling Take Profit.

Below is a comparative table of the best results without Trailing Stop and with Trailing Stop.

| Symbol | Trades (year/total) | Profit factor | Max. drawdown | Profit | Annual % | Max. losses |
| --- | --- | --- | --- | --- | --- | --- |
| TripAdvisor<br> TripAdvisor, trailing | 100 / 250<br> 99 / 248 | 1.5<br> 1.53 | 98<br> 100 | 315<br> 321 | 128 %<br> 128 % | 6<br> 6 |
| Sberbank<br> Sberbank, trailing | 157 / 236<br> 158 / 238 | 1.37<br> 1.55 | 45<br>44 | 98<br> 148 | 145 %<br> 224 % | 5<br> 5 |
| Nintendo\_US<br> Nintendo\_US, trailing | 171 / 342<br> 173 / 347 | 1.38<br> 1.2 | 18<br> 37 | 90<br> 51 | 250 %<br> 68 % | 6<br> 9 |
| Tencent<br> Tencent, trailing | 78 / 234<br> 69 / 209 | 1.9<br> 2.3 | 150<br> 82 | 415<br> 438 | 92 %<br> 178 % | 8<br> 7 |
| Michael\_Kors<br> Michael\_Kors, trailing | 38/ 114<br> 48 / 145 | 1.46<br> 1.46 | 134<br> 116 | 247<br> 169 | 61 %<br> 48 % | 5<br> 5 |
| Starbucks<br> Starbucks, trailing | 16 / 234<br> 15 / 227 | 1.66<br> 1.6 | 38<br> 51 | 247<br> 234 | 44 %<br> 31 % | 4<br> 4 |
| Gazprom<br> Gazprom, trailing | 288 / 433<br> 295 / 443 | 1.34<br> 1.4 | 59<br> 39 | 155<br> 162 | 175 %<br> 276 % | 6<br> 5 |
| Petrobras<br> Petrobras, trailing | 24 / 349<br> 23 / 344 | 1.4<br> 1.52 | 126<br> 102 | 547<br> 541 | 29 %<br> 36 % | 8<br> 5 |
| SQM<br> SQM, trailing | 64 / 164<br> 69 / 174 | 1.82<br> 1.8 | 55<br> 83 | 298<br> 400 | 216 %<br> 192 % | 5<br> 5 |
| ORCL.m<br> ORCL.m, trailing | 44 / 246<br> 46 / 254 | 1.07<br> 1.46 | 275<br> 225 | 46<br> 294 | 3 %<br> 23 % | 8<br> 7 |
| INTC.m<br> INTC.m, trailing | 31 / 172<br> 31 / 174 | 1.77<br> 1.7 | 52<br> 52 | 216<br> 218 | 75 %<br> 75 % | 5<br> 6 |
| FOXA.m<br> FOXA.m, trailing | 52 / 260<br> 68 / 342 | 0.91<br> 1.38 | 413<br> 62 | -57<br> 126 | -<br> 40 % | 8<br> 5 |
| SBUX.m<br> SBUX.m, trailing | 49 / 272<br> 49 / 269 | 1.5<br> 1.49 | 58<br> 63 | 218<br> 214 | 68 %<br> 67 % | 5<br> 5 |
| NKE.m<br> NKE.m, trailing | 35 / 198<br> 35 / 199 | 2.26<br> 2.01 | 140<br> 84 | 957<br> 407 | 122 %<br> 88 % | 6<br> 5 |
| HPE.m<br> HPE.m, trailing | 30 / 90<br> 41 / 125 | 1.4<br> 1.4 | 46<br> 21 | 75<br> 43 | 54 %<br> 68 % | 5<br> 3 |
| MSFT.m<br> MSFT.m, trailing | 37 / 207<br> 37 / 208 | 2.04<br> 1.12 | 70<br> 442 | 538<br> 112 | 139 %<br> 4 % | 7<br> 8 |
| KO.m<br> KO.m, trailing | 20 / 105<br> 20 / 109 | 1.56<br> 1.86 | 107<br> 44 | 133<br> 164 | 22 %<br> 67 % | 5<br> 5 |
| ATVI.m<br> ATVI.m, trailing | 79 / 435<br> 80 / 443 | 1.27<br> 1.26 | 56<br> 56 | 201<br> 199 | 65 %<br> 64 % | 5<br> 5 |
| YM<br> YM, trailing | 106 / 1 288<br> 106 / 1 290 | 1.3<br> 1.3 | 7 285<br> 6 989 | 15 090<br> 14 622 | 16 %<br> 16 % | 8<br> 8 |
| Brent<br> Brent, trailing | 151 / 759<br> 149 / 748 | 1.3<br> 1.44 | 8 721<br> 8 790 | 21 950<br> 28 537 | 50 %<br> 64 % | 21<br> 21 |

As you can see from the table, the Trailing Stop function increased profits in some cases and decreased them in others. So Trailing Stop cannot be considered for sure as a positive feature in this strategy. Moreover, if you check symbols with a large historical data period (Starbucks, Petrobras, YM), Trailing Stop reduces their profitability. So, an improvement in results of symbols without a large history can be random.

However, Trailing Stop can really slightly reduce the chances of having the entire chain closed by Stop Loss. See FOXA.m symbol results. We had an entire chain closed by Stop Loss without trailing stop. The trailing function helped us eliminate it.

It is strange that FOXA.m results in current tests are much worse than those from the previous table. From the Strategy Tester report on this symbol (attached below) we can see that Stop Loss on the entire chain was registered long time ago. The strange fact is that in all previous numerous tests we did not have this Stop Loss.

Unfortunately, the Strategy Tester does not perform identical tests. It may use 0%, 2%, 8$ or any other share of real data for the same financial instrument. That is why testing results also change.

I do not know why this happens (if you know any trick to have the same results in each testing run, please share it in comments). At least this is another reason why you should not fully trust testing results obtained with historical data.

### Signals based on the reversing technique

More than three months have passed since I launched [a signal in the Forex market](https://www.mql5.com/en/signals/465950), which applies the reversing trading strategy. It is not a very large time period, but quite enough for drawing intermediate conclusions.

The most important result is that it is still in profit. It makes 8% per month on the average, i.e. about 100% per year.

![Results of a signal using the reversing technique](https://c.mql5.com/2/34/signal.png)

We did not have losing months except for the first month. But the loss in the first month is quite understandable. First, the signal run only for a few days. Second, since we are using entries in one predetermined direction, it's natural that the first EA operation steps turned out to be unsuccessful.

As for negative sides, Gold managed to reach the 7th step in a chain. Remember, for 18-year testing Gold reached a maximum of 8th chain step. In my signal, it reached the seventh step for as little as three months. In that case I decided not to wait for Take Profit and closed the entire chain as soon as the profit exceeded losses of the entire chain. Some of the profit might be lost due to swaps, but they were fully covered by profits on other symbols.

As seen from the above image, the largest drawdown was obtained due to Stop Loss at the 6th and 7the step in Gold chains. Fortunately, profit on other symbols helped to reduce this drawdown.

As for other symbols, GBPUSD reached the 6th step. Other chains were closed by Take Profit at a maximum of 4th-5th chain step.

Only 5-10% of all entries were closed at the first step. This is an indication of our random entries =)

Continue to monitor the signal and evaluate the risks and profits of the strategy. I am not going to stop this signal.

### Writing a manual trading algorithm using levels

Automated trading has many advantages, such as the strict following of the trading strategy and free time, which you have, while the robot is trading for you. However, a successful manual trading can show much higher profitability. In addition, there are other disadvantages:

- a huge loss if the entire chain closes by Stop Loss;
- large free margin requirement for the last steps in a chain, if you use lot doubling at each step (other lot calculation methods are not suitable for the used SL to TP ratio).

If in the first step the maintenance margin is equal to 1 dollar, in the seventh step it would be equal to 64 dollars. It means you need to have 64 free dollars in your account to support a position for only one symbol. The resulting profit is relatively small compared to the required deposit.

Manual trading allows you to reduce the maximum required margin by applying lot doubling not at each step, but after 1, 2 or even 3 steps. You choose when to enter a trade, as well as which Stop Loss to Take Profit ratio to use.

Also, the use of lot doubling after one, two or three steps allows to increase the maximum acceptable number of steps in a chain, and therefore to increase chances of closing with profit.

That is why, in this section I want to offer one of the possible manual trading algorithms using the reversing technique. I cannot say for sure that this is the best version. But in addition to reversing, the strategy involves diversification, which may also reduce overall risks to some extent.

Moreover, diversification allows for a flexible risk management. For example, if your account balance increases by the end of the day or week, you can close all open positions, including losing ones. In this case you do not need to wait for a reversing step at which the price would move in the favorable direction.

**Manual trading tools**. Specially for the manual trading purposes, I have created the RevertManualEA Expert Advisor for MetaTrader 5 and RevertmeManualEA4 for MetaTrader 4 (both of these EAs are available in the article attachments). These EAs manage trades which you open manually. The symbol of the trade is not important for the EA. It will manage all symbols no matter on which symbol chart the EA is running. So, you need to launch the EA only once, for example on your VPS. After that you will be able to perform trades from any computer using your MetaTrader 5 account.

For RevertManualEA to manage a position, you should do the following when performing a trade:

- Specify the same Magic number used by RevertManualEA (by default _777_);
- In order comments, specify the step number (i.e. _1_).

Unfortunately, comments or Magic number cannot be specified during manual trading from MetaTrader 5. Therefore, we need a special Expert Advisor for opening positions with appropriate parameters. Such as the Expert Advisor [Creating orders with a fixed stop in dollars](https://www.mql5.com/en/market/product/29801).

As for MetaTrader 4, it allows the setting of a comment, but does not support the specification of a Magic number. So we also need a separate EA for opening positions. Such as the Expert Advisor [Creating orders with a fixed stop in dollars (MT4)](https://www.mql5.com/en/market/product/32839).

**Trade entry algorithm**. So, any trading activity will be profitable if Stop Loss triggers less frequently than Take Profit, while the amount lost at Stop Loss is less than the amount earned at Take Profit. The logic is very simple.

The reversing technique allows reducing the chances of hitting a Stop Loss level. And the more steps in a chain you perform, the less chances you have of losing money. However, this may also increase the amount of potential loss, in case the entire chain closes by Stop Loss.

That is why I suggest doubling the position size not at every step, but after two steps. This allows reducing potential loss in case the entire chain closes by Stop Loss, as well as reducing the margin requirement in the last chain steps.

In addition, in order to reduce the loss in case the entire chain closes by Stop Loss, we will use diversification and enter positions on 5 symbols at a time, instead of opening only one symbol deal.

Let us open positions in a total of $10 Stop Loss, i.e. the Stop Loss of each position will be equal to $2. In this case, depending on the Take Profit size and the doubling step, we may receive the following profit on each symbol:

Take Profit 3:1, lot doubling after 2 deals:

- Step 1: $6;
- Step 2: $4;
- Step 3: $2;
- Step 4: $6;
- Step 5: $2;
- Step 6: -$2;
- Step 7: $6;
- Step 8: $2;
- Stop Loss: -34 $.

Take Profit 4:1, lot doubling after 2 deals:

- Step 1: $8;
- Step 2: $6;
- Step 3: $4;
- Step 4: $10;
- Step 5: $6;
- Step 6: $2;
- Step 7: $14;
- Step 8: $6;
- Stop Loss on the entire chain: -34 $.

Take Profit 4:1, lot doubling after 3 deals:

- Step 1: $8;
- Step 2: $6;
- Step 3: $4;
- Step 4: $2;
- Step 5: $8;
- Step 6: $4;
- Step 7: $0;
- Step 8: -$4;
- Stop Loss on the entire chain: -$24.

In the above variants, losses in case of a Stop Loss on the entire chain are much higher than each position profit. Therefore, you should have a significantly greater number of profitable deals than whole losing chains. From my own experience, I may assume that this can only be achieved through the use of large Stop Loss values. Even if you believe your deal opening levels to be very reliable, situations when the price starts moving up and down, hitting your Stop Loss levels, may often happen during short-term trading.

Therefore, you may use the following set of parameters during short-term trading:

Take Profit 5:1, lot doubling after 3 deals, only 3 deals in a chain:

- Step 1: 10 $;
- Step 2: 8 $;
- Step 3: 6 $;
- Stop Loss on the entire chain: -$6.


In this case we only use three reversing steps. In the first step we may earn profit if we the price moves in the expected direction. In the first step, we may get profit if our assumption related to a roll back from a level was wrong, and the level was broken. In the third step, we may earn profit if this was a false breakout. After that we fully exit the position, expecting that the level is lost and further chain steps would not help us.

You may also use the Take Profit level of 5:1, doubling after 2 deals and a very large number of steps in a chain. Such as, for example, 20-30. The more steps you allow, the less is the possibility of closing the whole chain by Stop Loss. The main thing is to calculate the required deposit to make sure it can withstand the drawdown throughout the chain steps. Also, your entire deposit should not be lost in case of the whole chain loss. Even after such a situation, you need to have funds to trade further.

**Position exit**. Positions are managed and closed by RevertManualEA (or RevertmeManualEA4). In addition, you can always close a separate position or all open positions.

Closing all positions is simple. For example, if the summary result of all open positions is positive and you do not want to wait each of them to be closed by a Take Profit. Open the symbol chart on which the EA is running. The _Close all_ button is available on the chart. A click on this button will close all positions having the same Magic number, which is specified in the EA settings. In addition to closing all positions, this will delete all orders for further chain steps.

**Manual trading example**. The below video demonstrates manual trading using two Expert Advisors: RevertManualEA and [Creating orders with a fixed stop in dollars](https://www.mql5.com/en/market/product/29801).

The profitability of this system during manual trading depends greatly on the accuracy of your entry points.

### Conclusions

As a conclusion, I would like to state once again that the reversing trading strategy can really be used for trading.

It will not bring you huge profits. The expected maximum when used with an acceptable risk is about 100% per year.

The system will not save you from losses. The whole chain can at any time be closed by Stop Loss and you will lose all profit received in the last N years.

But the system can work. To make it work, you need to follow two main rules:

- Do not use tight Stop Loss, better use Stop Losses values at the medium-term trading levels;
- Use Take Profit larger than Stop Loss.

Both during automated trading testing and in manual trading, tight Stop Loss levels do not allow profiting, if the number of trades within a chain is less than 10-15. Ranging movements often happen in the market, during which the entire chain gets closed by Stop Loss. In this case, you lose all the profit you received earlier. Therefore, it is very problematic or even impossible to perform automatic intraday trading using the reversing technique.

There is another problem with tight Stop Loss. Spread widening can kill your entire trading chain. Here is a real trading example:

![Spread widening](https://c.mql5.com/2/34/spread_102_points_m5.png)

Look at the last step in the chain. The broker's spread by that instrument suddenly widened greatly and closed the previous step by stop loss, although the price was moving to the Take Profit direction. After that a limit order triggered and was immediately closed by Stop Loss, because the Stop Loss size was less than the spread value. No Expert Advisor would be able to create the next Limit order in such conditions. However, if another limit order were opened, it could also be closed with the same Stop Loss. The above example is a good evidence of that the use of tight stop levels with brokers offering floating spread may lead to complete deposit loss.

### Attachments

The following zip archives are attached below:

- _RevertEA.zip_. A zip archive with an Expert Advisor for MetaTrader 5
- _RevertManualEA.zip_. A zip archive with an Expert Advisor for MetaTrader 5
- _RevertmeManualEA4.zip_. A zip archive with an Expert Advisor for MetaTrader 4
- _SETfiles.zip_. A zip archive with SET files with optimal settings for various symbols and different brokers
- _TESTfiles\_ma.zip_. A zip archive with the testing reports for a trading strategy using the Moving Average
- _TESTfiles\_ma\_norevert.zip_. A zip archive with the testing reports for a trading strategy using the Moving Average, without the reversing technique
- _TESTfiles\_plain.zip_. A zip archive with the testing reports for a reversing trading strategy without a Moving Average

Testing report archives contain reports only related to symbols considered in this article. This is because an archive for all symbols considered within this series of articles is too large and cannot be attached to the article.

The following suffixes can be used in SET files and report name:

- _no suffix_ — a trading system based on the reversing technique without the use of a Moving Average, with Stop Loss and Take Profit specified in points;
- _\_ma_ — a trading system based on the reversing technique and a Moving Average;
- _\_norevert_ — a trading system based on the Moving Average, without the reversing technique;
- _\_inf_ — a trading system without the Moving Average, using the reversing technique and Trailing Stop
- _\_percent_ — a trading system based on the reversing technique, without a Moving Average, with Stop Loss specified as a percentage of the price
- _\_atr_ — a trading system based on the reversing technique, without a Moving Average, with Stop Loss specified as a percentage of ATR.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5268](https://www.mql5.com/ru/articles/5268)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5268.zip "Download all attachments in the single ZIP archive")

[RevertManualEA.zip](https://www.mql5.com/en/articles/download/5268/revertmanualea.zip "Download RevertManualEA.zip")(150.14 KB)

[RevertEA.zip](https://www.mql5.com/en/articles/download/5268/revertea.zip "Download RevertEA.zip")(214.22 KB)

[RevertmeManualEA4.zip](https://www.mql5.com/en/articles/download/5268/revertmemanualea4.zip "Download RevertmeManualEA4.zip")(48.04 KB)

[SETfiles.zip](https://www.mql5.com/en/articles/download/5268/setfiles.zip "Download SETfiles.zip")(587.96 KB)

[TESTfiles\_ma.zip](https://www.mql5.com/en/articles/download/5268/testfiles_ma.zip "Download TESTfiles_ma.zip")(4665.65 KB)

[TESTfiles\_ma\_norevert.zip](https://www.mql5.com/en/articles/download/5268/testfiles_ma_norevert.zip "Download TESTfiles_ma_norevert.zip")(1276.71 KB)

[TESTfiles\_plain.zip](https://www.mql5.com/en/articles/download/5268/testfiles_plain.zip "Download TESTfiles_plain.zip")(8351.57 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing a cross-platform grid EA: testing a multi-currency EA](https://www.mql5.com/en/articles/7777)
- [Developing a cross-platform grid EA (Last part): Diversification as a way to increase profitability](https://www.mql5.com/en/articles/7219)
- [Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://www.mql5.com/en/articles/7013)
- [Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings](https://www.mql5.com/en/articles/6986)
- [Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)
- [Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://www.mql5.com/en/articles/5614)
- [Developing a cross-platform grider EA](https://www.mql5.com/en/articles/5596)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/298386)**
(13)


![Florian Silver Grunert](https://c.mql5.com/avatar/avatar_na2.png)

**[Florian Silver Grunert](https://www.mql5.com/en/users/bnd)**
\|
25 Sep 2020 at 14:24

Is the revertme\_manual EA running for you?


![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
25 Sep 2020 at 19:00

I haven't installed it, but I think you have it running.

What do you expect that won't happen?

Naively asked: Revert means that a position is reversed - how is this first position created?

![Florian Silver Grunert](https://c.mql5.com/avatar/avatar_na2.png)

**[Florian Silver Grunert](https://www.mql5.com/en/users/bnd)**
\|
28 Sep 2020 at 14:20

**Carl Schreiber:**

I haven't installed it, but I think you have it running.

What do you expect that won't happen?

Naively asked: Revert means that a position is reversed - how is this first position created?

Selected by hand.

Just watch the video

https://youtu.be/CUjm-MLcsfw

Two windows appear in the video. I only see the first dialogue.

If you haven't installed the EA then you can't say anything about it, can you?

![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
28 Sep 2020 at 15:42

**BND:**

Selected by hand.

Just watch the video

https://youtu.be/CUjm-MLcsfw

Two windows appear in the video. I only see the first dialogue.

If you haven't installed the EA then you can't say anything about it, can you?

Not about the content, but about how to deal with it.

When I look at the code, the first dialogue is at the top:

```
/*

 - Send a notification at MetaQuotes ID when a position is opened at this chain step

*/
```

So either your expectation is wrong or you are using the wrong programme?

But start the EA in the debugger. That way you can see what happens.

![Florian Silver Grunert](https://c.mql5.com/avatar/avatar_na2.png)

**[Florian Silver Grunert](https://www.mql5.com/en/users/bnd)**
\|
29 Sep 2020 at 10:54

**Carl Schreiber:**

Not about the content, but about how to deal with it.

When I look at the code, the first thing I see is at the top:

So either your expectation is wrong or you are using the wrong programme?

But start the EA in the debugger. That way you can see what happens.

What happens is exactly what I wrote and not what you see in the video. Wrong programme?

Give it a rest. That's no good.

![DIY multi-threaded asynchronous MQL5 WebRequest](https://c.mql5.com/2/34/Multi_WebRequest_MQL5.png)[DIY multi-threaded asynchronous MQL5 WebRequest](https://www.mql5.com/en/articles/5337)

The article describes the library allowing you to increase the efficiency of working with HTTP requests in MQL5. Execution of WebRequest in non-blocking mode is implemented in additional threads that use auxiliary charts and Expert Advisors, exchanging custom events and reading shared resources. The source codes are applied as well.

![Reversing: Reducing maximum drawdown and testing other markets](https://c.mql5.com/2/34/Graal.png)[Reversing: Reducing maximum drawdown and testing other markets](https://www.mql5.com/en/articles/5111)

In this article, we continue to dwell on reversing techniques. We will try to reduce the maximum balance drawdown till an acceptable level for the instruments considered earlier. We will see if the measures will reduce the profit. We will also check how the reversing method performs on other markets, including stock, commodity, index, ETF and agricultural markets. Attention, the article contains a lot of images!

![Using OpenCL to test candlestick patterns](https://c.mql5.com/2/34/OpenCL_for_candle_patterns.png)[Using OpenCL to test candlestick patterns](https://www.mql5.com/en/articles/4236)

The article describes the algorithm for implementing the OpenCL candlestick patterns tester in the "1 minute OHLC" mode. We will also compare its speed with the built-in strategy tester launched in the fast and slow optimization modes.

![Reversal patterns: Testing the Head and Shoulders pattern](https://c.mql5.com/2/34/5358_avatar.png)[Reversal patterns: Testing the Head and Shoulders pattern](https://www.mql5.com/en/articles/5358)

This article is a follow-up to the previous one called "Reversal patterns: Testing the Double top/bottom pattern". Now we will have a look at another well-known reversal pattern called Head and Shoulders, compare the trading efficiency of the two patterns and make an attempt to combine them into a single trading system.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/5268&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068361026547677321)

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