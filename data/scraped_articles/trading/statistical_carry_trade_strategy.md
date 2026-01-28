---
title: Statistical Carry Trade Strategy
url: https://www.mql5.com/en/articles/491
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:39:07.053045
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/491&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083014054802363209)

MetaTrader 5 / Trading


### Economic Regulators

According to Adam Smith's theory set forth in his book "An Inquiry into the Nature and Causes of the Wealth of Nations"\[1\], all economic processes are automatically regulated by the market economy using the forces of supply and demand, thus keeping them in the optimal state.

Unfortunately, however, practice shows otherwise. Market supply and demand often lead to distorted financial relationships and bring about economic crises.

To reduce the impact of economic imbalances, government regulators are involved to help the market economy.

The objectives of government regulators consist in the indirect control of economic processes using:

- Bank reserves, i.e. insurance funds accumulated
- Export and import quotas
- Subsidizing certain economic fields that cannot independently survive world competition
- Regulating interest rates.


### Interest Rates

Interest rates are used by central banks to control economic processes at the government level:

- The official discount rate (ODR) is the most effective tool of the government economic regulation. This is the rate that a central bank charges commercial banks for borrowing money from it.
- Repo Rate is used by a central bank to account for and repurchase government securities from commercial banks
- Fund Rate is the reserve funds rate
- Lombard Rate is the rate charged for loans secured by pledge.

### Regulated Economy

Do not have illusions about living in the era of 'free' market economy. Adam Smith's ideas are more than utopian. Market participants do not have to regulate economic processes at their own risk and peril. And for commercial reasons, they often invest in the anti-market ones, e.g.:

- Investments in drug trafficking. The result being is that a part of the population gets incapacitated and the crime rate rises.
- Investments in bubbles. As a result, finances stop to be a part of economic production and consumption of goods and instead become a part of lottery scam. Such investments ultimately turn into lost savings for a considerable part of the population.
- Investments in derivatives. Derivatives act as a destabilizing factor for the market supply and demand and lead to dramatic economic shifts, all the way down to world crises.

We live in the era of the regulated economy. It is not centrally planned where economic processes are directly controlled by the government, yet it is regulated.

### The Official Discount Rate

The discount rate set by a certain country's Central Bank is one of the major investment factors. It gives an indication to investors, especially to those from other countries, of the percentage of profit they will get if their savings are kept in the national currency or government bonds of a particular country. For the higher the discount rate, the higher the interest.

Central banks therefore use the discount rate to regulate the state economy, i.e. to either attract investors by increasing the rate, if this is necessary, or to lower the rate in case of economic overheating.

However one should not indulge in illusions. A higher discount rate does not necessarily add to the attractiveness of the currency. There is another significant factor that investors take into account – inflation. If the inflation rate is considerably higher than the discount rate, there is no point in investing in such an economy.

For example, the Central Bank of the Republic of Zimbabwe once increased the discount rate to 950% which only scared investors away as the money printing operations in that country just could not keep pace with the inflation and the banknote printing paper was more expensive than the nominal value of banknotes.

Low discount rate does not always indicate that the real economy is overheated but often signals about the extreme prevalence of bubbles.

### Carry Trade Strategy

Carry Trade is the strategy of making profits based on positive swaps.

When trading currency pairs, discount rates are transformed into the difference between the discount rate of the currency to be bought and the currency to be sold, i.e. a swap. The difference can therefore be negative for either purchases or sales. Making money based on positive swaps is attractive to traders, especially given the leverage. However the leverage is a two-edged sword, i.e. if the prices start moving in the direction opposite to that of the open position, the losses can exceed the future potential profit and lead to a margin call. It is therefore a risky venture to make money based on swaps trading one currency pair.

Carry trade has some distinct advantages, such as the fact that, being a low frequency trading strategy, it is devoid of problems associated with high frequency trading, like the need to constantly monitor trading signals, connection failures and others. VPS hosting is not a necessity. Every now and then you just need to monitor statistics and follow the news.

This article will provide a variant of the carry trade protection strategy which allows to compensate for potential risk of the price movement in the direction opposite to that of the open position.

Statistical carry trade strategy is a multi-currency strategy as it involves two or more currency pairs so as to compensate for potential losses from unwanted price movements due to cross correlations. However it is implemented in such a way so as to gradually increase profit on equity, even when blocked by negatively correlated financial instruments.

### Statistical Carry Trade Mathematics

Statistical carry trade is based on assumptions:

1. Prices for currency pairs shall move in the direction of positive swaps.
2. If two or more currency pairs are quoted in terms of one highly liquid currency, their correlations are positive. The price movements can consequently be canceled out by oppositely directed and positively correlated positions.


Assumptions are however not treated as fixed rules; the two points above are therefore only a hypothesis that needs proving using statistical methods. It could be that the majority of investors are, for one reason or another, of a different opinion based on fundamental factors and prefer to avoid risks, regardless of positive swaps.

Since the carry trade protection strategy variant involves several currency pairs that mutually cancel out the unwanted price movements, the statistical analysis of quoting processes using historical data should be very thorough.

In a very simple case where n currency pairs are used, a statistical model of the quoting process is a linear equation as follows:

v1 \* d1 + v2 \* d2 + …  + vn \* dn  =
profit

where:

n is the total number of financial instruments.

v1, v2, …, vn are the volumes of positions being opened in the relevant financial instruments. If the volume value is negative, a short position is opened.

d1, d2, …, dn is the average price change over one trading day for a financial instrument.

profit is the average profit over one trading day.

The formula will be shorter if simplified for two financial instruments:

v1 \* d1 + v2 \* d2 =
profit

Transform it:

d1  =  (-v2 \* d2 + profit) / v1

In this case, if we assume that:

> v1 = 1
>
>  y = d1
>
>  a = -v2
>
>  b =
>  profit

We get the classical formula of linear equation with one argument and in two unknowns:

y = a \* x + b

The unknowns a and b can be calculated using the classical least squares method.

Following that, you should specify the profit size using swaps and get the final results of the potential profit over one trading day:

b’ = b – swap1 + a \* swap2

where:

> swap1, swap2 are the swaps of currency pairs calculated over one trading day for the relevant open position directions.

Since the algorithm strategy set forth in the article presupposes the concurrent satisfaction of two conditions:

1. Volumes and directions of currency pairs are selected so that they are, on average, profitable.
2. Swaps of all currency pairs involved in the strategy shall be positive.


the additional testing using the last formula according to the above conditions becomes unnecessary.

### Exemplification

Why do we have y = a \* x + b, b = profit in our formula?

![Calculation example](https://c.mql5.com/2/4/carry_trading_02.png)

Assume that daily price movements of two currency pairs denoted by the identifiers **y** and **x** can be described by the formula:

y = 2 \* x + 1

Transform it into a similar formula:

y – 2 \* x = 1

That is, we need to open a long position in the first financial instrument (positive sign) and a short position (negative sign), being twice the size of the first position (as a = 2), in the second financial instrument.

In our example, the current instrument prices are 10 and 8.

Assume that the price for the second instrument increased by 1 over one trading day, i.e. it reached 9. Consequently, the price for the first instrument will, on average, change by 2 \* x + 1 = 2 \* 1 + 1 = 3 and reach 13 (prices for both instruments increased simultaneously because the correlation is positive). Since the position of the second instrument is short, the loss on it will be 2, while the first instrument in the long position will earn 3. The difference, i.e. the profit will be +1.

Suppose that following the next trading day the second instrument price decreased by 1 and returned to the former value of 8. The first instrument price will in this case also decrease by the value of 2 \* x + 1 = 2 \* -1 + 1 = -1 and will be equal to 12. Calculating the results: there is a loss of 1 on the first instrument and a profit of 2 on the second instrument. The final result is again +1. That is, regardless of the direction and the range of the price movement, we would, on average, still gain profit in the amount specified in the formula and denoted by the identifier **b**.

Thus, knowing the formula in the form of the linear equation we can determine the directions and volumes of the opening positions in two cross correlated financial instruments so as to gain an average profit, regardless of the price direction.

But do not be too carried away as the formula is calculated by means of the least squares method, i.e. following a statistical approach and using historical data. It does not guarantee any future profit. We need statistics to make sure that the market entry directions selected are profitable both when using historical data and carry trading. If in the future something does not go as expected based on the calculations, we will still benefit from the difference on swaps.

### Implementation

Being too laborious for manual calculations, the least squares method calculation should better be assigned to an Expert Advisor.

The Expert Advisor calculates position directions and volumes in two financial instruments so as to gain an average profit. It then requests the server to provide information on the value of swaps for the opening position directions selected and should both swap values be positive, gives a recommendation.

The Expert Advisor source code:

```
//+------------------------------------------------------------------+
//|                                        StatisticCarryTrading.mq5 |
//|                                  Copyright 2012, Ruslan V. Lunev |
//|                              https://www.mql5.com/ru/articles/491 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, Ruslan V. Lunev"
#property link      "https://www.mql5.com/ru/articles/491"
#property version   "1.00"

// Second currency pair
input string secondpair="AUDUSD";
// Statistics collection period in bars
input int p=100;

// Arrays for storing historical opening prices
double open0[];
double open1[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Reading the time series of opening prices
   // for the currency pairs involved
   CopyOpen(_Symbol,PERIOD_D1,0,p+1,open0);
   ArraySetAsSeries(open0,true);
   CopyOpen(secondpair,PERIOD_D1,0,p+1,open1);
   ArraySetAsSeries(open1,true);

   int i=0;

   double pp=p;

   double s1 = 0;
   double s2 = 0;
   double s3 = 0;
   double s4=open1[0]-open1[p];
   double s5=open0[0]-open0[p];

   double averagex = s4 / pp;
   double averagey = s5 / pp;

   for(i=0; i<p; i++)
     {
      double x0 = open1[i] - open1[i + 1];
      double y0 = open0[i] - open0[i + 1];
      double x1 = x0 - averagex;
      double y1 = y0 - averagey;
      s1 = s1 + x1 * x1;
      s2 = s2 + y1 * y1;
      s3 = s3 + x1 * y1;
     }

   // Pearson's linear correlation coefficient
   double r=s3/MathSqrt(s1*s2);

   // Calculation of proportions of opening positions sizes given the contract sizes
   double a = signum(r) * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE) * MathSqrt(s2)
   / (MathSqrt(s1) * SymbolInfoDouble(secondpair, SYMBOL_TRADE_CONTRACT_SIZE));

   // Calculation of the average daily profit
   double b = averagey - averagex * a;

   // Derive the resulting formula of joint price movement
   if(b>0)
     {
      Print(_Symbol+" = ",a," * "+secondpair+" + ",b);
        } else {
      Print(_Symbol+" = ",a," * "+secondpair+" - ",MathAbs(b));
     }

   a=-a*signum(b);

   // Recommendations
   string recomendation="Buy "+_Symbol;

   if(b<0)
     {
      recomendation="Sell "+_Symbol;
      if(SymbolInfoDouble(_Symbol,SYMBOL_SWAP_SHORT)<0.0)
        {
         recomendation="Short positions swap in "+_Symbol+" is negative";
         MessageBox(recomendation,"Not recommended",1);
         return(0);
        }
        } else {
      if(SymbolInfoDouble(_Symbol,SYMBOL_SWAP_LONG)<0.0)
        {
         recomendation="Long positions swap in "+_Symbol+" is negative";
         MessageBox(recomendation,"Not recommended",1);
         return(0);
        }
     }

   if(a<0)
     {
      recomendation=recomendation+"\r\nSell "+a+" "+secondpair;
      if(SymbolInfoDouble(secondpair,SYMBOL_SWAP_SHORT)<0.0)
        {
         recomendation="Short positions swap in "+secondpair+" is negative";
         MessageBox(recomendation,"Not recommended",1);
         return(0);
        }
        } else {
      recomendation=recomendation+"\r\nBuy "+a+" "+secondpair;
      if(SymbolInfoDouble(secondpair,SYMBOL_SWAP_LONG)<0.0)
        {
         recomendation="Long positions swap in "+secondpair+" is negative";
         MessageBox(recomendation,"Not recommended",1);
         return(0);
        }
     }

   double profit=MathAbs(b)/SymbolInfoDouble(_Symbol,SYMBOL_POINT);

   if((SymbolInfoInteger(_Symbol,SYMBOL_DIGITS)==5) || (SymbolInfoInteger(_Symbol,SYMBOL_DIGITS)==3))
     {
      profit=profit/10;
     }

   recomendation = recomendation + "\r\nCorrelation coefficient: " + r;
   recomendation = recomendation + "\r\nAverage daily profit: "
   + profit + " points";

   MessageBox(recomendation,"Recommendation",1);

   return(0);
  }

// Step function - Signum
double signum(double x)
  {
   if(x<0.0)
     {
      return(-1.0);
     }
   if(x==0.0)
     {
      return(0);
     }
   return(1.0);
  }
//+-----------------------The End ------------------------
```

The Expert Advisor has two input parameters:

- **p** is the period in daily bars which provide all statistics required in calculations. It is preferred that this parameter value not exceed the time of the last change in discount rates made by central banks of the countries whose three currencies are indicated in the first and the second pair.
- **secondpair** is the second financial instrument. The first financial instrument is the currency pair of the chart the Expert Advisor is attached to. The second pair is selected so that the currency used for the calculation of points coincided with that of the first one (the last three characters in the currency pair identifier). For example: EURUSD and AUDUSD or GBPJPY and NZDJPY, etc. Before selecting the second pair, one should make sure that it has a positive swap by checking the contract specification.

The Expert Advisor should be attached to the chart of the first currency pair running on the D1 time frame, while the second currency pair should be specified in the input parameters.

This will be followed by the calculation over a number of bars 'p' set in the input parameters and a recommendation will be displayed.

If the recommendation given suits the trader, he can open positions manually sticking to the directions and volumes as directed by the Expert Advisor. After that, the Expert Advisor does not have to be removed from the chart as the strategy employed is a low frequency strategy and there is no need in having the terminal constantly connected to the server. Instead, every time the Expert Advisor accesses the terminal it will either give the same recommendation as before or, depending on the changes of the broker's swap rate or statistics based on historical data, display a "Not recommended" message which allows the trader to promptly change the previous strategy and close all trading orders that were already placed.

### Warning

It is known that financial instruments are not time independent and their statistical parameters can consequently change over time. The volume values and currency pair directions calculated under this strategy using statistics are therefore not predicted but rather confirm the hypothesis that the price moves in the direction of positive discount rates, even if canceled out by cross correlations of oppositely directed currency pairs. That is, there is a statistically confirmed market demand for currencies with high discount rates.

### References

1. Smith, А., An Inquiry of the Nature and Causes of the Wealth of Nations. — М.: Eksmo, 2007. — (Series: The Anthology of Economic Thought) — 960 p. — ISBN 978-5-699-18389-0


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/491](https://www.mql5.com/ru/articles/491)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/491.zip "Download all attachments in the single ZIP archive")

[StatisticCarryTrading.mq4](https://www.mql5.com/en/articles/download/491/statisticcarrytrading.mq4 "Download StatisticCarryTrading.mq4")(3.87 KB)

[StatisticCarryTrading.mq5](https://www.mql5.com/en/articles/download/491/statisticcarrytrading.mq5 "Download StatisticCarryTrading.mq5")(4.16 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/8289)**
(13)


![Heroix](https://c.mql5.com/avatar/2013/7/51D5B71D-6867.jpg)

**[Heroix](https://www.mql5.com/en/users/heroix)**
\|
14 Sep 2012 at 08:40

**pronych:**

1\. Do you realise that the author is making assumptions?

2\. i.e. if you open in the direction of positive swaps, not only _will the price be pulled in your direction_, but the swaps will increase. This is not arbitrage, this is "carry trading"

I didn't know about the first point. It's a useful thought.

So, if the assumptions are correct, the strategy [makes objective](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string "MQL5 Documentation: Object Properties") sense.

Respect.

1\. I understand, I don't see their point. There is no benefit, only harm.

2\. Delusion again. Enough of this chatter already. "Thought is useful" etc. - it's all to hell. Only practice shows what is valuable:

Let's take EURAUD cross and specification from DC A. Swap short = 1.14, long = -1.4. Not bad swaps. So what? Quotes **don't care about** that:

[![](https://c.mql5.com/3/11/1v.gif)](https://c.mql5.com/3/11/iu.gif "https://c.mql5.com/3/11/iu.gif")

However, it does not prevent some people from flying in the clouds of their assumptions and thoughts.

![jcbrosse2](https://c.mql5.com/avatar/avatar_na2.png)

**[jcbrosse2](https://www.mql5.com/en/users/jcbrosse2)**
\|
16 Oct 2012 at 18:36

**Automated-Trading:**

New article [Statistical Carry Trade Strategy](https://www.mql5.com/en/articles/491) is published:

Author: [Ruslan Lunev](https://www.mql5.com/en/users/ruslun "https://www.mql5.com/en/users/ruslun")

Hi Ruslan

I just wanted to know if it does open trade at all ?

and thank you for your article about carry trade .

JCB

![Kourosh Hossein Davallou](https://c.mql5.com/avatar/avatar_na2.png)

**[Kourosh Hossein Davallou](https://www.mql5.com/en/users/kourosh1347)**
\|
8 Dec 2012 at 16:45

Hi ruslan

thank you for article

![brazandeh](https://c.mql5.com/avatar/avatar_na2.png)

**[brazandeh](https://www.mql5.com/en/users/brazandeh)**
\|
15 Dec 2012 at 10:52

I am not math expert ,but pretty sure that the whole calculations are wrong

```
double averagex = open1[0]-open1[100] / 100;
```

what I expect is the summation of all the [open prices](https://www.mql5.com/en/docs/constants/indicatorconstants/prices#enum_applied_price_enum "MQL5 documentation: Price Constants") from 0 through 100, divided by 100

besides, the following formula

```
y = 2 * x + 1
```

does not seem to be true.

Can anyone with more knowledge in math justify me?


![ffoorr](https://c.mql5.com/avatar/avatar_na2.png)

**[ffoorr](https://www.mql5.com/en/users/ffoorr)**
\|
6 Apr 2019 at 14:09

There is an error in the message box :

```
         MessageBox(recomendation,"Not recommended", 1);
```

To be replaced with :

```
         MessageBox(recomendation + " ...Not recommended", 1);
```

So to get the message not recommanded

Post scriptum : it is not an error, the "not recommended" is the title of the box, but on an black screen, it is not seen.

The "return(0);"  should be all deleted

For this part

```
 if(b>0)
     {
      Print("###", _Symbol+" = ",a," * "+secondpair+" + ",b);
        } else {
      Print("###   ",_Symbol+" = ",a," * "+secondpair+" - ",MathAbs(b));
     }
```

here are the result I get :

2019.04.06 15:41:15.077    StatisticCarryTrading [USDCHF](https://www.mql5.com/en/quotes/currencies/usdchf "USDCHF chart: technical analysis"),Daily: ###   USDCHF = -0.9075968272144176 \* AUDUSD - 0.0001681237443192841

2019.04.06 15:41:20.802    StatisticCarryTrading USDCHF,Daily: recomendation  Average daily profit: 1.681237443192841 points

2019.04.06 15:41:19.539    StatisticCarryTrading USDCHF,Daily:  recomendation  Correlation coefficient: -0.212986151589174

![Communicating With MetaTrader 5 Using Named Pipes Without Using DLLs](https://c.mql5.com/2/0/pipe-ava__1.png)[Communicating With MetaTrader 5 Using Named Pipes Without Using DLLs](https://www.mql5.com/en/articles/503)

Many developers face the same problem - how to get to the trading terminal sandbox without using unsafe DLLs. One of the easiest and safest method is to use standard Named Pipes that work as normal file operations. They allow you to organize interprocessor client-server communication between programs. Take a look at practical examples in C++ and MQL5 that include server, client, data exchange between them and performance benchmark.

![How to Subscribe to Trading Signals](https://c.mql5.com/2/0/signals_avatar.png)[How to Subscribe to Trading Signals](https://www.mql5.com/en/articles/523)

The Signals service introduces social trading with MetaTrader 4 and MetaTrader 5. The Service is integrated into the trading platform, and allows anyone to easily copy trades of professional traders. Select any of the thousands of signal providers, subscribe in a few clicks and the provider's trades will be copied on your account.

![Interview with Francisco García García (ATC 2012)](https://c.mql5.com/2/0/avatar__15.png)[Interview with Francisco García García (ATC 2012)](https://www.mql5.com/en/articles/563)

Today we interview Francisco García García (chuliweb) from Spain. A week ago his Expert Advisor reached the 8th place, but the unfortunate logic error in programming threw it from the first page of the Championship leaders. As confirmed by statistics, such an error is not uncommon for many participants.

![Interview with Andrey Barinov (ATC 2012)](https://c.mql5.com/2/0/Wahoo_avatarm1q.png)[Interview with Andrey Barinov (ATC 2012)](https://www.mql5.com/en/articles/562)

It was on Friday of the Championship's first week that the trading robot of Andrey Barinov (Wahoo) occupied the fifth place in TOP-10. Andrey is a newcomer in the Championship but he has already managed to execute more than 100 orders in Jobs and develop a dozen of products for Market. We have arranged an interview with him and learned that the development of a "simple multicurrency Expert Advisor" is not an easy but a fairly easy task.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/491&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083014054802363209)

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