---
title: MQL4  as a Trader's Tool, or The Advanced Technical Analysis
url: https://www.mql5.com/en/articles/1410
categories: Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:09:25.255784
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/1410&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083371443326032484)

MetaTrader 4 / Examples


### Introduction

Trading is, first of all, a calculus of probabilities. The proverb about idleness
being an engine for progress reveals us the reason why all those indicators and
trading systems have been developed. It comes that the major of newcomers in trading
study "ready-made" trading theories. But, as luck would have it, there
are some more undiscovered market secrets, and tools used in analyzing of price
movements exist, basically, as those unrealized technical indicators or math and
stat packages. Thanks awfully to Bill Williams for his contribution to the market
movements theory. Though, perhaps, it's too early to rest on oars.

### Keeping Statistics

We can ask ourselves: "What color of candlesticks prevails in the one-hour
chart for EURUSD?" We can start to count black ones noting every new hundred
thereof in the block, then count white ones. But we can also write about a dozen
of code lines, which will do this automatically. Basically, everything is logical
and there is nothing unusual here. However, let us find an answer to the above
question. First of all, let us simplify the candlestick color identification:

```
bool isBlack(int shift)
  {
    if(Open[shift] > Close[shift])
        return (true);
    return (false);
  }
//+------------------------------------------------------------------+
bool isWhite(int shift)
  {
    if(Open[shift] < Close[shift])
        return (true);
    return (false);
  }
//+------------------------------------------------------------------+
```

Using the code already written, we will continue the experiment.

```
//EXAMPLE 1
      //Calculate black and white candles
      double BlackCandlesCount = 0;
      double WhiteCandlesCount = 0;
      double BProbability = 0;

      for(int i = 0; i < Bars - 1; i++)
        {
          if(isBlack(i) == true)
              BlackCandlesCount++;

          if(isWhite(i) == true)
              WhiteCandlesCount++;
        }

      BProbability = BlackCandlesCount / Bars;
```

The result is interesting and quite predictable: 52.5426% of 16000 candles are white.
Using the MQL4 compiler, we can also solve a problem of candles cyclicity. For
example, if a black candle has been closed, what is the probability of forming
a white one? This, of course, depends on a great variety of factors, but let us
refer to statistics.

```
//EXAMPLE 2
      //Calculate seqences of 1st order
      //BW means after black going white candle
      double BW = 0;
      double WB = 0;
      double BB = 0;
      double WW = 0;

      for(i = Bars; i > 0; i--)
        {
         if(isBlack(i) && isWhite(i-1))
             BW++;
         if(isWhite(i) && isBlack(i-1))
             WB++;
         if(isBlack(i) && isBlack(i-1))
             BB++;
         if(isWhite(i) && isWhite(i-1))
             WW++;
        }
```

The result obtained:

\- White followed by Black - 23.64 %

\- Black followed by White - 23.67 %

\- White followed by White - 21.14 %

\- Black followed by Black - 20.85 %

As we can see, the probability that a candle will be followed by a candle of the
same color is a bit less than that of the opposite color.

Using MQL4 and having historical data, a trader can make some more profound market
researches. The terminal allows drawing histograms. We will use this function to
draw the candle color distribution according to values of indicators WPR and RSI.

```
//EXAMPLE 3.1
      //Build histogram by RSI
      //RSI min/max - 0/100

      double RSIHistogramBlack[100];
      double RSIHistogramWhite[100];

      for(i = Bars; i > 0; i--)
        {
          int rsi_val = iRSI(NULL,0,12,PRICE_CLOSE,i);
          if(isWhite(i))
              RSIHistogramWhite[rsi_val]++;
          if(isBlack(i))
              RSIHistogramBlack[rsi_val]++;
        }
      for(i = 0; i < 100; i++)
        {
          ExtMapBuffer1[i] = RSIHistogramBlack[i];
          ExtMapBuffer2[i] = -RSIHistogramWhite[i];
        }

//EXAMPLE 3.2
      //Build histogram by %R
      //%R min/max - 0/-100

      double WPRHistogramBlack[100];
      double WPRHistogramWhite[100];

      for(i = Bars; i > 0; i--)
        {
          int wpr_val = iWPR(NULL,0,12,i);
          int idx = MathAbs(wpr_val);
          if (isWhite(i))
              WPRHistogramWhite[idx]++;
          if (isBlack(i))
              WPRHistogramBlack[idx]++;
        }
```

![](https://c.mql5.com/2/14/experiment.gif)

Anyway, it would be more objective, instead of counting black and white candlesticks,
to keep statistics of profitable and losing trades with different values of StopLoss
and TakeProfit. The procedure below will be helpful for this purpose:

```
int TestOrder(int shift, int barscount, int spread, int tp, int sl, int operation)
 {
   double open_price = Close[shift];

   if (operation == OP_BUY)
      open_price  = open_price + (Point * spread);

   if (operation == OP_SELL)
      open_price  = open_price - (Point * spread);


   for (int i = 0; i<barscount; i++)
    {
      if (operation == OP_BUY)
       {
         //sl
         if (Low[shift-i] <= open_price - (Point * sl) )
            return (MODE_STOPLOSS);
         //tp
         if (High[shift-i] >= open_price + (Point * tp) )
            return (MODE_TAKEPROFIT);
       }

      if (operation == OP_SELL)
       {
         //sl
         if (High[shift-i] >= open_price + (Point * sl) )
            return (MODE_STOPLOSS);
         //tp
         if (Low[shift-i] <= open_price - (Point * tp) )
            return (MODE_TAKEPROFIT);
       }

    }
   return (MODE_EXPIRATION);
 }
```

I am sure that the results will be surprising for you. Kohonen's maps, Gaussian distribution, Hurst coefficient will astonish you even more. Basically, there are many astonishing
things. The main thing is not to forget about the essence and sense of trading.

### Conclusion

Basically, every trader uses his or her own trading techniques. Of course, nothing
will prevent him or her to represent the effectiveness of his or her system pictorially,
analyze it and utilize in trading. No result is a result, too. Knowledge got by
the trader will just enhance his or her trading productivity.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1410](https://www.mql5.com/ru/articles/1410)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1410.zip "Download all attachments in the single ZIP archive")

[instrument.mq4](https://www.mql5.com/en/articles/download/1410/instrument.mq4 "Download instrument.mq4")(5.02 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Expert System 'Commentator'. Practical Use of Embedded Indicators in an MQL4 Program](https://www.mql5.com/en/articles/1406)
- [Working with Files. An Example of Important Market Events Visualization](https://www.mql5.com/en/articles/1382)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39288)**
(7)


![Andrey Opeyda](https://c.mql5.com/avatar/2010/7/4C50E70A-F9C4.jpg)

**[Andrey Opeyda](https://www.mql5.com/en/users/njel)**
\|
5 Apr 2007 at 17:18

the point is to use the MQL language for research aims.

**lalilo:**

what is the point of this article ?!!!

![Giampiero Raschetti](https://c.mql5.com/avatar/2010/12/4D03B0F6-464A.jpg)

**[Giampiero Raschetti](https://www.mql5.com/en/users/giaras)**
\|
28 Mar 2008 at 17:24

That's a very interesting article for identifying statistical pattern.

I would like to apply something like this to a statistical analysis like the one done on some interesting articles

on [Currency Trader](https://www.mql5.com/en/blogs/tags/forexnews "Latest news from foreign exchange market") on line magazine.

They create a table starting from this values:

![](https://c.mql5.com/3/54/picturee1.png)

![Myles Crouch-Anderson](https://c.mql5.com/avatar/avatar_na2.png)

**[Myles Crouch-Anderson](https://www.mql5.com/en/users/mpaca)**
\|
12 May 2016 at 11:40

how can this be?

The result obtained:

\- White followed by Black - 23.64 %

\- Black followed by White - 23.67 %

\- White followed by White - 21.14 %

\- Black followed by Black - 20.85 %

surely it must add up to 100%? there arent any possible outcomes?

![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
12 May 2016 at 20:08

**Myles Crouch-Anderson:**

how can this be?

surely it must add up to 100%? there arent any possible outcomes?

Dark matter?


![Keith Watford](https://c.mql5.com/avatar/avatar_na2.png)

**[Keith Watford](https://www.mql5.com/en/users/forexample)**
\|
17 Jun 2022 at 05:58

Comments that do not relate to this topic, have been moved to " [Off Topic Posts](https://www.mql5.com/en/forum/339471)".

![Poll: Traders’ Estimate of the Mobile Terminal](https://c.mql5.com/2/14/345_1.png)[Poll: Traders’ Estimate of the Mobile Terminal](https://www.mql5.com/en/articles/1471)

Unfortunately, there are no clear projections available at this moment about the future of the mobile trading. However, there are a lot of speculations surrounding this matter. In our attempt to resolve this ambiguity we decided to conduct a survey among traders to find out their opinion about our mobile terminals. Through the efforts of this survey, we have managed to established a clear picture of what our clients currently think about the product as well as their requests and wishes in future developments of our mobile terminals.

![Effective Averaging Algorithms with Minimal Lag: Use in Indicators](https://c.mql5.com/2/14/297_2.png)[Effective Averaging Algorithms with Minimal Lag: Use in Indicators](https://www.mql5.com/en/articles/1450)

The article describes custom averaging functions of higher quality developed by the author: JJMASeries(), JurXSeries(), JLiteSeries(), ParMASeries(), LRMASeries(), T3Series(). The article also deals with application of the above functions in indicators. The author introduces a rich indicators library based on the use of these functions.

![Synchronization of Expert Advisors, Scripts and Indicators](https://c.mql5.com/2/13/117_1.gif)[Synchronization of Expert Advisors, Scripts and Indicators](https://www.mql5.com/en/articles/1393)

The article considers the necessity and general principles of developing a bundled program that would contain both an Expert Advisor, a script and an indicator.

![What Is a Martingale?](https://c.mql5.com/2/14/288_3.png)[What Is a Martingale?](https://www.mql5.com/en/articles/1446)

A short description of various illusions that come up when people trade using martingale betting strategies or misuse spiking and the like approaches.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/1410&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083371443326032484)

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