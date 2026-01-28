---
title: Bid/Ask spread analysis in MetaTrader 5
url: https://www.mql5.com/en/articles/9804
categories: Trading, Indicators
relevance_score: 3
scraped_at: 2026-01-23T18:15:29.315488
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=oxxlqjqaxclwfbnmckgijgrwpynbnksj&ssn=1769181328244847761&ssn_dr=0&ssn_sr=0&fv_date=1769181328&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9804&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Bid%2FAsk%20spread%20analysis%20in%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918132832963890&fz_uniq=5069285324984615568&sv=2552)

MetaTrader 5 / Trading


### Introduction

If you don't use Limit or Stop orders for both trade entry and exit then you will use market orders and of course these depend on the size of the Bid/Ask spread to determine what prices you receive.

When you hit the buy button, you actually buy at the ASK price which is a spread size above the bid price that you probably used to decide to buy.

When you hit the sell button, you actually sell at the BID price which is a spread size below the ask price.

Of course when you hit the close button to close a position that you had previously bought, you actually sell at the current BID price.

And the reverse is true, when you hit the close button to close a position that you had previously shorted, you actually buy back or cover at the current ASK price.

Now we can use tick data from MetaTrader 5 to analyze what the historic true average Bid/Ask spread actually have recently been.

You shouldn't need to look at the current spread because that is available if you show both bid and ask price lines.

### Let's look at why and how

Looking at these charts, you can see that this broker says that most of the spreads are 5 points.

If that was the case then it should cost you 1 pip for the round trip of opening and closing a trade.

So, for a trade with a 1/1 reward risk ratio with a Stop Loss of 10pips and Take Profit of 10 pips, it should cost you 10% of your risk/stake.

This kind of spread is fair enough, for example a bookies over-round book is typically 15%, casinos profit margin is around 4%.

![BAS-EURUSD-M30](https://c.mql5.com/2/43/BAS-EURUSD-M30.png)

But, the actual average spreads, the red line, versus the brokers documented spread, (black dashed line) are mostly twice as large as the declared spread as confirmed by the data window below. Using the earlier example with the same SL & TP, the cost to you is normally at least 2pips or 20%.

![BAS-EURUSD-M30-DW](https://c.mql5.com/2/43/BAS-EURUSD-M30-DW.png)

If you are a smaller scale scalper, e.g. using SL of 5pips & TP 5pips, or if you decide to get out before the previous examples 10 pip SL or TP are hit, at say a 5pip loss, then the cost is the same 2 pips, but because you played safe after the trade started to go against you, the percentage cost is now 40% of your stake/risk.

When I was a newbie trader  I started off using 5 pip S/l and 10 pip T/P for a 2:1 risk/reward ratio, (as I suspect many new traders do). I wasn't very successful.

So I did a deep analysis of the EURUSD M1 chart with a reliable Zig Zag indicator. I set it to 5 pips as the minimum leg size, which for me signified the kind of retracement I could deal with.

The results seem to suggest that most small swings were around 7 pips and the 10 pip legs were relatively rare in comparison. Of course I allowed for news releases and volatile markets, so the results were from mainly average periods from the trading sessions only.

So, in consequence I started using a 10 pip stop loss and left the take profit open, so I could monitor the trade closely and decide when to get out if the trade had already hit a 7 pip loss or profit. This resulted in an improvement, but still left me short of a profit. It was only then that I noticed the high bid/ask spreads I was trading against with that broker, so of course looked for a better broker.

![BAS-EURUSD-M1](https://c.mql5.com/2/43/BAS-EURUSD-M1.png)

If you trade when news comes out or the market become volatile, you can see that the actual average spread goes up to around 15 points or 3 times the standard 5 points, so you have to pay 3pips or 60% of your stake.

![BAS-EURUSD-M1-DW](https://c.mql5.com/2/43/BAS-EURUSD-M1-DW.png)

Don't even consider trading after 20:30 UK time (21:30 on chart server time), it could well be 4, 5, 6 times or even much higher, especially if you decide to hold onto your trading position over a weekend, which as you can see below is almost 10 times the standard 5 point spread, unless you have extremely large stop loss and take profit levels.

![BAS-EURUSD-M30-WeekEnd](https://c.mql5.com/2/43/BAS-EURUSD-M30-WE.png)

### OnInit() Code example

```
#property indicator_separate_window

#property indicator_buffers 2
#property indicator_plots   2

//--- plots
#property indicator_label1  "ActSpread"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

#property indicator_label2  "DeclaredSpread"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrBlack
#property indicator_style2  STYLE_DASH
#property indicator_width2  2

//--- indicator parameters
input int      numRecentBarsBack=100; //#RecentBarsBack M30+~100, M5~200, M1~500
input bool     doPrint=true;          //true=prints to the toolbox\experts log

//--- indicator buffers
double         ActSpreadBuf[], DeclaredSpreadBuf[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   int numBars=iBars(_Symbol,PERIOD_CURRENT)-2;

   // Check we have enough data for the request before we begin
   if(numRecentBarsBack>numBars)
   {
      Alert("Can't Do ", numRecentBarsBack, "! Only ",
               numBars, " Bars are Available",
               " try 100 or so for 30+ minute charts,",
               " 200 for 5 minute, or 500 for 1 minute charts.",
               " Otherwise the indicator may be too slow"
           );

      return(INIT_PARAMETERS_INCORRECT);
   }

   double sumPrice=0;
   double avgPrice=0;

   // Get the standard 5 point spread for the standard EURUSD currency
   double stdSpread=0.00005/iClose("EURUSD",PERIOD_M1,1); // 1.2 ~=  EURUSD std price

   //Find out the current average price of the instrument we are using, so we can standardise the spread and _Point
   int CheckAvgPriceBars=MathMin(numRecentBarsBack, 200);

   int i=0;
   for(; i<CheckAvgPriceBars; i++)
   {
      sumPrice+=iClose(_Symbol,PERIOD_CURRENT,i);
   }
   avgPrice=sumPrice/(i? i: 1.0);

   //convert the stdSpread to stdPoint by dividing by 5, so we compare  apples with apples, not oranges
   double stdPoint=StringToDouble(DoubleToString(avgPrice*stdSpread/5.0,6));

   Print(i, "=bars done, avgPrice=", DoubleToString(avgPrice,6),
            " std=", DoubleToString(1.2*stdSpread, 6),
            " stdPoint=", DoubleToString(stdPoint, 6)
         );

   SetIndexBuffer(0,ActSpreadBuf,INDICATOR_DATA);
   SetIndexBuffer(1,DeclaredSpreadBuf,INDICATOR_DATA);

   string indName ="BAS("+_Symbol;
          indName+=" TF="+string(_Period);
          indName+=" stdPoint="+DoubleToString(stdPoint, 6);
          indName+=") Last("+string(numRecentBarsBack)+") Bars";

   IndicatorSetString(INDICATOR_SHORTNAME, indName);

   IndicatorSetInteger(INDICATOR_DIGITS,6);

   IndicatorSetDouble(INDICATOR_MINIMUM, 0.0);

   IndicatorSetInteger(INDICATOR_LEVELS, 20);

   //mark out each standard EURUSD 5 point spread, to compare this currencies spread with EURUSD
   IndicatorSetDouble(INDICATOR_LEVELVALUE,0,  0.000000);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,1,  5*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,2, 10*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,3, 15*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,4, 20*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,5, 25*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,6, 30*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,7, 35*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,8, 40*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,9, 45*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,10,50*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,11,55*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,12,60*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,13,65*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,14,70*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,15,75*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,16,80*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,17,85*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,18,90*stdPoint);
   IndicatorSetDouble(INDICATOR_LEVELVALUE,19,95*stdPoint);

return(INIT_SUCCEEDED);
}
```

For this simple 2 plot indicator there are only 2 parameters, the first 'numRecentBarsBack' is for how many bars we want analysed.

The first thing we do in OnInit() is do a check that we have enough data to satisfy the request, if we haven't then we alert the user and suggest some realistic values to use, then exit the indicator early with an error.

The rest of OnInit() is fairly standard except for the levels used in the indicator sub-window, which are set to values that correspond to multiples of the standard EURUSD 5 point spread.

This is a fairly important step because as well as wanting to see the comparison between the declared and actual average spread values, we want to also see how big the spread of different currencies are as compared with the standard EURUSD, which normally has the lowest available spread of all the currencies.

This is a fairly convoluted method because we have to get the current EURUSD price (and substitute 1.2 if it doesn't exist) and use 5 EURUSD points divided by that price to build a standard spread. Then we iterate through numRecentBarsBack prices of the current Forex instrument, (I haven't tested it with non-forex instruments) to get an average price of that instrument.

When we have the instruments average price, we then build a rounded standard point by multiplying the instruments average price by the previously built standard spread and dividing by 5, the standard EURUSD point of spread.

This rounded standard point is then used in each level value and is also included in the indicators short name, as can be seen within the indicators name in the 'exotic' USDMXN chart below.

In this USDMXN example the trading daytime declared spread is around 0.0025 which is about 3 spread levels up from the zero so corresponds to around 15 points in a EURUSD chart. Also note that the actual average spread varies wildly above even that high level for this broker.

![BAS-USDMXN-M30](https://c.mql5.com/2/43/BAS-USDMXN-M30.png)

The GBPAUD chart below shows that the trading daytime declared spread is around 0.00019 which is about 2.5 spread levels up from the zero so corresponds to around 12 points in a EURUSD chart. Also note that in this chart the actual average spread values are fairly close to the declared values for this broker.

![ BAS-GBPAUD-M30](https://c.mql5.com/2/43/BAS-GBPAUD-M30.png)

The GBPJPY chart below shows that the trading daytime declared spread is around 0.020 which is about 3 spread levels up from the zero so corresponds to around 15 points in a EURUSD chart. Also note that in this chart the actual average spread values are fairly close to the declared values again for this broker.

![BAS-GBPJPY-M30](https://c.mql5.com/2/43/BAS-GBPJPY-M30.png)

The USDJPY chart below shows that the trading daytime declared spread is around 0.0050 which is roughly only 1 spread level up from the zero level so corresponds to around the standard 5 points in a EURUSD chart. Also note that in this chart the actual average spread values are again roughly double the declared values, so the same comments as the EURUSD risk/reward percentage levels also apply here.

![BAS-USDJPY-M30](https://c.mql5.com/2/43/BAS-USDJPY-M30.png)

Here are a few more examples, you can make your own assessments about the relationships between the spreads levels.

![BAS-GBPUSD-M30](https://c.mql5.com/2/43/BAS-GBPUSD-M30.png)

![BAS-EURGBP-M30](https://c.mql5.com/2/43/BAS-EURGBP-M30.png)

The second parameter is the boolean 'doPrint', which is checked in the code and if true will print the individual bars stats to the experts log as the examples below demonstrate. This can slow the indicator down if the value of 'numRecentBarsBack' is too large, so the default value is 100.

if you set the 'doPrint' parameter to true and 'numRecentBarsBack' to a reasonable value of somewhere around 100 for a 30 minute chart or 300 for a 1 minute chart, then you can copy the log entries and send them to your broker as proof of their true Bid/Ask spreads.

![Bid/Ask Spread M20 Log](https://c.mql5.com/2/43/BidAskSpreadImgM20-2.png)

![Bid/Ask Spread M1 Log](https://c.mql5.com/2/43/BidAskSpreadImgM1-2.png)

### OnCalculate() Code Example

```
//--- Global variables
//--- Set the date formatting for printing to the log
const uint dtFormat=uint(TIME_DATE|TIME_MINUTES);

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   //--- Check for no data or Stop flag before we begin
   if(_StopFlag || rates_total<2)
   {
         Alert("Error, StopFlag=", _StopFlag, " #Bars=", rates_total);
         return(rates_total);
   }

   //only do the report at indicator start up or refresh
   if(prev_calculated>2)
   {
      // if we have already nulled the ActSpreadBuf just do the DeclaredSpreadBuf[] and return.
      if(prev_calculated==rates_total)
      {
         int currBar=rates_total-1;
         DeclaredSpreadBuf[currBar]=spread[currBar]*_Point;
         return(rates_total);
      }
      // else its the start of a new bar so null the ActSpreadBuf
      else
      {
         int currBar=rates_total-1;
         ActSpreadBuf[currBar]=EMPTY_VALUE;
         return(rates_total);
      }
   }


   static int start=rates_total-numRecentBarsBack;

   MqlTick tickBuf[];

   double sumSpread=0;
   double thisSpread=0;

   int ticks=0;
   int bid_tick=0;
   int ask_tick=0;
   int k=0;

   ArrayInitialize(ActSpreadBuf, EMPTY_VALUE);
   ArrayInitialize(DeclaredSpreadBuf, EMPTY_VALUE);

   for(int i=start; i<rates_total; i++)
   {
      sumSpread=0;
      thisSpread=0;
      bid_tick=0;
      ask_tick=0;
      k=0;

      ticks=CopyTicksRange(_Symbol, tickBuf,
                           COPY_TICKS_INFO, // Only bid and ask changes are required
                           time[i-1]*1000,  // Start time of previous bar
                           time[i  ]*1000   // End time of previous bar
                           );

      while(k<ticks)
      {
         if((tickBuf[k].flags&TICK_FLAG_ASK)==TICK_FLAG_ASK)
            ask_tick++;

         if((tickBuf[k].flags&TICK_FLAG_BID)==TICK_FLAG_BID)
            bid_tick++;

         sumSpread+=tickBuf[k].ask-tickBuf[k].bid;

         k++;
      }

      // Ensure no divide by zero errors for any missing tick data
      if(ticks>0) {
         thisSpread=sumSpread/ticks;
         ActSpreadBuf[i-1]=thisSpread;
      }
      else  {
         thisSpread=0.0;
         ActSpreadBuf[i-1]=EMPTY_VALUE;
      }

      DeclaredSpreadBuf[i-1]=spread[i-1]*_Point;

      if(doPrint)
      {
                  Print(TimeToString(time[i-1], dtFormat),
                  "  NumTicks="+string(ticks),
                  "  b="+string(bid_tick),
                  "  a="+string(ask_tick),
                  "  AvgSpread=",  DoubleToString(thisSpread/_Point, 1),
                  "  DeclaredSpread=", string(spread[i-1])
                  );
      }

   }

   //don't do stats for incomplete current bar, but can do DeclaredSpread if it has a value
   DeclaredSpreadBuf[rates_total-1]=(spread[rates_total-1]*_Point);

//--- return value of prev_calculated for next call
return(rates_total);
}
```

From the above OnCalculate() example, the main point to note is the use of CopyTicksRange() to only get the tick data between the start of the previous indexes bar/candle and the start of the current indexes bar/candle. Also note that we have to convert the time\[\] array to milliseconds by multiplying it by 1000 because datetime data is only accurate down to the second, and CopyTicksRange() requires milliseconds.

```
ticks=CopyTicksRange(_Symbol, tickBuf,
                           COPY_TICKS_INFO, // Only bid and ask changes are required
                           time[i-1]*1000,  // Start time of previous bar
                           time[i  ]*1000   // End time of previous bar
                           );

```

Also note that we accumulate the bid and ask ticks, although we don't use them in the plots. The value of bid ticks should match the value in the tick\_volume\[\] array, and does as shown in the Data Window.

### Extra note about downloading ticks...

If you want to check a currency that you don't normally use, you will need to add that currency from the View\\Symbols menu item by double clicking it to Show Symbol. Whilst in this window you should also go to the ticks tab and request All ticks a date a month or so before today in the first date menu; and then set the second date menu to tomorrow to seed your local ticks database.

### Conclusion

Before we trade a currency we should know what our risk percentages are for the type of trading (scalping, swing, position...) we are considering and compare those of our favourite currencies with others available using a common standard spread size .

From my studies, I would advise traders to stick to the major currencies that are directly attached to the USD, namely USDCAD, USDCHF, USDJPY, EURUSD and GBPUSD; as they have the lowest overall spreads.

We all need to let our brokers know that we can now see their true Bid/Ask spreads, even if we are trading commission only, if they increase their spreads to very high levels. Good luck, and remember don't trade if you can't find a broker with reasonable Bid/Ask spreads during the trading hours, as you CANNOT win!

Before anyone asks, this process can only be run against MetaTrader 5 because the tick data is not available in MetaTrader 4, so it's a good reason to upgrade.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9804.zip "Download all attachments in the single ZIP archive")

[BidAskSpreadStats.mq5](https://www.mql5.com/en/articles/download/9804/bidaskspreadstats.mq5 "Download BidAskSpreadStats.mq5")(8.4 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/377157)**
(11)


![trens](https://c.mql5.com/avatar/avatar_na2.png)

**[trens](https://www.mql5.com/en/users/trens)**
\|
19 Oct 2021 at 06:26

I'm wildly apologising, but read below:

1\. Any broker has a marketing department, which advertises the company's services, including through the lowest possible spreads. Don't confuse advertising with the real state of affairs. A marketer has one useful function for the public - to draw your attention; they are not responsible for quality.

2\. Spread as a difference between buyers' and sellers' sentiments strongly depends on market liquidity. Liquidity, in turn, changes overnight, and depending on the state of the market. I saw usdjpy standing still for half a year in 2014, and the spread there was quite wide. You couldn't trade at all - price wasn't moving well and the spread was 2-3 times higher than in 2012-13. And then towards the end of the year an unstoppable trend started, liquidity was abundant, the market was roaring! Spread sometimes reached 1-2 pips on 3 digits (usdjpy). As for the time of day, Americans have the most money (USA is the financial centre of the planet). Also, the general liquidity of Europe overlaps with the American session. Therefore, at this time spreads are low, liquidity is high, everyone plays with excitement, the market is mobile. If you trade intraday, choose the American session only.

3\. The market spread at one and the same moment of time for one and the same instrument is not the same! The more volume you buy or sell at the same time, the worse price you get. There is such a concept as a [price stack](https://www.mql5.com/en/docs/constants/tradingconstants/enum_book_type "MQL5 Documentation: Types of orders in the price stack"). Usually, the larger the volume, the higher the price difference between them. And the smallest volume is traded next to each other. This trifle is displayed in ticks, which the author analysed so scrupulously for some reason. But this trifle is not the market itself. That is why large investment funds do not buy the whole volume in one deal. For a big fish to enter the market, it is necessary to work, and usually not one day. The largest players start to buy the asset in small amounts from those who trade close to each other, gaining a position sequentially. And they also have to exit sequentially. And if they are on fire, there is a collapse or a sharp rise due to a quick exit at any acceptable prices. So the big players often have to pay more in market costs. So they always play on a week to quarter scale.

![Sergey Pavlov](https://c.mql5.com/avatar/2010/2/4B7AECD8-6F67.jpg)

**[Sergey Pavlov](https://www.mql5.com/en/users/dc2008)**
\|
19 Oct 2021 at 09:31

**trens price stack. Usually, the larger the volume, the higher the price difference between them. And the smallest volume is traded next to each other. This trifle is displayed in ticks, which the author analysed so scrupulously for some reason. But this trifle is not the market itself. That is why large investment funds do not buy the whole volume in one deal. For a big fish to enter the market, it is necessary to work, and usually not one day. The largest players start to buy the asset in small amounts from those who trade close to each other, gaining a position sequentially. And they also have to exit sequentially. And if they are on fire, there is a collapse or a sharp rise due to a quick exit at any acceptable prices. So the big players often have to pay more in market costs. So they always play on a week to quarter scale.**

Right on!


![Dmitiry Ananiev](https://c.mql5.com/avatar/2021/5/60A1913E-6AF5.jpg)

**[Dmitiry Ananiev](https://www.mql5.com/en/users/dimeon)**
\|
23 Oct 2021 at 18:49

And they pay $200 for an article like this ? Very controversial philosophy.

I and a number of signals - different. Based on the fact that we buy at the ask price, at the time of purchase we are interested only in this price. At closing it is actually a sell transaction, so we are interested only in the bid price. It does not matter what prices were, what spreads. If the price is good, we should trade in the appropriate direction.

Example: Usually in Rollover spreads are very wide and many pairs are flat. At one point you see that the Ask price is very much below the formed corridor. We buy knowing that the bid price will return to the top of the corridor when the spreads recover. And at the moment of buying we are not interested in Bid price at all.

You will see similar strategies in the signals service, where trading is carried out from 23 to 1 am. By the way, they are among the most stable. And even the increased spread does not prevent to earn steadily.

This is how the statement that it is not necessary to trade on the increased spread is crossed out in practice.

![Fast235](https://c.mql5.com/avatar/2019/11/5DDBA4DA-BF3F.png)

**[Fast235](https://www.mql5.com/en/users/igann)**
\|
23 Oct 2021 at 20:27

Dimitri,


![architecmt4](https://c.mql5.com/avatar/avatar_na2.png)

**[architecmt4](https://www.mql5.com/en/users/architecmt4)**
\|
14 Feb 2024 at 09:17

Hello Mr. [Paul Kelly](https://www.mql5.com/en/users/wrc_dba), what's the difference between a "pip" and a "point"?


![Exploring options for creating multicolored candlesticks](https://c.mql5.com/2/43/multicolored-candlesticks.png)[Exploring options for creating multicolored candlesticks](https://www.mql5.com/en/articles/7815)

In this article I will address the possibilities of creating customized indicators with candlesticks, pointing out their advantages and disadvantages.

![Graphics in DoEasy library (Part 79): "Animation frame" object class and its descendant objects](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__6.png)[Graphics in DoEasy library (Part 79): "Animation frame" object class and its descendant objects](https://www.mql5.com/en/articles/9652)

In this article, I will develop the class of a single animation frame and its descendants. The class is to allow drawing shapes while maintaining and then restoring the background under them.

![Graphics in DoEasy library (Part 80): "Geometric animation frame" object class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__7.png)[Graphics in DoEasy library (Part 80): "Geometric animation frame" object class](https://www.mql5.com/en/articles/9689)

In this article, I will optimize the code of classes from the previous articles and create the geometric animation frame object class allowing us to draw regular polygons with a given number of vertices.

![Combinatorics and probability theory for trading (Part I): The basics](https://c.mql5.com/2/42/dj1f.png)[Combinatorics and probability theory for trading (Part I): The basics](https://www.mql5.com/en/articles/9456)

In this series of article, we will try to find a practical application of probability theory to describe trading and pricing processes. In the first article, we will look into the basics of combinatorics and probability, and will analyze the first example of how to apply fractals in the framework of the probability theory.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/9804&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069285324984615568)

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