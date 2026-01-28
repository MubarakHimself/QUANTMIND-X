---
title: Filtering by History
url: https://www.mql5.com/en/articles/1441
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:57:37.187648
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/1441&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083240378104027039)

MetaTrader 4 / Trading systems


### Introduction

There are different filters: values of indicators, market volatility, time, weekday.
They all can be used for sifting loss trades. It is quite easy to add such a filter
into an Expert Advisor - just add one more condition before the opening block.

But what should be done, if you want to use the EA history as a filter? If you
switch off your trade system after a number of unsuccessful trades, you will not
have a history later, and therefore there will be nothing to be analyzed. To solve
this problem we need to teach an Expert Advisor trade virtually, i.e. simulate
opening, modification and closing of trades without real trading activity.

This is what this article is about.

### Experimental Strategy

For the implementation of our system we will make some changes in the Expert Advisor
CrossMACD\_DeLuxe.mq4:

- at opening/modification/closing of each position, changes will be written in an
array of virtual positions;
- add tracking of the actuation of SropLoss and TakeProfit of virtual positions;
- add filtration criterion - a condition, upon which real trades will not be opened.


I will try to describe every step of EA modification in maximum details. If you
are not interested in it, you can download the ready Expert Advisor and move to the part "Is the game not worth the candle?".

### Accounting Virtual Positions

So, a signal to open a position appeared. The parameters StopLoss and TakeProfit
are calculated, everything is ready for calling the function OrderSend(). Exactly
in this moment we open the virtual trade - simply save all necessary parameters
in appropriate variables:

```
void OpenBuy()
  {
    int _GetLastError = 0;
    double _OpenPriceLevel, _StopLossLevel, _TakeProfitLevel;
    _OpenPriceLevel = NormalizeDouble(Ask, Digits);

    if(StopLoss > 0)
        _StopLossLevel = NormalizeDouble(_OpenPriceLevel -
                                     StopLoss*Point, Digits);
    else
        _StopLossLevel = 0.0;
    if(TakeProfit > 0)
        _TakeProfitLevel = NormalizeDouble(_OpenPriceLevel +
                                   TakeProfit*Point, Digits);
    else
        _TakeProfitLevel = 0.0;

    //---- open the virtual position
    virtOrderSend(OP_BUY, _OpenPriceLevel, _StopLossLevel,
                  _TakeProfitLevel);

    if(OrderSend(Symbol(), OP_BUY, 0.1, _OpenPriceLevel, 3,
       _StopLossLevel, _TakeProfitLevel, "CrossMACD",
       _MagicNumber, 0, Green) < 0)
      {
        _GetLastError = GetLastError();
        Alert("Error OrderSend № ", _GetLastError);
        return(-1);
      }
  }

//---- Save parameters of the opened position in main variables
void virtualOrderSend(int type, double openprice, double stoploss,
                      double takeprofit)
  {
    virtTicket = 1;
    virtType = type;
    virtOpenPrice = openprice;
    virtStopLoss = stoploss;
    virtTakeProfit = takeprofit;
  }
```

You see, we use only five variables:

```
int       virtTicket     = 0;
// determines, if there is an open virtual position
int       virtType       = 0;   // position type
double    virtOpenPrice  = 0.0; // position opening price
double    virtStopLoss   = 0.0; // position StopLoss
double    virtTakeProfit = 0.0; // position TakeProfit
```

We do not need other characteristics for the salvation of our task. If you want
to widen the functionality of this example, just add the necessary amount of variables.

For tracing position closing and modification we need to do more. Copy the block
of open positions controlling, which is in the Expert Advisor and change the order
characteristics into virtual ones:

```
int start()
  {
    // skipped...

    //+------------------------------------------------------------------+
    //| Control block of "virtual" positions                             |
    //+------------------------------------------------------------------+
    if(virtTicket > 0)
      {
        //---- if BUY-position is open,
        if(virtType == OP_BUY)
          {
            //---- if MACD crossed 0-line downwards,
            if(NormalizeDouble(MACD_1 + CloseLuft*Point*0.1,
               Digits + 1) <= 0.0)
              {
                //---- close position
                virtOrderClose(Bid);
              }
            //---- if the signal did not change, accompany the position by
            //     TrailingStop
            else
              {
                if(TrailingStop > 0)
                  {
                    if(NormalizeDouble(Bid - virtOpenPrice,
                       Digits ) > 0.0)
                      {
                        if(NormalizeDouble( Bid - TrailingStop*Point -
                           virtStopLoss, Digits) > 0.0 || virtStopLoss < Point)
                        {
                          virtStopLoss = Bid - TrailingStop*Point;
                        }
                      }
                  }
              }
          }
        //---- if SELL position is open
        if(virtType == OP_SELL)
          {
            //---- if MACD crossed 0-line upwards,
            if(NormalizeDouble(MACD_1 - CloseLuft*Point*0.1,
               Digits + 1 ) >= 0.0)
              {
                //---- close the position
                virtOrderClose(Ask);
              }
            //---- if the signal did not change, accompany the position by
            //     TrailingStop
            else
              {
                if ( TrailingStop > 0 )
                  {
                    if(NormalizeDouble( virtOpenPrice - Ask,
                       Digits ) > 0.0 )
                      {
                        if(NormalizeDouble( virtStopLoss - ( Ask +
                           TrailingStop*Point ), Digits ) > 0.0 ||
                           virtStopLoss <= Point )
                          {
                            virtStopLoss = Ask + TrailingStop*Point;
                          }
                      }
                  }
              }
          }
      }
    // skipped...
    return(0);
  }


//---- virtual position closing function
void virtOrderClose(double closeprice)
  {
    //---- Save the parameters of the closed position in the array
    ArrayResize(virtClosedOrders, virtClosedOrdersCount + 1);

    virtClosedOrders[virtClosedOrdersCount][0] = virtType;
    virtClosedOrders[virtClosedOrdersCount][1] = virtOpenPrice;
    virtClosedOrders[virtClosedOrdersCount][2] = virtStopLoss;
    virtClosedOrders[virtClosedOrdersCount][3] = virtTakeProfit;
    virtClosedOrders[virtClosedOrdersCount][4] = closeprice;

    virtClosedOrdersCount ++;

    //---- clear variables
    virtTicket = 0;
    virtType = 0;
    virtOpenPrice = 0.0;
    virtStopLoss = 0.0;
    virtTakeProfit = 0.0;
  }
```

You see, the modification turned into a simple assigning of a new value to the
variable virtStopLoss. And closing is quite difficult - all the characteristics
of the closed order are saved in an array. Later the whole virtual history will
be saved in it. From it we will take the information about the closed positions
for making a decision about opening of a new position.

Now we need to process the position closing upon StopLoss and TakeProfit. For this
purpose add several strings into the created control block:

```
if(virtType == OP_BUY)
  {
    //---- check, whether SL was not activated
    if(virtStopLoss > 0.0 && NormalizeDouble(virtStopLoss - Bid,
       Digits ) >= 0.0)
      {
        virtOrderClose(virtStopLoss);
      }
    //---- check, whether TPL was not activated
    if(virtTakeProfit > 0.0 && NormalizeDouble( Bid - virtTakeProfit,
       Digits ) >= 0.0)
      {
        virtOrderClose(virtTakeProfit);
      }
  }
```

Now our virtual history is ready and we can add a filtration criterion.

### What Is "Good" and What is "Bad"?

We need to prohibit position opening after a certain condition is implemented.
But what condition to choose? Several losing trades in succession, StopLoss activation
or reduced average profit of several last trades. It is hard to answer for sure
\- each variant may have its advantages and disadvantages.

To check the efficiency of each condition, let us try to code all three and test
them on the history:

```
extern int TradeFiltrVariant = 0;

//---- Function of checking the necessity of the real trading
bool virtCheckCondition()
  {
    int pos, check_pos = 2;
    double last_profit = 0.0, pre_last_profit = 0.0;

    //---- depending on the value of TradeFiltrVariant:
    switch(TradeFiltrVariant)
      {
        //---- 1: prohibit real trading, if 2 last deals are losing
        case 1:
          {
            //---- if the virtual history contains enough orders,
            if(virtClosedOrdersCount >= check_pos)
              {
                for(pos = 1; pos check_pos; pos ++)
                  {
                    //---- if the deal is profitable, return true
                    if((virtClosedOrders[virtClosedOrdersCount-pos][0] == 0 &&
                        virtClosedOrders[virtClosedOrdersCount-pos][4] -
                        virtClosedOrders[virtClosedOrdersCount-pos][1] >= 0.0) ||
                        (virtClosedOrders[virtClosedOrdersCount-pos][0] == 1 &&
                        virtClosedOrders[virtClosedOrdersCount-pos][1] -
                        virtClosedOrders[virtClosedOrdersCount-pos][4] >= 0.0))
                      {
                        return(true);
                      }
                    }
              }
            return(false);
          }
        //---- 2: prohibit real trading if the last position was closed
        //        by StopLoss
        case 2:
          {
            //---- if the virtual history contains enough orders,
            if(virtClosedOrdersCount > 0)
              {
                //---- if the closing price of the last order is equal to StopLoss,
                if(virtClosedOrders[virtClosedOrdersCount-1][2] -
                   virtClosedOrders[virtClosedOrdersCount-1][4] < Point &&
                   virtClosedOrders[virtClosedOrdersCount-1][4] -
                   virtClosedOrders[virtClosedOrdersCount-1][2] < Point)
                  {
                    return(false);
                  }
              }
            return(true);
          }
        //---- 3: prohibit real trading, if the profit of the last position
        //----    is lower than that of the last but one position (or loss is higher)
        case 3:
          {
            if(virtClosedOrdersCount >= 2)
              {
                if(virtClosedOrders[virtClosedOrdersCount-1][0] == 0)
                  {
                    last_profit =  virtClosedOrders[virtClosedOrdersCount-1][4] -
                                   virtClosedOrders[virtClosedOrdersCount-1][1];
                  }
                else
                  {
                    last_profit =  virtClosedOrders[virtClosedOrdersCount-1][1] -
                                   virtClosedOrders[virtClosedOrdersCount-1][4];
                  }
                if(virtClosedOrders[virtClosedOrdersCount-2][0] == 0)
                  {
                    pre_last_profit = virtClosedOrders[virtClosedOrdersCount-2][4] -
                                      virtClosedOrders[virtClosedOrdersCount-2][1];
                  }
                else
                  {
                    pre_last_profit = virtClosedOrders[virtClosedOrdersCount-2][1] -
                                      virtClosedOrders[virtClosedOrdersCount-2][4];
                  }

                if(pre_last_profit - last_profit > 0.0)
                  {
                    return(false);
                  }
              }
            return(true);
          }
        //---- by default the filter is off, i.e. positions will be always opened in reality
        default: return(true);
      }
    return(true);
  }

void OpenBuy()
  {
    int _GetLastError = 0;
    double _OpenPriceLevel, _StopLossLevel, _TakeProfitLevel;
    _OpenPriceLevel = NormalizeDouble(Ask, Digits);

    if(StopLoss > 0)
      {
        _StopLossLevel = NormalizeDouble(_OpenPriceLevel - StopLoss*Point, Digits);
      }
    else
      {
        _StopLossLevel = 0.0;
      }

    if(TakeProfit > 0)
      {
        _TakeProfitLevel = NormalizeDouble(_OpenPriceLevel + TakeProfit*Point, Digits);
      }
    else
      {
        _TakeProfitLevel = 0.0;
      }

    //---- open a virtual position
    virtOrderSend(OP_BUY, _OpenPriceLevel, _StopLossLevel, _TakeProfitLevel);

    //---- if virtual positions filter prohibits trading, exit
    if(virtCheckCondition() == false)
      {
        return(0);
      }

    if(OrderSend( Symbol(), OP_BUY, 0.1, _OpenPriceLevel, 3, _StopLossLevel,
          _TakeProfitLevel, "CrossMACD", _MagicNumber, 0, Green) < 0 )
      {
        _GetLastError = GetLastError();
        Alert("Error OrderSend № ", _GetLastError);
        return(-1);
      }
  }
```

You see, now we have an external variable TradeFiltrVariant. It is in charge of
choosing the filtration criterion:

```
extern int TradeFiltrVariant = 0;
//---- 0: filter is off, i.e. position is always opened in reality
//---- 1: prohibit real trading, if 2 last positions are losing
//---- 2: prohibit real trading, if the last position closed by StopLoss
//---- 3: prohibit real trading, if the profit of the last position is lower,
//----    than that of the last but one psition (or loss is higher)
```

Now test the Expert Advisor with different filters and compare results.

### Is the Game Not Worth the Candle?

For testing I chose the following parameters:

Symbol: GBPUSD

Period: H4, 01.01.2005 - 01.01.2006

Modeling mode: all ticks (modeling quality 90%, quotes by HistoryCenter)

EA parameters:

StopLoss: 50

TakeProfit: 0 (disabled)

TrailingStop: 0 (disabled)

FastEMAPeriod: 12

SlowEMAPeriod: 26

OpenLuft: 10

CloseLuft: 0

The following table shows the dependence of results on the filter used:

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **TradeFiltrVariant** | **Total profit/loss** | **Total trades** | **Profitable trades** | **Loss trades** |
| 0 | 1678.75 | 41 | 9 (22%) | 32 (78%) |
| 1 | 105.65 | 20 | 2 (10%) | 18 (90%) |
| 2 | -550.20 | 11 | 0 (0%) | 11 (100%) |
| 3 | 1225.13 | 28 | 7 (25%) | 21 (75%) |

You see, the theory about the usefulness of the filter was not proved out. Moreover,
the results of trades with the filter are lower than those of trades without the
filter. The only exception is the third variant - the rate of profitable trades
is higher (25 % against 22%), but the total profit is lower in all three variants.

Then what was wrong? Probably, the filtration criterion was wrong. Let us try to
change all three filters into contrary ones, i.e. close real trading if:

- two last trades are profitable;
- the last position is profitable (we do not have an analogue of StopLoss, because
TakeProfit is disabled);
- The profit of the last position is higher than that of the last but one position.


In order not to cut down the Expert Advisor, just add three more values of TradeFiltrVariant
\- 4, 5 and 6:

```
//---- 4: prohibit real trading, if two last trades are profitable
//---- 5: prohibit real trading, if the last position is profitable
//---- 6: prohibit real trading, if the profit of the last position is higher,
//----    than that of the last but one position (or loss is lower)

    //---- 4: prohibit real trading, if two last trades are profitable
    case 4:
      {
        if(virtClosedOrdersCount >= check_pos)
          {
            for(pos = 1; pos check_pos; pos ++)
              {
                //---- if the trade is losing, return true
                if((virtClosedOrders[virtClosedOrdersCount-pos][0] == 0 &&
                   virtClosedOrders[virtClosedOrdersCount-pos][1] -
                   virtClosedOrders[virtClosedOrdersCount-pos][4] > 0.0) ||
                   (virtClosedOrders[virtClosedOrdersCount-pos][0] == 1 &&
                   virtClosedOrders[virtClosedOrdersCount-pos][4] -
                   virtClosedOrders[virtClosedOrdersCount-pos][1] > 0.0))
                  {
                    return(true);
                  }
              }
          }
        return(false);
      }
    //---- 5: prohibit real trading, if the last position is profitable
    case 5:
      {
        if(virtClosedOrdersCount >= 1)
          {
            if(virtClosedOrders[virtClosedOrdersCount-1][0] == 0)
              {
                last_profit =  virtClosedOrders[virtClosedOrdersCount-1][4] -
                               virtClosedOrders[virtClosedOrdersCount-1][1];
              }
            else
              {
                last_profit =  virtClosedOrders[virtClosedOrdersCount-1][1] -
                               virtClosedOrders[virtClosedOrdersCount-1][4];
              }

            if(last_profit > 0.0)
              {
                return(false);
              }
          }
        return(true);
      }
    //---- 6: prohibit real trading, if the profit of the last position is higher,
    //----    than that of the last but one position (or loss is lower)
    case 6:
      {
        if(virtClosedOrdersCount >= 2)
          {
            if(virtClosedOrders[virtClosedOrdersCount-1][0] == 0)
              {
                last_profit =  virtClosedOrders[virtClosedOrdersCount-1][4] -
                               virtClosedOrders[virtClosedOrdersCount-1][1];
              }
            else
              {
                last_profit =  virtClosedOrders[virtClosedOrdersCount-1][1] -
                               virtClosedOrders[virtClosedOrdersCount-1][4];
              }
            if(virtClosedOrders[virtClosedOrdersCount-2][0] == 0)
              {
                pre_last_profit = virtClosedOrders[virtClosedOrdersCount-2][4] -
                                  virtClosedOrders[virtClosedOrdersCount-2][1];
              }
            else
              {
                pre_last_profit = virtClosedOrders[virtClosedOrdersCount-2][1] -
                                  virtClosedOrders[virtClosedOrdersCount-2][4];
              }

            if(last_profit - pre_last_profit > 0.0)
              {
                return(false);
              }
          }
        return(true);
      }
```

Now let us test three new variants and add them to our table:

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **AdaptVariant** | **Total profit/loss** | **Total trades** | **Profitable trades** | **Loss trades** |
| 0 | 1678.75 | 41 | 9 (22%) | 32 (78%) |
| 1 | 105.65 | 20 | 2 (10%) | 18 (90%) |
| 2 | -550.20 | 11 | 0 (0%) | 11 (100%) |
| 3 | 1225.13 | 28 | 7 (25%) | 21 (75%) |
| 4 | 1779.24 | 39 | 9 (23%) | 30 (77%) |
| **5** | **2178.95** | **31** | **9 (29%)** | **22 (71%)** |
| 6 | 602.32 | 24 | 5 (21%) | 19 (79%) |

The sixth variant filtered out half of trades - both profitable and loss ones.
The forth one sifted away two loss trades, increasing the total profit by 100.49$.

And the best one is the variant, prohibiting trading after each profitable trade
\- it filtered out 10 loss trades and no profitable trades.

So, there is a hope - even such a simple and popular strategy may be improved!

### Conclusion

I think, such filters are not enough for a real system improvement - much deeper
research must be conducted and new conclusions made.

The filters described may be more complex and absolutely different for each strategy.
Their efficiency directly depends on the interchange of profitable and losing positions.

This article dwells only on the mater of filtering. But I hope it will give an
impulse to further developments.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1441](https://www.mql5.com/ru/articles/1441)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1441.zip "Download all attachments in the single ZIP archive")

[CrossMACD\_TradeFiltr.mq4](https://www.mql5.com/en/articles/download/1441/crossmacd_tradefiltr.mq4 "Download CrossMACD_TradeFiltr.mq4")(41.09 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235)
- [Equivolume Charting Revisited](https://www.mql5.com/en/articles/1504)
- [Testing Visualization: Account State Charts](https://www.mql5.com/en/articles/1487)
- [An Expert Advisor Made to Order. Manual for a Trader](https://www.mql5.com/en/articles/1460)
- [Testing Visualization: Trade History](https://www.mql5.com/en/articles/1452)
- [Sound Alerts in Indicators](https://www.mql5.com/en/articles/1448)

**[Go to discussion](https://www.mql5.com/en/forum/39364)**

![Trading Using Linux](https://c.mql5.com/2/14/230_2.png)[Trading Using Linux](https://www.mql5.com/en/articles/1438)

The article describes how to use indicators to watch the situation on financial markets online.

![How Not to Fall into Optimization Traps?](https://c.mql5.com/2/14/218_2.png)[How Not to Fall into Optimization Traps?](https://www.mql5.com/en/articles/1434)

The article describes the methods of how to understand the tester optimization results better. It also gives some tips that help to avoid "harmful optimization".

![Break Through The Strategy Tester Limit On Testing Hedge EA](https://c.mql5.com/2/14/445_33.gif)[Break Through The Strategy Tester Limit On Testing Hedge EA](https://www.mql5.com/en/articles/1493)

An idea of testing the hedge Expert Advisors using the strategy tester.

![Universal Expert Advisor Template](https://c.mql5.com/2/14/451_25.gif)[Universal Expert Advisor Template](https://www.mql5.com/en/articles/1495)

The article will help newbies in trading to create flexibly adjustable Expert Advisors.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=klfcpjprvqulkhbtqxtsgmedwwanxvwb&ssn=1769252256257040771&ssn_dr=0&ssn_sr=0&fv_date=1769252256&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1441&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Filtering%20by%20History%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925225618478496&fz_uniq=5083240378104027039&sv=2552)

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