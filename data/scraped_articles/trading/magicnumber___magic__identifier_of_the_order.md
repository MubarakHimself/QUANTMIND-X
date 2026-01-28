---
title: MagicNumber: "Magic" Identifier of the Order
url: https://www.mql5.com/en/articles/1359
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:43:36.965965
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/1359&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083069782003029079)

MetaTrader 4 / Examples


### 1\. Preamble

In МТ3, management of open positions was rather time-taking. Traders had at their disposal a rather limited
tool set to work with the list of open and closed positions. The problem of distinguishing between "own"
and "someone else's" positions was solved in rather complicated ways. In МТ4, the situation has cardinally
changed. Now, trader can use a great variety of functions and fully manage all open positions and placed orders
and get access to information about any closed positions.

A special parameter named MagicNumber was added to identify orders. This is the
parameter our article will deal with.

### 2\. What Is MagicNumber?

MQL4 Reference:

**int OrderSend**( string symbol, int cmd, double volume, double price, int slippage, double stoploss,
double takeprofit, string comment=NULL, **int magic=0**, datetime expiration=0, color arrow\_color=CLR\_NONE)

…

magic - Order magic number. May be used as user defined identifier

I.e., when an order is being placed (a position is being opened), one can assign
a unique number to it. This number will consequently be used to distinguish the
above order from others. There is no use (or even possibility) applying this feature
when trading manually, but it is really unexpendable when trading using an expert
(automated trading).

**Example 1:** A human trader and an expert are trading in the client terminal at the same time.

**Task:** The expert must trade according to its algorithm and may not do anything with positions
opened manually.

**Solution:** The expert must assign a unique, non-zero MagicNumber to the position being opened.
In future, it must manage only positions, the MagicNumber of which is equal to
the preset one.

**Example 2:** Two experts with different algorithms are trading in the client terminal at the
same time.

**Task:** The expert must manage only "their" orders.

**Solution:** Each expert must use its unique non-zero MagicNumber when opening positions. In
future, they must manage only positions, the MagicNumber of which is equal to the
preset one.

**Example 3:** Several experts, a human trader and an assisting expert realizing a non-standard
Trailing Stop are operating in the client terminal simultaneously.

**Task:** Trading experts must work according to their algorithms and may not do anything
with positions opened manually. The assisting expert that realizes Trailing Stop
may modify only positions opened manually, but not those opened by other experts.

**Solution:** The trading experts must use unique MagicNumbers and manage only "their"
positions. The assisting expert must modify only those positions having MagicNumber
equal to 0.

All three examples are quite realistic, and the users could probably have set such
problems for themselves. In all three cases, the MagicNumber is used to solve it.
This way is not the unique one, but the easiest.

### 3\. Realization

Now let us solve the specific task: create an expert that could work only with its
"own" positions without paying attention to positions opened manually
or by other experts.

Let us first write a simple expert, for which the signal to open a position will
be when the MACD indicator meets zero line. The expert will look like this:

```
int start()
{
    //---- Remember the indicator's values for further analysis
    //---- Note that we use the 1st and the 2nd bar. This allows a 1-bar delay
    //---- (i.e., the signal will appear later), but protects against repeated opening and closing
    //---- of positions within a bar
    double MACD_1 = iMACD( Symbol(), 0, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 1 );
    double MACD_2 = iMACD( Symbol(), 0, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 2 );

    int _GetLastError = 0, _OrdersTotal = OrdersTotal();
    //---- search in all open positions
    for ( int z = _OrdersTotal - 1; z >= 0; z -- )
    {
        //---- if an error occurs at selecting of a position, go to the next one
        if ( !OrderSelect( z, SELECT_BY_POS ) )
        {
            _GetLastError = GetLastError();
            Print( "OrderSelect( ", z, ", SELECT_BY_POS ) - Error #", _GetLastError );
            continue;
        }

        //---- if the position was opened not for the current symbol, skip it
        if ( OrderSymbol() != Symbol() ) continue;

        //---- if the BUY position has been opened,
        if ( OrderType() == OP_BUY )
        {
            //---- if the MACD has met the zero line top-down,
            if ( NormalizeDouble( MACD_1, Digits + 1 ) <  0.0 &&
                  NormalizeDouble( MACD_2, Digits + 1 ) >= 0.0    )
            {
                //---- close the position
                if ( !OrderClose( OrderTicket(), OrderLots(), Bid, 5, Green ) )
                {
                    _GetLastError = GetLastError();
                    Alert( "Error OrderClose # ", _GetLastError );
                    return(-1);
                }
            }
            //---- if the alert has not been changed, exit: it is too early for opening a new position
            else return(0);
        }
        //---- if the SELL position has been opened,
        if ( OrderType() == OP_SELL )
        {
            //---- if the MACD has met the zero line bottom-up,
            if ( NormalizeDouble( MACD_1, Digits + 1 ) >  0.0 &&
                  NormalizeDouble( MACD_2, Digits + 1 ) <= 0.0    )
            {
                //---- close the position
                if ( !OrderClose( OrderTicket(), OrderLots(), Ask, 5, Red ) )
                {
                    _GetLastError = GetLastError();
                    Alert( "Error OrderClose # ", _GetLastError );
                    return(-1);
                }
            }
            //---- if the alert has not been changed, exit: it is too early for opening a new position
            else return(0);
        }
    }

//+------------------------------------------------------------------+
//| if execution reached this point, there is no open position       |
//| check whether it is still possible to open a position            |
//+------------------------------------------------------------------+

    //---- if the MACD has met the zero line bottom-up,
    if ( NormalizeDouble( MACD_1, Digits + 1 ) >  0.0 &&
          NormalizeDouble( MACD_2, Digits + 1 ) <= 0.0    )
    {
        //---- open a BUY position
        if ( OrderSend( Symbol(), OP_BUY, 0.1, Ask, 5, 0.0, 0.0, "MACD_test", 0, 0, Green ) < 0 )
        {
            _GetLastError = GetLastError();
            Alert( "Error OrderSend # ", _GetLastError );
            return(-1);
        }
        return(0);
    }
    //---- if the MACD has met the zero line top-down,
    if ( NormalizeDouble( MACD_1, Digits + 1 ) <  0.0 &&
          NormalizeDouble( MACD_2, Digits + 1 ) >= 0.0    )
    {
        //---- open a SELL position
        if ( OrderSend( Symbol(), OP_SELL, 0.1, Bid, 5, 0.0, 0.0, "MACD_test", 0, 0, Red ) < 0 )
        {
            _GetLastError = GetLastError();
            Alert( "Error OrderSend # ", _GetLastError );
            return(-1);
        }
        return(0);
    }

    return(0);
}
```

Let us attach it to the chart and see how it works:

![](https://c.mql5.com/2/13/macd_test1.gif)

![](https://c.mql5.com/2/13/macd_test1.gif)

Everything is ok, but there is one problem here. If we open a position during the
expert's operation, it will consider this position as its "own" and act
accordingly. This is not what we want.

We will modify our expert in such a way that it manages only its "own"
positions:

- Add the external variable named Expert\_ID to be used for changing the MagicNumber values for positions opened by the expert
- After the position has been selected by the OrderSelect() function, add checking
for whether the MagicNumber of the selected order complies with that of the Expert\_ID
variable
- We will write the value of the Expert\_ID instead of 0 into the MagicNumber field
during position opening

Considering the above changes, the code will appear as follows:

```
extern int Expert_ID = 1234;

int start()
{
    //---- Remember the indicator's values for further analysis
    //---- Note that we use the 1st and the 2nd bar. This allows a 1-bar delay
    //---- (i.e., the signal will appear later), but protects against repeated opening and closing
    //---- positions within a bar
    double MACD_1 = iMACD( Symbol(), 0, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 1 );
    double MACD_2 = iMACD( Symbol(), 0, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 2 );

    int _GetLastError = 0, _OrdersTotal = OrdersTotal();
    //---- search in all open positions
    for ( int z = _OrdersTotal - 1; z >= 0; z -- )
    {
        //---- if an error occurs when searching for a position, go to the next one
        if ( !OrderSelect( z, SELECT_BY_POS ) )
        {
            _GetLastError = GetLastError();
            Print( "OrderSelect( ", z, ", SELECT_BY_POS ) - Error #", _GetLastError );
            continue;
        }

        //---- if a position is closed not for the current symbol, skip it
        if ( OrderSymbol() != Symbol() ) continue;

        //---- if the MagicNumber is not equal to the Expert_ID, skip this position
        if ( OrderMagicNumber() != Expert_ID ) continue;

        //---- if a BUY position is opened,
        if ( OrderType() == OP_BUY )
        {
            //---- if the MACD has met the zero line top-down,
            if ( NormalizeDouble( MACD_1, Digits + 1 ) <  0.0 &&
                  NormalizeDouble( MACD_2, Digits + 1 ) >= 0.0    )
            {
                //---- close the position
                if ( !OrderClose( OrderTicket(), OrderLots(), Bid, 5, Green ) )
                {
                    _GetLastError = GetLastError();
                    Alert( "Error OrderClose # ", _GetLastError );
                    return(-1);
                }
            }
            //---- if the alert has not changed, exit: it is too early to open a new position
            else
            { return(0); }
        }
        //---- if a SELL position is opened,
        if ( OrderType() == OP_SELL )
        {
            //---- if the MACD has met the zero line bottom-up,
            if ( NormalizeDouble( MACD_1, Digits + 1 ) >  0.0 &&
                  NormalizeDouble( MACD_2, Digits + 1 ) <= 0.0    )
            {
                //---- close the position
                if ( !OrderClose( OrderTicket(), OrderLots(), Ask, 5, Red ) )
                {
                    _GetLastError = GetLastError();
                    Alert( "Error OrderClose № ", _GetLastError );
                    return(-1);
                }
            }
            //---- if the alert has not changed, exit: it is too early to open a new position
            else return(0);
        }
    }

//+------------------------------------------------------------------+
//| if execution reached this point, there is no an open position    |
//| check whether it is still possible to open a position            |
//+------------------------------------------------------------------+

    //---- if the MACD has met the zero line bottom-up,
    if ( NormalizeDouble( MACD_1, Digits + 1 ) >  0.0 &&
          NormalizeDouble( MACD_2, Digits + 1 ) <= 0.0    )
    {
        //---- open a BUY position
        if ( OrderSend( Symbol(), OP_BUY, 0.1, Ask, 5, 0.0, 0.0, "MACD_test",
              Expert_ID, 0, Green ) < 0 )
        {
            _GetLastError = GetLastError();
            Alert( "Error OrderSend # ", _GetLastError );
            return(-1);
        }
        return(0);
    }
    //---- if the MACD has met the zero line top-down,
    if ( NormalizeDouble( MACD_1, Digits + 1 ) <  0.0 &&
          NormalizeDouble( MACD_2, Digits + 1 ) >= 0.0    )
    {
        //---- open a SELL position
        if ( OrderSend( Symbol(), OP_SELL, 0.1, Bid, 5, 0.0, 0.0, "MACD_test",
              Expert_ID, 0, Red ) < 0 )
        {
            _GetLastError = GetLastError();
            Alert( "Error OrderSend # ", _GetLastError );
            return(-1);
        }
        return(0);
    }

    return(0);
}
```

Now, when the expert is working, the user can open positions manually. The expert
will not touch them.

### 4\. Multiple Identic Experts on Different Charts of One Symbol

There are cases where the same EA must trade on the charts of the same symbol, but
with different timeframes, for instance. If we try to attach our expert to the
chart EURUSD, H1, and to the EURUSD, M30, simultaneously, they will interfere each
other: each will "consider" the open position to be "its" position
and modify it at its discretion.

This problem can be solved by assigning another Expert\_ID to the other expert. But
this is not very convenient. If there are many experts used, one can just get entangled
among their IDs.

We can meet this problem using the chart period as MagicNumber. How shall we do
it? If we just add the chart period to the Expert\_ID, it is possible that 2 different
experts on 2 different charts generate the same MagicNumber.

So we will better multiply Expert\_ID by 10 and put the chart period (its code from
1 to 9, to be exact) at the end.

It will look something like this:

```
    int Period_ID = 0;
    switch ( Period() )
    {
        case PERIOD_MN1: Period_ID = 9; break;
        case PERIOD_W1:  Period_ID = 8; break;
        case PERIOD_D1:  Period_ID = 7; break;
        case PERIOD_H4:  Period_ID = 6; break;
        case PERIOD_H1:  Period_ID = 5; break;
        case PERIOD_M30: Period_ID = 4; break;
        case PERIOD_M15: Period_ID = 3; break;
        case PERIOD_M5:  Period_ID = 2; break;
        case PERIOD_M1:  Period_ID = 1; break;
    }
    _MagicNumber = Expert_ID * 10 + Period_ID;
```

Now add this code to the expert's init() function and replace Expert\_ID with \_MagicNumber
everywhere.

The final version of the EA looks like this:

```
extern int Expert_ID = 1234;
int _MagicNumber = 0;

int init()
{
    int Period_ID = 0;
    switch ( Period() )
    {
        case PERIOD_MN1: Period_ID = 9; break;
        case PERIOD_W1:  Period_ID = 8; break;
        case PERIOD_D1:  Period_ID = 7; break;
        case PERIOD_H4:  Period_ID = 6; break;
        case PERIOD_H1:  Period_ID = 5; break;
        case PERIOD_M30: Period_ID = 4; break;
        case PERIOD_M15: Period_ID = 3; break;
        case PERIOD_M5:  Period_ID = 2; break;
        case PERIOD_M1:  Period_ID = 1; break;
    }
    _MagicNumber = Expert_ID * 10 + Period_ID;

    return(0);
}

int start()
{
    //---- Remember the indicator's values for further analysis
    //---- Note that we use the 1st and the 2nd bar. This allows a 1-bar delay
    //---- (i.e., the signal will appear later), but protects against repeated opening and closing
    //---- positions within a bar
    double MACD_1 = iMACD( Symbol(), 0, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 1 );
    double MACD_2 = iMACD( Symbol(), 0, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 2 );

    int _GetLastError = 0, _OrdersTotal = OrdersTotal();
    //---- search in all open positions
    for ( int z = _OrdersTotal - 1; z >= 0; z -- )
    {
        //---- if an error occurs when searching for a position, go to the next one
        if ( !OrderSelect( z, SELECT_BY_POS ) )
        {
            _GetLastError = GetLastError();
            Print( "OrderSelect( ", z, ", SELECT_BY_POS ) - Error #", _GetLastError );
            continue;
        }

        //---- if a position is opened not for the current symbol, skip it
        if ( OrderSymbol() != Symbol() ) continue;

        //---- if the MagicNumber is not equal to _MagicNumber, skip this position
        if ( OrderMagicNumber() != _MagicNumber ) continue;

        //---- if a BUY position is opened,
        if ( OrderType() == OP_BUY )
        {
            //---- if the MACD has met the zero line top-down,
            if ( NormalizeDouble( MACD_1, Digits + 1 ) <  0.0 &&
                  NormalizeDouble( MACD_2, Digits + 1 ) >= 0.0    )
            {
                //---- close the position
                if ( !OrderClose( OrderTicket(), OrderLots(), Bid, 5, Green ) )
                {
                    _GetLastError = GetLastError();
                    Alert( "Error OrderClose # ", _GetLastError );
                    return(-1);
                }
            }
            //---- if the alert has not been changed, quit: it is too early to open a new position
            else return(0);
        }
        //---- if a SELL position is opened,
        if ( OrderType() == OP_SELL )
        {
            //---- if the MACD has met the zero line bottom-up,
            if ( NormalizeDouble( MACD_1, Digits + 1 ) >  0.0 &&
                  NormalizeDouble( MACD_2, Digits + 1 ) <= 0.0    )
            {
                //---- close the position
                if ( !OrderClose( OrderTicket(), OrderLots(), Ask, 5, Red ) )
                {
                    _GetLastError = GetLastError();
                    Alert( "Error OrderClose № ", _GetLastError );
                    return(-1);
                }
            }
            //---- if the alert has not changed, quit: it is too early to open a new position
            else return(0);
        }
    }

//+------------------------------------------------------------------+
//| if execution reached this point, there is no an open position    |
//| check whether it is still possible to open a position            |
//+------------------------------------------------------------------+

    //---- if the MACD has met the zero line bottom-up,
    if ( NormalizeDouble( MACD_1, Digits + 1 ) >  0.0 &&
          NormalizeDouble( MACD_2, Digits + 1 ) <= 0.0    )
    {
        //---- open a BUY position
        if ( OrderSend( Symbol(), OP_BUY, 0.1, Ask, 5, 0.0, 0.0, "MACD_test",
              _MagicNumber, 0, Green ) < 0 )
        {
            _GetLastError = GetLastError();
            Alert( "Error OrderSend # ", _GetLastError );
            return(-1);
        }
        return(0);
    }
    //---- if the MACD has met the zero line top-down,
    if ( NormalizeDouble( MACD_1, Digits + 1 ) <  0.0 &&
          NormalizeDouble( MACD_2, Digits + 1 ) >= 0.0    )
    {
        //---- open a SELL position
        if ( OrderSend( Symbol(), OP_SELL, 0.1, Bid, 5, 0.0, 0.0, "MACD_test",
              _MagicNumber, 0, Red ) < 0 )
        {
            _GetLastError = GetLastError();
            Alert( "Error OrderSend # ", _GetLastError );
            return(-1);
        }
        return(0);
    }

    return(0);
}
```

In such appearance, the expert can be used on several charts with different periods.

The Expert\_ID variable value will be to change only if there is a need to launch
two experts on charts of the same symbol and period (for example, EURUSD H1 and
EURUSD H4), but this happens extremely rarely.

Similarly, using the above code, the user can improve his or her EAs and "teach"
them to distinguish "their" positions from the "foreign" ones.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1359](https://www.mql5.com/ru/articles/1359)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235)
- [Equivolume Charting Revisited](https://www.mql5.com/en/articles/1504)
- [Testing Visualization: Account State Charts](https://www.mql5.com/en/articles/1487)
- [An Expert Advisor Made to Order. Manual for a Trader](https://www.mql5.com/en/articles/1460)
- [Testing Visualization: Trade History](https://www.mql5.com/en/articles/1452)
- [Sound Alerts in Indicators](https://www.mql5.com/en/articles/1448)
- [Filtering by History](https://www.mql5.com/en/articles/1441)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39206)**
(4)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
4 Sep 2006 at 17:47

Great article!


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
25 Dec 2006 at 02:02

Andrey

have you considered having the EA generate its own ID at start up and have the
ability to not only see its own orders but to be able to track each of its orders
even if it has 100 orders open. It would have to have the ability to recover its
ID or restates as well. With that you can create a [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") to be called in the
int() function at start up.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
2 Jul 2012 at 13:18

This article helps a lot...thank you!


![wainer](https://c.mql5.com/avatar/avatar_na2.png)

**[wainer](https://www.mql5.com/en/users/wainer)**
\|
27 Mar 2015 at 01:17

I need to know how to get the minimal magic number available.

**Example:** there are two open orders with magic numbers "1" and "3", so the minimal available is "2"

![A Pause between Trades](https://c.mql5.com/2/12/103_1.gif)[A Pause between Trades](https://www.mql5.com/en/articles/1355)

The article deals with the problem of how to arrange pauses between trade operations when a number of experts work on one МТ 4 Client Terminal. It is intended for users who have basic skills in both working with the terminal and programming in MQL 4.

![How to Use Crashlogs to Debug Your Own DLLs](https://c.mql5.com/2/13/153_6.gif)[How to Use Crashlogs to Debug Your Own DLLs](https://www.mql5.com/en/articles/1414)

25 to 30% of all crashlogs received from users appear due to errors occurring when functions imported from custom dlls are executed.

![Considering Orders in a Large Program](https://c.mql5.com/2/13/114_3.gif)[Considering Orders in a Large Program](https://www.mql5.com/en/articles/1390)

General principles of considering orders in a large and complex program are discussed.

![Working with Files. An Example of Important Market Events Visualization](https://c.mql5.com/2/13/112_2.gif)[Working with Files. An Example of Important Market Events Visualization](https://www.mql5.com/en/articles/1382)

The article deals with the outlook of using MQL4 for more productive work at FOREX markets.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kvimmraoidvujutvvrprkxsmpzvfuvqs&ssn=1769251416852268714&ssn_dr=0&ssn_sr=0&fv_date=1769251416&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1359&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MagicNumber%3A%20%22Magic%22%20Identifier%20of%20the%20Order%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925141610480194&fz_uniq=5083069782003029079&sv=2552)

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