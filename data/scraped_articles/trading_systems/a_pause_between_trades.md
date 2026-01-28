---
title: A Pause between Trades
url: https://www.mql5.com/en/articles/1355
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:59:23.578883
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/1355&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083259911615289354)

MetaTrader 4 / Examples


### 1\. A Must or a Goodwill?

In the МetaТrader 3 Client Terminal, it was impossible to perform 2 trades at an interval under 10 seconds. Developing its МТ4, the MetaQuotes Software Corporation met the wishes of traders to remove this limitation. Indeed, there are situations where it is quite acceptable to perform a number of trades one after another (moving of StopLoss levels for several positions, removing of pending orders, etc.). But some traders took it in a wrong way and set about writing "killer" experts that could open positions one-by-one without a break. This resulted in blocked accounts or, at least, in the broker's unfriendly attitude.

This article is not for the above writers. It is for those who would like to make their trading comfortable for both themselves and their broker.

### 2\. One Expert or Several Experts: What Is the Difference?

If you have only one terminal launched and only one expert working on it, it is simplicity itself to arrange a pause between trades: it is sufficient to create a global variable (a variable declared at global level, not to be confused with the terminal's Global Variables) and store the time for the last operation in it. And, of course, you should check before performing of each trade operation, whether the time elapsed after the last attempt to trade is sufficient.

This will look like this:

```
datetime LastTradeTime = 0;
int start()
 {
  // check whether we should enter the market
  ...
  // calculate the levels of StopLoss and TakeProfit and the lot size
  ...
  // check whether sufficient time has elapsed after the last trade
  if(LocalTime() - LastTradeTime < 10)
   {
    Comment("Less than 10 seconds have elapsed after the last trade!",
            " The expert won't trade!");
    return(-1);
   }

  // open a position
  if(OrderSend(...) < 0)
   {
    Alert( "Error opening position # ", GetLastError() );
    return(-2);
   }

  // memorize the time of the last trade
  LastTradeTime = LocalTime();

  return(0);
 }
```

This example is suitable for one expert working on one terminal. If one or several more experts are launched simultaneously, they will not keep the 10-second pause. They will not just be aware of when another expert was trading. Every expert will have its own, local LastTradeTime variable. The way out of this is obvious: You should create a Global Variable and store the time of the trade in it. Here we mean the terminal's Global Variable, all experts will be able to access to it.

### 3\. The \_PauseBeforeTrade() Function

Since the code realizing the pause will be the same for all experts, it will be more reasonable to arrange it as a function. This will provide maximum usability and minimum volume of the expert code.

Prior to writing the code, let as define our task more concretely - this will save our time and efforts. Thus, this is what the function must do:

- **check whether a global variable has been created and, if not, create it**. It would be more logical and saving to do it from the expert's init() function, but then the would be a probability that the user will delete it and all experts working at the time will stop keeping the pause between trades. So we will place it in the function to be created;
- **memorize the current time in the global variable** in order for other experts to keep the pause;
- **check whether enough time has elapsed after the last trade**. For usability, it is also necessary to add an external variable that would set the necessary duration of the pause. Its value can be changed for each expert separately;
- **display information** about the process and about all the errors occurred during the work;
- **return different values depending on the performance results.**

If the function detects that not enough time has elapsed after the last trade, it must wait. The Sleep() function will provide both waiting and checking of the IsStopped(). I.e., if you delete the expert from the chart during it "sleeps", it will not hang up or be stopped forcedly.

But, for more self-descriptiveness, let us (every second during the "sleep") display information about how long it still remains to wait.

This is what we have to get as a result:

```
extern int PauseBeforeTrade = 10; // pause between trades (in seconds)

/////////////////////////////////////////////////////////////////////////////////
// int _PauseBeforeTrade()
//
// The function sets the local time value for the global variable LastTradeTime.
// If the local time value at the moment of launch is less than LastTradeTime +
// PauseBeforeTrade value, the function will wait.
// If there is no global variable LastTradeTime at all, the function will create it.
// Return codes:
//  1 - successful completion
// -1 - the expert was interrupted by the user (the expert was deleted from the chart,
//      the terminal was closed, the chart symbol or period was changed, etc.)
/////////////////////////////////////////////////////////////////////////////////
int _PauseBeforeTrade()
 {
  // there is no reason to keep the pause during testing - just terminate the function
  if(IsTesting())
    return(1);
  int _GetLastError = 0;
  int _LastTradeTime, RealPauseBeforeTrade;

  //+------------------------------------------------------------------+
  //| Check whether the global variable exists and, if not, create it  |
  //+------------------------------------------------------------------+
  while(true)
   {
    // if the expert was interrupted by the user, stop working
    if(IsStopped())
     {
      Print("The expert was stopped by the user!");
      return(-1);
     }
    // check whether a global variable exists
    // if yes, exit this loop
    if(GlobalVariableCheck("LastTradeTime"))
      break;
    else
     // if the GlobalVariableCheck returns FALSE, it means that either global variable does not exist, or
     // an error occurred during checking
     {
      _GetLastError = GetLastError();
      // if it is still an error, display information, wait during 0.1 sec, and start to
      // recheck
      if(_GetLastError != 0)
       {
        Print("_PauseBeforeTrade()-GlobalVariableCheck(\"LastTradeTime\")-Error #",
              _GetLastError );
        Sleep(100);
        continue;
       }
     }
    // if no error occurs, it just means that there is no global variable, try to create it
    // if GlobalVariableSet > 0, it means that the global variable has been successfully created.
    // Exit the function
    if(GlobalVariableSet("LastTradeTime", LocalTime() ) > 0)
      return(1);
    else
     // if GlobalVariableSet returns a value <= 0, it means that an error occurred during creation
     // of the variable
     {
      _GetLastError = GetLastError();
      // display information, wait 0.1 sec, and restart the attempt
      if(_GetLastError != 0)
       {
        Print("_PauseBeforeTrade()-GlobalVariableSet(\"LastTradeTime\", ",
              LocalTime(), ") - Error #", _GetLastError );
        Sleep(100);
        continue;
       }
     }
   }
  //+--------------------------------------------------------------------------------+
  //| If the function performance has reached this point, it means that the global   |
  //| variable exists.                                                                    |
  //| Wait until LocalTime() becomes > LastTradeTime + PauseBeforeTrade               |
  //+--------------------------------------------------------------------------------+
  while(true)
   {
    // if the expert was stopped by the user, stop working
    if(IsStopped())
     {
      Print("The expert was stopped by the user!");
      return(-1);
     }
    // get the value of the global variable
    _LastTradeTime = GlobalVariableGet("LastTradeTime");
    // if an error occurs at this, display information, wait 0.1 sec, and try
    // again
    _GetLastError = GetLastError();
    if(_GetLastError != 0)
     {
      Print("_PauseBeforeTrade()-GlobalVariableGet(\"LastTradeTime\")-Error #",
            _GetLastError );
      continue;
     }
    // count how many seconds have been elapsed since the last trade
    RealPauseBeforeTrade = LocalTime() - _LastTradeTime;
    // if less than PauseBeforeTrade seconds have elapsed,
    if(RealPauseBeforeTrade < PauseBeforeTrade)
     {
      // display information, wait one second, and check again
      Comment("Pause between trades. Remaining time: ",
               PauseBeforeTrade - RealPauseBeforeTrade, " sec" );
      Sleep(1000);
      continue;
     }
    // if the time elapsed exceeds PauseBeforeTrade seconds, stop the loop
    else
      break;
   }
  //+--------------------------------------------------------------------------------+
  //| If the function performance has reached this point, it means that the global   |
  //| variable exists and the local time exceeds LastTradeTime + PauseBeforeTrade    |
  //| Set the local time value for the global variable LastTradeTime                 |
  //+--------------------------------------------------------------------------------+
  while(true)
   {
    // if the expert was stopped by the user, stop working
    if(IsStopped())
     {
      Print("The expert was stopped by the user!");
      return(-1);
     }

    // Set the local time value for the global variable LastTradeTime.
    // In case it succeeds - exit
    if(GlobalVariableSet( "LastTradeTime", LocalTime() ) > 0)
     {
      Comment("");
      return(1);
     }
    else
    // if the GlobalVariableSet returns a value <= 0, it means that an error occurred
     {
      _GetLastError = GetLastError();
      // display the information, wait 0.1 sec, and restart attempt
      if(_GetLastError != 0)
       {
        Print("_PauseBeforeTrade()-GlobalVariableSet(\"LastTradeTime\", ",
              LocalTime(), " ) - Error #", _GetLastError );
        Sleep(100);
        continue;
       }
     }
   }
 }
```

### 4\. Integration into Experts and How to Use It

To check operability of the function, we created a diagnostic expert that had to trade keeping pauses between trades. The \_PauseBeforeTrade() function was preliminarily placed into the PauseBeforeTrade.mq4 file included into the expert using the #include directive.

Attention! This expert is only dedicated for checking the function operability! It may not be used for trading!

```
#include <PauseBeforeTrade.mq4>

int ticket = 0;
int start()
 {
  // if there is no position opened by this expert
  if(ticket <= 0)
   {
    // keep a pause between trades, if an error has occurred, exit
    if(_PauseBeforeTrade() < 0)
      return(-1);
    // update the market information
    RefreshRates();

    // and try to open a position
    ticket = OrderSend(Symbol(), OP_BUY, 0.1, Ask, 5, 0.0, 0.0, "PauseTest", 123, 0,
                       Lime);
    if(ticket < 0)
      Alert("Error OrderSend № ", GetLastError());
   }
  // if there is a position opened by this expert
  else
   {
    // keep the pause between trades (if an error has occurred, exit)
    if(_PauseBeforeTrade() < 0)
      return(-1);
    // update the market information
    RefreshRates();


    // and try to close the position
    if (!OrderClose( ticket, 0.1, Bid, 5, Lime ))
      Alert("Error OrderClose № ", GetLastError());
    else
      ticket = 0;
   }
  return(0);
 }
```

After that, one expert was attached to the EURUSD-M1 chart and another one, absolutely identical, to the GBPUSD-M1 chart. The result did not keep one waiting long: Both experts started trading keeping the prescribed 10-second pause between trades:

![](https://c.mql5.com/2/13/pausetest14285i29.gif)

![](https://c.mql5.com/2/13/pausetest25284929.gif)

![](https://c.mql5.com/2/13/pausetest3.gif)

### 5\. Possible Problems

When several experts work with one global variable, error can occur. To avoid this, we must delimit access to the variable. A working algorithm of such "delimitation" is described in details in the article named " [**Error 146 ("Trade context busy") and How to Deal with It**](https://www.mql5.com/en/articles/1412)". It is this algorithm that we will use.

The final version of the expert will look like this:

```
#include <PauseBeforeTrade.mq4>
#include <TradeContext.mq4>

int ticket = 0;
int start()
 {
  // if there is no a position opened by this expert
  if(ticket <= 0)
   {
    // wait until the trade is not busy and occupy it (if an error has occurred,
    // exit)
    if(TradeIsBusy() < 0)
      return(-1);
    // keep the pause between trades
    if(_PauseBeforeTrade() < 0)
     {
      // if an error has occurred, free the trade and exit
      TradeIsNotBusy();
      return(-1);
     }
    // update the market information
    RefreshRates();

    // and try to open a position
    ticket = OrderSend(Symbol(), OP_BUY, 0.1, Ask, 5, 0.0, 0.0, "PauseTest", 123, 0,
                       Lime);
    if (ticket < 0)
      Alert("Error OrderSend № ", GetLastError());
    // free the trade
    TradeIsNotBusy();
   }
  // if there is a position opened by this expert
  else
   {
    // wait until the trade is not busy and occupy it (if an error has occurred,
    // exit)
    if(TradeIsBusy() < 0)
      return(-1);
    // keep the pause between trades
    if(_PauseBeforeTrade() < 0)
     {
      // if an error occurs, free the trade and exit
      TradeIsNotBusy();
      return(-1);
     }
    // update the market information
    RefreshRates();

    // and try to close the position
    if(!OrderClose( ticket, 0.1, Bid, 5, Lime))
      Alert("Error OrderClose № ", GetLastError());
    else
      ticket = 0;

    // free the trade
    TradeIsNotBusy();
   }
  return(0);
 }
```

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1355](https://www.mql5.com/ru/articles/1355)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1355.zip "Download all attachments in the single ZIP archive")

[PauseBeforeTrade.mq4](https://www.mql5.com/en/articles/download/1355/PauseBeforeTrade.mq4 "Download PauseBeforeTrade.mq4")(5.74 KB)

[PauseTest\_expert.mq4](https://www.mql5.com/en/articles/download/1355/PauseTest_expert.mq4 "Download PauseTest_expert.mq4")(1.93 KB)

[TradeContext.mq4](https://www.mql5.com/en/articles/download/1355/TradeContext.mq4 "Download TradeContext.mq4")(9.27 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/39211)**
(6)


![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
26 Nov 2020 at 09:17

**Mancoba Message Malaza:**

How to install this

The article was written 14 years ago. You don't need this pause in 2020.

![Mancoba Message Malaza](https://c.mql5.com/avatar/2020/11/5FAFD189-ECD3.jpeg)

**[Mancoba Message Malaza](https://www.mql5.com/en/users/malaza1998)**
\|
9 Dec 2020 at 01:13

**Andrey Khatimlianskii:**

The article was written 14 years ago. You don't need this pause in 2020.

Why not?


![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
9 Dec 2020 at 08:33

**Mancoba Message Malaza:**

Why not?

Because MT server can accept a lot of orders every second.

If you need implement pause to your [trading algorithm](https://www.mql5.com/en/articles/8231 "Article: Scientific approach to the development of trading algorithms "), it is not hard. But you don't need this article.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
24 Dec 2020 at 15:48

**Andrey Khatimlianskii:**

Because MT server can accept a lot of orders every second.

If you need implement pause to your trading algorithm, it is not hard. But you don't need this article.

How can we implement this pause pls


![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
24 Dec 2020 at 21:17

**cosse:**

How can we implement this pause pls

You don't need this pause nowadays.

![Considering Orders in a Large Program](https://c.mql5.com/2/13/114_3.gif)[Considering Orders in a Large Program](https://www.mql5.com/en/articles/1390)

General principles of considering orders in a large and complex program are discussed.

![MagicNumber: "Magic" Identifier of the Order](https://c.mql5.com/2/13/105_2.gif)[MagicNumber: "Magic" Identifier of the Order](https://www.mql5.com/en/articles/1359)

The article deals with the problem of conflict-free trading of several experts on the same МТ 4 Client Terminal. It "teaches" the expert to manage only "its own" orders without modifying or closing "someone else's" positions (opened manually or by other experts). The article was written for users who have basic skills of working with the terminal and programming in MQL 4.

![My First "Grail"](https://c.mql5.com/2/13/144_2.png)[My First "Grail"](https://www.mql5.com/en/articles/1413)

Examined are the most frequent mistakes that lead the first-time programmers to creation of a "super-moneymaking" (when tested) trading systems. Exemplary experts that show fantastic results in tester, but result in losses during real trading are presented.

![How to Use Crashlogs to Debug Your Own DLLs](https://c.mql5.com/2/13/153_6.gif)[How to Use Crashlogs to Debug Your Own DLLs](https://www.mql5.com/en/articles/1414)

25 to 30% of all crashlogs received from users appear due to errors occurring when functions imported from custom dlls are executed.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qaxumsnrdqztljnriflcslgjqnktwyes&ssn=1769252362870940353&ssn_dr=0&ssn_sr=0&fv_date=1769252362&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1355&back_ref=https%3A%2F%2Fwww.google.com%2F&title=A%20Pause%20between%20Trades%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925236268788996&fz_uniq=5083259911615289354&sv=2552)

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