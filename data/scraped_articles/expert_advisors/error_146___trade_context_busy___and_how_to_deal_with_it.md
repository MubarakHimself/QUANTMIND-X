---
title: Error 146 ("Trade context busy") and How to Deal with It
url: https://www.mql5.com/en/articles/1412
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:13:43.954998
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=smquwuelbyoedxbpbblezynkllufzffi&ssn=1769253222750178231&ssn_dr=0&ssn_sr=0&fv_date=1769253222&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1412&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Error%20146%20(%22Trade%20context%20busy%22)%20and%20How%20to%20Deal%20with%20It%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925322287635479&fz_uniq=5083421234881895270&sv=2552)

MetaTrader 4 / Examples


### 1\. What Is "Trade Context" in Terms of MetaTrader 4 Client Terminal

Extract from MetaEditor Reference:

    To trade from experts and scripts, only one thread
was provided that was launched in the program trade context (context of
automated trading from experts and scripts). This is why, if this
context is occupied with an expert trading operation, another expert or
script cannot call trading functions at that moment due to error 146
(ERR\_TRADE\_CONTEXT\_BUSY).

Better to say, only
one expert (script) can trade at a time. All other experts that try to
start trading will be stopped by Error 146. This article will find
solutions for this problem.

### 2\. Function IsTradeAllowed()

The simplest way to find out whether the trade context is busy is to use the function named IsTradeAllowed().

Extract from MetaEditor Reference:

    "bool IsTradeAllowed()

    Returns true if the expert is allowed to trade and a thread for trading is not occupied, otherwise returns false.

This means that one can try to trade only if the IsTradeAllowed() function returns TRUE.

The check must be done just before a trade operation.

An example of **wrong** usage of the function:

```
int start()
  {
    // check whether the trade context is free
    if(!IsTradeAllowed())
      {
        // if the IsTradeAllowed() function has returned FALSE, inform the user about it,
        Print("Trade context is busy! The expert cannot open position!");
        // and terminate the expert operation. It will be restarted when the next tick
        // comes
        return(-1);
      }
    else
      {
        // if the IsTradeAllowed() function has returned TRUE, inform the user about it
        // and go on working
        Print("Trade context is free! We go on working...");
      }
    // check whether the market should be entered now
    ...
    // calculate Stop Loss and Take Profit levels and the lot size
    ...
    // open a position
    if(OrderSend(...) < 0)
        Alert("Error opening position # ", GetLastError());
    return(0);
  }
```

In this example, the trade context status is checked at the very
beginning of the start() function. It is a wrong idea: The trade
context can be occupied by another expert during the time taken by our
expert to calculate everything (the necessity to enter the market, Stop
Loss and Take Profit levels, lot size, etc.). In such case, the attempt
to open a position will not succeed.

An example of **proper** usage of the function:

```
int start()
  {
    // check whether the market should be entered now
    ...
    // calculate the Stop Loss and Take Profit levels, as well as the lot size
    ...
    // now let us check whether the trade context is free
    if(!IsTradeAllowed())
      {
        Print("Trade context is busy! The expert cannot open position!");
        return(-1);
      }
    else
        Print("Trade context is free! Trying to open position...");
    // if checking succeeded, open a position
    if(OrderSend(...) < 0)
        Alert("Error opening position # ", GetLastError());
    return(0);
  }
```

The trade context status is checked here **immediately**
before opening of position, and probability that another expert will
interpose between these two actions is much less. (It still exists,
though. This will be considered below.)

This method has two essential disadvantages:

- it is still probable that experts will check status simultaneously
and, having received positive results, will try to trade at the same
time
- if the checking fails, the expert will try to trade again only at the next tick; such delay is highly undesirable

The second problem can be
solved rather easily: one has just to wait until the trade context
becomes free. Then the expert will start trading immediately after the
other expert has finished it.

This will probably look like this:

```
int start()
  {
    // check whether the market should be entered now
    ...
    // calculate the Take Profit and Stop Loss levels, and the lot size
    ...
    // check whether the trade context is free
    if(!IsTradeAllowed())
      {
        Print("Trade context is busy! Wait until it is free...");
        // infinite loop
        while(true)
          {
            // if the expert was stopped by the user, stop operation
            if(IsStopped())
              {
                Print("The expert was stopped by the user!");
                return(-1);
              }
            // if trade context has become free, terminate the loop and start trading
            if(IsTradeAllowed())
              {
                Print("Trade context has become free!");
                break;
              }
            // if no loop breaking condition has been met, "wait" for 0.1 sec
            // and restart checking
            Sleep(100);
          }
      }
    else
        Print("Trade context is free! Trying to open a position...");
    // try to open a position
    if(OrderSend(...) < 0)
        Alert("Error opening position # ", GetLastError());
    return(0);
  }
```

In this current realization, we have some problem points again:

- since the IsTradeAllowed()
function is responsible not only for the trade context status, but also
for the enabling/disabling experts to trade, the expert can "hang" in
an infinite loop; it will then stop only if it is removed from the
chart manually
- if
the expert waits until the trade context is free, for just some
seconds, prices can change, and it will be impossible to trade using
them - the data should be refreshed, the open, Take Profit and Stop
Loss levels of the position to be opened should be recalculated

The corrected code will look like this:

```
// time (in seconds) whithin which the expert will wait until the trade
// context is free (if it is busy)
int MaxWaiting_sec = 30;
int start()
  {
    // check whether the market should be entered now
    ...
    // calculate the Stop Loss and Take Profit levels and the lot size
    ...
    // check whether the trade context is free
    if(!IsTradeAllowed())
      {
        int StartWaitingTime = GetTickCount();
        Print("Trade context is busy! Wait until it is free...");
        // infinite loop
        while(true)
          {
            // if the expert was terminated by the user, stop operation
            if(IsStopped())
              {
                Print("The expert was stopped by the user!");
                return(-1);
               }
            // if it is waited longer than it is specified in the variable named
            // MaxWaiting_sec, stop operation, as well
            if(GetTickCount() - StartWaitingTime > MaxWaiting_sec * 1000)
              {
                Print("The standby limit (" + MaxWaiting_sec + " sec) exceeded!");
                return(-2);
              }
            // if the trade context has become free,
            if(IsTradeAllowed())
              {
                Print("Trade context is free!");
                // refresh the market information
                RefreshRates();
                // recalculate the Stop Loss and Take Profit levels
                ...
                // leave the loop and start trading
                break;
              }
            // if no loop breaking condition has been met, "wait" for 0.1
            // second and then restart checking
            Sleep(100);
          }
      }
    else
        Print("Trade context is free! Trying to open a position...");

    // try to open a position
    if(OrderSend(...) < 0)
        Alert("Error opening position # ", GetLastError());

    return(0);
  }
```

In the above example, we added:

- refreshing of the market info (RefreshRates()) and consequent Stop Loss and Take Profit recalculation

- maximal time of waiting MaxWaiting\_sec, after exceeding of which the expert will stop operation

As such, the above code can be used in your experts already.

The final touch: Let us put
all concerning checking into a separate function. This will simplify
its integration in experts and its usage.

```
/////////////////////////////////////////////////////////////////////////////////
// int _IsTradeAllowed( int MaxWaiting_sec = 30 )
//
// the function checks the trade context status. Return codes:
//  1 - trade context is free, trade allowed
//  0 - trade context was busy, but became free. Trade is allowed only after
//      the market info has been refreshed.
// -1 - trade context is busy, waiting interrupted by the user (expert was removed from
//      the chart, terminal was shut down, the chart period and/or symbol was changed, etc.)
// -2 - trade context is busy, the waiting limit is reached (MaxWaiting_sec).
//      Possibly, the expert is not allowed to trade (checkbox "Allow live trading"
//      in the expert settings).
//
// MaxWaiting_sec - time (in seconds) within which the function will wait
// until the trade context is free (if it is busy). By default,30.
/////////////////////////////////////////////////////////////////////////////////
int _IsTradeAllowed(int MaxWaiting_sec = 30)
  {
    // check whether the trade context is free
    if(!IsTradeAllowed())
      {
        int StartWaitingTime = GetTickCount();
        Print("Trade context is busy! Wait until it is free...");
        // infinite loop
        while(true)
          {
            // if the expert was terminated by the user, stop operation
            if(IsStopped())
              {
                Print("The expert was terminated by the user!");
                return(-1);
              }
            // if the waiting time exceeds the time specified in the
            // MaxWaiting_sec variable, stop operation, as well
            if(GetTickCount() - StartWaitingTime > MaxWaiting_sec * 1000)
              {
                Print("The waiting limit exceeded (" + MaxWaiting_sec + " seconds)!");
                return(-2);
              }
            // if the trade context has become free,
            if(IsTradeAllowed())
              {
                Print("Trade context has become free!");
                return(0);
              }
              // if no loop breaking condition has been met, "wait" for 0.1
              // second and then restart checking
              Sleep(100);
          }
      }
    else
      {
        Print("Trade context is free!");
        return(1);
      }
  }
```

A template for the expert that uses the function:

```
int start()
  {
    // check whether the market should be entered now
    ...
    // calculate the Stop Loss and Take Profit levels, and lot size
    ...
    // check whether trade context is free
    int TradeAllow = _IsTradeAllowed();
    if(TradeAllow < 0)
      {
        return(-1);
      }
    if(TradeAllow == 0)
      {
        RefreshRates();
        // recalculate the Take Profit and Stop Loss levels
        ...
      }
    // open a position
    if(OrderSend(...) < 0)
        Alert("Error opening position # ", GetLastError());
    return(0);
  }
```

Let us draw some conclusions:

The IsTradeAllowed()
function is easy to use and ideally suits for differentiation of
accesses to trade context for two or three experts working
simultaneously. Due to some disadvantages thereof, its use does
not  ensure from Error 146 when many experts work simultaneously.
It can also cause "hanging" of the expert if the "Allow live trading"
is disabled.

This is why we will consider an alternative solution for this problem -a global variable as a "semaphore".

### 3\. Client Terminal Global Variables

First, the definition:

The client terminal global
variables are variables accessible to all experts, scripts and
indicators. This means a global variable created by one expert can be used in other experts
(in our case, to distribute accesses).

There are several functions provided in MQL 4 to work with global variables:

- GlobalVariableCheck() - to check whether a global variable exists

- GlobalVariableDel() - to delete a global variable

- GlobalVariableGet() - to get the value of the global variable

- GlobalVariableSet() - to create or modify a global variable

- GlobalVariableSetOnCondition() - to change the value of the
global variable specified by the user for another one. It differs from
GlobalVariableSet() in that the new value will be set only at a certain
previous value.
It is this function, which is a key function to create a semaphore.
- GlobalVariablesDeleteAll() - to delete all global variables (I cannot imagine who may need this:)

Why should the
GlobalVariableSetOnCondition() be used, but not the combination of
functions GlobalVariableGet() and GlobalVariableSet()? For the same
reasons: Some time can ellapse between uses of two functions. And
another expert can interpose into the semaphore switching. But this is
not what we need.

### 4\. The Basic Concept of Semaphore

Expert that is going to
trade should check the semaphore status. If the semaphore shows "red
light" (global variable = 1), it means that another expert is trading,
so it is necessary to wait. If it shows "green light"
(global variable = 0), the trading can be started immediately (but not
to forget to set the "red light" for other experts).

Thus, we have to create 2
functions: one for setting the "red light", another one for setting the
"green light". On the face of it, the task is simple. But we will not
jump to conclusions, but try to formulate the sequence of actions to be
executed by each function (let us name them TradeIsBusy() and
TradeIsNotBusy()) and finally realize them.

### 5\. Function TradeIsBusy()

As has been said before, the main task of the
function will be to wait until the "green light" appears and to switch
on the "red light".
Besides, we have to check whether the global variable exists, and
create it, if not. This checking would be more reasonable (and more
efficient) to perform from the init() function
of the expert.
But then a probability could exist that the user would delete it and no
one of working expert would be able to trade. This is why we will place
it in the body of the created function.

All this should be accompanied by information displaying and processing
of errors that have occurred when working with the global variable. The
"hanging" should not be forgotten, as well: The function operation time
should be limited.

This is what we will finally get:

```
/////////////////////////////////////////////////////////////////////////////////
// int TradeIsBusy( int MaxWaiting_sec = 30 )
//
// The function replaces the TradeIsBusy value 0 with 1.
// If TradeIsBusy = 1 at the moment of launch, the function waits until TradeIsBusy is 0,
// and then replaces.
// If there is no global variable TradeIsBusy, the function creates it.
// Return codes:
//  1 - successfully completed. The global variable TradeIsBusy was assigned with value 1
// -1 - TradeIsBusy = 1 at the moment of launch of the function, the waiting was interrupted by the user
//      (the expert was removed from the chart, the terminal was closed, the chart period and/or symbol
//      was changed, etc.)
// -2 - TradeIsBusy = 1 at the moment of launch of the function, the waiting limit was exceeded
//      (MaxWaiting_sec)
/////////////////////////////////////////////////////////////////////////////////
int TradeIsBusy( int MaxWaiting_sec = 30 )
  {
    // at testing, there is no reason to divide the trade context - just terminate
    // the function
    if(IsTesting())
        return(1);
    int _GetLastError = 0, StartWaitingTime = GetTickCount();
    //+------------------------------------------------------------------+
    //| Check whether a global variable exists and, if not, create it    |
    //+------------------------------------------------------------------+
    while(true)
      {
        // if the expert was terminated by the user, stop operation
        if(IsStopped())
          {
            Print("The expert was terminated by the user!");
            return(-1);
          }
        // if the waiting time exceeds that specified in the variable
        // MaxWaiting_sec, stop operation, as well
        if(GetTickCount() - StartWaitingTime > MaxWaiting_sec * 1000)
          {
            Print("Waiting time (" + MaxWaiting_sec + " sec) exceeded!");
            return(-2);
          }
        // check whether the global variable exists
        // if it does, leave the loop and go to the block of changing
        // TradeIsBusy value
        if(GlobalVariableCheck( "TradeIsBusy" ))
            break;
        else
        // if the GlobalVariableCheck returns FALSE, it means that it does not exist or
        // an error has occurred during checking
          {
            _GetLastError = GetLastError();
            // if it is still an error, display information, wait for 0.1 second, and
            // restart checking
            if(_GetLastError != 0)
             {
              Print("TradeIsBusy()-GlobalVariableCheck(\"TradeIsBusy\")-Error #",
                    _GetLastError );
              Sleep(100);
              continue;
             }
          }
        // if there is no error, it means that there is just no global variable, try to create
        // it
        // if the GlobalVariableSet > 0, it means that the global variable has been successfully created.
        // Leave the function
        if(GlobalVariableSet( "TradeIsBusy", 1.0 ) > 0 )
            return(1);
        else
        // if the GlobalVariableSet has returned a value <= 0, it means that an error
        // occurred at creation of the variable
         {
          _GetLastError = GetLastError();
          // display information, wait for 0.1 second, and try again
          if(_GetLastError != 0)
            {
              Print("TradeIsBusy()-GlobalVariableSet(\"TradeIsBusy\",0.0 )-Error #",
                    _GetLastError );
              Sleep(100);
              continue;
            }
         }
      }
    //+----------------------------------------------------------------------------------+
    //| If the function execution has reached this point, it means that global variable  |
    //| variable exists.                                                                 |
    //| Wait until the TradeIsBusy becomes = 0 and change the value of TradeIsBusy for 1 |
    //+----------------------------------------------------------------------------------+
    while(true)
     {
     // if the expert was terminated by the user, stop operation
     if(IsStopped())
       {
         Print("The expert was terminated by the user!");
         return(-1);
       }
     // if the waiting time exceeds that specified in the variable
     // MaxWaiting_sec, stop operation, as well
     if(GetTickCount() - StartWaitingTime > MaxWaiting_sec * 1000)
       {
         Print("The waiting time (" + MaxWaiting_sec + " sec) exceeded!");
         return(-2);
       }
     // try to change the value of the TradeIsBusy from 0 to 1
     // if succeed, leave the function returning 1 ("successfully completed")
     if(GlobalVariableSetOnCondition( "TradeIsBusy", 1.0, 0.0 ))
         return(1);
     else
     // if not, 2 reasons for it are possible: TradeIsBusy = 1 (then one has to wait), or

     // an error occurred (this is what we will check)
      {
      _GetLastError = GetLastError();
      // if it is still an error, display information and try again
      if(_GetLastError != 0)
      {
   Print("TradeIsBusy()-GlobalVariableSetOnCondition(\"TradeIsBusy\",1.0,0.0 )-Error #",
         _GetLastError );
       continue;
      }
     }
     //if there is no error, it means that TradeIsBusy = 1 (another expert is trading), then display
     // information and wait...
     Comment("Wait until another expert finishes trading...");
     Sleep(1000);
     Comment("");
    }
  }
```

Well, everything seems to be clear here:

- checking for whether the global variable exists and, if not, creation of it

- attempt to change the value of the global variable from 0 to 1; it will trigger only if its value is = 0.

The maximum amount of
seconds during which the function can work is MaxWaiting\_sec. The
function is no objection to deletion of the expert from the chart.

Information about all errors that occur is shown in the log.

### 6\. Function TradeIsNotBusy()

The TradeIsNotBusy function solves the inverse problem: It switches the "green light" on.

It is not limited by the
operation time and cannot be terminated by user. Motivation is rather
simple: If the "green light" is off, no expert will be able to trade.

It does not naturally have any return codes: The result can be only a successful completion.

This is how it looks:

```
/////////////////////////////////////////////////////////////////////////////////
// void TradeIsNotBusy()
//
// The function sets the value of the global variable TradeIsBusy = 0.
// If the TradeIsBusy does not exist, the function creates it.
/////////////////////////////////////////////////////////////////////////////////
void TradeIsNotBusy()
  {
    int _GetLastError;
    // at testing, there is no sense to divide the trade context - just terminate
    // the function
    if(IsTesting())
      {
        return(0);
      }
    while(true)
      {
        // if the expert was terminated by the user, stop working
        if(IsStopped())
          {
            Print("The expert was terminated by the user!");
            return(-1);
          }
        // try to set the global variable value = 0 (or create the global
        // variable)
        // if the GlobalVariableSet returns a value > 0, it means that everything
        // has succeeded. Leave the function
        if(GlobalVariableSet( "TradeIsBusy", 0.0 ) > 0)
            return(1);
        else
        // if the GlobalVariableSet returns a value <= 0, this means that an error has occurred.
        // Display information, wait, and try again
         {
         _GetLastError = GetLastError();
         if(_GetLastError != 0 )
           Print("TradeIsNotBusy()-GlobalVariableSet(\"TradeIsBusy\",0.0)-Error #",
                 _GetLastError );
         }
        Sleep(100);
      }
  }
```

### 7\. Integration into Experts and Use

Now we have 3 functions to
distribute access to the trading flow. To simplify their integration
into experts, we can create the TradeContext.mq4 file and enable it
using the #include directive (file attached).

Here is the template of the expert that uses functions TradeIsBusy() and TradeIsNotBusy():

```
#include <TradeContext.mq4>

int start()
  {
    // check whether the market should be entered now
    ...
    // calculate the StopLoss and TakeProfit levels, and the lot size
    ...
    // wait until the trade context is free and then occupy it (if an error occurs,
    // leave it)
    if(TradeIsBusy() < 0)
        return(-1);
    // refresh the market info
    RefreshRates();
    // recalculate the levels of StopLoss and TakeProfit
    ...
    // open a position
    if(OrderSend(...) < 0)
      {
        Alert("Error opening position # ", GetLastError());
      }

    // set the trade context free
    TradeIsNotBusy();

    return(0);
  }
```

In the use of functions
TradeIsBusy() and TradeIsNotBusy(), only one problem can occur: If the
expert is removed from the chart after the trade context has become
busy, the variable TradeIsBusy will remain equal to 1. Other experts
will not be able to trade after that.

The problem can be solved easily: The expert should not be removed from the chart when it is trading ;)

It is also possible that the
variable TradeIsBusy is not zeroized at disorderly close-down of the
terminal. In this case, the TradeIsNotBusy() function from the expert's
init() function can be used.

And, of course, the value of
the variable can be changed manually at any time: F3 button in the
terminal (it is an undocumented possibility to disable all experts to
trade).

komposter (komposterius@mail.ru), 2006.04.11

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1412](https://www.mql5.com/ru/articles/1412)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1412.zip "Download all attachments in the single ZIP archive")

[TradeContext.mq4](https://www.mql5.com/en/articles/download/1412/TradeContext.mq4 "Download TradeContext.mq4")(9.3 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/39203)**
(19)


![Florian Riedrich](https://c.mql5.com/avatar/2023/5/6471f647-94aa.jpg)

**[Florian Riedrich](https://www.mql5.com/en/users/1stdiablo-real1)**
\|
23 Aug 2019 at 11:21

Hey,

Iam try ing to use the 2 functions. But I don't get it to work

This is my test:

```
TradeIsNotBusy();

Print("Trade is Busy: " + TradeIsBusy());

TradeIsNotBusy(); // Entleeren den Trade-Context
```

The [Print function](https://www.mql5.com/en/docs/common/print "MQL5 documentation: Print function") returns: 1

But the Global Variable table returns: 0

Do I do something wrong?

![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
23 Aug 2019 at 14:21

**Florian Riedrich:**

Hey,

Iam try ing to use the 2 functions. But I don't get it to work

This is my test:

The [Print \\
function](https://www.mql5.com/en/docs/common/print "MQL5 documentation: Print function") returns: 1

But the Global Variable table returns: 0

Do I do something wrong?

```
TradeIsNotBusy();                                     // sets GV to 0

Print("Trade is Busy: " + TradeIsBusy());             // sets GV to 1, prints 1

TradeIsNotBusy(); // Entleeren den Trade-Context      // sets GV to 0
```

Here is explanation.

But you don't need to use these functions. There are 8 trade contexts in MT4 several years already.

![Florian Riedrich](https://c.mql5.com/avatar/2023/5/6471f647-94aa.jpg)

**[Florian Riedrich](https://www.mql5.com/en/users/1stdiablo-real1)**
\|
28 Aug 2019 at 16:56

Hey Andrey,

thanks for reply. So you mean using

```
IsConnected() && IsTradeAllowed() && !IsTradeContextBusy()
```

is enough?

What are the "8" contexts?

![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
2 Sep 2019 at 14:03

**Florian Riedrich:**

Hey Andrey,

thanks for reply. So you mean using

is enough?

What are the "8" contexts?

Terminal have 8 trade contexts now, so it can send as much as 8 trade requests simulateously.

If you use less then 8 EAs at the same time, you could even not to [check](https://www.mql5.com/en/docs/common/GetTickCount64 "MQL5 Documentation: GetTickCount64 function") IsTradeContextBusy(). Otherwise, just check it and have a little sleep in case it returns false ;)

![VikMorroHun](https://c.mql5.com/avatar/avatar_na2.png)

**[VikMorroHun](https://www.mql5.com/en/users/vikmorrohun)**
\|
5 Oct 2019 at 11:10

**Andrey Khatimlianskii:**

Terminal have 8 trade contexts now, so it can send as much as 8 trade requests simulateously.

If you use less then 8 EAs at the same time, you could even not to check IsTradeContextBusy(). Otherwise, just check it and have a
little sleep in case it returns false ;)

Wow, good to know. I've been fiddling with the incorporation of whrea\_v1.2's GetTradeContext()/RelTradeContext() [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") into my EA for
several hours... :)


![Working with Files. An Example of Important Market Events Visualization](https://c.mql5.com/2/13/112_2.gif)[Working with Files. An Example of Important Market Events Visualization](https://www.mql5.com/en/articles/1382)

The article deals with the outlook of using MQL4 for more productive work at FOREX markets.

![Information Storage and View](https://c.mql5.com/2/13/128_4.gif)[Information Storage and View](https://www.mql5.com/en/articles/1405)

The article deals with convenient and efficient methods of information storage and viewing. Alternatives to the terminal standard log file and the Comment() function are considered here.

![How to Use Crashlogs to Debug Your Own DLLs](https://c.mql5.com/2/13/153_6.gif)[How to Use Crashlogs to Debug Your Own DLLs](https://www.mql5.com/en/articles/1414)

25 to 30% of all crashlogs received from users appear due to errors occurring when functions imported from custom dlls are executed.

![Free-of-Holes Charts](https://c.mql5.com/2/13/130_1.png)[Free-of-Holes Charts](https://www.mql5.com/en/articles/1407)

The article deals with realization of charts without skipped bars.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/1412&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083421234881895270)

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