---
title: How to Make the Detection and Recovery of Errors in an Expert Advisor Code Easier
url: https://www.mql5.com/en/articles/1473
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:03:03.938560
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/1473&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071535308027144662)

MetaTrader 4 / Trading systems


### Introduction

The development of trading EAs in the MQL4 language is not an easy matter from the
points of view of several aspects:

- first, algorithmization of any more are less difficult trading system is already
a problem itself, because one needs to take into account many details, from the
peculiarities of an algorithmized EA and till the specific MetaTrader 4 environment;
- second, even the presence of further EA algorithm does not eliminate difficulties
that appear when transferring the developed algorithm into the programming language
MQL4.

We should do justice to the trading platform MetaTrader 4 - the existence of the
programming language for writing trading Expert Advisors is already a large step
forward as compared to earlier available alternatives. The compiler is of great
help in writing correct EAs. Immediately after clicking 'Compile' it will report
about all syntax errors in your code. If we dealt with an interpreted language,
such errors would be found only during EA's operation, and this would have increased
the difficulty and period of development. However, except syntax errors any EA
can contain logical errors. Now we will deal with such errors.

### Errors in Using Built-In Functions

While no trading EA can do without using built-in functions, let us try to ease
our life when analyzing errors, returned by such functions. First let us view the
operation results of functions, directly connected with trading operations, because
ignoring errors in such functions may lead to critical for an EA effects. However
further arguments refer also to other built-in functions.

Unfortunately using MQL4 options one cannot write a generalized library for processing
all possible error situations. In each separate case we need to process errors
separately. Still there is good news - we do not need to process all errors, many
of them are simply eliminated at the stage of EA development. But for doing this
they should be detected in time. As an example let us view 2 typical EA errors
in MQL4:

> 1) Error 130 - ERR\_INVALID\_STOPS
>
> 2) Error 146 - ERR\_TRADE\_CONTEXT\_BUSY

One of the cases, when the first error appears, is the attempt of an EA to place
a pending order too close to market. Its existence may seriously spoil the EA characteristics
in some cases. For example, suppose the EA, having opened a profitable position,
cuts profit every 150 points. If at the next attempt error 130 occurs and the price
returns to the previous stop level, this may deprive you of your legal profit.
Despite such possibility, this error can be eliminated from the very beginning,
inserting into the EA code the function of taking into account the minimally acceptable
distance between a price and stops.

The second error 'trade context is busy' cannot be fully eliminated. When several
EAs operate in one terminal we can face the situation, when one of the Expert Advisors
is trying to open a position, when the second one is doing the same. Consequently,
this error must be always processed.

So we always need to know, if any of built-in functions returns error during EA
operation. It can be achieved using the following simple additional function:

```
#include <stderror.mqh>
#include <stdlib.mqh>

void logError(string functionName, string msg, int errorCode = -1)
  {
    Print("ERROR: in " + functionName + "()");
    Print("ERROR: " + msg );

    int err = GetLastError();
    if(errorCode != -1)
        err = errorCode;

    if(err != ERR_NO_ERROR)
      {
        Print("ERROR: code=" + err + " - " + ErrorDescription( err ));
      }
  }
```

In a simplest case it should be used the following way:

```
void openLongTrade()
  {
    int ticket = OrderSend(Symbol(), OP_BUY, 1.0, Ask + 5, 5, 0, 0);
    if(ticket == -1)
        logError("openLongTrade", "could not open order");
  }
```

The first parameter of the function logError() shows the function name, in which
an error was detected, in our case it is the function openLongTrade(). If our Expert
Advisor calls the function OrderSend() in several places, we will have the opportunity
to determine exactly in which case the error occurred. The second parameter delivers
the error description, it enables to understand in which exact place inside the
function openLongTrade the error was detected. This can be a short description,
or a detailed one, including values of all parameters, passed into the built-in
function.

I prefer the last variant, because if an error appears, one can immediately get
all the information, necessary for analysis. Suppose, before calling OrderSend()
the current price greatly moved aside from the last known price. As a result an
error will occur and in the EA log file the following lines will appear:

```
 ERROR: in openLongTrade()
 ERROR: could not open order
 ERROR: code=138 - requote
```

I.e. it will be clear:

1. in what function the error occurred;
2. to what it refers (in our case to the attempt to open a position);
3. exactly what error appeared (error code and its description).

Now let us view the third, optional, parameter of the function logError(). It is
needed, when we want to process a certain error type, and other errors will be
included into a log file, as earlier:

```
void updateStopLoss(double newStopLoss)
  {
    bool modified = OrderModify(OrderTicket(), OrderOpenPrice(),
                newStopLoss, OrderTakeProfit(), OrderExpiration());

    if(!modified)
      {
        int errorCode = GetLastError();
        if(errorCode != ERR_NO_RESULT )
            logError("updateStopLoss", "failed to modify order", errorCode);
      }
  }
```

Here in the function updateStopLoss() the built-in function OrderModify() is called.
This function slightly differs from OrderSend() in terms of errors processing.
If no parameter of a changed order differs from its current parameters, the function
will return the error ERR\_NO\_RESULT. If such a situation is acceptable in our EA,
we should ignore this error. For this purpose we analyze the value, returned by
GetLastError(). If an error with the code ERR\_NO\_RESULT occurred, we do not write
anything into the log file.

However, if another error occurred, we should report about it as we did it earlier.
Exactly for this purpose we save the result of the function GetLastError() into
a intermediate variable and pass it using the third parameter into the function
logError(). Actually the built-in function GetLastError() automatically zeros the
last error code after it is called. If we do not pass the error code into logError(),
the EA log file would contain an error with the code 0 and description "no
error".

The similar actions should be done when processing other errors, for example, requotes.
The main idea is to process only errors, that need to be processed and to pass
other ones into the function logError(). In this case we will know, if an unexpected
error occurred during EA operation. After analyzing the log file, we will know
whether this error needs a separate processing or it can be eliminated improving
the EA code. Such approach makes our life easier and shortens time, spent for bug
fixing.

### Diagnosing Logic Errors

Logic errors in An Expert Advisor code may be trouble enough. The absence of EA
stepping-through option makes combating such errors a quite unpleasant task. The
main tool of diagnosing such errors at the present moment is the built-in function
Print(). Using it we may print current values of important variables and record
the EA operation flow. When debugging an EA during testing with visualization,
the built-in function Comment() can also be helpful. As a rule, when the wrong
work of an EA is confirmed, we have to add temporary calling of the function Print()
and record the EA inner state in the assumed places of error appearance.

But for the detection of difficult error situations we sometimes need to add dozens
of such diagnostic calls of the function Print(). And after the problem detection
and recovery, the function calls have to be deleted or commented, in order not
to overcharge the EA log file and not to make its testing slower. The situation
is even worse, if the EA code already includes the function Print() for periodic
recording of different states. Then the temporal Print() function calls cannot
be deleted by a simple search of 'Print' in EA code. One has to think whether to
delete a function, or not.

For example, when recording errors of functions OrderSend(), OrderModify() and OrderClose()
it is useful to print into a log file the current value of the variables Bid and
Ask. This makes finding reasons for such errors as ERR\_INVALID\_STOPS and ERR\_OFF\_QUOTES
easier.

For writing this diagnostic information into a log file, I recommend using the following
additional function:

```
void logInfo(string msg)
  {
    Print("INFO: " + msg);
  }
```

because:

- first, now such function will not be confused with 'Print' during search;
- second, this function has one more useful peculiarity, which will be discussed later.


It takes much time to add and delete temporary diagnostic calls of the function
Print(). That is why I suggest one more approach, which is efficient when detecting
logic errors in a code and helps saving our time. Let us analyze the following
simple function:

```
void openLongTrade(double stopLoss)
  {
    int ticket = OrderSend(Symbol(), OP_BUY, 1.0, Ask, 5, stopLoss, 0);
    if(ticket == -1)
        logError("openLongTrade", "could not open order");
  }
```

In this case, while we open a long position, it is clear, that in a correct EA operation
the value of he parameter stopLoss cannot be larger or equal to the current Bid
price. I.e. correctis the statement that when calling the function openLongTrade(),
the condition stopLoss < Bid is always fulfilled. As we know it already on the
stage of writing the analyzed function, we can use it the following way:

```
void openLongTrade( double stopLoss )
  {
    assert("openLongTrade", stopLoss < Bid, "stopLoss < Bid");

    int ticket = OrderSend(Symbol(), OP_BUY, 1.0, Ask, 5, stopLoss, 0);
    if(ticket == -1)
        logError("openLongTrade", "could not open order");
  }
```

I.e. we insert our statement into the code using the new additional function assert().
The function itself is quite simple:

```
void assert(string functionName, bool assertion, string description = "")
  {
    if(!assertion)
        Print("ASSERT: in " + functionName + "() - " + description);
  }
```

The first parameter of the function is the name of the function, in which our condition
is checked (by analogy with the function logError()). The second parameter shows
the results of this condition checking. And the third parameter denotes its description.
As a result, if an expected condition is not fulfilled, the EA log file will contain
the following information:

1. the name of the function, in which the condition was not fulfilled;
2. description of this condition.

As a description we may display, for example, the condition itself, or a more detailed
description, containing the values of the controlled variables at the moment of
checking the condition, if this can help to find the error causes.

Of course the given example is maximally simplified. But I hope it reflects the
idea quite clearly. As the EA functionality grows, we see clearly how it should
work and what conditions and input parameters are acceptable, and which ones are
not. Inserting it into an EA code using the function assert() we get a useful information
about the place, in which EA operation logic is broken. Moreover, we partially
eliminate the necessity to add and delete temporary calls of the function Print(),
because the function assert() generates diagnostic messages into the EA log file
only at the moment of detecting discrepancies in expected conditions.

One more useful method is using this function before each division operation. Actually,
this or that logic error can sometimes result in division by zero. In such cases
the Expert Advisor stops operating, and a single line appears in the log file:
'zero divide'. And if the division operation is used many times in the code, it
may be rather difficult to detect the place where the error occurred. And here
the function assert() may be very helpful. We simply need to insert the corresponding
checking before each division operation:

```
assert("buildChannel", distance > 0, "distance > 0");
double slope = delta / distance;
```

And now in case of a division by zero, we just look through the log file to find
where exactly the error appeared.

### Analyzing the EA Log File for Detecting Errors

The suggested functions for errors recording help to find them easily in the log
file. We just need to open the log file in text mode and search by words "ERROR:"
and "ASSERT:". However, sometimes we face situations, when during development
we omit the call results of this or that built-in function. Sometimes division
by zero is omitted. How can we detect messages about such errors among thousands
of lines, containing the information about opened, closed and changed positions?
If you face such a problem, I recommend you the following way out.

Open Microsoft Excel and download the EA operation log file as a CSV-file, indicating
that as a separator a space or several spaces should be used. Now enable "Autofilter".
This gives you a convenient opportunity to look through the filter options in two
neighboring columns (you will easily understand, what columns), to understand if
the log file contains errors, written into it by a terminal, or not. All entries,
generated by functions logInfo(), logError() and assert(), start with the prefix
("INFO:", "ERROR:" and "ASSERT:"). The entries of
the terminal about errors can also be easily seen among a few variants of typical
entries about working with orders.

This task can be solved in a more elegant way. For example if you are well acquainted
with tools of processing text files, you can write a small script, which will display
only those EA log file lines, which refer to EA operation errors. I strongly recommend
running each Expert Advisor through the tester on a large period and then analyze
the log file of its operation using the described methods. This allows detecting
the majority of possible errors before you run your EA on a demo-account. After
that the same from-time-to-time analyzing its operation log file will help you
detect in time errors, peculiar to EA operation only on accounts in real time.

### Conclusion

The described additional functions and simple methods allow simplifying the process
of errors detection and recovery in EA code, written in the programming language
MQL4. For your convenience the above described functions are included into attached
files. Simply add the file to your EA using the directory #include. Hope, the described
methods will be successfully used by traders, who worry about the robustness and
correctness of the Expert Advisors.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1473](https://www.mql5.com/ru/articles/1473)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1473.zip "Download all attachments in the single ZIP archive")

[debug.mqh](https://www.mql5.com/en/articles/download/1473/debug.mqh "Download debug.mqh")(1.07 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to Cut an EA Code for an Easier Life and Fewer Errors](https://www.mql5.com/en/articles/1491)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39372)**
(2)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
26 Sep 2008 at 12:40

is good.

nice article - more like this needed. assert is good. logging is good - black hole without. gives edge to coder and need all can get when coding/debug/runtime crazys ;)

i use assert long time - 4get where is in lots code, is good function param checker also, use all over, not trust self or mt ;) - is bit of magic when assert 'pops up' out of blue some times ;)

also take error logging step further - have one () for \*all\* mt [error codes](https://www.mql5.com/en/articles/70 "OOP in MQL5 by Example: Processing Warning and Error Codes"). only need call - it log to terminal and sends timestamped+callerLoc logfile entry - all use same msg format. is big lump code but now never mess with "msg..." anymore in mainline code. it return 'wat do next' suggest/status and caller make decision if retry,stop,refresh,abort,..

just ideas ok? not criticissms

nice read, yes.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
3 Dec 2009 at 17:47

Nice article.


![Indicator Alternative Ichimoku – Setup, Examples of Usage](https://c.mql5.com/2/14/339_13.jpg)[Indicator Alternative Ichimoku – Setup, Examples of Usage](https://www.mql5.com/en/articles/1469)

How to set up Alternative Ichimoku correctly? Read the description of parameters setting up. The article will help you understand the methods of setting up parameters not only of the indicator Ichimoku. Certainly you will also better understand how to set up the standard Ichimoku Kinko Hyo.

![Three-Dimensional Graphs - a Professional Tool of Market Analyzing](https://c.mql5.com/2/15/493_11.png)[Three-Dimensional Graphs - a Professional Tool of Market Analyzing](https://www.mql5.com/en/articles/1443)

In this article we will write a simple library for the construction of 3D graphs and their further viewing in Microsoft Excel. We will use standard MQL4 options to prepare and export data into \*.csv file.

![MetaTrader 4 Working under Antiviruses and Firewalls](https://c.mql5.com/2/14/295_1.gif)[MetaTrader 4 Working under Antiviruses and Firewalls](https://www.mql5.com/en/articles/1449)

The most of traders use special programs to protect their PCs. Unfortunately, these programs don't only protect computers against intrusions, viruses and Trojans, but also consume a significant amount of resources. This relates to network traffic, first of all, which is wholly controlled by various intelligent antiviruses and firewalls.
The reason for writing this article was that traders complained of slowed MetaTrader 4 Client Terminal when working with Outpost Firewall. We decided to make our own research using Kaspersky Antivirus 6.0 and Outpost Firewall Pro 4.0.

![Practical Application of Cluster Indicators in FOREX](https://c.mql5.com/2/14/352_5.gif)[Practical Application of Cluster Indicators in FOREX](https://www.mql5.com/en/articles/1472)

Cluster indicators are sets of indicators that divide currency pairs into separate currencies. Indicators allow to trace the relative currency fluctuation, determine the potential of forming new currency trends, receive trade signals and follow medium-term and long-term positions.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/1473&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071535308027144662)

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