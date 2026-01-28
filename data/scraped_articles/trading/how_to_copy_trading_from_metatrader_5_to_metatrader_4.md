---
title: How to Copy Trading from MetaTrader 5 to MetaTrader 4
url: https://www.mql5.com/en/articles/189
categories: Trading, Trading Systems, Integration, Expert Advisors
relevance_score: -2
scraped_at: 2026-01-24T14:16:39.370559
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/189&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083457789348551701)

MetaTrader 5 / Trading


### Introduction

Not so long ago, many traders believed MetaTrader 5 was a crude platform, unsuitable for real trading. But now, after a short time, the public opinion is wondering when the real trading will be available. Many traders have already appreciated the benefits implemented in MetaTrader 5. In addition, the [Championship](https://championship.mql5.com/2010/en) conducted by MetaQuotes Software Corp. increased interest of developers to the MQL5 language. And now this interesting wishes to be realized in the form of profiting from trading. The very question "When will the real trading be available?" is given to the wrong address. Its solution depends on the particular broker. It is they who make the final decision on the time of transition to the new platform.

What can a lone trader do in this situation? The answer is obvious, you should use the opportunities of real trading provided by MetaTrader 4, as a transit for MetaTrader 5. I.e. write a copyist. Binding between two MetaTrader 4 is not an innovation on the Web. Now it's time to implement such a binding with MetaTrader 5.

### Prologue

To implement the ideas stated in the topic, it is necessary to clarify the questions: "Where does the profit come from?" and "How can a trader control the growth of profits?" At first glance, the answers are obvious. Buy cheap, sell high. But let's consider the components of the profits. Profit is the difference of buying and selling, multiplied by the bet. That is, the profit has two components: quotes and the volume of a trading position.

What can the trader manage? Which of these two components is a steering wheel in trading? Of course it is the volume of a trade position. Quotes are received from the broker and cannot be changed by the trader. Here is the first conclusion: in order to make a copy of trading, it is necessary to keep synchronously the volume of trading positions.

### 1\. Comparison of the two platforms

**1.1 Differences in the accounting system**

The compared platforms have different trade accounting systems, and this can complicate the matter of copying. But we should not forget that the lead in this binding is MetaTrader 5. This means that in MetaTrader 4 we need to virtually repeat the same accounting system.

A trading position in MetaTrader 5 results from individual trade orders, which does not contradict the order accounting adopted in MetaTrader 4. The common Stop Loss and Take Profit for a position can be implemented through placing the same Stop Loss and Take Profit for every open order. Significant differences between the platforms appear only on the question of what order to close in MetaTrader 4. Since MetaTrader 5 does not have a separate accounting of orders in a trade position, this issue may become a stumbling block.

**1.2 Volumes of trade positions**

Let us consider in detail whether there is any difference which order to close. Will it affect profit? For example, we have two orders opened at different times, and closed the same way at different times, but having existed together for some time. That is, we try to emulate a trading position in the order accounting system.

Let's calculated different variants of what will happen to profit if we change the levels of the closing level of orders:

| Type | Volume | Open level | Close level |
| --- | --- | --- | --- |
| sell | 0.1 | 1.39388 | 1.38438 |
| sell | 0.1 | 1.38868 | 1.38149 |

Let's write a code of a calculator:

```
void OnStart()
  {
   double open1=1.39388,close1=1.38438,
         open2=1.38868,close2=1.38149;
   Print("total  ",n1(profit((open1-close1),0.1)+profit((open2-close2),0.1)));
   Print("order2  pp=",ns(open2-close2),"  profit=",n1(profit((open2-close2),0.1)));
   Print("order1  pp=",ns(open1-close1),"  profit=",n1(profit((open1-close1),0.1)));
  }
string ns(double v){return(DoubleToString(v,_Digits));}
string n1(double v){return(DoubleToString(v,1));}
double profit(double v,double lot){return(v/_Point*lot);}
```

Here is the calculation:

```
order1  pp=0.00950  profit=95.0
order2  pp=0.00719  profit=71.9
total  166.9
```

Now we swap the values of **close1** and **close2**.

```
order1  pp=0.01239  profit=123.9
order2  pp=0.00430  profit=43.0
total  166.9
```

![](https://c.mql5.com/2/2/2options.png)

Figure 1. Variants of order closing

Figure 1 shows that the areas AB and CD have in both versions volumes equal to 0.1, while BC has the volume of 0.2, and it does not depend on the fact the volume of which orders is closed.

In the profits of individual orders, we have differences, but the total profit of the positions are equal. I want to draw your attention to the fact that this example was calculated for equal volumes of orders. That is, at the same levels we have implemented closure of not an order, but of the same volume. And, if we strictly adhere to the principle of closing the volume, it does not matter what the volume of an order is. If the closing volume is larger than the order volume, there will be a partial closure.

So, here is the main conclusion: for the total position, it does not matter what order was closed, what's important is that the closed volumes must be equal at a given level.

**1.3 Copying trades**

From the above examples you see that to obtain the same profit, it is not necessary to transmit signals produced by the Expert Advisor written in MQL5. You only need to repeat the volumes of trading positions. Although, due to some reasons that will be discussed later, it will not be a completely identical trading. However, these reasons cannot be an obstacle to earning real profit using a profitable EA written in MQL5.

The first reason for the decline of profit is the difference in quotes. With some brokers it may exceed the spread. The fact is that the quotes are translated in real time. And when in MetaTrader 5 an EA decides to open a position, the broker to which MetaTrader 4 is connected can simply have a different price. And the price can be worse or better in this case.

The second reason for the decline of profit is the time factor. A position is copied after it has already appeared in MetaTrader 5, and hence the delay is inevitable.

These two reasons overshadow all scalping strategies. Therefore, until the real trading is available in MetaTrader 5, such strategies won't be applicable.

The system, implying a profit (from a trade), far exceeding the spread and insensitive to the quotes of a certain broker, can be used for making real money using the position copyist.

### 2\. Setting the problem

1. Binding to pass signals between MetaTrader 5 and MetaTrader 4
2. Translation of positions from MetaTrader 5
3. Receiving signals in MetaTrader 4
4. Repetition of trading positions in MetaTrader 4

**2.1. Binding to pass signals between MetaTrader 5 and MetaTrader 4**

The first variant is transmitting signals through a shared file. The question is: Wouldn't the frequent recording to the file harm the hardware? If you write new data only when a position has changed, then such records will not be very frequent. No more frequent than Windows developers make changes to the paging file. This, in turn, is a proven procedure that does not harm the hardware. Writing to a shared file is an acceptable implementation for not very frequent requests. This is another limitation for scalping strategies, although less significant than the previous limit.

Foe the binding, you may use the MetaTrader 5 feature of writing subdirectories to any depth. I have not checked that "any", but up to 10 subdirectories are written, that's for sure. And we don't need more. You can certainly use a DLL to provide access, but not to use the DLL, if the issue can be solved without them, is my principle position. To resolve this issue without DLL, simply install MetaTrader 4 to the directory \\Files\ of the MetaTrader 5 terminal (see [Working with files](https://www.mql5.com/en/docs/files)).

Thus, the path to the shared file will look like the following:

```
C:\Program Files\MetaTrader 5\MQL5\Files\MetaTrader 4\experts\files\xxx
//where xxx is the name of the shared file.
```

With this location, the file will be available both in MetaTrader 4 and MetaTrader 5, the possibility of file sharing is provided by the functions of MQL5.

**2.2. Translation of positions from MetaTrader 5**

For translating positions and economical use of resources, we need a function that will monitor the appearance/modification/closure of a trading position for all instruments. Above it was found that to transfer a trade we only need to know the volume of a trading position. Add to the volume the instrument symbol and the SL and TP levels.

To track the changes we need to know the previous state of a position. And if the previous and the present states are not equal (and hence the position has changed), it is necessary to display it in the file. We also need a function to write this information in the file. The file should be opened so that it is available for simultaneous use by multiple programs.

Not to miss the moment of position modification, the tracking system should be implemented in the [OnTimer()](https://www.mql5.com/en/docs/basis/function/events#ontimer) function, because we need to track all the instruments at once, and ticks come at different time for different symbols. Also we need to send a signal about the change in the file's contents.

**2.3. Receiving signals in MetaTrader 4**

It is necessary to track the signal of file update. This can be arranged through a variable whose state is monitored for the entrance to the zone of position change. We need a function for reading a file with the state of positions. It is just a standard function.

Passing the file contents into the arrays for calculation. Here we need the parser. Since not only numbers but also symbols will be passed, it is convenient to recode everything to string when transferring from MetaTrader 5. Besides, by writing all the data for one symbol into one text string, we eliminate data confusing.

**2.4. Repetition of trading positions in MetaTrader 4**

It is the largest set of functions. It should be divided into several subclasses.

1. Comparison of virtual positions;
2. Function of orders selection;
3. Function of order opening;
4. Function of order closing;
5. Function of order modification.

**2.4.1. Comparison of virtual positions**

Comparison of virtual positions is required for making sure that the positions are in conformity. The function should calculate the position for each symbol separately, and also be able to filter out positions for which trade is prohibited (if any).

In practice there may be situations when the broker does not have the symbol, a signal for which is passed from MetaTrader 5. But this should not block trading in general, although a warning should be provided. The user has the right to know about such a situation.

**2.4.2. Function of orders selection**

This function must choose the orders, depending on the symbol for further work with them. In this case, since we broadcast only the open positions, the orders must also be filtered so as not to have pending orders.

**2.4.3. Function of order opening**

It should contain the maximum number of calculations. Thus, it is enough to pass the volume and type.

**2.4.4. Function of order closing**

Just like the previous one, it should calculate everything before giving a command to close.

**2.4.5. Function of order modification**

The function should contain a check for nearness to the market. Also, it is desirable to dissolve over time placing of orders and stop levels, because placing of stop-levels during opening is not allowed with all brokers. In addition, the joint opening of orders and setting of stop-levels increases the likelihood of requoting.

Thus, the position will be quickly repeated. And placing of stop-levels is a minor thing, though no less important.

### 3\. Implementation

The codes are commented in detail, almost line by line. Therefore, when explaining the codes, I will dwell only on the most difficult moments.

**Binding to pass signals between MetaTrader 5 and MetaTrader 4**

The binding is implemented in MetaTrader 5 by the function:

```
void WriteFile(string folder="Translator positions") // by default it is the name of the shared file
```

The flags of opening mean:

```
FILE_WRITE|FILE_SHARE_READ|FILE_ANSI
```

file is open for writing \| shared use by different programs for reading is allowed \| ANSI encoding

In MetaTrader 4 the binding is implemented by the function:

```
int READS(string files,string &s[],bool resize)
```

The **resize** parameter prohibits redistributing the memory of the array of received data. In the code, memory for this array is allocated with each iteration, because a developer can't predict the number of lines. It depends on the number of symbols selected in MetaTrader 5. Therefore, it can't be calculated in advance in MetaTrader.

So, the array should be increased by one at each step. But this operation should be blocked at the second function call, because the length of the array is already defined and will not change. For this purpose use the variable **bool resize**.

**Translation of positions from MetaTrader 5**

To organize translation in the [OnTimer](https://www.mql5.com/en/docs/basis/function/events#ontimer) function with the frequency of 1 sec. data about all positions is received in the function:

```
void get_positions()
```

Then comparing the previous value of the positions with the current value of the function:

```
bool compare_positions()
```

And the exit with **return(true)** occurs in case at least one cell does not match. Exit with **return(true)** means that the positions are not equal and the file should be re-written. When re-writing the file, the counter **cnt\_command** is increased by one.

**Receiving signals in MetaTrader 4**

After reading the file using the **READS()** function, we have a filled array of strings **s\[\]**.

In order for these strings to turn into useful information, we need a parser.

The function:

```
int parser(int Size)
```

is just a wrapper for the call of line identification function:

```
void parser_string(int x)
```

Function recognizes all cells, except for symbols.

The symbol is recognized in a cycle, once at the beginning of an algorithm using the function:

```
void parser_string_Symbols(int x)
```

Next, we will not apply to the code in MQL5, and will discuss only the code in MQL4, unless it is specifically mentioned.

**Comparison of virtual positions**

Comparison of the positions is divided into two parts. Comparison of the volume and type of positions is implemented in the function:

```
bool compare_positions()
```

In this shell, call for getting the reals state of positions is implemented in the function:

```
void real_pos_volum()
```

and the comparison functions according to the above mentioned principle "all or nothing". It means, if at least one cell is not the same, all positions are considered different. In **real\_pos\_volum()** a number of filters are implemented, which are described in detail in the code and will be used repeatedly in other functions.

In particular, it will be used for summation of the volumes of all orders on one symbol into a virtual position. In order for the lock positions (if any) to be processed correctly, the Buy orders will have the volume with a minus, and the Sell orders with a plus.

The second part of the comparison is to compare the stop-levels (stop-levels are Stop Loss and Take Profit), it is implemented in the function, similar to the above one:

```
bool compare_sl_tp_levels()
```

Like with the volumes, inside the shell there is a call to get information about the stop-levels in the function:

```
void real_pos_sl_tp_levels()
```

**Function of orders selection**

Orders should be selected only for closing the volume, that's why the complicated specialized selection function is only implemented for closure:

```
void close_market_order(string symbol,double lot)
```

Has the parameters of the symbol and volume, which should be closed. In order break orders as little as possible, in the first cycle of the function it searches for the order whose volume is equal to the losing order passed in the parameter being sought a warrant, which is equal to the volume with the closing volume passed in the parameter.

If there is no such an order (which is known from the state **true** of the closure flag **FlagLot**), then the specified volume is closed in the order that is the first in the cycle (the check of the excess of the order volume is implemented in the close function **Closes()**).

Selection of orders for the modification of the stop levels is implemented in the function:

```
void modification_sl_tp_levels()
```

Orders are filtered only by the symbol, because all stop-levels within one symbol are equal.

**Function of order opening**

It is implemented in the following function:

```
int open_market_order(string symbol,int cmd,double volume,
                     int stoploss=0,int takeprofit=0,int magic=0)
```

It contains all the required checks for a comfortable opening of an order using the specified data.

**Function of order closing**

It is implemented in the following function:

```
bool Closes(string symbol,int ticket,double lot)
```

The code contains a check in case the **lot** parameter exceeds the real volume of the previously selected order.

**Function of order modification**

It is implemented in the following function:

```
bool OrderTradeModif(int ticket,string symbol,int cmd,double price,
                    double stoploss=0,double takeprofit=0,int magic=0)
```

The code has checks, in case stop-levels do not correspond to the type of order, the values will be exchanged. It also checks whether the levels already have the requested value.

### 4\. Functions of logic

The previously drawn plan is over, but the code still has some unexplained functions. They are the logic functions, we can say they are the basic functions, driving the process.

```
void processing_signals()
void processing_sl_tp_levels()
```

Both features are endless cycles with exit with the conditional **break**. Here we must note that the script itself is implemented as an infinite loop. To enable the user to comfortably remove the program, the main condition of the cycle has built-in [**IsStopped()**](https://docs.mql4.com/check/isstopped) function.

The code is transferred from an Expert Advisor to the looped script the following way:

```
// Init()
 while(!IsStopped())
    {
     // Start()
     Sleep(1000);
    }
 // Deinit()
```

The entire script logic is described in the same infinite loop in the standard function [**start**()](https://docs.mql4.com/runtime/running).

The code of the cycle located in **start**() will look like this:

```
If the trade flow is not busy
          Read the file and save data in an array (not changing the array size);
          if there have been changes in the file
               write new comments;
               remember the time when cycles of compliance check start (located below);
               if the positions whose volumes are being compared are not equal
                    process the positions by volumes;
               if the positions whose stops are being compared are not equal
                    process the positions by stops;
               calculate the end time of checks;
          If time is not exceeded
               make a pause for the remaining time;
```

The most complex logical constructions are located in functions **processing\_signals()** and **processing\_sl\_tp\_levels()**.

We begin describing the functions on the principle "from simple to complex." Although the call in the code is the opposite.

```
//+------------------------------------------------------------------+
//| processing stop levels                                           |
//+------------------------------------------------------------------+
void processing_sl_tp_levels()
  {
//--- remember the time of entering the cycle
   int start=GetTickCount();
   while(!IsStopped())
     {
      //--- if the trade flow is not busy
      if(Busy_and_Connected())
        {
         //--- select the order and modify stop levels
         modification_sl_tp_levels();
        }
      //--- if the delay time is over, update information from the file
      if(GetTickCount()-start>delay_time)READS("Translator positions",s,false);
      //--- if the update counter has changed in the file, exit the cycle
      if(cnt_command!=StrToInteger(s[0]))break;
      //--- micro-pause
      Sleep(50);
      //--- if real stop levels and those in the file are equal, exit the cycle
      if(!compare_sl_tp_levels())break;
     }
   return;
  }
```

As mentioned earlier, the function is an infinite loop with the exit on two conditions:

The first condition of exit from the loop occurs in case the value of **cnt\_command** is not equal to the same value in the file. Before that we receive the latest information about the file provided that the time of the loop operation exceeded the delay set in the global variable **delay\_time**.

The time may be exceeded because all modifications are protected by the filter **Busy\_and\_Connected()**. That is, enter only if the trade flow is free.

It should be explained here that in the MetaTrader 4 (in contrast to the MetaTrader 5) it is impossible to send a series of commands the server without having a requote. The server can only accept the first request, the rest will be lost. Therefore, before giving a command to the server, we need to check whether the trade flow is free.

The second check for exiting the cycle is the above described function of position comparison by stop levels **compare\_sl\_tp\_levels()**: if the positions are equal, then exit the cycle.

Now get to the complex: the **processing\_signals ()** function is organized in a similar way, but the logical part is very different in its functionality.

Let's analyze in details this part:

```
//--- convert the direction of the position stored in the file to the form -1,+1
int TF=SymPosType[i]*2-1;
//--- convert the direction of the real position to the form -1,+1
int TR=realSymPosType[i]*2-1;
//--- save the volume of the position stored in the file
double VF=SymPosVol[i];
//--- save the volume of the real position
double VR=realSymPosVol[i];
double lot;
//--- if the positions for the current symbol are nor equal
if(NormalizeDouble(VF*TF,8)!=NormalizeDouble(VR*TR,8))
  {
//--- if the real volume is not equal to zero and the directions are not equal or
//--- if the directions are equal and the real volume is larger than that in the file
   if((VR!=0 && TF!=TR) || (TF==TR && VF<VR))
     {
      //--- if the directions are equal and the real volume is larger than that in the file
      if(TF==TR && VF<VR)lot=realSymPosVol[i]-SymPosVol[i];
      //--- if the real volume is not equal to zero and the directions are not equal
      else lot=realSymPosVol[i];
      //--- close the calculated volume and exit the cycle
      close_market_order(Symbols[i],lot);
      break;
     }
   else
     {
      //--- if the directions are equal and the real volume is less than that in the file
      if(TF==TR && VF>VR)lot=SymPosVol[i]-realSymPosVol[i];
      //--- if the directions are not the same and the volume is equal to zero
      else lot=SymPosVol[i];
      //--- open the calculated volume and exit the cycle
      open_market_order(Symbols[i],SymPosType[i],lot);
      break;
     }
  }
```

The **TF** and **TR** variables store the value of the position type in the form of buy=-1,sell=1. Accordingly, **TF** is the value stored in the file, and **TR** is the real value of the virtual position. The same is with the volumes **VF**, **VR**.

Thus, the inequality:

```
if(VF*TF!=VR*TR)
```

will be **true** in case the volumes or position types are not equal.

Then comes the logical connective:

```
if((VR!=0 && TF!=TR) || (TF==TR && VF<VR))
```

which means that if the real volume is not equal to zero, and the types are not equal, then you should close the entire position.

This includes the options when the volume in the file is zero and the option when the position is reversed in direction. In the variant, when the position is reversed in direction, you must first prepare the position for opening, i.e. close the previous volume. Then with the next iteration, the logic goes to the other branch, to opening.

The second complex condition of the logical connective means that if the type is correct, but the real volume is more than that stored in the file, you should reduce the real volume. For this purpose, we first calculated the size of the lot by which it is necessary to reduce the volume.

If no closing condition is suitable for this situation, and the positions (as we discovered in the first filter) are not equal, a new order should be opened. Here are also two variants: to open an order for the entire size of the position in the file or add to existing orders. Here I'd like to note that the check for the excess of the limit volume is available in the function of opening, so the missing volume (if it was not possible with the check) will be opened at the next iteration of the algorithm. Due to the fact that first the close situation is handled, and only then - opening, the Lock situation is almost impossible.

I'd like to mention one subtle code place. The situation of reopening the order, which has just closed in MetaTrader 4 by stops. I've mentioned earlier that the discrepancy of quotes often is within the 2-3 points of 5-digit. With the spread equal to 15 the difference is insignificant. But with this difference, if stop loss or take profit triggered in MetaTrader 4 earlier than in MetaTrader 5, a situation appeared when the algorithm was trying to recreate the just closed position, with its subsequent removal with the stops triggered in MetaTrader 5.

It didn't bring to large losses, but one spread was wasted. Therefore, the algorithm has been redesigned so that after the removal of a position, MetaTrader 4 will not restore it, but will wait until the file state changes. And only then will start acting again. In this situation, the trader can remove the position manually, if he finds it wrong. And it will not be restored until MetaTrader 5 make changes to the file.

The only weak spot is the rare situation where the MetaTrader 4 stops will remove the position, and in MetaTrader 5 the position will not be closed. In this case, I can advise to restart script **Copyist positions**. And the last clause - the code does not check the work at the weekend. There's nothing serious, only the log will be full of worthless requotes.

### 5\. Checking the implementation in practice

Install MetaTrader 4 in the directory **C:\\Program Files\\MetaTrader 5\\MQL5\\Files\**

Run the compiled Expert Advisor **Translator positions** on any chart in MetaTrader 5 (the work of the Expert Advisor does not depend on the chart it is running on).

![](https://c.mql5.com/2/2/fig1__2.png)

Figure 2. Translator positions in MetaTrader 5

We see a multiline comment with the counter state on the first line and the log of all positions line by line.

Run the compiled script **Copyist positions** on any chart in MetaTrader 4 (the work of the looped script does not dependent on the chart, on which it is running).

![](https://c.mql5.com/2/2/fig3__1__1.png)

Figure 3. Copyist positions in MetaTrader 4

Then we can run any Expert Advisor in MetaTrader 5 . The results of its operation will be quickly copied to MetaTrader 4.

![](https://c.mql5.com/2/2/fig4__2.png)

Figure 4. Positions and orders in MetaTrader 4 (top) and MetaTrader 5 (bottom)

By the way, the account management in MetaTrader 5 can be carried out manually, or the account can be logged in using the investor password.

So, for example, you can start the copyist on any Championship account.

### Conclusion

This article is intended to accelerate the transition of traders to the new platform, and to encourage the study of MQL5.

In conclusion I would like to say that this code cannot fully replace direct trade on a real account in MetaTrader 5. It is written as a universal code for any trading system without taking into account the logic, therefore, like everything universal, it is not ideal. But based on it, you can write a translator of signals for a specific strategy. For many traders who are far from programming, it can serve as a transition stage in anticipation of the release.

To those who are well versed in programming, I recommend to modify the code to make it recognize orders by their magic number and implement the transfer and placing of pending orders. Placing of pending orders will not affect profit, provided there is stable connection to the server. If the connection loss happens often, all the server path, including pending orders, should be copied.

Learn the new language and use it to develop robust system. Good luck in your trading.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/189](https://www.mql5.com/ru/articles/189)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/189.zip "Download all attachments in the single ZIP archive")

[translator\_positions.mq5](https://www.mql5.com/en/articles/download/189/translator_positions.mq5 "Download translator_positions.mq5")(8.59 KB)

[copyist\_positions.mq4](https://www.mql5.com/en/articles/download/189/copyist_positions.mq4 "Download copyist_positions.mq4")(26.42 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Self-organizing feature maps (Kohonen maps) - revisiting the subject](https://www.mql5.com/en/articles/2043)
- [Debugging MQL5 Programs](https://www.mql5.com/en/articles/654)
- [The Player of Trading Based on Deal History](https://www.mql5.com/en/articles/242)
- [Electronic Tables in MQL5](https://www.mql5.com/en/articles/228)
- [Using Pseudo-Templates as Alternative to C++ Templates](https://www.mql5.com/en/articles/253)
- [Evaluation of Trade Systems - the Effectiveness of Entering, Exiting and Trades in General](https://www.mql5.com/en/articles/137)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3053)**
(161)


![Francis Fornari Passos](https://c.mql5.com/avatar/2017/9/59AD54E4-0F2B.jpg)

**[Francis Fornari Passos](https://www.mql5.com/en/users/francistbest)**
\|
17 Aug 2017 at 05:30

Can the reverse process be done? Copy orders from MT4 to MT5?

![Rodrigo Silva](https://c.mql5.com/avatar/2023/4/642f4df1-1782.png)

**[Rodrigo Silva](https://www.mql5.com/en/users/staetl_trader)**
\|
17 Jan 2018 at 12:59

here i am receiving error 5002


![erastogoncalves](https://c.mql5.com/avatar/2018/11/5BF1A566-0E31.jpg)

**[erastogoncalves](https://www.mql5.com/en/users/erastogoncalves)**
\|
24 Apr 2021 at 12:14

What part of the code should I change to make it possible to copy mt5 operations to mt5 itself?


![Daying Cao](https://c.mql5.com/avatar/2017/9/59C15CD0-3A22.png)

**[Daying Cao](https://www.mql5.com/en/users/cdymql4)**
\|
8 Nov 2022 at 11:16

Hi there!

I'm having a problem with MT5: I've installed it in C:\\Program Files\\Deriv.

My MT5 installation directory is C:\\Program Files\\Deriv, not **C:\\Program Files\\MetaTrader 5\\MQL5\\Files\** as you said.

Running the MT5 Translator positions\_\_1.ext5 is fine for storing files.

I have installed MT4 into the terminal directory of mt5: C:\\Users\\mac\\AppData\\Roaming\\MetaQuotes\\Terminal\\6AB79ED795024EC1B7F61552A87628BC\\MQL5\\Files\\MetaTrader 4 but the mt4 terminal Running script, can't find cvs file

The mt4 EA can only read the files under MQL4\\Files, I copied Translator positions.csv to this target and Copyist positions.ext4 works fine.

Can you please tell me how to solve this problem

[![](https://c.mql5.com/3/396/17yasssss.png)](https://c.mql5.com/3/396/1qoasssss.png "https://c.mql5.com/3/396/1qoasssss.png")

[![](https://c.mql5.com/3/396/pdaf_2022-11-08_y76.06.20.png)](https://c.mql5.com/3/396/0sl5_2022-11-08_a76.06.20.png "https://c.mql5.com/3/396/0sl5_2022-11-08_a76.06.20.png")

[![](https://c.mql5.com/3/396/4c55_2022-11-08_j36.04.58.png)](https://c.mql5.com/3/396/10gk_2022-11-08_hd6.04.58.png "https://c.mql5.com/3/396/10gk_2022-11-08_hd6.04.58.png")

![Daying Cao](https://c.mql5.com/avatar/2017/9/59C15CD0-3A22.png)

**[Daying Cao](https://www.mql5.com/en/users/cdymql4)**
\|
8 Nov 2022 at 15:20

Hi there.

I think you should use this directory: C:\\Users\\mac\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files to store Translator positions.csv

![Orders, Positions and Deals in MetaTrader 5](https://c.mql5.com/2/0/TradeIndo_MQL5.png)[Orders, Positions and Deals in MetaTrader 5](https://www.mql5.com/en/articles/211)

Creating a robust trading robot cannot be done without an understanding of the mechanisms of the MetaTrader 5 trading system. The client terminal receives the information about the positions, orders, and deals from the trading server. To handle this data properly using the MQL5, it's necessary to have a good understanding of the interaction between the MQL5-program and the client terminal.

![Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://c.mql5.com/2/0/MQL5_Mini-Max_Indicator.png)[Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

In the following article I am describing a process of implementing Moving Mini-Max indicator based on a paper by Z.G.Silagadze 'Moving Mini-max: a new indicator for technical analysis'. The idea of the indicator is based on simulation of quantum tunneling phenomena, proposed by G. Gamov in the theory of alpha decay.

![Exposing C# code to MQL5 using unmanaged exports](https://c.mql5.com/2/0/logo__5.png)[Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)

In this article I presented different methods of interaction between MQL5 code and managed C# code. I also provided several examples on how to marshal MQL5 structures against C# and how to invoke exported DLL functions in MQL5 scripts. I believe that the provided examples may serve as a basis for future research in writing DLLs in managed code. This article also open doors for MetaTrader to use many libraries that are already implemented in C#.

![Creating Multi-Expert Advisors on the basis of Trading Models](https://c.mql5.com/2/0/Multi_Expert_Advisor_MQL5__1.png)[Creating Multi-Expert Advisors on the basis of Trading Models](https://www.mql5.com/en/articles/217)

Using the object-oriented approach in MQL5 greatly simplifies the creation of multi-currency/multi-system /multi-time-frame Expert Advisors. Just imagine, your single EA trades on several dozens of trading strategies, on all of the available instruments, and on all of the possible time frames! In addition, the EA is easily tested in the tester, and for all of the strategies, included in its composition, it has one or several working systems of money management.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/189&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083457789348551701)

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