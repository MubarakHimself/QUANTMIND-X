---
title: My First "Grail"
url: https://www.mql5.com/en/articles/1413
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:43:18.868996
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/1413&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083066341734224963)

MetaTrader 4 / Tester


_"... art cannot be programmed, two senses of poetry_

_cannot be bred. Talents cannot be_

_grown by checkrow_

_planting. They are born. They are national_

_wealth, like radium deposits,_

_like September in Sigulda, or Bethesda ..."_

_(Andrey Voznesenskiy)_

The Holy Grail is generally considered to be the cup from which Christ drank at the Last Supper and the one used by Joseph of Arimathea to catch his blood as he hung on the cross. Some writers present the grail as a stone which provides sustenance and prevents anyone who beholds it from dying within the week. You can read more about it [here](http://d.lib.rochester.edu/camelot/theme/holy-grail "http://d.lib.rochester.edu/camelot/theme/holy-grail") or use any search engines in internet.

The word "grail" is now often used among modern programmers ironically. It means for them the impossibility to create a "universal" program for all occasions. As to programming in MQL4, this word means impossibility to create an expert that would give fantastic effects in the real trading.

In reality, forex is the reflection of a complex conglomerate of phenomena - economic and industrial relations, human characters, politics. Moreover, and this is even more important, it cannot be simply formalized. Experienced traders recommend to enter the market only if there are three to five or even more signs indicating the possible trend.

At the same time, the regularities determined by now cannot completely provide a deep basis for market forecasting with high probability of success. The contradictory prognoses made by leading analysts of eminent banks and financial organizations confirm this. All analysts, without any exception, can very well interpret the events that have already happened, but only a few of them can give a sequence of really confident prognoses.

Let us be just towards them: These people do what they can do, most of them have a long trading experience and much knowledge we can envy. However, let us call things by their proper names: practically all of them are often mistaken. They can look big, enjoy more or less popularity, sometimes make a handsome fortune ("gurus" of different kinds are really well described in the Alexander Elder's book titled Trading for a Living: Psychology, Trading Tactics, Money Management), but the fact remains that even **experienced analysts are often mistaken**.

So, considering these circumstances, what are the chances of a first-time programmer who is just making his or her first steps in trading on Forex? Let us try to retrace the pathway that the beginner goes in his or her quest of the "Grail".

### 1\. What the "Grail" Comprises

In formal logic, alleging an authority is not considered as evidence. Knowing this, the first-time "grailer" reasons approximately in this way: "Can you prove that creation of "grail" is impossible? No? If no, then it is possible!". The beginner does not take into account that the possibility of creation such a thing has not been proven yet, either. And thus, without considering or often even without studying the experiences of other "gold diggers", but inspired by the thought of "I will be able!" based exclusively on his or her enthusiasm and non-initiation, he or she starts programming.

**1.1. Formal Strategy**

In most cases, the first-time programmer does not set him or herself the task of creating a super profitable trading strategy within a short space of time. Roused by the dream of large and fast profits on Forex, he or she, none the less relizes that a precise set of trading criteria is required fro a profitable expert.

To find good criteria, our programmer opens MetaTrader 4 Client Terminal and looks at the EURUSD chart on timeframe of M1. It is very easy to notice that the currency rate changes in waves: up and down, up and down. The programmer decides to gain profits on these waves. But, to "catch" a wave, one has to know somehow that the wave has stopped, for example, rising and started descending.

If one chooses the simple rate motion as the direction criterion, this will not result in anything since black and white candles replace each other very frequently and the range of those small changes is within the spread value or very near to it. Besides, it would be more preferable to enter the market at the wave peak, not on its slope. And waves are, unfortunately, of different heights. Having reflected a little, our programmer determines the following criteria:

- Consider that the market must be entered in the down direction (Sell) if the rate has moved up a certain amount of points, for example, 10 to 15 (movement A-B in Fig.1) and then stopped moving up and moved some points down (movement B-C in Fig.1), say, about 3 points. At this, the descending motion of the market is prognosed (C-D in Fig.1). The same criterion to be used for closing of the Buy.
- Consider that the market must be entered in the up direction (Buy) if the rate has moved down a certain amount of points, for example, 10 to 15 (movement C-D in Fig.1) and then stopped moving down and moved some points up (movement D-E in Fig.1) about the same 3 points. At this, the ascending motion of the market is prognosed (E-F in Fig.1). The same criterion to be used for closing of the Sell.


![](https://c.mql5.com/2/13/graal_pic_1_1.png)

**Fig.1. Trading Criteria for Expert Graal\_1 (Grail 1).**

Well, life prepares amazing and unexpected experiences for us at times: It took the newly-fledged programmer just 3 days to create his or her first "grail".

```
extern int TP=100; extern int SL=100; extern int lim=1; extern int prodvig=3;
extern double  Prots= 10;
int   total, bb=0,ss=0; double max,min,lmax,lmin,Lot;
int start(){
total=OrdersTotal(); if (total==0){bb=0;ss=0;}
if (max<Bid) max=Bid;if (min>Ask) min=Ask;
if (((max-Bid)>=lim*Point)&&(Bid>lmax )) { for (int i=total;i>=0;i--) {
if (OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==true && OrderType()==OP_BUY)
{OrderClose(OrderTicket(),OrderLots(),Bid,3,CLR_NONE); bb=0;}} Strateg(1); }
if (((Ask-min)>=lim*Point)&&(lmin>Ask )) { for (i=total;i>=0;i--) {
if (OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==true && OrderType()==OP_SELL)
{OrderClose(OrderTicket(),OrderLots(),Ask,3,CLR_NONE);ss=0;}}  Strateg(2);}return;}
void Strateg (int vv)
{if (vv==1 && ss==0)
{OrderSend(Symbol(),OP_SELL,Lots(),Bid,3,Bid+SL*Point,Bid-TP*Point,"",0,0,Red); ss=1;}
if (vv==2 && bb==0)
{OrderSend(Symbol(),OP_BUY, Lots(),Ask,3,Ask-SL*Point,Ask+TP*Point,"",0,0,Blue);bb=1;}
lmax=Ask+prodvig*Point; lmin=Bid-prodvig*Point;   return;  }
double Lots(){ Lot=NormalizeDouble(AccountEquity()*Prots/100/1000,1);
double Min_Lot = MarketInfo(Symbol(), MODE_MINLOT);
if (Lot==0 ) Lot=Min_Lot; return(Lot);  }
```

So, we have what we have. Let us be charitable - this is the first experience of the first-time programmer whose style has not been moulded yet. And let us respect this expert since it works and shows a really fantastic result.

Let us present the code as more readable and try to get to the bottom of what is what. After being a bit edited, this expert can look like this:

```
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
// Graal_1.mq4 (Grail 1).
// Used as an example in the article My First "Grail".
// Sergey Kovalyov, Dnepropetrovsk (Ukraine),sk@mail.dnepr.net,ICQ 64015987, http://autograf.dp.ua/
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
extern int     TP = 100;                                 // TakeProfit orders
extern int     SL = 100;                                 // StopLoss orders
extern int     lim=   1;                                 // Distance of the rate return
extern int     prodvig=3;                                // Distance of the rate progress
extern double  Prots= 10;                                 // Percentage of the liquid assets
//--------------------------------------------------------------------------------------------
int
   total,                                                // Count of lots
   bb=0,                                                 // 1 = the Buy order is available
   ss=0;                                                 // 1 = the Sell order is available
//--------------------------------------------------------------------------------------------
double
   max,                                                  // Maximum price at the peak (abs.)
   min,                                                  // Minimum price in the trough(abs.)
   lmax,                                                 // Limiting price after the exceeding
                                                         // of which we consider selling(abs.)
   lmin,                                                 // The same for buying
   Lot;                                                  // Count of lots
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int start()
   {
//============================================================================================
   total=OrdersTotal();                                  // Count of lots
   if (total==0)                                         // If there are no orders, ..
      {
      bb=0;                                              // .. no Buys
      ss=0;                                              // .. no Sells
      }
   if (max<Bid) max=Bid;                                 // Calculate the max. price at the peak
   if (min>Ask) min=Ask;                                 // Calculate the min. price in the trough
//------------------------------------------------------------- The price turns down ----
   if (((max-Bid)>=lim*Point)&&(Bid>lmax ))              // Turn at a high level
      {
      for (int i=total;i>=0;i--)                         // On all orders
         {
         if (OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==true && OrderType()==OP_BUY)
            {
            OrderClose(OrderTicket(),OrderLots(),Bid,3,CLR_NONE);// Close Buy
            bb=0;                                        // No Buys anymore
            }
         }
      Strateg(1);                                        // Opening function
      }
//------------------------------------------------------------ The price turns up ----
   if (((Ask-min)>=lim*Point)&&(lmin>Ask ))              // Turn at the deep bottom
      {
      for (i=total;i>=0;i--)                             // On all orders
         {
         if (OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==true && OrderType()==OP_SELL)
            {
            OrderClose(OrderTicket(),OrderLots(),Ask,3,CLR_NONE);// Close Sell
            ss=0;                                        // No Sells anymore
            }
         }
      Strateg(2);                                        // Opening function
      }
//============================================================================================
   return;

   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
void Strateg (int vv)                                    // Opening function
   {
//============================================================================================
   if (vv==1 && ss==0)                                   // Sell situation and no Sells
      {
      OrderSend(Symbol(),OP_SELL,Lots(),Bid,3,Bid+SL*Point,Bid-TP*Point,"",0,0,Red);// Open
      ss=1;                                              // Now, there is a Sell
      }
//--------------------------------------------------------------------------------------------
   if (vv==2 && bb==0)                                   // Buy situation and no Buys
      {
      OrderSend(Symbol(),OP_BUY, Lots(),Ask,3,Ask-SL*Point,Ask+TP*Point,"",0,0,Blue);// Open
      bb=1;                                              // Now, there is a Buy
      }
//--------------------------------------------------------------------------------------------
   lmax=Ask+prodvig*Point;                               // Redefine the new limiting ..
   lmin=Bid-prodvig*Point;                               // .. levels for open and close
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
double Lots()                                            // Calculation of lots
   {
//============================================================================================
   Lot=NormalizeDouble(AccountEquity()*Prots/100/1000,1);// Calculate the amoung of lots
   double Min_Lot = MarketInfo(Symbol(), MODE_MINLOT);   // Minimum permissible cost of lots
   if (Lot == 0 ) Lot = Min_Lot;                         // For testing on const.min.lots
//============================================================================================
   return(Lot);
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
/*
```

Generally, the contents of the expert is quite understandable.

In the upper part, before the start() function, variables are collected. In the start() function, the current position of the rate at the slope of a peak (trough) is calculated first, and then the situation is analyzed for triggering of trading criteria. And, if the criteria trigger, the order already processed will be closed. Two more functions are used in the expert - Strateg() to open new orders and Lots() to determine the amount of lots. Variables ss and bb are used to register opened orders.
Strategy Tester is a very important tool in the MetaTrader 4 Client Terminal. Our programmer has tested the expert carefully in order to optimize the inputs and one of the best results is as follows:

![](https://c.mql5.com/2/13/graal_pic_2.gif)

**Fig. 2. A Classical "Grail". The result was obtained at testing of the expert named Graal\_1.mq4 (Grail 1) on history data from March 2005 to June 2006 года for M1 EURUSD under trading conditions of the [MetaQuotes Software Corp](https://www.metaquotes.net/ "https://www.metaquotes.net/"). demo server**

We can easily imagine the programmer's joy: He or she created this work of art with own hands! Here it is: The new, quite substantial, sparkling with all its perfect facets, his or her first "grail"! And our programmer makes a simple conclusion: **"Only me!"**

It is not necessary that any formal strategy results in creation of a "grail", but, in our example, it is so. Let us closely investigate what brought the programmer to this result.

**1.2. Advanced Investment**

Our programmer has not stopped half-way and decided to try and maximize the values of orders. He or she this value (at testing) up to 70% (the value of the extern variable Prots=70) of the liquid assets, but this approach proved to be ineffective as a result:

![](https://c.mql5.com/2/13/graal_pic_3.gif)

**Fig. 3. Agressive Investments Can Result in Unjustified Losses**

Our hero was astonished at such a result very much. He or she reasoned like this: "It the expert is profitable, the more you invest in it, the more you get!". But Tester results show that it is not nearly often so.

Let us examine what caused such a result.

**1.2.1. Geometric Progression**

Let us have a look at the technology of order value calculation. According to the above code, the order value grows proportionally to the amount of liquid assets:

```
   Lot=NormalizeDouble(AccountEquity()*Prots/100/1000,1);// Calculate the amount of lots
   double Min_Lot = MarketInfo(Symbol(), MODE_MINLOT);   // Minimum allowed lot value
   if (Lot == 0 ) Lot = Min_Lot;                         // For testing on const.min.lots
```

This algorithm represents a **geometric progression**. The Prots (percentage) variable allows to regulate the order value depending on the current amount of liquid assets. This way to calculate the order values is not quite correct since it does not take into consideration the margin calls when working with a specific dealer. At the same time, the above code fragment allows solving the main problem - the problem of proportional investments. Thus, the value of each subsequent order will proportionally depend on the results obtained: the value will grow after each profitable trade and decrease after each unprofitable one. The order value in the above Graal\_1.mq4 (Grail 1) code makes 10% of the liquid assets.

The progression itself is not a necessary attribute of the "grail". Any other normal expert can be created using this technology. However, geometric progression must be fully excluded in order to investigate the "grail" more carefully, and focuse only on testing results obtained at constant minimum order value (equal to 0.1 lot, if possible). In this case, we can easily estimate the amount of gained points.

Let us set value of the Prots variable to 0 and, according to the code below,

```
   if (Lot == 0 ) Lot = Min_Lot;                         // For testing on const.min. lots
```

test the expert at the constant order value.

In this case, the result will be as follows:

![](https://c.mql5.com/2/13/graal_pic_4.png)

**Fig.4. Testing Results for Expert Graal\_1.mq4 (Grail 1) at Constant Lot Values**

Exclusion of preogressive order values visualizes the true nature of the balance curve changes - surges, deep drawdowns, etc.

**1.2.2. Agressive Reinvestment**

Any profitable expert will always hunt for its advantage between profit and loss. Currently, there is no a technology that would allow to perform **only** profitable trades. Losing trades are a norm for any really working trading strategy. The matter is just what the ratio is between profitable and losing operations.

For example, ratio 3:1 between profit and loss can be considered as quite acceptable. This means that 100 consecutively opened and closed orders must contain 75 profitable and 25 losing trades. At the same time, we can never predict how the losing orders will be distributed among the profitable ones. This distribution is predominantly of random nature.

**А.** Ideal distribution of Profits and Losses would be their **uniform** distribution throughout the entire trading history:

**P P P** **L** **P P P** **L** **P P P** **L** **P P P** **L** **P P P L P P P L P P P L ...**

However, there are no guarantees that the distribution will always be so ideal. Quite the opposite, it is highly probable that a long series of successive losses will occur sooner or later.

**В.** Below is an example of the very probable situations of **nonuniform** distribution of profitable and losing trades during real trading:

**P P P P P P P P P P P P P P P LLLL P P P L P P P L ...**

A series of 5 consecutive losses is shown above, though such a series can be even longer. You should note that, in this case, the ratio between profitable and losing orders is kept as 3 : 1.

So, what can this **regular randomness** result in? Answer to this question depends on quality of the trading strategy the trader has chosen. One thing is if the trader trades deliberately, observing reasonable precautions, where the total value of his or her orders does not exceed 10 to 15% of the deposit; and another thing is altogether if the trader allows **agressive reinvestments**: invests the most of his or her deposit, and only once, but every time when he or she gains profits.

In case of agressive reinvestments, the trading account history will develop incalculably. A rare "lucky" can probably escape troubles. But, in a number of cases, the negative result is really imminent. To illustrate possible scenarios of how the expert **Graal\_1.mq4** (Grail 1) will work, let us analyze two investment alternatives all other conditions being equal (the same phase of history, equally set inputs).

- **Alternative 1**. The value of an order is **10%** of liquid assets (Prots=10;). Under such conditions, the testing allows expert to achieve a good pace of work, increasing the balance deliberately, but permanently. You can see the results of such working in **Fig.2**. You can also notice in the Fig.2 that the expert has been working for a very long time (about 10 000 orders)

- **Alternative 2.** The value of an order is **70%** of liquid assets (Prots=70;). The testing results are given in **Fig.3**. Two or three series of successive losing orders resulted in that the deposit had become absolutely empty before the expert made 400 trades.


Please pay attention to the report about test results: The maximum amount of consecutive losses (a series of losing orders) for expert Graal\_1.mq4 (Grail 1) makes only 10. Other losing series took place during the entire testing period, but the amount of losing orders in each of them did not exceed 10. These losing series did not influence significantly the total result of trading in Alternative 1, but were disastrous in Alternative 2.

Thus, a short series of losses at wrong investments can result in full disaster. There is an effective system developed for proper capital management - [Money Management](https://www.mql5.com/go?link=http://www.kamas.ru/forex/forex/moneymanagement/?template=3 "http://www.kamas.ru/forex/forex/moneymanagement/?template=3"). According to this system, the value of one order should not exceed 15% of the total balance, and the total amount of investments should not exceed 50% of it. This system offers some more useful rules our programmer has read about.

He or she has drawn a conclusion: **trading using agressive reinvestments is a self-deception** and results in ruination sooner or later. Every trader will decide independently about where to place the limit. The results will be obtained accordingly.

Having thus "twiddled" his or her "grail" and made sure that everything can work well, our programmer opens a real account and starts to "make money". After a number of orders has been successfully opened and closed, the balance has turned out not to be growing, but to be steadily dropping. Moreover, the staff of the dealing center have disabled our programmer's expert at the end of day.
Our programmer feels really vexed. He or she desires the reasons for such an unfairness from dealers and disagrees to any explanations. He or she puzzles: What "technical limitaitons"? What has it to do with "slippage"?! As a result of such first experience in the real trading, our programmer has come to a conclusion that this is a "bad dealing center", that he or she was just cheated. He or she breaks the contract with the dealing center.

However, our programmer does not give up, decides to get into details of the process for certain, and contacts the more experienced colleagues on the specialized forum at [MQL4](https://www.mql5.com/en/forum/mql4 "https://www.mql5.com/en/forum/mql4") where he or she is first of all proposed to show the code. He or she grudges the code, but he or she finally agrees since the expert seems to be unprofitable, in any case... The programmer understands the following after the things have been discussed on the forum.

**1.3. Errors**

Meta Editor constantly confirmed that no errors had occurred,

![](https://c.mql5.com/2/13/graal_pic_5.gif)

**Fig. 5. No Errors When Compiling**

but they had, nevertheless.

**1.3.1. Digression from the Formal Strategy**

Though a quite knowledgeable strategy had underlain the expert creation, it was essentially changed during coding and optimization of generic variables. So, the initial idea is now presented in the "grail" really down-sized. Indeed, the idea was to earn on waves 15 to 20 points long, but this parameter takes miserable value of 3 in the present "grail" version. Who would talk peaks and troughs here...

There is no definite line between black and white. All intermediate states can be characterized by the coefficient of their whitness or blackness. But one usually can distinguish that a color is more black than white, or vice versa. In our case, the initial "purely white" idea was "blacken" so much that it became as black as a sweep and lost all its force. We can consider this to be the reason for the "grail" obtained.

**1.3.2. Incorrect Programming**

Note how the prders are considered in variables ss and bb. This is a wrong way to consider orders. The order is considered to be already opened when it is being formed. But one cannot know in advance whether the order will be opened. To be sure of this, one has to wait until the server replies and then analyze the availability of the order in the terminal. For the real trading, this part of the code should be rewritten, though the given modification will work in the most cases including in strategy tester (see article [Considering Orders in a Large Program](https://www.mql5.com/ru/articles/1390) ).

**1.4. Hardware Restrictions**

The expert was tested on the timeframe, for which it had been intended: on M1. One has to be very attentive when modeling experts' trading on M1 timeframe, especially for price-sensitive experts.

If we want to precise the history of a bar on a larger timesframe, we can use the bar history of a smaller timeframe. But for M1, there are no smaller timeframes since the tick history is not stored or used in the MetaTrader 4 Client Terminal (see [Strategy Tester: Modes of Modeling during Testing](https://www.mql5.com/en/articles/1511)).

The main defference between a test model and a real trading strategy lies in that test mode does not provide slippage and requoting (when the dealer gives another price, other than that in the trader's order).

Different dealing centers work with different datafeeds (to be considered later). The more and the more frequently changes the rate, the more probable it is that the dealer will propose the trader to execute the order at another price. On the other hand, the more "smoothed" looks the traffic, the less probable is the requote, but the less space is it for the expert that trades frequently and with small achievements.

Our programmer has tested Graal\_1 (Grail 1) on 1-minute history of another dealing center (with the same spread) and saw with the own eyes that the result largely depends on the traffic type.

![](https://c.mql5.com/2/13/graal_pic_6_1_1.gif)

**Fig. 6. Testing Results for Graal\_1.mq4 (Grail 1) on M1 history of** **EURUSD from the 4th of April to the 26th of July 2006** **in trading terms of the [FX Integralbank](https://www.mql5.com/go?link=http://fxintegralbank.com/ "http://fxintegralbank.com/") demo server at constant orders costs**

The presence of deep drawdowns, instable profit development show that the expert balances on the brink of loss.

As a rule, the requoting prices differ by 1 or 2 points (though it can be more on a fast market). This phenomenon cannot influence essentially the expert's success if its expected payoff is much greater, for example, 10 - 15. But an expert having a small expected payoff, especially, if it is less than 2, cannot seriously rackon on general success since it depends much on the quote nature and the concerned requoting.

Whether a specific dealer will ot not requote on the current tick, is a separate question that can only be answered by the dealer. But, starting to work on the real market, one has to assume that requoting is a normal, natural phenomenon, and a trading technology that works on a real account must consider this.

**1.5. Economic Aspects**

- **Small Price of Orders**

Situation when the trader's attention is fully locked on his or her trading account and its events, including trading strategies, experts, etc., is most commonly encountered. However, beside the trader's concernment, there is the dealer's economic concernment.

At "normal" trading, both parties can win: the trader who uses a well-thought-out strategy and dealer who gets interests from trade operations performed by the trader. In this case, both parties are concerned in each other and both are ready to support each other.

However, there can occur situations where activities of one party oppose interests of the other party. For example, if the dealer increases the spread, disables automated trading, or does not open (close) orders at the price's first touch unilaterally, this opposes the trader's interests.

At the same time, there can be some trader's activities that oppose the dealer's economic interests. One of such activities is **frequent trade operations with small amounts of money**.

The dealer's wroking technology is in whole rather simple. The dealer collects the traders' orders to buy and to sell and enters into relations with a bank or other financial institution to process the differences in their prices. Let us examine a simple example. Suppose, the dealer serves for totally 100 traders, 60 of which have bought by 1 lot EURUSD, and 40 have sold by 1 lot of the same currency pair. In this case, the dealer must buy 20 lots (the difference between 60 and 40) from the bank. At that, it is the same for the dealer in what direction the price will move. In any case, the dealer will get the full spread for 80 lots (40+40) and a partial spread of 20 lots (some part will be given to the bank for its services).

In this case, the **source** of profit/loss for 40 traders is the loss/profit of other 40 traders. The source of profit/loss for the resting 20 traders is the bank or, to be more exact, legal persons served by the bank and selling/buying currencies for their export-import operations. This is a normal way of how the participants of money-market interrelate.

But there is one more detail od how the dealer interrelates with the bank or financial institution. The matter is that the dealer does not interrelate with the bank at each sell or buy operation if the trader sells or buy an **insignificant amount of money**. Trade operations between the dealer and the bank are performed a bit less frequently than the traders' clicks in the МТ4 terminal. Normally, the minimum amount in the relations between the dealer and the bank does not exceed US$50 000, which translated into the order price with a leverage of 1:100 makes 0.5 lot. This is why, if the trader always works with a small amount of money, he or sheб in effect, trades with the dealer. In this case, the **source** of the trader's profit is the dealer's money!

The dealer is as much the participant of the money market, as all others. The dealer must watch and analyze the work of all traders. Of course, dealers, within their economic concernment, can close eyes to irregular repeated investments if they result in the trader's evident losses or its total result for a certain period (for example, the daily result) is neutral. But no reasonable dealer will aloow the interrelations continue if these interrelations result in the dealer's losses. In this case, the dealer has to react somehow, otherwise there is nothing for this dealer on the money-market economic territory.

Please note that the modern routine on the money market is not a dealer's freak or ill will. The dealer's relations to the bank are contracted, as well. And the dealer strives to earn the spread by fair means (abuses on the dealer's side are not discussed within this present article). The dealer would probably be glad to trade with the bank every second, but there is no such possibility for the moment.

Under conditions of **frequent small trade operations** performed by one trader, the dealer has to take some measures. It can be a notification sent or said to the trader that this specific trader works on the brink of a foul. The dealer can, as well, disable automated trading for this trader or increase the amount of requoted operations.

- **Large Price of Orders**

There is one more limitation that differentiates real trading from testing or trading on a demo account. It is the maximum value of the lot price. No dealing center, whether it be a bank, a bookmaker's office or a financial institution, can practically operate with very large amounts of money.

If orders cost about US$10 000 to 20 000 (please note that there is a leverage, which is usually equal to 100), the dealing center will sell/buy as good as US$1 000 000. This is a really large amount, but dealers have learned how to work with such great amounts of money and they work successfully.

Some difficulties can occur in a medium-sized financial institution if the order price reaches US$50 000 or even US$100 000. In this case, the amount of 5 to 10 million US dollars figures in the interbank market. This amount of money is not that easy to sell/buy in one movement at a time.

Predicaments can occur if the order price exceeds the amount of US$ 300 000 to 500 000. In this case, the dealing center will certainly work with the trader individually. Well, 50 million dollars is a really great amount of money for any institution.

In case of great order prices, the dealing center's firm guarantees are out of the question, and the trading itself is performed based on that the trader understands and accepts this. (Indeed, what dealer can take the liberty to open an order that costs a great amount of money at the first touch under the conditions of unexpected and strong movements of the market?)

For this reason, **scalping experts trading with millions is an evidence of the trader's vast mistake** and does not have anything to do with the real life.

This reasoning suggested our programmer to earn the initial 100 "kilobucks" here and then find a "badder office". The programmer becomes quiet and thinks that "all is under control".

Let us not forget that dealers are all normal people with their educational and cultural levels, with their businesses running in some or other way. The same technology, which is unprofitable for dealers, can be immediately detected where analysts work effectively and cannot be quickly detected by dealers who work by halves or are not experienced enough, or do not have necessary software. But in the most cases, dealers are former "grailers", are experienced in trading and usually grip very well all niceties of trading at money-markets, so they know what is geometrical progression, economic concernment, agressive reinvestment, small and large prices of orders. Generally, every trader has to proceed from the positive statement that the dealers' qualifications are higher than those of a trader and any dealer's position on practically any topic is well-reasoned.

Judge yourselves: What (under the above economic and technical conditions) can be results of working of the expert that opens more than 20 000 orders a year?

![](https://c.mql5.com/2/13/graal_pic_7_2.png)

****Fig.7. Testing Report on the expert named Graal\_1 (Grail 1).****

Our programmer's brain swims with all these ticks, bars, points, spreads, dealers, and banks. He or she decides not to get into details or "single questions". He or she is satisfied with his or her own understanding that "test results do not coincide with the real trading". And he or she decides that a suitable technology will be built based on pending orders. "Well, they won't get to me here!" he or she thinks.

### 2\. The Second "Grail"

**2.1 Strategy** The idea to build the expert on pending orders comes to our programmer's mind after he or she has studied the behavior of cross-rates EURGBP and EURCHF on 1-minute charts.

![](https://c.mql5.com/2/13/graal_pic_8.png)

**Fig. 8. Cross-rates history regions with significant deviations in prices (spikes).**

As is easy to see, at night, when European session is already over and Asian session has not started yet, some symbol charts contain bars with unexpected prices that are absolutely different from a "normal" price stream. From formal logic, it is sufficient to place pending orders of the Limit type somewhat above and somewhat below the main stream, and a prat of them will be opened with a high probability of success. For pending orders to react adequately to changes of the main stream, they must be "dragged" along some midline and constantly supported at a given distance from the MA (Moving Average) line. Our programmer has written an expert in order to relize the idea:

```
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
// Graal_2.mq4 (Grail 2).
// Used as an example in the article My First "Grail".
// Sergey Kovalyov, Dnepropetrovsk (Ukraine),sk@mail.dnepr.net,ICQ 64015987, http://autograf.dp.ua/
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
extern int TakeProfit=5;                            // TakeProfit orders
extern int StopLoss= 29;                            // StopLoss orders
extern int Distan   = 2;                            // Distance from the MA line
extern int Cls      = 2;                            // Close at ** points of profit
extern int period_MA=16;                            // MA period
//extern int Time_1   = 0;                          // Starting time
//extern int Time_2   = 0;                          // Finishing time
extern int Prots    = 0;                            // Percentage of free assets

//--------------------------------------------------------------------------------------------
int
   Nom_bl,                                          // BuyLimit order number
   Nom_sl,                                          // SellLimit
   total,                                           // Count of lots
   bl = 0,                                          // 1 = BuyLimit order availability
   sl = 0,                                          // 1 = SellLimit order availability
   b  = 0,                                          // 1 = Buy order availability
   s  = 0;                                          // 1 = Sell order availability
//--------------------------------------------------------------------------------------------
double
   OP,                                              // OpenPrice (absolute points)
   SL,                                              // StopLoss orders (relative points)
   TP,                                              // TakeProfit orders (relative points)
   dist,                                            // Distance from MA (relative points)
   Level,                                           // Min.allowed distance of a pending order
   OP_bl,                                           // OpenPrice BuyLimit (absolute points)
   OP_sl,                                           // OpenPrice SellLimit(absolute points)
   cls,                                             // Close at ** profit (absolute points)
   MA,                                              // MA value (rate)
   spred,                                           // Spread (absolute points)
   Lot;                                             // Count of lots
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int init()
   {
   Level=MarketInfo(Symbol(),MODE_STOPLEVEL);       // Check what the server shows us
   Level=(Level+1)*Point;                           // ?:)
   SL=StopLoss*Point;                               // StopLoss orders (relative points)
   TP=TakeProfit*Point;                             // TakeProfit orders (relative points)
   dist=Distan*Point;                               // Distance from the MA line(relative points)
   cls=Cls*Point;                                   // Close at ** profit (absolute points)
   spred=Ask-Bid;                                   // Spread (absolute points)
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int start()
   {
//============================================================================================
   total=OrdersTotal();                             // Count of lots
   bl=0;                                            // Zeroize at the start of the pass
   sl=0;                                            // Zeroize at the start of the pass
   b=0;                                             // Zeroize at the start of the pass
   s=0;                                             // Zeroize at the start of the pass
//--------------------------------------------------------------------------------------------
   for (int i=total; i>=0; i--)                     // For all orders
      {
      if (OrderSelect(i,SELECT_BY_POS)==true &&     // Select an order
         OrderSymbol()==Symbol())
         {

//--------------------------------------------------------------------------------------------
         if (OrderType()==OP_BUY)                   // Buy order
            {
            b =1;                                   // The order found
            Close_B(OrderTicket(),OrderLots());     // Close the order (the function decides
                                                    // whether it is necessary)
            }
//--------------------------------------------------------------------------------------------
         if (OrderType()==OP_SELL)                   // Sell order
            {
            s =1;                                   // The order found
            Close_S(OrderTicket(),OrderLots());     // Close the order (if necessary)
            }
//--------------------------------------------------------------------------------------------
         if (OrderType()==OP_BUYLIMIT)              // BuyLimit order
            {
            OP_bl=NormalizeDouble(OrderOpenPrice(),Digits);//OpenPrice BuyLimit(absolute points)
            Nom_bl=OrderTicket();
            bl=1;                                   // The order found
            }
//--------------------------------------------------------------------------------------------
         if (OrderType()==OP_SELLLIMIT)             // SellLimit order
            {
            OP_sl=NormalizeDouble(OrderOpenPrice(),Digits);//OpenPrice SellLimit(absolute points)
            Nom_sl=OrderTicket();
            sl=1;                                   // The order found
            }
//--------------------------------------------------------------------------------------------
         }
      }
//--------------------------------------------------------------------------------------------
   MA = iMA(NULL,0, period_MA, 0,MODE_LWMA, PRICE_TYPICAL, 0);// The MA current value
   Modify_order();                                  // Activate modification
   Open_order() ;                                   // Activate opening
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
void Close_B(int Nomber, double lots)               // Close Buy orders
   {
//============================================================================================
   if (NormalizeDouble(Bid-OrderOpenPrice(),Digits)>=cls)// If the preset profit is reached
      {
      OrderClose( Nomber, lots, Bid, 1, Yellow);    // Close
      b = 0;                                        // No Buy order anymore
      }
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
void Close_S(int Nomber, double lots)               // Close Sell orders
   {
//============================================================================================
   if (NormalizeDouble(OrderOpenPrice()-Ask,Digits)>=cls)// If the preset order is reached
      {
      OrderClose( Nomber, lots, Ask, 1, Yellow);    // Close
      s = 0;                                        // No Sell order anymore
      }
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
void Modify_order()                                 // Modification of orders
   {
//============================================================================================
   if (bl==1)                                       // If there is BuyLimit
      {
      OP=MA-dist;                                   // it must be located here
      if (MathAbs(OP_bl-OP)>0.5*Point)              // if it is not located here
         {
         OrderModify(Nom_bl,OP,OP-SL,OP+TP,0,DeepSkyBlue);// The order modification
         }
      }
//--------------------------------------------------------------------------------------------
   if (sl==1)                                       // If there is SeelLimit
      {
      OP=MA+spred+dist;                             // It must be located here
      if (MathAbs(OP_sl-OP)>0.5*Point)              // If it is not located here
         {
         OrderModify( Nom_sl, OP, OP+SL, OP-TP, 0, Pink);// The order modification
         }
      }
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
void Open_order()                                   // An opening function
   {
//   int Tek_Time=TimeHour(CurTime());              // To test by the time
//   if (Tek_Time>Time_2 && Tek_Time>
//============================================================================================
   if (b==0 && bl==0)                               // No Buy orders, open bl
      {
      OP=MA-dist;                                   // bl order open rate
      if(OP>Ask-Level) OP=Ask-Level;                // OP precision according to the tolerance
      OP=NormalizeDouble(OP,Digits);                // Normalizing (MA gives the 5th digit)
      OrderSend(Symbol(),OP_BUYLIMIT, Lots(),OP,3,OP-SL,OP+TP,"",0,0,Blue);// Open
      bl=1;                                         // Now there is a Buy order b1
      }
//--------------------------------------------------------------------------------------------
   if (s==0 && sl==0)                               // No Sell orders, open sl
      {
      OP=MA+spred+dist;                             // sl order open rate
      if(OP<Bid+Level) OP=Bid+Level;                // OP precision according to the tolerance
      OP=NormalizeDouble(OP,Digits);                // Normalizing (MA gives the 5th digit)
      OrderSend(Symbol(),OP_SELLLIMIT,Lots(),OP,3,OP+SL,OP-TP,"",0,0,Red);// Open
      sl=1;                                         // Now there is a Sell order sl
      }
///============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
double Lots()                                       // Calculation of lots
   {
//============================================================================================
   Lot=NormalizeDouble(AccountEquity()*Prots/100/1000,1);// Calculate the amount of lots
   double Min_Lot = MarketInfo(Symbol(), MODE_MINLOT);   // Minimum lot values
   if (Lot == 0 ) Lot = Min_Lot;                         // For testing on const.min.lots
//============================================================================================
   return(Lot);
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
/*
```

StopLoss, TakeProfit, Distan (distance, at which the order keeps along the main stream), Cls (minimum amount of points gained on the order, the order must be closed when the amount of points reaches this value), period\_MA (the MA period; the MA is here the midline of rates of the latest history), and Prots (the percentage of free assets, the order price) have been chosen as adjustable variables. These are quite enough to realize the idea.

- In the special start() function, the orders are analyzed (the orders consideration methods are described [here](https://www.mql5.com/ru/articles/1390)) and the decision about what to do with some of them is made simultaneously. If there is an open order in the terminal, the corresponding function Close\_\*() will be immediately called where the necessity of closing is analyzed and the order is closed.

- At the end of the start() function, the Modify\_order() and Open\_order() functions are opened to, respectively, modify and open pending orders of the Limit type. Other orders are not opened.

- In the Modify\_order() function, the order location is calculated (at some distance from the midline), and, if it is located not on a proper place, it will be moved there.

- In the Open\_order() function, the desired place to locate the pending order is also determined. At this, the server limitations are considered and, if there is no order, it is placed.


In writing the program, our character has noticed that it can be perfected endlessly and decided that the program writing cannot be completed, it just must be stopped! The expert tests on EURCHF resulted in the following (we will study this example, though similar results can be obtained on EURGBP).

![](https://c.mql5.com/2/13/graal_pic_9.gif)

**Fig. 9. Testing Results for Graal\_2.mq4 (Grail 2) on History from March, 2005, to June, 2006, on the M1 EURUSD Chart under Trading Conditions of [MIG Investments](https://www.mql5.com/go?link=http://www.migfx.ch/index.php?id=5&L=2 "http://www.migfx.ch/index.php?id=5&amp;L=2") Company.**
**2.2. Geometrical Progression**
In this expert, as well as in the preceding one, the progressive investment technology is used. For the constant orders prices, the balance graph will look like this:

****![](https://c.mql5.com/2/13/graal_pic_10.gif)****

****Fig. 10.** **Testing Results for Graal\_2.mq4 (Grail 2) on History from March, 2005, to June, 2006, on the M1 EURUSD Chart under Trading Conditions of [MIG Investments](https://www.mql5.com/go?link=http://www.migfx.ch/index.php?id=5&L=2 "http://www.migfx.ch/index.php?id=5&amp;L=2") Company** **at Constant Order Prices.****
Originally, our programmer wanted to earn money only at night, but then was surprised by that the expert successfully worked during the whole trade day. Is it any use saying that after having seen this beauty our character decides again: **"Only me!"** and starts to "make money" again? But nothing of the sort happens... This time, the programmer has to get acquainted with "spikes".

**2.3. Errors**

In the above expert example, there is a large amount of defects and incorrect programming solutions. In the Modify\_order(), the minimum allowed distance for pending orders is not taken into consideration, the open orders considering is organized incorrectly, TakeProfit is not modified separately in order to "play up" the profit to Cls, the analytical part of the expert is "smeared out" on the entire code, etc.

Incorrect and, the more so, uncareful programming often results in undistinguished errors and, sometimes, in error replication if the code is integrated in other experts.

**2.4. Technical Restrictions (Spikes)**

All restrictions, either technical or organizational, will finally turn into economic losses, as a rule. Question arises at whose expense these losses will be covered.

A normal candle chart that is seen by the trader in the monitor is an averaged result of the market development on a specific territory (among European banks, for example). The rate behavior on a short period of history at different dealers can be analyzed. At this, as is easy seen, all these charts will differ from each other, though slightly. Some dealers' charts will contain lots of longmultidirectional candles, while other dealers' charts will show "peace and quiet".

A tick chart is also an object of sale. It is the work result of some processing program that filters excessive and unnatural price ticks. And they ("unnatural" ticks) appear. This happens when a person in a bank, for some personal reasons (anything can happen, right?) has bought or sold some amount at a price very much differing from the market price. The processing program accepts this information and, if such a trasnaction took place only once, does not consider it in forming the final chart. In some cases, the program finds difficulty in undertanding of what price is "normal" and what is not. For this reason, single candles of different length (depending on the processing program's algorithm) sometimes appear in the chart. These are **spikes**. They appear as different types, but they **do not show** the real situation at the money-market.

It is not very difficult to make an expert that would catch these "pins", "spines", "spikes" and open orders when the price is most profitable. But one has to understand well that, under contracts concluded between the dealer and the bank, the bank does not usually pay such a transaction to the dealer. This means that the trader's profit source is the dealer's pocket again! And the dealer does not want this, of course, so the dealer battles with traders for performing of trade operations at "normal", market prices.

As measures providing this opportunity for the dealer, the contract between the dealer and the trader usually contains clauses about the dealer's rights including possibility to requote, non-guaranteed opening of orders at strong price movements, opening of pending orders at the second touch, and other precautions. In some cases, the dealer can cancel an order already opened if, in the dealer's opinion, the order has been opened on a spike.

This is what has happened to our character. Nice as it is on the demo account, the grail was corrected again by the life. This time, our programmer does not slam down the receiver and listens to dealers very attentively every time when they explain the reasons why orders had been closed forcedly. But, when the dealer disabled the experts, the programmer could not stick it and broke the contract again.

Our character now supposes the one-minute timeframes to be guilty of his or her repeated failures. Having read posts on forums and communicated with other traders, the programmer sees that most of them work on larger timeframes with the same symbols. "This must make sense", thinks he or she. Gradually, the programmer arrives at the conclusion of **"Mozart and me"** and takes off coat to develop his or her third creation - not on M1 charts now ("Never, nevermore!").

### 3\. The Third "Grail"

**3.1 Formal Strategy**
Having carefully studied the EURUSD H1 chart and used different indicators to analyze it, our programmer has discovered a very attractive regularity: If the MA (Moving Average) of a smaller period meets the MA of a larger period, the market usually moves in the direction where the MA of the smaller period moves.

![](https://c.mql5.com/2/13/graal_pic_11.png)

**Fig. 11. Graphical Representation of the Strategy Based on Intercrossing of MAs with Different Periods**

The programmer has also noticed that this interrelation appears on all timeframes, but has set on working with only large ones. It only remains for him or her to translate the extracted knowledge into MQL4 in order to inform the expert in what direction and under what conditions it must open orders. And, when the expert is ready, it is necessary to optimize its parameters - to sort out the most effective MA lengths and place StopLoss and TakeProfit properly.
This time, the following expert has been born:

```
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
// Graal_3.mq4 (Grail 3).
// Used as an example in the article My First "Grail".
// Sergey Kovalyov, Dnepropetrovsk (Ukraine),sk@mail.dnepr.net,ICQ 64015987, http://autograf.dp.ua/
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
//
//
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
extern int    MA1 = 11;                                  // Period of the first MA
extern int    MA2 = 23;                                  // Period of the second MA
extern double TP =  50;                                  // TakeProfit orders
extern double SL =  15;                                  // StopLoss orders
extern double Prots= 0;                                  // Percentage of free assets
//--------------------------------------------------------------------------------------------
int
   ret,                                                  // Direction of intersection
   total;                                                // Count of open orders
//--------------------------------------------------------------------------------------------
double
   Lot,                                                  // Count of lots
   Pred,                                                 // Preceding value of the 1st MA(red)
   Tek,                                                  // Current value of the 1st MA (red)
   Golub;                                                // Current value of the 2nd MA (blue)
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int init()
   {
//============================================================================================
   SL = SL*Point;                                        // StopLoss in points
   TP = TP*Point;                                        // TakeProfit in points
   return;
//============================================================================================
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int start()
   {
//============================================================================================
   total=OrdersTotal();                                  // Total amount of orders
   if (total==2)return;                                  // Both orders are already open
//--------------------------------------------------------------------------------------------
   Tek  =iMA(NULL,0, MA1, 0,MODE_LWMA, PRICE_TYPICAL, 0);// Current value of the 1st MA
   Pred =iMA(NULL,0, MA1, 0,MODE_LWMA, PRICE_TYPICAL, 1);// Preceding value of the 2nd MA
   Golub=iMA(NULL,0, MA2, 0,MODE_LWMA, PRICE_TYPICAL, 0);// Current value of the 2nd MA
//--------------------------------------------------------------------------------------------
   if (Peresechenie()==1) Open_Buy();                    // Movement bottom-up = open Buy
   if (Peresechenie()==2) Open_Sell();                   // Movement top-down  = open Sell
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Peresechenie()                                       // Function determining intersection
   {
//============================================================================================
   if ((Pred<=Golub && Tek> Golub) ||
       (Pred< Golub && Tek>=Golub)  ) ret=1;             // Intersection bottom-up
//--------------------------------------------------------------------------------------------
   if ((Pred>=Golub && Tek< Golub) ||
       (Pred> Golub && Tek<=Golub)  ) ret=2;             // Intersection top-down
//============================================================================================
   return(ret);                                          // Return the intersection direction
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Open_Buy()                                           // Function opening Buy
   {
//============================================================================================
   if (total==1)                                         // If there is only one order...
      {                                                  // ... it means another can be opened
      OrderSelect(0, SELECT_BY_POS);                     // Select the order
      if (OrderType()==0)return;                         // If it is Buy, do not open
      }
   OrderSend(Symbol(),0, Lots(), Ask, 0, Ask-SL, Ask+TP, "", 0, 0, Blue);// open
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Open_Sell()                                          // Function opening Sell
   {
//============================================================================================
   if (total==1)                                         // If there is only one order...
      {                                                  // ... it means another can be opened
      OrderSelect(0, SELECT_BY_POS);                     // Select the order
      if (OrderType()==1)return;                         // If it is Sell, do not open
      }
   OrderSend(Symbol(),1, Lots(), Bid, 0, Bid+SL, Bid-TP, "", 0, 0, Red);// Open
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
double Lots()                                            // Calculation of lots
   {
//============================================================================================
   Lot=NormalizeDouble(AccountEquity()*Prots/100/1000,1);// Calculate the amount of lots
   double Min_Lot = MarketInfo(Symbol(), MODE_MINLOT);   // Minimum lot values
   if (Lot == 0 ) Lot = Min_Lot;                         // For testing on const. min. lots
//============================================================================================
   return(Lot);
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
/*
```

This expert, like all preceding ones, turned out to be rather simple.

The expert starts with the block that describes variables. In the start() function, the total amount of orders, as well as the MA values for the larger and for the smaller period are defined. If the MAs meet, the corresponding function is called that will open orders: Open\_\*(). To state that MAs have met, the Peresechenie() function is used, to find out about the lot values - the Lots() function is.

Our programmer looks though his or her expert once again, does not see any defects ("Simple, but nice!") and started testing. After optimization of variables, the result was splendid:

![](https://c.mql5.com/2/13/graal_pic_12.gif)

****Fig. 12.** ****Testing Results for Graal\_3.mq4 (Grail 3) on History from March, 2005, to July, 2005,******

******on the H1 EURUSD Chart under Trading Conditions of******[MetaQuotes Software Corp](https://www.metaquotes.net/ "https://www.metaquotes.net/"). demo server.****

Three million within five months! Wonderful! But something in this balance graph resemples our character the bitterness of the previous experience. And the programmer decides to be a bit slower.

**3.2. Progression**

After the same expert has been tested at constant order prices, it showed a rather acceptable balance graph:

![](https://c.mql5.com/2/13/graal_pic_13.gif)

****Fig. 13.** Testing Results for Graal\_3.mq4 (Grail 3) on History from March, 2005, to July, 2005,****on the H1 EURUSD Chart******

******under Trading Conditions of******[MetaQuotes Software Corp](https://www.metaquotes.net/ "https://www.metaquotes.net/"). demo server** **at constant lots prices.****

Six thousand points within five months - you could cut it with a knife. Over one thousand points a month! But our character doubts whether or not to set the expert for real trading. After all, he or she has already mistaken twice...

Then the programmer has decided to look more attentively at the timeframe with markings of orders openings and closings and much to his or her surprise discovered that the expert opens orders not only at intersections of MAs, but also at other points, without any reasons!

![](https://c.mql5.com/2/13/graal_pic_14.png)

****Fig. 14. Opening of orders by the expert named Graal\_3.mq4 (Grail 3) in situations that have not been foreseen within the initial strategy.****

**3.3. Errors**

"What a strange news! There must be something wrong with MetaEditor!" was the first thought. But then, having gradually and consecutively studied the code, the programmer finds errors again. Let us learn about them, too.

**3.3.1 Deviations from the Initial Strategy - Algorithmic Errors**

Look at the simple function below:

```
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Peresechenie()                                       // Function that detects the intersection
   {
//=================================================================================================
   if ((Pred<=Golub && Tek> Golub) ||
       (Pred< Golub && Tek>=Golub)  ) ret=1;             // Bottom-up intersection
//-------------------------------------------------------------------------------------------------
   if ((Pred>=Golub && Tek< Golub) ||
       (Pred> Golub && Tek<=Golub)  ) ret=2;             // Top-down intersection
//=================================================================================================
   return(ret);                                          // Return the intersection direction
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
```

It all seems to be quite clear, what can be found here? But there is still an error here. It is in that, when the global variable **ret** is used, its value is stored equal to the last calculated one, i.e. 1 or 2. For this reason, the Peresechenie() function, regardless of the current situation, just returns the last calculated value. This is why it is constantly ordered to open in the start() function:

```
   if (Peresechenie()==1) Open_Buy();                    // Bottom-up movement = open Buy
   if (Peresechenie()==2) Open_Sell();                   // Top-down movement  = open Sell
```

To make changes, it is sufficient to zeroize the **ret** variable at the beginning of the Peresechenie() function. Having caught at the algorithmic error found, our character corrects it and writes a fragment at the same time that allows to open on only one bar. Now the expert looks like this:

```
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
// Graal_31.mq4 (Grail 31)
// Used as an example in the article My First "Grail"
// Sergey Kovalyov, Dnepropetrovsk, sk@mail.dnepr.net, ICQ 64015987, http://autograf.dp.ua/.
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
//
//
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
extern int    MA1 = 11;                                  // Period of the 1st MA
extern int    MA2 = 23;                                  // Period of the 2nd MA
extern double TP =  50;                                  // TakeProfit orders
extern double SL =  15;                                  // StopLoss orders
extern double Prots= 0;                                  // Percentage of free assets
//--------------------------------------------------------------------------------------------
int
   New_Bar,                                              // 0/1 The fact of new bar forming
   Time_0,                                               // The new bar beginning time
   ret,                                                  // Intersection direction
   total;                                                // Count of open orders
//--------------------------------------------------------------------------------------------
double
   Lot,                                                  // Count of lots
   Pred,                                                 // Previous value of the 1st MA (red)
   Tek,                                                  // Current value of the 1st MA (red)
   Golub;                                                // Current value of the 2nd MA (blue)
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int init()
   {
//============================================================================================
   SL = SL*Point;                                        // SL in points
   TP = TP*Point;                                        // ТР in points
   return;
//============================================================================================
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int start()
   {
//============================================================================================
   total=OrdersTotal();                                  // Total count of orders
   if (total==2)return;                                  // Both orders already opened

//----------------------------------------------------------------------------- New bar ------
   New_Bar=0;                                            // First zeroize
   if (Time_0 != Time[0])                                // If the bar beginning time changed
      {
      New_Bar= 1;                                        // Here we have a new bar
      Time_0 = Time[0];                                  // Remember the new bar beginning time
      }

//--------------------------------------------------------------------------------------------
   Tek  =iMA(NULL,0, MA1, 0,MODE_LWMA, PRICE_TYPICAL, 0);// Current value of the 1st MA
   Pred =iMA(NULL,0, MA1, 0,MODE_LWMA, PRICE_TYPICAL, 1);// Previous value of the 2nd MA
   Golub=iMA(NULL,0, MA2, 0,MODE_LWMA, PRICE_TYPICAL, 0);// Current value of the 2nd MA

//--------------------------------------------------------------------------------------------
   if (Peresechenie()==1 && New_Bar==1) Open_Buy();      // Bottom-up movement = open Buy
   if (Peresechenie()==2 && New_Bar==1) Open_Sell();     // Top-down movement  = open Sell
//============================================================================================
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Peresechenie()
   {
   ret=0;                                                // That's the heart of the matter!:)
//============================================================================================
   if ((Pred<=Golub && Tek> Golub) ||
       (Pred< Golub && Tek>=Golub)  ) ret=1;             // Bottom-up intersection
//--------------------------------------------------------------------------------------------
   if ((Pred>=Golub && Tek< Golub) ||
       (Pred> Golub && Tek<=Golub)  ) ret=2;             // Top-down intersection
//============================================================================================
   return(ret);                                          // Return the intersection direction
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Open_Buy()
   {
   if (total==1)
      {
      OrderSelect(0, SELECT_BY_POS);                     // Select the order
      if (OrderType()==0)return;                         // If it is buy, don't open
      }
   OrderSend(Symbol(),0, Lots(), Ask, 0, Ask-SL, Ask+TP, "", 0, 0, Blue);// Open
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Open_Sell()
   {
   if (total==1)
      {
      OrderSelect(0, SELECT_BY_POS);                     // Select the order
      if (OrderType()==1)return;                         // If it is sell, don't open
      }
   OrderSend(Symbol(),1, Lots(), Bid, 0, Bid+SL, Bid-TP, "", 0, 0, Red);// Open
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
double Lots()
   {
   Lot = NormalizeDouble( AccountEquity()*Prots/100/1000, 1);// Calculate the amount of lots
   if (Lot<0.1)Lot = 0.1;                                // For testing on const. min. lots
   return(Lot);
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
/*
```

Testing of this expert at constant lots prices has met with only limited success:

![](https://c.mql5.com/2/13/graal_pic_15.gif)

**Fig. 15.** **Testing Results for Graal\_31.mq4 (Grail 31) on History from March, 2005, to July, 2005, ****on the H1 EURUSD Chart******

******under Trading Conditions of****[MetaQuotes Software Corp](https://www.metaquotes.net/ "https://www.metaquotes.net/"). demo server at constant lots prices.**

Our programmer feels utterly discouraged. So much work and all in vain. Time waxed on, the programmer went on questing for the "grail". After a couple of months, he or she decides to return to the expert named Graal\_3.mq4 (Grail 3). "Well, there was an error there, but it gave good results. Maybe, this is the matter-of-course! May the expert open orders as it wants if only good results are kept". And starts testing again.

What was the programmer's surprise at seeing that the expert started to lose deposit monotonically, by itself, without any changes in the code!

![](https://c.mql5.com/2/13/graal_pic_16.gif)

**Fig. 16.** **Testing Results for Graal\_3.mq4 (Grail 3) on History from March, 2005, to July, 2006, ****on the H1 EURUSD Chart******

******under Trading Conditions of****[MetaQuotes Software Corp](https://www.metaquotes.net/ "https://www.metaquotes.net/"). demo server at constant lots prices.**

**3.4. Fitting of Results**

Out character finds himself or herself to be ostriching. It proves that there are no difficulties at all to fitt optimal parameters for an a little effective expert in a history segment. The same expert will always give different results on different history periods.

The whole question reduces itself to the question of how to find such optimal inputs for the expert, if possible, at which small market fluctuations in price velocity would not influence the expert's workability and efficiency. Generally, one has to focus on a strategy, which is resistant to any changes in the market, and optimization of parameters must just help to some extent in getting the best of possible results.

The search for optimal settings for the expert named Graal\_3.mq4 (Grail 3) has resulted in nothing. Testing of the expert at different values of external variables has either showed losses or resulted in significantly different settings for different history periods. For this reason, our character has not succeeded in finding the best settings. So he or she has concluded that this strategy has no universal settings for all occasions.

Our programmer has returned to the expert named Graal\_31.mq4 (Grail 31) and agonizes to get the desired results. But the best ones look like this:

![](https://c.mql5.com/2/13/graal_pic_17.gif)

**Fig. 17.** **Testing Results for Graal\_31.mq4 (Grail 31) on History from March, 2005, to July, 2006, ****on the H1 EURUSD Chart******

******under Trading Conditions of****[MetaQuotes Software Corp](https://www.metaquotes.net/ "https://www.metaquotes.net/"). demo server at constant lots prices.**

That what the programmer sees above does not resemble any "grail" at all. Because of the naked-eye deep drawdown in the middle of the chart, the expert cannot be used for working on a real account. But it is still a working expert that shows a far removed from fantastic, but still positive result.

This encourages our character.

### 4\. Conclusion

Disillusionment is always a wrench, especially if the illusions are so sanguine and happy. But life puts everything in its place again sooner or later. The cares of the world took the shine out of our character, too: The programmer has damaged relations with two dealing centers, lost a certain amount of money hardly earned... Having gained priceless (for his or her own) experiences in programming and trading on Forex, our character has almost drawn a conclusion of **"Mozart only..."**, but has stopped in time and decided that he or she will never draw hasty conclusions.

Instead of this, the programmer takes an empty sheet of paper and formulates conclusions drawn on basis of lessons digested:

- **Progressive investments** in real trading are normal. However, orders may not be of a too high price.

- **Adhere to the initial strategy**

If it turns during the expert development that significant changes in adjustable parameters result in significant changes in outcomes, it means that the initial strategy is levelled up or down. It is quite possible that the resulting new strategy is more effective. In this case, it should be held in abeyance for a short while and called later, after the reasons for such an unexpected success have been studied well. But one has to return to the initial idea first in order to investigate its usability to the full extent admitting changes in parameters only within the reasonable limits.

- **Keep a close watch on algorithmic errors** The presence of algorithmic errors is a quite natural phenomenon. It sometimes needs to have much experience to discover such errors. At the same time, some of them can and must be found at the final results of the expert work. To often order openings or openings "off the mark" are a sharp evidence of an algorithmic error.

- **Properly interpret the expert testing results in Strategy Tester**

The work of the expert on a real account and its modeling in Strategy Tester differ from each other. The difference consists in that modeling does not consider rejections to execute orders and requotes that occur in the real trading. This does not matter for experts with high expected payoff, but for those with a low one, the working results can differ significantly from those during testing. In a number of cases, an expert having low expected payoff and showing fantastic profits in testing can result in losses iduring real trading.

- **No agressive investments**

Any trading strategy must consider losing operations. If the order prices are unlimitedly high, even a short series of losses can cause irrepairable wrong in the total balance, and, in some cases, even a total loss of the deposit. Allowance of unsupervized agressive reinvestments means a voluntary ostrichism, a guaranteed bankruptcy.

- **No attempts to earn money on chance fluctuations**

Spikes do not reflect normal, natural market rates. If traders earn on such spikes, their profit source is the dealer's money. This fact will be sooner or later discovered by the dealer with all it implies: cancellation of transactions, disabling of experts, blocking of the account.
- **Crosscheck test results on different history periods**

Every expert will show different results on different timeframes. For each history interval, its optimal settings can be found, at which the results will be the best. One has to be very attentive at testing, not pursue ostrichism (fitting the expert parameters), but crosscheck the obtained results on as many history intervals as possible.

SK. Dnepropetrovsk. 2006.

Translated from Russian by MetaQuotes Software Corp.

Original article: [/ru/articles/1413](https://www.mql5.com/ru/articles/1413)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1413](https://www.mql5.com/ru/articles/1413)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1413.zip "Download all attachments in the single ZIP archive")

[Grail\_1.mq4](https://www.mql5.com/en/articles/download/1413/grail_1.mq4 "Download Grail_1.mq4")(5.99 KB)

[Grail\_2.mq4](https://www.mql5.com/en/articles/download/1413/grail_2.mq4 "Download Grail_2.mq4")(11.58 KB)

[Grail\_3.mq4](https://www.mql5.com/en/articles/download/1413/grail_3.mq4 "Download Grail_3.mq4")(6.38 KB)

[Grail\_31.mq4](https://www.mql5.com/en/articles/download/1413/grail_31.mq4 "Download Grail_31.mq4")(5.9 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Synchronization of Expert Advisors, Scripts and Indicators](https://www.mql5.com/en/articles/1393)
- [Considering Orders in a Large Program](https://www.mql5.com/en/articles/1390)
- [Graphic Expert Advisor: AutoGraf](https://www.mql5.com/en/articles/1378)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39212)**
(23)


![map.q5](https://c.mql5.com/avatar/2013/8/51F9F99D-B8DB.jpg)

**[map.q5](https://www.mql5.com/en/users/map.q5)**
\|
16 Apr 2015 at 12:48

Good article, thanks for sharing

![Sithembiso Gama](https://c.mql5.com/avatar/2015/6/55930470-19BF.JPG)

**[Sithembiso Gama](https://www.mql5.com/en/users/access2sterror)**
\|
5 Jul 2015 at 20:52

Thanks, this was a good read. Seems like I am not the only one with such a terrible and seemingly devastating experience.


![karar97](https://c.mql5.com/avatar/2016/6/576D7ACA-65E4.jpg)

**[karar97](https://www.mql5.com/en/users/karar97)**
\|
24 Jun 2016 at 18:18

**MQL4 Comments:**

**Agree. Some of them are not using MT4 at all though.**

Hi Dear I test it on my MT4 the result is this so pleas any one can help me to

set grail\_1 minus to plus

![Robot for everyone](https://c.mql5.com/avatar/2019/4/5CB3CCA6-99BF.jpg)

**[Robot for everyone](https://www.mql5.com/en/users/traderandre)**
\|
15 Nov 2016 at 16:43

Karar Hello !! Good ..

I am also very interested in this EA !!

As we can at least determine the [parameters](https://www.mql5.com/en/docs/directx/dxinputset "MQL5 Documentation: DXInputSet function")

to make it "profitable"?

I was able to at least $ 20 or $ 30

In the afternoon, I was happy !! :)

![Robot for everyone](https://c.mql5.com/avatar/2019/4/5CB3CCA6-99BF.jpg)

**[Robot for everyone](https://www.mql5.com/en/users/traderandre)**
\|
15 Nov 2016 at 17:13

Hello friends!

I am a beginner, please,

someone could indicate

an EA or explain what [parameters](https://www.mql5.com/en/docs/directx/dxinputset "MQL5 Documentation: DXInputSet function")

ideal for some of these posted EAs

in this article?

Thanks, hugs to all!

![Secrets of MetaTrader 4 Client Terminal](https://c.mql5.com/2/13/158_10.png)[Secrets of MetaTrader 4 Client Terminal](https://www.mql5.com/en/articles/1415)

21 way to ease the life: Latent features in MetaTrader 4 Client Terminal.
Full screen; hot keys; Fast Navigation bar; minimizing windows; favorites; traffic reduction; disabling of news; symbol sets; Market Watch; templates for testing and independent charts; profiles; crosshair; electronic ruler; barwise chart paging; account history in the chart; types of pending orders; modifying of StopLoss and TakeProfit; undo deletion; chart print.

![Considering Orders in a Large Program](https://c.mql5.com/2/13/114_3.gif)[Considering Orders in a Large Program](https://www.mql5.com/en/articles/1390)

General principles of considering orders in a large and complex program are discussed.

![Secrets of MetaTrader 4 Client Terminal: Alerting System](https://c.mql5.com/2/13/159_6.png)[Secrets of MetaTrader 4 Client Terminal: Alerting System](https://www.mql5.com/en/articles/1416)

How to be aware of what happens in the terminal and on your account without permanent looking at the monitor.
System events; custom events; wave and executable files; electronic messages; setting up SMTP server access; publications; setting up FTP server access.

![A Pause between Trades](https://c.mql5.com/2/12/103_1.gif)[A Pause between Trades](https://www.mql5.com/en/articles/1355)

The article deals with the problem of how to arrange pauses between trade operations when a number of experts work on one МТ 4 Client Terminal. It is intended for users who have basic skills in both working with the terminal and programming in MQL 4.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=uzwavdnqkxptpvfkawetwdhhszvienes&ssn=1769251395422687071&ssn_dr=0&ssn_sr=0&fv_date=1769251395&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1413&back_ref=https%3A%2F%2Fwww.google.com%2F&title=My%20First%20%22Grail%22%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925139582548810&fz_uniq=5083066341734224963&sv=2552)

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