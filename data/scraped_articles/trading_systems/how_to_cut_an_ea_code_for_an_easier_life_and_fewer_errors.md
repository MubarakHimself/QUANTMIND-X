---
title: How to Cut an EA Code for an Easier Life and Fewer Errors
url: https://www.mql5.com/en/articles/1491
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:57:56.566026
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/1491&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083243942926882739)

MetaTrader 4 / Trading systems


### Introduction

There is many trading systems based on technical analysis whether they are indicators or graphical drawings, that have an important property. I mean the symmetry of such systems in the trading direction. Due to this property, trading signals and the mechanics of placing trade orders in such systems can be generally expressed as relative to their directions.

A simple approach described below allows effective using this property to significantly reduce the code length of Expert Advisors based on such symmetric systems. Expert Advisors using this approach utilize the same code for detecting trade signals and generating trade orders, for both long and short positions.

It is a common practice that developing an Expert Advisor based on a symmetric system, one first codes generation and processing of trade signals in one direction and then copies the code and refines it for the other direction. In this case, it is very easy to make an error and then it is very difficult to detect such an error. So the decreasing the amount of possible errors in the Expert Advisor's logic is an additional advantage of the approach considered.

### 1\. Expert Advisor's Embeddings Invariant with Respect to Trade Direction

The concept under consideration is based on _trade direction_. This direction can be either long (buying signals and orders) or short (selling signals and orders). Our object is writing Expert Advisors in such a way that their codes are invariant with respect to the current trade direction. To avoid pads in the text, we will call this code invariant considering that it is invariant only towards trade direction.

For this, we will input a function or a variable, the value of which will always show the current trade direction with one of two possible values.

Representation of this variable in the code is a very important aspect. Though the **bool** type seems to fit for these purposes, it would be more effective to use an a bit different representation - an integer one. The trade direction itself is coded as follows:

- long trade direction: **+1**
- short trade direction: **-1**

An advantage of this representation as compared to the logical one, is that it can be used effectively to make various calculations and checks in the Expert Advisor's code without conditional branching used in conventional approaches.

### 2\. Example of How to Change from Conventional Code to the Invariant One

Let us clear this statement on some examples. However, let's begin with considering a couple of auxiliary functions that we will repeatedly use later:

```
int sign( double v )
{
    if( v < 0 ) return( -1 );
    return( 1 );
}

double iif( bool condition, double ifTrue, double ifFalse )
{
    if( condition ) return( ifTrue );

    return( ifFalse );
}

string iifStr( bool condition, string ifTrue, string ifFalse )
{
    if( condition ) return( ifTrue );

    return( ifFalse );
}

int orderDirection()
{
    return( 1 - 2 * ( OrderType() % 2 ) );
}
```

The purpose of the sign() function is obvious:it returns 1 for nonnegative values of the argument and -1 for negative ones.

Function iif() is an equivalent of the C-language operator named "condition ? ifTrue : ifFalse" and allows significant simplifying invariant Expert Advisors making it more compact and representative. It takes arguments of the **double** type, so it can be used with values of both this type and of types **int** and **datetime**. For the same work with strings, we will need a fully analogous function iifStr() that takes values of the **string** type.

Function orderDirection() returns the direction of the current trade order (i.e., that selected by function OrderSelect()) according to our agreements about how to represent trade directions.

Now let us consider on specific examples how the invariant approach with such coding of trade directions allows simplifying Expert Advisor codes:

#### 2.1 Example 1. Transform Trailing Stop Realization

A typical code:

```
if( OrderType() == OP_BUY )
{
    bool modified = OrderModify( OrderTicket(), OrderOpenPrice(), Bid - Point *
        TrailingStop, OrderTakeProfit(), OrderExpiration() );

    int error = GetLastError();
    if( !modified && error != ERR_NO_RESULT )
    {
        Print( "Failed to modify order " + OrderTicket() + ", error code: " +
            error );
    }
}
else
{
    modified = OrderModify( OrderTicket(), OrderOpenPrice(), Ask + Point *
        TrailingStop, OrderTakeProfit(), OrderExpiration() );

    error = GetLastError();
    if( !modified && error != ERR_NO_RESULT )
    {
        Print( "Failed to modify order " + OrderTicket() + ", error code: " +
            error );
    }
}
```

The invariant code:

```
double closePrice = iif( orderDirection() > 0, Bid, Ask );

bool modified = OrderModify( OrderTicket(), OrderOpenPrice(), closePrice -
    orderDirection() * Point * TrailingStop, OrderTakeProfit(),
    OrderExpiration() );

int error = GetLastError();
if( !modified && error != ERR_NO_RESULT )
{
    Print( "Failed to modify order " + OrderTicket() + ", error code: " +
        error );
}
```

Summarizing:

1. we managed to avoid heavy conditional branching;
2. we used only one string calling function OrderModify() instead of two initial ones; and,

3. as implication of (2), we shortened the code for error processing.

Please note that we managed to use only one call for OrderModify() due to that we utilize the trade order direction immediately in the arithmetic expression for calculation of stop level. If we used a logical representation of the trade direction, it would be impossible.

Basically, an experienced Expert Advisors' writer would be able to do with only one call for OrderModify() using the conventional approach. However, in our case, this happens absolutely naturally and does not require any additional steps.

#### 2.2 Example 2.Transform Trade Signal Detection

As an example, let us consider detection of trade signals in a system of two moving averages:

```
double slowMA = iMA( Symbol(), Period(), SlowMovingPeriod, 0, MODE_SMA,
    PRICE_CLOSE, 0 );
double fastMA = iMA( Symbol(), Period(), FastMovingPeriod, 0, MODE_SMA,
    PRICE_CLOSE, 0 );

if( fastMA > slowMA + Threshold * Point )
{
    // open a long position
    int ticket = OrderSend( Symbol(), OP_BUY, Lots, Ask, Slippage, 0, 0 );

    if( ticket == -1 )
    {
        Print( "Failed to open BUY order, error code: " + GetLastError() );
    }
}
else if( fastMA < slowMA - Threshold * Point )
{
    // open a short position
    ticket = OrderSend( Symbol(), OP_SELL, Lots, Bid, Slippage, 0, 0 );

    if( ticket == -1 )
    {
        Print( "Failed to open SELL order, error code: " + GetLastError() );
    }
}
```

Now let us make the code invariant with respect to the trade direction:

```
double slowMA = iMA( Symbol(), Period(), SlowMovingPeriod, 0, MODE_SMA,
    PRICE_CLOSE, 0 );
double fastMA = iMA( Symbol(), Period(), FastMovingPeriod, 0, MODE_SMA,
    PRICE_CLOSE, 0 );

if( MathAbs( fastMA - slowMA ) > Threshold * Point )
{
    // open a position
    int tradeDirection = sign( fastMA - slowMA );
    int ticket = OrderSend( Symbol(), iif( tradeDirection > 0, OP_BUY, OP_SELL ),
        Lots, iif( tradeDirection > 0, Ask, Bid ), Slippage, 0, 0 );

    if( ticket == -1 )
    {
        Print( "Failed to open " + iifStr( tradeDirection > 0, "BUY", "SELL" ) +
            " order, error code: " + GetLastError() );
    }
}
```

I think it is absolutely obvious that the code has become more compact. And, naturally, two checks for errors turned into only one.

In spite of the fact that the above examples are very simple, the main advantages of the approach under consideration must be very obvious. In some of more complicated cases, the difference between the traditional approach and that under consideration is even more significant. Let us make sure of it on the example of a standard Expert Advisor, MACD Sample

### 3\. How to Simplify MACD Sample

In order not to pad the article, we will not consider the full code of this Expert Advisor here. Let us go into the code areas that will be changed under the concept considered.

The full code of this EA is included in the MetaTrader 4 delivery set. It is also attached to this article (file MACD Sample.mq4) together with its simplified version (MACD Sample-2.mq4) for your convenience.

Let us start with the block written for detection of trade signals. Its initial code is given below:

```
// check for long position (BUY) possibility
if(MacdCurrent<0 && MacdCurrent>SignalCurrent && MacdPrevious<SignalPrevious &&
   MathAbs(MacdCurrent)>(MACDOpenLevel*Point) && MaCurrent>MaPrevious)
  {
   ticket=OrderSend(Symbol(),OP_BUY,Lots,Ask,3,0,Ask+TakeProfit*Point,
     "macd sample",16384,0,Green);
   if(ticket>0)
     {
      if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES))
        Print("BUY order opened : ",OrderOpenPrice());
     }
   else Print("Error opening BUY order : ",GetLastError());
   return(0);
  }
// check for short position (SELL) possibility
if(MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious &&
   MacdCurrent>(MACDOpenLevel*Point) && MaCurrent<MaPrevious)
  {
   ticket=OrderSend(Symbol(),OP_SELL,Lots,Bid,3,0,Bid-TakeProfit*Point,
     "macd sample",16384,0,Red);
   if(ticket>0)
     {
      if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES))
        Print("SELL order opened : ",OrderOpenPrice());
     }
   else Print("Error opening SELL order : ",GetLastError());
   return(0);
  }
```

Now, using the above methods, let us re-write the code in such a way that it is the same for both buying and selling signals:

```
int tradeDirection = -sign( MacdCurrent );

// check if we can enter the market
if( MacdCurrent * tradeDirection < 0 && ( MacdCurrent - SignalCurrent ) *
    tradeDirection > 0 && ( MacdPrevious - SignalPrevious ) * tradeDirection < 0
    && MathAbs(MacdCurrent)>(MACDOpenLevel*Point) && ( MaCurrent - MaPrevious ) *
    tradeDirection > 0 )
  {
   int orderType = iif( tradeDirection > 0, OP_BUY, OP_SELL );
   string orderTypeName = iifStr( tradeDirection > 0, "BUY", "SELL" );
   double openPrice = iif( tradeDirection > 0, Ask, Bid );
   color c = iif( tradeDirection > 0, Green, Red );
   ticket = OrderSend( Symbol(), orderType, Lots, openPrice, 3 , 0, openPrice +
     tradeDirection * TakeProfit * Point, "macd sample", 16384, 0, c );
   if(ticket>0)
     {
      if(OrderSelect(ticket,SELECT_BY_TICKET,MODE_TRADES))
        Print( orderTypeName + " order opened : ", OrderOpenPrice() );
     }
   else Print("Error opening " + orderTypeName + " order : ",GetLastError());
   return(0);
  }
```

Now let us go to the block responsible for closing open positions and processing trailing stops. Let us first study its initial version, as before:

```
if(OrderType()==OP_BUY)   // long position is opened
  {
   // should it be closed?
   if(MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious &&
      MacdCurrent>(MACDCloseLevel*Point))
       {
        OrderClose(OrderTicket(),OrderLots(),Bid,3,Violet); // close position
        return(0); // exit
       }
   // check for trailing stop
   if(TrailingStop>0)
     {
      if(Bid-OrderOpenPrice()>Point*TrailingStop)
        {
         if(OrderStopLoss()<Bid-Point*TrailingStop)
           {
            OrderModify(OrderTicket(),OrderOpenPrice(),Bid-Point*TrailingStop,
               OrderTakeProfit(),0,Green);
            return(0);
           }
        }
     }
  }
else // go to short position
  {
   // should it be closed?
   if(MacdCurrent<0 && MacdCurrent>SignalCurrent &&
      MacdPrevious<SignalPrevious && MathAbs(MacdCurrent)>(MACDCloseLevel*Point))
     {
      OrderClose(OrderTicket(),OrderLots(),Ask,3,Violet); // close position
      return(0); // exit
     }
   // check for trailing stop
   if(TrailingStop>0)
     {
      if((OrderOpenPrice()-Ask)>(Point*TrailingStop))
        {
         if((OrderStopLoss()>(Ask+Point*TrailingStop)) || (OrderStopLoss()==0))
           {
            OrderModify(OrderTicket(),OrderOpenPrice(),Ask+Point*TrailingStop,
               OrderTakeProfit(),0,Red);
            return(0);
           }
        }
     }
  }
```

Let us transform this code into an invariant one with respect to the trade direction:

```
tradeDirection = orderDirection();
double closePrice = iif( tradeDirection > 0, Bid, Ask );
c = iif( tradeDirection > 0, Green, Red );

// should it be closed?
if( MacdCurrent * tradeDirection > 0 && ( MacdCurrent - SignalCurrent ) *
    tradeDirection < 0 && ( MacdPrevious - SignalPrevious ) * tradeDirection > 0
    && MathAbs( MacdCurrent ) > ( MACDCloseLevel * Point ) )
    {
     OrderClose(OrderTicket(),OrderLots(), closePrice, 3,Violet); // close position
     return(0); // exit
    }
// check for trailing stop
if(TrailingStop>0)
  {
   if( ( closePrice - OrderOpenPrice() ) * tradeDirection > Point * TrailingStop )
     {
      if( OrderStopLoss() == 0 || ( OrderStopLoss() - ( closePrice - tradeDirection *
        Point * TrailingStop ) ) * tradeDirection < 0 )
        {
         OrderModify( OrderTicket(), OrderOpenPrice(), closePrice - tradeDirection *
            Point * TrailingStop, OrderTakeProfit(), 0, c );
         return(0);
        }
     }
  }
```

Please note that the initial version of the EA checks condition OrderStopLoss() == 0 only for short positions in processing of a trailing stop. This is necessary to process situations when one did not manage to set the initial stop level (for example, due to that it was too close to the market price).

The fact that this condition is not checked for long positions can be considered as an error that is very typical for writing such symmetric Expert Advisors using the copy-and-paste method.

Please note that this error has been automatically fixed for both trade directions in the improved code. It must also be noted that, if this error made writing the invariant code, it would occur at processing both long and short positions. It goes without saying that this would increase probability of its detection during testing.

Well, that's about it. If you test the Expert Advisors with the same settings on the same data, you will see that they are absolutely equivalent. However, the simplified version is much more compact and maintainable.

### 4\. Recommendations on Writing Symmetric Expert Advisors "from Scratch"

Until now, we considered possibilities of how to change from conventional code of an Expert Advisor to the invariant one. However, development of a trading robot using the above principles "from scratch" is even more effective.

At a first glance, it does not seem to be easy since it needs certain skills ans experiences in formulating conditions and expressions invariant with respect to the trade direction. However, after having some practice, writing code in this style will go easily, as by itself.

I will now try to give some recommendations that may help to start using a more effective approach:

1. When developing this or that code area, first of all, process the long trade direction - in the most cases, it will be easier to synthesize the invariant code, since this trade direction is represented by value +1 and does not take much of doing when writing and analyzing invariant relationships.
2. If you start working on the long direction, first try to write a condition without a variable/function that reflects the trade direction. Make sure that the expression is correct and add the trade direction into it. After having gained some experiences, you may continue without this stepwise division.
3. Do not "hang up" on the long trade direction - it is sometimes more effective to express the condition for the short direction.
4. Try to avoid conditional branching and using of function iif() where it is possible to do with arithmetic calculationsо.

As to the last clause, I would add that there can be situations where you cannot do without conditional branching. However, you should try and pool such situations and appropriate them to separate helper functions of trade direction that don't depend on a specific EA. These functions, as well as the above functions, sign(), iif() and orderDirection(), can be appropriate to a special library that will later be used by all your Expert Advisors.

To clear everything again, let us consider the following problem:

> _in a trade order, the stop level must be at the minimum level of the preceding bar for a long position and at the maximum level of the preceding bar for a short position._

It can be represented as follows in the code:

```
double stopLevel = iif( tradeDirection > 0, Low[ 1 ], High[ 1 ] );
```

It seems to be easy and clear, however, even these simple constructions can and must be pooled in small and simple functions to be used repeatedly.

Let us avoid conditional operator by placing it in the helper function for more general purposes:

```
double barPeakPrice( int barIndex, int peakDirection )
{
    return( iif( peakDirection > 0, High[ barIndex ], Low[ barIndex ] ) );
}
```

Now we can express calculation of stop levels as follows:

```
double stopLevel = barPeakPrice( 1, -tradeDirection );
```

Please don't let you be led by your first impressions if the difference seems to be quite insignificant. This variant has serious advantages:

- it expresses its purpose explicitly and in an invariant form;
- it stimulates writing the EA's code in a proper style;
- it is easier to read and makes further development simpler.

This is just one example. As a matter of fact, many standard elements of the code of Expert Advisors can be expressed in a similar form. You will be able to do it by yourselves, too. So I highly recommend you to analyze your code and revise it in this way.

### 5\. Why All This?

The concept described above has, in my opinion, the following serious advantages:

- reducing the source code without losing functionality and, subsequently, lower time consuming during development and adjusting of trading systems;
- reducing of amounts of potential errors;
- increasing probability of detection of the existing errors;
- simplifying of further modification of the Expert Advisor (the changes automatically apply to both long and short signals and positions).

I noticed only one disadvantage: This concept can cause some slight difficulties in understanding and studying at earlier stages. However, this disadvantage is more than compensated by the advantages listed above. Moreover, it is just the matter of time and some experience - and development of invariant code becomes natural and easy.

### Conclusion

The above approach to writing Expert Advisors' codes in MQL4 is based on the use and effective representation of the notion of _trade direction_. It allows to avoid doubling some practically identical code areas that can be normally seen in Expert Advisors written using conventional approach. The utilizing of the described method results in essential reducing the volume of the source code with all advantages following from this.

The examples are given in the article in order to help newbies and some more experienced developers of trading systems in elaborating their existing codes if they wish. These examples being complemented by some author's recommendations will help them to write compact codes to be invariant with respect to the trade direction.

The examples considered were rather simple for the purposes of easier reading and understanding the matter. However, the described approach was successfully used in realization of much more complicated systems that had applied such technical analysis tools as trendlines, channels, Andrews' Pitchfork, Elliot Waves, and some other middle and advanced methods of market analysis.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1491](https://www.mql5.com/ru/articles/1491)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1491.zip "Download all attachments in the single ZIP archive")

[MACD\_Sample-2.mq4](https://www.mql5.com/en/articles/download/1491/MACD_Sample-2.mq4 "Download MACD_Sample-2.mq4")(5.69 KB)

[MACD\_Sample.mq4](https://www.mql5.com/en/articles/download/1491/MACD_Sample.mq4 "Download MACD_Sample.mq4")(5.48 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to Make the Detection and Recovery of Errors in an Expert Advisor Code Easier](https://www.mql5.com/en/articles/1473)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39350)**
(3)


![miky](https://c.mql5.com/avatar/avatar_na2.png)

**[miky](https://www.mql5.com/en/users/miky)**
\|
29 Aug 2007 at 19:05

thanks for a artical this.

I do have a question , could you please make some explanation about this sentence

int orderDirection()

{

return( 1 - 2 \* ( OrderType() % 2 ) );

}

what particulary "% 2" does here?

and what value contains "OrderType()" for buy and [sell orders](https://www.mql5.com/en/docs/constants/tradingconstants/enum_book_type "MQL5 documentation: Trade Orders in Depth Of Market") (1 or 2 or smth else)?


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
21 Feb 2008 at 15:45

**niniu:**

thanks for a artical this.

I do have a question , could you please make some explanation about this sentence

int orderDirection()

{

return( 1 - 2 \* ( OrderType() % 2 ) );

}

what particulary "% 2" does here?

and what value contains "OrderType()" for buy and sell orders (1 or 2 or smth else)?

Yes... very interesting article and gives many ideas and methods for reducing code size of EA. However, I have one problem with the conclusion and it is that this invariant code style makes code harder to _quickly_ "see at a glance" what's going on - imho. I totally agree with premises put forward, yet when code becomes effectively _subjectively interpretive_ due to application of reader's own knowledgebase - I would suggest that issues can result. Either due to misunderstandings and hence how to effectively employ code or, due to misundersandings causing reader to modify/tweak code to _fit in_ with their 'misunderstandings'. Regardless of issue, result is predictable - failure.

I am (in the scheme of software universe) very old... Simply put, I do not (cannot or will not) _bother_ to remember what all my code does. It makes no difference if I write pages of comment or not - bottom line is the " **see at a glance**" moment in which I find myself reading the code. Sure, _I believe_ code 'works' (eg, I have distant memories of having sweated over hot keyboard-lol) but I must know in _that moment_ that I also trully _understand_ the code **now!** This statement must go against the 'black box' paradigm, yes? eg, design, code, test, _fully document_ and... forget just _how_ the black box really functions! - just use it. Sure, great... but this is **money on the table** we are talking about, yes? That will make even me constantly look over my shoulder "just in case" something has been missed, not understood, etc.

_Cobol_  springs to mind! Very HLL indeed, yes? I know of many systems which suffered massive issues simply because of using such a HLL as Cobol. Nobody ever thought to look under the hood occasionally to make sure system firing on all cylinders... Inevitable issues of catastrophic nature resulted... **always**. Similarly, this articles approach is analog (imho) to Cobol and all the \[potential\] issues therein.

I do not agree this is more understandable: barPeakPrice(1, -tradeDirection);

Each and everytime I would use, I would have to take time to figure out what "minus tradeDirection" actually did - enevitably leading me to again, always finding the code and figuring out _step-by-painfull-step_ what the heck this code was doing! This is of course _for me_ and others may be massively fleet of mind or worse still... just **assume** that all is a-ok.....

Similiarly, if(MacdCurrent \\* tradeDirection \> 0 && (MacdCurrent \- SignalCurrent) \*

tradeDirection < 0 && (MacdPrevious \- SignalPrevious) \\* tradeDirection \> 0

    && MathAbs(MacdCurrent) \> (MACDCloseLevel \\* Point))

compared to, if(MacdCurrent>0 && MacdCurrent<SignalCurrent && MacdPrevious>SignalPrevious &&

MacdCurrent>(MACDCloseLevel\*Point))

is not imho very _clear_. Conditionals complexity increase is comparable to mental complexity increase in likelihood of meaningful understand... bottom line: BIG if() clauses cause many (not just me) to run away _faster than the proverbial 'speeding bullet'..._ I do not buy into the thinking that speed is all important in EA coding. Have you ever monitored how many data ticks incoming there are per period? Missing a datum is not, I would argue, hazardous to equity. However, complex statements like above invariant  if() are massively hazardous to equity. Additionally, given the serious shortcommings with MQL4's parsing of conditionals (ie, **NO** early exit and all the implications therein of such mishandling of a universally understood (and expected) _and practiced_ technique) I personally stay well clear of even hint of multi-clause if()'s. Is, of course, free world and each will do as wish... just not on my watch - that's all - lol.

Notwithstanding above - I reiterate that I have found this article **most interesting** and I will be taking on board, those snippets which I feel can benifit my EA robustness.

Thanking you  [Крамарь Роман](https://www.mql5.com/en/users/bstone) for article ;-)

((( Babel Fish gives me _Kramar' the novel_  but somehow not think correct? _Constantly wish had learnt Russian - .ru forum is awesome, if somewhat cryptic for non Russians!!!_ )))

**Hello  niniu**, May I offer my interpretation/answer to your reply?:

OrderType() returns integer from 0..5 (see: [https://docs.mql4.com/trading/ordertype](https://docs.mql4.com/trading/ordertype) and specifically [https://docs.mql4.com/constants/tradingconstants/orderproperties](https://docs.mql4.com/constants/tradingconstants/orderproperties))

"% 2" is _division remainder_ mathmatical operation ( **%** also called the modulus operator).  Example using _OrderType()_ is: 0%2=0, 1%2=1, 2%2=0, 3%2=1, 4%2=0, 5%2=1;  Notice how result can only ever be 0 or 1.  Now, by multiplying 0 or 1 by 2 we get 0 or 2, yes? The final part of return() statement is: 1 - 0 or 1 - 2, yes?

Which resolves to 1-0= **+1** =long trade direction: **+1**  or  1-2= **-1** =short trade direction: - **1**

Meaning BUY/LONG _OrderType()'s_ always cause function to return **+1**  and  SELL/SHORT _OrderType()'s_ always cause function to return **-1**

HTH

![William Roeder](https://c.mql5.com/avatar/2016/12/584F20BE-8336.png)

**[William Roeder](https://www.mql5.com/en/users/whroeder1)**
\|
18 Jul 2010 at 19:10

My approach was similar but I like your IIF better. I used a function setDIR(int op) that set global variables now.open, now.close, and DIR to Bid/Ask, Ask/Bid, and +1/-1. Thus I could use now.open in OrderSend and now.close in [stop loss](https://www.metatrader5.com/en/terminal/help/trading/general_concept "User Guide: Types of orders") calculations

Additionally I used two different stop losses so I needed a max function:

```
stopBars    = Low[Lowest(...)]...,
stopATR     = now.close -DIR* (SL.Trail.ATR*ATR.value + SL.Trail.Pips*pips2dbl),
SL          = MathMaxDIR(stop.bars, stopATR);
double  MathMaxDIR(double a, double b, double d=0){
    if (0 == d) d = DIR;
    if (d > 0)  return( MathMax(a,b) );     else    return( MathMin(a,b) ); }
double  MathMinDIR(double a, double b, double d=0){
    if (0 == d) d = DIR;
    if (d > 0)  return( MathMin(a,b) );     else    return( MathMax(a,b) ); }
```

![Creation of an Automated Trading System](https://c.mql5.com/2/13/196_9.png)[Creation of an Automated Trading System](https://www.mql5.com/en/articles/1426)

You must admit that it sounds alluringly - you become a fortunate possessor of a program that can develop for you a profitable automated trading system (ATC) within a few minutes. All you need is to enter desirable inputs and press Enter. And - here you are, take your ATC tested and having positive expected payoff. Where thousands of people spend thousands of hours on developing that very unique ATC, which will "wine and dine", these statements soundб to put it mildly, very hollow. On the one hand, this really looks a little larger than life... However, to my mind, this problem can be solved.

![Practical Use of the Virtual Private Server (VPS) for Autotrading](https://c.mql5.com/2/14/373_44.png)[Practical Use of the Virtual Private Server (VPS) for Autotrading](https://www.mql5.com/en/articles/1478)

Autotrading using VPS. This article is intended exceptionally for autotraders and autotrading supporters.

![MQL4 Language for Newbies. Technical Indicators and Built-In Functions](https://c.mql5.com/2/15/466_27.gif)[MQL4 Language for Newbies. Technical Indicators and Built-In Functions](https://www.mql5.com/en/articles/1496)

This is the third article from the series "MQL4 Language for Newbies". Now we will learn to use built-in functions and functions for working with technical indicators. The latter ones will be essential in the future development of your own Expert Advisors and indicators. Besides we will see on a simple example, how we can trace trading signals for entering the market, for you to understand, how to use indicators correctly. And at the end of the article you will learn something new and interesting about the language itself.

![Price Forecasting Using Neural Networks](https://c.mql5.com/2/14/395_11.png)[Price Forecasting Using Neural Networks](https://www.mql5.com/en/articles/1482)

Many traders speak about neural networks, but what they are and what they really can is known to few people. This article sheds some light on the world of artificial intelligence. It describes, how to prepare correctly the data for the network. Here you will also find an example of forecasting using means of the program Matlab.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/1491&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083243942926882739)

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