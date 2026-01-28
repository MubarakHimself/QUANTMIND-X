---
title: Multibot in MetaTrader: Launching multiple robots from a single chart
url: https://www.mql5.com/en/articles/12434
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:26:23.869446
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=olkvxvryitpcmfsiiuflukxpmxlyxqsy&ssn=1769192782442601509&ssn_dr=0&ssn_sr=0&fv_date=1769192782&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12434&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Multibot%20in%20MetaTrader%3A%20Launching%20multiple%20robots%20from%20a%20single%20chart%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919278261622010&fz_uniq=5071833451771932494&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/12434#para1)
- [Problem statement and limits of applicability](https://www.mql5.com/en/articles/12434#para2)
- [Differences between the MetaTrader 4 and MetaTrader 5 terminals in terms of using a multibot](https://www.mql5.com/en/articles/12434#para3)
- [The nuances of building a universal template](https://www.mql5.com/en/articles/12434#para4)
- [Writing a universal template](https://www.mql5.com/en/articles/12434#para5)
- [Conclusion](https://www.mql5.com/en/articles/12434#para6)

### Introduction

In the world of financial markets, automated trading systems have become an integral part of the decision-making process. These systems can be configured to analyze the market, make entry and exit decisions, and execute trades using predefined rules and algorithms. However, setting up and running robots on multiple charts can be a time consuming task. Each robot should be configured individually for each chart, which requires additional effort.

In this article, I will show you my implementation of a simple template that allows you to create a universal robot for multiple charts in MetaTrader 4 and 5. Our template will allow you to attach the robot to one chart, while the rest of the charts will be processed inside the EA. Thus, our template greatly simplifies the process of setting up and running robots on multiple charts, saving traders time and effort. In this article, I will consider in detail the process of creating such a robot in MQL5 from the idea to the test.

### Problem statement and limits of applicability

This idea came to me not so long ago, although I I have been observing similar decisions from professional sellers for a long time. In other words, I'm not the first and not the last one to come up with an idea in this field, but as always, some conditions must arise in order for the programmer to begin to come to such decisions. The main reason for developing such Expert Advisors in the MQL5 store is the desire for user comfort. However, in my case there was a slightly different motivation. My motivation was that I first had to test either several strategies simultaneously for several instruments, or the same strategy, but in order to see its multicurrency characteristics.

In addition, a very important factor when testing a strategy in the tester, especially in multi-currency mode, is a **general curve** of profitability, which is the basis of any evaluation of automatic trading systems when backtesting on historical data. When testing trading systems separately on one instrument, it is quite difficult to combine such reports later. I am not aware of such tools, at least for MetaTrader 5. As for the fourth version of the terminal, there is one unofficial tool for such manipulations. I used it in at least one article, but of course such an approach is not preferable.

In addition to the testing process, there is an equally important process of automatic trading itself, and synchronization of similar EAs that work independently, each on its own chart. If there are too many such charts, it may require additional computer resources **slowing down or worsening trading performance** and **leading to unexpected errors** and other unpleasant incidents that can have a detrimental effect on the final trading result. For each such EA, we need to come up with **unique order IDs**, protection against high-frequency server requests, as well as many other things that are not obvious at first glance.

**Processing of the graphical part of the EA** is a separate and very sensitive issue. Now all the more or less skillful EA creators make at least a minimal version of some indication on the chart the EA is attached to. This way the EA looks more serious and inspires more confidence, and finally, almost always, displaying some information on the chart to a user sometimes allows for more effective control over the EA trading process. Besides, it is possible to add elements for manual control if necessary. All this is called a **user interface**. While distributing such EAs by charts, **the load** on updating both graphical, textual and numerical information in the interfaces increases exponentially. Of course, when using a multi-template, we have one interface that requires the **minimum amount of resources** fro the terminal.

Of course, such a template does not solve all problems, but nevertheless, it helps me a lot in my projects. I use different robots and, generally, all approaches have the right to exist, but I think many novice programmers may find this pattern useful. It is not necessary to copy it completely, but if you wish, you can easily adjust it to suit your needs. My goal is not to give you something extraordinary, but to **show and explain** **one of the options** of solving such a problem.

### Differences between the MetaTrader 4 and MetaTrader 5 terminals in terms of using a multibot

What I like about the latest MetaTrader 5 is the power of its tester, which gives you all the features you need to test on multiple instruments at the same time provided that you use the approach to EA development stated above. The tester **auto synchronizes quotes** by time providing you a clearly synchronized profitability curve on a time scale. MetaTrader 4 has no such functionality. I think, this is its biggest disadvantage. Nevertheless, it is worth noting that MetaQuotes is doing its best to support the fourth terminal and **its popularity is still high**. As an active user of MetaTrader 4, I can say that these shortcomings are not as significant as they may seem.

The MQL4 language was recently updated toMQL5\. This means that when writing similar templates like ours, we will have a minimum of differences in the code. It is in my good tradition to try to implement things for **both terminals**, so you will receive a template for both terminals. Such improvements to the old terminal, among other things, allow us to use the following functions that we really need:

- CopyClose \- request for bar closing prices
- CopyOpen \- request for bar opening prices
- CopyHigh \- request for bar peaks
- CopyLow \- request for bar lows
- CopyTime \- request for bar opening time
- SymbolInfoTick \- request for the last incoming tick for the requested symbol
- SymbolInfoInteger \- request for symbol data, which can be described by integers and numbered lists
- SymbolInfo\\*\\*\\*\*\*\*\* \- other functions we need

These features are present in both MQL4 and MQL5\. These functions allow you to get **bar data for any symbol and period**. Thus, the only unpleasant difference between the tester of the fourth and fifth versions is the fact that these functions in the fourth terminal will work only for the current chart on which testing is being carried out, and the rest of the requests will simply inform you that **there is no data** due to the peculiarities of MetaTrader 4 tester. Therefore, when testing our template, you will only get trading on the selected symbol, and only one of the profit curves for a single robot.

In the fifth terminal, you will already receive trading for all the requested symbols **and the common line** of profitability. As for **applying in trading**, when trading directly with such a robot in both terminals, **you receive** the **full performance of such a template**. In other words, the difference is only in the tester. But even in such cases, you can get away with the fact that when creating an EA, it is better to start with the version for MetaTrader 5. After all the necessary tests, you can quickly make the version for MetaTrader 4.

Of course, there are a number of differences that I have not covered. I just want to emphasize the importance of some of them, since these nuances must be known when building an elaborate structure for such a template. MetaTrader 5 is definitely better than its predecessor, but nevertheless I have no desire to get rid of the fourth terminal, because **in many situations, its demand for computational resources is not so great compared to the fifth one**. Both tools are still good.

### The nuances of building a universal template

To build such a template, you should understand how the terminal works, what an Expert Advisor is, and what a MetaTrader chart is. In addition, you should understand that **each chart is a separate object**. Each such chart can be associated with several indicators and **only one EA**. There may be **several identical** charts. Several charts are usually made in order to run several different EAs on one **symbol period** or to run **multiple copies** of one EA **with different settings**. Understanding these subtleties, we should come to the conclusion that in order to abandon multiple charts in favor of our template, we will have to implement all this inside our template. This can be represented in the form of a diagram:

![objects structure](https://c.mql5.com/2/53/diagram_basic.png)

Separately, it should be said about **ticks**. The disadvantage of this approach is that we will not be able to subscribe to the handler for the appearance of a new tick for each chart. We **will have to apply** ticks from the chart our robot is working on, or use the **timer**. Ultimately, this will mean unpleasant moments for tick robots, which are as follows:

- We will have to write custom OnTick handlers
- These handlers will have to be implemented as a derivative of OnTimer
- Ticks will not be perfect because OnTimer works with a delay (the value of the delay is not important, but its presence is important)
- To get ticks, you need the SymbolInfoTick function

I think for those who count every millisecond, this can be an irresistible moment, especially for those who love arbitrage. However, I do not emphasize this in my template. Over the years of building different systems, I came to the **bar trading** paradigm. This means that trading operations and other calculations for the most part occur when a new bar appears. This approach has a number of obvious advantages:

- Inaccuracy in determining the start of a new bar does not significantly affect trading
- The longer the period of the bar, the less this influence.
- Discretization in the form of bars provides an increase in testing speed by orders of magnitude
- The same quality of testing both when testing on real ticks and on artificial ones

The approach teaches a certain paradigm of building EAs. This paradigm **eliminates many problems associated with tick EAs**, speeds up the testing process, provides a higher mathematical expectation of profit, which is the main obstacle, and also saves a lot of time and computing power. I think, we can find many more advantages, but I think this is enough in the context of this article.

To implement our template, it is not necessary to implement **the entire structure of the workspace** of the trading terminal inside our template, but it is enough just to implement a separate chart for each robot. This is not the most optimal structure, but if we agree that each individual instrument will be present only once in the list of instruments, then this optimization is not required. It will look like this:

![our realization](https://c.mql5.com/2/53/our_realization.png)

We have implemented the simplest structure for the implementation of charts. Now it is time to think aboutinputs of such a template, and more importantly, **how to take into account the dynamic number of charts and EAs for each situation** within the allowable possibilities of the MQL5 language. The only way to solve this problem is **using string input variables**. A string allows us to store a very large amount of data. In fact, in order to describe all the necessary parameters for such a template, we will need **dynamic arrays** in the input data. Of course, no one will implement such things simply because few people would use such opportunities. **The string** is our **dynamic array**, in which we can put anything we want. So let's use it. For my simplest template, I decided to introduce three variables like this:

- Charts \- our charts (list)

- Chart Lots \- lots for trading (list)
- Chart Timeframes \- chart periods (list)

In general, we can combine all this data into **a single string**, but then its structure will be complex and it will be difficult for a potential user to figure out how to correctly describe the data. In addition, it will be very easy to make mistakes in filling it out and we can get a lot of very unpleasant things when using it, not to mention the incredible complexity of the conversion function, which will take this data out of the strings. I saw similar solutions among sellers and in general they did everything right. All data is simply listed separated by commas. At the start of the EA, these **data is taken out** from a string using special functions **and filled in the corresponding dynamic arrays**, which are then used in the code. We will also follow this path. We can add more similar strings with identical enumeration rules. I decided to use **":"** as a separator. If we use a comma, then it is not clear how to deal with double arrays such as Chart Lots. It is possible to add **more** of such string variables, and in general it is possible to construct **even more complete and versatile template**, but my task here is only to show how to implement this and give you the first version of the template you can **modify quickly and easily**.

It is not enough to implement such arrays, it is also necessary to implement common variables, for example:

- Work Timeframe For Unsigned \- a chart period where it is not specified
- Fix Lot For Unsigned \- a lot where it is not specified

The Charts list should be filled. The same action is optional for Chart Lots and Chart Timeframes. For example, we may take **single lots for all charts** and **the same period for all charts**. A similar functionality will be implemented in our template. It is desirable to apply such implementation rules wherever possible to ensure brevity and clarity when setting the input parameters of an EA built on the basis of such templates.

Let's now define a few more important variables for a minimal implementation of such a pattern:

- Last Bars Count \- the number of last bars for a chart that we store for each chart
- Deposit For Lot \- deposit for the use of a specified lot
- First Magic \- unique ID for a separate EA's deals

I think, the first variable is pretty clear. The second variable is much harder to grasp. This is how I regulate the **auto lot** in my EAs. If I set it to "0", then I inform the algorithm that it only needs to trade a **fixed lot** specified **in the corresponding string** or **in a shared variable** we have considered above. Otherwise, I set the required deposit so that the lot **specified in the settings** can be applied. It is easy to understand that with a smaller or larger deposit, this lot changes its value according to the equation:

- Lot = Input Lot \* ( Current Deposit / Deposit For Lot )

I think everything should be clear now. If we want a fixed lot, we set zero, and in other cases we adjust the deposit in the input settings to the risks. I think, it is cheap and cheerful. If necessary, you can change the risk assessment approach for auto lot, but I personally like this option, it makes no sense to overthink it.

It is worth mentioning about synchronization, and in particular about such an issue as setting the **Expert Magic Number**. When trading with EAs or even in a mixed form, all self-respecting programmers pay special attention to this particular variable. The thing is that when using multiple EAs, it is very important to ensure that each such EA has **a unique ID**. Otherwise, when working with orders, deals or positions, you will get a complete mess and your strategies will stop working correctly, and in most cases they will stop working completely. I hope I do not have to explain why. **Each time an EA is placed on the chart**, we need to configure these IDs and make sure that they **do not repeat**. Even a single mistake may lead to **disastrous consequences**. In addition, if you accidentally close the chart with the EA, you will have to **reconfigure** it anew. As a result, the probability of a mistake is greatly increased. In addition, it is very unpleasant in many other aspects. For example, you close the chart and forget which ID was used there. In this case, you will have to dig into the trading history to look for it. Without the ID, the newly restarted EA may work incorrectly and many more unpleasant things can happen.

Using a template like mine **sets us free** from such control and **minimizes** possible errors since we need to set only the starting ID in the EA settings, while the rest of the IDs will be **automatically generated** using an increment and assigned to the corresponding copies of EAs. This process will happen at each restart **automatically**. Anyway, remembering only one starting ID is much easier than remembering some random ID in the middle of the way.

### Writing a universal template

It is time to implement the template. I will try to omit excessive elements, so everyone who needs this template may download and see the rest in the source code. Here I will show **only things that are directly related to our ideas**. Stop levels and other parameters are defined by users. You can find my implementation in the source code. First, let's define our input variables, which we will definitely need:

```
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input string SymbolsE="EURUSD:GBPUSD:USDCHF:USDJPY:NZDUSD:AUDUSD:USDCAD";//Charts
input string LotsE="0.01:0.01:0.01:0.01:0.01:0.01:0.01";//Chart Lots
input string TimeframesE="H1:H1:H1:H1:H1:H1:H1";//Chart Timeframes
input int LastBars=10;//Last Bars Count
input ENUM_TIMEFRAMES TimeframeE=PERIOD_M1;//Work Timeframe For Unsigned
input double RepurchaseLotE=0.01;//Fix Lot For Unsigned
input double DepositForRepurchaseLotE=0.00;//Deposit For Lot (if "0" then fix)
input int MagicE=156;//First Magic
```

Here you can see an example of filling in **string variables** reflecting our **dynamic arrays**, just like the example of shared variables. By the way, this code will look the same both in MQL4 and in MQL5\. I tried to make everything as similar as possible.

Now let's decide how we will **get** our string data. This will be done by the corresponding function, but first we will create arrays where our function will add the data obtained from the strings:

```
//+------------------------------------------------------------------+
//|Arrays                                                            |
//+------------------------------------------------------------------+
string S[];// Symbols array
double L[];//Lots array
ENUM_TIMEFRAMES T[];//Timeframes array
```

The following function fills in these arrays:

```
//+------------------------------------------------------------------+
//| Fill arrays                                                      |
//+------------------------------------------------------------------+
void ConstructArrays()
   {
      int SCount=1;
      for (int i = 0; i < StringLen(SymbolsE); i++)//calculation of the number of tools
         {
         if (SymbolsE[i] == ':')
            {
            SCount++;
            }
         }
      ArrayResize(S,SCount);//set the size of the character array
      ArrayResize(CN,SCount);//set the size of the array to use bars for each character
      int Hc=0;//found instrument index
      for (int i = 0; i < StringLen(SymbolsE); i++)//building an array of tools
         {
         if (i == 0)//if we just started
            {
            int LastIndex=-1;
            for (int j = i; j < StringLen(SymbolsE); j++)
               {
               if (StringGetCharacter(SymbolsE,j) == ':')
                  {
                  LastIndex=j;
                  break;
                  }
               }
            if (LastIndex != -1)//if no separating colon was found
               {
               S[Hc]=StringSubstr(SymbolsE,i,LastIndex);
               Hc++;
               }
            else
               {
               S[Hc]=SymbolsE;
               Hc++;
               }
            }
         if (SymbolsE[i] == ':')
            {
            int LastIndex=-1;
            for (int j = i+1; j < StringLen(SymbolsE); j++)
               {
               if (StringGetCharacter(SymbolsE,j) == ':')
                  {
                  LastIndex=j;
                  break;
                  }
               }
            if (LastIndex != -1)//if no separating colon was found
               {
               S[Hc]=StringSubstr(SymbolsE,i+1,LastIndex-(i+1));
               Hc++;
               }
            else
               {
               S[Hc]=StringSubstr(SymbolsE,i+1,StringLen(SymbolsE)-(i+1));
               Hc++;
               }
            }
         }
      for (int i = 0; i < ArraySize(S); i++)//assignment of the requested number of bars
         {
         CN[i]=LastBars;
         }
      ConstructLots();
      ConstructTimeframe();
   }
```

In short, here the amount of data in a string is calculated **thanks to the separators**. **Based on the first array, the size of all other arrays is set** similarly to an array with symbols, after which the symbols are filled first followed by such functions asConstruct Lots and ConstructTimeframe. Their implementation is similar to the implementation of this function with some differences. You can see their implementation in the source code. I have not added them to the article so as not to show the duplicate code.

Now we need to create the appropriate classes for the **virtual chart** and **the virtual robot** linked to it accordingly. Let's start by defining that virtual charts and EAs will be stored in arrays:

```
//+------------------------------------------------------------------+
//| Charts & experts pointers                                        |
//+------------------------------------------------------------------+
Chart *Charts[];
BotInstance *Bots[];
```

Let's start from the **chart class**:

```
//+------------------------------------------------------------------+
//| Chart class                                                      |
//+------------------------------------------------------------------+
class Chart
   {
   public:
   datetime TimeI[];
   double CloseI[];
   double OpenI[];
   double HighI[];
   double LowI[];
   string BasicSymbol;//the base instrument that was extracted from the substring
   double ChartPoint;//point size of the current chart
   double ChartAsk;//Ask
   double ChartBid;//Bid
   datetime tTimeI[];//auxiliary array to control the appearance of a new bar
   static int TCN;//tcn
   string CurrentSymbol;//symbol
   ENUM_TIMEFRAMES Timeframe;//timeframe
   int copied;//how much data is copied
   int lastcopied;//last amount of data copied
   datetime LastCloseTime;//last bar time
   MqlTick LastTick;//last tick fos this instrument

   Chart()
      {
      ArrayResize(tTimeI,2);
      }

   void ChartTick()//this chart tick
      {
      SymbolInfoTick(CurrentSymbol,LastTick);
      ArraySetAsSeries(tTimeI,false);
      copied=CopyTime(CurrentSymbol,Timeframe,0,2,tTimeI);
      ArraySetAsSeries(tTimeI,true);
      if ( copied == 2 && tTimeI[1] > LastCloseTime )
         {
         ArraySetAsSeries(CloseI,false);
         ArraySetAsSeries(OpenI,false);
         ArraySetAsSeries(HighI,false);
         ArraySetAsSeries(LowI,false);
         ArraySetAsSeries(TimeI,false);
         lastcopied=CopyClose(CurrentSymbol,Timeframe,0,Chart::TCN+2,CloseI);
         lastcopied=CopyOpen(CurrentSymbol,Timeframe,0,Chart::TCN+2,OpenI);
         lastcopied=CopyHigh(CurrentSymbol,Timeframe,0,Chart::TCN+2,HighI);
         lastcopied=CopyLow(CurrentSymbol,Timeframe,0,Chart::TCN+2,LowI);
         lastcopied=CopyTime(CurrentSymbol,Timeframe,0,Chart::TCN+2,TimeI);
         ArraySetAsSeries(CloseI,true);
         ArraySetAsSeries(OpenI,true);
         ArraySetAsSeries(HighI,true);
         ArraySetAsSeries(LowI,true);
         ArraySetAsSeries(TimeI,true);
         LastCloseTime=tTimeI[1];
         }
      ChartBid=LastTick.bid;
      ChartAsk=LastTick.ask;
      ChartPoint=SymbolInfoDouble(CurrentSymbol,SYMBOL_POINT);
      }
   };
int Chart::TCN = 0;
```

The class has only one function, which **controls the update of ticks and bars**, as well as the necessary **fields for identifying some necessary parameters of a particular chart**. Some parameters are missing there. If desired, you can add the missing ones by adding their update, for example, like updating ChartPoint. Bar arrays are made in MQL4 style. I still find it very convenient to work with predetermined arrays in MQL4. It is very convenient when you know that **zero bar is the current bar**. Anyway, this is just my vision. You are free to follow the one of your own.

Now we need to describe **the class of a separate virtual EA**:

```
//+------------------------------------------------------------------+
//| Bot instance class                                               |
//+------------------------------------------------------------------+
class BotInstance//expert advisor object
   {
   public:
   CPositionInfo  m_position;// trade position object
   CTrade         m_trade;// trading object
   ///-------------------this robot settings----------------------
   int MagicF;//Magic
   string CurrentSymbol;//Symbol
   double CurrentLot;//Start Lot
   int chartindex;//Chart Index
   ///------------------------------------------------------------


   ///constructor
   BotInstance(int index,int chartindex0)//load all data from hat using index, + chart index
      {
      chartindex=chartindex0;
      MagicF=MagicE+index;
      CurrentSymbol=Charts[chartindex].CurrentSymbol;
      CurrentLot=L[index];
      m_trade.SetExpertMagicNumber(MagicF);
      }
   ///

   void InstanceTick()//bot tick
      {
      if ( bNewBar() ) Trade();
      }

   private:
   datetime Time0;
   bool bNewBar()//new bar
      {
      if ( Time0 < Charts[chartindex].TimeI[1] && Charts[chartindex].ChartPoint != 0.0 )
         {
         if (Time0 != 0)
            {
            Time0=Charts[chartindex].TimeI[1];
            return true;
            }
         else
            {
            Time0=Charts[chartindex].TimeI[1];
            return false;
            }
         }
      else return false;
      }

   //////************************************Main Logic********************************************************************
   void Trade()//main trade function
      {
      //Close[0]   -->   Charts[chartindex].CloseI[0] - example of access to data arrays of bars of the corresponding chart
      //Open[0]   -->   Charts[chartindex].OpenI[0] -----------------------------------------------------------------------
      //High[0]   -->   Charts[chartindex].HighI[0] -----------------------------------------------------------------------
      //Low[0]   -->   Charts[chartindex].LowI[0] -------------------------------------------------------------------------
      //Time[0]   -->   Charts[chartindex].TimeI[0] -----------------------------------------------------------------------

      if ( true )
         {
            CloseBuyF();
            //CloseSellF();
         }
      if ( true )
         {
            BuyF();
            //SellF();
         }

      }

   double OptimalLot()//optimal lot calculation
      {
      if (DepositForRepurchaseLotE != 0.0) return CurrentLot * (AccountInfoDouble(ACCOUNT_BALANCE)/DepositForRepurchaseLotE);
      else return CurrentLot;
      }

   //here you can add functionality or variables if the trading function turns out to be too complicated
   //////*******************************************************************************************************************

   ///trade functions
   int OrdersG()//the number of open positions / orders of this virtual robot
      {
      ulong ticket;
      bool ord;
      int OrdersG=0;
      for ( int i=0; i<PositionsTotal(); i++ )
         {
         ticket=PositionGetTicket(i);
         ord=PositionSelectByTicket(ticket);
         if ( ord && PositionGetInteger(POSITION_MAGIC) == MagicF && PositionGetString(POSITION_SYMBOL) == CurrentSymbol )
            {
            OrdersG++;
            }
         }
      return OrdersG;
      }

   /////////********/////////********//////////***********/////////trade function code block
   void BuyF()//buy market
      {
      double DtA;
      double CorrectedLot;

      DtA=double(TimeCurrent())-GlobalVariableGet("TimeStart161_"+IntegerToString(MagicF));//unique bot marker last try datetime
      if ( (DtA > 0 || DtA < 0) )
         {
         CorrectedLot=OptimalLot(Charts[chartindex]);
         if ( CorrectedLot > 0.0 )
            {
            //try buy logic
            }
         }
      }

   void SellF()//sell market
      {
      //Same logic
      }

   void CloseSellF()//close sell position
      {
      ulong ticket;
      bool ord;
      for ( int i=0; i<PositionsTotal(); i++ )
         {
         ticket=PositionGetTicket(i);
         ord=PositionSelectByTicket(ticket);
         if ( ord && PositionGetInteger(POSITION_MAGIC) == MagicF && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL
         && PositionGetString(POSITION_SYMBOL) == Charts[chartindex].CurrentSymbol )
            {
            //Close Sell logic
            }
         }
      }

   void CloseBuyF()//close buy position
      {
      //same logic
      }

   bool bOurMagic(ulong ticket,int magiccount)//whether the magic of the current deal matches one of the possible magics of our robot
      {
      int MagicT[];
      ArrayResize(MagicT,magiccount);
      for ( int i=0; i<magiccount; i++ )
         {
         MagicT[i]=MagicE+i;
         }
      for ( int i=0; i<ArraySize(MagicT); i++ )
         {
         if ( HistoryDealGetInteger(ticket,DEAL_MAGIC) == MagicT[i] ) return true;
         }
      return false;
      }
   /////////********/////////********//////////***********/////////end trade function code block
   };
```

I have removed some of the repetitive logic in order to reduce the amount of code. **This is the class**, in which **the entire algorithm of your EA is to be implemented**. The main functionality present in this class:

- Trade() \- main trading function that is called in the bar handler for the corresponding chart
- BuyF() \- buying by market function
- SellF() \- selling by market function
- CloseBuyF() \- function of closing buy positions by market
- CloseSellF() \- function of closing sell positions by market

This is the **minimum set** of functions for demonstrating trading by bars. For this demonstration, we just need to open any position and close it on the next bar. This is sufficient within the framework of this article. There is some additional functionality from this class that should complement the understanding:

- OrdersG() \- counting positions that are open on a specific symbol linked to the chart
- OptimalLot() \- preparing a lot before sending it to the trading function (selecting a fixed lot or calculating an auto lot)
- bOurMagic()  \- checking transactions from the history for compliance with the list of allowed ones (for sorting out a custom history only)

These functions may be needed to implement trading logic. It would also be reasonable to remind of **the new bar handler**:

- InstanceTick() \- tick simulation on a separate EA instance
- bNewBar() \- predicate for checking the appearance of a new bar (used inside InstanceTick)

If the predicate shows a new bar, then the Trade function is triggered. This is the function, in which **the main trading logic** is to be set. The connection with the corresponding chart is carried out using the chartindex variable assigned when the instance is created. So each EA instance knows the chart it should take a quote from.

Now let's consider **the process of creating** the **virtual charts** and **EAs** themselves. **Virtual charts** are created first:

```
//+------------------------------------------------------------------+
//| Creation of graph objects                                        |
//+------------------------------------------------------------------+
void CreateCharts()
   {
   bool bAlready;
   int num=0;
   string TempSymbols[];
   string Symbols[];
   ConstructArrays();//array preparation
   int tempcnum=CN[0];
   Chart::TCN=tempcnum;//required number of stored bars for all instruments
   for (int j = 0; j < ArraySize(Charts); j++)//fill in all the names and set the dimensions of all time series, each graph
      {
      Charts[j] = new Chart();
      Charts[j].lastcopied=0;
      ArrayResize(Charts[j].CloseI,tempcnum+2);//assign size to character arrays
      ArrayResize(Charts[j].OpenI,tempcnum+2);//----------------------------------
      ArrayResize(Charts[j].HighI,tempcnum+2);//----------------------------------
      ArrayResize(Charts[j].LowI,tempcnum+2);//-----------------------------------
      ArrayResize(Charts[j].TimeI,tempcnum+2);//----------------------------------
      Charts[j].CurrentSymbol = S[j];//symbol
      Charts[j].Timeframe = T[j];//timeframe
      }
   ArrayResize(Bots,ArraySize(S));//assign a size to the array of bots
   }
```

After creating charts and setting the size of the array with virtual EAs, we need to create the instances of the EAs themselves and implement **the connection of virtual EAs with charts**:

```
//+------------------------------------------------------------------+
//| create and hang all virtual robots on charts                     |
//+------------------------------------------------------------------+
void CreateInstances()
   {
   for (int i = 0; i < ArraySize(S); i++)
      {
      for (int j = 0; j < ArraySize(Charts); j++)
         {
         if ( Charts[j].CurrentSymbol == S[i] )
            {
            Bots[i] = new BotInstance(i,j);
            break;
            }
         }
      }
   }
```

Connection is carried out using the "j" index set **in each instance of the virtual EA** when creating it. The corresponding variable shown above is highlighted there. Of course, all this can be done in many ways and much more elegantly, but I think that the main thing is that the general idea is clear.

All that is left is to show how **ticks are simulated** on each chart and EA associated with it:

```
//+------------------------------------------------------------------+
//| All bcharts & all bots tick imitation                            |
//+------------------------------------------------------------------+
void AllChartsTick()
   {
   for (int i = 0; i < ArraySize(Charts); i++)
      {
      Charts[i].ChartTick();
      }
   }

void AllBotsTick()
   {
   for (int i = 0; i < ArraySize(S); i++)
      {
      if ( Charts[Bots[i].chartindex].lastcopied >= Chart::TCN+1 ) Bots[i].InstanceTick();
      }
   }
```

The only thing I want to note is that this template **was obtained by reworking my more complex template**, which was intended for much more serious purposes, so there may be excessive elements here and there. I think, you can easily remove them and make the code more neat if you want.

In addition to the template, there is a simple interface, which, I think, can also come in handy, for example, when writing an order in freelance or for other purposes:

![](https://c.mql5.com/2/53/647883702904.png)

I left free space in this interface, it will be enough for three entries in case you do not have enough space. You can easily expand or change its structure completely if necessary. If we want to add the three missing fields in this particular example, we need to find the following places in the code:

```
//+------------------------------------------------------------------+
//| Reserved elements                                                |
//+------------------------------------------------------------------+

   "template-UNSIGNED1",//UNSIGNED1
   "template-UNSIGNED2",//UNSIGNED2
   "template-UNSIGNED3",//UNSIGNED3

   //LabelCreate(0,OwnObjectNames[13],0,x+Border+2,y+17+Border+20*5+20*5+23,corner,"","Arial",11,clrWhite,0.0,ANCHOR_LEFT);//UNSIGNED1
   //LabelCreate(0,OwnObjectNames[14],0,x+Border+2,y+17+Border+20*5+20*5+23+20*1,corner,"","Arial",11,clrWhite,0.0,ANCHOR_LEFT);//UNSIGNED2
   //LabelCreate(0,OwnObjectNames[15],0,x+Border+2,y+17+Border+20*5+20*5+23+20*2,corner,"","Arial",11,clrWhite,0.0,ANCHOR_LEFT);//UNSIGNED3

   ////////////////////////////
   //TempText="UNSIGNED1 : ";
   //TempText+=DoubleToString(NormalizeDouble(0.0),3);
   //ObjectSetString(0,OwnObjectNames[13],OBJPROP_TEXT,TempText);
   //TempText="UNSIGNED2 : ";
   //TempText+=DoubleToString(NormalizeDouble(0.0),3);
   //ObjectSetString(0,OwnObjectNames[14],OBJPROP_TEXT,TempText);
   //TempText="UNSIGNED3 : ";
   //TempText+=DoubleToString(NormalizeDouble(0.0),3);
   //ObjectSetString(0,OwnObjectNames[15],OBJPROP_TEXT,TempText);
   ///////////////////////////

```

The first three entries assign names of new elements on the interface, the second three ones are used when creating the interface at the start of the EA, while the last three are used in the function to update information on the interface. Now it is time to **test the performance of both templates**. **The tester visualizer will be sufficient** for a visual demonstration. I will show only the option for MetaTrader 5, because its visualizer is much better. Besides, the result of the work will clearly show everything that is needed to confirm the efficiency:

![checking using MetaTrader 5 tester visualization](https://c.mql5.com/2/53/t5ne498az_dydl8.png)

As you can see, we have uploaded all seven charts for the major Forex pairs. The visualization log shows that trading is in progress **for all listed symbols**. Trading is performed independently, as required. In other words, the EAs **trade each on their own chart** and **do not interact at all**.

### Conclusion

In this article, we reviewed the main nuances of building universal templates for the MetaTrader 4 and MetaTrader 5 terminals, made a simple but working template, analyzed the most important points of its work, and also confirmed its viability using the MetaTrader 5 tester visualizer. I think, it is pretty obvious by now that a template like this is not that complicated. In general, you can make various implementations of such templates, but it is obvious that such templates can be completely different while remaining applicable. The main thing is to understand the basic nuances of building such structures. If necessary, you can rework the templates for personal use.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12434](https://www.mql5.com/ru/articles/12434)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12434.zip "Download all attachments in the single ZIP archive")

[MultiTemplate.mq4](https://www.mql5.com/en/articles/download/12434/multitemplate.mq4 "Download MultiTemplate.mq4")(93.94 KB)

[MultiTemplate.mq5](https://www.mql5.com/en/articles/download/12434/multitemplate.mq5 "Download MultiTemplate.mq5")(91.41 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Multibot in MetaTrader (Part II): Improved dynamic template](https://www.mql5.com/en/articles/14251)
- [Brute force approach to patterns search (Part VI): Cyclic optimization](https://www.mql5.com/en/articles/9305)
- [Brute force approach to patterns search (Part V): Fresh angle](https://www.mql5.com/en/articles/12446)
- [OpenAI's ChatGPT features within the framework of MQL4 and MQL5 development](https://www.mql5.com/en/articles/12475)
- [Rebuy algorithm: Multicurrency trading simulation](https://www.mql5.com/en/articles/12579)
- [Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/447502)**
(4)


![Duc Anh Le](https://c.mql5.com/avatar/2024/1/65b470c3-0d80.png)

**[Duc Anh Le](https://www.mql5.com/en/users/expdal3)**
\|
19 May 2023 at 00:57

**MetaQuotes:**

New article [Multibot in MetaTrader: Launching multiple robots from a single chart](https://www.mql5.com/en/articles/12434) has been published:

Author: [Evgeniy Ilin](https://www.mql5.com/en/users/W.HUDSON "W.HUDSON")

Hi, this is really cool. May I ask if with this template, can I  use the \`BotsInstance\` class to attach another Expert from the terminal (external EA outside of this EA) ? If this is possible then means that we can actually backtest multiple EAs simultaneously in StrategyTester.

```
//+------------------------------------------------------------------+
//| create and hang all virtual robots on charts                     |
//+------------------------------------------------------------------+
void CreateInstances()
   {
   for (int i = 0; i < ArraySize(S); i++)
      {
      for (int j = 0; j < ArraySize(Charts); j++)
         {
         if ( Charts[j].CurrentSymbol == S[i] )
            {
            Bots[i] = new BotInstance(i,j);
            break;
            }
         }
      }
   }
```

![Evgeniy Ilin](https://c.mql5.com/avatar/2021/1/60171FA1-EC8E.png)

**[Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)**
\|
19 May 2023 at 09:05

**Duc Anh Le [#](https://www.mql5.com/en/forum/447502#comment_46979526):**

Hi, this is really cool. May I ask if with this template, can I  use the \`BotsInstance\` class to attach another Expert from the terminal (external EA outside of this EA) ? If this is possible then means that we can actually backtest multiple EAs simultaneously in StrategyTester.

yes, it is possible, but you will have to rewrite the adviser code a little, for this a template was made, it will allow you to trade and test such an adviser multicurrency. you just need to place the code in the body of the BotInstance class, and fit it to it

![Peng Peng Liu](https://c.mql5.com/avatar/avatar_na2.png)

**[Peng Peng Liu](https://www.mql5.com/en/users/yylnthz)**
\|
24 Dec 2023 at 03:02

It's not bad at all.


![Evgeniy Ilin](https://c.mql5.com/avatar/2021/1/60171FA1-EC8E.png)

**[Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)**
\|
6 Nov 2025 at 18:53

**Peng Peng Liu [#](https://www.mql5.com/ru/forum/445289#comment_57328723):**

It's not bad at all.

The initial version actually. Already improved this template a lot, just need to find time just for the article.


![Understand and Use MQL5 Strategy Tester Effectively](https://c.mql5.com/2/54/use_mql5_strategy_tester_effectively_avatar.png)[Understand and Use MQL5 Strategy Tester Effectively](https://www.mql5.com/en/articles/12635)

There is an essential need for MQL5 programmers or developers to master important and valuable tools. One of these tools is the Strategy Tester, this article is a practical guide to understanding and using the strategy tester of MQL5.

![Implementing an ARIMA training algorithm in MQL5](https://c.mql5.com/2/54/Implementing_an_ARIMA_training_algorithm_in_MQL5_Avatar.png)[Implementing an ARIMA training algorithm in MQL5](https://www.mql5.com/en/articles/12583)

In this article we will implement an algorithm that applies the Box and Jenkins Autoregressive Integrated Moving Average model by using Powells method of function minimization. Box and Jenkins stated that most time series could be modeled by one or both of two frameworks.

![Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://c.mql5.com/2/53/neural_network_experiments_p5_avatar.png)[Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://www.mql5.com/en/articles/12459)

Neural networks are an ultimate tool in traders' toolkit. Let's check if this assumption is true. MetaTrader 5 is approached as a self-sufficient medium for using neural networks in trading. A simple explanation is provided.

![MQL5 Wizard techniques you should know (Part 06): Fourier Transform](https://c.mql5.com/2/54/fourier_transform_avatar.png)[MQL5 Wizard techniques you should know (Part 06): Fourier Transform](https://www.mql5.com/en/articles/12599)

The Fourier transform introduced by Joseph Fourier is a means of deconstructing complex data wave points into simple constituent waves. This feature could be resourceful to traders and this article takes a look at that.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/12434&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071833451771932494)

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