---
title: MQL5 Cookbook: Handling BookEvent
url: https://www.mql5.com/en/articles/1179
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:19:37.267675
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/1179&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069345914088260487)

MetaTrader 5 / Examples


### Introduction

As is well known, the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") trading terminal is a multi-market platform, that facilitates trading on Forex, stock markets, Futures and Contracts for Difference. According to the [Freelance](https://www.mql5.com/en/job) section stats, the number of traders trading not only on Forex market is growing.

In this article I would like to introduce novice MQL5 programmers to the [BookEvent](https://www.mql5.com/en/docs/runtime/event_fire#bookevent) handling. This event is connected with Depth of Market—an instrument for trading stock assets and their derivatives. Forex traders, however, may find Depth of Market useful too. In ECN accounts, liquidity providers supply data on the orders, though only within their aggregator model. These accounts are becoming more popular.

### 1\. BookEvent

According to the [Documentation](https://www.mql5.com/en/docs), this event is generated when Depth of Market status changes. Let us agree that [BookEvent](https://www.mql5.com/en/docs/runtime/event_fire#bookevent) is a Depth of Market event.

Depth of Market is an array of orders, which differ in direction (sell and buy), price and volume. Prices in Depth of Market are close to the market ones and therefore are considered as the best.

![Fig.1 Depth of Market in MetaTrader 5](https://c.mql5.com/2/11/1__26.png)

Fig.1 Depth of Market in MetaTrader 5

In MetaTrader 5 an "order book" is named as the "Depth of Market" (Fig.1). Detailed information about Depth of Market can be found in the User Guide to the Client Terminal.

The [MqlBookInfo](https://www.mql5.com/en/docs/constants/structures/mqlbookinfo) structure providing information on the Depth of Market should be mentioned separately.

```
struct MqlBookInfo
  {
   ENUM_BOOK_TYPE   type;       // order type from the ENUM_BOOK_TYPE enumeration
   double           price;      // price
   long             volume;     // volume
  };
```

It contains three fields. Data on the order type, price and volume can be obtained by processing the order structure.

### 2\. Event Handler of BookEvent

The OnBookEvent() event handling function takes one constant as a parameter. It is a reference to a string parameter.

```
void OnBookEvent (const string& symbol)
```

The string parameter contains the name of the symbol, for which a Depth of Market event took place.

The event handler itself requires preliminary preparation. For the EA to handle a Depth of Market event, this event has to be subscribed for, using the built-in function MarketBookAdd(). Usually it is located in the block of EA's initialization. If a Depth of Market event has not been subscribed for, then the EA will ignore it.

Providing a possibility to unsubscribe from receiving events is considered to be a good programming practice. At deinitialization, it is necessary to unsubscribe from receiving that data by calling the MarketBookRelease() function.

The mechanisms of subscribing and unsubscribing from receiving data is similar to creating and processing a timer, which has to be activated before processing.

### 3\. BookEvent Handling Template

We are going to create a simple template of EA, which calls the OnBookEvent() function, and name it BookEventProcessor1.mq5.

The template comprises a minimum set of handlers, which are handlers of initialization and deinitialization of the EA, as well as the Depth of Market event handler.

The Depth of Market event handler itself is very simple:

```
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
   Print("Book event for: "+symbol);
//--- select the symbol
   if(symbol==_Symbol)
     {
      //--- array of the DOM structures
      MqlBookInfo last_bookArray[];

      //--- get the book
      if(MarketBookGet(_Symbol,last_bookArray))
        {
         //--- process book data
         for(int idx=0;idx<ArraySize(last_bookArray);idx++)
           {
            MqlBookInfo curr_info=last_bookArray[idx];
            //--- print
            PrintFormat("Type: %s",EnumToString(curr_info.type));
            PrintFormat("Price: %0."+IntegerToString(_Digits)+"f",curr_info.price);
            PrintFormat("Volume: %d",curr_info.volume);
           }
        }
     }
  }
```

As this Depth of Market event is a broadcast one (after subscription it will appear for all symbols), a required instrument has to be specified.

It should be noted though that in the latest few builds, changes have been introduced in the work of Depth of Market. In the current version (build 975) I did not find any signs of broadcasting. The handler was called only for the assigned symbol. To prove this fact, I simply arranged entering information to the "Experts Log".

To do that, we are going to use the built-in function MarketBookGet(). It will return all the information about Depth of Market, namely the MqlBookInfo, array of structures, which contains Depth of Market records for the specified symbol. It should be noted, that this array will vary in size, depending on a broker.

The template allows displaying values of the array structures in the Log.

The EA was launched for the [SBRF-12.14](https://www.mql5.com/go?link=http://www.moex.com/ru/contract.aspx?code=SBRF-12.14 "http://moex.com/ru/contract.aspx?code=SBRF-12.14") futures in a debug mode. The sequence of records in the Experts log will be as follows:

```
EL      0       11:24:32.250    BookEventProcessor1 (SBRF-12.14,M1)     Book event for: SBRF-12.14
MF      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
KP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7708
LJ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 6
MP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
HF      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7705
LP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 6
MJ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
GL      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7704
ON      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 3
MD      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
FR      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7703
PD      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 2
MN      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
DH      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7701
QQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 10
GH      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
QM      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7700
OK      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 1011
ER      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
KS      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7698
EE      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 50
OM      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
KI      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7696
IO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 21
QG      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
LO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7695
MK      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 1
QQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
KE      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7694
QQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 5
QK      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
PK      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7691
LO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 2
QE      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
HQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7688
OE      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 106
MO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
RG      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7686
IP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 18
GI      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
FL      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7684
QI      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 3
GS      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
IR      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7681
LG      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 4
GM      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
JH      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7680
RM      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 2
GG      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
DN      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7679
HH      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 19
IQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
ED      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7678
EQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 1
IK      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
OJ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7676
DO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 2
IE      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_SELL
RP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7675
EE      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 1
QR      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
LF      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7671
LP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 40
QD      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
KL      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7670
QJ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 21
QN      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
CR      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7669
RD      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 20
QP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
DH      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7668
NN      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 17
QJ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
QN      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7667
RK      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 2
OL      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
DE      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7666
MQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 151
QF      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
OJ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7665
RO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 2
OH      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
FQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7664
EF      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 49
OR      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
GG      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7663
OS      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 3
ED      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
JM      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7662
PI      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 4
CN      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
ES      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7661
LD      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 13
CP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
FI      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7660
LM      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 2
IJ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
NO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7659
II      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 12
IL      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
ME      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7658
IP      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 3
GF      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
NK      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7657
FM      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 15
GH      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
MQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7656
DD      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 6
MS      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
NG      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7655
KR      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 9
KE      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
OM      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7654
IK      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 14
KO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
NS      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7653
DF      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 534
MQ      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Type: BOOK_TYPE_BUY
OI      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Price: 7652
IO      0       11:24:33.812    BookEventProcessor1 (SBRF-12.14,M1)     Volume: 25
```

It is what Depth of the Market looked like at a certain moment in time. The first element of the array of structures is an order to sell at the highest price (7708 Rub). The last element of the array is an order to buy at the lowest price (7652 Rub). So, the Depth of Market data is read to the array from top to bottom.

For convenience, I have gathered data in Table 1.

![Table 1. Depth of Market for SBRF-12.14](https://c.mql5.com/2/11/2__4.png)

Table 1. Depth of Market for SBRF-12.14

The upper block highlighted in red contains sell orders (sell-limits). The lower block highlighted in green includes buy orders (buy-limits).

It is obvious that out of all sell orders, order #6 had the largest volume with the price of 7700 Rub and volume of 1011 lots. Order #39 had the largest volume out of all buy orders with the price of 7653 Rub and volume of 534 lots.

The source data in Depth of Market is the information that a trader analyses to work out a trading strategy. The simplest trading idea is that gathering and clustering orders with significant volumes at contiguous price levels create zones of support and resistance. In the following section, we shall create an indicator that will track changes in Depth of Market.

### 4\. Depth of Market

There are a lot of various indicators working with Depth of Market data. You can find a few interesting ones in the MetaTrader 5 [Market](https://www.mql5.com/en/market). I like [IShift](https://www.mql5.com/en/market/product/758) most of all. [Yury Kulikov](https://www.mql5.com/en/users/yurich "https://www.mql5.com/en/users/yurich"), the developer, succeeded in creating a compact and informative tool.

All programs, working with Depth of Market data, will have a form of either an Expert Advisor or an indicator, as only these MQL5 programs feature the [BookEvent](https://www.mql5.com/en/docs/runtime/event_fire#bookevent) event handler.

Let us create a short program that will be showing live Depth of Market data. At first we need to specify the data to be displayed. It will be a panel, with horizontal bars indicating the volume of the order. The size of the bars, however, will be of relative nature. Maximum volume of all current orders will be considered as 100%. Fig. 2 shows that the order at the price of 7507 Rub has the largest volume of 519 lots.

For correct panel initialization, the exact Depth of Market or the number of levels has to be specified (the "DOM depth" parameter). This number differs from broker to broker.

![Fig.2 Depth of Market Panel](https://c.mql5.com/2/11/4__4.png)

Fig. 2 Depth of Market Panel

The program code for the panel is written with [Object-Oriented Programming](https://www.mql5.com/en/docs/basis/oop) approach. The class responsible for the panel operation is named as CBookBarsPanel.

```
//+------------------------------------------------------------------+
//| Book bars class                                                  |
//+------------------------------------------------------------------+
class CBookBarsPanel
  {
private:
   //--- Data members
   CArrayObj         m_obj_arr;
   uint              m_arr_size;
   //---
   uint              m_width;
   uint              m_height;

   //--- Methods
public:
   void              CBookBarsPanel(const uint _arr_size);
   void             ~CBookBarsPanel(void){};
   //---
   bool              Init(const uint _width,const uint _height);
   void              Deinit(void){this.m_obj_arr.Clear();};
   void              Refresh(const MqlBookInfo &_bookArray[]);
  };
```

This class contains four data members.

- The **m\_obj\_arr** attribute is a container for pointers to the objects of the CObject type. It is used for storing Depth of Market data. A separate class will be created for the latter.
- The **m\_arr\_size** attribute is responsible for the number of Depth of Market levels.
- The **m\_width** and **m\_height** attributes store the panel dimensions (width and height in pixels).

As far as the methods are concerned, their set besides standard constructor and destructor includes:

- method of initialization
- method of deinitialization
- method of updating

We shall create the separate CBookRecord class for each row of the panel (Depth of Market level) specifying the price, horizontal bar and volume.

It will comprise three pointers. One of them will be pointing to the object of the CChartObjectRectLabel type (rectangular label for working with a horizontal bar) and two pointers of the CChartObjectLabel type (text labels for the price and volume).

```
//+------------------------------------------------------------------+
//| Book record class                                                |
//+------------------------------------------------------------------+
class CBookRecord : public CObject
  {
private:
   //--- Data members
   CChartObjectRectLabel *m_rect;
   CChartObjectLabel *m_price;
   CChartObjectLabel *m_vol;
   //---
   color             m_color;

   //--- Methods
public:
   void              CBookRecord(void);
   void             ~CBookRecord(void);
   bool              Create(const string _name,const color _color,const int _X,
                            const int _Y,const int _X_size,const int _Y_size);
   //--- data
   bool              DataSet(const long _vol,const double _pr,const uint _len);
   bool              DataGet(long &_vol,double &_pr) const;

private:
   string            StringVolumeFormat(const long _vol);
  };
```

Among the methods are:

- method of a record creation;
- method of data setting;
- method of data receiving;
- method of big numbers formatting.

Then the handler of [BookEvent](https://www.mql5.com/en/docs/runtime/event_fire#bookevent) for the EA and, essentially, the indicator, will look as follows:

```
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
//--- select the symbol
   if(symbol==_Symbol)
     {
      //--- array of the DOM structures
      MqlBookInfo last_bookArray[];

      //--- get the book
      if(MarketBookGet(_Symbol,last_bookArray))
         //--- refresh panel
         myPanel.Refresh(last_bookArray);
     }
  }
```

So, every time [BookEvent](https://www.mql5.com/en/docs/runtime/event_fire#bookevent) is generated, the panel will update its data. We are going to name the updated version of the program as BookEventProcessor2.mq5.

YouTube

In the video above you can see how the EA is working.

### Conclusion

This article is dedicated to another event of the Terminal - the Depth of Market event. This event is often at the core of high frequency trading algorithms (HFT). This type of trading is gaining popularity among traders.

I hope that traders starting to program on MQL5 will find included examples of handling the Depth of Market event useful. The source files attached to this article are convenient to be put into the project folder. In my case it is \\MQL5\\Projects\\BookEvent.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1179](https://www.mql5.com/ru/articles/1179)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1179.zip "Download all attachments in the single ZIP archive")

[BookEvent.zip](https://www.mql5.com/en/articles/download/1179/bookevent.zip "Download BookEvent.zip")(3.84 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Cookbook — Macroeconomic events database](https://www.mql5.com/en/articles/11977)
- [MQL5 Cookbook — Services](https://www.mql5.com/en/articles/11826)
- [MQL5 Cookbook – Economic Calendar](https://www.mql5.com/en/articles/9874)
- [MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://www.mql5.com/en/articles/7166)
- [MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)
- [MQL5 Cookbook - Pivot trading signals](https://www.mql5.com/en/articles/2853)
- [MQL5 Cookbook - Trading signals of moving channels](https://www.mql5.com/en/articles/1863)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/37110)**
(26)


![FABYCASTILLO](https://c.mql5.com/avatar/avatar_na2.png)

**[FABYCASTILLO](https://www.mql5.com/en/users/fabycastillo)**
\|
10 Jun 2021 at 20:08

Hello Denis!

Quick question. Can we access all types of [pending orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 documentation:") not only buy below market price but also sell below it and vice versa, buy above market price. In other words Im trying to map out all forms of liquidity orders in the book. Similar like Oanda broker used to have in the FX lab tools.

Thanks a lot

![Camilo Ramirez](https://c.mql5.com/avatar/2019/5/5CE77DB0-8B12.png)

**[Camilo Ramirez](https://www.mql5.com/en/users/minealexgames)**
\|
19 May 2022 at 21:57

good day friend

i saw your block

I couldn't copy the file

[![](https://c.mql5.com/3/386/1539128950429__1.png)](https://c.mql5.com/3/386/1539128950429.png "https://c.mql5.com/3/386/1539128950429.png")

I await your prompt response, thank you

```

```

![Florian Silver Grunert](https://c.mql5.com/avatar/avatar_na2.png)

**[Florian Silver Grunert](https://www.mql5.com/en/users/bnd)**
\|
10 Aug 2022 at 14:52

**Camilo Ramirez [#](https://www.mql5.com/en/forum/37110#comment_39692655):**

good day friend

i saw your block

I couldn't copy the file

I await your prompt response, thank you

You have to copy CBookBarsPanel.mqh to the Experts folder too. After that compile BookEventProcessor2.mq5 again.


![B. Goksin](https://c.mql5.com/avatar/2022/4/6263A6BC-42E6.png)

**[B. Goksin](https://www.mql5.com/en/users/b.goksin)**
\|
3 Sep 2022 at 11:28

hello, i am a beginner in [mql5 programming](https://www.mql5.com/en/articles/117 "Article: How to Order a Trading Robot in MQL5 and MQL4 "). and I am looking that if it is possible to see the realized ask and bid number of the market. depth of market may give some information but I think it is not realized data. thank you.


![teufeurlastreet](https://c.mql5.com/avatar/avatar_na2.png)

**[teufeurlastreet](https://www.mql5.com/en/users/teufeurlastreet)**
\|
20 May 2024 at 20:12

[@Denis Kirichenko](https://www.mql5.com/en/users/denkir)

Hello Sir,

Hope you are doing well, thanks a lot for your code about the DOM automated it is really useful for me so thanks a lot. :)

I have downloaded and installed the code fine, but I only see the sell orders. I think there was an update in the API Reference or something and your buy orders is not showing now.

As you can see in the screenshot, the Sell works fine, but no data for the Buy. I am in MT5 free, I don't have subscribed to any market data Level 2.

What do you think I need to check please ?

Thanks for your help Denis,

Kind regards,

Alexandre

![Why Virtual Hosting On The MetaTrader 4 And MetaTrader 5 Is Better Than Usual VPS](https://c.mql5.com/2/11/Virtual_hosting.png)[Why Virtual Hosting On The MetaTrader 4 And MetaTrader 5 Is Better Than Usual VPS](https://www.mql5.com/en/articles/1171)

The Virtual Hosting Cloud network was developed specially for MetaTrader 4 and MetaTrader 5 and has all the advantages of a native solution. Get the benefit of our free 24 hours offer - test out a virtual server right now.

![How to Access the MySQL Database from MQL5 (MQL4)](https://c.mql5.com/2/11/MQLMySQL.png)[How to Access the MySQL Database from MQL5 (MQL4)](https://www.mql5.com/en/articles/932)

The article describes the development of an interface between MQL and the MySQL database. It discusses existing practical solutions and offers a more convenient way to implement a library for working with databases. The article contains a detailed description of the functions, the interface structure, examples and some of specific features of working with MySQL. As for the software solutions, the article attachments include the files of dynamic libraries, documentation and script examples for the MQL4 and MQL5 languages.

![MQL5 Programming Basics: Global Variables of the Terminal](https://c.mql5.com/2/12/MQL5_Basics_Global_variables_terminal_MetaTrader5.png)[MQL5 Programming Basics: Global Variables of the Terminal](https://www.mql5.com/en/articles/1210)

This article highlights object-oriented capabilities of the MQL5 language for creating objects facilitating work with global variables of the terminal. As a practical example I consider a case when global variables are used as control points for implementation of program stages.

![MQL5 Cookbook: Handling Custom Chart Events](https://c.mql5.com/2/11/avatar.png)[MQL5 Cookbook: Handling Custom Chart Events](https://www.mql5.com/en/articles/1163)

This article considers aspects of design and development of custom chart events system in the MQL5 environment. An example of an approach to the events classification can also be found here, as well as a program code for a class of events and a class of custom events handler.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/1179&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069345914088260487)

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