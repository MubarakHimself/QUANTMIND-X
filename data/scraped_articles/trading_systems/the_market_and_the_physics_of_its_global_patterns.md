---
title: The market and the physics of its global patterns
url: https://www.mql5.com/en/articles/8411
categories: Trading Systems
relevance_score: 1
scraped_at: 2026-01-23T21:38:58.507592
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/8411&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071991235985486400)

MetaTrader 5 / Trading systems


### Introduction

In this article, we will try to understand how market physics can be used for automated trading. The language of mathematics implies transition from abstractness and uncertainty to forecasting. This allows operating with clear formulas or criteria, rather than with some approximate and vague values, in an effort to improve the quality of the created systems. I will not invent any theories or patterns, but I will only use known facts, gradually translating these facts into the language of mathematical analysis. The market physics is impossible without mathematics, because the signals that we generate are mathematical substance. Many people try to create various theories and formulas without any statistical analysis or using very limited statistics, which is often not enough for such bold conclusions. Practice alone is the criterion of truth. First, I will try to reflect a little, and then, based on these reflections, I will create an Expert Advisor (EA). This will be followed by the EA testing.

### Price and what it provides

Any market context implies the existence of various products. In the currency market, the product is the currency. Currency means the right to own a certain product or information, which is set as a benchmark all over the world. Consider for example the EURUSD pair and the current value of its chart. The current chart value will mean USD/EUR = CurrentPrice. Or: USD =  EUR\*CurrentPrice, which means the amount of dollars contained in one euro. In other words, the value shows the ratio of each currency weight, while, of course, it is assumed that there is some common equivalent of exchange for each currency, i.e. some common commodity or something else. The price is formed in the order book, and the dynamics of the order book determines the price movement. It should always be remembered that we will never be able to take into account all factors of price formation. For example, FORTS is linked to FOREX and both markets influence each other. I am not an expert in this variety, but I am able to understand that everything is bound and that the more data channels, the better. It seems to me that it is better not to go deep into such detail, but to focus on simple things that move the price.

Any dependency can be represented as a function of many variables, like any quote. Initially, the price is:

- P=P(t)

In other words, the price is a function of time. The form of the function cannot be reliably established for each currency pair or any other instrument, since this will take an infinite time. But this presentation gives us nothing. However, the price is of a dual natures, as it has both a predictable and a random component. The predictable part is not the function itself, but its first time derivative. It makes no sense to represent this function as some terms, since this has no purpose for trading. But if we consider its first derivative, there is the following:

- P'(t)=Pa'(t)+Pu'(t)

Here, the first term reflects the part that can be analyzed in some way using mathematical analysis, the second is an unpredictable part. Based on this formula, we can say that it is not possible to predict the movement size and direction with a 100% accuracy. There is no need to think what the last term will be, since we cannot determine it. But we can determine the first one. It can be assumed that this term can be represented in a different form, taking into account that the value function is discrete and we cannot apply differential operations. But instead, we can take the average derivative over time "st". When applied to the price, this will be the duration of a bar; when applied to ticks, it is the minimum time between two ticks.

- PaM(t)=(Pa(t)-Pa(t-st))/st - the average price movement (time derivative) over a fixed period of time
- Ma(t)=Ma(P(t),P(t-st) + ... + P(t-N\*st), D(t),D(t-st) + ... + D(t-N\*st),U\[1\](),U\[2\](t) + ... + U\[N\](t) )
- P(t\[i\]) - old price values (bar or tick data)
- D(t\[i\]) - old price values on other currency pairs
- U\[i\](t) - other unknown or known values affecting the market
- Ma(t) \- mathematical expectation of the PaM(t) value at a given point in time


In other words, we assume that the predictable part of the price may depend on previous bars or ticks, as well as on the price data of other currency pairs, data from other exchanges and world events. However, it should be understood that even this part of the price cannot be predicted with a 100% accuracy, but we can only calculate some of its characteristics. Such a characteristic can be only a probability or parameters of a random variable, such as the mathematical expectation, variance, standard deviation and other quantities of the theory of probability. Operating with mathematical expectations can be enough to make trading profitable. After taking time and thinking carefully, we can come to the conclusion that the market can be analyzed not only using this logic. The thing is that the predictable part of the price develops on the basis of the activity of market players. We can discard various market parameters, except for the factors created by the market players themselves. Of course, all this leads to a decrease in the reliability of our analysis method, but this greatly simplifies the model. Here, the smaller the "st" value, the more accurately our formulas describe the market.

- VMa(t)=VMa(P(t),P(t-st) + ... + P(t-N\*st))
- VMa(t)=VBuy(t)-VSell(t)
- VMa(t) - total volumes
- VBuy(t) - volumes of open Buy orders
- VSell(t) - volumes of open Sell orders

The above function describes the difference in the volumes of all currently open Buy and Sell positions. Part of these positions compensate each other, while the rest positions are independent. Since the positions are open, they symbolize a promise to close after a while. We all know that buying moves the price up, and selling moves the price down. The only way to know where the price will go is to measure the volumes of open positions and to estimate the direction of these positions, taking into account only open market orders.

The wave nature of the market is actually connected with this simple fact. This is only one special case of a more general process of fluctuations in position volumes, or the actions of bulls and bears.

When dealing with bars, it is also possible to take into account the fact that there are 4 prices inside a bar, which can give us better formulas. More data means more accurate analysis, which is why it is important to consider all price data. However, I do not like counting every tick, as this would slow down the algorithms by ten times. Furthermore, tick data can be different with different brokers. On the contrary, bar open and close prices are almost identical for most brokers. Let us amend the volume function to take into account all price data:

- VMa(t)=VMa(O(t),O(t-st) +...+ O(t-N\*st) + C(t),C(t-st) + C(t-N\*st),H(t),H(t-st)...H(t-N\*st),L(t),L(t-st)...L(t-N\*st))

We could add more variables to this function, such as time, days of the week, months and weeks, but this would produce a lot of functions linked to specific market areas, while our purpose is to determine the general market physics. We will know that it cannot be broken and thus it can be used as long as the market exists. Another advantages of the formula is its multi-currency nature.

In practical terms, the use of this representation type does not make sense, since we need to know exactly how and based on what data to build this function. We cannot just write the form of this function and determine dependencies. But these expressions can help us gain some initial understanding of how to analyze and how move on to the following assumptions. Any set of logical conditions can ultimately be represented as such a function. Conversely, the function itself can be turned into a set of conditions. It does not matter which form we use. It is only important to understand it. Any algorithm can be reduced to some single formula. Sometimes it is easier to describe signals as conditions or a system of conditions than to build a super-complex function. Another big question is how to build such a function.

In a real trading system, we cannot analyze the entire history at once, but we can only analyze a fixed period of time. There are 4 possible approaches to such analysis. I will create names for them and explain:

- Formulas (indicators or their functions)
- Simulation
- General mathematics
- Types of machine learning

The first option assumes that we use one certain value, or a set of values. An example is an indicator or our own formula. The advantage of this approach is the availability of a large toolkit in the MetaTrader 4/5 terminals. Moreover, there are a lot of indicators based in popular market theories, available in the Market and on the web. The disadvantage of this approach is that in most cases we will not understand based on what the indicator works. Even if we do understand, such understanding can be of no value.

In the second option, we do not use the data that we do not understand or that can be of no use. Instead, we can try to simulate orders in the market and thus we will know that our system will be able to some extent to describe how many positions are open in one direction, and how many in the other. This information can produce the necessary forecasts, allowing a fine description of the market in a global perspective. This is the only alternative to machine learning.

Mathematics means an understanding of some fundamental laws or knowledge of certain mathematical principles which allow exploiting any quote, regardless of the current market situation. The fact is that any functions, including discrete ones, have certain characteristics that can be exploited. Of course, here we assume that the dependence is not chaotic (in our case, forex is not chaotic, so any quote can be exploited). In the next article, we will analyze one such principle, which almost everyone knows. But to know and to be able to use are two different things. The advantage of this approach is that if we manage to build a successful system, we will not need to worry about how it will behave in the future.

The fourth approach is the most progressive one, as machine learning can make the most of any data. The more computing power you have, the higher the quality of the analysis. The downside of this approach is that it does not assist in understanding the market physics. The advantage is the simplicity of the approach, the maximum quality of the results with a minimum of time spent. But this approach is not applicable within this article.

### About Patterns

Everyday trading implies a plethora of terms, each having a different importance level. Some terms are used very often, though not all traders understand their real purpose. One of these terms is the Pattern. I will try to explain it is in the language of mathematics. A pattern is always linked to a specific period of time and a specific currency pair and chart period. Some patterns are strong. Such patterns can be of multicurrency or global nature. An ideal pattern is the Grail. There are some statements that can be applied to any pattern:

- The existence of a formula or set of conditions that symbolizes the pattern
- Minimum test values in the strategy tester or performance values on a demo or real account
- Classification of a pattern in terms of performance for all currency pairs and chart periods
- The period of history in which the pattern was found
- The time interval in the future, during which the pattern remains operational
- The second period of time in the future, following the first one, during which the original pattern retains some parameters, or inverts them

If you carefully read each property, you can understand that a pattern is a formula or a set of conditions, which most accurately describe the price movement at selected intervals. A pattern can turn out to be random, especially if found on a too small period or if the relevant system produces overoptimistic values. It is very important to understand that when testing a system on short periods, the chances of finding global patterns tend to zero. This is connected with the sample size. The smaller the sample, the higher the randomness of results.

We have determined what a pattern is. But how to effectively use patterns? It all depends on how this pattern was found and on its quality. If we do not take into account the analysis methods utilizing computing power, then we come to analytics. In my opinion, analytics cannot compete with any types of machine analysis - even a good team of analysts cannot process the data that can be processed by one machine. Anyway, the process of finding global patterns requires computing power. Well, except for the case when you saw obvious things with your own eyes and understood their physics.

### Writing the Simplest Position Simulator

In order to try to find global patterns, it would be interesting to develop an Expert Advisor that can describe the mood of market players. For this, I decided to try to create a simulator of market positions. Positions will be simulated within the bars close to the market edge. It will be necessary to assume that the market players are different, and the weight of their orders is also different. At the same time, this should be presented in a simple form. If a simple prototype shows profit, then its principles can be used further.

The logic will be conditionally divided into 3 separate simulations and any possible mixed combination of them:

- Simulation of Stop orders
- Simulation of Limit orders
- Simulation of market orders
- Any possible combinations

The following logic will be used for placing orders:

![Orders Grid Logic](https://c.mql5.com/2/41/Positions.png)

These grids are places at each new bar, in an effort to simulate the mood of some part of the market participants. The status of old order grids is updated based on the new bar that appears on the chart. This approach is not very accurate, but the tick-by-tick simulation would lead to endless calculations. Moreover, I don’t quite trust ticks.

There are two types of distribution of relative volumes, with attenuation and even, but only for Stop and Limit orders. Market orders have even distribution. It will be also possible to expand the types of distribution, if this will be viewed in perspective. Here is the illustration:

![Relative Volumes Filling Types](https://c.mql5.com/2/41/Volumes.png)

Here, the length of the line symbolizing the order is proportional to the volume of the same order. I think such illustrations will be simple and understandable for everyone.

In this case, everything can be done using an object-oriented approach. Let us start by describing numbered lists:

```
enum CLOSE_MODE// how to close orders
   {
   CLOSE_FAST,// fast
   CLOSE_QUALITY// wait for an opposite signal
   };

enum WORK_MODE// operation mode
   {
   MODE_SIMPLE,// slow mode
   MODE_FAST// fast mode
   };

enum ENUM_GRID_WEIGHT//weight fill type for limit and stop orders
   {
   WEIGHT_DECREASE,// with attenuation if moving from the price
   WEIGHT_SAME// the same for the entire grid
   };

enum ENUM_STATUS_ORDER// statuses of orders or positions
   {
   STATUS_VIRTUAL,// stop or limit order
   STATUS_MARKET,// market order
   STATUS_ABORTED// canceled stop or limit order
   };
```

The simulator will work in two modes, slow and fast. The slow mode is mainly needed for starting analysis. In the starting analysis, the calculation is performed in the first "n" nearest to market candlesticks. In the fast mode, calculation is performed only on the newly appeared candlestick. But the simple approach turned to be not enough. An additional functionality was required to increase the algorithm speed. Quite a big computation is performed at the Expert Advisor initialization. But then we only need to update the simulation for one new candlestick at each candlestick. There will be two types of volume distribution, for limit and stop orders, depending on the distance from the current market price, which is Open\[i\] for each bar. This is because a grid of stop and limit orders is opened at each bar, with different distributions and weights. After some time, stop and limit orders turn into market orders. If the price does not reach the required price during the specified time, the stop and limit orders are canceled.

Let us start building this simulation from simple to complex, gradually putting everything together. First, define what an order is:

```
struct Order// structure symbolizing a player's order
   {
   public:
   double WantedPrice;// desired open price
   int BarsExpirationOpen;// If the order remains for certain number of bars, the player can't wait any more and cancels the order
   int BarsExpirationClose;//If this is a market order and the player does not want to wait, he closes the position
   double UpPriceToClose;//The total upward price movement at which the player closes the order (points)
   double LowPriceToClose;//The total downward price movement at which the player closes the order
   double VolumeAlpha;// current volume equivalent [0...1]
   double VolumeStart;// starting volume equivalent [0...1]
   int IndexMarket;// the index of the bar on which the virtual market turned into market
   ENUM_STATUS_ORDER Status;// order status
   Order(ENUM_STATUS_ORDER S)// constructor that creates a certain order
      {
      Status=S;
      }
   };
```

There are not many parameters, and each parameter is important for the general algorithm. Many of the fields are applicable to any order, while some of them only apply to Limit or Stop orders. For example, the desired price is the open price for a market order, while it is exactly the desired price for Limit and Stop orders.

The upper and lower closing prices act as stop levels. At the same time, we will assume that the grid order is not quite an order, and this order contains a whole bunch of orders, while they just merge into one order, which is opened with a certain volume at a certain price. The variables of the starting volume and the current volume tell us how important the orders are at a specific level of a specific bar.

Starting volume is the volume at the order placing time. Current volume is the volume as the events develop further. What is important is the ratio of the volumes of Buy and Sell orders rather than profits from certain orders. Trading signals will be generated based on these considerations. Of course, we could come up with other signals, but this requires some other considerations. Also note that orders will not be closed when they reach certain levels, but closing will happen gradually at each bar, in an effort to simulate the real development of events as much as possible.

Next, we need to define the storage for each bar. The bar will store the orders that open at this bar:

```
class OrderBox// Order box of a specific bar
   {
   public:
   Order BuyStopOrders[];
   Order BuyLimitOrders[];
   Order BuyMarketOrders[];
   Order SellStopOrders[];
   Order SellLimitOrders[];
   Order SellMarketOrders[];

   OrderBox(int OrdersToOneBar)
      {
      ArrayResize(BuyStopOrders,OrdersToOneBar);
      ArrayResize(BuyLimitOrders,OrdersToOneBar);
      ArrayResize(BuyMarketOrders,OrdersToOneBar);
      ArrayResize(SellStopOrders,OrdersToOneBar);
      ArrayResize(SellLimitOrders,OrdersToOneBar);
      ArrayResize(SellMarketOrders,OrdersToOneBar);
      for ( int i=0; i<ArraySize(BuyStopOrders); i++ )// Set types for all orders
         {
         BuyStopOrders[i]=Order(STATUS_VIRTUAL);
         BuyLimitOrders[i]=Order(STATUS_VIRTUAL);
         BuyMarketOrders[i]=Order(STATUS_MARKET);
         SellStopOrders[i]=Order(STATUS_VIRTUAL);
         SellLimitOrders[i]=Order(STATUS_VIRTUAL);
         SellMarketOrders[i]=Order(STATUS_MARKET);
         }
      }
   };
```

Everything is quite simple here. Six types of orders, which are described by arrays. This is done to avoid confusion. The class will not be used in its pure form, while it is only a brick in the construction.

Next, define the common storage of all bars as one object, from which later inheritance will be performed. The techniques here are quite simple.

```
class BarBox// Storage for all orders
   {
   protected:
   OrderBox BarOrders[];

   BarBox(int OrdersToOneBar,int BarsTotal)
      {
      ArrayResize(BarOrders,BarsTotal);
      for ( int i=0; i<ArraySize(BarOrders); i++ )// Set types for all orders
         {
         BarOrders[i]=OrderBox(OrdersToOneBar);
         }
      }
   };
```

This is just a storage with bar data (orders) and nothing more. So far, everything is quite simple. Further, things become more complicated.

After we have determined a convenient data storage for orders, we need to determine how and according to which rules orders are created, what is the importance of specific order types, etc. I have created the following class for this purpose:

```
class PositionGenerator:public BarBox// Inherit class from the box to avoid the need to include it as an internal member and to avoid multiple references
   {
   protected:
   double VolumeAlphaStop;// importance of volumes of STOP orders
   double VolumeAlphaLimit;// importance of volumes of LIMIT orders
   double VolumeAlphaMarket;// importance of volumes of MARKET orders
   double HalfCorridorLimitStop;// step of the corridor of Limit and Stop orders in points
   int ExpirationOpenLimit;// after how many bars the volumes of the grid of limit orders for opening will completely attenuate
   int ExpirationOpenStop;// after how many bars the volumes of the grid of stop orders for opening will completely attenuate
   int ExpirationClose;// after how many bars the volumes of orders for closing will completely attenuate
   int ProfitPointsCorridorPart;// half corridor size for the profit of all orders
   int LossPointsCorridorPart;// half corridor size for the loss of all orders
   int OrdersToOneBar;// orders of one type per 1 bar
   ENUM_GRID_WEIGHT WeightStopLimitFillingType;

   PositionGenerator( ENUM_GRID_WEIGHT WeightStopLimitFillingType0
                     ,int HalfCorridorLimitStop0,int OrdersToOneBar0,int BarsTotal0
                     ,int ExpirationOpenLimit0,int ExpirationOpenStop0
                     ,int ExpirationClose0
                     ,int ProfitPointsCorridorPart0,int LossPointsCorridorPart0
                     ,double VolumeAlphaStop0,double VolumeAlphaLimit0,double VolumeAlphaMarket0)
                     : BarBox(OrdersToOneBar0,BarsTotal0)
      {
      VolumeAlphaStop=VolumeAlphaStop0;
      VolumeAlphaLimit=VolumeAlphaLimit0;
      VolumeAlphaMarket=VolumeAlphaMarket0;
      OrdersToOneBar=OrdersToOneBar0;
      HalfCorridorLimitStop=double(HalfCorridorLimitStop0)/double(OrdersToOneBar);
      ExpirationOpenLimit=ExpirationOpenLimit0;
      ExpirationOpenStop=ExpirationOpenStop0;
      ExpirationClose=ExpirationClose0;
      ProfitPointsCorridorPart=ProfitPointsCorridorPart0;
      LossPointsCorridorPart=LossPointsCorridorPart0;
      OrdersToOneBar=OrdersToOneBar0;
      WeightStopLimitFillingType=WeightStopLimitFillingType0;
      }
   private:

   double CalcVolumeDecrease(double TypeWeight,int i,int size)// attenuation volume
      {
      if ( size > 1 )
         {
         double K=1.0/(1.0-size);
         double C=1.0;
         return TypeWeight*K*i+C;
         }
      else return 0.0;
      }

   double CalcVolumeSimple(double TypeWeight)// equal volume
      {
      return TypeWeight;
      }

   void RebuildStops()// rebuild stop orders
      {
      int size=ArraySize(BarOrders[0].BuyStopOrders);
      for ( int j=ArraySize(BarOrders)-1; j>=0; j-- )
         {
         for ( int i=0; i<size; i++ )// reset all
            {
            BarOrders[j].BuyStopOrders[i].Status=STATUS_VIRTUAL;// reset status to initial
            BarOrders[j].BuyStopOrders[i].WantedPrice=Open[j+1]+HalfCorridorLimitStop*(i+1)*Point;// prices of the order grid
            if ( WeightStopLimitFillingType == WEIGHT_DECREASE ) BarOrders[j].BuyStopOrders[i].VolumeAlpha=CalcVolumeDecrease(VolumeAlphaStop,i,size);// weight of each element of the grid
            if ( WeightStopLimitFillingType == WEIGHT_SAME ) BarOrders[j].BuyStopOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaStop);// current weight of each element of the grid
            BarOrders[j].BuyStopOrders[i].VolumeStart=BarOrders[j].BuyStopOrders[i].VolumeAlpha;// starting weight of each element of the grid
            BarOrders[j].BuyStopOrders[i].UpPriceToClose=BarOrders[j].BuyStopOrders[i].WantedPrice+ProfitPointsCorridorPart*Point;// upper border to close
            BarOrders[j].BuyStopOrders[i].LowPriceToClose=BarOrders[j].BuyStopOrders[i].WantedPrice-LossPointsCorridorPart*Point;// lower border to close
            BarOrders[j].BuyStopOrders[i].BarsExpirationOpen=ExpirationOpenStop;
            BarOrders[j].BuyStopOrders[i].BarsExpirationClose=ExpirationClose;

            BarOrders[j].SellStopOrders[i].Status=STATUS_VIRTUAL;
            BarOrders[j].SellStopOrders[i].WantedPrice=Open[j+1]-HalfCorridorLimitStop*(i+1)*Point;// prices of the order grid
            if ( WeightStopLimitFillingType == WEIGHT_DECREASE ) BarOrders[j].SellStopOrders[i].VolumeAlpha=CalcVolumeDecrease(VolumeAlphaStop,i,size);// weight of each element of the grid
            if ( WeightStopLimitFillingType == WEIGHT_SAME ) BarOrders[j].SellStopOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaStop);// current weight of each element of the grid
            BarOrders[j].SellStopOrders[i].VolumeStart=BarOrders[j].SellStopOrders[i].VolumeAlpha;// starting weight of each element of the grid
            BarOrders[j].SellStopOrders[i].UpPriceToClose=BarOrders[j].SellStopOrders[i].WantedPrice+LossPointsCorridorPart*Point;// upper border to close
            BarOrders[j].SellStopOrders[i].LowPriceToClose=BarOrders[j].SellStopOrders[i].WantedPrice-ProfitPointsCorridorPart*Point;// lower border to close
            BarOrders[j].SellStopOrders[i].BarsExpirationOpen=ExpirationOpenStop;
            BarOrders[j].SellStopOrders[i].BarsExpirationClose=ExpirationClose;
            }
         }
      }

   void RebuildLimits()// rebuild limit orders
      {
      int size=ArraySize(BarOrders[0].BuyLimitOrders);
      for ( int j=ArraySize(BarOrders)-1; j>=0; j-- )
         {
         for ( int i=0; i<size; i++ )// reset all
            {
            BarOrders[j].BuyLimitOrders[i].Status=STATUS_VIRTUAL;// reset status to initial
            BarOrders[j].BuyLimitOrders[i].WantedPrice=Open[j+1]-HalfCorridorLimitStop*(i+1)*Point;// prices of the order grid
            if ( WeightStopLimitFillingType == WEIGHT_DECREASE ) BarOrders[j].BuyLimitOrders[i].VolumeAlpha=CalcVolumeDecrease(VolumeAlphaLimit,i,size);// weight of each element of the grid
            if ( WeightStopLimitFillingType == WEIGHT_SAME ) BarOrders[j].BuyLimitOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaLimit);// current weight of each element of the grid
            BarOrders[j].BuyLimitOrders[i].VolumeStart=BarOrders[j].BuyLimitOrders[i].VolumeAlpha;// starting weight of each element of the grid
            BarOrders[j].BuyLimitOrders[i].UpPriceToClose=BarOrders[j].BuyLimitOrders[i].WantedPrice+ProfitPointsCorridorPart*Point;// upper border to close
            BarOrders[j].BuyLimitOrders[i].LowPriceToClose=BarOrders[j].BuyLimitOrders[i].WantedPrice-LossPointsCorridorPart*Point;// lower border to close
            BarOrders[j].BuyLimitOrders[i].BarsExpirationOpen=ExpirationOpenLimit;
            BarOrders[j].BuyLimitOrders[i].BarsExpirationClose=ExpirationClose;

            BarOrders[j].SellLimitOrders[i].Status=STATUS_VIRTUAL;
            BarOrders[j].SellLimitOrders[i].WantedPrice=Open[j+1]+HalfCorridorLimitStop*(i+1)*Point;// prices of the order grid
            if ( WeightStopLimitFillingType == WEIGHT_DECREASE ) BarOrders[j].SellLimitOrders[i].VolumeAlpha=CalcVolumeDecrease(VolumeAlphaLimit,i,size);// weight of each element of the grid
            if ( WeightStopLimitFillingType == WEIGHT_SAME ) BarOrders[j].SellLimitOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaLimit);// current weight of each element of the grid
            BarOrders[j].SellLimitOrders[i].VolumeStart=BarOrders[j].SellLimitOrders[i].VolumeAlpha;// starting weight of each element of the grid
            BarOrders[j].SellLimitOrders[i].UpPriceToClose=BarOrders[j].SellLimitOrders[i].WantedPrice+LossPointsCorridorPart*Point;// upper border to close
            BarOrders[j].SellLimitOrders[i].LowPriceToClose=BarOrders[j].SellLimitOrders[i].WantedPrice-ProfitPointsCorridorPart*Point;// lower border to close
            BarOrders[j].SellLimitOrders[i].BarsExpirationOpen=ExpirationOpenLimit;
            BarOrders[j].SellLimitOrders[i].BarsExpirationClose=ExpirationClose;
            }
         }
      }

   void RebuildMarkets()// rebuild market orders
      {
      int size=ArraySize(BarOrders[0].BuyMarketOrders);
      double MarketStep;
      for ( int j=ArraySize(BarOrders)-1; j>0; j-- )
         {
         MarketStep=(High[j+1]-Low[j+1])/double(OrdersToOneBar);

         for ( int i=0; i<size; i++ )// reset all
            {
            BarOrders[j].BuyMarketOrders[i].Status=STATUS_MARKET;// reset status to initial
            BarOrders[j].BuyMarketOrders[i].WantedPrice=Low[j+1]+MarketStep*i;// prices of the order grid
            BarOrders[j].BuyMarketOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaMarket);// current weight of each element of the grid
            BarOrders[j].BuyMarketOrders[i].VolumeStart=BarOrders[j].BuyMarketOrders[i].VolumeAlpha;// starting weight of each element of the grid
            BarOrders[j].BuyMarketOrders[i].UpPriceToClose=BarOrders[j].BuyMarketOrders[i].WantedPrice+ProfitPointsCorridorPart*Point;// upper border to close
            BarOrders[j].BuyMarketOrders[i].LowPriceToClose=BarOrders[j].BuyMarketOrders[i].WantedPrice-LossPointsCorridorPart*Point;// lower border to close
            BarOrders[j].BuyMarketOrders[i].BarsExpirationClose=ExpirationClose;

            BarOrders[j].SellMarketOrders[i].Status=STATUS_MARKET;
            BarOrders[j].SellMarketOrders[i].WantedPrice=High[j+1]-MarketStep*i;// prices of the order grid
            BarOrders[j].SellMarketOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaMarket);// current weight of each element of the grid
            BarOrders[j].SellMarketOrders[i].VolumeStart=BarOrders[j].SellMarketOrders[i].VolumeAlpha;// starting weight of each element of the grid
            BarOrders[j].SellMarketOrders[i].UpPriceToClose=BarOrders[j].SellMarketOrders[i].WantedPrice+LossPointsCorridorPart*Point;// upper border to close
            BarOrders[j].SellMarketOrders[i].LowPriceToClose=BarOrders[j].SellMarketOrders[i].WantedPrice-ProfitPointsCorridorPart*Point;// lower border to close
            BarOrders[j].SellMarketOrders[i].BarsExpirationClose=ExpirationClose;
            }
         }
      }

   ///// Fast methods
   void RebuildStopsFast()// rebuild stop orders
      {
      int size=ArraySize(BarOrders[0].BuyStopOrders);
      for ( int j=ArraySize(BarOrders)-1; j>0; j-- )
         {
         for ( int i=0; i<size; i++ )// shift orders
            {
            BarOrders[j].BuyStopOrders[i]=BarOrders[j-1].BuyStopOrders[i];
            BarOrders[j].SellStopOrders[i]=BarOrders[j-1].SellStopOrders[i];
            BarOrders[j].SellStopOrders[i].IndexMarket++;
            }
         }

      for ( int i=0; i<size; i++ )// create a new grid at a new bar
         {
         BarOrders[0].BuyStopOrders[i].Status=STATUS_VIRTUAL;// reset status to initial
         BarOrders[0].BuyStopOrders[i].WantedPrice=Close[1]+HalfCorridorLimitStop*(i+1)*Point;// prices of the order grid
         if ( WeightStopLimitFillingType == WEIGHT_DECREASE ) BarOrders[0].BuyStopOrders[i].VolumeAlpha=CalcVolumeDecrease(VolumeAlphaStop,i,size);// weight of each element of the grid
         if ( WeightStopLimitFillingType == WEIGHT_SAME ) BarOrders[0].BuyStopOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaStop);// current weight of each element of the grid
         BarOrders[0].BuyStopOrders[i].VolumeStart=BarOrders[0].BuyStopOrders[i].VolumeAlpha;// starting weight of each element of the grid
         BarOrders[0].BuyStopOrders[i].UpPriceToClose=BarOrders[0].BuyStopOrders[i].WantedPrice+ProfitPointsCorridorPart*Point;//upper border to close
         BarOrders[0].BuyStopOrders[i].LowPriceToClose=BarOrders[0].BuyStopOrders[i].WantedPrice-LossPointsCorridorPart*Point;// lower border to close
         BarOrders[0].BuyStopOrders[i].BarsExpirationOpen=ExpirationOpenStop;
         BarOrders[0].BuyStopOrders[i].BarsExpirationClose=ExpirationClose;

         BarOrders[0].SellStopOrders[i].Status=STATUS_VIRTUAL;
         BarOrders[0].SellStopOrders[i].WantedPrice=Close[1]-HalfCorridorLimitStop*(i+1)*Point;// prices of the order grid
         if ( WeightStopLimitFillingType == WEIGHT_DECREASE ) BarOrders[0].SellStopOrders[i].VolumeAlpha=CalcVolumeDecrease(VolumeAlphaStop,i,size);// weight of each element of the grid
         if ( WeightStopLimitFillingType == WEIGHT_SAME ) BarOrders[0].SellStopOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaStop);// current weight of each element of the grid
         BarOrders[0].SellStopOrders[i].VolumeStart=BarOrders[0].SellStopOrders[i].VolumeAlpha;// starting weight of each element of the grid
         BarOrders[0].SellStopOrders[i].UpPriceToClose=BarOrders[0].SellStopOrders[i].WantedPrice+LossPointsCorridorPart*Point;// upper border to close
         BarOrders[0].SellStopOrders[i].LowPriceToClose=BarOrders[0].SellStopOrders[i].WantedPrice-ProfitPointsCorridorPart*Point;// lower border to close
         BarOrders[0].SellStopOrders[i].BarsExpirationOpen=ExpirationOpenStop;
         BarOrders[0].SellStopOrders[i].BarsExpirationClose=ExpirationClose;
         }
      }

   void RebuildLimitsFast()// rebuild limit orders
      {
      int size=ArraySize(BarOrders[0].BuyLimitOrders);
      for ( int j=ArraySize(BarOrders)-1; j>0; j-- )
         {
         for ( int i=0; i<size; i++ )// shift orders
            {
            BarOrders[j].BuyLimitOrders[i]=BarOrders[j-1].BuyLimitOrders[i];
            BarOrders[j].SellLimitOrders[i]=BarOrders[j-1].SellLimitOrders[i];
            BarOrders[j].SellLimitOrders[i].IndexMarket++;
            }
         }

      for ( int i=0; i<size; i++ )// create a new grid at a new bar
         {
         BarOrders[0].BuyLimitOrders[i].Status=STATUS_VIRTUAL;// reset status to initial
         BarOrders[0].BuyLimitOrders[i].WantedPrice=Open[1]-HalfCorridorLimitStop*(i+1)*Point;// prices of the order grid
         if ( WeightStopLimitFillingType == WEIGHT_DECREASE ) BarOrders[0].BuyLimitOrders[i].VolumeAlpha=CalcVolumeDecrease(VolumeAlphaLimit,i,size);// weight of each element of the grid
         if ( WeightStopLimitFillingType == WEIGHT_SAME ) BarOrders[0].BuyLimitOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaLimit);// current weight of each element of the grid
         BarOrders[0].BuyLimitOrders[i].VolumeStart=BarOrders[0].BuyLimitOrders[i].VolumeAlpha;// starting weight of each element of the grid
         BarOrders[0].BuyLimitOrders[i].UpPriceToClose=BarOrders[0].BuyLimitOrders[i].WantedPrice+ProfitPointsCorridorPart*Point;// upper border to close
         BarOrders[0].BuyLimitOrders[i].LowPriceToClose=BarOrders[0].BuyLimitOrders[i].WantedPrice-LossPointsCorridorPart*Point;// lower border to close
         BarOrders[0].BuyLimitOrders[i].BarsExpirationOpen=ExpirationOpenLimit;
         BarOrders[0].BuyLimitOrders[i].BarsExpirationClose=ExpirationClose;

         BarOrders[0].SellLimitOrders[i].Status=STATUS_VIRTUAL;
         BarOrders[0].SellLimitOrders[i].WantedPrice=Open[1]+HalfCorridorLimitStop*(i+1)*Point;// prices of the order grid
         if ( WeightStopLimitFillingType == WEIGHT_DECREASE ) BarOrders[0].SellLimitOrders[i].VolumeAlpha=CalcVolumeDecrease(VolumeAlphaLimit,i,size);// weight of each element of the grid
         if ( WeightStopLimitFillingType == WEIGHT_SAME ) BarOrders[0].SellLimitOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaLimit);// current weight of each element of the grid
         BarOrders[0].SellLimitOrders[i].VolumeStart=BarOrders[0].SellLimitOrders[i].VolumeAlpha;// starting weight of each element of the grid
         BarOrders[0].SellLimitOrders[i].UpPriceToClose=BarOrders[0].SellLimitOrders[i].WantedPrice+LossPointsCorridorPart*Point;// upper border to close
         BarOrders[0].SellLimitOrders[i].LowPriceToClose=BarOrders[0].SellLimitOrders[i].WantedPrice-ProfitPointsCorridorPart*Point;// lower order to close
         BarOrders[0].SellLimitOrders[i].BarsExpirationOpen=ExpirationOpenLimit;
         BarOrders[0].SellLimitOrders[i].BarsExpirationClose=ExpirationClose;
         }
      }

   void RebuildMarketsFast()// rebuild market orders
      {
      int size=ArraySize(BarOrders[0].BuyMarketOrders);
      double MarketStep;
      for ( int j=ArraySize(BarOrders)-1; j>0; j-- )
         {
         for ( int i=0; i<size; i++ )// shift orders
            {
            BarOrders[j].BuyMarketOrders[i]=BarOrders[j-1].BuyMarketOrders[i];
            BarOrders[j].SellMarketOrders[i]=BarOrders[j-1].SellMarketOrders[i];
            }
         }
      MarketStep=(High[1]-Low[1])/double(OrdersToOneBar);
      for ( int i=0; i<size; i++ )// create a new grid at a new bar
         {
         BarOrders[0].BuyMarketOrders[i].Status=STATUS_MARKET;// reset status to initial
         BarOrders[0].BuyMarketOrders[i].WantedPrice=Low[1]+MarketStep*i;// prices of the order grid
         BarOrders[0].BuyMarketOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaMarket);// current weight of each element of the grid
         BarOrders[0].BuyMarketOrders[i].VolumeStart=BarOrders[0].BuyMarketOrders[i].VolumeAlpha;// starting weight of each element of the grid
         BarOrders[0].BuyMarketOrders[i].UpPriceToClose=BarOrders[0].BuyMarketOrders[i].WantedPrice+ProfitPointsCorridorPart*Point;// upper border to close
         BarOrders[0].BuyMarketOrders[i].LowPriceToClose=BarOrders[0].BuyMarketOrders[i].WantedPrice-LossPointsCorridorPart*Point;// lower border to close
         BarOrders[0].BuyMarketOrders[i].BarsExpirationClose=ExpirationClose;

         BarOrders[0].SellMarketOrders[i].Status=STATUS_MARKET;
         BarOrders[0].SellMarketOrders[i].WantedPrice=High[1]-MarketStep*i;// prices of the order grid
         BarOrders[0].SellMarketOrders[i].VolumeAlpha=CalcVolumeSimple(VolumeAlphaMarket);// current weight of each element of the grid
         BarOrders[0].SellMarketOrders[i].VolumeStart=BarOrders[0].SellMarketOrders[i].VolumeAlpha;// starting weight of each element of the grid
         BarOrders[0].SellMarketOrders[i].UpPriceToClose=BarOrders[0].SellMarketOrders[i].WantedPrice+LossPointsCorridorPart*Point;// upper border to close
         BarOrders[0].SellMarketOrders[i].LowPriceToClose=BarOrders[0].SellMarketOrders[i].WantedPrice-ProfitPointsCorridorPart*Point;// lower border to close
         BarOrders[0].SellMarketOrders[i].BarsExpirationClose=ExpirationClose;
         }
      }

   protected:
   void CreateNewOrders()// create new orders at each candlestick
      {
      if ( VolumeAlphaStop != 0.0 ) RebuildStops();
      if ( VolumeAlphaLimit != 0.0 ) RebuildLimits();
      if ( VolumeAlphaMarket != 0.0 ) RebuildMarkets();
      }

   void CreateNewOrdersFast()//
      {
      if ( VolumeAlphaStop != 0.0 ) RebuildStopsFast();
      if ( VolumeAlphaLimit != 0.0 ) RebuildLimitsFast();
      if ( VolumeAlphaMarket != 0.0 ) RebuildMarketsFast();
      }

   public:
   virtual void Update()// state updating function (will be expanded in child classes)
      {
      CreateNewOrders();
      }

   virtual void UpdateFast()// fast state update
      {
      CreateNewOrdersFast();
      }
   };
```

In fact, this class only creates an implementation of the Update() and UpdateFast() methods, which are similar, with the only difference that the latter is much faster. These methods create new orders at each bar and delete old ones, thereby preparing data for the next class which will simulate order life cycle. All required order parameters are assigned in this class, including type, open prices, volumes and other important parameters needed for further operation.

The next class implements the order simulation process, as well as calculations of parameters necessary for trading, on the basis of which signals are generated:

```
class Simulation:public PositionGenerator // then assemble a simulator of positions (inherited from the position generator)
   {// market parameter calculations will also performed in this class
   protected:
   double BuyPercent;// percent of open Buy positions
   double SellPercent;// percent of open Sell positions
   double StartVolume;// starting total volume of open Buy positions (the same for Buys and Sells)
   double RelativeVolume;// relative volume
   double SummVolumeBuy;// total volume for Buys
   double SummVolumeSell;// total volume for Sells

   public:
   Simulation( ENUM_GRID_WEIGHT WeightStopLimitFillingType0
                     ,int HalfCorridorLimitStop0,int OrdersToOneBar0,int BarsTotal0
                     ,int ExpirationOpenLimit0,int ExpirationOpenStop0
                     ,int ExpirationClose0
                     ,int ProfitPointsCorridorPart0,int LossPointsCorridorPart0
                     ,double VolumeAlphaStop0,double VolumeAlphaLimit0,double VolumeAlphaMarket0)
   :PositionGenerator(WeightStopLimitFillingType0
                     ,HalfCorridorLimitStop0,OrdersToOneBar0,BarsTotal0
                     ,ExpirationOpenLimit0,ExpirationOpenStop0
                     ,ExpirationClose0
                     ,ProfitPointsCorridorPart0,LossPointsCorridorPart0
                     ,VolumeAlphaStop0,VolumeAlphaLimit0,VolumeAlphaMarket0)
      {
      CreateNewOrders();
      CalculateStartVolume();// calculate starting volumes
      UpdateVirtual();// first update virtual orders to process part of them as market orders in the next function
      UpdateMarket();// now update the state of all market orders
      CalculateCurrentVolume();// calculate current volumes of all open orders
      CalculatePercent();// calculate the percentage of positions
      CalculateRelativeVolume();// calculate relative volume
      }

   double GetBuyPercent()// get the percentage of open Buy deals
      {
      return BuyPercent;
      }

   double GetSellPercent()// get the percentage of open Sell deals
      {
      return SellPercent;
      }

   double GetRelativeVolume()// get relative volume
      {
      return RelativeVolume;
      }

   virtual void Update() override
      {
      PositionGenerator::Update();// call everything that was before
      UpdateVirtual();// first update virtual orders to process part of them as market orders in the next function
      UpdateMarket();// now update the state of all market orders
      CalculateCurrentVolume();// calculate current volumes of all open orders
      CalculatePercent();// calculate the percentage of positions
      CalculateRelativeVolume();// calculate relative volume
      }

   virtual void UpdateFast() override
      {
      PositionGenerator::UpdateFast();// call everything that was before
      UpdateVirtualFast();// first update virtual orders to process part of them as market orders in the next function
      UpdateMarketFast();// now update the state of all market orders
      CalculateCurrentVolume();// calculate current volumes of all open orders
      CalculatePercent();// calculate the percentage of positions
      CalculateRelativeVolume();// calculate relative volume
      }

   private:

   void UpdateVirtual()// update the status of virtual orders
      {
      int size=ArraySize(BarOrders[0].BuyLimitOrders);
      int SizeBarOrders=ArraySize(BarOrders);

      if ( VolumeAlphaLimit != 0.0 )
         {
         for ( int i=SizeBarOrders; i>0; i-- )// update the state of limit orders simulating each candlestick
            {
            for ( int j=SizeBarOrders-1; j>i; j-- )// update the state of all candlesticks preceding this one
               {
               for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
                  {
                  if ( BarOrders[j].BuyLimitOrders[k].Status == STATUS_VIRTUAL
                  && BarOrders[j].BuyLimitOrders[k].WantedPrice <= High[i]
                  && BarOrders[j].BuyLimitOrders[k].WantedPrice >= Low[i] )// if the order is virtual and is inside a candlestick, then it turns into a market one
                     {
                     BarOrders[j].BuyLimitOrders[k].Status = STATUS_MARKET;
                     BarOrders[j].BuyLimitOrders[k].IndexMarket = i;
                     }
                  if ( BarOrders[j].SellLimitOrders[k].Status == STATUS_VIRTUAL
                  && BarOrders[j].SellLimitOrders[k].WantedPrice <= High[i]
                  && BarOrders[j].SellLimitOrders[k].WantedPrice >= Low[i] )// the same
                     {
                     BarOrders[j].SellLimitOrders[k].Status = STATUS_MARKET;
                     BarOrders[j].SellLimitOrders[k].IndexMarket = i;
                     }

                  /////// Check for interest expiration of limit players
                  if ( BarOrders[j].BuyLimitOrders[k].Status == STATUS_VIRTUAL )//
                     {
                     if ( BarOrders[j].BuyLimitOrders[k].IndexMarket - 1 >= BarOrders[j].BuyLimitOrders[k].BarsExpirationOpen )
                     BarOrders[j].BuyLimitOrders[k].Status=STATUS_ABORTED;
                     else
                        {
                        if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha > 0.0 )
                        BarOrders[j].BuyLimitOrders[k].VolumeAlpha-=BarOrders[j].BuyLimitOrders[k].VolumeStart/double(BarOrders[j].BuyLimitOrders[k].BarsExpirationOpen);
                        if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyLimitOrders[k].VolumeAlpha=0.0;
                        }
                     }
                  if ( BarOrders[j].SellLimitOrders[k].Status == STATUS_VIRTUAL )//
                     {
                     if ( BarOrders[j].SellLimitOrders[k].IndexMarket - 1 >= BarOrders[j].SellLimitOrders[k].BarsExpirationOpen  )
                     BarOrders[j].SellLimitOrders[k].Status=STATUS_ABORTED;
                     else
                        {
                        if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha > 0.0 )
                        BarOrders[j].SellLimitOrders[k].VolumeAlpha-=BarOrders[j].SellLimitOrders[k].VolumeStart/double(BarOrders[j].SellLimitOrders[k].BarsExpirationOpen);
                        if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellLimitOrders[k].VolumeAlpha=0.0;
                        }
                     }
                  }
               }
            }
         }

      if ( VolumeAlphaStop != 0.0 )
         {
         for ( int i=SizeBarOrders; i>0; i-- )// update the state of limit orders simulating each candlestick
            {
            for ( int j=SizeBarOrders-1; j>i; j-- )// update the state of all candlesticks preceding this one
               {
               for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
                  {
                  if ( BarOrders[j].SellStopOrders[k].Status == STATUS_VIRTUAL
                  && BarOrders[j].SellStopOrders[k].WantedPrice <= High[i]
                  && BarOrders[j].SellStopOrders[k].WantedPrice >= Low[i] )// the same
                     {
                     BarOrders[j].SellStopOrders[k].Status = STATUS_MARKET;
                     BarOrders[j].SellStopOrders[k].IndexMarket = i;
                     }
                  if ( BarOrders[j].BuyStopOrders[k].Status == STATUS_VIRTUAL
                  && BarOrders[j].BuyStopOrders[k].WantedPrice <= High[i]
                  && BarOrders[j].BuyStopOrders[k].WantedPrice >= Low[i] )// the same
                     {
                     BarOrders[j].BuyStopOrders[k].Status = STATUS_MARKET;
                     BarOrders[j].BuyStopOrders[k].IndexMarket = i;
                     }

                  /////// Check for interest expiration of stop and limit players
                  if ( BarOrders[j].BuyStopOrders[k].Status == STATUS_VIRTUAL )//
                     {
                     if ( BarOrders[j].BuyStopOrders[k].IndexMarket - 1 >= BarOrders[j].BuyStopOrders[k].BarsExpirationOpen  )
                     BarOrders[j].BuyStopOrders[k].Status=STATUS_ABORTED;
                     else
                        {
                        if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha > 0.0 )
                        BarOrders[j].BuyStopOrders[k].VolumeAlpha-=BarOrders[j].BuyStopOrders[k].VolumeStart/double(BarOrders[j].BuyStopOrders[k].BarsExpirationOpen);
                        if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyStopOrders[k].VolumeAlpha=0.0;
                        }
                     }
                  if ( BarOrders[j].SellStopOrders[k].Status == STATUS_VIRTUAL )//
                     {
                     if ( BarOrders[j].SellStopOrders[k].IndexMarket - 1 >= BarOrders[j].SellStopOrders[k].BarsExpirationOpen  )
                     BarOrders[j].SellStopOrders[k].Status=STATUS_ABORTED;
                     else
                        {
                        if ( BarOrders[j].SellStopOrders[k].VolumeAlpha > 0.0 )
                        BarOrders[j].SellStopOrders[k].VolumeAlpha-=BarOrders[j].SellStopOrders[k].VolumeStart/double(BarOrders[j].SellStopOrders[k].BarsExpirationOpen);
                        if ( BarOrders[j].SellStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellStopOrders[k].VolumeAlpha=0.0;
                        }
                     }
                  }
               }
            }
         }
      }

   void UpdateMarket()// update the status of market orders
      {
      int size=ArraySize(BarOrders[0].BuyLimitOrders);
      int SizeBarOrders=ArraySize(BarOrders);

      if ( VolumeAlphaLimit != 0.0 )
         {
         for ( int i=SizeBarOrders; i>1; i-- )// update the state of orders simulating each candlestick
            {
            for ( int j=SizeBarOrders-1; j>i; j-- )// update the state of all candlesticks preceding this one
               {
               for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
                  {
                  // Block for closing when prices change
                  if ( BarOrders[j].BuyLimitOrders[k].Status == STATUS_MARKET )//
                     {
                     if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha > 0.0 )
                        {
                        BarOrders[j].BuyLimitOrders[k].VolumeAlpha-=BarOrders[j].BuyLimitOrders[k].VolumeStart*(High[i]-Open[i])/(BarOrders[j].BuyLimitOrders[k].UpPriceToClose-BarOrders[j].BuyLimitOrders[k].WantedPrice);// with profit
                        BarOrders[j].BuyLimitOrders[k].VolumeAlpha-=BarOrders[j].BuyLimitOrders[k].VolumeStart*(Open[i]-Low[i])/(BarOrders[j].BuyLimitOrders[k].WantedPrice-BarOrders[j].BuyLimitOrders[k].LowPriceToClose);// with loss
                        }
                     if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyLimitOrders[k].VolumeAlpha=0.0;
                     }
                  if ( BarOrders[j].SellLimitOrders[k].Status == STATUS_MARKET )//
                     {
                     if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha > 0.0 )
                        {
                        BarOrders[j].SellLimitOrders[k].VolumeAlpha-=BarOrders[j].SellLimitOrders[k].VolumeStart*(Open[i]-Low[i])/(BarOrders[j].SellLimitOrders[k].WantedPrice-BarOrders[j].SellLimitOrders[k].LowPriceToClose);//with profit
                        BarOrders[j].SellLimitOrders[k].VolumeAlpha-=BarOrders[j].SellLimitOrders[k].VolumeStart*(High[i]-Open[i])/(BarOrders[j].SellLimitOrders[k].UpPriceToClose-BarOrders[j].SellLimitOrders[k].WantedPrice);//with loss
                        }
                     if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellLimitOrders[k].VolumeAlpha=0.0;
                     }
                  // End of lock for closing when prices change

                  // Block for closing when time changes******************************************************
                  if ( BarOrders[j].BuyLimitOrders[k].Status == STATUS_MARKET )//
                     {
                     if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha > 0.0 )
                     BarOrders[j].BuyLimitOrders[k].VolumeAlpha-=BarOrders[j].BuyLimitOrders[k].VolumeStart/double(BarOrders[j].BuyLimitOrders[k].BarsExpirationClose);
                     if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyLimitOrders[k].VolumeAlpha=0.0;
                     }
                  if ( BarOrders[j].SellLimitOrders[k].Status == STATUS_MARKET )//
                     {
                     if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha > 0.0 )
                     BarOrders[j].SellLimitOrders[k].VolumeAlpha-=BarOrders[j].SellLimitOrders[k].VolumeStart/double(BarOrders[j].SellLimitOrders[k].BarsExpirationClose);
                     if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellLimitOrders[k].VolumeAlpha=0.0;
                     }
                  }
               }
            }
         }

      if ( VolumeAlphaStop != 0.0 )
         {
         for ( int i=SizeBarOrders; i>1; i-- )// update the state of orders simulating each candlestick
            {
            for ( int j=SizeBarOrders-1; j>i; j-- )// update the state of all candlesticks preceding this one
               {
               for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
                  {
                  // Block for closing when prices change
                  if ( BarOrders[j].BuyStopOrders[k].Status == STATUS_MARKET )//
                     {
                     if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha > 0.0 )
                        {
                        BarOrders[j].BuyStopOrders[k].VolumeAlpha-=BarOrders[j].BuyStopOrders[k].VolumeStart*(High[i]-Open[i])/(BarOrders[j].BuyStopOrders[k].UpPriceToClose-BarOrders[j].BuyStopOrders[k].WantedPrice);// with profit
                        BarOrders[j].BuyStopOrders[k].VolumeAlpha-=BarOrders[j].BuyStopOrders[k].VolumeStart*(Open[i]-Low[i])/(BarOrders[j].BuyStopOrders[k].WantedPrice-BarOrders[j].BuyStopOrders[k].LowPriceToClose);// with loss
                        }
                     if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyStopOrders[k].VolumeAlpha=0.0;
                     }
                  if ( BarOrders[j].SellStopOrders[k].Status == STATUS_MARKET )//
                     {
                     if ( BarOrders[j].SellStopOrders[k].VolumeAlpha > 0.0 )
                        {
                        BarOrders[j].SellStopOrders[k].VolumeAlpha-=BarOrders[j].SellStopOrders[k].VolumeStart*(Open[i]-Low[i])/(BarOrders[j].SellStopOrders[k].WantedPrice-BarOrders[j].SellStopOrders[k].LowPriceToClose);//with profit
                        BarOrders[j].SellStopOrders[k].VolumeAlpha-=BarOrders[j].SellStopOrders[k].VolumeStart*(High[i]-Open[i])/(BarOrders[j].SellStopOrders[k].UpPriceToClose-BarOrders[j].SellStopOrders[k].WantedPrice);//with loss
                        }
                     if ( BarOrders[j].SellStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellStopOrders[k].VolumeAlpha=0.0;
                     }

                  // End of lock for closing when prices change

                  // Block for closing when time changes******************************************************
                  if ( BarOrders[j].BuyStopOrders[k].Status == STATUS_MARKET )//
                     {
                     if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha > 0.0 )
                     BarOrders[j].BuyStopOrders[k].VolumeAlpha-=BarOrders[j].BuyStopOrders[k].VolumeStart/double(BarOrders[j].BuyStopOrders[k].BarsExpirationClose);
                     if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyStopOrders[k].VolumeAlpha=0.0;
                     }
                  if ( BarOrders[j].SellStopOrders[k].Status == STATUS_MARKET )//
                     {
                     if ( BarOrders[j].SellStopOrders[k].VolumeAlpha > 0.0 )
                     BarOrders[j].SellStopOrders[k].VolumeAlpha-=BarOrders[j].SellStopOrders[k].VolumeStart/double(BarOrders[j].SellStopOrders[k].BarsExpirationClose);
                     if ( BarOrders[j].SellStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellStopOrders[k].VolumeAlpha=0.0;
                     }
                  }
               }
            }
         }

      if ( VolumeAlphaMarket != 0.0 )
         {
         for ( int i=SizeBarOrders; i>1; i-- )// update the state of orders simulating each candlestick
            {
            for ( int j=SizeBarOrders-1; j>i; j-- )// update the state of all candlesticks preceding this one
               {
               for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
                  {
                  // Block for closing when prices change
                  /// For obviously market positions
                  if ( BarOrders[j].BuyMarketOrders[k].VolumeAlpha > 0.0 )
                     {
                     BarOrders[j].BuyMarketOrders[k].VolumeAlpha-=BarOrders[j].BuyMarketOrders[k].VolumeStart*(High[i]-Open[i])/(BarOrders[j].BuyMarketOrders[k].UpPriceToClose-BarOrders[j].BuyMarketOrders[k].WantedPrice);// with profit
                     BarOrders[j].BuyMarketOrders[k].VolumeAlpha-=BarOrders[j].BuyMarketOrders[k].VolumeStart*(Open[i]-Low[i])/(BarOrders[j].BuyMarketOrders[k].WantedPrice-BarOrders[j].BuyMarketOrders[k].LowPriceToClose);// with loss
                     }
                  if ( BarOrders[j].BuyMarketOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyMarketOrders[k].VolumeAlpha=0.0;

                  if ( BarOrders[j].SellMarketOrders[k].VolumeAlpha > 0.0 )
                     {
                     BarOrders[j].SellMarketOrders[k].VolumeAlpha-=BarOrders[j].SellMarketOrders[k].VolumeStart*(Open[i]-Low[i])/(BarOrders[j].SellMarketOrders[k].WantedPrice-BarOrders[j].SellMarketOrders[k].LowPriceToClose);// with profit
                     BarOrders[j].SellMarketOrders[k].VolumeAlpha-=BarOrders[j].SellMarketOrders[k].VolumeStart*(High[i]-Open[i])/(BarOrders[j].SellMarketOrders[k].UpPriceToClose-BarOrders[j].SellMarketOrders[k].WantedPrice);// with loss
                     }
                  if ( BarOrders[j].SellMarketOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellMarketOrders[k].VolumeAlpha=0.0;
                  // End of lock for closing when prices change

                  // Block for closing when time changes******************************************************

                  /// For obviously market positions
                  if ( BarOrders[j].BuyMarketOrders[k].VolumeAlpha > 0.0 )
                  BarOrders[j].BuyMarketOrders[k].VolumeAlpha-=BarOrders[j].BuyMarketOrders[k].VolumeStart/double(BarOrders[j].BuyMarketOrders[k].BarsExpirationClose);
                  if ( BarOrders[j].BuyMarketOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyMarketOrders[k].VolumeAlpha=0.0;
                  if ( BarOrders[j].SellMarketOrders[k].VolumeAlpha > 0.0 )
                  BarOrders[j].SellMarketOrders[k].VolumeAlpha-=BarOrders[j].SellMarketOrders[k].VolumeStart/double(BarOrders[j].SellMarketOrders[k].BarsExpirationClose);
                  if ( BarOrders[j].SellMarketOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellMarketOrders[k].VolumeAlpha=0.0;
                  //
                  }
               }
            }
         }
      }

   /// fast methods****
   void UpdateVirtualFast()// update the status of virtual orders
      {
      int SizeBarOrders=ArraySize(BarOrders);
      int size=ArraySize(BarOrders[0].BuyLimitOrders);

      if ( VolumeAlphaLimit != 0.0 )
         {
         for ( int j=SizeBarOrders-1; j>0; j-- )// update the state of all candlesticks preceding this one
            {
            for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
               {
               if ( BarOrders[j].BuyLimitOrders[k].Status == STATUS_VIRTUAL
               && BarOrders[j].BuyLimitOrders[k].WantedPrice <= High[1]
               && BarOrders[j].BuyLimitOrders[k].WantedPrice >= Low[1] )// if the order is virtual and is inside a candlestick, then it turns into a market one
                  {
                  BarOrders[j].BuyLimitOrders[k].Status = STATUS_MARKET;
                  BarOrders[j].BuyLimitOrders[k].IndexMarket = 1;
                  }
               if ( BarOrders[j].SellLimitOrders[k].Status == STATUS_VIRTUAL
               && BarOrders[j].SellLimitOrders[k].WantedPrice <= High[1]
               && BarOrders[j].SellLimitOrders[k].WantedPrice >= Low[1] )// the same
                  {
                  BarOrders[j].SellLimitOrders[k].Status = STATUS_MARKET;
                  BarOrders[j].SellLimitOrders[k].IndexMarket = 1;
                  }

               /////// Check for interest expiration of limit players
               if ( BarOrders[j].BuyLimitOrders[k].Status == STATUS_VIRTUAL )//
                  {
                  if ( BarOrders[j].BuyLimitOrders[k].IndexMarket - 1 >= BarOrders[j].BuyLimitOrders[k].BarsExpirationOpen )
                  BarOrders[j].BuyLimitOrders[k].Status=STATUS_ABORTED;
                  else
                     {
                     if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha > 0.0 )
                     BarOrders[j].BuyLimitOrders[k].VolumeAlpha-=BarOrders[j].BuyLimitOrders[k].VolumeStart/double(BarOrders[j].BuyLimitOrders[k].BarsExpirationOpen);
                     if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyLimitOrders[k].VolumeAlpha=0.0;
                     }
                  }
               if ( BarOrders[j].SellLimitOrders[k].Status == STATUS_VIRTUAL )//
                  {
                  if ( BarOrders[j].SellLimitOrders[k].IndexMarket - 1 >= BarOrders[j].SellLimitOrders[k].BarsExpirationOpen  )
                  BarOrders[j].SellLimitOrders[k].Status=STATUS_ABORTED;
                  else
                     {
                     if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha > 0.0 )
                     BarOrders[j].SellLimitOrders[k].VolumeAlpha-=BarOrders[j].SellLimitOrders[k].VolumeStart/double(BarOrders[j].SellLimitOrders[k].BarsExpirationOpen);
                     if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellLimitOrders[k].VolumeAlpha=0.0;
                     }
                  }
               }
            }
         }

      if ( VolumeAlphaStop != 0.0 )
         {
         for ( int j=SizeBarOrders-1; j>0; j-- )// update the state of all candlesticks preceding this one
            {
            for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
               {
               if ( BarOrders[j].SellStopOrders[k].Status == STATUS_VIRTUAL
               && BarOrders[j].SellStopOrders[k].WantedPrice <= High[1]
               && BarOrders[j].SellStopOrders[k].WantedPrice >= Low[1] )// the same
                  {
                  BarOrders[j].SellStopOrders[k].Status = STATUS_MARKET;
                  BarOrders[j].SellStopOrders[k].IndexMarket = 1;
                  }
               if ( BarOrders[j].BuyStopOrders[k].Status == STATUS_VIRTUAL
               && BarOrders[j].BuyStopOrders[k].WantedPrice <= High[1]
               && BarOrders[j].BuyStopOrders[k].WantedPrice >= Low[1] )// the same
                  {
                  BarOrders[j].BuyStopOrders[k].Status = STATUS_MARKET;
                  BarOrders[j].BuyStopOrders[k].IndexMarket = 1;
                  }

               /////// Check for interest expiration of stop players
               if ( BarOrders[j].BuyStopOrders[k].Status == STATUS_VIRTUAL )//
                  {
                  if ( BarOrders[j].BuyStopOrders[k].IndexMarket - 1 >= BarOrders[j].BuyStopOrders[k].BarsExpirationOpen  )
                  BarOrders[j].BuyStopOrders[k].Status=STATUS_ABORTED;
                  else
                     {
                     if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha > 0.0 )
                     BarOrders[j].BuyStopOrders[k].VolumeAlpha-=BarOrders[j].BuyStopOrders[k].VolumeStart/double(BarOrders[j].BuyStopOrders[k].BarsExpirationOpen);
                     if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyStopOrders[k].VolumeAlpha=0.0;
                     }
                  }
               if ( BarOrders[j].SellStopOrders[k].Status == STATUS_VIRTUAL )//
                  {
                  if ( BarOrders[j].SellStopOrders[k].IndexMarket - 1 >= BarOrders[j].SellStopOrders[k].BarsExpirationOpen  )
                  BarOrders[j].SellStopOrders[k].Status=STATUS_ABORTED;
                  else
                     {
                     if ( BarOrders[j].SellStopOrders[k].VolumeAlpha > 0.0 )
                     BarOrders[j].SellStopOrders[k].VolumeAlpha-=BarOrders[j].SellStopOrders[k].VolumeStart/double(BarOrders[j].SellStopOrders[k].BarsExpirationOpen);
                     if ( BarOrders[j].SellStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellStopOrders[k].VolumeAlpha=0.0;
                     }
                  }
               }
            }
         }
      }

   void UpdateMarketFast()// update the status of market orders
      {
      int size=ArraySize(BarOrders[0].BuyLimitOrders);
      int SizeBarOrders=ArraySize(BarOrders);

      if ( VolumeAlphaLimit != 0.0 )
         {
         for ( int j=SizeBarOrders-1; j>0; j-- )// update the state of all candlesticks preceding this one
            {
            for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
               {
               // Block for closing when prices change
               if ( BarOrders[j].BuyLimitOrders[k].Status == STATUS_MARKET )//
                  {
                  if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha > 0.0 )
                     {
                     BarOrders[j].BuyLimitOrders[k].VolumeAlpha-=BarOrders[j].BuyLimitOrders[k].VolumeStart*(High[1]-Open[1])/(BarOrders[j].BuyLimitOrders[k].UpPriceToClose-BarOrders[j].BuyLimitOrders[k].WantedPrice);// with profit
                     BarOrders[j].BuyLimitOrders[k].VolumeAlpha-=BarOrders[j].BuyLimitOrders[k].VolumeStart*(Open[1]-Low[1])/(BarOrders[j].BuyLimitOrders[k].WantedPrice-BarOrders[j].BuyLimitOrders[k].LowPriceToClose);// with loss
                     }
                  if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyLimitOrders[k].VolumeAlpha=0.0;
                  }
               if ( BarOrders[j].SellLimitOrders[k].Status == STATUS_MARKET )//
                  {
                  if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha > 0.0 )
                     {
                     BarOrders[j].SellLimitOrders[k].VolumeAlpha-=BarOrders[j].SellLimitOrders[k].VolumeStart*(Open[1]-Low[1])/(BarOrders[j].SellLimitOrders[k].WantedPrice-BarOrders[j].SellLimitOrders[k].LowPriceToClose);// with profit
                     BarOrders[j].SellLimitOrders[k].VolumeAlpha-=BarOrders[j].SellLimitOrders[k].VolumeStart*(High[1]-Open[1])/(BarOrders[j].SellLimitOrders[k].UpPriceToClose-BarOrders[j].SellLimitOrders[k].WantedPrice);// with loss
                     }
                  if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellLimitOrders[k].VolumeAlpha=0.0;
                  }
               // End of lock for closing when prices change

               // Block for closing when time changes******************************************************
               if ( BarOrders[j].BuyLimitOrders[k].Status == STATUS_MARKET )//
                  {
                  if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha > 0.0 )
                  BarOrders[j].BuyLimitOrders[k].VolumeAlpha-=BarOrders[j].BuyLimitOrders[k].VolumeStart/double(BarOrders[j].BuyLimitOrders[k].BarsExpirationClose);
                  if ( BarOrders[j].BuyLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyLimitOrders[k].VolumeAlpha=0.0;
                  }
               if ( BarOrders[j].SellLimitOrders[k].Status == STATUS_MARKET )//
                  {
                  if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha > 0.0 )
                  BarOrders[j].SellLimitOrders[k].VolumeAlpha-=BarOrders[j].SellLimitOrders[k].VolumeStart/double(BarOrders[j].SellLimitOrders[k].BarsExpirationClose);
                  if ( BarOrders[j].SellLimitOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellLimitOrders[k].VolumeAlpha=0.0;
                  }
               //
               }
            }
         }

      if ( VolumeAlphaStop != 0.0 )
         {
         for ( int j=SizeBarOrders-1; j>0; j-- )// update the state of all candlesticks preceding this one
            {
            for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
               {
               // Block for closing when prices change
               if ( BarOrders[j].BuyStopOrders[k].Status == STATUS_MARKET )//
                  {
                  if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha > 0.0 )
                     {
                     BarOrders[j].BuyStopOrders[k].VolumeAlpha-=BarOrders[j].BuyStopOrders[k].VolumeStart*(High[1]-Open[1])/(BarOrders[j].BuyStopOrders[k].UpPriceToClose-BarOrders[j].BuyStopOrders[k].WantedPrice);// with profit
                     BarOrders[j].BuyStopOrders[k].VolumeAlpha-=BarOrders[j].BuyStopOrders[k].VolumeStart*(Open[1]-Low[1])/(BarOrders[j].BuyStopOrders[k].WantedPrice-BarOrders[j].BuyStopOrders[k].LowPriceToClose);// with loss
                     }
                  if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyStopOrders[k].VolumeAlpha=0.0;
                  }
               if ( BarOrders[j].SellStopOrders[k].Status == STATUS_MARKET )//
                  {
                  if ( BarOrders[j].SellStopOrders[k].VolumeAlpha > 0.0 )
                     {
                     BarOrders[j].SellStopOrders[k].VolumeAlpha-=BarOrders[j].SellStopOrders[k].VolumeStart*(Open[1]-Low[1])/(BarOrders[j].SellStopOrders[k].WantedPrice-BarOrders[j].SellStopOrders[k].LowPriceToClose);// with profit
                     BarOrders[j].SellStopOrders[k].VolumeAlpha-=BarOrders[j].SellStopOrders[k].VolumeStart*(High[1]-Open[1])/(BarOrders[j].SellStopOrders[k].UpPriceToClose-BarOrders[j].SellStopOrders[k].WantedPrice);// with loss
                     }
                  if ( BarOrders[j].SellStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellStopOrders[k].VolumeAlpha=0.0;
                  }

               // End of lock for closing when prices change

               // Block for closing when time changes******************************************************
               if ( BarOrders[j].BuyStopOrders[k].Status == STATUS_MARKET )//
                  {
                  if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha > 0.0 )
                  BarOrders[j].BuyStopOrders[k].VolumeAlpha-=BarOrders[j].BuyStopOrders[k].VolumeStart/double(BarOrders[j].BuyStopOrders[k].BarsExpirationClose);
                  if ( BarOrders[j].BuyStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyStopOrders[k].VolumeAlpha=0.0;
                  }
               if ( BarOrders[j].SellStopOrders[k].Status == STATUS_MARKET )//
                  {
                  if ( BarOrders[j].SellStopOrders[k].VolumeAlpha > 0.0 )
                  BarOrders[j].SellStopOrders[k].VolumeAlpha-=BarOrders[j].SellStopOrders[k].VolumeStart/double(BarOrders[j].SellStopOrders[k].BarsExpirationClose);
                  if ( BarOrders[j].SellStopOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellStopOrders[k].VolumeAlpha=0.0;
                  }
               //
               }
            }
         }

       if ( VolumeAlphaMarket != 0.0 )
         {
         for ( int j=SizeBarOrders-1; j>0; j-- )// update the state of all candlesticks preceding this one
            {
            for ( int k=0; k<size; k++ )// update the state inside each preceding candlestick
               {
               /// For obviously market positions
               if ( BarOrders[j].BuyMarketOrders[k].VolumeAlpha > 0.0 )
                  {
                  BarOrders[j].BuyMarketOrders[k].VolumeAlpha-=BarOrders[j].BuyMarketOrders[k].VolumeStart*(High[1]-Open[1])/(BarOrders[j].BuyMarketOrders[k].UpPriceToClose-BarOrders[j].BuyMarketOrders[k].WantedPrice);// with profit
                  BarOrders[j].BuyMarketOrders[k].VolumeAlpha-=BarOrders[j].BuyMarketOrders[k].VolumeStart*(Open[1]-Low[1])/(BarOrders[j].BuyMarketOrders[k].WantedPrice-BarOrders[j].BuyMarketOrders[k].LowPriceToClose);// with loss
                  }
               if ( BarOrders[j].BuyMarketOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyMarketOrders[k].VolumeAlpha=0.0;

               if ( BarOrders[j].SellMarketOrders[k].VolumeAlpha > 0.0 )
                  {
                  BarOrders[j].SellMarketOrders[k].VolumeAlpha-=BarOrders[j].SellMarketOrders[k].VolumeStart*(Open[1]-Low[1])/(BarOrders[j].SellMarketOrders[k].WantedPrice-BarOrders[j].SellMarketOrders[k].LowPriceToClose);// with profit
                  BarOrders[j].SellMarketOrders[k].VolumeAlpha-=BarOrders[j].SellMarketOrders[k].VolumeStart*(High[1]-Open[1])/(BarOrders[j].SellMarketOrders[k].UpPriceToClose-BarOrders[j].SellMarketOrders[k].WantedPrice);// with loss
                  }
               if ( BarOrders[j].SellMarketOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellMarketOrders[k].VolumeAlpha=0.0;
               // End of lock for closing when prices change

               // Block for closing when time changes******************************************************

               /// For obviously market positions
               if ( BarOrders[j].BuyMarketOrders[k].VolumeAlpha > 0.0 )
               BarOrders[j].BuyMarketOrders[k].VolumeAlpha-=BarOrders[j].BuyMarketOrders[k].VolumeStart/double(BarOrders[j].BuyMarketOrders[k].BarsExpirationClose);
               if ( BarOrders[j].BuyMarketOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].BuyMarketOrders[k].VolumeAlpha=0.0;
               if ( BarOrders[j].SellMarketOrders[k].VolumeAlpha > 0.0 )
               BarOrders[j].SellMarketOrders[k].VolumeAlpha-=BarOrders[j].SellMarketOrders[k].VolumeStart/double(BarOrders[j].SellMarketOrders[k].BarsExpirationClose);
               if ( BarOrders[j].SellMarketOrders[k].VolumeAlpha < 0.0 ) BarOrders[j].SellMarketOrders[k].VolumeAlpha=0.0;
               //
               }
            }
         }
      }
   ///******

   void CalculateStartVolume()// calculate the starting total volume of all positions (relative to it we will estimate market fullness)
      {
      StartVolume=0;
      int size=ArraySize(BarOrders[0].BuyStopOrders);
      if ( VolumeAlphaStop != 0.0 )
         {
         for ( int j=ArraySize(BarOrders)-1; j>=0; j-- )
            {
            for ( int i=0; i<size; i++ )
               {
               StartVolume+=BarOrders[j].BuyStopOrders[i].VolumeStart;
               }
            }
         }

      if ( VolumeAlphaLimit != 0.0 )
         {
         size=ArraySize(BarOrders[0].BuyLimitOrders);
         for ( int j=ArraySize(BarOrders)-1; j>=0; j-- )
            {
            for ( int i=0; i<size; i++ )
               {
               StartVolume+=BarOrders[j].BuyLimitOrders[i].VolumeStart;
               }
            }
         }

      if ( VolumeAlphaMarket != 0.0 )
         {
         size=ArraySize(BarOrders[0].BuyMarketOrders);
         for ( int j=ArraySize(BarOrders)-1; j>=0; j-- )
            {
            for ( int i=0; i<size; i++ )
               {
               StartVolume+=BarOrders[j].BuyMarketOrders[i].VolumeStart;
               }
            }
         }
      }

   void CalculateCurrentVolume()// calculate the current total volume of all positions
      {
      SummVolumeBuy=0;
      SummVolumeSell=0;
      int size=ArraySize(BarOrders[0].BuyStopOrders);

      if ( VolumeAlphaStop != 0.0 )
         {
         for ( int j=ArraySize(BarOrders)-1; j>=0; j-- )
            {
            for ( int i=0; i<size; i++ )
               {
               if ( BarOrders[j].BuyStopOrders[i].Status == STATUS_MARKET )
               SummVolumeBuy+=BarOrders[j].BuyStopOrders[i].VolumeAlpha;
               if ( BarOrders[j].SellStopOrders[i].Status == STATUS_MARKET )
               SummVolumeSell+=BarOrders[j].SellStopOrders[i].VolumeAlpha;
               }
            }
         }

      if ( VolumeAlphaLimit != 0.0 )
         {
         size=ArraySize(BarOrders[0].BuyLimitOrders);
         for ( int j=ArraySize(BarOrders)-1; j>=0; j-- )
            {
            for ( int i=0; i<size; i++ )
               {
               if ( BarOrders[j].BuyLimitOrders[i].Status == STATUS_MARKET )
               SummVolumeBuy+=BarOrders[j].BuyLimitOrders[i].VolumeAlpha;
               if ( BarOrders[j].SellLimitOrders[i].Status == STATUS_MARKET )
               SummVolumeSell+=BarOrders[j].SellLimitOrders[i].VolumeAlpha;
               }
            }
         }

      if ( VolumeAlphaMarket != 0.0 )
         {
         size=ArraySize(BarOrders[0].BuyMarketOrders);
         for ( int j=ArraySize(BarOrders)-1; j>=0; j-- )
            {
            for ( int i=0; i<size; i++ )
               {
               SummVolumeBuy+=BarOrders[j].BuyMarketOrders[i].VolumeAlpha;
               SummVolumeSell+=BarOrders[j].SellMarketOrders[i].VolumeAlpha;
               }
            }
         }
      }

   void CalculatePercent()// calculate the percentage of Buys and Sells relative to all positions
      {
      if ( (SummVolumeBuy+SummVolumeSell) != 0.0 ) BuyPercent=100.0*SummVolumeBuy/(SummVolumeBuy+SummVolumeSell);
      else BuyPercent=50;
      if ( (SummVolumeBuy+SummVolumeSell) != 0.0 ) SellPercent=100.0*SummVolumeSell/(SummVolumeBuy+SummVolumeSell);
      else SellPercent=50;
      }

   void CalculateRelativeVolume()// calculate relative volumes of Buys and Sells (calculate only uncovered part of positions)
      {
      if ( SummVolumeBuy >= SummVolumeSell ) RelativeVolume=(SummVolumeBuy-SummVolumeSell)/StartVolume;
      else RelativeVolume=(SummVolumeSell-SummVolumeBuy)/StartVolume;
      }

   };
```

The entire code is valid for both MetaTrader 4 and MetaTrader 5. These classes can be compiled in both platforms. Of course in MetaTrader 5 you should implement predefined arrays in advance, the same as in MQL4. I will not provide this code here. You can check it in the attached source code. My code is not very original. All we need to do now is to implement the variables responsible for trading, and the relevant functionality. Expert Advisors for both terminals are attached below.

The code is very resource intensive, that is why I organized slow logic implementation for starting analysis and fast implementation for working on bars. All my Expert Advisors work by bars to avoid dependence on artificial tick generation and other undesirable consequences of every-tick testing. Of course, we could completely avoid slow functions. But in this case there would have been no starting analysis. Moreover, I do not like to implement functionality outside the class body, as this ruins the integrity of the picture, in my opinion.

The last class constructor will accept the following parameters, which will be inputs. If you wish, you can make many such simulations, because it is only a class instance:

```
input bool bPrintE=false;// print market parameters
input CLOSE_MODE CloseModeE=CLOSE_FAST;// order closing mode
input WORK_MODE ModeE=MODE_SIMPLE;// simulation mode
input ENUM_GRID_WEIGHT WeightFillingE=WEIGHT_SAME;// weight distribution type
input double LimitVolumeE=0.5;// significance of limit orders
input double StopVolumeE=0.5;// significance of stop orders
input double MarketVolume=0.5;// significance of market orders
input int ExpirationBars=100;// bars for full expiration of open orders
input int ExpirationOpenStopBars=1000;// patience of a stop player in bars, after which the order is canceled
input int ExpirationOpenLimitBars=1000;// patience of a limit player in bars, after which the order is canceled
input int ProfitPointsCloseE=200;// points to close with profit
input int LossPointsCloseE=400;// points to close with loss
input int HalfCorridorE=500;// half-corridor for limit and stop orders
input int OrdersToOneBarE=50;// orders for half-grid per 1 bar
input int BarsE=250;// bars for analysis
input double MinPercentE=60;// minimum superiority of one trading side in percentage
input double MaxPercentE=80;// maximum percentage
input double MinRelativeVolumeE=0.0001;// minimum market filling [0...1]
input double MaxRelativeVolumeE=1.00;// maximum market filling [0...1]
```

I will not provide here variables related to trading, as everything is simple and clear.

Here is my function in which trading is implemented:

```
void Trade()
   {
   if ( Area0 == NULL )
      {
      CalcAllMQL5Values();
      Area0 = new Simulation(WeightFillingE,HalfCorridorE,OrdersToOneBarE,BarsE
       ,ExpirationOpenLimitBars,ExpirationOpenStopBars,ExpirationBars,ProfitPointsCloseE,LossPointsCloseE
       ,StopVolumeE,LimitVolumeE,MarketVolume);
      }

   switch(ModeE)
      {
      case MODE_SIMPLE:
         Area0.Update();// update simulation
      case MODE_FAST:
         Area0.UpdateFast();// fast update simulation
      }

   if (bPrintE)
      {
      Print("BuyPercent= ",Area0.GetBuyPercent());
      Print("SellPercent= ",Area0.GetSellPercent());
      Print("RelativeVolume= ",Area0.GetRelativeVolume());
      }

   if ( CloseModeE == CLOSE_FAST && Area0.GetBuyPercent() > 50.0 )
      {
      if ( !bInvert ) CloseBuyF();
      else CloseSellF();
      }

   if ( CloseModeE == CLOSE_FAST && Area0.GetSellPercent() > 50.0 )
      {
      if ( !bInvert ) CloseSellF();
      else CloseBuyF();
      }

   if ( Area0.GetBuyPercent() > MinPercentE && Area0.GetBuyPercent() < MaxPercentE
   && Area0.GetRelativeVolume() >= MinRelativeVolumeE && Area0.GetRelativeVolume() <= MaxRelativeVolumeE )
      {
      if ( !bInvert )
         {
         CloseBuyF();
         SellF();
         }
      else
         {
         CloseSellF();
         BuyF();
         }
      }

   if ( Area0.GetSellPercent() > MinPercentE && Area0.GetSellPercent() < MaxPercentE
   && Area0.GetRelativeVolume() >= MinRelativeVolumeE && Area0.GetRelativeVolume() <= MaxRelativeVolumeE )
      {
      if ( !bInvert )
         {
         CloseSellF();
         BuyF();
         }
      else
         {
         CloseBuyF();
         SellF();
         }
      }
   }
```

If desired, it is possible to make better trading conditions and a more complicated function, but so far I do not see the point in this. I always try to separate the trading from the logical part. The logic is implemented in the object. The simulation object is created dynamically in the trading function at the first trigger; it is deleted during deinitialization. This is because of unavailability of predefined arrays in MQL5, and they have to be created artificially to ensure class operation similar to MQL4.

### How to find working settings?

Based on my experience, I can say that it is better to select settings manually. Furthermore, I suggest ignoring spread at the first stage for Expert Advisors running on low timeframes and having small mathematical expectation - this will help you not to miss performance signs at the very beginning. MetaTrader 4 is very good for this purpose. A successful search is always followed by revision, edits and so on, in an effort to increase the final mathematical expectation and profit factor (signal strength). Also, the structure of the input data, ideally, should allow for independent modes of operation. In other words, one setting should have the most independent effect on the system performance, despite the value of the other settings. This approach can enhance the overall signal by setting balanced settings which will combine all the found signals together. Such an implementation is not always possible, but in our case it is applicable as we analyze each order type separately.

### Testing the Expert Advisor

This Expert Advisor has been written without any preparation, from scratch. So this EA code is new for me. I had a similar EA, but it was much simpler and had a completely different logic. I want to emphasize this, because the purpose of the article was to show that any idea that has some correct underlying basics, even if it does not fully describe the market physics, has a very high chance of showing some performance. If the system shows basic performance abilities, we can test it, find what works in this system and go to the next level in market understanding. The next level assumes better quality Expert Advisors which you will generate yourself.

When testing the Expert Advisor, I had to spend several days patiently selecting trading parameters. Everything was boring and long, but I managed to find working settings. Of course, these settings are very weak and the simulation is performed using only 1 order type, but this was done on purpose. Because it is better to analyze separately how a specific order type affects a signal, and then to try to simulate other order types. Here is my result in MetaTrader 4:

![EURUSD 2010.01.01 -2020.11.01](https://c.mql5.com/2/41/1__11.png)

I first created a version for MetaTrader 4 and tested it with the lowest possible spread. The fact is, that we need to see every tick in order to search for patterns, especially on lower timeframes. When testing the MetaTrader 5 version, we cannot see that because of the spreads, which MetaTester 5 can adjusts at its discretion.

This is a perfect tool for stress testing and for evaluating real system performance. It should be used at the last stage to test the system on real ticks. In our case, it is better to start testing the system in MetaTrader 4. I advise everyone to use this approach, because if you immediately start testing in the fifth version, you can lose a lot of excellent setting options, which can serve as the basis for better-quality settings.

I do not recommend using optimization for many reasons, but the main reason is that it is a simple iteration of parameters. You will not be able to understand how it works, if you do not try the parameters manually. If you take an old radio, connect electric motors to its handles and start them, most probably you will not find a radio station. The same happens here. Even if you manage to catch something, it will be hard to understand what it is.

Another very important aspect is the number of orders. The more orders relative to the number of bars are opened, the stronger physics is found, and you should not bother about its performance in the future. Also remember that the found patterns can lie inside the spread, which can make the system useless if these moments are left uncontrolled!

This is where we need the MetaTrader 5 tester. MetaTrader 5 allows testing a strategy using real ticks. Unfortunately, real ticks exist for a relatively recent period for all currency pairs and instruments. I will test the system version for MetaTrader 5 over the last using real ticks and very strict spread requirements to see if the system works in 2020. But first, I will test the system in the "Every tick" mode on the previously used period:

![](https://c.mql5.com/2/42/1_MT5.png)

This testing mode is not as good as the one using real ticks. However, it is obvious that only a very small part of the signals from the initial result remained here. Nevertheless there are signals that cover the spread and even give a small profit. Also, I am not sure about the spreads which the tester takes from deep history. In this test, the lot was 0.01, which means that the mathematical expectation is 5 points, which is even higher than that of the original test, although the chart looks not very good. This data can still be trusted because there is a huge sample of 100,000 trades in the initial test behind it.

Now let us have a look at the last year:

![](https://c.mql5.com/2/42/1_forward_real_ticks_MT5.png)

In this test, I set the lot to 0.1, thus the mathematical expectation is 23.4 points, which is quite good, considering that the initial test on MetaTrader 4 had the expectation of only 3 points. The expectation may go down in the future, but not by many points. So it will still be enough for a break-even trading.

The Expert Advisor for both terminals is available in the attachment below. You can change settings and try to find working parameters for limit and for market orders. Then you can combine sets and set some average parameters. Unfortunately, I did not have enough time to take the most of it, so there is room for further actions.

Of course, please note that its is not a ready-to-use EA, which you can simply launch on a chart and enjoy. Test the simulation using other types of orders, and then combine the settings and repeat these two testing cycles until the EA shows reliable trading results. Possibly, you can introduce some filters or adjust the algorithms themselves. As can be seen from the tests, further improvements are possible. So you can use the attached program and refine it in an effort to achieve stable results. Also, I did not manage to find suitable settings for higher timeframes. The EA works best on M5, but maybe you can find other timeframes. Also, I did not have time to check how the set performs on other currency pairs. However, usually such flat lines mean that the EA should also work on other currency pairs. I will try to continue refining the EA if I have time. When writing this article, I found some flaws and errors in the EA, so there is still a lot to do.

### Conclusion

The simulator has proved the good expectations regarding this analysis method. After obtaining the first results, it is time to think about how to improve them, how to modernize the algorithm and to make it better, faster and more variable. The biggest advantage of this approach is its maximum simplicity. I am not talking about the logic implementation, but about the underlying logic of this method from a physical point of view.

We cannot take into account all the factors that affect the price movement. But even if there is at least one such factor (even if we cannot describe it reliably), then even its inaccurate code description will produce certain signals. These signals may not be of a very high quality, but they will be quite enough to generate some profit from this idea. Do not try to find an ideal formula, while this is impossible, because you can never take into account all the factors affecting the market.

Moreover, you do not have to use exactly my approach. The purpose of this article was to utilize some physics of the market, which, in my opinion, at least partially describes the market, and write an Expert Advisor which can prove the idea. As a result, I have developed an Expert Advisor, which has shown that such assumptions can serve as a basis for a working system.

Also take into account the fact that the code has minor and bigger flaws and bugs. Even in this form the EA works. You should only refine it. In the next article, I will present another type of multi-asset market analysis, which is simpler, more effective and is known to everyone. Again we will create an Expert Advisor.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8411](https://www.mql5.com/ru/articles/8411)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8411.zip "Download all attachments in the single ZIP archive")

[Simulation.zip](https://www.mql5.com/en/articles/download/8411/simulation.zip "Download Simulation.zip")(27.31 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Multibot in MetaTrader (Part II): Improved dynamic template](https://www.mql5.com/en/articles/14251)
- [Brute force approach to patterns search (Part VI): Cyclic optimization](https://www.mql5.com/en/articles/9305)
- [Brute force approach to patterns search (Part V): Fresh angle](https://www.mql5.com/en/articles/12446)
- [OpenAI's ChatGPT features within the framework of MQL4 and MQL5 development](https://www.mql5.com/en/articles/12475)
- [Rebuy algorithm: Multicurrency trading simulation](https://www.mql5.com/en/articles/12579)
- [Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)
- [Multibot in MetaTrader: Launching multiple robots from a single chart](https://www.mql5.com/en/articles/12434)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/362678)**
(32)


![Dmytro Nazarchuk](https://c.mql5.com/avatar/avatar_na2.png)

**[Dmytro Nazarchuk](https://www.mql5.com/en/users/dmytron)**
\|
25 Jan 2021 at 06:33

**Evgeniy Ilin:**

here is a detailed description of what a flat is [https://www.mql5.com/en/articles/8274](https://www.mql5.com/en/articles/8274)

Thanks - I read it and laughed. I was especially amused by the definition of "chaotic market". It turns out that the probability of triggering protective orders depends on the market, not on their size!

So what is a flat market - give a mathematical definition?

![Evgeniy Ilin](https://c.mql5.com/avatar/2021/1/60171FA1-EC8E.png)

**[Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)**
\|
25 Jan 2021 at 20:49

**Dmytryi Nazarchuk:**

Thanks - read it and laughed. I was especially amused by the definition of "chaotic market". It turns out that the probability of triggering protective orders depends on the market, not on their size!

So what is a flat market - give a mathematical definition of it

I already gave you a link to the article. You laughed. Well, I can't blame you for that. Flat is defined graphically. But we can also give a scalar characterisation of these processes. If we take a fixed number of steps and consider the reference distribution (distribution under the condition that the market is chaotic) we consider the modulo [average price](https://www.mql5.com/en/docs/constants/indicatorconstants/prices#enum_applied_price_enum "MQL5 documentation: Price Constants") movement in the reference distribution and then we consider the modulo average price movement in the available distribution on the market, then the trend is the latter more than the reference distribution. Flat is respectively the opposite situation. The average modulus movement should be less than the reference one. You can also come up with a coefficient of movement to buy or sell similarly, plus ascending, minus descending. By the way, the article contains an example of an indicator that implements these two characteristics, if you have not understood. Satisfied? Or do you want me to throw formulas at you in Paints? Now I will ask you a question. What are you looking for in the articles for yourself?

![Dmytro Nazarchuk](https://c.mql5.com/avatar/avatar_na2.png)

**[Dmytro Nazarchuk](https://www.mql5.com/en/users/dmytron)**
\|
25 Jan 2021 at 21:27

**Evgeniy Ilin:**

I already gave you a link to the article. You laughed. Well, I can't blame you for that. Flat is graphically defined. But we can also give a scalar characterisation of these processes. If we take a fixed number of steps and consider the reference distribution (distribution under the condition that the market is chaotic) we consider the modulo average price movement in the reference distribution and then we consider the modulo average price movement in the available distribution on the market, then the trend is the latter more than the reference distribution. Flat is respectively the opposite situation. The average modulus movement should be less than the reference one. You can also come up with a coefficient of movement to buy or sell similarly, plus ascending, minus descending. By the way, the article contains an example of an indicator that implements these two characteristics, if you have not understood. Satisfied? Or do you want me to throw formulas at you in Paints? Now I will ask you a question. What are you looking for in the articles for yourself?

In this article, I was looking for an explanation of what physics has to do with it. It has nothing to do with physics.

In the first article I was interested in the definition of a 'chaotic market' - a funny link to the probability of protective orders triggering. I have a vague suspicion that as the size of orders decreases, the 'chaoticness' of any market increases.....

![Evgeniy Ilin](https://c.mql5.com/avatar/2021/1/60171FA1-EC8E.png)

**[Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)**
\|
26 Jan 2021 at 09:30

**Dmytryi Nazarchuk:**

In this article, I was looking for an explanation of what physics has to do with it. Physics has nothing to do with it.

In the first article I was interested in the definition of a 'chaotic market' - a funny connection to the probability of triggering protective orders. I have a vague suspicion that as the size of orders decreases, the 'chaoticness' of any market increases.....

This can only be verified experimentally. There is a difference from randomness on all ranges, but in general almost 100 per cent everywhere is a shift towards flatness, because people when they buy they buy in a cluster and sell in a cluster, so waves appear, entry and exit are two half-waves. In many cases it [happens by chance](https://www.mql5.com/en/docs/math/mathrand "MQL5 documentation: MathRand function"), and we can not reliably determine what was in the wave. Those experimental Expert Advisors that I made were able to give about 20-40 points of expectation, while working globally throughout history. This mechanics works everywhere and multicurrency. If you play correctly to make the detection of these waves and filtering of signals, and if you also add averaging with martin ... Only this is a serious project, for which I do not have time yet. There are analogues. Maxim Romanov has a similar approach, but he uses higher TFs. This is true, by the way. I promoted mainly on M5

![LINE123](https://c.mql5.com/avatar/avatar_na2.png)

**[LINE123](https://www.mql5.com/en/users/line123)**
\|
15 May 2021 at 21:09

Well constructed article. Good job Evgenly.


![Finding seasonal patterns in the forex market using the CatBoost algorithm](https://c.mql5.com/2/41/yandex_catboost__3.png)[Finding seasonal patterns in the forex market using the CatBoost algorithm](https://www.mql5.com/en/articles/8863)

The article considers the creation of machine learning models with time filters and discusses the effectiveness of this approach. The human factor can be eliminated now by simply instructing the model to trade at a certain hour of a certain day of the week. Pattern search can be provided by a separate algorithm.

![Neural networks made easy (Part 9): Documenting the work](https://c.mql5.com/2/48/Neural_networks_made_easy_009.png)[Neural networks made easy (Part 9): Documenting the work](https://www.mql5.com/en/articles/8819)

We have already passed a long way and the code in our library is becoming bigger and bigger. This makes it difficult to keep track of all connections and dependencies. Therefore, I suggest creating documentation for the earlier created code and to keep it updating with each new step. Properly prepared documentation will help us see the integrity of our work.

![Brute force approach to pattern search (Part III): New horizons](https://c.mql5.com/2/41/0964a7dfa2f4664c34e6be839022c67a.png)[Brute force approach to pattern search (Part III): New horizons](https://www.mql5.com/en/articles/8661)

This article provides a continuation to the brute force topic, and it introduces new opportunities for market analysis into the program algorithm, thereby accelerating the speed of analysis and improving the quality of results. New additions enable the highest-quality view of global patterns within this approach.

![Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://c.mql5.com/2/41/50_percents__1.png)[Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://www.mql5.com/en/articles/8616)

In the upcoming series of articles, I will demonstrate the development of self-adapting algorithms considering most market factors, as well as show how to systematize these situations, describe them in logic and take them into account in your trading activity. I will start with a very simple algorithm that will gradually acquire theory and evolve into a very complex project.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/8411&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071991235985486400)

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