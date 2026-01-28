---
title: Developing a trading Expert Advisor from scratch (Part 20): New order system (III)
url: https://www.mql5.com/en/articles/10497
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:46:31.047946
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/10497&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062705873173784421)

MetaTrader 5 / Trading systems


### Introduction

In the previous article, [Developing a trading Expert Advisor from scratch (Part 19)](https://www.mql5.com/en/articles/10474), we focused on code changes which were implemented to enable the operation of the new order system. Since these changes have been implemented, I can focus 100% on the real problem. This is to implement the order system, which is 100% visual and understandable for those who trade without having to know the tick value or where to place an order to earn X, or where to set Stop Loss in order not to lose Y.

The creation of such a system requires a good command of MQL5, as well as an understanding of how the MetaTrader 5 platform actually works and what resources it provides.

### 1.0. Planning

### 1.0.1. Designing indicators

The idea here, well not just the idea but what I'm actually going to do, is to show how to implement the system in this article. We will create something similar to what is shown in the image below:

![](https://c.mql5.com/2/45/001__3.png)

It is very easy to understand even without my explanation. There is a closing button, a value and a point to make it easier to drag and place orders. But that's not all. When Stop Loss turns into Stop Gain, the system will handle this as follows:

![](https://c.mql5.com/2/45/002__4.png)

So, we can easily know when, how much, where, and whether or not it is worth holding a certain position.

The figures above only show the objects of the OCO order or OCO position limits, but I have not forgotten the part related to the opening price, as this is also equally important.

For pending orders, this will look as follows:

![](https://c.mql5.com/2/45/003__6.png)

This will look a bit different for a position:

![](https://c.mql5.com/2/45/004__3.png)

However, the proportions are not very encouraging... But this is the idea that will be implemented. As for the colors, I will use the ones shown here. But you can choose the ones that you like.

As we continue planning, we can notice that we will basically have 5 objects in each indicator. It means that MetaTrader 5 will have to process 5 objects at the same time for each indicator. In case of an OCO order, MetaTrader 5 will have to deal with 15 objects, just like with an OCO position, where MetaTrader 5 has to handle 15 objects, which is per one order or position. It means that if you have 4 pending OCO orders and 1 open OCO position, MetaTrader 5 will have to deal with 25 objects, apart from the others that will also be on the chart. And this is if you are using only one asset in the platform.

I say this because it is important to know possible memory and processing required for each order that you are going to place for the trading instrument. Well, this is not a problem for modern computers, but it is necessary to know what exactly we will be demanding from the hardware. Previously, there was only one object on the screen for each point of the order. Now there will be 5 objects in each of the points, and they will have to somehow remain connected. This connection will be implemented by the platform, while we will only instruct how they should be connected and what should happen when each of the object triggers.

### 1.0.2. Choosing objects

The next question is the choice of objects that we will use. This may seem like a simple question but is it a very important one as it determines how the implementation will actually go. The first choice is based on the way objects are positioned on the screen.

We have two ways to implement this. Fortunately, MetaTrader 5 covers both: the first one is using positioning by time and price coordinates and the second one is using Cartesian X and Y coordinates.

However, before we proceed with the one of them in detail, I will immediately discard the model that uses time and price coordinates. Despite being ideal at first sight, it will not be at useful when we deale with so many objects that will be connected to each other and that must remain together. Therefore, we will have to use the Cartesian system.

In one of the previous articles, we already looked at this system and discussed how to select objects. For details see [Multiple indicators on one chart (Part 05)](https://www.mql5.com/en/articles/10277).

We have completed planning, and now we can finally move on to coding itself: it's time to implement things in practice.

### 2.0. Implementation

My purpose is not simply to implement the system, but to explain what exactly is going on in it, so that you could also create your own system based on the considered one, I will provide the details little by little. This will help you understand how it is created. Do not forget that any functionality related to pending orders will also work for positions, since the system follows the same principles and has a common code.

### 2.0.1. Creating an interface framework

The result of this first step can be seen below. This is my way of presenting the benefits so that you get excited as I was when I developed and decided to share these codes with all of you. I hope this will serve as a motivation for those who want to learn programming or to develop deeper knowledge on the subject.

![](https://c.mql5.com/2/45/ScreenRecorderProject85.gif)

Looking at the image above you may be thinking that the functionality was created in a usual way, excluding all the code created so far. But no, we will use exactly what has been built so far.

So, we will use the code that was presented in the previous article and make some changes to it. So, let us focus on what's new in the code. First, we will add three new classes.

### 2.0.1.1. C\_Object\_Base class

We will start by creating a new class - C\_Object\_Base. This is the lowest class in our system. The first codes of the class are shown below:

```
class C_Object_Base
{
        public  :
//+------------------------------------------------------------------+
void Create(string szObjectName, ENUM_OBJECT typeObj)
{
        ObjectCreate(Terminal.Get_ID(), szObjectName, typeObj, 0, 0, 0);
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_SELECTABLE, false);
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_SELECTED, false);
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_BACK, true);
        ObjectSetString(Terminal.Get_ID(), szObjectName, OBJPROP_TOOLTIP, "\n");
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_BACK, false);
        ObjectSetString(Terminal.Get_ID(), szObjectName, OBJPROP_TOOLTIP, "\n");
        PositionAxleY(szObjectName, 9999);
};

// ... The rest of the class code
```

Note that we have general code which will make our life a lot easier. In the same class, we have standard X and Y positioning codes.

```
void PositionAxleX(string szObjectName, int X)
{
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_XDISTANCE, X);
};
//+------------------------------------------------------------------+
virtual void PositionAxleY(string szObjectName, int Y)
{
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_YDISTANCE, Y);
};
```

The Y positioning code will depend on a specific object, but even if the object does not have a specific code, the class provides a general one. We have a general way of specifying the color of the object shown below.

```
virtual void SetColor(string szObjectName, color cor)
{
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_COLOR, cor);
}
```

And here is the way to define the dimensions of the objects.

```
void Size(string szObjectName, int Width, int Height)
{
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_XSIZE, Width);
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_YSIZE, Height);
};
```

This is all for now with the C\_Object\_Base class, but we will get back to it later.

### 2.0.1.2. C\_Object\_BackGround class

Now, let us create two other classes to support our two graphical objects. The first one of them is C\_Object\_BackGround. It creates a background box to receive other elements. Its code is pretty simple. You can see it below in full:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "C_Object_Base.mqh"
//+------------------------------------------------------------------+
class C_Object_BackGround : public C_Object_Base
{
        public:
//+------------------------------------------------------------------+
		void Create(string szObjectName, color cor)
                        {
                                C_Object_Base::Create(szObjectName, OBJ_RECTANGLE_LABEL);
                                ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_BORDER_TYPE, BORDER_FLAT);
                                ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
                                ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_COLOR, clrNONE);
                                ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_BGCOLOR, cor);
                        }
//+------------------------------------------------------------------+
virtual void PositionAxleY(string szObjectName, int Y)
                        {
                                int desl = (int)(ObjectGetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_YSIZE) / 2);
                                ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_YDISTANCE, Y - desl);
                        }
//+------------------------------------------------------------------+
};
```

Note that we are using inheritance to assemble the object with minimal code. Thus, we get the class to modify and model itself as necessary, so that later we do not need to make these adjustments. This can be seen in the highlighted code, where the class will automatically position itself in the correct place simply by knowing the value of the Y axis — it will check the size and will position itself so that it is in the middle of the axis that we are passing to it.

### 2.0.1.3. C\_Object\_TradeLine class

The C\_Object\_TradeLine class is responsible for replacing that horizontal line which was previously used to indicate where the order price line was located. This class is very interesting, so take a look at its code: it has a private static variable, as you can see in the code below.

```
#property copyright "Daniel Jose"
#include "C_Object_BackGround.mqh"
//+------------------------------------------------------------------+
class C_Object_TradeLine : public C_Object_BackGround
{
        private :
                static string m_MemNameObj;
        public  :
//+------------------------------------------------------------------+

// ... Internal class code

//+------------------------------------------------------------------+
};
//+------------------------------------------------------------------+
string C_Object_TradeLine::m_MemNameObj = NULL;
//+------------------------------------------------------------------+
```

It is highlighted to show how to declare it and how to properly initialize it. Well, we could create a global variable to replace what the static variable will do, but I want to maintain control over things: this way each object has everything it needs, and the information is stored in them. And if we want to replace one object with another, we can easily do that.

The next thing to pay attention to is the object creation code.

```
void Create(string szObjectName, color cor)
{
        C_Object_BackGround::Create(szObjectName, cor);
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_XSIZE, TerminalInfoInteger(TERMINAL_SCREEN_WIDTH));
        SpotLight(szObjectName);
};
```

To implement it correctly, we us the C\_Object\_BackGround class, in which we actually create a box that will serve as the line. Again, this is because if another type of object were used we would not have the same behavior that we have now, and the only object that does is as we need is what is present in the C\_Object\_Background class. So, we will modify it in order to adapt to our needs, and thus create a line.

Next, we will see the code responsible for highlighting a line.

```
void SpotLight(string szObjectName = NULL)
{
        if (szObjectName != NULL) ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_YSIZE, (szObjectName != NULL ? 4 : 3));
        if (m_MemNameObj != NULL) ObjectSetInteger(Terminal.Get_ID(), m_MemNameObj, OBJPROP_YSIZE, 3);
        m_MemNameObj = szObjectName;
};
```

This code is quite interesting, since when we highlight a line we don't need to know which one was highlighted, while the object itself does this for us. And when a new line deserves to be highlighted, the line that was highlighted loses this status automatically and the new line takes its place. Now if no line should be highlighted, we just call the function, and it will take care of removing the highlighting from any line.

Knowing this, the code above together with the following code make the old selection code disappear. This way MetaTrader 5 will let us know which indicator we are manipulating.

```
string GetObjectSelected(void) const { return m_MemNameObj; }
```

There is another function worth paying attention to. It positions the line along the Y axis. We will see it below.

```
virtual void PositionAxleY(string szObjectName, int Y)
{
        int desly = (m_MemNameObj == szObjectName ? 2 : 1);
        ObjectSetInteger(Terminal.Get_ID(), szObjectName, OBJPROP_YDISTANCE, Y - desly);
};
```

Like the function shown in the BackGround object, this function also adjust itself the correct point depending on whether the line is highlighted or not.

We have two objects fully completed. But before we can actually see them on the screen (as shown above), we will need to do some things in the C\_ObjectsTrade class.

### 2.0.2. Modification of the C\_ObjectsTrade class

The modifications to be made are not very complicated at first glance, but the number of times we will have to repeat the same code may be a little discouraging at times, so I tried to find a way around this. The first thing we are going to do is create an event enumeration This is using macros, but if you find it confusing to follow code full of macros, feel free to switch from macros to functions, or procedures, and in extreme cases to replace the macros with proper internal code. I prefer to use macros, as I have been doing this for many years.

First, we create an event enumeration.

```
enum eEventType {EV_GROUND = 65, EV_LINE};
```

As objects are created, we must add new events here, and they must be something important. However, each object will only have one type of event, and that event will be generated by MetaTrader 5. Or code will only make sure that the event is handled correctly.

Once this is done, we will create variables that will provide access to each of the objects.

```
C_Object_BackGround     m_BackGround;
C_Object_TradeLine      m_TradeLine;
```

They are in the global scope of the class, but they are private. We could declare them in every function that uses them, but that does not make much sense since the whole class will take care of the objects.

So, we make the corresponding change in the code from the previous article.

```
inline string MountName(ulong ticket, eIndicatorTrade it, eEventType ev)
{
        return StringFormat("%s%c%c%c%d%c%c", def_NameObjectsTrade, def_SeparatorInfo, (char)it, def_SeparatorInfo, ticket, def_SeparatorInfo, (char)ev);
}
```

The highlighted parts did not exist in the previous version, but now they will help MetaTrader 5 to keep us informed about what is happening.

Also, we have a new function.

```
void SetPositionMinimalAxleX(void)
{
        m_PositionMinimalAlxeX = (int)(ChartGetInteger(ChartID(), CHART_WIDTH_IN_PIXELS) * 0.2);
}
```

It creates a starting point along the X axis for the objects. Each of the objects will have a certain point, but here we provide an initial reference. You can modify it to change the starting position by simply changing the point in this code above.

The selection function has undergone a lot of changes, but it will change a little more later. At the time, it is as follows.

```
inline void Select(const string &sparam)
{
        ulong tick;
        double price;
        eIndicatorTrade it;
        eEventType ev;
        string sz = sparam;

        if (!GetInfosOrder(sparam, tick, price, it, ev)) sz = NULL;
        m_TradeLine.SpotLight(sz);
}
```

Another function that has also changed is the one that creates the indicator.

```
inline void CreateIndicatorTrade(ulong ticket, double price, eIndicatorTrade it, bool select)
{
        if (price <= 0) RemoveIndicatorTrade(ticket, it); else
        {
                CreateIndicatorTrade(ticket, it, select);
                PositionAxlePrice(price, ticket, it, -1, -1, 0, false);
        }
}
```

But the code above is not that important. What really does all the hard work is shown in the code below.

```
inline void CreateIndicatorTrade(ulong ticket, eIndicatorTrade it)
{
        color cor1, cor2;
        string sz0;

        switch (it)
        {
                case IT_TAKE    :
                        cor1 = clrPaleGreen;
                        cor2 = clrDarkGreen;
                        break;
                case IT_STOP    :
                        cor1 = clrCoral;
                        cor2 = clrMaroon;
                        break;
                case IT_PENDING:
                default:
                        cor1 = clrGold;
                        cor2 = clrDarkGoldenrod;
                        break;
        }
        m_TradeLine.Create(MountName(ticket, it, EV_LINE), cor2);
        if (ticket == def_IndicatorTicket0) m_TradeLine.SpotLight(MountName(ticket, IT_PENDING, EV_LINE));
        m_BackGround.Create(sz0 = MountName(ticket, it, EV_GROUND), cor1);
        switch (it)
        {
                case IT_TAKE:
                case IT_STOP:
                        m_BackGround.Size(sz0, 92, 22);
                        break;
                case IT_PENDING:
                        m_BackGround.Size(sz0, 110, 22);
                        break;
        }
}
```

In this function, we determine the colors and sequence of creation of objects, as well as determine their sizes. Any object that is added to the indicator should be placed using this function so that everything is centered and always checked. If you start making a function to create indicators, you will end up with a type of code that is difficult to maintain, and it may lack proper checks. You may think that everything is fine, that it works, and you place it on a real account - only then it will be actually checked, and you may suddenly realize that some things do not work properly. Here is advice: always try to assemble functions inside things that do the same job; even if it seems pointless at first, it will make sense over time as you will always be checking things that change.

Below is the next function that has changed.

```
#define macroDelete(A)  {                                                               \
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, A, EV_GROUND));       \
                ObjectDelete(Terminal.Get_ID(), MountName(ticket, A, EV_LINE));         \
                        }

inline void RemoveIndicatorTrade(ulong ticket, eIndicatorTrade it = IT_NULL)
                        {
                                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, false);
                                if ((it != NULL) && (it != IT_PENDING) && (it != IT_RESULT)) macroDelete(it)
                                else
                                {
                                        macroDelete(IT_PENDING);
                                        macroDelete(IT_RESULT);
                                        macroDelete(IT_TAKE);
                                        macroDelete(IT_STOP);
                                }
                                ChartSetInteger(Terminal.Get_ID(), CHART_EVENT_OBJECT_DELETE, true);
                        }
#undef macroDelete
```

Like the next one, it is a bit boring. The function should work so that you are the one who selects each of the created objects one by one. This should apply to each of the indicators. Think about it. If we do not use a macro that makes the task easier, this will be a nightmare. It would be extremely tedious to code this function, since each indicator at the end of the code will have 5 objects. Knowing that each set in an OCO order would have 3 indicators, this would result in us having to use 15 objects, in which case the chances of making a mistake (because the difference between them is only in name) would be huge. So, with the help of a macro, the code is reduced to what is highlighted in the code: we will only code 5 objects at the end. But this is just the first phase to get the result shown above.

To complete the first phase, we have another equally tedious feature. If we were not using macros, we could use procedures instead of macros. But we have selected this way.

```
#define macroSetAxleY(A)        {                                               \
                m_BackGround.PositionAxleY(MountName(ticket, A, EV_GROUND), y); \
                m_TradeLine.PositionAxleY(MountName(ticket, A, EV_LINE), y);    \
                                }

#define macroSetAxleX(A, B)     {                                               \
                m_BackGround.PositionAxleX(MountName(ticket, A, EV_GROUND), B); \
                m_TradeLine.PositionAxleX(MountName(ticket, A, EV_LINE), B);    \
                                }

inline void PositionAxlePrice(double price, ulong ticket, eIndicatorTrade it, int FinanceTake, int FinanceStop, int Leverange, bool isBuy)
                        {
                                double ad;
                                int x, y;

                                ChartTimePriceToXY(Terminal.Get_ID(), 0, 0, price, x, y);
                                macroSetAxleY(it);
                                macroSetAxleX(it, m_PositionMinimalAlxeX);
                                if (Leverange == 0) return;
                                if (it == IT_PENDING)
                                {
                                        ad = Terminal.GetAdjustToTrade() / (Leverange * Terminal.GetVolumeMinimal());
                                        ChartTimePriceToXY(Terminal.Get_ID(), 0, 0, price + Terminal.AdjustPrice(FinanceTake * (isBuy ? ad : (-ad))), x, y);
                                        macroSetAxleY(IT_TAKE);
                                        macroSetAxleX(IT_TAKE, m_PositionMinimalAlxeX + 120);
                                        ChartTimePriceToXY(Terminal.Get_ID(), 0, 0, price + Terminal.AdjustPrice(FinanceStop * (isBuy ? (-ad) : ad)), x, y);
                                        macroSetAxleY(IT_STOP);
                                        macroSetAxleX(IT_STOP, m_PositionMinimalAlxeX + 220);
                                }
                        }
#undef macroSetAxleX
#undef macroSetAxleY
```

If you think the previous feature was boring, then check out this one. Here the work would be doubled, but thanks to the highlighted codes the thing becomes acceptable.

Well, there are other small changes that had to take place, but they are not really worth mentioning, so when we run this code, we get exactly what we expect, namely the indication appeared on the screen.

![](https://c.mql5.com/2/45/ScreenRecorderProject85__1.gif)

### Conclusion

It remains quite a bit before the system is completed and can fully display the order directly on the chart. But now we have to do it all at once, since it is necessary to make very significant changes in other places in the code.

So, we'll leave that for the next article, as the changes will be very deep. And if something goes wrong, then you will have to go back a step and try again until you can change the system the way you want. In this way, you can customize the system, leaving it so that you are comfortable with. In the next article, we will have the system as shown below:

![](https://c.mql5.com/2/45/ScreenRecorderProject87__1.gif)

Seems easy to implement, doesn't it? But trust me, there are a lot of changes that need to be made. So, see you in the next article.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10497](https://www.mql5.com/pt/articles/10497)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10497.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Sistema\_de\_ordens\_k\_III\_1.zip](https://www.mql5.com/en/articles/download/10497/ea_-_sistema_de_ordens_k_iii_1.zip "Download EA_-_Sistema_de_ordens_k_III_1.zip")(12026.29 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)
- [Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/431109)**
(2)


![Sidjkcleto](https://c.mql5.com/avatar/2022/6/629A2734-3EE8.jpg)

**[Sidjkcleto](https://www.mql5.com/en/users/sidjkcleto)**
\|
27 Jun 2022 at 20:04

Good afternoon.

Congratulations on the [project](https://www.mql5.com/en/articles/7863 "Article: Projects let you create profitable trading robots! But it's not exactly"), but I can't seem to get the EA to stick to the chart.

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
29 Jun 2022 at 16:12

**Sidjkcleto project, but I can't seem to get the EA to stick to the chart.**

Algotrading has to be enabled, otherwise the EA won't work ... it will be disabled and you won't be able to use it ...

![Data Science and Machine Learning — Neural Network (Part 02): Feed forward NN Architectures Design](https://c.mql5.com/2/48/forward_neural_network_design.png)[Data Science and Machine Learning — Neural Network (Part 02): Feed forward NN Architectures Design](https://www.mql5.com/en/articles/11334)

There are minor things to cover on the feed-forward neural network before we are through, the design being one of them. Let's see how we can build and design a flexible neural network to our inputs, the number of hidden layers, and the nodes for each of the network.

![Developing a trading Expert Advisor from scratch (Part 19): New order system (II)](https://c.mql5.com/2/47/development__2.png)[Developing a trading Expert Advisor from scratch (Part 19): New order system (II)](https://www.mql5.com/en/articles/10474)

In this article, we will develop a graphical order system of the "look what happens" type. Please note that we are not starting from scratch this time, but we will modify the existing system by adding more objects and events on the chart of the asset we are trading.

![Neural networks made easy (Part 18): Association rules](https://c.mql5.com/2/48/Neural_networks_made_easy_018.png)[Neural networks made easy (Part 18): Association rules](https://www.mql5.com/en/articles/11090)

As a continuation of this series of articles, let's consider another type of problems within unsupervised learning methods: mining association rules. This problem type was first used in retail, namely supermarkets, to analyze market baskets. In this article, we will talk about the applicability of such algorithms in trading.

![Metamodels in machine learning and trading: Original timing of trading orders](https://c.mql5.com/2/42/yandex_catboost__4.png)[Metamodels in machine learning and trading: Original timing of trading orders](https://www.mql5.com/en/articles/9138)

Metamodels in machine learning: Auto creation of trading systems with little or no human intervention — The model decides when and how to trade on its own.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/10497&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062705873173784421)

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