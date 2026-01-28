---
title: Creating an EA that works automatically (Part 02): Getting started with the code
url: https://www.mql5.com/en/articles/11223
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:10:04.329134
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/11223&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069185406865441098)

MetaTrader 5 / Trading


### Introduction

In the previous article [Creating an EA that works automatically (Part 01): Concepts and structures](https://www.mql5.com/en/articles/11216), we discussed the first steps that anyone needs to understand before proceeding to creating an Expert Advisor that trades automatically. In that article, I showed which concepts should be taken into account and what structure should be created.

But we didn't discuss how to write the code. My purpose was to level the knowledge of the readers, especially beginners who have no practical knowledge in the field of programming. Thus, I tried to bring these people closer to the world of MQL5, showing how to create an EA that can work in a fully automated mode. If you haven't read the previous article, I strongly recommend reading it because it's important to understand the context in which you will work while creating an automated EA.

Okay, let's proceed to writing the code. We will start with the basic system, which is the order system. This will be the starting point for any EA that we ever decide to create.

### Planning

The MetaTrader 5 Standard Library provides a convenient and appropriate way to work with the order system. However, in this article, I want to go further and show you what is behind the scenes of the MetaTrader 5 TRADE library. But my goal today is not to discourage you from using the library. On the contrary, I want to shed the light on what exists in this black box. To do this, we will develop our own library for sending orders. It will not have all those resources present in the MetaTrader 5 Standard Library, while it will only contain the functions required to create and maintain a functional, robust and reliable order system.

In our developments, we will have to use a specific type of construction. I apologize to those new to programming, as you will have to make effort in order to follow the explanations. I will try my best so to keep everything as simple as possible so that you can follow and understand the ideal and the concepts. I'm afraid that without effort you will be just admiring the thing we are going to build but won't be able to develop in MQL5 by yourself Now, let's get started.

### Creating the C\_Orders class

To create a class, we first need to create a file that will contain our class code. In the MetaEditor browser window, navigate to the "Include" folder and right-click on it. Select "New file" and follow the instructions shown in the images below:

![Figure 1](https://c.mql5.com/2/48/001__1.png)

Figure 01. Adding an include file

![Figure 02](https://c.mql5.com/2/48/003__1.png)

Figure 02. This is how the required file is created

After you complete steps from Figures 01 and 02, a file will be created and will be opened in MetaEditor. Its contents can be found below. Before moving further, I want to quickly explain something. Look at Figure 01 — it shows that you can create a class directly. You may wonder why we don't do this. This is the right question. The reason is that if we create a class using the point shown in Figure 01, then the class will not be created from scratch, but will already contain some predefined information or format. However, within this series of articles, it is important for us to create the class from scratch. Let's get back to the code.

```
#property copyright "Daniel Jose"
#property link      ""
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
// #define MacrosHello   "Hello, world!"
// #define MacrosYear    2010
//+------------------------------------------------------------------+
//| DLL imports                                                      |
//+------------------------------------------------------------------+
// #import "user32.dll"
//   int      SendMessageA(int hWnd,int Msg,int wParam,int lParam);
// #import "my_expert.dll"
//   int      ExpertRecalculate(int wParam,int lParam);
// #import
//+------------------------------------------------------------------+
//| EX5 imports                                                      |
//+------------------------------------------------------------------+
// #import "stdlib.ex5"
//   string ErrorDescription(int error_code);
// #import
//+------------------------------------------------------------------+
```

All lines with the light gray texts are comments (as example code) that can be removed if desired. What interests us starts from here. Once the file is opened, we can star adding code to it. Here we will add the code, which will help us to work with our order system.

But to make the code safe, reliable and robust, we will use the tool that MQL5 brought over from C++: **classes**. If you don't know what classes are, I recommend finding this out. You don't need to go directly to the C++ documentation related to classes. You can read about [classes in the MQL5 documentation](https://www.mql5.com/en/docs/basis/types/classes), which will provide you with a good starting point. In addition, the content of the MQL5 documentation is easier to understand than all the confusion that C++ can cause for people who have never heard of classes.

In general, the class is by far the safest and most efficient way to isolate code from other parts of the code. This means that the class is not treated as code, but as a special data type, with which you can do much more than just use primitive types such as **_integer_**, **_double_**, **_boolean_** and others. In other words, for a program, a class is a multifunctional tool. It doesn't really need to know anything about the class was created and what it contains. The program just needs to know how to use it. Think of a class as a power tool: you don't need to know how it was built or what components it has. All you need to know is how to plug in and use it, the way it works won't make any difference to how you use it. This is a simple definition of a class.

And now, let's move on. The first thing we will do is generate the following lines:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
class C_Orders
{
        private :
        public  :
};
//+------------------------------------------------------------------+
```

They are the most basic part of our class. We usually use the same class name as the name of the file in which it is contained, but this is not required. However, this is what can help us find the class later. Inside the class, I indicate two reserved words, they indicate the level of sharing information between them. If you do not add these words, all elements of the class will be considered as public, i.e. anyone can read, write or access them. This breaks some of the hallmarks that make us use class — security and robustness — since any part of the code will be able to access and change what will be inside the class.

In short: everything that is between the declaration of the word private and the word public can be accessed only within the class. You can use global variables here and they will not be accessible outside the class code. Anything declared after the word public can be accessed anywhere in the code, whether it is part of a class or not. Anyone can access whatever it is there.

Once we have created this file as shown above, we can add it to our EA. Then our EA will have the following code:

```
#property copyright "Daniel Jose"
#property version   "1.00"
#property link      "https://www.mql5.com/pt/articles/11223"
//+------------------------------------------------------------------+
#include <Generic Auto Trader\C_Orders.mqh>
//+------------------------------------------------------------------+
int OnInit()
{
        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
```

At this step we have included our class to the EA code using the [Include](https://www.mql5.com/en/docs/basis/preprosessor/include) compilation directive. So, when we use this directive, the compiler understands that from that moment, the header file **C\_Orders.mqh** from the **include** directory of the **Generic Auto Trader** folder must be included into the system and compiled. There are some tricks about this, but I won't go into details because less experienced people can be completely lost trying to understand some details. But what is happening is exactly what I have just described.

### Defining the first functions of the C\_Orders class

Contrary to what many people think, a programmer does not type code non-stop. On the contrary, before starting something, the programmer must think through and study what needs to be done.

Therefore, you, who are just starting your career, should do the same: before adding any lines of code, you should **think over** all the moments. Let's think about what we really need to have in the C\_Orders class in order to do as little work as possible and get the most out of the MetaTrader 5 platform?

The only thing that comes to mind is how the orders are sent. We don't really need some kind of means to move and delete orders or to close positions as the MetaTrader 5 platform gives us all these possibilities. When we place an order on the chart, it appears in the Trade tab of the Toolbox window, so we don't need to do much about this.

Another thing that we don't really need, at least at this first stage, is some mechanism to show orders or positions on the chart, since the platform does this for us. So, in fact the only thing we need to implement is some way to send orders directly from the EA, without going through the order system in MetaTrader 5.

Based on this idea, we can start programming what we really need at this early stage, i.e. a function or a procedure that allows the EA to send orders. At this point many people, especially those who do not have much knowledge or experience in programming start making mistakes.

Since we have already determined that we will use MetaTrader 5 to manage orders and positions through the platform's toolbox, we have to develop a system for sending orders to the server, which is the most basic part of all. But to do this, we need to know some things about the asset which the EA will work with, in order to avoid having to constantly look for this information via MQL5 standard library function calls. Let's start by defining some global variables inside our class.

```
class C_Orders
{
        private :
//+------------------------------------------------------------------+
                MqlTradeRequest m_TradeRequest;
                struct st00
                {
                        int     nDigits;
                        double  VolMinimal,
                                VolStep,
                                PointPerTick,
                                ValuePerPoint,
                                AdjustToTrade;
                        bool    PlotLast;
                }m_Infos;
```

These variables will store the data we need and will be visible throughout the class. They cannot be accessed outside the class, as they are declared after the word private. This means that they can only be viewed and accessed within the class.

Now we need to initialize these variables somehow. There are several methods to do it, but from my experience, I know that many times I end up forgetting to do this. So, let's make it so that we will never forget this — we will use the class constructor as shown below. Let's look at an example below:

```
                C_Orders()
                        {
                                m_Infos.nDigits         = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
                                m_Infos.VolMinimal      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
                                m_Infos.VolStep         = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
                                m_Infos.PointPerTick    = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
                                m_Infos.ValuePerPoint   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
                                m_Infos.AdjustToTrade   = m_Infos.ValuePerPoint / m_Infos.PointPerTick;
                                m_Infos.PlotLast        = (SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE) == SYMBOL_CHART_MODE_LAST);
                        };
```

Class constructors are a safe way to initialize a class before working with it. Note that here I am using a default constructor, i.e. it does not receive any parameters. There are cases in which the constructor may receive parameters to initialize something within the class. But here we do not need to do this.

Now pay attention to the fact that each of the variables is initialized in a certain way, so that when we access any of the values, whether for a test or for a specific calculation, they will be properly initialized. For this reason, you should always give preference to using class constructors to initialize your global and internal class variables, before anything else is done in the code.

Since we are using a constructor, we also need to declare a destructor for the class. It can be as simple as below:

```
                ~C_Orders() { }
```

Look at the syntax. The constructor and the destructor names are the same as the name of the class. But the destructor declaration is preceded by a tilde (~). Another thing to pay attention to is that both the constructor and the destructor do not have a return value. An attempt to do so is considered and error and will make the code impossible to compile.

If you need to return a value during class initialization or termination, you must use a regular procedure call. Neither constructors nor destructors can be used for this purpose.

So, how exactly this type of coding, using constructors and destructor, can help? As mentioned above, it is very common when you try to access or use a global variable without initializing it. This can lead to unusual results and programming errors, even for experienced programmers. By using constructors to initialize global class variables, we ensure that they are always available when we need them.

Therefore, don't get into the bad habit of coding using object-oriented programming without actually using constructors and destructors. Otherwise you can get a lot of headache trying to find out the reason why a program is not working properly.

Before continuing, please pay attention to one detail that is present in the class constructor. It is highlighted below:

```
        m_Infos.PlotLast = (SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE) == SYMBOL_CHART_MODE_LAST);
```

Contrary to what many people think, there are differences between the markets, but not in the way you might imagine. The difference is in the way the assets are plotted on the chart. The previous line determines the type of the histogram used for the asset.

But why is it so important? If we want to create an EA that works with different markets or assets, we must remember that they can have different presentation systems. Basically, there are two main methods:

- BID graphical plot which can be seen in the Forex market;
- LAST graphical plot which us commonly seen in the stock market.

Although they may look the same, the EA and the trade server consider them as different plotting types. This becomes especially clear and important when the EA start sending orders — not market ones which are executed at the best price but pending ones. In this case it is important for us to create a class capable of sending such orders that will remain in the Depth of Market.

But why does an EA need to know which graphical representation system is used? The reason is simple: when an asset uses BID-based representation, the LAST price is always zero. In this case, the EA will not be able to send some types of orders to the server, since it will not know which order type is correct. Even if it manages to send an order and the server accepts it, the order will not be executed correctly since the order type was filled incorrectly.

This problem will occur if you create an EA for the stock market and try to use it on forex. The opposite is also true: if the system uses LAST prices for plotting, but the EA was created for the forex market where charts are plotted based on BID prices, the EA may fail to send the order at the right moment since the Last price can change, while BID and ASK remain static.

However, in the Stock market, the difference between BID and ASK will always be greater than zero, which means there will always be spread between them. The problem is that when creating an order to send to the server, the EA may create it incorrectly, especially if it does not respect the BID and ASK values. This can be fatal for the EA that attempts to trade within the existing spread.

For this reason, the EA checks which market, or rather plotting system, it works with, so that it could be ported from forex to the stocks market or back without any modification or recompilation.

We have already seen how important small nuances are. A simple detail can jeopardize everything. Now let's see how to send a pending order to a trade server.

### Sending a pending order to the server

While studying MQL5 language documentation in relation to how the trading system works, I found the [OrderSend](https://www.mql5.com/en/docs/trading/ordersend) function. You don't have to make the direct use of the function since you can use the MetaTrader 5 standard library and its [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class. However, here I want to show how this feature works behind the scenes.

To send operation requests to the server, we simply use the OrderSend function, but we must be careful when using this function directly. To make sure that it is only used within the class, we will put it inside private. In this way, we can separate all the complexity related to the filling out of the request in the public functions of the class, which can be accessed by the EA.

It is important to be very careful and to perform the required tests so that we know when things go wrong and when they go right. Since all requests will go through the OrderSend function, it is convenient to create a procedure just for it, as shown below:

```
                ulong ToServer(void)
                        {
                                MqlTradeCheckResult TradeCheck;
                                MqlTradeResult      TradeResult;
                                bool bTmp;

                                ResetLastError();
                                ZeroMemory(TradeCheck);
                                ZeroMemory(TradeResult);
                                bTmp = OrderCheck(m_TradeRequest, TradeCheck);
                                if (_LastError == ERR_SUCCESS) bTmp = OrderSend(m_TradeRequest, TradeResult);
                                if (_LastError != ERR_SUCCESS) MessageBox(StringFormat("Error Number: %d", GetLastError()), "Order System", MB_OK);

                                return (_LastError == ERR_SUCCESS ? TradeResult.order : 0);
                        }
```

Don't be scared with the code above. In short, it guarantees that the request is sent and informs us if it was sent correctly. If sending failed, it will instantly notify us. Let's now take a closer look at how it works. First, we reset the internal error variable so that you can analyze any generated errors at any point in the EA.

Do not forget to always check the return values of the most critical procedures. Once this is done, we reset the internal structures of the function and send a request to check the order. **Attention**: instead of checking whether failure is returned in order to find out which error occurred, check the value of the error variable. If it shows a value other than expected, the order will not be sent to the server. If the check shows no errors, the order will be sent to the trade server.

The detail that makes all the difference: regardless of who generates the error, if an error occurs, it code will be displayed on the chart. If everything is perfect, the message will not be shown. This way it doesn't matter which type of request we are sending to the server. If we use only the procedure shown above, we will always have the same type of handling. If more information or tests are needed, it will be enough to add them in this procedure above, doing so you will always guarantee, that the system is as stable and reliable as possible.

The function returns two possible values: order ticket which is returned by the server in case of success or zero which indicates an error. To find out the error type, check the value of the [\_LastError](https://www.mql5.com/en/docs/predefined/_lasterror) variable. In an automated EA this message would be in another place or it would just be a log message, depending on the EA purpose. But since the idea is to show you how to create an EA, I add this information so that you know where the message comes from.

Now we have another problem to solve: you cannot send any random price to the server because the order might be rejected. You should enter the correct price. Some people say that it is enough to normalize the value, but in most cases this does not work. In fact, we need to do a small calculation to use a correct price. We will use the following function here:

```
inline double AdjustPrice(const double value)
                        {
                                return MathRound(value / m_Infos.PointPerTick) * m_Infos.PointPerTick;
                        }
```

This simple calculation will adjust the price, so that regardless of the value entered it will be corrected to an adequate value and will be accepted by the server.

One less thing to worry about. However, the mere fact that we created the above function will not create or send an order to the server. To do this, we must fill in the [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure, which we have already seen in the previous code and accessed through a private global variable **m\_TradeRequest**. This structure should also be filled correctly.

Each request should be filled in a specific form. But here we only want to send a request so that the server will correctly create a pending order. Once this is done, the MetaTrader 5 platform will display the pending order on the chart and in the toolbox.

Filling this structure is one of the main reasons that causes problems in Expert Advisors, both in manual and automated mode. Therefore, we will use our class to create an abstraction and thus facilitate the correct filling so that our request is accepted by the trade server.

But before creating the function, let's think a little. We need to answer a few questions:

- Which type of operation we are going to implement (buying or selling)?
- At what price do we want to execute the deal?
- What will be the volume?
- What is the desired period for the operation?
- What type of orders will we use: buy limit, buy stop, sell limit or sell stop (we are not talking about sell stop limit or buy stop limit now)?
- Which market will we trade: forex or stock market?
- What stop loss to set?
- What take profit are we aiming to achieve?

As you can see, there are several questions, and some of them cannot be used directly in the structure required by the server. Therefore, we will create an abstraction to implement a more practical modeling so that our EA can work without problems. All the adjustments and corrections will be made by the class. Thus, we get the following function:

```
                ulong CreateOrder(const ENUM_ORDER_TYPE type, double Price, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
                        {
                                        double  bid, ask, Desloc;

                                	Price = AdjustPrice(Price);
                                        bid = SymbolInfoDouble(_Symbol, (m_Infos.PlotLast ? SYMBOL_LAST : SYMBOL_BID));
                                        ask = (m_Infos.PlotLast ? bid : SymbolInfoDouble(_Symbol, SYMBOL_ASK));
                                        ZeroMemory(m_TradeRequest);
                                        m_TradeRequest.action           = TRADE_ACTION_PENDING;
                                        m_TradeRequest.symbol           = _Symbol;
                                        m_TradeRequest.volume           = NormalizeDouble(m_Infos.VolMinimal + (m_Infos.VolStep * (Leverage - 1)), m_Infos.nDigits);
                                        m_TradeRequest.type             = (type == ORDER_TYPE_BUY ? (ask >= Price ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_BUY_STOP) :
                                                                                                    (bid < Price ? ORDER_TYPE_SELL_LIMIT : ORDER_TYPE_SELL_STOP));
                                        m_TradeRequest.price            = NormalizeDouble(Price, m_Infos.nDigits);
                                        Desloc = FinanceToPoints(FinanceStop, Leverage);
                                        m_TradeRequest.sl               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? -1 : 1)), m_Infos.nDigits);
                                        Desloc = FinanceToPoints(FinanceTake, Leverage);
                                        m_TradeRequest.tp               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? 1 : -1)), m_Infos.nDigits);
                                        m_TradeRequest.type_time        = (IsDayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
                                        m_TradeRequest.type_filling     = ORDER_FILLING_RETURN;
                                        m_TradeRequest.deviation        = 1000;
                                        m_TradeRequest.comment          = "Order Generated by Experts Advisor.";

                                        return (((type == ORDER_TYPE_BUY) || (type == ORDER_TYPE_SELL)) ? ToServer() : 0);
                };
```

This simple function does a wonderful thing. It takes the data which we want to trade and converts it to the form the server expects to receive, regardless of whether we are trading forex or stocks.

To use this feature, we simply tell it what we want to buy or sell, the price we want, the stop loss we want, the take profit we want, the leverage we want, and whether the trade is intraday or long term. The function will calculate the rest to make sure the order is created correctly. But for this, we will need to make some calculations and adjustments to have adequate values. When using the above function, it is important to keep two points in mind in order to avoid problems when the server creates orders.

Because of these problems orders are often refused by the server. These are Stop Loss and Take Profit. If we indicate zero in any financial value, the function will not generate Take Profit and Stop Loss. So, you must be careful specifying zero if the strategy requires it, because the stop levels will not be created (you can do it later, but at this point they will not be added). Another thing is that negative Stop Loss value for buy deals and negative Take Profit levels for sell deals make no sense. The function will ignore these things. You can just work with positive values, and everything will be fine.

By looking at this function, you can get confused, thinking it is too complicated. But if we see how it works, you will change your mind. But before seeing how it works, let's analyze the FinanceToPoints function that is mentioned in the procedure above. The code of the **FinanceToPoints** function is shown below:

```
inline double FinanceToPoints(const double Finance, const uint Leverage)
                        {
                                double volume = m_Infos.VolMinimal + (m_Infos.VolStep * (Leverage - 1));

                                return AdjustPrice(MathAbs(((Finance / volume) / m_Infos.AdjustToTrade)));
                        };
```

Let's try to figure out what's going on here, as it can be a little tricky. If you do not understand the function, it will be difficult for you to understand what happens when the class sends an order to the server and the server places Take Profit and Stop Loss orders with a certain shift from the entry price. Therefore, it is important to understand how the system works in order to understand this function.

Any asset has information that can be used to convert a financial value into a points. To understand this, look at the images below:

![Figure 05](https://c.mql5.com/2/48/005__1.png)![Figure 06](https://c.mql5.com/2/48/006__1.png)![Figure 07](https://c.mql5.com/2/48/007.png)

All the figures above have highlighted points which are used for converting a financial value into points. For Forex symbols, the ticket size and value are not shown, but they are 1.0 for the ticket value and 0.00001 for the size (number of points). _Do not forget these values._

Now let's consider the following: financial value is the result of dividing the trading volume by the number of points, multiplied by the value of each point. For example, if we assume that we are trading some asset with a minimum volume of 100 and the order size of 2x, that is, twice the minimum volume. In this case, the volume will be 100 x 2, i.e. 200. Don't worry about this value for now, let's continue. To find out the value of each point, we can do the following calculation:

The value of a point is equal to the value per point divided by the number of points. A lot of people assume that the number of points is 1, but this is a mistake. You can see in the figure above that the number of points can differ. For forex it is 0.00001. Therefor you should take care about the point value. Make the program capture the correct value and use it. For example, let's assume that the number of points is 0.01 and the value of each point is 0.01. In this case, dividing one value by the other will give a value of 1. But this value can be different. For example, while in the forex it is 0.00001, for the dollar traded on B3 (Bolsa do Brasil) the value is 10.

Now let's move on to one more thing. To make it easier, let us assume that the user wants to execute a trade for an asset with the minimum volume of 100 with the volume of x3. The size of the ticket is 0.01 and the value is 0.01. Yet the user wants to take a financial risk of 250. How many points should be the offset from the entry price to match this value of 250? This is what the above procedure does. It calculates the value and adjusts it so that we will have a positive or negative shift depending on the case so that the financial values will be 250. In this case we will have 2.5 or 250 points.

This example seems simple. But try to do it quickly while trading the market when the volatility is high, and you have to decide on which size to use very quickly so that you do not exceed the accepted risk limits and do not regret that you opened less than could. At this point, you will understand how important it is to have a well-programmed EA on your side.

With this class, the EA can send a properly configured order to the server. However, to see how it is done, let is create a way to test our order system.

### Testing the order system

I would like to emphasize that the purpose of this series is not to create an EA that can be controlled manually. The idea is to show how a beginner programmer can create an automatically trading EA with as little effort as possible. If you want to know how to create an EA for manual trading, please read another series " [Developing an EA from scratch](https://www.mql5.com/en/articles/10678)". In that series, I show how to create an EA for manual trading. Although, that series can be a little outdated and soon you will see why. For now, we need to focus on our main goal.

In order to test the EA's order system and check if it sends orders to the trade server, we will use some tricks. Let's see how to do it. Usually, I add a horizontal line to the chart, but in this case we just want to check if the order system is working. Therefore, to make testing easier, we use the following:

```
#property copyright "Daniel Jose"
#property description "This one is an automatic Expert Advisor"
#property description "for demonstration. To understand how to"
#property description "develop yours in order to use a particular"
#property description "operational, see the articles where there"
#property description "is an explanation of how to proceed."
#property version   "1.03"
#property link      "https://www.mql5.com/pt/articles/11223"
//+------------------------------------------------------------------+
#include <Generic Auto Trader\C_Orders.mqh>
//+------------------------------------------------------------------+
C_Orders *orders;
ulong m_ticket;
//+------------------------------------------------------------------+
input int       user01   = 1;           //Lot increase
input int       user02   = 100;         //Take Profit (financial)
input int       user03   = 75;          //Stop Loss (financial)
input bool      user04   = true;        //Day Trade ?
input double    user05   = 84.00;       //Entry price...
//+------------------------------------------------------------------+
int OnInit()
{
        orders = new C_Orders();

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        delete orders;
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
#define KEY_UP          38
#define KEY_DOWN        40

        switch (id)
        {
                case CHARTEVENT_KEYDOWN:
                        switch ((int)lparam)
                        {
                                case KEY_UP:
                                        m_ticket = orders.CreateOrder(ORDER_TYPE_BUY, user05, user03, user02, user01, user04);
                                        break;
                                case KEY_DOWN:
                                        m_ticket = orders.CreateOrder(ORDER_TYPE_SELL, user05, user03, user02, user01, user04);
                                        break;
                        }
                        break;
        }
#undef KEY_DOWN
#undef KEY_UP
}
//+------------------------------------------------------------------+
```

Despite its simplicity, this code will allow us to test the order system. To do this, follow these steps to place an order at the desired price:

1. Specify the price at which the order should be placed. Remember that you cannot use the current price;
2. Use the up arrow if you think the price will rise, or the down arrow if you think the price will fall;
3. Then go to the "Trade" tab in the toolbox window. An order should appear with the conditions specified by the EA.

The data will be filled as indicated in the form where the EA can interact with the user. Pay attention how the class is initialized via the constructor. Also pay attention to the destructor code. This way we can be sure that the class will always be initialized before we call any of its internal procedures.

### Conclusion

Although this EA is very simple, you will be able to test the order system and send orders using it. However, remember not to use the current price as you won't be able to see if the system is placing orders in the correct position and with the correct values.

In the following video you can see a demonstration of the correct operation of the system. The attached file provides the full version of the code that we have covered in this article. Use it to experiment and study.

But we are still at the very beginning. Things will get a little more interesting in the next article. However, at this stage it is important to have a good understanding of how orders are placed in the order book.

Demonstração Parte 02 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11223)

MQL5.community

1.91K subscribers

[Demonstração Parte 02](https://www.youtube.com/watch?v=Mlk5U3_JBvM)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 2:57

•Live

•

Demo video

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11223](https://www.mql5.com/pt/articles/11223)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11223.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_02.zip](https://www.mql5.com/en/articles/download/11223/ea_automatico_-_02.zip "Download EA_Automatico_-_02.zip")(3.12 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/442087)**

![Population optimization algorithms: Bat algorithm (BA)](https://c.mql5.com/2/51/Bat-algorithm-avatar.png)[Population optimization algorithms: Bat algorithm (BA)](https://www.mql5.com/en/articles/11915)

In this article, I will consider the Bat Algorithm (BA), which shows good convergence on smooth functions.

![Creating an EA that works automatically (Part 01): Concepts and structures](https://c.mql5.com/2/49/Aprendendo-a-construindo.png)[Creating an EA that works automatically (Part 01): Concepts and structures](https://www.mql5.com/en/articles/11216)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode.

![Creating an EA that works automatically (Part 03): New functions](https://c.mql5.com/2/50/aprendendo_construindo_003_avatar.png)[Creating an EA that works automatically (Part 03): New functions](https://www.mql5.com/en/articles/11226)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. In the previous article, we started to develop an order system that we will use in our automated EA. However, we have created only one of the necessary functions.

![How to choose an Expert Advisor: Twenty strong criteria to reject a trading bot](https://c.mql5.com/2/0/Avatar_Twenty_strong_criteria_to_reject_a_trading_bot.png)[How to choose an Expert Advisor: Twenty strong criteria to reject a trading bot](https://www.mql5.com/en/articles/11933)

This article tries to answer the question: how can we choose the right expert advisors? Which are the best for our portfolio, and how can we filter the large trading bots list available on the market? This article will present twenty clear and strong criteria to reject an expert advisor. Each criterion will be presented and well explained to help you make a more sustained decision and build a more profitable expert advisor collection for your profits.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/11223&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069185406865441098)

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