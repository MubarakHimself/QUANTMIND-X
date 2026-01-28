---
title: Market Simulation (Part 04): Creating the C_Orders Class (I)
url: https://www.mql5.com/en/articles/12589
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:29:30.028610
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/12589&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069484014466696664)

MetaTrader 5 / Examples


### Introduction

In the previous article, [Market Simulation (Part 3): A Matter of Performance](https://www.mql5.com/en/articles/12580), we made some adjustments to the classes in order to work around, at least temporarily, certain issues we were experiencing. These issues were degrading the overall performance of the system. While we have resolved these problems for now, we are now facing something genuinely quite complex. Not in this first part, as I have already touched on the topic discussed here in earlier articles. However, we are now approaching things slightly differently. As a result, the way we will handle this issue will also be somewhat different.

I know many of you are eager to see the replay/simulator finally executing orders. However, before doing that, we need to ensure that the order system is fully operational. This is essential for communicating with the real trading server, whether through a DEMO account or a REAL account. In any case, the applications involved — namely the Chart Trade indicator, the Mouse indicator, and the Expert Advisor — must work in perfect harmony, ensuring smooth communication with the real trading server.

In addition to the applications mentioned above, we will need to create other components. However, these can be set aside for now, since their development (both in the conceptual phase and during implementation) depends on several factors that will be addressed in this stage of development.

In this article, I will begin explaining how we will communicate with the trading server. Many of you may already know how to do this quite well. In that case, I ask for a little patience, as we will proceed deliberately rather than rushing. It is crucial that you fully understand what is actually happening. Unlike many typical programming approaches, this system will be built in modules, each responsible for a very specific task. If one module fails, the entire system fails, as there is no backup system to ensure functionality in other ways.

### Understanding the Concept

If you have been following this series, you may have noticed in the article [Developing a Replay System (Part 78): New Chart Trade (V](https://www.mql5.com/en/articles/12492)), where I began showing how interaction would occur, that the Expert Advisor does not actually know where the orders are coming from. However, the same Expert Advisor does know how to interpret the incoming messages. Although the methods used at that time did not allow for a cross-order system, we addressed this issue in subsequent articles. In [Market Simulation (Part 02): Cross Order (II)](https://www.mql5.com/en/articles/12537), I demonstrated how the messaging system would be structured to communicate with the Expert Advisor.

This entire mechanism is part of something even larger. Therefore, understanding this messaging mechanism is crucial for understanding what we will begin exploring in this article. Do not underestimate or overlook what was explained in previous articles. Most importantly, do not assume you fully understand it until you see it in action.

If you have grasped the Expert Advisor code presented earlier, you should not have much difficulty understanding what will be programmed here. Again, though, do not assume knowledge just by looking at the code. You should understand how each detail functions in order to fully understand how the system works as a whole.

### Initiating Market Orders

Since there are many concepts to explain, I will not show the entire class right away in the code below. Similarly, I will not dump a large amount of code into this article. This section requires careful understanding. Because the code we will see will actually handle money. Your money, dear reader. Understanding how each part works will allow you to feel confident and secure with the code presented. I do not want you to modify the code simply because you do not understand it. I want any modifications to be made only when you need to add functionality that is not currently present, and not because you are used to using a particular coding style. Study the code carefully, as it will also be used when we implement order simulation.

To simplify the explanation as much as possible, let us begin by examining the initial class responsible for sending orders to the server — whether the real server, which we are dealing with now, or the simulated server, which we will see later. In any case, the code begins as shown below. The first code snippet is as follows.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "..\Defines.mqh"
005. //+------------------------------------------------------------------+
006. class C_Orders
007. {
008.    protected:
009. //+------------------------------------------------------------------+
010. inline const ulong GetMagicNumber(void) const {return m_MagicNumber;}
011. //+------------------------------------------------------------------+
012.    private   :
013. //+------------------------------------------------------------------+
014.       MqlTradeRequest   m_TradeRequest;
015.       ulong             m_MagicNumber;
016.       bool              m_bTrash;
017. //+------------------------------------------------------------------+
018.       struct stChartTrade
019.       {
020.          struct stEvent
021.          {
022.             EnumEvents  ev;
023.             string      szSymbol,
024.                         szContract;
025.             bool        IsDayTrade;
026.             ushort      Leverange;
027.             double      PointsTake,
028.                         PointsStop;
029.          }Data;
030. //---
031.          bool Decode(const EnumEvents ev, const string sparam)
032.             {
033.                string Res[];
034.
035.                if (StringSplit(sparam, '?', Res) != 7) return false;
036.                stEvent loc = {(EnumEvents) StringToInteger(Res[0]), Res[1], Res[2], (bool)(Res[3] == "D"), (ushort) StringToInteger(Res[4]), StringToDouble(Res[5]), StringToDouble(Res[6])};
037.                if ((ev == loc.ev) && (loc.szSymbol == _Symbol)) Data = loc;
038.
039.                return true;
040.             }
041. //---
042.       }m_ChartTrade;
043. //+------------------------------------------------------------------+
044.       ulong SendToPhysicalServer(void)
045.          {
046.             MqlTradeCheckResult  TradeCheck;
047.             MqlTradeResult       TradeResult;
048.
049.             ZeroMemory(TradeCheck);
050.             ZeroMemory(TradeResult);
051.             if (!OrderCheck(m_TradeRequest, TradeCheck))
052.             {
053.                PrintFormat("Order System - Check Error: %d", GetLastError());
054.                return 0;
055.             }
056.             m_bTrash = OrderSend(m_TradeRequest, TradeResult);
057.             if (TradeResult.retcode != TRADE_RETCODE_DONE)
058.             {
059.                PrintFormat("Order System - Send Error: %d", TradeResult.retcode);
060.                return 0;
061.             };
062.
063.             return TradeResult.order;
064.          }
065. //+------------------------------------------------------------------+
066.       ulong ToMarket(const ENUM_ORDER_TYPE type)
067.          {
068.             double price  = SymbolInfoDouble(m_ChartTrade.Data.szContract, (type == ORDER_TYPE_BUY ? SYMBOL_ASK : SYMBOL_BID));
069.             double vol    = SymbolInfoDouble(m_ChartTrade.Data.szContract, SYMBOL_VOLUME_STEP);
070.             uchar  nDigit = (uchar)SymbolInfoInteger(m_ChartTrade.Data.szContract, SYMBOL_DIGITS);
071.
072.             ZeroMemory(m_TradeRequest);
073.             m_TradeRequest.magic        = m_MagicNumber;
074.             m_TradeRequest.symbol       = m_ChartTrade.Data.szContract;
075.             m_TradeRequest.price        = NormalizeDouble(price, nDigit);
076.             m_TradeRequest.action       = TRADE_ACTION_DEAL;
077.             m_TradeRequest.sl           = NormalizeDouble(m_ChartTrade.Data.PointsStop == 0 ? 0 : price + (m_ChartTrade.Data.PointsStop * (type == ORDER_TYPE_BUY ? -1 : 1)), nDigit);
078.             m_TradeRequest.tp           = NormalizeDouble(m_ChartTrade.Data.PointsTake == 0 ? 0 : price + (m_ChartTrade.Data.PointsTake * (type == ORDER_TYPE_BUY ? 1 : -1)), nDigit);
079.             m_TradeRequest.volume       = NormalizeDouble(vol + (vol * (m_ChartTrade.Data.Leverange - 1)), nDigit);
080.             m_TradeRequest.type         = type;
081.             m_TradeRequest.type_time    = (m_ChartTrade.Data.IsDayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
082.             m_TradeRequest.stoplimit    = 0;
083.             m_TradeRequest.expiration   = 0;
084.             m_TradeRequest.type_filling = ORDER_FILLING_RETURN;
085.             m_TradeRequest.deviation    = 1000;
086.             m_TradeRequest.comment      = "Order Generated by Experts Advisor.";
087.
088.             MqlTradeRequest TradeRequest[1];
089.
090.             TradeRequest[0] = m_TradeRequest;
091.             ArrayPrint(TradeRequest);
092.
093.             return (((type == ORDER_TYPE_BUY) || (type == ORDER_TYPE_SELL)) ? SendToPhysicalServer() : 0);
094.          };
095. //+------------------------------------------------------------------+
096.    public   :
097. //+------------------------------------------------------------------+
098.       C_Orders(const ulong magic)
099.          :m_MagicNumber(magic)
100.          {
101.          }
102. //+------------------------------------------------------------------+
103.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
104.          {
105.             switch (id)
106.             {
107.                case CHARTEVENT_CUSTOM + evChartTradeBuy     :
108.                case CHARTEVENT_CUSTOM + evChartTradeSell    :
109.                case CHARTEVENT_CUSTOM + evChartTradeCloseAll:
110.                   if (m_ChartTrade.Decode((EnumEvents)(id - CHARTEVENT_CUSTOM), sparam)) switch (m_ChartTrade.Data.ev)
111.                   {
112.                      case evChartTradeBuy:
113.                         ToMarket(ORDER_TYPE_BUY);
114.                         break;
115.                      case evChartTradeSell:
116.                         ToMarket(ORDER_TYPE_SELL);
117.                         break;
118.                      case evChartTradeCloseAll:
119.                         break;
120.                   }
121.                   break;
122.             }
123.          }
124. //+------------------------------------------------------------------+
125. };
126. //+------------------------------------------------------------------+
```

Source code of the C\_Replay.mqh file

Very well, let's understand what this code actually does. Although it may seem complicated, it is quite simple and performs a single function: it sends market buy or sell orders upon interaction between the user and the Chart Trade indicator. But how does it manage to do this? You might be thinking that we are somehow going to modify the Chart Trade indicator. If that was your thought, then you have not fully understood how the system works. Indicators cannot send orders to the trading server. They serve to display information on the chart. In the case of the Chart Trade indicator, it enables user interactions with the rest of the system.

Before introducing the Expert Advisor, which will actually make use of this header file, let's take a step back and recall what has already been explained. This will make it easier to understand the header file.

In [Developing a Replay System (Part 78): New Chart Trade (V)](https://www.mql5.com/en/articles/12492), we explained the messages that the Expert Advisor should intercept. You may have noticed that we used the OnChartEvent procedure to capture these messages. In this procedure, we called a class. That class translated the received message and printed it to the terminal for analysis. That was the easy part, as we did not need to use that data to communicate with the trading server.

When you look at the Expert Advisor's source code, you will see that in the OnChartEvent procedure, there is a call to DispatchMessage. This call will actually invoke the header file. Specifically line 103. But before we get there, let's start at the beginning — the class constructor at line 98.

The constructor is quite simple. When called, it receives an argument. This argument will be used to identify the class with a magic number. Note this detail: we are not identifying the Expert Advisor itself, but the class. Why make this distinction? Sometimes it is useful to use similar but separate classes to perform the same type of task. At this moment it may not seem necessary, but it will become clear as we continue. This particular class does not use certain elements, which will be explained later.

For instance, once no orders exist and only positions remain, you may want to handle them in a specific way. Using multiple Expert Advisors for this is prone to errors. Moreover, only one Expert Advisor can operate per chart. (I am not saying you cannot use multiple Expert Advisors for the same instrument. You just cannot have more than one on the same chart.)

Opening multiple charts for the same instrument, just to run different Expert Advisors, is difficult to manage. Some people get it, but I find it very confusing. However, placing slightly different classes within the same Expert Advisor, so they work harmoniously, is perfectly manageable. To make this easier, when the constructor creates the class, it is assigned a magic number. This number will later be used in orders and positions. We will see this later.

The magic number is initialized at line 99, and the variable storing it is declared at line 15. Note that line 12 contains a private clause, which encapsulates everything from lines 12 to 96 within the class, including the variables declared between lines 14 and 16.

Before continuing, pay attention to line 10. It contains a function that returns the magic number defined for this class. Since this line is located between line 8 (a protected clause) and line 12 (the private clause), this function cannot be used outside the inheritance system. In other words, attempting to access it outside a class inheriting from C\_Orders will result in a compiler error.

This function exists now for future purposes. and is only intended to return the magic number. We will not focus on this for now. For now, it's enough to know that this function on line 10 cannot be accessed outside the inheritance system and only serves to return the magic number of this class.

To make the rest of the code easier to understand, we will divide it into topics. However, each topic will refer to the same code.

### A Procedure Within a Structure?

In [Developing a Replay System (Part 78): New Chart Trade (V)](https://www.mql5.com/en/articles/12492), I used a class to implement the translation system. Here, I am using a structure. Can we do that? Yes, because classes are essentially more elaborate structures. Since what we need can be modeled more simply, I chose to use a structure. This is visible between lines 18 and 42.

Pay attention: line 18 declares the structure. If this were a class, everything from lines 18 to 42 would be private. This would require a public clause to adjust access. There is no problem with that. But why would you declare a class within another class? It usually just makes the code more confusing.

The data we need comes from messages, which can be viewed as a set of variables. Why not think of the message as a large variable set? This simplifies understanding. Right?

The message structure is found between lines 20 and 29. Although the overall data structure spans lines 18–42, the message itself is considered as the subset between 20–29. As explained in a previous article, the message is a string. If we tried to parse the string purely based on the structure length, we could easily misinterpret it. Therefore, we need some additional code to decode the message.

Lines 31–40 contain a function within the main structure that handles this. But it is not part of the message structure — it separates the logic, reducing the risk of errors. Why? Because the elements exist separately. We could put it all together: message structure, functions, and procedures. But by doing so, we risk running into problems during implementation.

Now please be attentive because this is quite confusing. If you can understand this, it will become clear to you why many programmers avoid using structures even when they can. And when some programmers try to use structures, they end up turning their programs into time bombs, especially if the code is written in the legacy C language.

Pay close attention to line 36. This line represents a real danger that everything could get mixed up. This line populates the data structure with all the values we need. If this were a function or procedure instead, what would happen upon the execution of line 35? Or even worse, what happens when you call a function or procedure in memory that was overwritten on line 36? overwriting the data could lead to severe consequences, which is a key reason legacy C highlights the importance of using classes. But if you know how it happens, you understand why these classes were created.

We won't go into details of what might happen. But don't forget that a truly evil programmer can do things you can't even imagine. However, let's get back to our code. The function at line 31 performs the same task as before. But please look at line 110 where the function is called, Note that we do not call a function directly by its name, but we need an additional reference This is a reference to the variable that contains the data representing our structure structure. While this may seem confusing, it is not. Forget for a moment that we're using a structure and think of this as calling a class method. The logic is the same.

Some may wonder why DECODE isn't part of C\_Orders. If it were, call on line 110 would not be necessary. But DECODE exists separately to decode the message containing our data. Therefore, it makes no sense to declare it in the C\_Orders class. Always think about separating elements from a logical point of view, not from a convenience point of view. If the DECODE function were declared in the C\_Orders class, then if we wanted to use a procedure with a similar name in the future, but targeting a different data structure, we would have to create a completely different name to avoid conflicts. Even worse, over time, as we change the code, it always becomes difficult to understand what's what. So, logically separating responsibilities prevents future naming conflicts and reduces potential errors.

### Communicating with the Server

This part is often confusing for new programmers. If you find this section confusing, it means you haven't fully understood everything you actually need to do. Unlike a trader looking at a chart, a programmer must view the process as filling out a form correctly. If the form has errors, the server (like a meticulous employer) will reject it.

The server does not understand mistakes. It simply refuses improper requests. However, it will tell us the reason for the refusal. A well-structured communication procedure ensures smooth interaction.

Lines 44–64 contain the function handling server communication. Despite its simplicity, it includes all necessary steps. Pay attention to the logical sequence of steps to be followed. We can't just do things the same way as before. The following order must be observed: first, we clear the response memory. This is done on lines 49–50. Next, we validate the request. I.e. we check data before submitting. This check is performed on line 51. If there is an error, a relevant error code will be returned for us to fix the error and retry. If everything is correct, the request is sent to the server (line 56).

The server's response is stored in a placeholder variable to avoid compiler warnings. The reason is that we are not interested in the response of the function itself. The real concern is the content of the response structure. Line 57 checks whether the result differs from **TRADE\_RETCODE\_DONE**. If so, an error occurred. The relevant error code is printed in the terminal. This is done on line 59.

If an error occurs, the function on line 44 returns zero. Otherwise, it returns the server's response ticket, identifying the order or position. We'll talk about this later when we start manipulating these positions/orders. For now, we are only sending a market order request.

Such a request is done in another function which appears on line 66. The function takes a single argument indicating buy or sell. It is called at lines 113 and 116 when a Chart Trade message instructs a market action. But how does the server know the stop loss, take profit, leverage, or instrument details? Or whether we are using cross orders? Well. That is where the next step comes in.

### Filling Out the Request Form

If you looked at the server communication function on line 44, you'll notice that on lines 51 and 56, we use a structure that is not declared in that function. This structure is declared in C\_Orders at line 14. This private, global-to-the-class structure. That is, the entire body of the class can access this variable, but no code outside the class has access to it.

So, this variable, used in the function on line 44, must be populated correctly. This population is performed on lines 66–94. Correct structure filling ensures the server executes the desired request: either market buy or sell. Pending orders will be covered later, along with modifying stop loss and take profit levels.

To populate the structure, we need several pieces of data. Some come from Chart Trade, others from MetaTrader 5. Anyway, there is a certain logical sequence that should be followed. First, get the current price (line 68). Pay attention to every detail in this line, because each of them is very important.

We also need to know what the trading volume will be. Many people experience difficulties at this stage, since the volume that the server expects is a multiple of another number. The volume shown on the trading chart is not the trading volume, but the leverage level. These two concepts are different, but related. Leverage multiplies the minimum volume. Therefore, do not confuse the concepts. The server also requires the number of decimal places for the instrument. This information can be found on line 70.

And now we have the most important thing. We can start populating the structure. First, we need to clear the structure, which is done on line 72. Next, each of the following lines specifies what exactly the server should do. This request filling shown on lines 73–86, where each field is populated appropriately, works for stocks, OTC markets, and Forex. It is important to know that each of these fields has its own meaning. Misunderstanding any field can lead to financial loss. A detailed explanation will be provided in a future article on pending orders.

Finally, the filled structure is printed to the terminal where we can check it. This is shown on lines 88–91. While there are different ways to do this, we call ArrayPrint from MQL5 Standard Library for simplicity. The function then returns the request response (line 93).

### Final Thoughts

Although this article covers only a portion of the code — specifically, sending market orders — this header file, together with the Chart Trade indicator, can already execute market buys and sells. Some questions may remain, especially regarding the MqlTradeRequest structure. Don't worry; pending orders will clarify all of this. In the next article, we will continue exploring the the Expert Advisor source code.

| File | Description |
| --- | --- |
| Experts\\Expert Advisor.mq5 | Demonstrates the interaction between Chart Trade and the Expert Advisor (Mouse Study is required for interaction) |
| Indicators\\Chart Trade.mq5 | Creates the window for configuring the order to be sent (Mouse Study is required for interaction) |
| Indicators\\Market Replay.mq5 | Creates controls for interacting with the replay/simulator service (Mouse Study is required for interaction) |
| Indicators\\Mouse Study.mq5 | Enables interaction between graphical controls and the user (Required for operating both the replay simulator and live market trading) |
| Services\\Market Replay.mq5 | Creates and maintains the market replay and simulation service (Main file of the entire system) |

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12589](https://www.mql5.com/pt/articles/12589)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12589.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12589/anexo.zip "Download Anexo.zip")(490.53 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

**[Go to discussion](https://www.mql5.com/en/forum/498295)**

![Mastering Quick Trades: Overcoming Execution Paralysis](https://c.mql5.com/2/176/19576-mastering-quick-trades-overcoming-logo.png)[Mastering Quick Trades: Overcoming Execution Paralysis](https://www.mql5.com/en/articles/19576)

The UT BOT ATR Trailing Indicator is a personal and customizable indicator that is very effective for traders who like to make quick decisions and make money from differences in price referred to as short-term trading (scalpers) and also proves to be vital and very effective for long-term traders (positional traders).

![Introduction to MQL5 (Part 25): Building an EA that Trades with Chart Objects (II)](https://c.mql5.com/2/176/19968-introduction-to-mql5-part-25-logo__1.png)[Introduction to MQL5 (Part 25): Building an EA that Trades with Chart Objects (II)](https://www.mql5.com/en/articles/19968)

This article explains how to build an Expert Advisor (EA) that interacts with chart objects, particularly trend lines, to identify and trade breakout and reversal opportunities. You will learn how the EA confirms valid signals, manages trade frequency, and maintains consistency with user-selected strategies.

![Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (Final Part)](https://c.mql5.com/2/109/Neural_Networks_in_Trading_Multimodal_Agent_Augmented_with_Instruments____LOGO.png)[Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (Final Part)](https://www.mql5.com/en/articles/16867)

We continue to develop the algorithms for FinAgent, a multimodal financial trading agent designed to analyze multimodal market dynamics data and historical trading patterns.

![Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (FinAgent)](https://c.mql5.com/2/108/Neural_Networks_in_Trading_Multimodal_Agent_Augmented_with_Instruments____LOGO.png)[Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (FinAgent)](https://www.mql5.com/en/articles/16850)

We invite you to explore FinAgent, a multimodal financial trading agent framework designed to analyze various types of data reflecting market dynamics and historical trading patterns.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/12589&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069484014466696664)

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