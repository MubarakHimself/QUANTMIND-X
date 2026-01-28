---
title: Developing a Replay System (Part 32): Order System (I)
url: https://www.mql5.com/en/articles/11393
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:17:40.753368
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11393&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070170698132951330)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 31): Expert Advisor project — C\_Mouse class (V)](https://www.mql5.com/en/articles/11378), we have developed the basic part for using the mouse in the replay/simulation system. We didn't show the problem with the mouse scroll wheel because I didn't initially see the need to use this feature. Now we can start working on the other part, which is undoubtedly much more difficult. What we have to implement, both in code and in other related things, is without a doubt the most difficult thing in the entire replay/modeling system. Without this part it is impossible to perform any practical and simple analysis. We are talking about the order system.

Of all the things that we have developed so far, this system, as you will probably notice and eventually agree, is the most complex. Now we need to do something very simple: make our system simulate the operation of a trading server. This need to accurately implement the way the trading server operates seems like a no-brainer. At least in words. But we need to do this so that the everything is seamless and transparent for the user of the replay/simulation system. When using the system, the user cannot distinguish between the real system and the simulated one. But that's not the hardest part. The hardest part is to allow the EA to be the same in every situation.

To say that the EA must be the same means not to compile it for use in a tester and then compile it again for use in the real market. This would be much simpler and easier from a programming point of view, but would create problems for the user, requiring the constant recompilation of the EA. So, to ensure the best user experience, we need to create an EA that will require only one compilation. Once this is done, it can be used both in the real market and in the tester.

The critical part of using the EA in the real market, even on a demo account, is the easiest to implement. So, we start there. Please note that we will start with a very basic model. We will gradually increase the complexity of the EA and enhance its capabilities to eventually create the EA that can be used both for replay/simulation and on a demo or live account. The first thing we will do is borrow most of the code previously explained in other articles published here in the community. These articles belong to me, so I see no problem in using this information. We will make some changes to make the system flexible, modular, reliable and robust. Otherwise, we may at some point find ourselves in a dead end, which will make further development of the system in all respects impossible.

This is just the beginning, as the task is very complex. First of all, we need to know how the class inheritance system is currently implemented in the EA. We have previously seen this in other articles within this series. The current inheritance diagram is shown in Figure 01:

![Figure 01](https://c.mql5.com/2/48/001__11.png)

Figure 01: Current EA class inheritance diagram

Although this diagram works very well for what has been done so far, it is still far from what we really need. This is not due to the fact that it is difficult to add the C\_Orders class to this inheritance scheme. Actually, this can be achieved by deriving the C\_Orders class from the C\_Study class. But I don't want to do this. The reason lies in a very practical issue that is sometimes ignored by most programmers working with OOP (Object Oriented Programming). This issue is known as encapsulation. That is, know only what you need to do your job. When we create a hierarchy of classes, we shouldn't let some classes know more than they really need to know. We should always favor the kind of programming where each class knows only what it really needs to know to perform its task. Therefore, maintaining encapsulation while adding the C\_Orders class to the diagram shown in Figure 01 is practically impossible. Therefore, the best solution to this problem is to remove the C\_Terminal class from the inheritance block as shown in Figure 01 and pass it into the same block as an argument or parameter, which can be used in a much more appropriate way. So the control over who receives what information will be exercised by the EA code, and this will assist in maintaining information encapsulation.

Thus, the new diagram of classes that we will use in this article is shown in Figure 02.

![Figure 02](https://c.mql5.com/2/48/002__5.png)

Figure 02: New inheritance diagram

In the new diagram, individual classes will only become accessible if the EA code allows it. As you may have guessed, we will have to make small changes to the existing code. But these changes will not affect the entire system much. We can quickly consider these changes just to see what's new.

### Preparing the ground

The first thing to do is to create an enumeration in the C\_Terminal class. Here it is:

```
class C_Terminal
{

       protected:
      enum eErrUser {ERR_Unknown, ERR_PointerInvalid};

// ... Internal code ...

};
```

This enumeration will allow us to use the \_LastError variable to notify us if an error occurs in the system for some reason. Now we will define only these two types of errors.

At this point, we will change the C\_Mouse class. I will not go into details, since the changes do not affect the functioning of the class. They will simply direct the flow of messages slightly differently than when using an inheritance system. The changes are shown below:

```
#define def_AcessTerminal (*Terminal)
#define def_InfoTerminal def_AcessTerminal.GetInfoTerminal()
//+------------------------------------------------------------------+
class C_Mouse : public C_Terminal
{
   protected:
//+------------------------------------------------------------------+
// ... Internal fragment ....
//+------------------------------------------------------------------+
   private :
//+------------------------------------------------------------------+
// ... Internal fragment ...
      C_Terminal *Terminal;
//+------------------------------------------------------------------+
```

To avoid repeating code at all times, we have added two new definitions. This allows extensive configuration options. Additionally, a private global variable has been added to allows correct access to the C\_Terminal class. Also, as can be seen in the above code, we will no longer be using C\_Terminal class inheritance.

Since we do not use inheritance, there are two more changes that need to be discussed. The first one is in the constructor of the C\_Mouse class:

```
C_Mouse(C_Terminal *arg, color corH, color corP, color corN)
	:C_Terminal()
   {
      Terminal = arg;
      if (CheckPointer(Terminal) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
      if (_LastError != ERR_SUCCESS) return;
      ZeroMemory(m_Info);
      m_Info.corLineH  = corH;
      m_Info.corTrendP = corP;
      m_Info.corTrendN = corN;
      m_Info.Study = eStudyNull;
      m_Mem.CrossHair = (bool)ChartGetInteger(def_InfoTerminal.ID, CHART_CROSSHAIR_TOOL);
      ChartSetInteger(def_InfoTerminal.ID, CHART_EVENT_MOUSE_MOVE, true);
      ChartSetInteger(def_InfoTerminal.ID, CHART_CROSSHAIR_TOOL, false);
      def_AcessTerminal.CreateObjectGraphics(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
   }
```

Here we remove the call of the C\_Terminal class constructor from the C\_Mouse class constructor. Now we need to get a new parameter to initialize the pointer to the class. For security reasons, as we don't want our code to break in an unsuitable situation, we'll run a test to verify that the pointer that allows us to use the C\_Terminal class was initialized correctly.

For this, we use the [CheckPointer](https://www.mql5.com/en/docs/common/checkpointer) function, but like the constructor, it does not allow error information to be returned. We will indicate the error condition using a predefined value in the enumeration that is present in the C\_Terminal class. However, since we cannot directly change the value of the **\_LastError** variable, we need to use the call [SetUserError](https://www.mql5.com/en/docs/common/setusererror). After that, we can check **\_LastError** to find out what happened.

However, we need to be careful if the C\_Terminal class has not been initialized correctly: the C\_Mouse class constructor will return without doing anything because it will not be able to use the C\_Terminal class that has not been initialized.

Another change is related to the following function:

```
virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
   {
      int w = 0;
      static double memPrice = 0;

      C_Terminal::DispatchMessage(id, lparam, dparam, sparam);
      switch (id)
      {
//....
```

The specified code must be added to the EA to process events reported by the MetaTrader 5 platform. As you can see, if this is not done, problems will arise with some events, which can lead to a violation of the position of the element on the chart. For now, we'll just remove code from this location. We can even allow the C\_Mouse class to call the C\_Terminal class messaging system. But since we're not using inheritance, this will leave the code with a rather unusual dependency.

Just like we did in the C\_Mouse class, we will do it in the C\_Study class. Pay attention to the class constructor, which can be seen below:

```
C_Study(C_Terminal *arg, color corH, color corP, color corN)
        :C_Mouse(arg, corH, corP, corN)
   {
      Terminal = arg;
      if (CheckPointer(Terminal) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
      if (_LastError != ERR_SUCCESS) return;
      ZeroMemory(m_Info);
      m_Info.Status = eCloseMarket;
      m_Info.Rate.close = iClose(def_InfoTerminal.szSymbol, PERIOD_D1, ((def_InfoTerminal.szSymbol == def_SymbolReplay) || (macroGetDate(TimeCurrent()) != macroGetDate(iTime(def_InfoTerminal.szSymbol, PERIOD_D1, 0))) ? 0 : 1));
      m_Info.corP = corP;
      m_Info.corN = corN;
      CreateObjectInfo(2, 110, def_ExpansionBtn1, clrPaleTurquoise);
      CreateObjectInfo(2, 53, def_ExpansionBtn2);
      CreateObjectInfo(58, 53, def_ExpansionBtn3);
   }
```

We take a parameter pointing to the C\_Terminal class pointer and pass it to the C\_Mouse class. Since we inherited it, we have to initialize it correctly, but either way, we'll do the same checks we did in the C\_Mouse class constructor to make sure we're using the correct pointer. Now we have to pay attention to one thing: if you notice, in both the constructors in C\_Mouse and C\_Study, we check the value of \_LastError to know if something is not as expected. However, depending on the asset being used, the C\_Terminal class may need to initialize its name in order for the EA to know which asset is currently on the chart.

If this happens by chance, the \_LastError variable will contain the value 4301 ( **ERR\_MARKET\_UNKNOWN\_SYMBOL** ), which indicates that the asset was not detected correctly. But this will not be true, since the C\_Terminal class, in its current programmed state, can access the desired asset. To avoid EA removal from the chart because of this error, you need to make a small change to the constructor of the C\_Terminal class. Here it is:

```
C_Terminal()
   {
      m_Infos.ID = ChartID();
      CurrentSymbol();
      m_Mem.Show_Descr = ChartGetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR);
      m_Mem.Show_Date  = ChartGetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE);
      ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, false);
      ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, 0, true);
      ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, false);
      m_Infos.nDigits = (int) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_DIGITS);
      m_Infos.Width   = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
      m_Infos.Height  = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
      m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
      m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
      m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
      m_Infos.AdjustToTrade = m_Infos.PointPerTick / m_Infos.ValuePerPoint;
      ResetLastError();
   }
```

By adding this code, we will indicate that there is no initial error. Thus, the constructor system will be used to initialize classes in the EA code. We will not need to actually add this line. Because in some cases we may forget to make this addition, or worse, do it at the wrong time, which will make the code completely unstable and unsafe to use.

### The C\_Orders class

What we have seen up to this point will help us set the stage for the next step. We still need to make a few more changes to the C\_Terminal class. We will make some of these changes later in this article. Let's move on to creating the C\_Orders class which will enable interaction with the trading server. In this case, it will be a real server, accessed to which is provided by a broker. But you can use a demo account to test the system. Actually, **it is not advisable** to use the system directly on a real account

The code for this class starts as follows:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "..\C_Terminal.mqh"
//+------------------------------------------------------------------+
#define def_AcessTerminal (*Terminal)
#define def_InfoTerminal def_AcessTerminal.GetInfoTerminal()
//+------------------------------------------------------------------+
class C_Orders
{
```

Here, to facilitate coding, we will define several things to access the C\_Terminal class. Now these definitions will be located not at the end of the class file but inside the class code. This will be the way to access the C\_Terminal class. Now, if we make any changes in the future, we will not have to change the class code, we will only need to change this definition. Note that the class does not inherit anything. It's important to keep this in mind so you don't get confused when programming this class and when writing other classes that will appear later.

Next, we declare the first global and internal class variables. Here it is:

```
   private :
//+------------------------------------------------------------------+
      MqlTradeRequest m_TradeRequest;
      ulong           m_MagicNumber;
      C_Terminal      *Terminal;
```

Note that these global variables are declared private, i.e. they cannot be accessed outside of the class code. Pay attention to how the variable that will provide access to the C\_Terminal class is declared. It is actually declared as a pointer, although the use of pointers in MQL5 is different from that in C/C++.

```
ulong ToServer(void)
   {
      MqlTradeCheckResult     TradeCheck;
      MqlTradeResult          TradeResult;
      bool bTmp;

      ResetLastError();
      ZeroMemory(TradeCheck);
      ZeroMemory(TradeResult);
      bTmp = OrderCheck(m_TradeRequest, TradeCheck);
      if (_LastError == ERR_SUCCESS) bTmp = OrderSend(m_TradeRequest, TradeResult);
      if (_LastError != ERR_SUCCESS) PrintFormat("Order System - Error Number: %d", _LastError);
      return (_LastError == ERR_SUCCESS ? TradeResult.order : 0);
   }
```

The above function, which will be private, serves to "centralize" calls. I decided to centralize calls because in the future it will be easier to adapt the system. This is necessary to be able to use the same diagram both with a real server and with a simulated one. The previous function has been removed, along with others that we discussed in the article: [Creating an EA that works automatically (Part 15): Automation (VII)](https://www.mql5.com/en/articles/11438). That article explained how to create an automated EA from a manual one. We will use a few functions from this article to expedite a littles our work here. This way, if we want to use the same concepts, we can test an automated EA using the replay/simulation system, without having to use the MetaTrader 5 strategy tester.

Basically, the above function will check several things on the broker's server. If everything is ok, it will send a request to the trading server to fill the user's or the EA's (since we can work in an automated mode) request.

```
inline void CommonData(const ENUM_ORDER_TYPE type, const double Price, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
   {
      double Desloc;

      ZeroMemory(m_TradeRequest);
      m_TradeRequest.magic        = m_MagicNumber;
      m_TradeRequest.symbol       = def_InfoTerminal.szSymbol;
      m_TradeRequest.volume       = NormalizeDouble(def_InfoTerminal.VolumeMinimal + (def_InfoTerminal.VolumeMinimal * (Leverage - 1)), def_InfoTerminal.nDigits);
      m_TradeRequest.price        = NormalizeDouble(Price, def_InfoTerminal.nDigits);
      Desloc = def_AcessTerminal.FinanceToPoints(FinanceStop, Leverage);
      m_TradeRequest.sl           = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? -1 : 1)), def_InfoTerminal.nDigits);
      Desloc = def_AcessTerminal.FinanceToPoints(FinanceTake, Leverage);
      m_TradeRequest.tp           = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? 1 : -1)), def_InfoTerminal.nDigits);
      m_TradeRequest.type_time    = (IsDayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
      m_TradeRequest.stoplimit    = 0;
      m_TradeRequest.expiration   = 0;
      m_TradeRequest.type_filling = ORDER_FILLING_RETURN;
      m_TradeRequest.deviation    = 1000;
      m_TradeRequest.comment      = "Order Generated by Experts Advisor.";
   }
```

The above function has also been imported from the same series of articles. However, here we had to adapt it to the new system being implemented. The operating principle is basically the same as in the automation series. But for those who haven't read that series, let's take a quick look at this function. First, it has a code that converts a financial values into points. This is done so that we, as users, do not have to worry about setting the number of points for a given leverage. Thus, we won't have to deal with finances more than we want. Performing this operation manually is can lead to errors and failures, but using this function, the values are converted quite easily. The function works regardless of the asset. Whatever the asset type you use, conversion will always be done correctly and efficiently.

Let's now look at the next function which is in the C\_Terminal class. Its code can be seen below:

```
inline double FinanceToPoints(const double Finance, const uint Leverage)
   {
      double volume = m_Infos.VolumeMinimal + (m_Infos.VolumeMinimal * (Leverage - 1));

      return AdjustPrice(MathAbs(((Finance / volume) / m_Infos.AdjustToTrade)));
   };
```

The main secret of this function is in the value that is calculated as shown in the fragment below:

```
m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
m_Infos.AdjustToTrade = m_Infos.ValuePerPoint / m_Infos.PointPerTick;
```

All of the above values used in FinanceToPoints depend on the asset that we manage and use for trading. So when FinanceToPoints does the conversion, it will actually adapt to the asset that we are using on the chart. Therefore, the EA does not care on what asset and on what market it was launched. Similarly, it can work with any user. Now that we've seen the private part of the class, let's look at the public part. We'll start with the constructor:

```
C_Orders(C_Terminal *arg, const ulong magic)
         :m_MagicNumber(magic)
   {
      if (CheckPointer(Terminal = arg) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
   }
```

In a simple and effective way, we will make sure that the class constructor has access to the C\_Terminal class. Notice how this actually happens: when the EA creates a C\_Terminal object for the class to use, it also creates an object that will be passed to all other classes that need that object. This happens as follows: the class receives a pointer created by the EA in order to have access to the already initialized class. We then save this value in our private global variable so that we can access any data or functions of the C\_Terminal class when needed. In fact, if such an object, in this case a class, does not point to something useful, it will be reported as an error. Since the constructor cannot return a value, we use this method which will set appropriate value for the \_LastError variable. It will allow us to see the reason.

Now let's proceed to the last two functions present in the class at this stage of development. The first one is shown below:

```
ulong ToMarket(const ENUM_ORDER_TYPE type, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
   {
      CommonData(type, SymbolInfoDouble(def_InfoTerminal.szSymbol, (type == ORDER_TYPE_BUY ? SYMBOL_ASK : SYMBOL_BID)), FinanceStop, FinanceTake, Leverage, IsDayTrade);
      m_TradeRequest.action   = TRADE_ACTION_DEAL;
      m_TradeRequest.type     = type;

      return (((type == ORDER_TYPE_BUY) || (type == ORDER_TYPE_SELL)) ? ToServer() : 0);
   };
```

This function is responsible for sending a request for execution at the market price. Here we use practically the entire code that we considered earlier. This is a good case of reuse. Such reuse promotes greater security and performance over time. Improving any part of the reused system improves the entire code. Pay attention to the following details regarding the code above:

- First, we will indicate stop levels (take profit and stop loss) as financial values, rather than points.
- Second, we will tell the server to execute the order immediately, at the best possible price available at the time the order is executed.
- Third, although we have access to more order types, here we can only use these two types, indicating whether we want to buy or sell. Without this indication the order will not be sent.

These details are important. We should take care of them, otherwise we won't be able to use this system. If you do not know or ignore these points, you will have a lot of headaches and doubts at the next stage of development.

Here comes the last functions of the C\_Orders class. Below we show the current stage of development:

```
ulong CreateOrder(const ENUM_ORDER_TYPE type, const double Price, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
   {
      double  bid, ask;

      bid = SymbolInfoDouble(def_InfoTerminal.szSymbol, (def_InfoTerminal.ChartMode == SYMBOL_CHART_MODE_LAST ? SYMBOL_LAST : SYMBOL_BID));
      ask = (def_InfoTerminal.ChartMode == SYMBOL_CHART_MODE_LAST ? bid : SymbolInfoDouble(def_InfoTerminal.szSymbol, SYMBOL_ASK));
      CommonData(type, def_AcessTerminal.AdjustPrice(Price), FinanceStop, FinanceTake, Leverage, IsDayTrade);
      m_TradeRequest.action   = TRADE_ACTION_PENDING;
      m_TradeRequest.type     = (type == ORDER_TYPE_BUY ? (ask >= Price ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_BUY_STOP) :
							  (bid < Price ? ORDER_TYPE_SELL_LIMIT : ORDER_TYPE_SELL_STOP));

      return (((type == ORDER_TYPE_BUY) || (type == ORDER_TYPE_SELL)) ? ToServer() : 0);
   };
```

Here are some things that are very similar to the market execution function. Such as the fact that the stop levels are set in financial values and that we must indicate whether we will buy or sell using only one of these two values. However, there is something different form the vast majority of codes present in the EA. Typically, when we create an Expert Advisor, it is targeted for use on a very specific type of market, be it the Forex market or the stock exchange. Since MetaTrader 5 supports both types of markets, we need to do some standardization to make our life easier. Wouldn't it be the same thing to work on Forex and the Stock Exchange? From the user's point of view **YES**, but from a programming point of view **NO**. If you look closely, you can see that we are checking what charting type is currently being used. Based on this, we cab conclude if the system works with the Last price or with Bid and Ask prices. Knowing this is important not in order to place an order, but in order to know what type of order to use. Later we will need to implement such orders into the system to simulate the operation of a trading server. But all we need to know at this stage is that the type of order is as important as the price at which it will be executed. If we put the price in the right place but get the order type wrong, then there will be a problem because the order will be executed at a different time than what you expected it would be executed by the server.

Very often, the novice users of MetaTrader 5 make mistakes when filling out pending orders. Not in any kind of market, since over time, users get used to the market and do not go wrong so easily. However, when we move from one market to another, things get more complicated. If the charting system is based on BID-ASK, the method for setting the order type is different from the LAST-based charting system. The differences are subtle, but they are there, and lead to the fact that the order does not remain pending, but is executed at the market price.

### Conclusion

Despite the material discussed, there will be no code attached to this article, and the reason is that we are not implementing an order system, but simply creating a BASIC class to implement such a system. You may have noticed that, compared to the C\_Orders class code shown here, several functions and methods are missing. I mean, compared to codes considered in previous articles, where we discussed the order system.

The fact that this is happening is due to my decision to split this order system into several parts, some larger and some smaller. This will help me to clearly and simply explain how the system will be integrated with the replay/simulation service. Believe me, this is not the easiest task, on the contrary, it is quite complex and includes many concepts that may not be yet familiar to you. Therefore, I have to gradually present the explanation so that the articles are understandable and their content does not turn into complete chaos.

In the next article we will look at how to get this order system to start interacting with the trading server. At least on a physical level so that we can use the EA on a demo or real account. There we will begin to understand how order types work so that we can start with the simulated system. If we do the opposite or put the simulated and real systems together, the result will be total confusion. See you in the next article!

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11393](https://www.mql5.com/pt/articles/11393)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11393.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11393/anexo.zip "Download Anexo.zip")(130.63 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/463880)**

![Quantization in machine learning (Part 2): Data preprocessing, table selection, training CatBoost models](https://c.mql5.com/2/59/Quantization_in_Machine_Learning_Logo_2___Logo.png)[Quantization in machine learning (Part 2): Data preprocessing, table selection, training CatBoost models](https://www.mql5.com/en/articles/13648)

The article considers the practical application of quantization in the construction of tree models. The methods for selecting quantum tables and data preprocessing are considered. No complex mathematical equations are used.

![Neural networks made easy (Part 62): Using Decision Transformer in hierarchical models](https://c.mql5.com/2/59/Neural_networks_are_easy_0Part_62s_logo.png)[Neural networks made easy (Part 62): Using Decision Transformer in hierarchical models](https://www.mql5.com/en/articles/13674)

In recent articles, we have seen several options for using the Decision Transformer method. The method allows analyzing not only the current state, but also the trajectory of previous states and actions performed in them. In this article, we will focus on using this method in hierarchical models.

![Introduction to MQL5 (Part 5): A Beginner's Guide to Array Functions in MQL5](https://c.mql5.com/2/73/Introduction_to_MQL5_Part_5___LOGO.png)[Introduction to MQL5 (Part 5): A Beginner's Guide to Array Functions in MQL5](https://www.mql5.com/en/articles/14306)

Explore the world of MQL5 arrays in Part 5, designed for absolute beginners. Simplifying complex coding concepts, this article focuses on clarity and inclusivity. Join our community of learners, where questions are embraced, and knowledge is shared!

![Developing a Replay System (Part 31): Expert Advisor project — C_Mouse class (V)](https://c.mql5.com/2/59/sistema_de_Replay_logo.png)[Developing a Replay System (Part 31): Expert Advisor project — C\_Mouse class (V)](https://www.mql5.com/en/articles/11378)

We need a timer that can show how much time is left till the end of the replay/simulation run. This may seem at first glance to be a simple and quick solution. Many simply try to adapt and use the same system that the trading server uses. But there's one thing that many people don't consider when thinking about this solution: with replay, and even m ore with simulation, the clock works differently. All this complicates the creation of such a system.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/11393&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070170698132951330)

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