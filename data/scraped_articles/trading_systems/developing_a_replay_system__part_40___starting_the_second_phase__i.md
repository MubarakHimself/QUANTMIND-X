---
title: Developing a Replay System (Part 40): Starting the second phase (I)
url: https://www.mql5.com/en/articles/11624
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:13:48.208153
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/11624&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070116770523582522)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 39): Paving the Path (III)](https://www.mql5.com/en/articles/11599), we looked at how we could organize communication between processes to enable certain actions. At the moment we are using an EA and an indicator, but as necessary we will be able to expand these tools.

The main advantage of this type of communication is that we can build our system in modules. You may not yet understand what we can really do. Ultimately, we will be able to exchange information between processes over a more "secure" channel than when using global terminal variables.

To implement this and show how to integrate our replay/simulator system into a modular system, let's take a step back and then a step forward. In this article, we will remove the study system that uses the EA mouse. We will convert the same system into an indicator so that it is compatible with the next step we will implement.

If we succeed, it will become clear that after this we can do much more. There is a high chance that the indicator will not end here, since we will need to do other things later, but eventually it will be completed. The big advantage is that we will be able to update or modify the module without affecting the rest of our main system.

### Starting the coding

Our code will be built from what we already have at the moment. We will make minimal changes so that the EA code becomes an indicator.

The first thing to do is check the code in the InterProcess.mqh header file. The full code can be seen below.

**InterProcess.mqh file code**:

```
01. #property copyright "Daniel Jose"
02. //+------------------------------------------------------------------+
03. #define def_SymbolReplay                    "RePlay"
04. #define def_GlobalVariableReplay            def_SymbolReplay + "_Infos"
05. #define def_GlobalVariableIdGraphics        def_SymbolReplay + "_ID"
06. #define def_GlobalVariableServerTime        def_SymbolReplay + "_Time"
07. #define def_MaxPosSlider                    400
08. //+------------------------------------------------------------------+
09. union u_Interprocess
10. {
11.     union u_0
12.     {
13.             double  df_Value;  // Value of the terminal global variable...
14.             ulong   IdGraphic; // Contains the Graph ID of the asset...
15.     }u_Value;
16.     struct st_0
17.     {
18.             bool    isPlay;     // Indicates whether we are in Play or Pause mode...
19.             bool    isWait;     // Tells the user to wait...
20.             bool    isHedging;  // If true we are in a Hedging account, if false the account is Netting...
21.             bool    isSync;     // If true indicates that the service is synchronized...
22.             ushort  iPosShift;  // Value between 0 and 400...
23.     }s_Infos;
24.     datetime        ServerTime;
25. };
26. //+------------------------------------------------------------------+
27. union uCast_Double
28. {
29.     double   dValue;
30.     long     _long;                    // 1 Information
31.     datetime _datetime;                // 1 Information
32.     int      _int[sizeof(double)];     // 2 Informations
33.     char     _char[sizeof(double)];    // 8 Informations
34. };
35. //+------------------------------------------------------------------+
```

What we're really interested in in this code is between lines 27 and 34. Here we have a reference that enables the transfer of information between processes. This connection now has some kinds of data that we will need and use from the very beginning. The idea is to facilitate data transfer that we already discussed in the previous article, but in a practical and targeted way.

These lines have been added to the InterProcess.mqh file. Let's continue, but now we'll make other changes to the code. They must be implemented in the class file C\_Study.mqh.

First, we'll change a few things at the beginning of the class. They can be seen below:

```
class C_Study : public C_Mouse
{
        protected:
                enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
        private :
                enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
```

The crossed out line has been removed from the private part of the code and is now in the protected part. Private parts indicate that their contents cannot be accessed from outside the class, but this is not the case with enumerations: they are always treated as public, regardless of the defining part.

Below is the part where the changes actually took place:

```
//+------------------------------------------------------------------+
void Update(const eStatusMarket arg)
void Update(void)
{
   datetime dt;

   switch (m_Info.Status = (m_Info.Status != arg ? arg : m_Info.Status))
   switch (m_Info.Status)
   {
      case eCloseMarket : m_Info.szInfo = "Closed Market";
         break;
      case eInReplay    :
      case eInTrading   :
         if ((dt = GetBarTime()) < ULONG_MAX)
         {
            m_Info.szInfo = TimeToString(dt, TIME_SECONDS);
            break;
         }
      case eAuction     : m_Info.szInfo = "Auction";
         break;
      default           : m_Info.szInfo = "ERROR";
   }
   Draw();
}
//+------------------------------------------------------------------+
void Update(const MqlBookInfo &book[])
{
   m_Info.Status = (ArraySize(book) == 0 ? eCloseMarket : (def_InfoTerminal.szSymbol == def_SymbolReplay ? eInReplay : eInTrading));
   for (int c0 = 0; (c0 < ArraySize(book)) && (m_Info.Status != eAuction); c0++)
      if ((book[c0].type == BOOK_TYPE_BUY_MARKET) || (book[c0].type == BOOK_TYPE_SELL_MARKET)) m_Info.Status = eAuction;
   this.Update();
}
//+------------------------------------------------------------------+
```

All crossed out lines have been removed from the class and have been replaced with new highlighted lines.

Why does this happen? To understand this, you need to realize that turning the Expert Advisor code into an indicator will lead to a change in the method of action.

The code in question relates to the use of the mouse as a study tool.

When we write code in such a way that the EA performs all the actions, we can act in a certain way. But when we do the same thing with an indicator, we have to adapt to a different way. This is because the indicator lives in a thread, and if it is somehow affected, it will affect all the others present of the chart.

But we don't want the indicator to be blocked at some point. Because of this, we will have to approach things a little differently.

Let's look at the indicator code below. This is the full code.

**Indicator source code**:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. #property description "This is an indicator for graphical studies using the mouse."
004. #property description "This is an integral part of the Replay / Simulator system."
005. #property description "However it can be used in the real market."
006. #property version "1.40"
007. #property icon "/Images/Market Replay/Icons/Indicators.ico"
008. #property link "https://www.mql5.com/en/articles/11624"
009. #property indicator_chart_window
010. #property indicator_plots 0
011. #property indicator_buffers 1
012. //+------------------------------------------------------------------+
013. #include <Market Replay\Auxiliar\Study\C_Study.mqh>
014. #include <Market Replay\Auxiliar\InterProcess.mqh>
015. //+------------------------------------------------------------------+
016. C_Terminal *Terminal  = NULL;
017. C_Study    *Study     = NULL;
018. //+------------------------------------------------------------------+
019. input C_Study::eStatusMarket user00 = C_Study::eAuction;   //Market Status
020. input color user01 = clrBlack;                             //Price Line
021. input color user02 = clrPaleGreen;                         //Positive Study
022. input color user03 = clrLightCoral;                        //Negative Study
023. //+------------------------------------------------------------------+
024. C_Study::eStatusMarket m_Status;
025. int m_posBuff = 0;
026. double m_Buff[];
027. //+------------------------------------------------------------------+
028. int OnInit()
029. {
030.    if (!CheckPass("Indicator Mouse Study")) return INIT_FAILED;
031.
032.    Terminal = new C_Terminal();
033.    Study = new C_Study(Terminal, user01, user02, user03);
034.    if ((*Terminal).GetInfoTerminal().szSymbol != def_SymbolReplay)
035.    {
036.            MarketBookAdd((*Terminal).GetInfoTerminal().szSymbol);
037.            OnBookEvent((*Terminal).GetInfoTerminal().szSymbol);
038.            m_Status = C_Study::eCloseMarket;
039.    }else
040.            m_Status = user00;
041.    SetIndexBuffer(0, m_Buff, INDICATOR_DATA);
042.    ArrayInitialize(m_Buff, EMPTY_VALUE);
043.
044.    return INIT_SUCCEEDED;
045. }
046. //+------------------------------------------------------------------+
047. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
048. {
049.    m_posBuff = rates_total - 4;
050.    (*Study).Update(m_Status);
051.
052.    return rates_total;
053. }
054. //+------------------------------------------------------------------+
055. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
056. {
057.    (*Study).DispatchMessage(id, lparam, dparam, sparam);
058.    SetBuffer();
059.
060.    ChartRedraw();
061. }
062. //+------------------------------------------------------------------+
063. void OnBookEvent(const string &symbol)
064. {
065.    MqlBookInfo book[];
066.    C_Study::eStatusMarket loc = m_Status;
067.
068.    if (symbol != (*Terminal).GetInfoTerminal().szSymbol) return;
069.    MarketBookGet((*Terminal).GetInfoTerminal().szSymbol, book);
070.    m_Status = (ArraySize(book) == 0 ? C_Study::eCloseMarket : C_Study::eInTrading);
071.    for (int c0 = 0; (c0 < ArraySize(book)) && (m_Status != C_Study::eAuction); c0++)
072.            if ((book[c0].type == BOOK_TYPE_BUY_MARKET) || (book[c0].type == BOOK_TYPE_SELL_MARKET)) m_Status = C_Study::eAuction;
073.    if (loc != m_Status) (*Study).Update(m_Status);
074. }
075. //+------------------------------------------------------------------+
076. void OnDeinit(const int reason)
077. {
078.    if (reason != REASON_INITFAILED)
079.    {
080.            if ((*Terminal).GetInfoTerminal().szSymbol != def_SymbolReplay)
081.                    MarketBookRelease((*Terminal).GetInfoTerminal().szSymbol);
082.            delete Study;
083.            delete Terminal;
084.    }
085. }
086. //+------------------------------------------------------------------+
087. bool CheckPass(const string szShortName)
088. {
089.    IndicatorSetString(INDICATOR_SHORTNAME, szShortName + "_TMP");
090.    if (ChartWindowFind(ChartID(), szShortName) != -1)
091.    {
092.            ChartIndicatorDelete(ChartID(), 0, szShortName + "_TMP");
093.            Print("Only one instance is allowed...");
094.
095.            return false;
096.    }
097.    IndicatorSetString(INDICATOR_SHORTNAME, szShortName);
098.
099.    return true;
100. }
101. //+------------------------------------------------------------------+
102. inline void SetBuffer(void)
103. {
104.    uCast_Double Info;
105.
106.    m_posBuff = (m_posBuff < 0 ? 0 : m_posBuff);
107.    m_Buff[m_posBuff + 0] = (*Study).GetInfoMouse().Position.Price;
108.    Info._datetime = (*Study).GetInfoMouse().Position.dt;
109.    m_Buff[m_posBuff + 1] = Info.dValue;
110.    Info._int[0] = (*Study).GetInfoMouse().Position.X;
111.    Info._int[1] = (*Study).GetInfoMouse().Position.Y;
112.    m_Buff[m_posBuff + 2] = Info.dValue;
113.    Info._char[0] = ((*Study).GetInfoMouse().ExecStudy == C_Mouse::eStudyNull ? (char)(*Study).GetInfoMouse().ButtonStatus : 0);
114.    m_Buff[m_posBuff + 3] = Info.dValue;
115. }
116. //+------------------------------------------------------------------+
```

If you look at the indicator source code and compare it with the code provided in the article [Developing a Replay System (Part 39): Paving the Path (III)](https://www.mql5.com/en/articles/11599), you can see a lot of similar parts.

If you have been following this series of articles on the replay/simulator system, then you should have no problem understanding most of the codes mentioned. However, there are a few things that are worth explaining in more detail, since many people won't be able to figure out what's going on just by looking at the code. I want the motive and idea of this approach to be as clear as possible, since a proper understanding of these and other source codes will be of great importance to those who really want to understand how the system is implemented.

Between lines 2 and 8 there is information that will change over time. This information links the code to the article in which it was created, modified, or submitted. This information is provided here to guide less experienced users.

In lines 9 and 11 we tell the compiler exactly how the indicator will be projected. All indicator codes must have at least one line 9 and one line 10. **ALL**. Line 11 tells us that we will use a buffer, and that any process that wants to read something through the CopyBuffer function can do so by accessing the indicator's internal buffer.

Lines 13 and 14 are includes that tell the compiler which header files to use. Although we have already made some changes to these files, we will need to make other changes. For now, we can use the provided header files without any problems.

On lines 16 and 17 we declare two internal global indicator variables. These pointers will help us gain controlled access to classes. Typically, a global variable is not initialized by default, as the compiler does this automatically. But here I want to clarify that although these variables (pointers) do not point to any specific memory location, if you declare them explicitly, then we must remember that the compiler will by default do so implicitly.

Between lines 19 and 22 are the external configuration variables. From now on, I suggest you stop seeing such variables as simply a way for the user to access or configure the indicator. If we think like this, we will end up with a very limited vision and will not be able to imagine everything that can be accomplished with their help.

No program should be considered as being pure and simple. You should always think of a program as a function since it receives information, processes it in a certain way, and produces some output. What element in programming has the same behavior? **A FUNCTION**. So, let's start looking at things from a broader perspective. Let's stop imagining or dreaming of possibilities and start seeing things for what they really are: programs are functions, no matter what we can do. Programs are functions.

On line 19 we have a parameter that has no meaning when the user places the indicator on the chart. This parameter is not intended to be configured by the user; it exists so that another program can use this indicator that we create. How is this possible? Is it hard to understand now 🤔? Don't worry about that for now. Note that this is the first article in the second phase of creating our replay/simulator system. The following articles will show you a few things to help you understand why line 19 exists. So do not panic.

Lines 20-22, on the other hand, are parameters that actually make sense because they are used to adjust the color used by the mouse on the chart.

Now we have a question: why do we need line 24? Why do we have a global variable of the same type as in line 19? The reason is that the parameter on line 19 is not actually considered a variable, but a constant. Therefore, we need to declare a variable that will be used and set while the indicator is running. This is the reason for the declaration on line 24.

The variable declared on line 25 serves as a form of memory, but with a more noble purpose than ordinary memory. We will return to this part shortly. On line 26 we declare our buffer, which will serve as the indicator's output. **Attention:** We declared a buffer, but it doesn't work yet. If you try to do anything in it, a runtime error warning will appear, and if you try to read it, you will not be able to do so.

From this point on, we move on to the interaction of the indicator with MetaTrader 5. Remember that the indicator reacts to events reported by MetaTrader 5, regardless of how our code is structured and what work it is supposed to do. It will always react to given events. Some of these may be interaction events and others may be activity events.

Among them is the Init event, which, when fired, generates the OnInit function call. This function starts on line 28 and does not take any parameters, but we can use the interaction parameters declared between lines 19 and 22 as parameters in the OnInit function.

Let's see how the OnInit function works. The first thing we'll do is check-in. This happens on line 30 where we call the function. The execution thread is then forwarded to line 87. Let's analyze it to understand what this check-in is and why, if the OnInit function does not work, it returns an indicator initialization error.

The function on line 87 takes one argument - the short name of the indicator. This name must always be defined to ensure certain conditions to operate. The first thing we do (and this happens on line 89) is give the indicator a temporary name. As soon as the indicator receives a temporary name, we can continue this check-in stage. The next thing we do is search for our indicator on the chart. This is done on line 90. If the indicator is already present on the chart, MetaTrader 5 will indicate exactly where. If not, we end up with the value -1.

Thus, the check on line 90 tells us whether the indicator is present or not on the chart. If the indicator is present, then line 92 is executed, which will remove the temporary indicator, that is, the indicator that we are trying to place on the chart. Next, we print a message warning about this, and on line 95 we return to the caller and inform it that the registration failed.

If the test on line 90 reports that the indicator is missing, we execute line 97, which will tell MetaTrader 5 the real name of the indicator. On line 99, we return to the caller and inform it that the indicator can be successfully placed on the chart. But this does not mean that it was actually placed on the chart. This is only the information that MetaTrader 5 did not find it on the chart. Now it can be launched.

This will take us back to line 30 where we actually start to get the indicator up and running. If successful, we will move to line 32. In case of failure, MetaTrader 5 will trigger an event to call the function on line 76. But because on line 30 we inform about a failure, as soon as code on line 78 is executed, it will prevent the code between lines 80 and 83 from executing. This is important to avoid failures at the indicator output.

But let's go back to line 32, where we really start using the indicator. On line 32 we start a pointer to access the C\_Terminal class. From this time on, we will be able to use the functions of the C\_Terminal class. Immediately after this, on line 33, we run a pointer to access the C\_Study class. This class will allow us to use the mouse and generate graphical analysis. Since we need access to the C\_Terminal class, we need to perform the following sequence of actions.

On line 34 we have a new test that will determine whether we are using a replay/simulator asset or another asset. If this is a replay/simulator asset, the contents of the indicator input configuration parameter declared on line 19 will need to be placed in our global variable. This is done on line 40.

If line 34 indicates that we are working on a different type of asset, then lines 36 to 38 will be executed. These lines initialize the use of the order book, telling MetaTrader 5 that we want to receive events that occurred in the order book. In addition, on line 38 we indicate that the market is closed. This condition is temporary as we will receive guidance from the order book as to what the correct information will be.

Now comes line 41, which is very important for the indicator. Remember that declaring a buffer on lines 11 and 26 does not guarantee that it can be accessed, and that attempting to access it will generate a runtime error. Without line 41, the indicator is almost completely useless. In this article, we talked about the index, matrix and their benefits. Typically, we start with a zero value in the index field. In the array field we indicate which variable is used as a buffer. As for the third field, which tells us what the buffer is used for, we could use one of the [ENUM\_INDEXBUFFER\_TYPE](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties) enumeration values. However, since we are going to store data, we use **INDICATOR\_DATA**. But it is possible to use any other type in this specific code.

Once this is done, we move on to line 42. This may seem unnecessary at this time, but I want to keep the buffer clean. This is exactly what we are doing now. We could add something else here, but I don't see the need. We want to create some kind of cache for mouse data, as you will see in the next articles.

This was an explanation of how the indicator is initialized. Now we need to know how to react to events triggered by MetaTrader 5. To explain this part, let's start by looking at the order book event code in the OnBookEvent function. This function starts on line 63 and is called by MetaTrader 5 every time something happens in the order book.

If you look closely, you can see that its content, located between lines 70 and 72, is practically no different from the content of the C\_Study class. Why didn't we leave this procedure in the C\_Study class? Now it is difficult to explain why, but believe me, there is a reason that will become clearer later. This part is probably already familiar to you, if, of course, you followed the series of articles and studied them. This function contains additional code.

On line 66 we declare a local and temporary variable that will temporarily store the market state as it may change on lines 70 or 72. If this happens, we should call line 73 to conveniently update the chart information. Before we look at the chart update part, let's take a quick look at the other lines.

On line 68, we filter BOOK events to only those that are actually part of the asset on the chart. MetaTrader 5 does not make such distinctions, so if any asset in the market watch window triggers an order book event, MetaTrader 5 will trigger an order book event. So, we have to make the distinction right away.

After this, on line 69, we record the data contained in the order book so that we can analyze it according to our needs. Then, on line 70, we tell the indicator whether the market is closed (or open) by referring to the physical market, i.e. whether the trading server is within the allowed trading window or not. To find this out, we check whether the order book array is empty or not.

Now, on line 71, we enter a loop that will check the array point by point to see if there are positions that indicate the asset is in auction. An asset is in auction status if one of the two conditions on line 72 is true. Since we don't know where exactly the BID and ASK are in the order book, we use a loop on line 71. However, if the asset we are using has a fixed BID and ASK position, we can skip the loop by removing line 71 and simply specifying the point in the array where the BID and ASK are located. If any of them indicate that the asset is in auction, the state will be updated.

Let's get back to the update part. This information is not always transmitted. We will only call the market state update if the value stored in line 66 is different from the new state.

The reason to do this here on line 73 is that when an asset enters auction or is in a suspended state during the trading window, only the order book event will have this information and a correct view of the state change. If we wait for something else to happen, we may end up in a situation where we are not aware of what is happening.

The second point at which the state is updated is exactly the event we'll look at: the OnCalculate function.

Every time the price is updated or a new bar appears on the chart, MetaTrader 5 triggers an event. This event calls the OnCalculate function in the indicator code. Of course, we usually always want this code to be executed as quickly as possible so as not to miss events that interest us. Unlike the Order Book event, which reports a state change as soon as it occurs, the OnCalculate event only takes effect when something affects the price.

However, we can see that on line 49 we are storing and adjusting a value that is not used there. If we have a function call, we need to use the given value, but why does this happen 🤔? Again, we need the OnCalculate function to execute as quickly as possible, and there is no point in updating the buffer every time the price moves or every new bar appears. Don't forget that we are not working with the price or bars on the chart, but with the mouse.

Therefore, what is really responsible for the update of the buffer is the next event triggered by MetaTrader 5: the OnChartEvent function.

This OnChartEvent function that starts on line 55 is quite simple and straightforward. The function on line 57 contains a call to ensure that the study class handles events correctly. This in turn will do the internal work required to fully process the mouse event. You can see more details in the article ["Developing a Replay System (Part 31): Expert Advisor project — C\_Mouse class (V)"](https://www.mql5.com/en/articles/11378). Regardless of this, we will call the procedure and the execution thread will stop at line 102.

Before moving on to line 102, let's look at the event on line 76 that was mentioned earlier when explaining the initialization process. Now let's see how the same code will behave when the indicator is initialized correctly. When this happens, lines 80 to 83 will be executed. On line 80, we will check which asset the indicator is used for. The reason is that if the asset is one of those that can be monitored for order book events, we must tell MetaTrader 5 that we no longer want to receive such events; this is done on line 81. Lines 82 and 83 will simply destroy the class, ensuring that the indicator is correctly removed from the chart.

Since the code from line 102 onwards is essentially what we want and need to understand in this article, it deserves special attention in its own topic. It starts just below:

### SetBuffer: where the magic happens

If you have been following the articles in this series, you may have noticed that in the article ["Developing a Replay System (part 39): Paving the Path (III)](https://www.mql5.com/en/articles/11599)", there is an entire section dedicated solely to explaining how to buffer values to send indicator information to another process.

In this section, I explained that information must be placed at a very specific point, but above all, I emphasized that information must be under dual control. You may not have realized in that article what we can do, but the point is that we can go much further than what many have done so far.

If you look at the code between lines 102 and 115, you will see that I do everything my way. But why am I doing it this way?

To avoid the need to scroll down the page to follow the explanation, I will put the lines closer to that explanation. This will make it easier to follow the idea.

```
102. inline void SetBuffer(void)
103. {
104.    uCast_Double Info;
105.
106.    m_posBuff = (m_posBuff < 0  ? 0 : m_posBuff);
107.    m_Buff[m_posBuff + 0] = (*Study).GetInfoMouse().Position.Price;
108.    Info._datetime = (*Study).GetInfoMouse().Position.dt;
109.    m_Buff[m_posBuff + 1] = Info.dValue;
110.    Info._int[0] = (*Study).GetInfoMouse().Position.X;
111.    Info._int[1] = (*Study).GetInfoMouse().Position.Y;
112.    m_Buff[m_posBuff + 2] = Info.dValue;
113.    Info._char[0] = ((*Study).GetInfoMouse().ExecStudy == C_Mouse::eStudyNull ? (char)(*Study).GetInfoMouse().ButtonStatus : 0);
114.    m_Buff[m_posBuff + 3] = Info.dValue;
115. }
```

On line 49, we indicate that we are going to put four double values into the buffer. But where exactly in the buffer? This was discussed in the article mentioned above. We will use the **rates\_total** position exactly four positions back from this point. If we are in the replay/simulator mode, we can start from position zero because we are the ones who start the process. Initialization was done on line 25.

To make the required conversions easier, we use line 104 to declare a local variable. Now notice the required check on line 106. It prevents abnormal situations which may occur mainly due to line 49, which belongs to the OnCalculate function. In the case of a real asset, we are unlikely to see a negative value in **m\_posBuff**. But in a replay or simulation asset, we can get a negative value. Line 106 fixes this issue by making m\_posBuff point to the correct position.

If m\_posBuff were to point to an index less than zero, the indicator would break. Please note that on line 107 we start updating the indicator data. This point is the simplest. Now comes the most difficult part, presented in lines 109, 112 and 114. In these lines, we buffer other values, always starting from the initially calculated position.

Can we publish these same values in other locations? **NO**. Can we arrange them in a different order? **YES**. But if you change the order, then you will have to resolve, or rather correct, future codes in order to have the correct information. If the order changes, the values will be calculated differently, and adjustments will have to be made here or in any other code created based on this indicator. This also concerns any other code we want to build.

Therefore, it would be nice to start developing some kind of methodology. Otherwise, everything will become very confusing, and we will not be able to take full advantage of MetaTrader 5. So, the methodology used here is:

- First of all, in the ZERO position, we will save the price where the mouse line is located.
- In the FIRST position, we will save the time value at which the mouse pointer is located.
- In the SECOND position, we will store values regarding the screen position, that is, in X and Y coordinates. First the X value and then Y.
- In the THIRD position we will store additional elements. The current state of the mouse buttons. It must be placed in the least significant byte (LSB). That is, in the zero byte.

The methodology used here will be applied to this indicator until the end of its life unless it is changed or updated in any way.

### Conclusion

There are a few other issues we need to resolve. So, I want you to follow the articles so that you understand what is written in the code, because often the code in the attachment may be different from what is explained in the article. There is a simple reason for this: the attached code is stable and works perfectly, while the code that will be provided in the article may undergo some changes. You will need to stay up to date with the contents of the article in order to understand the code once it becomes stable.

However, someone can find it difficult to compile complex code as it involves multiple applications. But don't worry. From time to time, I will attach the compiled code so that everyone can follow the development of the system. In addition, the already compiled code that will be presented in some articles will help you understand what applications will look like once everything is compiled perfectly. However, if you already have a little more experience and want to improve your programming skills, you can add what is shown in the articles. This way you will learn how to change the system so that you can use whatever you want in the future. But if you decide to do this, I advise you to proceed with caution and remember to always test what you add to your code.

Either way, it's not over yet. In the next article, we will continue this topic and will work on some unresolved issues.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11624](https://www.mql5.com/pt/articles/11624)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11624.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11624/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/469451)**

![Developing an MQL5 RL agent with RestAPI integration (Part 4): Organizing functions in classes in MQL5](https://c.mql5.com/2/64/RestAPIs_em_MQL5_Logo.png)[Developing an MQL5 RL agent with RestAPI integration (Part 4): Organizing functions in classes in MQL5](https://www.mql5.com/en/articles/13863)

This article discusses the transition from procedural coding to object-oriented programming (OOP) in MQL5 with an emphasis on integration with the REST API. Today we will discuss how to organize HTTP request functions (GET and POST) into classes. We will take a closer look at code refactoring and show how to replace isolated functions with class methods. The article contains practical examples and tests.

![MetaTrader 4 on macOS](https://c.mql5.com/2/12/1045_13.png)[MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)

We provide a special installer for the MetaTrader 4 trading platform on macOS. It is a full-fledged wizard that allows you to install the application natively. The installer performs all the required steps: it identifies your system, downloads and installs the latest Wine version, configures it, and then installs MetaTrader within it. All steps are completed in the automated mode, and you can start using the platform immediately after installation.

![Data Science and Machine Learning (Part 25): Forex Timeseries Forecasting Using a Recurrent Neural Network (RNN)](https://c.mql5.com/2/82/Data_Science_and_ML_Part_25__LOGO.png)[Data Science and Machine Learning (Part 25): Forex Timeseries Forecasting Using a Recurrent Neural Network (RNN)](https://www.mql5.com/en/articles/15114)

Recurrent neural networks (RNNs) excel at leveraging past information to predict future events. Their remarkable predictive capabilities have been applied across various domains with great success. In this article, we will deploy RNN models to predict trends in the forex market, demonstrating their potential to enhance forecasting accuracy in forex trading.

![Propensity score in causal inference](https://c.mql5.com/2/72/Propensity_score_in_causal_inference____LOGO.png)[Propensity score in causal inference](https://www.mql5.com/en/articles/14360)

The article examines the topic of matching in causal inference. Matching is used to compare similar observations in a data set. This is necessary to correctly determine causal effects and get rid of bias. The author explains how this helps in building trading systems based on machine learning, which become more stable on new data they were not trained on. The propensity score plays a central role and is widely used in causal inference.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/11624&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070116770523582522)

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