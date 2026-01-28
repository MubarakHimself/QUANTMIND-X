---
title: Developing a Replay System (Part 31): Expert Advisor project — C_Mouse class (V)
url: https://www.mql5.com/en/articles/11378
categories: Trading, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T18:01:57.271296
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/11378&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049574543242145251)

MetaTrader 5 / Tester


### Introduction

In the previous article [Developing a Replay System (Part 28): Expert Advisor project — C\_Mouse class (IV)](https://www.mql5.com/en/articles/11372), we looked at how you can change, add or adapt to your style the class system in order to test a new resource or model. This way, we avoid dependencies in the main code, keeping it always reliable, stable and robust, since any new resources are pushed into the main system only after the model you create is completely suitable. The main challenge in developing the replay/simulation system, and perhaps what makes the job so difficult, is creating mechanisms that are as close as possible, if not identical, to the system we will use when we trade on a real account. There is no point in creating a system to create or run replay/simulation if the same resources will not be available when the real account is used.

Looking at the C\_Mouse class system and the analytical classes shown in previous articles, you can notice that when used in a live market, be it a demo or real account, the timer will always tell you when the next bar will start. But when using a replication/simulation system, we don't count on that. Here we get a message. At first glance, it may seem that such a violation of symmetry is not particularly significant. But if we allow useless things to accumulate without correcting or removing them, we will end up with a pile of completely useless junk that will only hinder the solution of issues we really need to solve. We need a timer that can show how much time is left till the end of the replay/simulation run. This may seem at first glance to be a simple and quick solution. Many simply try to adapt and use the same system that the trading server uses. But there's one thing that many people don't consider when thinking about this solution: with replay, and even m ore with simulation, the clock works differently. This happens for several reasons:

- Replay always refers to the past. Thus, a clock on a platform or computer is by no means a sufficient indication of time.
- During replay/simulation, we can fast forward, pause, or regress time. The latter case is no longer possible, and this has happened a long time ago for various reasons, which we talked about in previous articles. We can still fast forward and pause. So the timing used on the trading server is no longer appropriate.

To give you an idea of what we're actually dealing with and how complex setting a timer in a replay/simulation system can be, let's look at Figure 01.

![Figure 01](https://c.mql5.com/2/48/001__10.png)

Figure 01 - Timer on the real market

This Figure 01 shows how the timer works to indicate when a new bar will appear on the chart. This is a very short diagram. From time to time we have the **OnTime** event. It triggers the **Update** event which will update the timer value. This way we can present how much time is left before the new bar appears. However, for the **Update** function to know what value will be presented, it requests from the **GetBarTime** function how long it will take for the bar to appear. **GetBarTime** uses the **TimeCurrent** which does not execute on the server but fixes local time on it. Thus, we can find out how much time has passed since the server was triggered on the last bar, and based on this value, calculate how much time is left until the new bar is triggered. This is the simplest part of this whole story, since we don't have to worry about whether the system has paused or whether a certain amount of time has passed. This will happen if we use an asset whose data comes directly from the trading server. When we work with the replay/simulation system, things get much more complicated. The big problem is not that we are working with the replay/simulation mode. The problem is to come up with a way to get around the **TimeCurrent** call in replay/simulation, since it is at this moment that the whole problem arises. But this must be done with minimal changes. We just want to bypass the **TimeCurrent** calling system. However, when working with the server, we want the system to work as shown in Figure 01.

Fortunately, there is one way, using MetaTrader 5, that allows you to do this with minimal hassle and significantly fewer modifications or additions to the already implemented code. This is exactly what we will talk about in this article.

### Planning

Planning of how to do this is perhaps the easiest part of the story. We will simply send the system the time value calculated by the replay/simulation service. This is the easy part. We will just use a global terminal variable for this. Earlier in the series of articles we have seen at how to use these variables. In one of the articles, we introduced a control to inform the service what the user wants to do. Many people think that using these variables you can only transfer double data. But they forget about one fact: binaries are just binaries, that is, they in no case represent any information other than zeros and ones. So we can pass any information through these global terminal variables. Assuming, of course, that we can arrange the bits in a logical manner so that we can reconstruct the information later. Fortunately, the DateTime type is actually a ulong value. We use 64 bits, which means that a DateTime value requires 8 bytes of information where we will have a representation of the complete date and time, including year, month, day, hour, minute and seconds. That's all we need to bypass the **TimeCurrent** call, since it uses and returns the same value in 8 bytes. Since the Double type uses exactly 64 bits to transfer information within the platform, it will be our solution.

But it's not that simple. We have some problems that will become clearer as we explain how this system is implemented. Although the service is quite simple and easy to build, it does have a slight complication related to the required modification aimed at providing the class with information that can be represented in the timer. At the end of the construction we will get the result shown in Figure 02:

![Figure 02](https://c.mql5.com/2/48/002__4.png)

Figure 02 - Generic timer.

This generic timer will be able to react accordingly and will be completely transparent so that the user will have a replay/simulation experience very close to what they would have on a demo/live account. This is the purpose of creating such a system: so that the experience is the same, and the user knows exactly how to act, without having to relearn how to use the system from scratch.

### Implementation

This part is the most interesting in this system. At this point, we'll look at how we can bypass the TimeCurrent call. The user should not know whether he is using replay or is interacting with the server. To start implementing the system, we need to add (as should be clear from previous topics) a new global terminal variable. At the same time, we need the right means to pass DateTime information through the Double variable. For this we will use the Interprocess.mqh header file. We also add the following things:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_SymbolReplay                "RePlay"
#define def_GlobalVariableReplay        def_SymbolReplay + " Infos"
#define def_GlobalVariableIdGraphics    def_SymbolReplay + " ID"
#define def_GlobalVariableServerTime    def_SymbolReplay + " Time"
#define def_MaxPosSlider                400
#define def_ShortName                   "Market " + def_SymbolReplay
//+------------------------------------------------------------------+
union u_Interprocess
{
   union u_0
   {
      double  df_Value;       // Value of the terminal global variable...
      ulong   IdGraphic;      // Contains the Graph ID of the asset...
   }u_Value;
   struct st_0
   {
      bool    isPlay;         // Indicates whether we are in Play or Pause mode...
      bool    isWait;         // Tells the user to wait...
      ushort  iPosShift;      // Value between 0 and 400...
   }s_Infos;
   datetime   ServerTime;
};
//+------------------------------------------------------------------+
```

Here we set the name of the global terminal variable that will be used for communication. Next, we define a variable that will be used to access data in datetime format. After that, we go to the C\_Replay class and add the following to the class destructor:

```
~C_Replay()
   {
      ArrayFree(m_Ticks.Info);
      ArrayFree(m_Ticks.Rate);
      m_IdReplay = ChartFirst();
      do
      {
         if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
         ChartClose(m_IdReplay);
      }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
      for (int c0 = 0; (c0 < 2) && (!SymbolSelect(def_SymbolReplay, false)); c0++);
      CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
      CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
      CustomSymbolDelete(def_SymbolReplay);
      GlobalVariableDel(def_GlobalVariableReplay);
      GlobalVariableDel(def_GlobalVariableIdGraphics);
      GlobalVariableDel(def_GlobalVariableServerTime);
      Print("Finished replay service...");
   }
```

By adding this line, we ensure that when we close the replay/simulation service, we remove the global terminal variable responsible for passing the time value to the timer. Now we need to ensure that this global terminal variable is created at the right time. You should never create the same global variable while a loop is running, as it may fire at different times. As soon as the system is enabled, it should already have access to the contents of the global terminal variable. In some ways I considered using a creation system similar to the control indicator. But since at the moment the EA may or may not be present on the chart, then at least for now, let's make the terminal variable responsible for the timer created by the service. This way we will have at least minimally adequate control over what we do. Perhaps the situation will change in the future. The below code shows where it is created:

```
bool ViewReplay(ENUM_TIMEFRAMES arg1)
   {
#define macroError(A) { Print(A); return false; }
      u_Interprocess info;

      if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
         macroError("Asset configuration is not complete, it remains to declare the size of the ticket.");
      if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
         macroError("Asset configuration is not complete, need to declare the ticket value.");
      if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
         macroError("Asset configuration not complete, need to declare the minimum volume.");
      if (m_IdReplay == -1) return false;
      if ((m_IdReplay = ChartFirst()) > 0) do
      {
         if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
         {
            ChartClose(m_IdReplay);
            ChartRedraw();
         }
      }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
      Print("Waiting for [Market Replay] indicator permission to start replay ...");
      info.ServerTime = m_Ticks.Info[m_ReplayCount].time;
      CreateGlobalVariable(def_GlobalVariableServerTime, info.u_Value.df_Value);
      info.u_Value.IdGraphic = m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
      ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");
      CreateGlobalVariable(def_GlobalVariableIdGraphics, info.u_Value.df_Value);
      while ((!GlobalVariableCheck(def_GlobalVariableReplay)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);

      return ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""));
#undef macroError
   }
```

In the middle of all this clutter of lines is a point that initializes the value of the variable. This ensures that the system already has a value to work with and allows us to check if the replay/simulation is actually in sync. Immediately after this we have a call that creates and initializes a global terminal variable. This initialization code is shown below:

```
void CreateGlobalVariable(const string szName, const double value)
   {
      GlobalVariableDel(szName);
      GlobalVariableTemp(szName);
      GlobalVariableSet(szName, value);
   }
```

The creation and initialization code is very simple. When the platform is closed and a request is made to save global terminal variables, those variables that are used in the replay/simulation system are not saved. We don't want such values to be stored and retrieved later. Now we can start looking at the timer readings. However, if we do this at the moment, the value contained in it will always be the same, and will not necessarily bring us any practical benefit. It will appear as if the replay/simulation service has been paused. But we want the timer to move when it is active, that is, in play mode. Thus, we will model what the requirements for the **TimeCurrent** function, in which we get the corresponding time and date data found on the server. For this, we need the value of the global variable to change approximately every second. It would be correct to set a timer for this process. But since we don't have the ability to do this, we need another means to implement such a change in the value of a global variable.

To determine the time, we also need to implement some additions to the C\_Replay class. They can be seen in the code below:

```
bool LoopEventOnTime(const bool bViewBuider)
   {
      u_Interprocess Info;
      int iPos, iTest, iCount;

      if (!m_Infos.bInit) ViewInfos();
      iTest = 0;
      while ((iTest == 0) && (!_StopFlag))
      {
         iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);
         iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value) ? iTest : -1);
         iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);
         if (iTest == 0) Sleep(100);
      }
      if ((iTest < 0) || (_StopFlag)) return false;
      AdjustPositionToReplay(bViewBuider);
      Info.ServerTime = m_Ticks.Info[m_ReplayCount].time;
      GlobalVariableSet(def_GlobalVariableServerTime, Info.u_Value.df_Value);
      iPos = iCount = 0;
      while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))
      {
         iPos += (int)(m_ReplayCount < (m_Ticks.nTicks - 1) ? m_Ticks.Info[m_ReplayCount + 1].time_msc - m_Ticks.Info[m_ReplayCount].time_msc : 0);
         CreateBarInReplay(true);
         while ((iPos > 200) && (!_StopFlag))
         {
            if (ChartSymbol(m_IdReplay) == "") return false;
            GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value);
            if (!Info.s_Infos.isPlay) return true;
            Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
            GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
            Sleep(195);
            iPos -= 200;
	    iCount++;
            if (iCount > 4)
            {
               iCount = 0;
               GlobalVariableGet(def_GlobalVariableServerTime, Info.u_Value.df_Value);
               Info.ServerTime += 1;
               GlobalVariableSet(def_GlobalVariableServerTime, Info.u_Value.df_Value);
            }
         }
      }
      return (m_ReplayCount == m_Ticks.nTicks);
   }
```

The newly added variable will help us try to approximate the time where a new bar will start. However, since we can move the time forward, we need the value specified by our "server" to be reset. This will be done at the moment when the function points to the new observation point. Even if we remain in pause mode for a while, we must be sure that the value will be appropriate. The real problem comes when we are about to set the timer. We can't just capture any value. The system may be ahead or behind the real time displayed on the computer clock. For this reason, at this early stage we are trying to get the timing right. We have a built-in timer in the bar generator. We will use it as a guide. It produces a tick approximately every 195 milliseconds, which brings us closer to a count of 5 units. Since we started with a value of zero, we will check if the counter value is greater than 4. When this happens, we will increase the time by one unit, that is, 1 second, and this value will be placed in a global terminal variable so that the rest of the system can use it. Then the whole loop will repeat.

This allows us to know when a new bar will appear. Yes, this is exactly the idea, but slight variations are possible at different times. And as they accumulate, time becomes out of sync. What we need is to have the time acceptably synchronous, not really achieve perfection. In the vast majority of cases we can get pretty close. To do this, we will slightly change the function described above to ensure increased synchrony between the bar and the timer. These changes are shown below:

```
//...

   Sleep(195);
   iPos -= 200;
   iCount++;
   if (iCount > 4)
   {
      iCount = 0;
      GlobalVariableGet(def_GlobalVariableServerTime, Info.u_Value.df_Value);
      Info.ServerTime += 1;
      Info.ServerTime = ((Info.ServerTime + 1) < m_Ticks.Info[m_ReplayCount].time ? Info.ServerTime : m_Ticks.Info[m_ReplayCount].time);
      GlobalVariableSet(def_GlobalVariableServerTime, Info.u_Value.df_Value);
   }

//...
```

You may think this is crazy. In a way, I agree that this is a bit of a crazy move on my part. But by adding this particular line, we will be able to maintain synchronization at an acceptable level. If the instrument has good liquidity to the extent that trades are created in a relatively short period of time, preferably less than 1 second, then we will have a fairly synchronized system. It may be very close to a near-perfect system. But this is achieved only if the instrument actually has sufficient liquidity. What is going on here? Let's think a little: Every 5 loops of approximately 195 milliseconds, we will execute code to update the timer. Thus, the update rate is about 975 milliseconds, which means that each loop is missing 25 milliseconds. But in fact, this value is not constant. Sometimes it can be a little greater, sometimes a little less. You should not try to set up synchronization using the new **sleep** command to force the system to slow down to overcome this difference. At first glance, this should work. But over time, these micro differences become so large that the entire system will be out of sync. To solve this problem, we do something different. Instead of trying to get the time, we use the bar itself to ensure synchrony. When the **CreateBarInReplay** function is executed, it always points to the current tick. By comparing the time value of this tick with the time value that is in the global variable, in some cases we can get a value greater than one, in this case 1 second. If this value is lower, that is, the **Info.ServerTime** variable is delayed in time for an accumulated 25 milliseconds, the current tick time value will be used to correct the difference and make the timer very close to the perfect value. But, as I already reported at the beginning of the explanation, this mechanism automatically adjusts the system if the instrument we are using has sufficient liquidity. If trading is stopped for a long time, and 5-10 minutes pass between one trade and another, then the accuracy of the timing system will suffer. This is because every second it will lag by an average of 25 milliseconds (this is an average, not an exact value).

Now we can move on to the next part. It is located in the C\_Study.mqh header file, where we will force the system to report data so that we can correctly estimate when a new bar will appear on the chart.

### Adapting the C\_Study class

To start the modifications, we first need to make the changes as shown below:

```
void Update(void)
   {
      switch (m_Info.Status)
      {
         case eCloseMarket: m_Info.szInfo = "Closed Market";                             break;
         case eAuction    : m_Info.szInfo = "Auction";                                   break;
         case eInReplay   :
         case eInTrading  : m_Info.szInfo = TimeToString(GetBarTime(), TIME_SECONDS);    break;
         case eInReplay   : m_Info.szInfo = "In Replay";                                 break;
         default          : m_Info.szInfo = "ERROR";
      }
      Draw();
   }
```

We remove the crossed-out line and move it to a higher level to have the same status and functionality as when using the system in the physical market. In this way we level the things, i.e., we will have the same behavior in both situations, both during replay/simulation and on a demo/real account. Now that we've done that, we can look at the first modification made to the **GetBarTime** function code:

```
const datetime GetBarTime(void)
   {
      datetime dt;
      u_Interprocess info;

      if (m_Info.Status == eInReplay)
      {
         if (!GlobalVariableGet(def_GlobalVariableServerTime, info.u_Value.df_Value)) return ULONG_MAX;
         dt = info.ServerTime;
      }else dt = TimeCurrent();

      if (m_Info.Rate.time <= dt)
         m_Info.Rate.time = iTime(GetInfoTerminal().szSymbol, PERIOD_CURRENT, 0) + PeriodSeconds();

      return m_Info.Rate.time - dt;
   }
```

This is where the magic happens in the analytics system. In the old version of this function, this call set a timer. But it is not suitable for running in a replay/simulation system. Therefore, we create a method to bypass this call so that the system has the same concepts and information no matter where it is used. So we add extra things that will be used during the period when the code is on the chart whose asset is used in replay. This is not something extraordinary. We simply take the value reported and placed in the global variable and use it as if it came from the trade server. This is where we actually bypass the system. But we still have a problem when we are in a low charting time where the number of trades is not enough to build the bars correctly. This is especially true when the data used contains intraday trades (a very common case for some assets). We will end up with a failure. This failure appears as a gap between the presented and displayed information. It's not because service stopped updating. But the indicator does not display any information, leaving us in the dark, without knowing what is really happening. Even if the there are no trades for the instrument, for some Forex trading pairs it can be quite common when only one trade occurs when the bar opens. This can be seen in the attached files, where at the beginning of the day there is a gap between the point where the bar opened and the point where the trade actually occurred. During this phase we will not see any information about what is happening. This needs to be corrected somehow, either because the trade is taking place outside of the point where the system expects it, or because the asset being used is participating in an auction. We really need to bring things as close to reality as possible.

To solve these problems, we need to make two changes to the class code. We will do it wright now. The first change is shown below:

```
void Update(void)
   {
      datetime dt;

      switch (m_Info.Status)
      {
         case eCloseMarket:
            m_Info.szInfo = "Closed Market";
            break;
         case eInReplay   :
         case eInTrading  :
            dt = GetBarTime();
            if (dt < ULONG_MAX)
            {
               m_Info.szInfo = TimeToString(dt, TIME_SECONDS);
               break;
            }
         case eAuction    :
            m_Info.szInfo = "Auction";
            break;
         default          :
            m_Info.szInfo = "ERROR";
      }
      Draw();
   }
```

The **Update** code shown above, despite its strange and complex appearance, is much simpler and more convenient than it seems. We have the following scenario. If we are in a replay system or even in a real market and receive from the **GetBarTime** function an **ULONG\_MAX** value, then we will display a message about the auction. If the value is less than this **ULONG\_MAX**, and in normal situations this is always the case, the timer value will be displayed.

Based on this information we can return to the **GetBarTime** function and generate the data needed for the Update function to plot the correct data so the user knows how things are going. So, the new GetBarTime function can be seen in the code below.

```
const datetime GetBarTime(void)
   {
      datetime dt;
      u_Interprocess info;
      int i0 = PeriodSeconds();

      if (m_Info.Status == eInReplay)
      {
         if (!GlobalVariableGet(def_GlobalVariableServerTime, info.u_Value.df_Value)) return ULONG_MAX;
         dt = info.ServerTime;
         if (dt == ULONG_MAX) return ULONG_MAX;
      }else dt = TimeCurrent();
      if (m_Info.Rate.time <= dt)
         m_Info.Rate.time = (datetime)(((ulong) dt / i0) * i0)) + i0;

      return m_Info.Rate.time - dt;
   }
```

This nice code completely solves our problem, at least for now, since we will have to make additions to the service code. Let's look at what's going on here. In the case of the physical market, where we use the **TimeCurrent** function, nothing changes. This is in the beginning. But when we are in the replay system, things change in a very peculiar way. Therefore, pay attention to understanding how the system manages to represent what is happening, regardless of what is happening with the replay or simulation data. If the service places an **ULONG\_MAX** value to a global terminal variable, or if this variable is not found, The **GetBarTime** function should return a **ULONG\_MAX** value. After this the **Update** method will tell us that we are in auction mode. It is not possible to advance the timer. Now comes the interesting part, which solves our second problem. Unlike using the system on an instrument connected to a trading server, where we will always be in sync, when using the replay/simulation mode, things can get out of control and we may encounter some pretty unusual situations. To solve this problem, we use this calculation, which works for both the physical market and the system we are developing. In this calculation, we replace the old method, when it was necessary to know the opening time of the current bar. Thus, we were able to solve both problems using replay/simulation.

However, we need to go back to the C\_Replay class so that the system can indicate when the asset went into auction. This part is relatively easy since all we need to do is set the **ULONG\_MAX** value to the global terminal variable. See that this was explained relatively simply, because we have other problems before us. But let's see how this will look in practice.

### Adapting the C\_Replay class to the communication system

The first thing we'll do in the C\_Replay class is change the following code:

```
bool ViewReplay(ENUM_TIMEFRAMES arg1)
   {
#define macroError(A) { Print(A); return false; }
      u_Interprocess info;

      if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE) == 0)
         macroError("Asset configuration is not complete, it remains to declare the size of the ticket.");
      if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE) == 0)
         macroError("Asset configuration is not complete, need to declare the ticket value.");
      if (SymbolInfoDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP) == 0)
         macroError("Asset configuration not complete, need to declare the minimum volume.");
      if (m_IdReplay == -1) return false;
      if ((m_IdReplay = ChartFirst()) > 0) do
      {
         if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
         {
            ChartClose(m_IdReplay);
            ChartRedraw();
         }
      }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
      Print("Waiting for [Market Replay] indicator permission to start replay ...");
      info.ServerTime = ULONG_MAX;
      info.ServerTime = m_Ticks.Info[m_ReplayCount].time;
      CreateGlobalVariable(def_GlobalVariableServerTime, info.u_Value.df_Value);
      info.u_Value.IdGraphic = m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
      ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");
      CreateGlobalVariable(def_GlobalVariableIdGraphics, info.u_Value.df_Value);
      while ((!GlobalVariableCheck(def_GlobalVariableReplay)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);

      return ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""));
#undef macroError
   }
```

Notice that we have removed one line and replaced it with another. With this change, when the replay/simulation service is launched and the asset is running on the chart, a message about the auction will be shown. It will indicate that the system is communicating as it should. That was the easy part, now let's move on to the hard part. It may happen that when the service has already been launched, and we need to find out whether the asset has been auctioned or not. An asset can be put for at auction for a number of reasons, all of which could be completely unexpected. There could have been a much greater variation or the order book could have crashed and been completely cleared, and so on. The reason why the asset was put on auction is not important. What is important is what happens when it put on auction. There are specific rules that determine how the auction will be conducted, but the most important rule for us is this: what is the minimum time an asset must be at auction? This is exactly the point. If this time is less than 1 minute, then the asset in the real market can certainly enter and exit trades without the replay/simulation system being able to detect it. Or rather, it will not be possible to notice this variation, since it will always be occurring within the shortest time that can be used to define the time of a bar.

The simplest mechanism that can be used in the replay/simulation system to determine that an asset has been put on auction is to check the time difference between one bar and another. If this difference exceeds 1 minute, we must inform the user that the asset has just entered the auction process, and thus suspend it for the entire period. This type of mechanism will be useful in the future. For now, we will only deal with its development and implementation, leaving other issues for another moment. Let's see how to solve this problem. This can be seen in the following code:

```
bool LoopEventOnTime(const bool bViewBuider)
   {
      u_Interprocess Info;
      int iPos, iTest, iCount;

      if (!m_Infos.bInit) ViewInfos();
      iTest = 0;
      while ((iTest == 0) && (!_StopFlag))
      {
         iTest = (ChartSymbol(m_IdReplay) != "" ? iTest : -1);
         iTest = (GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value) ? iTest : -1);
         iTest = (iTest == 0 ? (Info.s_Infos.isPlay ? 1 : iTest) : iTest);
         if (iTest == 0) Sleep(100);
      }
      if ((iTest < 0) || (_StopFlag)) return false;
      AdjustPositionToReplay(bViewBuider);
      Info.ServerTime = m_Ticks.Info[m_ReplayCount].time;
      GlobalVariableSet(def_GlobalVariableServerTime, Info.u_Value.df_Value);
      iPos = iCount = 0;
      while ((m_ReplayCount < m_Ticks.nTicks) && (!_StopFlag))
      {
         iPos += (int)(m_ReplayCount < (m_Ticks.nTicks - 1) ? m_Ticks.Info[m_ReplayCount + 1].time_msc - m_Ticks.Info[m_ReplayCount].time_msc : 0);
         CreateBarInReplay(true);
         while ((iPos > 200) && (!_StopFlag))
         {
            if (ChartSymbol(m_IdReplay) == "") return false;
            GlobalVariableGet(def_GlobalVariableReplay, Info.u_Value.df_Value);
            if (!Info.s_Infos.isPlay) return true;
            Info.s_Infos.iPosShift = (ushort)((m_ReplayCount * def_MaxPosSlider) / m_Ticks.nTicks);
            GlobalVariableSet(def_GlobalVariableReplay, Info.u_Value.df_Value);
            Sleep(195);
            iPos -= 200;
            iCount++;
            if (iCount > 4)
            {
               iCount = 0;
               GlobalVariableGet(def_GlobalVariableServerTime, Info.u_Value.df_Value);
               if ((m_Ticks.Info[m_ReplayCount].time - m_Ticks.Info[m_ReplayCount - 1].time) > 60) Info.ServerTime = ULONG_MAX; else
               {
                  Info.ServerTime += 1;
                  Info.ServerTime = ((Info.ServerTime + 1) < m_Ticks.Info[m_ReplayCount].time ? Info.ServerTime : m_Ticks.Info[m_ReplayCount].time);
               };
               GlobalVariableSet(def_GlobalVariableServerTime, Info.u_Value.df_Value);
            }
         }
      }
      return (m_ReplayCount == m_Ticks.nTicks);
   }
```

Note that we checked the time difference between one tick and the next. If this difference is greater than 60 seconds, that is, greater than the shortest bar creation time, we will report that this is an "auction call" and the entire replay/simulation system will indicate the auction. If the time difference is less than or equal to 60 seconds, it means the asset is still active and the timer should be started as discussed in this article. With this we have completed the current stage.

### Conclusion

Today we looked at how to add a timer indicating the appearance of a new bar in a completely practical, robust, reliable and effective way. We have went through several moments when it seemed impossible, but in reality nothing is impossible. Perhaps it will be a little more difficult to overcome, but we can always find a suitable way to solve problems. The main point here is to show that you should always try to create a solution that should be applicable in all situations where it is required. It makes no sense to program something for the user, or even ourselves (to learn the possibilities), if the application of such a model requires the use of absolutely different too;s. This is completely demotivating. Therefore, always remember: If you need to do something, then it needs to be done correctly, and not so that everything works in theory but in practice demonstrates completely different behavior.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11378](https://www.mql5.com/pt/articles/11378)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11378.zip "Download all attachments in the single ZIP archive")

[Files\_-\_BOLSA.zip](https://www.mql5.com/en/articles/download/11378/files_-_bolsa.zip "Download Files_-_BOLSA.zip")(1358.24 KB)

[Files\_-\_FOREX.zip](https://www.mql5.com/en/articles/download/11378/files_-_forex.zip "Download Files_-_FOREX.zip")(3743.96 KB)

[Files\_-\_FUTUROS.zip](https://www.mql5.com/en/articles/download/11378/files_-_futuros.zip "Download Files_-_FUTUROS.zip")(11397.51 KB)

[Market\_Replay\_-\_31.zip](https://www.mql5.com/en/articles/download/11378/market_replay_-_31.zip "Download Market_Replay_-_31.zip")(55.93 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/463832)**

![Neural networks made easy (Part 62): Using Decision Transformer in hierarchical models](https://c.mql5.com/2/59/Neural_networks_are_easy_0Part_62s_logo.png)[Neural networks made easy (Part 62): Using Decision Transformer in hierarchical models](https://www.mql5.com/en/articles/13674)

In recent articles, we have seen several options for using the Decision Transformer method. The method allows analyzing not only the current state, but also the trajectory of previous states and actions performed in them. In this article, we will focus on using this method in hierarchical models.

![Developing a Replay System (Part 30): Expert Advisor project — C_Mouse class (IV)](https://c.mql5.com/2/58/replay-p30-avatar.png)[Developing a Replay System (Part 30): Expert Advisor project — C\_Mouse class (IV)](https://www.mql5.com/en/articles/11372)

Today we will learn a technique that can help us a lot in different stages of our professional life as a programmer. Often it is not the platform itself that is limited, but the knowledge of the person who talks about the limitations. This article will tell you that with common sense and creativity you can make the MetaTrader 5 platform much more interesting and versatile without resorting to creating crazy programs or anything like that, and create simple yet safe and reliable code. We will use our creativity to modify existing code without deleting or adding a single line to the source code.

![Developing a Replay System (Part 32): Order System (I)](https://c.mql5.com/2/59/sistema_de_Replay_32_logo_.png)[Developing a Replay System (Part 32): Order System (I)](https://www.mql5.com/en/articles/11393)

Of all the things that we have developed so far, this system, as you will probably notice and eventually agree, is the most complex. Now we need to do something very simple: make our system simulate the operation of a trading server. This need to accurately implement the way the trading server operates seems like a no-brainer. At least in words. But we need to do this so that the everything is seamless and transparent for the user of the replay/simulation system.

![The Disagreement Problem: Diving Deeper into The Complexity Explainability in AI](https://c.mql5.com/2/72/The_Disagreement_Problem_Diving_Deeper_into_The_Complexity_Explainability_in_AI____LOGO.png)[The Disagreement Problem: Diving Deeper into The Complexity Explainability in AI](https://www.mql5.com/en/articles/13729)

In this article, we explore the challenge of understanding how AI works. AI models often make decisions in ways that are hard to explain, leading to what's known as the "disagreement problem". This issue is key to making AI more transparent and trustworthy.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11378&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049574543242145251)

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