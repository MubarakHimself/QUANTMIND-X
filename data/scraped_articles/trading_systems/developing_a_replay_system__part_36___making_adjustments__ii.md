---
title: Developing a Replay System (Part 36): Making Adjustments (II)
url: https://www.mql5.com/en/articles/11510
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:16:16.078240
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=lxqlvngtlzpxirfnycaegixwxsgkqmio&ssn=1769184974569230165&ssn_dr=0&ssn_sr=0&fv_date=1769184974&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11510&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2036)%3A%20Making%20Adjustments%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918497465170689&fz_uniq=5070151074427375822&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the previous article, [Developing a Replay System (Part 35): Making Adjustments](https://www.mql5.com/en/articles/11492), we created an EA that can easily manage the market replay service. While the fixes in this article have improved the user experience, we have not yet completely redesigned everything related to the replay/simulation system.

Despite the improved user experience, we still have a small problem. In this article, we will solve this problem, which, despite its relative simplicity (in theory), will turn out to be quite difficult to solve correctly. Thus, we will expand our knowledge about MetaTrader 5 and learn how to use the MQL5 language in more detail.

### Displaying the market type

When the EA is located on the chart, it will inform about the type of account detected. This is important in order to know how the EA should act. But although this works very well, when the system is run on a chart of on a REAL or DEMO account, the system does not use what controls the replay/simulation system, and does not report the type of account to which the asset belongs, but rather the type of account on which the platform operates. This problem, although small, does cause us some inconvenience.

You might think that the solution to this problem is not complex or even difficult, it is very simple. Let's make sure that the replay/simulation system somehow tells you which account type is correct. This of course depends on the asset used. In fact, this idea is simple. But let's put it into practice - this will be another story. The truth is that getting the replay/simulation system to tell us which account type to use is not that easy. But fortunately, the MetaTrader 5 platform offers us the opportunity to implement a solution that is adequate and plausible for the actual use.

However, we will not do this in a reckless way. We will implement the solution in a certain way that will prevent certain things on certain account types. This information will be important for us when creating an order system. First, let's think about what we are talking about. The replay/simulation system will be able to use assets from different markets. It means that we can use assets which imply NETTING or HEDGING account types.

Note. Since the EXCHANGE type is very similar to NETTING, we will not actually use this EXCHANGE type. We will consider it as NETTING.

As a user of the replay/simulation system, you know what type of market, or more specifically what type of account, the asset is using. We can then add the ability for the user to indicate this to the replay/simulation system. This is the easiest part because all we need to do is add a new command to the configuration file of the trading symbol. But this does not guarantee that the information will be available to those places that really need it. So where is this information used in our replay/simulation system? This information is used by the Expert Advisor, namely the C\_Manager class together with the C\_Orders class. This is where this specific information is actively used. Even if this is not done at the moment, it is necessary to think more globally and generally. We may want to use something from what we saw earlier in the series [Creating an EA that works automatically](https://www.mql5.com/en/articles/11438)

In that series of articles, the EA needed to know the account type being used. Precisely for the same reason, we needed to make sure that the replay/modeling system could inform the EA about this as well. Otherwise, the functionality discussed in that series will be lost and we will not be able to transfer it to our system. Well, we already have an idea of how to tell the replay/modeling service what type of account a particular asset uses. But the question is how to pass this information into the EA?

This is where we really need to stop and think. If you look at the code, you can find out the type of account. Take a look at it:

```
C_Manager(C_Terminal *arg1, C_Study *arg2, color cPrice, color cStop, color cTake, const ulong magic, const double FinanceStop, const double FinanceTake, uint Leverage, bool IsDayTrade)
         :C_ControlOfTime(arg1, magic)
   {
      string szInfo = "HEDGING";

      Terminal = arg1;
      Study = arg2;
      if (CheckPointer(Terminal) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
      if (CheckPointer(Study) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
      if (_LastError != ERR_SUCCESS) return;
      m_Infos.FinanceStop     = FinanceStop;
      m_Infos.FinanceTake     = FinanceTake;
      m_Infos.Leverage        = Leverage;
      m_Infos.IsDayTrade      = IsDayTrade;
      m_Infos.AccountHedging  = false;
      m_Objects.corPrice      = cPrice;
      m_Objects.corStop       = cStop;
      m_Objects.corTake       = cTake;
      m_Objects.bCreate       = false;
      switch ((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE))
      {
         case ACCOUNT_MARGIN_MODE_RETAIL_HEDGING: m_Infos.AccountHedging = true; break;
         case ACCOUNT_MARGIN_MODE_RETAIL_NETTING: szInfo = "NETTING";            break;
         case ACCOUNT_MARGIN_MODE_EXCHANGE      : szInfo = "EXCHANGE";           break;
      }
      Print("Detected Account ", szInfo);
   }
```

Thus the EA can determine the type of account being used. But there's a problem here, and that's the [AccountInfoInteger](https://www.mql5.com/en/docs/account/accountinfointeger) function. Well, not the function itself is a problem, since it reports exactly what we ask it to report. The problem is that when using the replay/simulation service, the result of the AccountInfoInteger function will be information about the type of account on the trading server. In other words, if we are connected to a NETTING server, we will end up running on a NETTING account, even if the replay/simulation asset is HEDGING.

This is the problem. Now for the ideas. We could instruct the EA too read the configuration file of the replay/simulation asset. In a sense, this would be appropriate if we only used a single file for it. Well, we could therefore ask the replay/simulation service to pass this account type information. Then the EA would know the correct type. Yes, exactly. But there is a small point here. Unlike C/C++, MQL5 does not have specific code construction types. It's not that it is completely not possible to use some forms of coding to pass information between programs. We have already seen multiple times that this is possible. However, the problem is that our information must be contained in an 8-byte block. This allows the use of terminal global variables to pass information between programs. Remember that we will be doing this using MQL5. There are other ways to do this, but here I want to use the capabilities of the MQL5 platform and language.

Let's get back to our problem: we can use a method that has been in use for a long time. For this we used the InterProcess.mqh file to provide the communication we needed. But this file has one problem which we will solve in this article. To understand the problem, let's look at its code.

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_SymbolReplay                "RePlay"
#define def_GlobalVariableReplay        def_SymbolReplay + "_Infos"
#define def_GlobalVariableIdGraphics    def_SymbolReplay + "_ID"
#define def_GlobalVariableServerTime    def_SymbolReplay + "_Time"
#define def_MaxPosSlider                400
#define def_ShortName                   "Market_" + def_SymbolReplay
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
        datetime ServerTime;
};
//+------------------------------------------------------------------+
```

The problem is with these boolean values. Unlike C/C++, where each Boolean value can be contained in one bit, in MQL5 each Boolean value will occupy eight bits. Maybe you don't understand what this really means, but an integer byte will be used when we only need one bit to store what we need. Ignorance of this fact causes a problem. Remember that ushort requires two bytes to transmit information. The st\_0 structure actually takes up four bytes, not three as we expected. If we add four more, and only four booleans to this st\_0 structure, we will be at the limit of eight bytes that we can use.

Things like this make programming a little more complicated, so the idea of doing everything exclusively within MQL5 is a little more complicated than it seems. If things get more complicated, or we need to pass more data in boolean mode, we will have to radically change this structure. This is a logistical nightmare. However, until now, we can retain the same structure. Although it would be nice if the developers of the MQL5 language allowed us to use the same programming mode as in C/C++, at least in the case of Boolean types. This way we can determine the number of bits used by each variable, and the compiler will do all the work of organizing, separating and grouping the variables. This will save us from having to do low-level programming to achieve such goals. With only one difference: if this is done by MQL5 developers, the result would be much more efficient, since it would be possible to organize and build using assembly code or something very similar to machine language. If only we, the developers, did this work, the efficiency of the code would not be as great and would require much more work.

**So, here's a hint of an improvement for MQL5. Something seemingly trivial, but very useful in programming in this language.**

Let's now go back to our code now, since we can still work only with what we have in hand. The new code for the Interprocess.mqh file is shown below:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_SymbolReplay                "RePlay"
#define def_GlobalVariableReplay        def_SymbolReplay + "_Infos"
#define def_GlobalVariableIdGraphics    def_SymbolReplay + "_ID"
#define def_GlobalVariableServerTime    def_SymbolReplay + "_Time"
#define def_MaxPosSlider                400
#define def_ShortName                   "Market_" + def_SymbolReplay
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
                bool    isHedging;      // If true we are in a Hedging account, if false the account is Netting...
                bool    isSync;         // If true indicates that the service is synchronized...
                ushort  iPosShift;      // Value between 0 and 400...
        }s_Infos;
        datetime ServerTime;
};
//+------------------------------------------------------------------+
```

Notice that all we've done here is add two new boolean variables. This will be enough to solve the immediate problem, but it brings us very close to the eight-byte limit. Although we are only using four Boolean numbers, which correspond to four bits, the fact that each Boolean number takes up eight bits means that the st\_0 structure is six bytes in size, not three as expected.

**IMPORTANT NOTE:** For those who do not know what the same code would look like if the MQL5 language used similar C/C++ modeling to define Boolean structures, I will just say that reading and writing the code would make  no troubles at all. The same code that takes up six bytes would take up three:

```
union u_Interprocess
{
        union u_0
        {
                double  df_Value;      // Value of the terminal global variable...
                ulong   IdGraphic;     // Contains the Graph ID of the asset...
        }u_Value;
        struct st_0
        {
    		char  isPlay    : 1;   // Indicates whether we are in Play or Pause mode...
    		char  isWait    : 1;   // Tells the user to wait...
    		char  isHedging : 1;   // If true we are in a Hedging account, if false the account is Netting...
    		char  isSync    : 1;   // If true indicates that the service is synchronized...
                ushort  iPosShift;     // Value between 0 and 400...
        }s_Infos;
        datetime ServerTime;
};
```

The MQL5 language compiler does not understand the above code, at least at the time of writing, but here we tell the compiler that we will only use one bit in each declaration. It's the compiler's job, not the programmer's, to set the variable appropriately so that when we access a bit we can do so in a more controlled manner, and use a label to do so rather than some compilation definition directive.

Although this construction is still possible in native MQL5, doing this using compilation directives is subject to errors and failures, and also makes the code very difficult to maintain. I'd like to make this point clear so you don't think that language doesn't allow us to do certain things.

But if the world gives us lemons, let's make lemonade. And let's not complain that the world is not what we expect it to be. Having made these changes to the Interprocess.mqh file, we can move on to the next point. Now we will modify C\_ConfigService.mqh. We will make some changes, starting with the code below:

```
        private :
                enum eWhatExec {eTickReplay, eBarToTick, eTickToBar, eBarPrev};
                enum eTranscriptionDefine {Transcription_INFO, Transcription_DEFINE};
                struct st001
                {
                        C_Array *pTicksToReplay, *pBarsToTicks, *pTicksToBars, *pBarsToPrev;
                        int     Line;
                }m_GlPrivate;
                string  m_szPath;
                bool    m_AccountHedging;
```

We have added a private variable to the C\_ConfigService class. Therefore, we can remember what we will process next. First we must initialize this variable. This is done in the class constructor:

```
C_ConfigService()
        :m_szPath(NULL),
         m_ModelLoading(1),
         m_AccountHedging(false)
   {
   }
```

The need for a variable that will serve as memory is explained by the fact that we need the Control indicator to create a global terminal variable, which will receive the already configured data. However, this global terminal variable will only appear when the template loads the control indicator onto the chart. And the chart will only appear after the replay/simulation service has finished setting up the entire system. If the user-specified setting value was not stored somewhere, we would not be able to tell the EA the account type. But with careful programming we can pass the account type configured in the replay/simulation service to the EA. This way we can know how to work. But this is provided that you use the knowledge on how to create an automated EA.

Since the variable is private and the C\_Replay class sets a global terminal variable, we need a C\_Replay class method to access the account type variable. This is done in the following code:

```
inline const bool TypeAccountIsHedging(void) const
   {
      return m_AccountHedging;
   }
```

This will return a value. Now we need to make sure that the user specified settings can be captured and used by our code. To do this, we will need to add new code to the capture system. It is shown below:

```
inline bool Configs(const string szInfo)
   {
      const string szList[] = {
                               "PATH",
                               "POINTSPERTICK",
                               "VALUEPERPOINTS",
                               "VOLUMEMINIMAL",
                               "LOADMODEL",
                               "ACCOUNT"
                              };
      string  szRet[];
      char    cWho;

      if (StringSplit(szInfo, '=', szRet) == 2)
      {
         StringTrimRight(szRet[0]);
         StringTrimLeft(szRet[1]);
         for (cWho = 0; cWho < ArraySize(szList); cWho++) if (szList[cWho] == szRet[0]) break;
         switch (cWho)
         {
            case 0:
               m_szPath = szRet[1];
               return true;
            case 1:
               CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, StringToDouble(szRet[1]));
               return true;
            case 2:
               CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, StringToDouble(szRet[1]));
               return true;
            case 3:
               CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_VOLUME_STEP, StringToDouble(szRet[1]));
               return true;
            case 4:
               m_ModelLoading = StringInit(szRet[1]);
               m_ModelLoading = ((m_ModelLoading < 1) && (m_ModelLoading > 4) ? 1 : m_ModelLoading);
               return true;
            case 5:
               if (szRet[1] == "HEDGING") m_AccountHedging = true;
               else if (szRet[1] == "NETTING") m_AccountHedging = false;
               else
               {
                  Print("Entered account type is not invalid.");
                  return false;
               }
               return true;
         }
         Print("Variable >>", szRet[0], "<< not defined.");
      }else
         Print("Configuration definition >>", szInfo, "<< invalidates.");

      return false;
   }
```

This new code allows us to know which account to use. It is very easy to do and at the same time it is absolutely functional. Now, the configuration file may look something like the following:

```
[Config]
Path = Forex\EURUSD
PointsPerTick = 0.00001
ValuePerPoints = 1.0
VolumeMinimal = 0.01
Account = HEDGING
```

or like this:

```
[Config]
Path = Petrobras PN
PointsPerTick = 0.01
ValuePerPoints = 1.0
VolumeMinimal = 100.0
Account = NETTING
```

We have various assets that can be used by the replication/simulation service, but as a user, you can determine the type of account to be used. It's quite simple and straightforward. This way, we can extend our analysis to any type of market. It allows us to use any methodology known to us or developed by us. Now let's look at how to make the C\_Replay class set this data for us, since any work related to the C\_ConfigService class has been finished. To make this happen, and to make the user-specified configuration visible to the entire replay/simulation system, we don't have to do much work. Everything we need to do is shown in the following code:

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
      CreateGlobalVariable(def_GlobalVariableServerTime, info.u_Value.df_Value);
      info.u_Value.IdGraphic = m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
      ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");
      CreateGlobalVariable(def_GlobalVariableIdGraphics, info.u_Value.df_Value);
      while ((!GlobalVariableCheck(def_GlobalVariableReplay)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);
      while ((!GlobalVariableGet(def_GlobalVariableReplay, info.u_Value.df_Value)) && (!_StopFlag) && (ChartSymbol(m_IdReplay) != "")) Sleep(750);
      info.s_Infos.isHedging = TypeAccountIsHedging();
      info.s_Infos.isSync = true;
      GlobalVariableSet(def_GlobalVariableReplay, info.u_Value.df_Value);

      return ((!_StopFlag) && (ChartSymbol(m_IdReplay) != ""));
#undef macroError
   }
```

We removed one line from the code. The reason is that we need the C\_Replay class to load the value of the variable that was placed by the control indicator so that we know what we are dealing with. Remember, we should not make assumptions. In the future, we may have to pass values to the service once the control indicator starts running. After this, we will read the values specified by the user and report that the system is synchronized, after which we will send the value to MetaTrader 5, making the data available through a global terminal variable. This completes the part about the replay/simulation service, but this we have not writing the code yet. Now we need to modify the EA so that it can know how replay/simulation has been configured by the user. Again, we will consider all related explanations in a separate topic:

### Make the Expert Advisor know the account type

In order to understand how to make the EA recognize the user custom account type, we need to understand how the entire system works. Let's look at image 01, which briefly describes how the initialization process occurs.

![Figure 01](https://c.mql5.com/2/49/001.png)

Figure 01 - Replay/simulation system initialization process.

Why is it so important to understand this figure? Because the C\_Terminal class is a common class for the control indicator and the Expert Advisor. If you don't understand this, you won't be able to understand how and why the system can follow user-configured settings in MetaTrader 5. Moreover, if we don't understand this initialization, we can do something stupid by editing the file that will configure the replay/simulation process. This will make the entire system work incorrectly. Thus, instead of gaining an experience that is close to the real market, we may get a false impression of how we should act when we use in practice what we have learned through the replay/simulation process. So, it is obligatory to understand what the first figure is.

- All blue arrows indicate the first phase of initialization. This happens at the very moment when we, as a user, launch the service in MetaTrader 5.
- After this, we move on to the second stage, which is performed when the service creates a chart and uses a template to initialize the EA and control indicator. This is indicated by a yellow arrow.
- The next step, indicated by the GREEN arrows, is to create a global terminal variable using the control indicator and configure this variable with the replay/simulation service. At this point, we see that the C\_Terminal class is sending data to the control indicator, but not to the C\_Manager class, which is currently being initialized by the EA.
- The last step (PURPLE arrows) is the setup from the data provided by the user in the file that specifies how the replay/simulation will work. The EA needs this information to know what type of account is being used.

Although it may seem confusing, we must be extremely careful to ensure that the C\_Terminal class allows the control indicator to do its work. At the same time, don't let the EA do anything before the C\_Manager class is properly configured. Why am I saying that management should be done in the C\_Terminal class and not in the C\_Manager class? It might be easier to do this in the C\_Manager class. Why not? The reason is not simply to choose which class to store things in, the real reason is the C\_Orders class. If the control system is in the C\_Manager class, the C\_Orders class will not be able to access the data we need. Since we put the control in the C\_Terminal class, then even if we later use another class to manage orders, it will still be able gain access to the data we need. As shown in Figure 01, we cannot choose just a random way to implement this. We need to make sure it's all done correctly. Otherwise, the entire system may fail immediately after startup. I know it's hard to understand and may even be shocking, but believe me, if programming is not done strictly enough, the system will fail. So, let's see how the system is implemented. Let's start with the following code:

```
class C_Terminal
{
        protected:
                enum eErrUser {ERR_Unknown, ERR_PointerInvalid};
                enum eEvents {ev_Update};
//+------------------------------------------------------------------+
                struct st_Terminal
                {
                        ENUM_SYMBOL_CHART_MODE   ChartMode;
                        ENUM_ACCOUNT_MARGIN_MODE TypeAccount;
                        long    ID;
                        string  szSymbol;
                        int     Width,
                                Height,
                                nDigits;
                        double  PointPerTick,
                                ValuePerPoint,
                                VolumeMinimal,
                                AdjustToTrade;
                };
//+------------------------------------------------------------------+
        private :
                st_Terminal m_Infos;
                struct mem
                {
                        long    Show_Descr,
                                Show_Date;
                        bool    AccountLock;
                }m_Mem;
```

This variable provides access to the account type using the methods already implemented in this article. We won't have to add new code into the class to get this information, which is great. But we also have another variable. It will serve as a LOCK, but you will soon understand why and how it will be used. Let's now look at the constructor of the class:

```
C_Terminal()
   {
      m_Infos.ID = ChartID();
      m_Mem.AccountLock = false;
      CurrentSymbol();
      m_Mem.Show_Descr = ChartGetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR);
      m_Mem.Show_Date  = ChartGetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE);
      ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, false);
      ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, 0, true);
      ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_CREATE, 0, true);
      ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, false);
      m_Infos.nDigits = (int) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_DIGITS);
      m_Infos.Width   = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
      m_Infos.Height  = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
      m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
      m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
      m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
      m_Infos.AdjustToTrade = m_Infos.ValuePerPoint / m_Infos.PointPerTick;
      m_Infos.ChartMode     = (ENUM_SYMBOL_CHART_MODE) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_CHART_MODE);
      if(m_Infos.szSymbol != def_SymbolReplay) SetTypeAccount((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE));
      ResetLastError();
   }
```

Here we initialize the 'lock' variable to indicate to the C\_Terminal class that the account type value has not yet been implemented. But we are doing something about it, and this is very important. When the system detects that an asset IS NOT A REPLAY ASSET, the C\_Terminal class must initialize the account data. If this is a replay asset, then the data will not be initialized now, but later in the C\_Manager class. This must be understood, because very soon it will be of great importance. The C\_Manager class initializes the data if the asset is a replay asset. If the asset is **not** a replay asset, it is initialized at this stage. Next we have the following method call:

```
inline void SetTypeAccount(const ENUM_ACCOUNT_MARGIN_MODE arg)
   {
      if (m_Mem.AccountLock) return; else m_Mem.AccountLock = true;
      m_Infos.TypeAccount = (arg == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING ? arg : ACCOUNT_MARGIN_MODE_RETAIL_NETTING);
   }
```

This method is very important. Without it, the type of account used would be unknown. Since in this case you would be assuming that this or that another type is used. And it is important that once you started with a certain account type, the variable value does not change. That is why we use this lock. I emphasize this: only if the lock variable is set to false, we will be able to initialize its value based on the argument passed to the method. After this, the variable cannot be changed until the end of the code execution. Another detail is that we will limit ourselves to the NETTING and HEDGING types. This is because EXCHANGE operates the same as the NETTING type. Since we are using a trading system very close to the real market, I don't see a problem in limiting ourselves to NETTING and HEDGING types. This completes the code part of the C\_Terminal class. Now let's see what needs to be changed in the C\_Manager class. In this C\_Manager class, all we had to do was change the constructor code. Now it looks like this:

```
C_Manager(C_Terminal *arg1, C_Study *arg2, color cPrice, color cStop, color cTake, const ulong magic, const double FinanceStop, const double FinanceTake, uint Leverage, bool IsDayTrade)
         :C_ControlOfTime(arg1, magic)
   {
      string szInfo = "HEDGING";
      u_Interprocess info;

      Terminal = arg1;
      Study = arg2;
      if (CheckPointer(Terminal) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
      if (CheckPointer(Study) == POINTER_INVALID) SetUserError(C_Terminal::ERR_PointerInvalid);
      if (_LastError != ERR_SUCCESS) return;
      m_Infos.FinanceStop     = FinanceStop;
      m_Infos.FinanceTake     = FinanceTake;
      m_Infos.Leverage        = Leverage;
      m_Infos.IsDayTrade      = IsDayTrade;
      m_Infos.AccountHedging  = false;
      m_Objects.corPrice      = cPrice;
      m_Objects.corStop       = cStop;
      m_Objects.corTake       = cTake;
      m_Objects.bCreate       = false;
      if (def_InfoTerminal.szSymbol == def_SymbolReplay)
      {
         do
         {
            while ((!GlobalVariableGet(def_GlobalVariableReplay, info.u_Value.df_Value)) && (!_StopFlag)) Sleep(750);
         }while ((!info.s_Infos.isSync) && (!_StopFlag));
         def_AcessTerminal.SetTypeAccount(info.s_Infos.isHedging ? ACCOUNT_MARGIN_MODE_RETAIL_HEDGING : ACCOUNT_MARGIN_MODE_RETAIL_NETTING);
      };
      switch (def_InfoTerminal.TypeAccount)
      {
         case ACCOUNT_MARGIN_MODE_RETAIL_HEDGING: m_Infos.AccountHedging = true; break;
         case ACCOUNT_MARGIN_MODE_RETAIL_NETTING: szInfo = "NETTING";            break;
      }
      Print("Detected Account ", szInfo);
   }
```

You may look and think that nothing has changed. Or worse, you might not understand what's going on. Since here we have a loop that, under certain conditions, can go into an endless loop. But most of the code remains the same, with one small difference, which lies precisely in this point. Previously we checked here all account types. Now we will leave only two. And the information that allows you to find out the type of account is in the variable created in the C\_Terminal class. But let's take a closer look at the truly new part of the code. It starts when it is determined that the asset being used is the same as the one in the replay/simulation system. If the check passes, we will end up in a double loop, where there is an outer loop and another one nested within this first loop.

In this nested loop, we will wait for the control indicator to create a global terminal variable so that the replay/simulation service can set it. If the program is terminated by the user or the control indicator creates a global terminal variable, we will exit this nested loop and enter the outer loop. This outer loop will only end in two cases: first, if the replay/simulation service sets a global terminal variable; the second - if the user closes the program. Other than these two situations, the loop will not end and will reenter the nested loop. If everything goes well and the outer loop ends, then the account type value will be sent to the C\_Terminal class.

### Conclusion

In this article, we solved a problem that, although small, could cause us a lot of headaches in the future. I hope you understand the importance of these changes, and most importantly, how we managed to solve the problem using MQL5. The attachment contains the source code as well as updated files for use in replay/simulation. If you are already using this service, even if you cannot use it for analysis yet, be sure to also update the files responsible for setting up the replay/simulation service so as to use the appropriate account type. Otherwise, the system will use the default HEDGING account type.

This can lead to problems in the future. So don't forget to update.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11510](https://www.mql5.com/pt/articles/11510)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11510.zip "Download all attachments in the single ZIP archive")

[Files\_-\_FOREX.zip](https://www.mql5.com/en/articles/download/11510/files_-_forex.zip "Download Files_-_FOREX.zip")(3744 KB)

[Files\_-\_BOLSA.zip](https://www.mql5.com/en/articles/download/11510/files_-_bolsa.zip "Download Files_-_BOLSA.zip")(1358.28 KB)

[Files\_-\_FUTUROS.zip](https://www.mql5.com/en/articles/download/11510/files_-_futuros.zip "Download Files_-_FUTUROS.zip")(11397.55 KB)

[Market\_Replay\_-\_36.zip](https://www.mql5.com/en/articles/download/11510/market_replay_-_36.zip "Download Market_Replay_-_36.zip")(131.58 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/466184)**

![Neural networks made easy (Part 68): Offline Preference-guided Policy Optimization](https://c.mql5.com/2/62/midjourney_image_13912_49_444__1-logo.png)[Neural networks made easy (Part 68): Offline Preference-guided Policy Optimization](https://www.mql5.com/en/articles/13912)

Since the first articles devoted to reinforcement learning, we have in one way or another touched upon 2 problems: exploring the environment and determining the reward function. Recent articles have been devoted to the problem of exploration in offline learning. In this article, I would like to introduce you to an algorithm whose authors completely eliminated the reward function.

![How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://c.mql5.com/2/76/How_to_build_and_optimize_a_volatility-based_trading_system_gChaikin_Volatility_-_CHVz____LOGO.png)[How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)

In this article, we will provide another volatility-based indicator named Chaikin Volatility. We will understand how to build a custom indicator after identifying how it can be used and constructed. We will share some simple strategies that can be used and then test them to understand which one can be better.

![Developing a Replay System (Part 37): Paving the Path (I)](https://c.mql5.com/2/61/Desenvolvendo_um_sistema_de_Replay__Parte_37__LOGO.png)[Developing a Replay System (Part 37): Paving the Path (I)](https://www.mql5.com/en/articles/11585)

In this article, we will finally begin to do what we wanted to do much earlier. However, due to the lack of "solid ground", I did not feel confident to present this part publicly. Now I have the basis to do this. I suggest that you focus as much as possible on understanding the content of this article. I mean not simply reading it. I want to emphasize that if you do not understand this article, you can completely give up hope of understanding the content of the following ones.

![Population optimization algorithms: Micro Artificial immune system (Micro-AIS)](https://c.mql5.com/2/64/Bacterial_Foraging_Optimization_-_Genetic_Algorithmi_BFO-GA____LOGO.png)[Population optimization algorithms: Micro Artificial immune system (Micro-AIS)](https://www.mql5.com/en/articles/13951)

The article considers an optimization method based on the principles of the body's immune system - Micro Artificial Immune System (Micro-AIS) - a modification of AIS. Micro-AIS uses a simpler model of the immune system and simple immune information processing operations. The article also discusses the advantages and disadvantages of Micro-AIS compared to conventional AIS.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kcwftxycqhfjpootefbmkcswahqxaliz&ssn=1769184974569230165&ssn_dr=0&ssn_sr=0&fv_date=1769184974&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11510&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2036)%3A%20Making%20Adjustments%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918497465187953&fz_uniq=5070151074427375822&sv=2552)

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