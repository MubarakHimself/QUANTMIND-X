---
title: Developing a Replay System — Market simulation (Part 08): Locking the indicator
url: https://www.mql5.com/en/articles/10797
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:44:15.649395
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/10797&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062679502074586845)

MetaTrader 5 / Examples


### Introduction

In the previous article, [Developing a Replay System — Market simulation (Part 07): First improvements (II)](https://www.mql5.com/en/articles/10784), we have made some corrections and adjustments. However, there was still an error, as shown in the video attached to that article.

In this text we will look at how to fix this error. While this may seem simple at first, there are several steps we need to follow. The process will be intriguing and interesting. Our goal is to make the indicator apply exclusively to a specific chart and symbol. Even if the user tries, they will not be able to apply the indicator to another chart or open it more than once in one session.

I encourage you to continue reading as the content promises to be very useful.

### Locking the indicator on a specific symbol.

The first step is to link the control indicator to the symbol used to for market replay. This step, although it seems simple, is necessary to develop our main task. Let's see what the indicator code will look like in this context:

```
#property copyright "Daniel Jose"
#property indicator_chart_window
#property indicator_plots 0
//+------------------------------------------------------------------+
#include <Market Replay\C_Controls.mqh>
//+------------------------------------------------------------------+
C_Controls      Control;
//+------------------------------------------------------------------+
int OnInit()
{
        u_Interprocess Info;

        IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
        if (_Symbol != def_SymbolReplay)
        {
                ChartIndicatorDelete(ChartID(), 0, def_ShortName);
                return INIT_FAILED;
        }
        if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.Value = 0;
        Control.Init(Info.s_Infos.isPlay);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
        return rates_total;
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        Control.DispatchMessage(id, lparam, dparam, sparam);
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        switch (reason)
        {
                case REASON_REMOVE:
                case REASON_CHARTCLOSE:
                        if (_Symbol != def_SymbolReplay) break;
                        GlobalVariableDel(def_GlobalVariableReplay);
                        ChartClose(ChartID());
                        break;
        }
}
//+------------------------------------------------------------------+
```

We initially check whether the symbol in question is the one used for market replay. If this is not the case, the indicator will close automatically. Please note that it is important to know the name of the indicator. Therefore, the first function executed during initialization calls our indicator, which allows us to remove it without complications.

Now one important point: when you remove it from the chart, MetaTrader 5 generates the DeInit event. This event triggers the OnDeInit function subject to the **REASON\_REMOVE** event indicating the removal of the indicator from the chart. This is because the symbol is not the same as the one the indicator was designed to use. If we don't re-check and prevent the code from running, the symbol chart will close. However, thanks to our check, it will remain open.

Don't be surprised if the indicator code is different from the code presented in the previous article: the previous text focused on other improvements and fixes. However, after writing the article and code, and recording the video accompanying this article, I realized that although one of the problems was fixed, another remained undetected. That's why I had to change the code.

Despite the changes, we will not detail all the modifications made here. A significant portion had to be removed because it was not effective in providing the locking discussed here. Therefore, the above code is very different from the previous one. However, I believe that the knowledge presented in the previous article may be useful to someone at some point. I saved that article to show that sometimes we all make mistakes, but we should still strive to do things right.

Thus, we have established the first locking step by ensuring that the control indicator only exists on the market replay symbol's chart. However, this measure does not prevent adding more than one indicator to the same chart or to different charts, and this must be adjusted.

### We should avoid using multiple indicators on the same chart.

We've solved one problem, now let's tackle another. There are various solutions presented here depending on what we really want and are willing to do. Personally, I do not see an ideal and final solution to this issue. However, I will try to present an approach that the reader can familiarize with and understand. The most important thing is that the solution will be based exclusively on MQL5. I even considered the possibility of using external coding, but decided to use pure MQL5. The idea of resorting to external coding and using DLLs for locking is tempting, but this would be too easy.

I think we have a lot more to learn in MQL5 before resorting to an external DLL to fill in the gaps that the MQL5 language does not fill. This will provide a solution that looks "cleaner" when using external code. However, but this won't help in understanding MQL5 better. Additionally, this may reinforce the misconception that MetaTrader 5 is a limited platform. Misunderstanding and underutilization of the platform fuels this misconception.

To apply our proposed solution, you will have to make some changes and undo others. The first step is to change the InterProcess.mqh header file so that it has the following structure:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_GlobalVariableReplay        "Replay Infos"
#define def_GlobalVariableIdGraphics    "Replay ID"
#define def_SymbolReplay                "RePlay"
#define def_MaxPosSlider                400
#define def_ShortName                   "Market Replay"
//+------------------------------------------------------------------+
union u_Interprocess
{
        union u_0
        {
                double  df_Value;
                long    IdGraphic;
        }u_Value;
        struct st_0
        {
                bool    isPlay;
                int     iPosShift;
        }s_Infos;
};
//+------------------------------------------------------------------+
```

This may seem a little strange to many of you who are not familiar with programming, but surprisingly, the above structure only uses 8 bytes of memory. You may have noticed that a variable was removed in the structure of the previous article. The reason is that we will no longer use this locking method. We'll take a different approach that's a little more complicated, but much more effective at limiting the control indicator to a single chart. It will be a very specific and defined replay service.

**_NOTE:_** It would be interesting if the developers of the MetaTrader 5 platform and the MQL5 language provided the service with the ability to add an indicator to a specific chart or allowed the service to call and execute a script on the chart. Using scripts, we can add an indicator to a specific chart, but **for now this is not possible with services.** We can open a chart, but we cannot add an indicator to it. When trying to perform this action, an error message is always displayed, even if we use MQL5 functions. At the time of this writing, MetaTrader 5 version is **build 3280.**

**Important note:** At this stage of writing the article, which is a more advanced stage, I was able to achieve this. However, when I was writing this article, I could not find any references that could help in this matter. So follow this replay/simulation series to see how I came up with the solution.

In this context, by running the script below, we will be able to open and add an indicator to the chart:

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{

        ENUM_TIMEFRAMES time = PERIOD_D1;
        string szSymbol = "EURUSD";
        long id = ChartOpen(szSymbol, time);
        ChartRedraw(id);

        ChartIndicatorAdd(id, 0, iCustom(szSymbol, time, "Media Movel.ex5"));
}
```

However, if we turn the same script into a service, we will not get the same result.

```
#property service
//+------------------------------------------------------------------+
void OnStart()
{
        ENUM_TIMEFRAMES time = PERIOD_D1;
        string szSymbol = "EURUSD";
        long id = ChartOpen(szSymbol, time);
        ChartRedraw(id);

        ChartIndicatorAdd(id, 0, iCustom(szSymbol, time, "Media Movel.ex5"));
}
```

Note that the only change here is the compilation property, which now specifies that the compiled code will be a service. Simply turning a script into a service using a reserved word completely changes the way the code works, even if it performs the same task as before. Therefore, you will need to use a template to add an indicator to the chart. If it were possible to add an indicator through a service, we could compile the indicator as an internal service resource. Thus, when opening a chart, it will receive the indicator directly from the service, without the need to mix it with other indicators.

Even if we prevent, as shown above, the addition of an indicator to a chart that is not associated with the replay symbol, the user will be able to insert an indicator onto a chart which has market replay as the symbol. This cannot be allowed. So, once we've made changes to the header file Interprocess.mqh, let's focus on the service code. To be more exact, we will move to the header file C\_Replay.mqh.

In short, here's what we'll do: We'll tell the indicator whether the service is active or not. When it is active, we will indicate which chart is the main one. To do this, we need to modify the code, which will now look as shown below:

```
long ViewReplay(ENUM_TIMEFRAMES arg1)
{
        u_Interprocess info;

        if ((m_IdReplay = ChartFirst()) > 0) do
        {
                if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
                {
                        ChartClose(m_IdReplay);
                        ChartRedraw();
                }
        }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
        info.u_Value.IdGraphic = m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
        ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");
        ChartRedraw(m_IdReplay);
        GlobalVariableDel(def_GlobalVariableIdGraphics);
        GlobalVariableTemp(def_GlobalVariableIdGraphics);
        GlobalVariableSet(def_GlobalVariableIdGraphics, info.u_Value.df_Value);
        return m_IdReplay;
}
```

First we clear the global terminal variable. We then create the same variable again, ensuring that it is now temporary. Then we write into this variable the identifier of the chart opened by the service. Thus, we have already significantly simplified the work with the indicator, since we will only have to analyze this value recorded by the service.

However, we should not forget that when we terminate the service, we will also need to remove the additional global variable we created, as indicated in the code below:

```
void CloseReplay(void)
{
        ArrayFree(m_Ticks.Info);
        ChartClose(m_IdReplay);
        SymbolSelect(def_SymbolReplay, false);
        CustomSymbolDelete(def_SymbolReplay);
        GlobalVariableDel(def_GlobalVariableReplay);
        GlobalVariableDel(def_GlobalVariableIdGraphics);
}
```

With these changes, we can add a new control to the indicator.

```
int OnInit()
{
        u_Interprocess Info;

        IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
        if ((_Symbol != def_SymbolReplay) || (!GlobalVariableCheck(def_GlobalVariableIdGraphics)))
        {
                ChartIndicatorDelete(ChartID(), 0, def_ShortName);
                return INIT_FAILED;
        }
        if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.u_Value.df_Value = 0;
        Control.Init(Info.s_Infos.isPlay);

        return INIT_SUCCEEDED;
}
```

Even if we try to add a control indicator to a market replay symbol, we will not be able to do so unless the replay service creates a global terminal variable. Only if this variable is present, it is possible to run an indicator even on the replay symbol chart. However, this action still does not solve our problem. We need to perform a few more checks.

Next, we will implement a check that will begin linking the indicator to the corresponding chart. The first step is shown below:

```
int OnInit()
{
#define macro_INIT_FAILED { ChartIndicatorDelete(ChartID(), 0, def_ShortName); return INIT_FAILED; }
        u_Interprocess Info;

        IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
        if ((_Symbol != def_SymbolReplay) || (!GlobalVariableCheck(def_GlobalVariableIdGraphics))) macro_INIT_FAILED;
        Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableIdGraphics);
        if (Info.u_Value.IdGraphic != ChartID()) macro_INIT_FAILED;
        if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.u_Value.df_Value = 0;
        Control.Init(Info.s_Infos.isPlay);

        return INIT_SUCCEEDED;

#undef macro_INIT_FAILED
}
```

Due to the frequent repetition of the same code in the initialization function, I decided to define a helper macro. This will allow us to avoid possible errors when writing the code. Now let's move on to the first stage of locking. When the symbol chart is created, we pass this chart's ID via the global terminal variable. This way, we can capture this value to confirm that the control indicator is indeed on the expected chart that should be generated by the replay service. If an attempt is made to add an indicator to a chart that differs from the chart created by the replay service, this action will be denied.

While this helps in a wide variety of situations, we still run into a problem: the ability to add additional control indicators to the same chart. To finally solve this problem, we will use a slightly unconventional approach. Keep in mind that at the time of writing this, the platform is in the version shown in the image below:

![](https://c.mql5.com/2/54/terminal_mt5__1.png)

It's possible that by the time you read this, the platform has undergone significant updates, making the mechanism presented here obsolete. But let's look at the mechanism that allows you to configure the control indicator so that new control indicators cannot be added to the same chart, limiting it to a specific chart. If the indicator is removed, the chart will close and the replay service will also be closed.

Before moving on to the actual code, it's important to note that the mechanism I'm going to use is based on Boolean logic. If you are not familiar with these concepts, I recommend doing some research on the topic as these concepts are fundamental to the creation and development of any code. Some people might think that the simplest solution would be to use a DLL to solve the situation in a more direct way. I partially agree, but there is the disadvantage of creating a solution in an external program.

This will not give us a complete and accurate understanding of the limits of the MQL5 language, so we won't be able to find suggestions for its improvement. Many people claim that C/C++ is the most powerful language, and they are right. However, it did not appear as a single entity; it evolved and gained new capabilities as its developers explored its limits. When these boundaries were reached and the desired functionality could not be realized, new functions were created that made previously unattainable projects feasible. This is why C/C++ has proven to be a reliable language that can handle almost any project.

I am sure that MQL5 has the same qualities and potential as C/C++. We only need to study and test the MQL5 language as much as possible. Then, when these boundaries are reached, developers can offer improvements and new features to MQL5. Over time, it could become an extremely powerful application development language for MetaTrader 5.

To understand what we are actually going to do, we need to understand the current limitations of the language where abstracting certain information is not possible. Pay attention that **I DIDN'T SAY** it is impossible to do this, I mean that we cannot create an abstraction that makes our work easier. These are different concepts. It is one thing to develop the abstraction of data and information, and another thing to be able to manipulate the data in the way we need it. Do not confuse these things.

In C/C++ we can create a data abstraction that allows us to isolate one particular bit in a sequence of bits. This is achieved quite simply, see below:

```
union u01
{

double  value;
        struct st
        {
                ulong info : 63;
                bool signal;
        }c;
}data;
```

Even though this seems strange and silly, we are creating a form of data abstraction that allows us to identify and even change the sign of information. The code is not very useful here and now, but let's consider another point. Let's say we want to send information and we use bits to control something. We could have something like this:

```
struct st
{
        bool PlayPause;
        bool Reservad : 6;
        bool RS_Info;
}ctrl;
```

In this scenario, the abstraction level would help us isolate the bits we actually want to access, making the code easier to read. However, as mentioned already, currently MQL5 does not allow us to use the presented abstraction level. We have to take a different approach because pure abstraction is not possible, but if we understand the limitations of the language we can still manipulate the data. Therefore, we resort to Boolean logic to process data that would otherwise be processed by abstraction. However, the use of Boolean logic makes the program more difficult to interpret. Thus, we move from a high-level system with abstractions to a low-level system in which abstraction is reduced to Boolean logic.

We will return to this discussion later. But you will see that the reason will be more justified than what is shown now. Everything I've mentioned may seem unimportant, but if you look at the final control indicator code, you'll understand what I'm trying to illustrate. Often it is not that MQL5 cannot do certain things. In fact, many programmers do not want to move to a deeper level where data abstraction simply does not exist, which makes it impossible to create certain things.

Below is the complete and comprehensive code for the control indicator at the current stage of development. The code makes it possible to lock the indicator on the chart and prohibit the user from adding other control indicators in one MetaTrader 5 session.

```
#property copyright "Daniel Jose"
#property description "This indicator cannot be used\noutside of the market replay service."
#property indicator_chart_window
#property indicator_plots 0
//+------------------------------------------------------------------+
#include <Market Replay\C_Controls.mqh>
//+------------------------------------------------------------------+
C_Controls      Control;
//+------------------------------------------------------------------+
#define def_BitShift ((sizeof(ulong) * 8) - 1)
//+------------------------------------------------------------------+
int OnInit()
{
#define macro_INIT_FAILED { ChartIndicatorDelete(id, 0, def_ShortName); return INIT_FAILED; }
        u_Interprocess Info;
        long id = ChartID();
        ulong ul = 1;

        ul <<= def_BitShift;
        IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
        if ((_Symbol != def_SymbolReplay) || (!GlobalVariableCheck(def_GlobalVariableIdGraphics))) macro_INIT_FAILED;
        Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableIdGraphics);
        if (Info.u_Value.IdGraphic != id) macro_INIT_FAILED;
        if ((Info.u_Value.IdGraphic >> def_BitShift) == 1) macro_INIT_FAILED;
        IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName + "Device");
        Info.u_Value.IdGraphic |= ul;
        GlobalVariableSet(def_GlobalVariableIdGraphics, Info.u_Value.df_Value);
        if (GlobalVariableCheck(def_GlobalVariableReplay)) Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableReplay); else Info.u_Value.df_Value = 0;
        Control.Init(Info.s_Infos.isPlay);

        return INIT_SUCCEEDED;

#undef macro_INIT_FAILED
}
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
        return rates_total;
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        Control.DispatchMessage(id, lparam, dparam, sparam);
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        u_Interprocess Info;
        ulong ul = 1;

        switch (reason)
        {
                case REASON_CHARTCHANGE:
                        ul <<= def_BitShift;
                        Info.u_Value.df_Value = GlobalVariableGet(def_GlobalVariableIdGraphics);
                        Info.u_Value.IdGraphic ^= ul;
                        GlobalVariableSet(def_GlobalVariableIdGraphics, Info.u_Value.df_Value);
                        break;
                case REASON_REMOVE:
                case REASON_CHARTCLOSE:
                        if (_Symbol != def_SymbolReplay) break;
                        GlobalVariableDel(def_GlobalVariableReplay);
                        ChartClose(ChartID());
                        Control.Finish();
                        break;
        }
}
//+------------------------------------------------------------------+
```

Do you have trouble understanding what's really going on? Can't get how our control indicator works? Well, that's the points. There is no abstraction in this code that makes it easy to understand. This is because MQL5 does not allow us to achieve the same level of abstraction that C/C++ provides. Therefore, we are forced to resort to Boolean logic, which, although more complex, allows us to manipulate data and achieve the same results that we would obtain using abstraction.

If you look closely, you will see that the double type requires 8 bytes to store its value. Likewise, a long (signed) or ulong (unsigned) type also takes up the same 8 bytes. Considering that the chart ID obtained via **ChartID** returns the long type, we have 1 extra bit, which is precisely used to indicate the sign. What we will do is use this particular bit to lock the indicator on the chart. And we will manipulate the indicator name to prevent adding another control indicator to the same chart. How? Follow the explanation.

First we determine how many bits we will have and which ones we will work with. So, regardless of whether we are using a 64-bit, 32-bit or 128-bit system, this definition will use the appropriate type and length. Even though we know it's 64-bit, I want the settings to be flexible rather than static. So we subtract 1 from this value, thus isolating the sign bit of the long.

Next, we activate the least significant bit of these 64 bits. Having done this, we will get the value 1, which will be our starting point. Next, we perform a 63-bit shift to the left, resulting in the value 0x8000000000000000000000000, where the most significant bit has a value of 1, that is, true. This step could be avoided if we set this value directly, but the risk of entering it incorrectly is high. By doing this we minimize the possibility of error.

Once we get this value, we have two options. The first one is to lock the system. The second option is to unlock the system allowing MetaTrader 5 to reapply the indicator on the chart as soon as necessary. The second option is simpler, so let's look at it first.

To unlock the system, we take the value of the global terminal variable which contains the chart ID and perform an XOR operation on this value, so as to keep all bits except the most significant one. Ideally, instead of an XOR operation, we should do a NOT operation followed by an AND operation. This would remove any information contained in the most significant bit. However, since this operation only occurs when the given bit already contains some information, I don't see any problem with using the XOR operation. If you have problems, replace the XOR operation with the following line:

```
Info.u_Value.IdGraphic &= (~ul);
```

In any case, we have achieved our goal: resetting the most significant bit. This way we can store the value back into a global terminal variable before MetaTrader 5 attempts to return the control indicator to the chart. This completes the simplest part of the locking system, now let's move on to the more complex part.

The first thing to do at this stage is to check whether the chat symbol matches the replay symbol and whether the replay service is functioning. If any of these conditions are not met, the indicator will be removed from the chart. We then capture the value contained in the global terminal variable which provides the ID of the chart that the replay service created. Next, we compare this value with the ID of the chart window: if they are different, the indicator is also deleted. Next, we take the resulting value and shift it 63 bits to the right, checking whether this bit is active. If so, the identifier is removed again.

While this may seem like enough, there is another problem we need to solve. This particular problem added some work for me while keeping everything within MQL5. I am telling this because I had to add a specific line to the code. Without it, every time I tried to prevent the system from adding an indicator to the chart, it still added one. Even if it didn't cause problems, it remained visible in the indicator window, which annoyed me. It was then that I had the idea to change the name of the indicator, but so that only the first indicator would receive this new name. This indicator is generated along with the chart at the time it is created by the service.

When a template is applied to the chart, the indicator is activated automatically. Speaking of templates, there is one more point. But first, let's finish this explanation. Finally, after all these steps, we perform an OR operation and save the result in the global terminal variable to lock the control indicator. To conclude this topic, one more amendment needs to be made. All the work we have done would have been useless and would not have functioned correctly if we had not made the final change. I could skip this information, boasting for having done something what many consider impossible. Then, if anyone tried to do the same, they would probably fail. But I'm not here to brag or claim that I'm indispensable. I'm not looking for such recognition, on the contrary, I want to show that it is possible to go beyond what many people think is possible, and that sometimes the solution to a problem can be found in unexpected places.

If you follow all the instructions provided and try to execute the entire process, you will find that you can succeed in almost all aspects except one. No matter how hard you try, you would not be able to prevent new control indicators from being added to the chart created by the replay service. You might be asking: "But why? I followed all the steps and even added a special line for this. Are you joking?"

I'd like to say this is a joke, but it's not. In fact, you will not be able to avoid this situation. This is due to a "fault" (note the quotes) in the system. I'm not sure what exactly is causing this or how it could happen, but take a look at the following code:

```
long ViewReplay(ENUM_TIMEFRAMES arg1)
{
        u_Interprocess info;

        if ((m_IdReplay = ChartFirst()) > 0) do
        {
                if (ChartSymbol(m_IdReplay) == def_SymbolReplay)
                {
                        ChartClose(m_IdReplay);
                        ChartRedraw();
                }
        }while ((m_IdReplay = ChartNext(m_IdReplay)) > 0);
        info.u_Value.IdGraphic = m_IdReplay = ChartOpen(def_SymbolReplay, arg1);
        ChartApplyTemplate(m_IdReplay, "Market Replay.tpl");
        ChartRedraw(m_IdReplay);
        GlobalVariableDel(def_GlobalVariableIdGraphics);
        GlobalVariableTemp(def_GlobalVariableIdGraphics);
        GlobalVariableSet(def_GlobalVariableIdGraphics, info.u_Value.df_Value);
        return m_IdReplay;
}
```

Something strange happens when this line is executed. I racked my brain for a long time until I decided to open the template to understand how it works. That's when I realized that we should force the system not to use a template, but to automatically create and apply an indicator to the chart, as I explained at the beginning of the article. I showed there that the use of a service does not allow us to add indicators to the chart, while charts can only be added if we use scripts. The system worked, but when I used the template, it didn't work.

I searched hard to find out what the problem was. I didn't find anyone who had the answer. However, using techniques and skills gained from years of programming, I identified the problem. If you look at the previous articles and data in the template file, you can find the following data:

```
<indicator>
name=Custom Indicator
path=Indicators\Market Replay.ex5
apply=0
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1
```

At first glance, it may seem that everything is correct, and in fact this is so. However, by changing something in the previous fragment, the system begins to work as expected, managing to lock the control indicator and not allowing others to be added.

The corrected version is shown below:

```
<indicator>
name=Custom Indicator
path=Indicators\Market Replay.ex5
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1
```

I'm not going to show you exactly the changed point. I want you to see it and try to understand. But don't worry, the attached code version provides proper system operation.

### Conclusion

As I mentioned earlier, I could skip this information, and when someone asked me why they couldn't get the system to work, I could act like I was a better programmer. But I don't like to brag. I want people to learn, understand, and feel motivated to find solutions. Whenever possible, share your knowledge, because this is how we contribute to evolution. Hiding knowledge is not a sign of superiority, but of fear or lack of confidence.

I have overcome this phase. That's why I am explaining how to work. I hope many will be inspired to do the same. Hugs to everyone and see you in the next article. Our work has just begun.

Narrado 08 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10797)

MQL5.community

1.91K subscribers

[Narrado 08](https://www.youtube.com/watch?v=eoAffPabp98)

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

0:00 / 2:22

•Live

•

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10797](https://www.mql5.com/pt/articles/10797)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10797.zip "Download all attachments in the single ZIP archive")

[Market\_Replay.zip](https://www.mql5.com/en/articles/download/10797/market_replay.zip "Download Market_Replay.zip")(13058.5 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/455138)**

![Neural networks made easy (Part 39): Go-Explore, a different approach to exploration](https://c.mql5.com/2/54/NN_39_Go_Explore_Avatar.png)[Neural networks made easy (Part 39): Go-Explore, a different approach to exploration](https://www.mql5.com/en/articles/12558)

We continue studying the environment in reinforcement learning models. And in this article we will look at another algorithm – Go-Explore, which allows you to effectively explore the environment at the model training stage.

![Developing a Replay System — Market simulation (Part 07): First improvements (II)](https://c.mql5.com/2/54/replay-p7-avatar.png)[Developing a Replay System — Market simulation (Part 07): First improvements (II)](https://www.mql5.com/en/articles/10784)

In the previous article, we made some fixes and added tests to our replication system to ensure the best possible stability. We also started creating and using a configuration file for this system.

![Category Theory in MQL5 (Part 22): A different look at Moving Averages](https://c.mql5.com/2/58/Category-Theory-p22-avatar.png)[Category Theory in MQL5 (Part 22): A different look at Moving Averages](https://www.mql5.com/en/articles/13416)

In this article we attempt to simplify our illustration of concepts covered in these series by dwelling on just one indicator, the most common and probably the easiest to understand. The moving average. In doing so we consider significance and possible applications of vertical natural transformations.

![Neural networks made easy (Part 38): Self-Supervised Exploration via Disagreement](https://c.mql5.com/2/54/self_supervised_exploration_via_disagreement_038_avatar.png)[Neural networks made easy (Part 38): Self-Supervised Exploration via Disagreement](https://www.mql5.com/en/articles/12508)

One of the key problems within reinforcement learning is environmental exploration. Previously, we have already seen the research method based on Intrinsic Curiosity. Today I propose to look at another algorithm: Exploration via Disagreement.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=uutixtizrxqevgsdxqccizoqkwkerfxs&ssn=1769157854560524727&ssn_dr=0&ssn_sr=0&fv_date=1769157854&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10797&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20%E2%80%94%20Market%20simulation%20(Part%2008)%3A%20Locking%20the%20indicator%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915785429457952&fz_uniq=5062679502074586845&sv=2552)

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