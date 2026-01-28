---
title: Developing a Replay System — Market simulation (Part 02): First experiments (II)
url: https://www.mql5.com/en/articles/10551
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T19:25:08.176062
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/10551&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070267893242860258)

MetaTrader 5 / Tester


### Introduction

In the previous article " [Developing a Replay System — Market simulation (Part 01): First experiments (I)](https://www.mql5.com/en/articles/10543)", we have seen some limitations when trying to create an event system with a short execution time to generate an adequate market simulation. It became clear that it was impossible to get less than 10 milliseconds with this approach. In many cases, this time is quite low. However, if you study the files attached to the article, you will see that 10 milliseconds is not a low enough time period. Is there any other method that would allow us to achieve the desired time of 1 or 2 milliseconds?

Before considering anything connected with the use of time in the range of a few milliseconds, it's important to remind everyone that this is not an easy task. The fact is that the timer provided by the operating system itself cannot reach these values. Therefore, this is a big, if not GIANT, problem. In this article, I will try to answer this question and to show how you can try to solve it by going beyond the time limit set by the operating system. I know that many people think that a modern processor can perform billions of calculations per second. However, it is one thing when the processor performs calculations, and quite another question is whether all the processes inside the computer can cope with the required tasks. Please note that we are trying to use exclusively MQL5 for this, without using any external code or DLL. We are only using pure MQL5.

### Planning

To verify this, we will have to make some changes to the methodology. If this process works out, we won't have to mess with the replay creation system again. We will focus on other questions that will help us conduct research or training using real tick values or simulated values. The way to assemble 1-minute bars remains the same. This will be the main focus of this article.

We are going to use a maximally generic approach, and the best way I have found is to use a client-server-like system. I already explained this technique in my earlier article " [Developing a trading Expert Advisor from scratch (Part 16): Accessing data on the web (II)](https://www.mql5.com/en/articles/10442)". In that article, I showed three ways to transmit information within MetaTrader 5. Here we will use one of these methods, namely SERVICE. So, Market Replay will become a MetaTrader 5 service.

You might be thinking that I am going to create everything from scratch. But why would I do such a thing? Basically, the system is already running, however it does not reach the desired 1 minute time. You might be asking: "Do you think that changing the system to a service will solve this problem?" In fact, simply replacing the system with a service will not solve our problem. But if we isolate the creation of 1-minute bars from the rest of the EA system from the very beginning, then we will have less work later on, because the EA itself will cause a slight delay in the execution of bar construction. I will explain the reasons for this later.

Do you understand now why we are using a service? It is much more practical than other methods discussed above. We will be able to control it in the same way as I explained in the article about how to exchange messages between an EA and a service: [Developing a trading EA from scratch (Part 17): Accessing data on the web (III)](https://www.mql5.com/en/articles/10447). But here we will not focus on generating this control, we only want the service to generate the bars that will be placed on the chart. To make things more interesting, we are going to use the platform in a more creative way, not just using an EA and a service.

Just as a reminder, in the last attempt to reduce the time, we got the following result:

![](https://c.mql5.com/2/45/001__6.png)

This was the best time we had. Here we will crush this time right away. However, I don't want you to get totally attached to these values or to the tests shown here. This series of articles related to the creation of the replay/simulator system, is already at a much more advanced stage in which I changed some concepts several times in order to actually get the system to work as expected. Even if at this point it all seems to be adequate, deep down here I made some mistakes related to timing tests. Such errors, or misconceptions, are not easy to notice in such an early system. As this series of articles develops, you will notice that this timing-related issue is much more complex and that it involves much more than just getting the CPU and the MetaTrader 5 platform provide a certain amount of data on the chart so that you could have an immersion in the replay/simulator system.

So don't take everything you see here literally. Follow this series of articles because what we are going to do here is not simple or easy to do.

### Implementation

Let's start by creating the foundation of our system. These include:

1. Service for creating 1-minute bars
2. Script to start the service
3. EA for simulation (this will be discussed later)

### Defining the Market Replay Service

To properly work with the service, we need to update our C\_Replay class. But these changes are very small, so we won't go into detail. Basically, these are the return codes. However, there is one point that is worth noting separately, as it implements something additional. The code is as follows:

```
#define macroGetMin(A)  (int)((A - (A - ((A % 3600) - (A % 60)))) / 60)
                int Event_OnTime(void)
                        {
                                bool isNew;
                                int mili;
                                static datetime _dt = 0;

                                if (m_ReplayCount >= m_ArrayCount) return -1;
                                if (m_dt == 0)
                                {
                                        m_Rate[0].close = m_Rate[0].open =  m_Rate[0].high = m_Rate[0].low = m_ArrayInfoTicks[m_ReplayCount].Last;
                                        m_Rate[0].tick_volume = 0;
                                        m_Rate[0].time = m_ArrayInfoTicks[m_ReplayCount].dt - 60;
                                        CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
                                        _dt = TimeLocal();
                                }
                                isNew = m_dt != m_ArrayInfoTicks[m_ReplayCount].dt;
                                m_dt = (isNew ? m_ArrayInfoTicks[m_ReplayCount].dt : m_dt);
                                mili = m_ArrayInfoTicks[m_ReplayCount].milisec;
                                while (mili == m_ArrayInfoTicks[m_ReplayCount].milisec)
                                {
                                        m_Rate[0].close = m_ArrayInfoTicks[m_ReplayCount].Last;
                                        m_Rate[0].open = (isNew ? m_Rate[0].close : m_Rate[0].open);
                                        m_Rate[0].high = (isNew || (m_Rate[0].close > m_Rate[0].high) ? m_Rate[0].close : m_Rate[0].high);
                                        m_Rate[0].low = (isNew || (m_Rate[0].close < m_Rate[0].low) ? m_Rate[0].close : m_Rate[0].low);
                                        m_Rate[0].tick_volume = (isNew ? m_ArrayInfoTicks[m_ReplayCount].Vol : m_Rate[0].tick_volume + m_ArrayInfoTicks[m_ReplayCount].Vol);
                                        isNew = false;
                                        m_ReplayCount++;
                                }
                                m_Rate[0].time = m_dt;
                                CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
                                mili = (m_ArrayInfoTicks[m_ReplayCount].milisec < mili ? m_ArrayInfoTicks[m_ReplayCount].milisec + (1000 - mili) : m_ArrayInfoTicks[m_ReplayCount].milisec - mili);
                                if ((macroGetMin(m_dt) == 1) && (_dt > 0))
                                {
                                        Print("Elapsed time: ", TimeToString(TimeLocal() - _dt, TIME_SECONDS));
                                        _dt = 0;
                                }
                                return (mili < 0 ? 0 : mili);
                        };
#undef macroGetMin
```

The highlighted parts have been added to the source code of the C\_Replay class. What we do is define the delay time, that is, we will use exactly the value obtained in the line, but in milliseconds. Do not forget that this time will not be exact, as it depends on some variables. However, we will try to keep it as close to 1 millisecond as possible.

With these changes in mind, let's look at the service code below:

```
#property service
#property copyright "Daniel Jose"
#property version   "1.00"
//+------------------------------------------------------------------+
#include <Market Replay\C_Replay.mqh>
//+------------------------------------------------------------------+
input string    user01 = "WINZ21_202110220900_202110221759"; //File with ticks
//+------------------------------------------------------------------+
C_Replay Replay;
//+------------------------------------------------------------------+
void OnStart()
{
        ulong t1;
        int delay = 3;

        if (!Replay.CreateSymbolReplay(user01)) return;
        Print("Waiting for permission to start replay ...");
        GlobalVariableTemp(def_GlobalVariable01);
        while (!GlobalVariableCheck(def_SymbolReplay)) Sleep(750);
        Print("Replay service started ...");
        t1 = GetTickCount64();
        while (GlobalVariableCheck(def_SymbolReplay))
        {
                if ((GetTickCount64() - t1) >= (uint)(delay))
                {
                        if ((delay = Replay.Event_OnTime()) < 0) break;
                        t1 = GetTickCount64();
                }
        }
        GlobalVariableDel(def_GlobalVariable01);
        Print("Replay service finished ...");
}
//+------------------------------------------------------------------+
```

The above code is responsible for creating the bars. By placing this code here, we make the replay system function independently: the operation of the MetaTrader 5 platform will hardly affect or be affected by it. So, we can work with other things related to the control system, analysis and simulation of the replay. But this will be done later on.

Now comes an interesting thing: pay attention that the highlighted parts gave the [GetTickCount64](https://www.mql5.com/en/docs/common/gettickcount64) function. This will provide a system equivalent to what we saw in the previous article but with one advantage: here the resolution will drop to a time of 1 millisecond. This precision is not exact, it is approximate, but the level of approximation is very close to what the real market movement would be. This does not depend on the hardware you use. After all, you can even create a loop which would guarantee greater precision, but it would be quite laborious since this time it would depend on the hardware used.

The next thing to do is the following script. Here is its full code:

```
#property copyright "Daniel Jose"
#property version   "1.00"
//+------------------------------------------------------------------+
#include <Market Replay\C_Replay.mqh>
//+------------------------------------------------------------------+
C_Replay Replay;
//+------------------------------------------------------------------+
void OnStart()
{
        Print("Waiting for the Replay System ...");
        while((!GlobalVariableCheck(def_GlobalVariable01)) && (!IsStopped())) Sleep(500);
        if (IsStopped()) return;
        Replay.ViewReplay();
        GlobalVariableTemp(def_SymbolReplay);
        while ((!IsStopped()) && (GlobalVariableCheck(def_GlobalVariable01))) Sleep(500);
        GlobalVariableDel(def_SymbolReplay);
        Print("Replay Script finished...");
        Replay.CloseReplay();
}
//+------------------------------------------------------------------+
```

As you can see, both codes are quite simple. However, they communicate with each other through global variables supported by the platform. Thus, we have the following scheme:

![](https://c.mql5.com/2/45/003__8.png)

These scheme will be maintained by the platform itself. If the script closes, the service will be stopped. If the service stops, then the symbol we are using to execute the replay system will stop receiving data. This makes it super simple and highly sustainable. Any improvements (both in the platform and in the hardware) are automatically reflected in the overall performance. This is not a miracle – it is all achieved due to the small latencies that occur during each operation performed by the service process. Only this will actually affect the system in general, we don't need to worry about the script or the WA that we will develop in the future. Any improvements will only affect the service.

To save you the trouble of testing the system, you can preview the result in the image below. So you, dear reader, won't have to wait a whole minute to see the result on your chart.

![](https://c.mql5.com/2/45/002__5.png)

As you can see, the result is very close to ideal. The extra 9 seconds can be easily eliminated using the system settings. Ideally, the time should be less than 1 minute, which will make it easier to adjust things, since we will only need to add a delay to the system. **It's easier to add latency than to reduce it.**. But if you think that the system time cannot be reduced, let's take a closer look at this.

There is a point that produces a delay in the system, which is in the service. This point that will actually generate a delay is highlighted in the code below. But what if we make this line a comment? What will happen to the system?

```
        t1 = GetTickCount64();
        while (GlobalVariableCheck(def_SymbolReplay))
        {
// ...  COMMENT ...  if ((GetTickCount64() - t1) >= (uint)(delay))
                {
                        if ((delay = Replay.Event_OnTime()) < 0) break;
                        t1 = GetTickCount64();
                }
        }
        GlobalVariableDel(def_GlobalVariable01);
```

The highlighted line will no longer be executed. In this case, I will save you from the need to test the system locally and having to wait one minute again. The execution result is shown in the video below. You can watch it entirely or jump to the part where only the final result is shown. Feel free to make your choice.

Demonstrando Velocidade Maxima - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10551)

MQL5.community

1.91K subscribers

[Demonstrando Velocidade Maxima](https://www.youtube.com/watch?v=-h1oBzpBxVA)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=-h1oBzpBxVA&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10551)

0:00

0:00 / 1:03

•Live

•

That is, the biggest challenge is to properly generate a delay. But the small deviation in time of 1 minute for the bar to be created is not really a problem. Since even on a real account, we don't have the exact time, as there is latency in information transmission. This latency is quite small, but it still exists.

### Maximum speed. Really?

Here we will make one last attempt to make the system operate in less than 1 minute.

When you look at millisecond values, you can notice that sometimes we have a variation of only 1 millisecond between one line and another. But we will be treating everything within the same second. So, we can make a small change to the code. We will add a loop inside it, which may make a very big difference to the overall system.

The changes are shown below:

```
#define macroGetMin(A)  (int)((A - (A - ((A % 3600) - (A % 60)))) / 60)
inline int Event_OnTime(void)
                        {
                                bool isNew;
                                int mili;
                                static datetime _dt = 0;

                                if (m_ReplayCount >= m_ArrayCount) return -1;
                                if (m_dt == 0)
                                {
                                        m_Rate[0].close = m_Rate[0].open =  m_Rate[0].high = m_Rate[0].low = m_ArrayInfoTicks[m_ReplayCount].Last;
                                        m_Rate[0].tick_volume = 0;
                                        m_Rate[0].time = m_ArrayInfoTicks[m_ReplayCount].dt - 60;
                                        CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
                                        _dt = TimeLocal();
                                }
                                isNew = m_dt != m_ArrayInfoTicks[m_ReplayCount].dt;
                                m_dt = (isNew ? m_ArrayInfoTicks[m_ReplayCount].dt : m_dt);
                                mili = m_ArrayInfoTicks[m_ReplayCount].milisec;
                                do
                                {
                                        while (mili == m_ArrayInfoTicks[m_ReplayCount].milisec)
                                        {
                                                m_Rate[0].close = m_ArrayInfoTicks[m_ReplayCount].Last;
                                                m_Rate[0].open = (isNew ? m_Rate[0].close : m_Rate[0].open);
                                                m_Rate[0].high = (isNew || (m_Rate[0].close > m_Rate[0].high) ? m_Rate[0].close : m_Rate[0].high);
                                                m_Rate[0].low = (isNew || (m_Rate[0].close < m_Rate[0].low) ? m_Rate[0].close : m_Rate[0].low);
                                                m_Rate[0].tick_volume = (isNew ? m_ArrayInfoTicks[m_ReplayCount].Vol : m_Rate[0].tick_volume + m_ArrayInfoTicks[m_ReplayCount].Vol);
                                                isNew = false;
                                                m_ReplayCount++;
                                        }
                                        mili++;
                                }while (mili == m_ArrayInfoTicks[m_ReplayCount].milisec);
                                m_Rate[0].time = m_dt;
                                CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
                                mili = (m_ArrayInfoTicks[m_ReplayCount].milisec < mili ? m_ArrayInfoTicks[m_ReplayCount].milisec + (1000 - mili) : m_ArrayInfoTicks[m_ReplayCount].milisec - mili);
                                if ((macroGetMin(m_dt) == 1) && (_dt > 0))
                                {
                                        Print("Elapsed time: ", TimeToString(TimeLocal() - _dt, TIME_SECONDS));
                                        _dt = 0;
                                }
                                return (mili < 0 ? 0 : mili);
                        };
#undef macroGetMin
```

If you notice, we now have an outer loop that does this 1ms test. Since it is very difficult to make a correct adjustment within the system so that we would take advantage of using this single millisecond, maybe it is better to take it out of the play.

We've only made one change. You can see the result in the video below.

VOANDO BAIXO - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10551)

MQL5.community

1.91K subscribers

[VOANDO BAIXO](https://www.youtube.com/watch?v=c6G9vpT1tRo)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 1:08

•Live

•

For those who want something even faster, look at the result:

![](https://c.mql5.com/2/45/004.1__1.png)

I think that's enough. We now have the creation of a 1-minute bar below this time. We can make adjustments to reach the perfect time, adding delays to the system. But we will not do it because the idea is to have a system that would allow us to do simulated studies. Anything close to 1 minute is fine for training and practicing. It doesn't have to be something exact.

### Conclusion

Now we have the basics of the Replay system we are creating, and we can move on to the next points. See that everything has been resolved only by using settings and functions present in the MQL5 language, which proves that it can actually do much more than many people think.

But please note that our work has only just begun. There is still a lot to be done.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10551](https://www.mql5.com/pt/articles/10551)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10551.zip "Download all attachments in the single ZIP archive")

[Replay.zip](https://www.mql5.com/en/articles/download/10551/replay.zip "Download Replay.zip")(10746.69 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/450739)**
(7)


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
21 Jun 2023 at 12:49

**Miguel Carmona [#](https://www.mql5.com/es/forum/448575#comment_47585716) :**

I found a solution.

A linha anterior deve ser adaptadacom base nos dados fornecidos para o programa "C Replay".

Thanks for the suggestion and for your interest in the article. But follow the sequence and you will see what was the solution I found at the time these articles were written. I think you will start to see the market in a different way.

![Triton7](https://c.mql5.com/avatar/avatar_na2.png)

**[Triton7](https://www.mql5.com/en/users/triton7)**
\|
17 Jul 2023 at 23:02

Hi Daniel,

why You using so complicated macro with definition: "(int)((A - (A - ((A % 3600) - (A % 60)))) / 60)" ?

I my opinion "(int)( (A % 3600) / 60)" gives the same result.

Jack

![Rasoul Mojtahedzadeh](https://c.mql5.com/avatar/2015/6/558F004E-DFBD.png)

**[Rasoul Mojtahedzadeh](https://www.mql5.com/en/users/rasoul)**
\|
6 Aug 2023 at 16:36

Nice work! :)


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
7 Aug 2023 at 16:41

**Rasoul Mojtahedzadeh [#](https://www.mql5.com/en/forum/450739#comment_48587815):**

Nice work! :)

Thanks ... 😁👍

![Florida Penguin](https://c.mql5.com/avatar/2021/4/607D6029-9C7D.png)

**[Florida Penguin](https://www.mql5.com/en/users/floridapenguin)**
\|
5 Oct 2023 at 06:20

**MetaQuotes:**

Check out the new article: [Developing a Replay System — Market simulation (Part 02): First experiments (II)](https://www.mql5.com/en/articles/10551).

Author: [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831 "DJ_TLoG_831")

Thank you!

![Category Theory in MQL5 (Part 13): Calendar Events with Database Schemas](https://c.mql5.com/2/56/Category-Theory-p13-avatar.png)[Category Theory in MQL5 (Part 13): Calendar Events with Database Schemas](https://www.mql5.com/en/articles/12950)

This article, that follows Category Theory implementation of Orders in MQL5, considers how database schemas can be incorporated for classification in MQL5. We take an introductory look at how database schema concepts could be married with category theory when identifying trade relevant text(string) information. Calendar events are the focus.

![Developing a Replay System — Market simulation (Part 01): First experiments (I)](https://c.mql5.com/2/52/replay-p1-avatar.png)[Developing a Replay System — Market simulation (Part 01): First experiments (I)](https://www.mql5.com/en/articles/10543)

How about creating a system that would allow us to study the market when it is closed or even to simulate market situations? Here we are going to start a new series of articles in which we will deal with this topic.

![Understanding functions in MQL5 with applications](https://c.mql5.com/2/56/understanding-functions-avatar.png)[Understanding functions in MQL5 with applications](https://www.mql5.com/en/articles/12970)

Functions are critical things in any programming language, it helps developers apply the concept of (DRY) which means do not repeat yourself, and many other benefits. In this article, you will find much more information about functions and how we can create our own functions in MQL5 with simple applications that can be used or called in any system you have to enrich your trading system without complicating things.

![Revisiting an Old Trend Trading Strategy: Two Stochastic oscillators, a MA and Fibonacci](https://c.mql5.com/2/56/tranding_strategy_avatar.png)[Revisiting an Old Trend Trading Strategy: Two Stochastic oscillators, a MA and Fibonacci](https://www.mql5.com/en/articles/12809)

Old trading strategies. This article presents one of the strategies used to follow the trend in a purely technical way. The strategy is purely technical and uses a few technical indicators and tools to deliver signals and targets. The components of the strategy are as follows: A 14-period stochastic oscillator. A 5-period stochastic oscillator. A 200-period moving average. A Fibonacci projection tool (for target setting).

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/10551&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070267893242860258)

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