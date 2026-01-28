---
title: Developing a Replay System — Market simulation (Part 01): First experiments (I)
url: https://www.mql5.com/en/articles/10543
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T19:25:17.883387
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/10543&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070269821683176172)

MetaTrader 5 / Tester


### Introduction

When writing the series of articles " [Developing a trading EA from scratch](https://www.mql5.com/en/articles/10085)", I encountered several moments that made me realize that it was possible to do much more than what was done via MQL5 programing. One of such moments was when I developed a graphic [Times & Trade](https://www.mql5.com/en/articles/10410) system. In that article, I wondered if it was possible to go beyond the previously built things.

One of the most common complaints from novice traders is the lack of certain features in the MetaTrader 5 platform. Among these features, there is one that, in my opinion, makes sense: a market simulation or replay system. It would be good for new market participants to have some kind of mechanism or tool that would allow them to test, verify, or even study assets. One of such tools is the replay and simulation system.

MetaTrader 5 does not include this feature in the standard installation package. Thus, it is up to each user to decide how to conduct such studies. However, in MetaTrader 5 you can find solutions for so many tasks since this platform is very functional. But for you to actually be able to use it to its fullest potential, you need to have good programming experience. I don't necessarily mean MQL5 programming but programming in general.

If you don't have much experience in this area, you will just stick to the basics. You will be devoid of more adequate means or better ways to develop in the market (in terms of becoming an exceptional trader). So, unless you have a good level of programming knowledge, you won't really be able to use everything MetaTrader 5 has to offer. Even experienced programmers may still not be interested in creating certain types of programs or applications for MetaTrader 5.

The fact is that few people can create a workable system for beginners. There are even some free proposals to create a market replay system. But in my opinion, they don't really exploit the features that MQL5 provides. They often require the use of external DLLs with closed code. I think it is not a good idea. Even more so because you don't really know the origin, or the content present in such DLLs, which puts the entire system at risk.

I don't know how many articles this series will include, but it will be about developing a working replay. I'll show you how you can create code to implement this replay. But that's not all. We are also going to develop a system that will allow us to simulate any market situation no matter how strange or rare it is.

A curious fact is that many people, when they talk about trade quantity, actually do not have the real idea of what they are talking about, because there is no practical way of doing studies involving this type of thing. But if you understand the concepts that I will describe in this series, you will be able to transform MetaTrader 5 into a quantitative analysis system. Thus, the possibilities will extend far beyond what I will actually expose here.

In order not to get too repetitive and tiring, I will treat the system just like a replay. Although the correct term is replay/simulation, because in addition to analyzing past movements, we can also develop our own movements to study them. So, don't treat this system just as a market replay, but as a market simulator, or even a market "Game", since it will involve a lot of game programming as well. This type of programming involved in games will be necessary at some points. But we will see this step by step while developing and enhancing the system.

### Planning

First, we need to understand what we are dealing with. It may seem strange, but do you really know what you want to achieve when you use a replay/simulation system?

There are some pretty tricky issues when it comes to creating a market replay. One of them, and perhaps the main one, is the lifetime of assets and information about them. If you do not understand this, it is important that you understand the following: The trading system records all the information, tick by tick, for each executed trade for all assets, one by one. But do you realize how much data they represent? Have you ever thought about how long it will take to organize and sort all the assets?

Well, some typical assets can contain about 80 MBytes of data in their daily movements. In some cases, it may be a little more or a little less. This is only for 1 single asset in a single day. Now think about having to store the same asset for 1 month, 1 year, 10 years... Or who knows, forever. Think of the enormous amount of data that will have to be stored and then retrieved. Because if you just save them on the disk, soon you won't be able to find anything. There is a phrase that identifies this well:

**_The bigger the space, the bigger the mess_**.

To make things easier, after a while the data is compressed into 1-minute bars, which contain the minimum necessary information, so that we can do some kind of study. But when that bar is actually created, the ticks that built it disappear and are no longer accessible. After that, it is no longer possible to do a real market replay. From this moment on, what we have is a simulator. Since the real movement is no longer accessible, we will have to create some way to simulate it based on some plausible market movement.

To understand, see the figures below:

![](https://c.mql5.com/2/45/001__5.png)![](https://c.mql5.com/2/45/04__4.png)![](https://c.mql5.com/2/45/002.1__1.png)![](https://c.mql5.com/2/45/04__5.png)![](https://c.mql5.com/2/45/002.2.png)

The sequence above shows how data is lost over time. The image on the left shows the actual tick values. When the data is compressed, we get the image in the center. Based on it, we won't be able to get the left values. It's IMPOSSIBLE to do that. But we can create something like the image on the right, where we will be simulating market movements based on the knowledge about how the market normally moves. However, it looks nothing like the original image.

Keep this in mind when working with the replay. If you don't have the original data, you won't be able to do a real study. You will only be able to make a statistical study, which may or may not be close to a possible real movement. Always remember this. Throughout this sequence, we'll be exploring more how to do this. But this will happen little by little.

With that, let's move on to the really challenging part: Implement a replay system.

### Implementation

This part, although it seems simple, is quite complicated since there are issues and limitations involved in terms of hardware and other problems in the software part. So, we have to try to create something, at least the most basic, functional and acceptable. It won't do any good to try to do something more complex if the bases are weak.

Our main and biggest problem, oddly enough, is time. Time is a big or even huge problem to be overcome.

In the attachment, I will always leave (in this first phase) at least 2 REAL sets of ticks of any asset for any past period. This data can no longer be obtained as the data has been lost and cannot be downloaded. This will help us to study every detail. But still, you can also create your own REAL tick base.

### Create your own database

Fortunately, MetaTrader 5 provides the means to do this. This is quite simple to do, but you have to do this steadily, otherwise the values can be lost, and it will no longer be possible to complete this task.

To do this, open MetaTrader 5 and press the default shortcut keys: CTRL+U. This will open a screen. Specify here the asset as well as data collection start and end dates, press the button to request data, and wait a few minutes. The server will return all the data you need. After that, just export this data and store it carefully, as it's very valuable.

Below is the screen you will use to capture.

![](https://c.mql5.com/2/45/003__7.png)

Although you can create a program to do this, I think it's better to do this manually. There are things that we cannot trust blindly. We have to actually see what is going on, otherwise we will not have the proper confidence in what we are using.

Trust me, this is the easiest part of the whole system that we are going to learn to create. From this point on, things get much more complicated.

### First replay test

Some may think that this will be a simple task, but we will soon disprove this idea. Others may wonder: Why don't we use the MetaTrader 5 Strategy Tester to do the replay? The reason is that it does not allow us to replay as if we were trading in the market. There are limitations and difficulties in replaying through the tester, and as a result, we will not have perfect immersion in the replay, as if we were actually trading the market.

We will face great challenges, but we must take the first steps on this long journey. Let's start with a very simple implementation. For this we need the [OnTime](https://www.mql5.com/en/docs/basis/function/events#ontimer) event which will generate a data flow to create bars (Candlesticks). This event is provided for EAs and indicators, but we should not use indicators in this case, because if a failure occurs, it will jeopardize much more than the replay system. We will start the code as follows:

```
#property copyright "Daniel Jose"
#property icon "Resources\\App.ico"
#property description "Expert Advisor - Market Replay"
//+------------------------------------------------------------------+
int OnInit()
{
        EventSetTimer(60);
        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        EventKillTimer();
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
void OnTimer()
{
}
//+------------------------------------------------------------------+
```

However, the highlighted code is not suitable for our purposes, because in this case the smallest period we can use is 1 second, which is a long time, a very long time. Since market events occur in a much smaller time frame, we need to get down to milliseconds, and for this we will have to use another function: [EventSetMillisecondTimer](https://www.mql5.com/en/docs/eventfunctions/eventsetmillisecondtimer). But we have a problem here.

### Limitations of the EventSetMillisecondTimer function

Let's see the documentation:

**"When working in real-time mode, timer events are generated no more than 1 time in 10-16 milliseconds due to hardware limitations."**

This may not be a problem, but we need to run various checks to verify what actually happens. So, let's create some simple code to check the results.

Let's start with the EA code below:

```
#property copyright "Daniel Jose"
#property icon "Resources\\App.ico"
#property description "Expert Advisor - Market Replay"
//+------------------------------------------------------------------+
#include "Include\\C_Replay.mqh"
//+------------------------------------------------------------------+
input string user01 = "WINZ21_202110220900_202110221759";       //Tick archive
//+------------------------------------------------------------------+
C_Replay        Replay;
//+------------------------------------------------------------------+
int OnInit()
{
        Replay.CreateSymbolReplay(user01);
        EventSetMillisecondTimer(20);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        EventKillTimer();
}
//+------------------------------------------------------------------+
void OnTick() {}
//+------------------------------------------------------------------+
void OnTimer()
{
        Replay.Event_OnTime();
}
//+------------------------------------------------------------------+
```

Note that our OnTime event will occur approximately every 20 milliseconds, as indicated by the highlighted line in the EA code. You may think this is too fast, but is it really so? Let's check it out. Remember that the documentation says that we can't go below 10 to 16 milliseconds. Therefore, it makes no sense to set the value to 1 millisecond, since the event will not be generated during this time.

Pay attention that in the EA code, we have only two external links. Now let's see the class in which these codes are implemented.

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_MaxSizeArray 134217727 // 128 Mbytes of positions
#define def_SymbolReplay "Replay"
//+------------------------------------------------------------------+
class C_Replay
{
};
```

It is important to note that the class has a definition of 128 MB, as indicated in the highlighted point above. This means that the file containing the data of all ticks must not exceed this size. You can increase this size if you want or need it, but personally I had no problems with this value.

The next line specifies the name of the asset that will be used as a replay. Pretty creative of me to name the asset REPLAY, isn't it? 😂 Well, let's continue with the study of the class. The next code to discuss is shown below:

```
void CreateSymbolReplay(const string FileTicksCSV)
{
        SymbolSelect(def_SymbolReplay, false);
        CustomSymbolDelete(def_SymbolReplay);
        CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\Replay\\%s", def_SymbolReplay), _Symbol);
        CustomRatesDelete(def_SymbolReplay, 0, LONG_MAX);
        CustomTicksDelete(def_SymbolReplay, 0, LONG_MAX);
        SymbolSelect(def_SymbolReplay, true);
        m_IdReplay = ChartOpen(def_SymbolReplay, PERIOD_M1);
        LoadFile(FileTicksCSV);
        Print("Running speed test.");
}
```

The two highlighted lines do some pretty interesting things. For those who don't know, the [CustomSymbolCreate](https://www.mql5.com/en/docs/customsymbols/customsymbolcreate) function creates a custom symbol. In this case, we can adjust a few things, but since this is just a test, I won't go into it for now. [ChartOpen](https://www.mql5.com/en/docs/chart_operations/chartopen) will open the chart of our custom symbol, which in this case will be the replay. It's all very nice, but we need to load our replay from the file, and this is done by the following function.

```
#define macroRemoveSec(A) (A - (A % 60))
                void LoadFile(const string szFileName)
                        {
                                int file;
                                string szInfo;
                                double last;
                                long    vol;
                                uchar flag;

                                if ((file = FileOpen("Market Replay\\Ticks\\" + szFileName + ".csv", FILE_CSV | FILE_READ | FILE_ANSI)) != INVALID_HANDLE)
                                {
                                        ArrayResize(m_ArrayInfoTicks, def_MaxSizeArray);
                                        m_ArrayCount = 0;
                                        last = 0;
                                        vol = 0;
                                        for (int c0 = 0; c0 < 7; c0++) FileReadString(file);
                                        Print("Loading data for Replay.\nPlease wait ....");
                                        while ((!FileIsEnding(file)) && (m_ArrayCount < def_MaxSizeArray))
                                        {
                                                szInfo = FileReadString(file);
                                                szInfo += " " + FileReadString(file);
                                                m_ArrayInfoTicks[m_ArrayCount].dt = macroRemoveSec(StringToTime(StringSubstr(szInfo, 0, 19)));
                                                m_ArrayInfoTicks[m_ArrayCount].milisec = (int)StringToInteger(StringSubstr(szInfo, 20, 3));
                                                m_ArrayInfoTicks[m_ArrayCount].Bid = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Ask = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Last = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Vol = StringToInteger(FileReadString(file));
                                                flag = m_ArrayInfoTicks[m_ArrayCount].flag = (uchar)StringToInteger(FileReadString(file));
                                                if (((flag & TICK_FLAG_ASK) == TICK_FLAG_ASK) || ((flag & TICK_FLAG_BID) == TICK_FLAG_BID)) continue;
                                                m_ArrayCount++;
                                        }
                                        FileClose(file);
                                        Print("Loading completed.\nReplay started.");
                                        return;
                                }
                                Print("Failed to access tick data file.");
                                ExpertRemove();
                        };
#undef macroRemoveSec
```

This function will load all tick data, line by line. If the file does not exist or cannot be accessed, [ExpertRemove](https://www.mql5.com/en/docs/common/expertremove) will close the EA.

All data will be temporarily stored in memory to speed up further processing. This is because you may be using a disk drive, which is likely to be slower than system memory. Therefore, it is worth making sure from the very beginning that all data is present.

But there is something rather interesting in the above code: the [FileReadString](https://www.mql5.com/en/docs/files/filereadstring) function. It reads the data until it finds some delimiter. It is interesting to see that when we look at the binary data of the tick file generated by MetaTrader 5 and saved in CSV format, as explained at the beginning of the article, we get the following result.

![](https://c.mql5.com/2/45/005__3.png)

The yellow area is the file header which shows us the organization of the internal structure that will follow. The green area represents the first data row. Now let's look at the blue dots (they are delimiters) present in this format. 0D and 0A denote a new line, and 09 denotes a tab (TAB key). When we use the FileReadString function, we don't need to accumulate data to test it. The function will do that itself. All we have to do is convert data to the required type. Let's see the next code part.

```
if (((flag & TICK_FLAG_ASK) == TICK_FLAG_ASK) || ((flag & TICK_FLAG_BID) == TICK_FLAG_BID)) continue;
```

This code prevents unnecessary data from appearing in our data matrix, but why am I filtering out these values? Because they are not suitable for a replay. If you want to use these values, you can let them pass, but you will have to filter them later, when creating the bars. So, I prefer to filter them here.

Below we show the last routine in our test system:

```
#define macroGetMin(A)  (int)((A - (A - ((A % 3600) - (A % 60)))) / 60)
                void Event_OnTime(void)
                        {
                                bool isNew;
                                static datetime _dt = 0;

                                if (m_ReplayCount >= m_ArrayCount) return;
                                if (m_dt == 0)
                                {
                                        m_Rate[0].close = m_Rate[0].open =  m_Rate[0].high = m_Rate[0].low = m_ArrayInfoTicks[m_ReplayCount].Last;
                                        m_Rate[0].tick_volume = 0;
                                        _dt = TimeLocal();
                                }
                                isNew = m_dt != m_ArrayInfoTicks[m_ReplayCount].dt;
                                m_dt = (isNew ? m_ArrayInfoTicks[m_ReplayCount].dt : m_dt);
                                m_Rate[0].close = m_ArrayInfoTicks[m_ReplayCount].Last;
                                m_Rate[0].open = (isNew ? m_Rate[0].close : m_Rate[0].open);
                                m_Rate[0].high = (isNew || (m_Rate[0].close > m_Rate[0].high) ? m_Rate[0].close : m_Rate[0].high);
                                m_Rate[0].low = (isNew || (m_Rate[0].close < m_Rate[0].low) ? m_Rate[0].close : m_Rate[0].low);
                                m_Rate[0].tick_volume = (isNew ? m_ArrayInfoTicks[m_ReplayCount].Vol : m_Rate[0].tick_volume + m_ArrayInfoTicks[m_ReplayCount].Vol);
                                m_Rate[0].time = m_dt;
                                CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
                                m_ReplayCount++;
                                if ((macroGetMin(m_dt) == 1) && (_dt > 0))
                                {
                                        Print(TimeToString(_dt, TIME_DATE | TIME_SECONDS), " ---- ", TimeToString(TimeLocal(), TIME_DATE | TIME_SECONDS));
                                        _dt = 0;
                                }
                        };
#undef macroGetMin
```

This code will create bars with a period of 1 minute, which is the minimum platform requirement for creating any other chart period. The highlighted parts are not part of the code itself but are useful for analyzing the 1-minute bar. We need to check if it is really created within this timeframe. Because if it takes much longer than 1 minute to be created, we will have to do something about it. If it is created in less than a minute, this may indicate that the system is viable right away.

After executing this system, we will get the result that we show in the following video:

Replay 01 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10543)

MQL5.community

1.91K subscribers

[Replay 01](https://www.youtube.com/watch?v=ovIrjRpbs9s)

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

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=ovIrjRpbs9s&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10543)

0:00

0:00 / 3:33

•Live

•

Some feel that the timing is much longer than expected, but we can make some improvements to the code, and maybe this will make a difference.

### Improving the code

Despite the huge delay, we might be able to improve things a little and help the system perform a little closer to expectations. But I don't really believe in miracles. We know the limitation of the EventSetMillisecondTimer function, and the problem isn't due to MQL5 but it's a hardware limitation. However, let's see if we can help the system.

If you look at the data, you will see that there are several moments when the system just doesn't move, the price stays still, or it could have happened that the Book absorbs every aggression, and the price just doesn't move. This can be seen in the image below.

![](https://c.mql5.com/2/45/006__3.png)

Notice that we have two different conditions: in one of them, the time and price have not changed. This does not tell us that the data is incorrect, but it tells that not enough time has passed for the millisecond measurement to make a difference. We also have another type of event where the price didn't move, but time moved by only 1 millisecond. In both cases, when we combine information, the difference in bar creation time can be 1 minute. This will avoid additional calls to the creation functions, and every nanosecond saved can make a big difference in the long run. Everything adds up, and little by little a lot is achieved.

To check whether there will be a difference, we will need to check the amount of information being generated. This is a matter of statistics, so it is not exact. A small mistake is acceptable. But the time that can be seen in the video is completely unacceptable for a simulation close to reality.

To verify this, we make the first modification to the code:

```
#define macroRemoveSec(A) (A - (A % 60))
                void LoadFile(const string szFileName)
                        {

// ... Internal code ...
                                        FileClose(file);
                                        Print("Loading completed.\n", m_ArrayCount, " movement positions were generated.\nStarting Replay.");
                                        return;
                                }
                                Print("Failed to access tick data file.");
                                ExpertRemove();
                        };
#undef macroRemoveSec
```

The specified additional part will do it for us. Let's now take a look at the first launch and see what happens. All this can be seen in the image below:

![](https://c.mql5.com/2/45/007__3.png)

Now we have some parameter that allows us to check if the modifications help or not. If we use it, we will see that it took almost 3 minutes to generate 1 minute of data. In other words, the system is very far from acceptable.

Therefore, we will make small improvements to the code:

```
#define macroRemoveSec(A) (A - (A % 60))
                void LoadFile(const string szFileName)
                        {
                                int file;
                                string szInfo;
                                double last;
                                long    vol;
                                uchar flag;


                                if ((file = FileOpen("Market Replay\\Ticks\\" + szFileName + ".csv", FILE_CSV | FILE_READ | FILE_ANSI)) != INVALID_HANDLE)
                                {
                                        ArrayResize(m_ArrayInfoTicks, def_MaxSizeArray);
                                        m_ArrayCount = 0;
                                        last = 0;
                                        vol = 0;
                                        for (int c0 = 0; c0 < 7; c0++) FileReadString(file);
                                        Print("Loading data to Replay.\nPlease wait ....");
                                        while ((!FileIsEnding(file)) && (m_ArrayCount < def_MaxSizeArray))
                                        {
                                                szInfo = FileReadString(file);
                                                szInfo += " " + FileReadString(file);
                                                m_ArrayInfoTicks[m_ArrayCount].dt = macroRemoveSec(StringToTime(StringSubstr(szInfo, 0, 19)));
                                                m_ArrayInfoTicks[m_ArrayCount].milisec = (int)StringToInteger(StringSubstr(szInfo, 20, 3));
                                                m_ArrayInfoTicks[m_ArrayCount].Bid = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Ask = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Last = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Vol = vol + StringToInteger(FileReadString(file));
                                                flag = m_ArrayInfoTicks[m_ArrayCount].flag = (uchar)StringToInteger(FileReadString(file));
                                                if (((flag & TICK_FLAG_ASK) == TICK_FLAG_ASK) || ((flag & TICK_FLAG_BID) == TICK_FLAG_BID)) continue;
                                                if (m_ArrayInfoTicks[m_ArrayCount].Last != last)
                                                {
                                                        last = m_ArrayInfoTicks[m_ArrayCount].Last;
                                                        vol = 0;
                                                        m_ArrayCount++;
                                                }else
                                                        vol += m_ArrayInfoTicks[m_ArrayCount].Vol;
                                        }
                                        FileClose(file);
                                        Print("Loading complete.\n", m_ArrayCount, " movement positions were generated.\nStarting Replay.");
                                        return;
                                }
                                Print("Failed to access tick data file.");
                                ExpertRemove();
                        };
#undef macroRemoveSec
```

Adding the highlighted bold lines greatly improves the results, as seen in the image below:

![](https://c.mql5.com/2/45/008__2.png)

Here we have improved system performance. It may not seem like much, but it still shows that early changes play a decisive role. We have reached an approximate time of 2 minutes 29 seconds to generate a 1 minute bar. In other words, there has been an overall improvement in the system, but while that sounds encouraging, we have a problem that complicates matters. We **can't** reduce the time between events generated by the EventSetMillisecondTimer function, which makes us think about a different approach.

However, a small improvement has been made to the system, as shown below:

```
                void Event_OnTime(void)
                        {
                                bool isNew;
                                static datetime _dt = 0;

                                if (m_ReplayCount >= m_ArrayCount) return;
                                if (m_dt == 0)
                                {
                                        m_Rate[0].close = m_Rate[0].open =  m_Rate[0].high = m_Rate[0].low = m_ArrayInfoTicks[m_ReplayCount].Last;
                                        m_Rate[0].tick_volume = 0;
                                        m_Rate[0].time = m_ArrayInfoTicks[m_ReplayCount].dt - 60;
                                        CustomRatesUpdate(def_SymbolReplay, m_Rate, 1);
                                        _dt = TimeLocal();
                                }

// ... Internal code ....

                        }
```

What the highlighted lines do can be seen on the chart. Without them, the first bar is always cut off, making it difficult to read correctly. But when we add these two lines, the visual representation becomes much nicer, allowing us to properly see the generated bars. This happens from the first bar. It's something very simple, but it makes a lot of difference in the end.

But let's get back to our original question, which is trying to create a suitable system for presenting and creating the bars. Even if it were possible to reduce the time, we wouldn't have an adequate system. In fact, we will have to change the approach. This is why the EA is not the best way to create a replay system But even so, I want to show another thing that may be interesting for you. How much can we actually reduce or improve the creation of a 1-minute bar if we use the shortest possible time to generate the OnTime event? What if when the value does not change within the same 1 minute, we further compress the data into the tick scope? Will it make any difference?

### Going to the extreme

To do this, we need to make the last change to the code. It is shown below:

```
#define macroRemoveSec(A) (A - (A % 60))
#define macroGetMin(A)  (int)((A - (A - ((A % 3600) - (A % 60)))) / 60)
                void LoadFile(const string szFileName)
                        {
                                int file;
                                string szInfo;
                                double last;
                                long    vol;
                                uchar flag;
                                datetime mem_dt = 0;

                                if ((file = FileOpen("Market Replay\\Ticks\\" + szFileName + ".csv", FILE_CSV | FILE_READ | FILE_ANSI)) != INVALID_HANDLE)
                                {
                                        ArrayResize(m_ArrayInfoTicks, def_MaxSizeArray);
                                        m_ArrayCount = 0;
                                        last = 0;
                                        vol = 0;
                                        for (int c0 = 0; c0 < 7; c0++) FileReadString(file);
                                        Print("Loading data to Replay.\nPlease wait ....");
                                        while ((!FileIsEnding(file)) && (m_ArrayCount < def_MaxSizeArray))
                                        {
                                                szInfo = FileReadString(file);
                                                szInfo += " " + FileReadString(file);
                                                m_ArrayInfoTicks[m_ArrayCount].dt = macroRemoveSec(StringToTime(StringSubstr(szInfo, 0, 19)));
                                                m_ArrayInfoTicks[m_ArrayCount].milisec = (int)StringToInteger(StringSubstr(szInfo, 20, 3));
                                                m_ArrayInfoTicks[m_ArrayCount].Bid = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Ask = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Last = StringToDouble(FileReadString(file));
                                                m_ArrayInfoTicks[m_ArrayCount].Vol = vol + StringToInteger(FileReadString(file));
                                                flag = m_ArrayInfoTicks[m_ArrayCount].flag = (uchar)StringToInteger(FileReadString(file));
                                                if (((flag & TICK_FLAG_ASK) == TICK_FLAG_ASK) || ((flag & TICK_FLAG_BID) == TICK_FLAG_BID)) continue;
                                                if ((mem_dt == macroGetMin(m_ArrayInfoTicks[m_ArrayCount].dt)) && (last == m_ArrayInfoTicks[m_ArrayCount].Last)) vol += m_ArrayInfoTicks[m_ArrayCount].Vol; else
                                                {
                                                        mem_dt = macroGetMin(m_ArrayInfoTicks[m_ArrayCount].dt);
                                                        last = m_ArrayInfoTicks[m_ArrayCount].Last;
                                                        vol = 0;
                                                        m_ArrayCount++;
                                                }
                                        }
                                        FileClose(file);
                                        Print("Upload completed.\n", m_ArrayCount, " movement positions were generated.\nStarting Replay.");
                                        return;
                                }
                                Print("Failed to access tick data file.");
                                ExpertRemove();
                        };
#undef macroRemoveSec
#undef macroGetMin
```

The highlighted code fixes a small problem which existed before but which we did not notice. When the price remained unchanged, but there was a transition from one bar to another, it took some time to create a new bar. However, the real problem was that the opening price was different from what was shown on the original chart. So, this has been corrected. Now, if all other parameters are the same or have a small difference in milliseconds, we will only have one saved position.

After that, we can test the system with EventSetMillisecondTimer of 20. The result is as follows:

![](https://c.mql5.com/2/45/009__2.png)

In this case, the result was 2 minutes 34 seconds for a 20 millisecond event... Let's then change the value of the EventSetMillisecondTimer to 10 (which is the minimum value specified in the documentation). Here is the result:

![](https://c.mql5.com/2/45/010__1.png)

In this case, the result was 1 minute 56 seconds for a 10 millisecond event. The result has improved, but it is still far from what we need. And now there is no way to further reduce the time event using the method adopted in this article, since the documentation itself informs us that this is not possible, or we will not have enough stability to be able to take the next step.

### Conclusion

In this article, I have presented the basic principles for those who want to create a Replay/Simulation system. These principles underlie the entire system, but for those with no programming experience, understanding how the MetaTrader 5 platform works can be a daunting task. Seeing how these principles are applied in practice can be a great motivation to start learning programming. Because things only become interesting when we see how they work; just looking at the code is not motivating.

Once you realize what can be done and understand how it works, everything changes. It's like a magic door is opening, revealing a whole new unknown world full of possibilities. You will see how this happens throughout this series. I will develop this system as I create articles, so please be patient. Even when it seems that there is no progress, there is always progress. And knowledge is never too much. Maybe it can make us less happy, but it never hurts.

The attachment contains the two versions that we discussed here. You will also find two real tick files so you can experiment and see how the system behaves on your own hardware. The results won't be much different from what I've shown, but it can be quite interesting to see how a computer handles certain problems, solving them in quite creative ways.

In the next article, we will make some changes to try to achieve a more adequate system. We will also look at another rather interesting solution, which will also be useful for those who are just starting their journey in the programming world. So, the work has just begun.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10543](https://www.mql5.com/pt/articles/10543)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10543.zip "Download all attachments in the single ZIP archive")

[Replay.zip](https://www.mql5.com/en/articles/download/10543/replay.zip "Download Replay.zip")(10910.23 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/450687)**
(6)


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
25 Nov 2023 at 18:09

In addition, you have not solved another problem that allows you to reduce the construction of a minute bar.

For example, the times highlighted in red and blue are the same, where the last price may or may not be the same.

These ticks can be compressed and greatly reduce the time it takes to build a minute bar.

```
 449      2021.10 . 22        09 : 00 : 38.649      107900    107905                    6
 450      2021.10 . 22        09 : 00 : 38.651                      107900    1.00000000        88
 451      2021.10 . 22        09 : 00 : 38.651                      107895    5.00000000        88
 452      2021.10 . 22        09 : 00 : 38.651                      107890    5.00000000        88
 453      2021.10 . 22        09 : 00 : 38.651                      107885    3.00000000        88
 454      2021.10 . 22        09 : 00 : 38.651                      107880    15.00000000       88
 455      2021.10 . 22        09 : 00 : 38.651                      107880    3.00000000        88
 456      2021.10 . 22        09 : 00 : 38.651                      107875    16.00000000       88
 457      2021.10 . 22        09 : 00 : 38.651                      107870    2.00000000        88
 458      2021.10 . 22        09 : 00 : 38.654                      107875    1.00000000        88
 459      2021.10 . 22        09 : 00 : 38.654                      107875    1.00000000        88
 460      2021.10 . 22        09 : 00 : 38.654                      107880    1.00000000        88
 461      2021.10 . 22        09 : 00 : 38.659                      107880    2.00000000        88
 462      2021.10 . 22        09 : 00 : 38.659                      107885    2.00000000        88
 463      2021.10 . 22        09 : 00 : 38.660                      107885    1.00000000        88
 464      2021.10 . 22        09 : 00 : 38.660                      107890    3.00000000        88
 465      2021.10 . 22        09 : 00 : 38.662                      107885    3.00000000        88
 466      2021.10 . 22        09 : 00 : 38.662                      107880    3.00000000        88
 467      2021.10 . 22        09 : 00 : 38.662                      107875    2.00000000        88
 468      2021.10 . 22        09 : 00 : 38.662                      107895    3.00000000        88
 469      2021.10 . 22        09 : 00 : 38.662                      107900    1.00000000        88
 470      2021.10 . 22        09 : 00 : 38.664                      107880    1.00000000        88
```

But, you removed the seconds on line 53.

```

53      m_ArrayInfoTicks[m_ArrayCount].dt = macroRemoveSec ( StringToTime ( StringSubstr (szInfo, 0 , 19 )));
54      m_ArrayInfoTicks[m_ArrayCount].milisec = ( int ) StringToInteger ( StringSubstr (szInfo, 20 , 3 ));
```

And left milliseconds in line 54.

Which makes it impossible for you to perform this compression accurately. There is no binding of milliseconds to seconds - it has been removed.

Of course, there is a low probability that the millisecond value will move to the next second, and even continue in the next ticks. But it is there.

There is a 100% guarantee of accuracy only when [moving](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "MetaTrader 5 Help: Moving Average indicator") from a minute bar to the next - there is a binding of milliseconds to minutes.


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
25 Nov 2023 at 18:27

Do you need milliseconds to compress ticks into the future? I understand correctly?

Well, “in the future” - for me - I haven't read further yet. For you, I suppose, this is already “in the past”...)))

If this is so, then I agree that the seconds can be removed - the probability of a coincidence is extremely low.)))


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
26 Nov 2023 at 07:33

Oh, a miracle! If you compress milliseconds, then the formation time of the minute candle will be 00:01:06. Against 00:01:52 - without compression. We won 46 seconds!)))


![qqq](https://c.mql5.com/avatar/avatar_na2.png)

**[qqq](https://www.mql5.com/en/users/soroboru)**
\|
1 Dec 2023 at 23:56

**As a result, with all the edits.**

1153,809 movement positions were created.

Deleted ticks  = 1066231

Checking the [execution speed](https://www.mql5.com/en/articles/4310 "Article: Battle for speed: QLUA vs MQL5 - why is MQL5 50 to 600 times faster? ") . 2023.12.02 01:52:21 ---- 2023.12.02 01:53:17

Time to build the first candle: 00:00:56  seconds.)))

We won 56 seconds!

Exactly half of it.

![Dermeni73](https://c.mql5.com/avatar/avatar_na2.png)

**[Dermeni73](https://www.mql5.com/en/users/dermeni73)**
\|
8 Nov 2024 at 16:50

Thanks for your great work!

I got a question. Is it possible to add rewinding option to the replay?? I mean going back to previous candles and playing again.

![Developing a Replay System — Market simulation (Part 02): First experiments (II)](https://c.mql5.com/2/52/replay-p2-avatar.png)[Developing a Replay System — Market simulation (Part 02): First experiments (II)](https://www.mql5.com/en/articles/10551)

This time, let's try a different approach to achieve the 1 minute goal. However, this task is not as simple as one might think.

![Revisiting an Old Trend Trading Strategy: Two Stochastic oscillators, a MA and Fibonacci](https://c.mql5.com/2/56/tranding_strategy_avatar.png)[Revisiting an Old Trend Trading Strategy: Two Stochastic oscillators, a MA and Fibonacci](https://www.mql5.com/en/articles/12809)

Old trading strategies. This article presents one of the strategies used to follow the trend in a purely technical way. The strategy is purely technical and uses a few technical indicators and tools to deliver signals and targets. The components of the strategy are as follows: A 14-period stochastic oscillator. A 5-period stochastic oscillator. A 200-period moving average. A Fibonacci projection tool (for target setting).

![Category Theory in MQL5 (Part 13): Calendar Events with Database Schemas](https://c.mql5.com/2/56/Category-Theory-p13-avatar.png)[Category Theory in MQL5 (Part 13): Calendar Events with Database Schemas](https://www.mql5.com/en/articles/12950)

This article, that follows Category Theory implementation of Orders in MQL5, considers how database schemas can be incorporated for classification in MQL5. We take an introductory look at how database schema concepts could be married with category theory when identifying trade relevant text(string) information. Calendar events are the focus.

![Creating Graphical Panels Became Easy in MQL5](https://c.mql5.com/2/56/creating_graphical_panels_avatar.png)[Creating Graphical Panels Became Easy in MQL5](https://www.mql5.com/en/articles/12903)

In this article, we will provide a simple and easy guide to anyone who needs to create one of the most valuable and helpful tools in trading which is the graphical panel to simplify and ease doing tasks around trading which helps to save time and focus more on your trading process itself without any distractions.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/10543&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070269821683176172)

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