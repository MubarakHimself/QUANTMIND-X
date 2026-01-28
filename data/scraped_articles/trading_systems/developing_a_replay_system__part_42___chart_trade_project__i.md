---
title: Developing a Replay System (Part 42): Chart Trade Project (I)
url: https://www.mql5.com/en/articles/11652
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:12:18.709952
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/11652&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070096094551019488)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 41): Starting the second phase (II)](https://www.mql5.com/en/articles/11607), I showed how to create an indicator to be used for the mouse. You may be thinking that this is nonsense and won't do any good in the long run. If this is the case, it is only because you never understood what was happening.

Every time we create something new, we will have to conduct a series of tests and create conditions so that a certain program or process can coexist with others. The implementation of this condition shows whether you are ready to declare yourself as a professional.

Beginners often have problems when one program or process conflicts with others. In such a case strange things may happen. A beginner may find it hard or impossible to solve these problems, but a professional should be able to cope with such situations.

My point is this: there is no point in programming an Expert Advisor that can perform a thousand and one operations, or have a series of indicators and graphical elements, or many programs or processes for working on the market, if when you try to combine them on one chart, they begin to conflict with each other.

Or worse, having created one extremely reliable and stable program, sometimes you need to develop another that uses part of the functionality of the first one, and in doing so the new program begins to exhibit a series of errors and failures, especially when placed together with other processes. Troubleshooting these glitches will definitely take time, as some of them will be simple, but some will be very complex.

Why am I saying this? Because we have already created an indicator for the mouse. The MetaTrader 5 platform has various tools that can help us. Some of them are simpler and need improvements to suit everyone's individual needs. You can do this by creating various programs to get the perfect rifle, the perfect weapon for use in the market. But what I'd like you to think about is this: What if, instead of trying to create the perfect weapon, we started creating various compatible parts and components that, when combined, could be assembled each time to create a weapon designed specifically for what we encounter?

This seems like a good solution, doesn't it? And that's exactly what we did when we created the mouse indicator. Based on this indicator, we will begin to create a series of parts and components that will help us a lot.

From the very beginning of this series on developing the replay/simulator system, I was saying that the idea is to use the MetaTrader 5 platform in the same way both in the system we are developing and in the real market. It is important that this is done properly. No one wants to train and learn to fight using one tool while having to use another one during the fight. Believe me, using the replay/simulator is learning, but trading on a REAL account is a real battle. In a real battle, you need to use what you have learned and become accustomed to in training.

After the mouse indicator, there are two more tools that need to be created as quickly as possible. One is simple and the other is quite complex due to the fact that we will have to deal with data simulation. So let's start with something simple to get used to, and when we move on to something much more complex, we'll already have a good idea of how everything will work.

The simple tool is Chart Trade. In its latest version it was presented in the article: [Developing a trading Expert Advisor from scratch (Part 30): CHART TRADE as an indicator?!](https://www.mql5.com/en/articles/10653). Compared to what we will do now, this indicator will seem like a child's toy, as it is so simple and non-universal. Let's create a new Chart Trade that can harmoniously coexist with the already created indicator, namely the mouse indicator.

### New Chart Trade

Chart Trade, which we discussed in the above mentioned article, will serve as the basis, but we will not use its code or template. We will create another one, much more mature and adequate to our needs.

You will see that the new code that I will present is much simpler and more compact, which means that it will be easier to modify and correct things. The idea is to use the mouse indicator to move, click and interact with any element on the chart. To do this we will need to create some rather unusual code. I will demonstrate everything gradually so that it is easy to follow what is being done and how, since it is very different from what many are used to seeing and using in MetaTrader 5. When it comes to programming, many may not understand at first.

Let's start with the fact that our Chart Trade uses a concept I recently introduced: RAD programming. The word RAD stands for **R** apid **A** application **D** evelopment. I was introduced to this concept many years ago when I started programming for Windows systems. It greatly helps in creating applications efficiently, allowing you to spend more time working directly with the code instead of worrying about the appearance of the program. In the article [Multiple indicators on one chart (Part 06): Turning MetaTrader 5 into a RAD system (II)](https://www.mql5.com/en/articles/10301), I showed how to use something similar in MetaTrader 5.

The same concept will be used again here, however, in an improved version. If you want to start off on the right foot, read the mentioned article as well as the previous one, which is also important for understanding what we will be doing. Here is the article: [Multiple indicators on one chart (Part 05): Turning MetaTrader 5 into a RAD system (I)](https://www.mql5.com/en/articles/10277). It is very important to understand what is explained there. The current topic will be more rich and complex, although this indicator is the easiest part compared to what awaits us in the future.

The new template will look like in Figure 01 below:

![Figure 01](https://c.mql5.com/2/50/001.png)

Figure 01 - RAD Template

Figure 01 shows the actual view that Chart Trade will have. This is a Chart Trade template, not Chart Trade itself. The latter can be seen in video 01, just below, where I demonstrate the interaction between it and the mouse indicator. Watch the video and notice how it works. The video clearly presents everything that will be discussed in this article, so it is recommended for viewing.

Vídeo from the article "Desenvolvendo um sistema de Replay (Parte 42): Projeto do Chart Trade (I)" - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11652)

MQL5.community

1.91K subscribers

[Vídeo from the article "Desenvolvendo um sistema de Replay (Parte 42): Projeto do Chart Trade (I)"](https://www.youtube.com/watch?v=I1FAfCqFfs0)

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

[Watch on](https://www.youtube.com/watch?v=I1FAfCqFfs0&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11652)

0:00

0:00 / 1:41

•Live

•

Video 01 - Demo video

Considering that you have watched video 01, let's start looking at the Chart Trade indicator code. But first we need to make a small addition to the code of the C\_Terminal class, as shown in the following snippet.

**Chart Trade indicator source code**

```
156. //+------------------------------------------------------------------+
157.            bool IndicatorCheckPass(const string szShortName)
158.                    {
159.                            string szTmp = szShortName + "_TMP";
160.
161.                            if (_LastError != ERR_SUCCESS) return false;
162.                            IndicatorSetString(INDICATOR_SHORTNAME, szTmp);
163.                            if (ChartWindowFind(m_Infos.ID, szShortName) != -1)
164.                            {
165.                                    ChartIndicatorDelete(m_Infos.ID, 0, szTmp);
166.                                    Print("Only one instance is allowed...");
167.
168.                                    return false;
169.                            }
170.                            IndicatorSetString(INDICATOR_SHORTNAME, szShortName);
171.                            ResetLastError();
172.
173.                            return true;
174.                    }
175. //+------------------------------------------------------------------+
```

**Fragment of the C\_Terminal class**

This code may not seem unusual. We have already seen it before, but now it is part of the C\_Terminal class, so all classes or indicators that use this class take advantage of this code. This code prevents the indicator from being placed on the chart more than once. There is a way around this and we'll use it in the future, but don't worry about that detail for now. Keep in mind that this code prevents two identical indicators from appearing on the chart at the same time.

But before moving on to the source class that will support our Chart Trade, let's first look at the indicator code. I am changing the order of presentation to make the class explanation more appropriate. The full indicator code can be seen below.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Base version for Chart Trade (DEMO version)"
04. #property version   "1.42"
05. #property link "https://www.mql5.com/pt/articles/11652"
06. #property indicator_chart_window
07. #property indicator_plots 0
08. //+------------------------------------------------------------------+
09. #include <Market Replay\Chart Trader\C_ChartFloatingRAD.mqh>
10. //+------------------------------------------------------------------+
11. C_ChartFloatingRAD *chart = NULL;
12. //+------------------------------------------------------------------+
13. int OnInit()
14. {
15.     chart = new C_ChartFloatingRAD("Indicator Chart Trade", new C_Mouse("Indicator Mouse Study"));
16.     if (_LastError != ERR_SUCCESS)
17.     {
18.             Print("Error number:", _LastError);
19.             return INIT_FAILED;
20.     }
21.
22.     return INIT_SUCCEEDED;
23. }
24. //+------------------------------------------------------------------+
25. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
26. {
27.     return rates_total;
28. }
29. //+------------------------------------------------------------------+
30. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
31. {
32.     (*chart).DispatchMessage(id, lparam, dparam, sparam);
33.
34.     ChartRedraw();
35. }
36. //+------------------------------------------------------------------+
37. void OnDeinit(const int reason)
38. {
39.     delete chart;
40.
41.     ChartRedraw();
42. }
43. //+------------------------------------------------------------------+
```

As shown above, the code for this indicator is quite simple. Of course we will add more information to make an adequate description, but for now this is enough for what we need.

Please note that the only thing we are doing is specifying that no data will be displayed on line 07. And the rest of the work consists of initializing a pointer to use the class that will support trading from the chart, and destroying the same pointer. However, we will have the behavior demonstrated in video 01.

I think there shouldn't be any difficulties in understanding the lines of indicator code. Perhaps the only one that will require a little more work is line 15. But it is also quite simple. We use two operators **new** to initialize both classes.

Pay attention to the information passed in the form of string variables. The first string variable refers to the name that will be recognized by MetaTrader 5 when the indicator is on the chart. The second refers to the name of the mouse indicator. This is defined in the mouse indicator code. More detailed information can be found in the article: [Developing a Replay System (Part 41): Starting the second phase (II)](https://www.mql5.com/en/articles/11607).

Now that you have seen how simple the Chart Trade indicator code is, let's look at the class code. Please remember that the code you will see is the original code and additional elements will be added to make the indicator truly functional.

### Class C\_ChartFloatingRAD

Despite the name, this class is mainly designed to work with the Chart Trade indicator. At the current stage of development, it already contains all the necessary logic to provide us with behavior like in video 01. That is, if we are doing research, we will not be able to move the window even if we click on it and drag the mouse. The window cannot be moved if study is running.

It may seem that to achieve this required the creation of a complex diagram, as well as an image in the video, which already shows the final appearance of the indicator. Let's see if this is true. Just look at the class code. Its full code is as follows.

**C\_ChartFloatingRAD class source code**

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. //+------------------------------------------------------------------+
04. #include "../Auxiliar/C_Terminal.mqh"
05. #include "../Auxiliar/C_Mouse.mqh"
06. //+------------------------------------------------------------------+
07. class C_ChartFloatingRAD : private C_Terminal
08. {
09.     private :
10.             struct st00
11.             {
12.                     int     x, y, cx, cy;
13.                     string  szObj_Chart;
14.                     long    WinHandle;
15.             }m_Info;
16. //+------------------------------------------------------------------+
17.             C_Mouse *m_Mouse;
18. //+------------------------------------------------------------------+
19.             void CreateWindowRAD(int x, int y, int w, int h)
20.                     {
21.                             m_Info.szObj_Chart = (string)ObjectsTotal(GetInfoTerminal().ID);
22.                             ObjectCreate(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJ_CHART, 0, 0, 0);
23.                             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x = x);
24.                             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y = y);
25.                             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XSIZE, w);
26.                             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, h);
27.                             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_DATE_SCALE, false);
28.                             ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_PRICE_SCALE, false);
29.                             m_Info.WinHandle = ObjectGetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_CHART_ID);
30.                             m_Info.cx = w;
31.                             m_Info.cy = 26;
32.                     };
33. //+------------------------------------------------------------------+
34. inline void UpdateChartTemplate(void)
35.                     {
36.                             ChartApplyTemplate(m_Info.WinHandle, "IDE_RAD.tpl");
37.                             ChartRedraw(m_Info.WinHandle);
38.                     }
39. //+------------------------------------------------------------------+
40.     public  :
41. //+------------------------------------------------------------------+
42.             C_ChartFloatingRAD(string szShortName, C_Mouse *MousePtr)
43.                     :C_Terminal()
44.                     {
45.                             if (!IndicatorCheckPass(szShortName)) SetUserError(C_Terminal::ERR_Unknown);
46.                             m_Mouse = MousePtr;
47.                             CreateWindowRAD(0, 0, 170, 240);
48.                             UpdateChartTemplate();
49.                     }
50. //+------------------------------------------------------------------+
51.             ~C_ChartFloatingRAD()
52.                     {
53.                             ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Chart);
54.                             delete m_Mouse;
55.                     }
56. //+------------------------------------------------------------------+
57.             void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
58.                     {
59.                             static int sx = -1, sy = -1;
60.                             int x, y, mx, my;
61.
62.                             switch (id)
63.                             {
64.                                     case CHARTEVENT_MOUSE_MOVE:
65.                                             if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft))
66.                                             {
67.                                                     x = (int)lparam;
68.                                                     y = (int)dparam;
69.                                                     if ((x > m_Info.x) && (x < (m_Info.x + m_Info.cx)) && (y > m_Info.y) && (y < (m_Info.y + m_Info.cy)))
70.                                                     {
71.                                                             if (sx < 0)
72.                                                             {
73.                                                                     ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
74.                                                                     sx = x - m_Info.x;
75.                                                                     sy = y - m_Info.y;
76.                                                             }
77.                                                             if ((mx = x - sx) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x = mx);
78.                                                             if ((my = y - sy) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y = my);
79.                                                     }
80.                                             }else if (sx > 0)
81.                                             {
82.                                                     ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
83.                                                     sx = sy = -1;
84.                                             }
85.                                             break;
86.                             }
87.                     }
88. //+------------------------------------------------------------------+
89. };
90. //+------------------------------------------------------------------+
91.
```

Is that all? Where are the object creation codes? Mouse manipulation?

I can't believe that these two simple codes can do what is shown in video 01. Seems like a joke. But I am not joking. These few lines can make the window seen in Figure 01 behave as shown in Video 01.

You can study the chart using your mouse, you can click and drag the Chart Trade window, all thanks to just this code. A few more things will be added to this class code to make the window functional and allow the asset to be displayed. The buttons will become functional and you will be able to edit values that are not yet presented. There are actually a few more tweaks missing here, but don't expect to see those objects being created here. I use another way to make our window functional and will show it in the next article.

But wait, we're not done yet. To really understand what's going on, let's look at the explanation of the class code and why it contains that particular code.

Let's start from line 07. Note that the C\_ChartFloatingRAD class inherits from the C\_Terminal class. Because we must always give minimal privileges to inheritance, we make inheritance private. This way, no other code will have access to the C\_Terminal class, at least not through the C\_ChartFloatingRAD class. This is not a problem since the indicator code does not mention any C\_Terminal class code.

If you have understood this point, you can move on to the constructor. This will make the execution process easier to understand.

The constructor starts on line 42. Note that it will take a string and an object, actually a pointer, to refer to the mouse indicator. On line 43, we perform the necessary initialization of the C\_Terminal class. When we go to the constructor code, starting on line 45, we make a call to the verification code to ensure that the Chart Trade indicator is not on the chart. If this check fails, we indicate this in the **\_LastError** constant using the [SetUserError](https://www.mql5.com/en/docs/common/setusererror) call.

**Please note the following. If you don't understand what I'm about to explain, you'll be completely lost in the rest of this series. Therefore, you Should pay close attention to what I say.**

On line 46, we store the pointer that was passed using the new operator call. This happens in the Chart Trade indicator code. You can see this call in line 15 of the indicator code. It **DOESN'T MATTER** whether the mouse indicator is present on the chart at the time line 46 is executed. I repeat, because this is important: the presence of an indicator on the chart at the time of the call **DOESN'T MATTER**.

What really matters (and many may not understand this) is the name that is passed in the call visible on line 15 of the Chart Trade indicator code. This string variable, which is passed to the constructor of the C\_Mouse class, is what matters. But not the presence or absence of the indicator on the chart.

This may seem like it doesn't make sense. But it does. To understand it, you need to return to the previous article [Developing a Replay System (Part 41): Starting the second phase (II)](https://www.mql5.com/en/articles/11607) and pay attention to line 169 in the code of the C\_Mouse class. Only this line will ensure the correct operation of the code presented in the C\_ChartFloatingRAD class. Therefore, it is necessary to specify the name of the mouse indicator correctly so that the MetaTrader 5 platform can find it when it is necessary to execute line 169 in the C\_Mouse class.

This may still seem confusing, so let's move on to explain how the code works. We'll come back to this point later to make things clearer.

On line 47, we call the function that is on line 19. This function on line 19 is simply a procedure that creates an **OBJ\_CHART** object on the chart. Only this. However, on line 29, we save the ID of this object so that we can use it in a very specific way later. I think that no one will have any difficulty understanding the remaining lines of this procedure.

Returning to the constructor code, there is another call on line 48. This call will execute what is shown on line 34. At the moment, we have two lines 36 and 37. Line 36 will look for the template we specify, which is exactly the one shown in Figure 01. On line 37, we force MetaTrader 5 to update the OBJ\_CHART object so that the template is displayed.

Thus, we present our Chart Trade. Well, it is not yet functional. However, we can do some "manipulation" to understand the interaction between the mouse indicator and the Chart Trade indicator.

The destructor code is quite simple and does not require much explanation. Let's take a closer look at the message processing code: the DispatchMessage procedure starts on line 57. The interaction between the two indicators takes place here.

Basically, for now, we will only handle the mouse movement event that starts at line 64. Attention: we won't do this the usual way. We will handle the mouse movement event completely differently. For those who are used to programming in a certain way, this may seem _"_ **_Very crazy_**" _._

For better understanding (and this is why I don't show other things in this article), you need to refer to the previous material [Developing a Replay System (Part 41): Starting the second phase (II)](https://www.mql5.com/en/articles/11607). We are dealing not with one, but with two indicators at the same time. They interact to coexist in harmony and the user could do what is shown in video 01.

In this Chart Trade indicator message processing code, we will mention the mouse indicator only once. It is mentioned in line 65. Try to understand what is happening. If you understand this, the rest will be easy.

There can be two situations here. The first is when the mouse indicator is on the chart, and the second is when the indicator is not on the chart. Let me remind you that the indicator must have the name that we specified in the constructor of the C\_ChartFloatingRAD class. If the name is different, MetaTrader 5 will not help us, and we will be faced with a situation where the indicator is not on the chart. Pay attention to this detail.

Once this is clear and you are clearly aware of these two situations, we can begin further explanation.

When this happens, the call on line 65 will make the call on line 158 of the C\_Mouse class (from the previous article). In line 15 of the Chart Trade indicator code, we tell the constructor of the C\_Mouse class that we will use the C\_Mouse class as a translator. Line 158 of the C\_Mouse class will call line 163 to check whether the button specified in line 65 of the C\_ChartFloatingRAD class has been clicked.

The moment line 165 is executed in the C\_Mouse class, the code will understand that we are using a translator. We will then execute line 169 to capture the mouse indicator handle. At this time we will turn to MetaTrader 5 for help, so the names provided must match. If MetaTrader 5 finds the indicator we are referencing, the handle will receive an index to access the indicator's buffer. If the buffer can be read, line 172 will capture the mouse pointer data. If the buffer cannot be read, the function will fail, which will also cause the function on line 158 to fail. This will cause the check present on line 65 of the C\_ChartFloatingRAD class to fail. Thus, we will receive an indication that the click did not occur.

This logic will work in two situations:

- When the mouse indicator is missing from the chart. Remember that it must have the same name as what was specified in order for MetaTrader 5 to find it.
- When we do research on the chart using the mouse indicator.

In either of these two situations, line 65 of the C\_ChartFloatingRAD class will indicate a mouse click failure.

Does this part seem confusing? Don't worry, things get even more interesting from here.

Now let's analyze a situation where MetaTrader 5 finds the mouse indicator and we want to move the Chart Trade indicator on the chart, as shown in video 01. To understand how this will happen, you need to understand the following articles:

- [Developing a Replay System (Part 37): Paving the Path (I)](https://www.mql5.com/en/articles/11585)
- [Developing a Replay System (Part 38): Paving the Path (II)](https://www.mql5.com/en/articles/11591)
- [Developing a Replay System (Part 39): Paving the Path (III)](https://www.mql5.com/en/articles/11599)
- [Developing a Replay System (Part 40): Starting the second phase (I)](https://www.mql5.com/en/articles/11624)
- [Developing a Replay System (Part 41): Starting the second phase (II)](https://www.mql5.com/en/articles/11607)

Each of these articles is a preparation for what is to come. If you haven't read them or don't understand the content, I suggest you go back to them and understand how communication happens. Although it may seem simple, these concepts become quite difficult to understand in practice, especially for those who do not apply such methods.

What I'm showing here is the easiest part of all. Compared to what comes next, it will seem like preschool education. Therefore, it is very important to understand how this code works.

At some point, the call of C\_ChartFloatingRAD on line 65 will encounter the call of C\_Mouse shown on line 181. When this happens, line 160 of the C\_Mouse class will check the button code passed to it. If the code matches the one passed in, the function will return true. At this moment, and for as long as the mouse button is pressed, the code present between lines 67 and 79 will be executed.

There is a small flaw in this code, but since the code is a demo, you can ignore it for now.

Now the question arises: why in lines 67 and 68 am I using the values provided by MetaTrader 5 and not the mouse indicator? Of course not because it is prettier or simpler. There is a reason to use the values provided by MetaTrader 5 rather than the indicator values. To understand this, you need to look at the code in the C\_Mouse class between lines 203 and 206. In these lines, we perform the transformation and adjustment of mouse coordinates. In this case, the mouse coordinates cease to be on-screen (x and y) and become active (price and time).

This is very useful for working with certain types of problems, but problematic when we need to work differently. If we use code similar to the following on lines 67 and 68 of the C\_ChartFloatingRAD class:

```
x = (*m_Mouse).GetInfoMouse().Position.X;
y = (*m_Mouse).GetInfoMouse().Position.Y;
```

We will get the movement of the OBJ\_CHART object as if it were tied to the asset coordinates (price and time). But in reality, such an object uses screen coordinates (x, y), so the movement will be quite strange and not as smooth as in video 01. For some types of objects it is actually desirable to use the coordinates provided by the mouse indicator.

### Conclusion

In this article, we looked at the first stage of creating a Chart Trade indicator. I showed how you can use the mouse indicator in perfect harmony with the Chart Trade indicator we are creating, which, however, does not yet have any functionality other than the ability to drag and drop on the chart.

In the next article, we will start working on expanding the functionality of Chart Trade.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11652](https://www.mql5.com/pt/articles/11652)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11652.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11652/anexo.zip "Download Anexo.zip")(420.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/470524)**
(1)


![Sergei Lebedev](https://c.mql5.com/avatar/2019/7/5D3E1392-839A.jpg)

**[Sergei Lebedev](https://www.mql5.com/en/users/salebedev)**
\|
2 Aug 2024 at 10:24

Very interesting series of articles, as MT5 has no internal MarketReply functionality.

Kindly ask English section moderators to speed up translation to English of next parts, namely 43-60!

![Building A Candlestick Trend Constraint Model (Part 7): Refining our model for EA development](https://c.mql5.com/2/86/Building_A_Candlestick_Trend_Constraint_Model_Part_7___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 7): Refining our model for EA development](https://www.mql5.com/en/articles/15154)

In this article, we will delve into the detailed preparation of our indicator for Expert Advisor (EA) development. Our discussion will encompass further refinements to the current version of the indicator to enhance its accuracy and functionality. Additionally, we will introduce new features that mark exit points, addressing a limitation of the previous version, which only identified entry points.

![Population optimization algorithms: Whale Optimization Algorithm (WOA)](https://c.mql5.com/2/73/Whale_Optimization_Algorithm___LOGO.png)[Population optimization algorithms: Whale Optimization Algorithm (WOA)](https://www.mql5.com/en/articles/14414)

Whale Optimization Algorithm (WOA) is a metaheuristic algorithm inspired by the behavior and hunting strategies of humpback whales. The main idea of WOA is to mimic the so-called "bubble-net" feeding method, in which whales create bubbles around prey and then attack it in a spiral motion.

![MQL5 Wizard Techniques you should know (Part 29): Continuation on Learning Rates with MLPs](https://c.mql5.com/2/86/MQL5_Wizard_Techniques_you_should_know_Part_29___LOGO.png)[MQL5 Wizard Techniques you should know (Part 29): Continuation on Learning Rates with MLPs](https://www.mql5.com/en/articles/15405)

We wrap up our look at learning rate sensitivity to the performance of Expert Advisors by primarily examining the Adaptive Learning Rates. These learning rates aim to be customized for each parameter in a layer during the training process and so we assess potential benefits vs the expected performance toll.

![Build Self Optimizing Expert Advisors With MQL5 And Python](https://c.mql5.com/2/85/Build_Self_Optimizing_Expert_Advisors_With_MQL5_And_Python__LOGO.png)[Build Self Optimizing Expert Advisors With MQL5 And Python](https://www.mql5.com/en/articles/15040)

In this article, we will discuss how we can build Expert Advisors capable of autonomously selecting and changing trading strategies based on prevailing market conditions. We will learn about Markov Chains and how they can be helpful to us as algorithmic traders.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=tnqcbssbbiwruqnujumyugiqetqcmfmw&ssn=1769184737502803368&ssn_dr=0&ssn_sr=0&fv_date=1769184737&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11652&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2042)%3A%20Chart%20Trade%20Project%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691847370995441&fz_uniq=5070096094551019488&sv=2552)

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