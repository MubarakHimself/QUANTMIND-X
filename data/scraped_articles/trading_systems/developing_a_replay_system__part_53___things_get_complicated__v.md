---
title: Developing a Replay System (Part 53): Things Get Complicated (V)
url: https://www.mql5.com/en/articles/11932
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:05:47.109377
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/11932&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070004727711731289)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 52): Things Get Complicated (IV)](https://www.mql5.com/en/articles/11925), we created a new data structure so that the mouse indicator could interact with the control indicator. Although the interaction may go smoothly and steadily at first, we encounter some problems that force us to change things up a bit.

The problem is not in the code, not in the platform, not in the concepts used, but in our intentions and in the way we work. Personally, I apologize to anyone who followed this series of articles until the replay/simulator system came along. To be honest, I didn't expect to have to use some of the techniques found in modular systems, but it's impossible to continue developing the replay/simulator system without using some of the techniques developed decades ago.

What I am going to explain here may seem extremely strange for some readers and very familiar for others. However, no matter how you look at it, the mechanism that I will explain here and that we will start using more and more often is present in MetaTrader 5, and MQL5 allows us to use it. But when using MQL5, we are limited in that we can only do this inside the chart or inside MetaTrader 5 itself. This is not a bad thing, on the contrary, it is an imposed security measure, since the platform is designed to work with money, and we do not want to lose money because of something strange.

So, despite the restrictions, there is no reason for complaining. What we're going to start using here may make you, as a new programmer, feel lost for a while. But this is only because you don't know the content and how to use it. However, I have been using this technique for some time now, but not in the way we will start using it from now on. So it's time to raise the bar and apply more complex concepts in your articles.

### Explaining the concept

We're going to start making heavy use of messaging between programs, and if you're new to this area, please pay attention because there's a quick introduction now. Yes, MQL5 allows us to do this. When used and designed correctly, this approach is very powerful. But without understanding some details, you will feel completely confused and this will lead to incorrect behavior of programs when they are placed together on the same chart in MetaTrader 5.

Up to this point, the programs I've presented in this series have used messages, not between themselves, but within the code, so that one class can communicate with another even if they are at different levels or not related by inheritance. You can see this by looking at my class codes. Almost all of them have a common procedure: DispatchMessage. The main purpose of this procedure is to manage messages sent to the class, although there are other ways to communicate with the class, using other functions. DispatchMessage is used to manage messages addressed to a class.

This idea is not something new for me, it has been around for a long time and is aimed at creating a common interface between programs or procedures in general. Those who have been working in the field of professional programming for a long time know what we are talking about. So when you need to send data, values or queries to another program whose code you don't know at all, you use this principle. You send a message to a very specific function, and it returns specific information. They communicate precisely through this single function.

Its name may be different, but the set and sequence of data provided to it is always the same.

This may seem superficial and completely meaningless, but if you are studying programming, in this case MQL5, then you have probably seen this function more than once, and almost all indicator or Expert Advisor code contains this function. This function, or rather procedure, is called OnChartEvent in MQL5.

You might be thinking, "How is this possible? Are you saying that MetaTrader 5 communicates with my software?" The answer is "Yes." By analyzing the value specified in the integer constant ID within this call, you can filter and determine what message MetaTrader 5 sent you.

This will surprise you very much, but for beginners everything becomes even more difficult. However, in order not to complicate the situation, I will conduct the explanation within the framework of MetaTrader 5, but limit myself to it. It's all much broader and more complex. So you need to understand this point well: MetaTrader 5 sends messages to your program. These messages are intercepted and managed by some procedures present in your program. Among these procedures, there is one that is more general and allows sending much more complex elements, and also uses a common and well-defined interface. The name of this procedure is OnChartEvent. We're done with that. Now comes the tricky part.

MQL5 (I'll only talk about it for now, so as not to complicate the situation) allows us to define custom events. These events are denoted by a constant and a value. The name of this constant is CHARTEVENT\_CUSTOM. Thus, using this constant, we can send a message to the message handler from any point in our program, which allows us to concentrate the processing of both specific and general events in one point, but in any case without calling the message dispatcher. You have to do it the right way, and to make things easier, MQL5 provides us with a function for this purpose: EventChartCustom. Using this function, you can send messages to the default message handler, the aforementioned OnChartEvent, which in my case calls DispatchMessage.

It all looks great and works great, allowing us to do a lot of things. However, there is a danger here. I am talking about CHARTEVENT\_CUSTOM. The biggest danger of these user calls is not in my program, or yours, or any other good programmer's program, but in the fact that the user doesn't know what each program does. Often users have no idea what is really going on. This is why it is important to NEVER use something without understanding what it really is. The program may work fine in several scenarios, but there will be only one in which the interaction between the programs will turn into a complete nightmare: it can lead to the platform crashing, objects disappearing or appearing out of nowhere, and you will not be able to understand what is happening. It's good when the effects are noticeable, but what if they happen completely unnoticed? It may seem to you that everything is going smoothly, but in reality the situation may be as if you are on the edge of an abyss.

If you are a professional programmer and you are developing solutions, don't get me wrong, but you have to explain to the client that your program can interact in a way that will cause problems for other programs, or that other programs can cause problems for yours. Users should be careful not to mix one programmer's code with another's. This is the reason why some time ago Windows would often crash for no apparent reason and people would simply blame a program, when in fact the crashes usually occurred when several programs were running at the same time in the same environment.

I won't go into this topic in depth because it is beyond the main purpose of this explanation, but it is important to understand something: MetaTrader 5 is similar to the Windows operating system. If we use the right tools in the right way, we will never have problems with the platform. But if you start mixing everything, be careful, as you may encounter a number of problems. Knowing that MetaTrader 5 is a graphical environment designed for trading in financial markets, the question arises: what will happen if two programs use EventChartCustom, because this function makes MetaTrader 5 transmit messages at the request of the programmer?

Well. The question is very simple, and indeed, it is the right question. To understand this, let's start with a simpler case: what happens when ONE program uses the EventChartCustom function? Essentially, MetaTrader 5 will send a custom event that will be handled by the OnChartEvent procedure. It seems obvious, doesn't it? Actually, no. It's not that obvious, and that's where the danger lies.

A typical use of EventChartCustom can be seen in the control indicator code shown in the previous article. In line 30 of the control indicator code, you can see the following:

```
30.     EventChartCustom(user00, C_Controls::evInit, Info.s_Infos.iPosShift, Info.df_Value, "");
```

So when MetaTrader 5 executes this line, a ChartEvent will appear, which will cause the OnChartEvent to be executed by the code that is present on the chart. This will continue until we find the DispatchMessage function, which is in the C\_Control class, it line 161 of the code shown in the previous article. For you convenience I will add that snippet here.

```
161.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
162.                    {
163.                            u_Interprocess Info;
164.
165.                            switch (id)
166.                            {
167.                                    case CHARTEVENT_CUSTOM + C_Controls::evInit:
168.                                            Info.df_Value = dparam;
169.                                            m_Slider.Minimal = Info.s_Infos.iPosShift;
170.                                            SetPlay(Info.s_Infos.isPlay);
171.                                            if (!Info.s_Infos.isPlay) CreateCtrlSlider();
172.                                            break;
```

Part of code present in the C\_Control class

Now, since the event is custom, MetaTrader 5 will set the ID value in the OnChartEvent procedure call so that the ID is the value corresponding to CHARTEVENT\_CUSTOM plus one more value. In the demonstrated case, the value will correspond to the value of C\_Control:evInit. But what is the value of C\_Control:evInit? To find this out, you need to go to line 35 of the C\_Control class code and check the value.

```
035.            enum EventCustom {evInit};
```

This value is part of the enumeration. Since the enum only has this value and is not initialized, it will start with the default value, which is NULL. Then the same line 30 present in the control indicator will actually be interpreted by MetaTrader 5 as follows:

```
30.     EventChartCustom(user00, CHARTEVENT_CUSTOM + 0, Info.s_Infos.iPosShift, Info.df_Value, "");
```

This code will work perfectly and safely, allowing us as programmers to make a custom call at any time to force the DispatchMessage procedure present inside the C\_Control class to initialize some values that were not set in the class constructor. Things like this happen very often and are a perfectly appropriate form of programming that is very useful in a variety of situations.

The same applies to the mouse indicator. Let's look at it now to understand something else, but still within ONE program using custom events in MetaTrader 5.

There is not a single call to EventChartCustom visible in the entire mouse indicator code. However, the message handler contains code to respond to the custom event. This code existed for quite a long time, anticipating its future use. You can see this handling code in lines 196 and 199 of the C\_Mouse class. If you look closely, you'll notice a few things. I am including this fragment within the article to better explain the idea.

```
189. virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
190.                    {
191.                            int w = 0;
192.                            static double memPrice = 0;
193.
194.                            if (m_Mem.szShortName == NULL) switch (id)
195.                            {
196.                                    case (CHARTEVENT_CUSTOM + ev_HideMouse):
197.                                            if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
198.                                            break;
199.                                    case (CHARTEVENT_CUSTOM + ev_ShowMouse):
200.                                            if (m_Mem.IsFull) ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, m_Info.corLineH);
201.                                            break;
```

Part of code present in the C\_Mouse class

Note that in this snippet we use ev\_HideMouse and ev\_ShowMouse. If any program wants to hide the mouse indicator line, we can just ask MetaTrader 5 to send a custom event to the mouse indicator. This way you can hide or show the mouse line. Note that we are not destroying the object, we are simply changing the color property.

These values of ev\_HideMouse and ev\_ShowMouse are enumerations, but where do they come from? You can see them in line 34 of the C\_Mouse class. Again, I am showing the code here for ease of explanation.

```
034.            enum eEventsMouse {ev_HideMouse, ev_ShowMouse};
```

You may not yet understand what I'm trying to explain. Please watch video 01below, which shows how this system functions.

YouTube

Video 01 - Demonstration

Everything works in perfect harmony without creating any problems. During initialization, the control indicator tells MetaTrader 5 that a custom event needs to be handled, and the mouse indicator waits for a custom event from some program to hide or show the mouse line. When separated, these two indicators do not create problems and, moreover, do not conflict with each other. However, when they find themselves together, the situation becomes more complicated. You can see this in video 02.

Demonstração Parte 53 2 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11932)

MQL5.community

1.91K subscribers

[Demonstração Parte 53 2](https://www.youtube.com/watch?v=pOPaQjGu1Uo)

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

[Watch on](https://www.youtube.com/watch?v=pOPaQjGu1Uo&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11932)

0:00

0:00 / 4:40

•Live

•

Video 02 - Conflicts

Why is this happening, why are they okay when they are separated but conflict when they are placed together? This is a big problem that may be discouraging for makes many beginning programmers, and those who have no real motivation to learn programming give up at this stage. However, a professional programmer or someone who aspires to become one sees this as an opportunity for learning and strives to gain more knowledge and resolve this conflict. This is exactly the problem with using programs from different programmers, or even from the same programmer who does not take the time to test programs more creatively. But let's leave this issue aside and focus on the conflict shown in video 02.

Why do two indicators that seem to work well independently of each other work together in such a strange way?

### Understanding the concept of work environment

If you are a more experienced user, then it is likely that you have already heard of or used work environments. The concept is based on separating elements so that they can coexist independently in one ecosystem.

In MetaTrader 5 this issue of the environment is taken very seriously. If you don't understand this idea, you won't understand the reason for the conflict shown in video 02. Most importantly, you will not be able to resolve this very conflict. Remember: We want the mouse indicator to work in complete harmony with any other indicator. So we eliminate a number of problems, and it becomes a module of an even larger system.

Many would simply give up, but you can't give up. We are making the system modular, and for this we need the mouse indicator to work together with the control indicator.

After watching video 01 we can see that the indicators do not conflict, however in video 02 strange things happen with the mouse indicator and the control indicator when they are placed together on the same chart. It is clearly visible that MetaTrader 5 treats each chart as an environment completely isolated from the others.

This type of observation is very important and will be used in the future, but there is another equally important point in video 02. Several things happen when MetaTrader 5 updates a chart in response to a custom event to change the timeframe. To understand this, you need to know how MetaTrader 5 works.

When we request a timeframe change (as shown in the video) MetaTrader 5 temporarily deletes everything from the chart and then restores the necessary things. For this reason, only indicators and Expert Advisors are restored on the chart. If the programmer does not take this into account, all programs will conflict. This is because the order in which MetaTrader 5 restores objects will most likely not match the order in which the user placed them on the chart. It is possible that the user places everything on the chart in such an order that nothing happens. But once something happens and MetaTrader 5 rebuilds the chart, the order may be different and then problems will start. Many may blame the platform, some the operating system, some God or the devil. But the real problem is that the program you or someone else created does not allow for multiple elements to be shared. And the fact that many codes are closed makes it even more difficult to understand the reasons for the conflict.

Let's get back to the main question. In this article I mentioned that the mouse indicator does not use any custom events. But it responds to two custom events: one is when the mouse is minimized, and the other is when its line is shown. As for the control indicator, it both uses and responds to the custom event, which serves, among other things, to initialize some elements in the control class.

Let's get back to the code now. Now I want you to pay very close attention to what I am about to explain. If you can understand this, you will have taken a big step in understanding how MetaTrader 5 works.

Control and mouse indicators are compiled separately. In the code we use enumerations that must be somehow related to the class that will respond to messages. However, this kind of reasoning means that we are still making some assumptions, and in programming you shouldn't make assumptions. For the compiler, this custom event code will always be a value shifted from a constant. The constant is CHARTEVENT\_CUSTOM, and since enumerations start with the default value (zero), the code for handling mouse indicator and control indicator messages both start with the same index - CHARTEVENT\_CUSTOM plus zero.

You might think that this doesn't explain the reason for the conflict, but you would be wrong. This explain the reason. Even more. This also explains how the MetaTrader 5 working environment is structured.

Each chart represents a working environment in MetaTrader 5. When a program sends a custom event to MetaTrader 5, and that event is not sent to the program or any other specific program, but is sent to MetaTrader 5. The platform will fire this event for any program present in the work environment, i.e. on the chart. So in video 02 you can see how this happens. This is because all programs within the chart, as soon as a custom event is triggered, receive the same notification from MetaTrader 5. Now think about an Expert Advisor that uses a custom event to open and close a position. If the same event uses an index equal to the event sending a signal in an indicator that is on the same chart as the EA, what happens when the Expert Advisor or indicator triggers a custom event? You will be in big trouble.

I'm trying to cover things step by step. The issue is much more complicated than it may seem, and we will make use of this same system for other purposes. That is why it is necessary that you understand what is happening. If you don't, you'll end up in an alley ending with a cliff.

You are probably asking yourself: is there no solution to this problem? There is, and it is quite simple, although it requires some knowledge. However, we will not dwell on it in this article, as we will have to make some changes and explain some points that will be too complicated at this stage.

### Conclusion

In this article, I began to outline the contents of the next articles. I know that for many this topic is very complex and difficult to understand right away. But it is important that you start preparing and studying this topic, which I have talked about in this article. We will consider the main issue in the next articles.

To better understand what has been discussed here, you can create small programs, such as simple indicators, that generate and respond to custom events. Place them on one or more charts. Observe how they behave when they are together and when they are apart. But this is, first of all, the golden key to this article: try to make one indicator change something in another indicator, both on the same chart and on different ones.

If you can't get them to interact via custom events when they're on different charts, don't worry. Don't think that you are a bad programmer. Perhaps you just don't have the necessary knowledge yet. I will not only show you how to do it, but also take it to another level of understanding, because as you may have noticed, I avoid as much as possible using some things, like external DLLs, to cover some of the shortcomings of MQL5, which so far has done a great job of what we do, without requiring fancy solutions.

Good luck in studying and see you in the next article! Be prepared for serious challenges ahead.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11932](https://www.mql5.com/pt/articles/11932)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11932.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11932/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/477010)**

![Trading Insights Through Volume: Moving Beyond OHLC Charts](https://c.mql5.com/2/102/Trading_Insights_Through_Volume_Moving_Beyond_OHLC_Charts___LOGO.png)[Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)

Algorithmic trading system that combines volume analysis with machine learning techniques, specifically LSTM neural networks. Unlike traditional trading approaches that primarily focus on price movements, this system emphasizes volume patterns and their derivatives to predict market movements. The methodology incorporates three main components: volume derivatives analysis (first and second derivatives), LSTM predictions for volume patterns, and traditional technical indicators.

![Creating a Trading Administrator Panel in MQL5 (Part VI):Trade Management Panel (II)](https://c.mql5.com/2/102/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_VI____Art2___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part VI):Trade Management Panel (II)](https://www.mql5.com/en/articles/16328)

In this article, we enhance the Trade Management Panel of our multi-functional Admin Panel. We introduce a powerful helper function that simplifies the code, improving readability, maintainability, and efficiency. We will also demonstrate how to seamlessly integrate additional buttons and enhance the interface to handle a wider range of trading tasks. Whether managing positions, adjusting orders, or simplifying user interactions, this guide will help you develop a robust, user-friendly Trade Management Panel.

![MQL5 Wizard Techniques you should know (Part 49): Reinforcement Learning with Proximal Policy Optimization](https://c.mql5.com/2/103/MQL5_Wizard_Techniques_you_should_know_Part_49___LOGO.png)[MQL5 Wizard Techniques you should know (Part 49): Reinforcement Learning with Proximal Policy Optimization](https://www.mql5.com/en/articles/16448)

Proximal Policy Optimization is another algorithm in reinforcement learning that updates the policy, often in network form, in very small incremental steps to ensure the model stability. We examine how this could be of use, as we have with previous articles, in a wizard assembled Expert Advisor.

![Price Action Analysis Toolkit Development (Part 2):  Analytical Comment Script](https://c.mql5.com/2/102/Price_Action_Analysis_Toolkit_Development_Part_2____LOGO.png)[Price Action Analysis Toolkit Development (Part 2): Analytical Comment Script](https://www.mql5.com/en/articles/15927)

Aligned with our vision of simplifying price action, we are pleased to introduce another tool that can significantly enhance your market analysis and help you make well-informed decisions. This tool displays key technical indicators such as previous day's prices, significant support and resistance levels, and trading volume, while automatically generating visual cues on the chart.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zmwaatwmdoqvetwiqwqjolboeqxlgdvu&ssn=1769184345884210170&ssn_dr=0&ssn_sr=0&fv_date=1769184345&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11932&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2053)%3A%20Things%20Get%20Complicated%20(V)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691843457921819&fz_uniq=5070004727711731289&sv=2552)

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