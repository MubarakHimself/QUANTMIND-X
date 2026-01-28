---
title: Developing a Replay System (Part 37): Paving the Path (I)
url: https://www.mql5.com/en/articles/11585
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:16:05.223565
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/11585&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070147986345889986)

MetaTrader 5 / Examples


### Introduction

In this article, we will touch on a very important issue. I suggest that you focus as much as possible on understanding the content of this article. I mean not simply reading it. I want to emphasize that if you do not understand this article, you can completely give up hope of understanding the content of the following ones.

Everything that I am going to show, explain and tell in this article will be essential for you to at least understand the following articles to a minimal extent.

It's not a joke. Because I think a lot of people don't even know what we're capable of. We use purely MQL5 to force different elements to behave in a certain way in MetaTrader 5.

All this time I have been writing articles to explain to you, dear reader, how to do certain things in MetaTrader 5. I have been putting off talking about a certain topic, beating around the bush, trying to talk about what I have been doing for a long time in my programs.

These things allow me to modulate my system in such a way that I often have to work more on modifying the source code than on displaying it using only MQL5.

All the codes you have seen all this time are actually modified codes. They are nothing like the ones I originally used, but now I'm at a crossroads and don't know what else to do. Either I show the original code, or the one very close to what I actually use, or I won't be able to continue with the replay/simulation system project.

The problem isn't even the writing of the code itself. In my opinion, this is quite simple to do. The problem is that I want to give you the necessary knowledge so that you can not only use the system, but also change it. This is necessary so that you can make it behave as it should and is desirable for you, so that you can conduct your own analysis.

I want everyone to learn to work on their own, fully, without outside help. I hope this is clear.

What we will do in this article will show how we can, using indicators and some Expert Advisors, have very specific behavior and very modular programming, even to the point of creating some kind of complex system. But here's the problem: we'll compile as few things as possible.

The replay/simulation system currently contains 3 executables: an Expert Advisor, a service, and an indicator. And yet this allows us to develop the order system. Remember that such a system will replace the trading server, so we can run analysis by trading on a demo or live account.

We still have to bring some things from the past into the replay/modeling system. This applies, for example, to Chart Trader. Therefore, we need some kind of system, but it needs to be stable and easy to use. Although we did this in the article [Developing a trading Expert Advisor from scratch (Part 30): CHART TRADE as an indicator?!](https://www.mql5.com/en/articles/10653), this doesn't suit us.

We need things to be done much more transparently and without causing us any inconvenience.

We can do this much better and show it with a simpler example. If we go straight to the desired application, I'm almost absolutely sure that few, if any, will be able to keep up with what we are discussing. But I don't want this to happen. So, let's now see how to proceed. Get ready to enter my UNIVERSE.

### Starting to build the indicator

What we will do is very difficult from the point of view of a person who does not know how MetaTrader 5 works. I won't add any attachments at the end. I want you to follow the explanations so that you know that you can experience what you see here and learn in detail how everything works.

Let's begin by creating an indicator. Then you should open MetaEditor and do the following:

![Figure 01](https://c.mql5.com/2/49/001__1.png)

Figure 01 - Select what to create.

![Figure 02](https://c.mql5.com/2/49/002.png)

Figure 02 - Create a directory.

![Figure 03](https://c.mql5.com/2/49/003.png)

Figure 03 - Name the indicator.

![Figure 04](https://c.mql5.com/2/49/004.png)

Figure 04 - Keep things simple.

As you can see in Figure 04, we create the same for whatever type of indicator we are going to make. At this point in Figure 04, we may sometimes have one or the other, but in either case, we can add or change event management functionality later if needed. So don't worry, you can continue creating the indicator as usual.

![Figure 05](https://c.mql5.com/2/49/005.png)

Figure 05 - Click "Finish"

As a result, you will see something like in Figure 05. At this point we just need to click the "Finish" button.

The idea is to create an indicator to be placed in the ZERO window. However, the same concept can be used to place an indicator in any window. But again, don't worry. Pay attention to the main idea of the article.

At the end of this process, you will see the following code in the MetaEditor window:

```
#property copyright "Daniel Jose"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
  {
//---

//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

I know many people prefer to use MetaEditor's native formatting. This is understandable, but for me this format is less pleasant to use. But this is everyone's personal matter. What's really important is that we can easily read our code and understand it.

So, let's start modifying this standard code into what we need.

If you try to compile this standard code, you will get a warning.

![Figure 06](https://c.mql5.com/2/49/006.png)

Figure 06 - Compiler output.

Although Figure 06 shows that the code was compiled, it was not generated in exactly the right way. Many programmers ignore the fact that the compiler warns them about errors. This can be seen by the red arrow.

The fact that an error occurs, even a seemingly non-critical one, can put your code at risk. Therefore, under no circumstances should you use code if the compiler has reported it to be defective. NEVER do this.

To solve this problem, let's do something very simple: tell the compiler that we know what the use of the indicator will do.

```
#property copyright "Daniel Jose"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_plots 0
//+------------------------------------------------------------------+
int OnInit()
{
        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
{
        return rates_total;
}
//+------------------------------------------------------------------+
```

By adding a highlighted line, we let the compiler know that we know what we're doing. The result is shown in Figure 07.

![Figure 07](https://c.mql5.com/2/49/007.png)

Figure 07 - Result of ideal compilation.

Whenever you compile code, pay attention to cases where the compiler is telling you the same thing as in Figure 07. This is the only way to get perfectly compiled code without critical problems.

You might think that you already know all that. Okay, but I want to be clear. I don't want anyone to complain that the system doesn't work or works in the wrong way because of some modification. I would like to encourage everyone to modify and adapt the system to their own needs. And to achieve this, you first need to know the basics, have a solid and well-formed foundation of knowledge and ideas.

Now that we have a basic indicator, let's create a basic Expert Advisor.

### Starting to build the Expert Advisor

We'll build an EA to implement what we need. Let's do this step by step.

![Figure 08](https://c.mql5.com/2/49/010.png)

Figure 08 - What we are going to build.

![Figure 09](https://c.mql5.com/2/49/011.png)

Figure 09 - Determine the directory to use.

![Figure 10](https://c.mql5.com/2/49/012.png)

Figure 10 - Determine the name of the executable file.

As you can see, we follow all the same steps as in the indicator stage, the only difference is the choice we made at the beginning of creation.

![Figure 11](https://c.mql5.com/2/49/013.png)

Figure 11 - Leave default things.

![Figure 12](https://c.mql5.com/2/49/014.png)

Figure 12 - Click Finish.

After we complete all these steps, a file will appear in the MetaEditor, as can be seen below:

```
#property copyright "Daniel Jose"
#property link      ""
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

As with the indicator, I will also make some formatting changes here. But unlike the indicator, when we ask MetaEditor to compile this code, we will receive the following information from the compiler.

![Figure 13](https://c.mql5.com/2/49/015.png)

Figure 13 - Result of the compilation request.

What does it mean? Calm down, dear reader. Do not worry. You'll soon understand what we're going to do.

Now that we have a basic system for the indicator and Expert Advisor, let's do what we started this article for: connect them.

It is important to know how to create an indicator or how to create an Expert Advisor, and also to understand how to make them interact and work together. Many people know the basics that allow the indicator to be used so that the Expert Advisor can use the data calculated by the indicator.

If you don't know how to do this, I recommend you learn it first. A good starting point is the article: [Developing a trading Expert Advisor from scratch (Part 10): Accessing custom indicators](https://www.mql5.com/en/articles/10329). It gives a very simple explanation of how to access the values calculated by the indicator. At the same time, we will look at how to initialize the indicator in a very simple way, even before we can access any information that was calculated depending on what we are going to do.

All this is very beautiful and simple, because using the idea in Figure 14, you can read the contents of the indicator. Now, using Figure 15, you can create an indicator using an Expert Advisor for this.

![Figure 14](https://c.mql5.com/2/49/017.png)

Figure 14 - EA reading the indicator data.

![Figure 15](https://c.mql5.com/2/49/016.png)

Figure 15 - Creating an indicator by an EA.

But despite this, this method does not work when we need to do something like what we did using Chart Trader. When we turned Chart Trader into an indicator, we had to use another path. Even at that time I already mentioned that this would change in the future. In that article, we used a scheme shown in Figure 16.

![Figure 16](https://c.mql5.com/2/49/008.png)

Figure 16 - Bidirectional communication.

When it is necessary to transfer information between different parties, or rather between different processes, we need certain means for this. In MetaTrader 5, we can use global terminal variables. This same concept has long been used in the replay/simulation system. This ensures that the service can somehow interact with the indicator.

There are other ways to promote the same communication, but I do not want to use techniques that will not allow me to take full advantage of all the benefits of MQL5 and MetaTrader 5. When we use the platform and its language to its fullest, we can benefit from any future improvements. If we start inventing solutions that don't take full advantage of the features offered, how can we benefit if MQL5 or even MetaTrader 5 improves?

Although the system shown in Figure 16 is very suitable for many situations, it does not provide much benefit.

To understand this, let's try to understand the following fact: How many indicators should we put on a chart to get a good experience in the replay/simulation system? One? Two? Five? The truth is that it is not really possible to know this. But here's what we already know: We will need at least two indicators, and this is for now, not for later. We need them now. We need at least two indicators in the replay/simulation system.

You may be wondering why two. The first indicator is used to control the replay/simulation. We already have this indicator. Now we need another one: Chart Trader. It will allow us to place orders directly. We need Chart Trader because, unlike the physical market, in replay/simulation we do not need the buttons in the upper corner (Fig. 17).

![Figure 17](https://c.mql5.com/2/49/018.png)

Figure 17 - Quick Trader buttons.

We do need a way to place orders, but it should be the same whether we use a DEMO or REAL account. We should have the same method for interacting with the platform.

Although Figure 16 enables interaction between the EA and the indicator to send orders, it is not the best way. Because in this case we need to keep global variables that could be better used for other purposes. We can take a completely different path to establish exactly the communication we need.

Based on this, we began to apply a new approach, as can be seen in Figure 18.

![Figure 18](https://c.mql5.com/2/49/009.png)

Figure 18 - New communication protocol.

Looking at this figure, you might think I got crazy. How are we going to make the EA and the indicator communicate directly? Where's the catch in this story? No, this is not a joke, and no, I'm not crazy. The fact is, there are means to make this communication directly. This can be seen in Figures 14 and 15. But in addition to these means, we have one more at our disposal. What we really need is to understand how to best use these means.

Before I continue, let me ask you: Have you ever tested or tried to use MetaTrader 5 in a way different than everyone else does? Have you experimented with MQL5 to the point where you said that certain things could be done that way? If the answer to either of these two questions is NO, I suggest you take a look and see how far the rabbit hole goes.

### Initiating communication between processes

As you could see from the previous topics, here we are going to build a way of communication between the indicator and the Expert Advisor so as not to use global terminal variables, except in special cases. This will enable efficient information exchange.

The idea is simple, and the concept is even simpler. However, without understanding some details, you will **NOT** succeed. It is very important to properly understand what I am about to explain to you. Don't think that you already know how to program in MQL5 and it's enough. What I'm about to explain goes beyond the normal and familiar use of both MetaTrader 5 and the MQL5 language.

We should start with certain things – it will be like creating a protocol. No matter what you think about it, follow the path and the results will be achieved. Deviate and you will fail.

In the indicator code, we start by adding the first lines as shown below.

```
 1. #property copyright "Daniel Jose"
 2. #property link      ""
 3. #property version   "1.00"
 4. #property indicator_chart_window
 5. #property indicator_plots 0
 6. //+------------------------------------------------------------------+
 7. #define def_ShortName       "SWAP MSG"
 8. #define def_ShortNameTmp    def_ShortName + "_Tmp"
 9. //+------------------------------------------------------------------+
10. input double user00 = 0.0;
11. //+------------------------------------------------------------------+
12. long m_id;
13. //+------------------------------------------------------------------+
14. int OnInit()
15. {
16.     m_id = ChartID();
17.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortNameTmp);
18.     if (ChartWindowFind(m_id, def_ShortName) != -1)
19.     {
20.              ChartIndicatorDelete(m_id, 0, def_ShortNameTmp);
21.             Print("Only one instance is allowed...");
22.             return INIT_FAILED;
23.     }
24.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
25.
26.     return INIT_SUCCEEDED;
27. }
28. //+------------------------------------------------------------------+
29. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
30. {
31.     return rates_total;
32. }
33. //+------------------------------------------------------------------+
```

Looks like a mess? But there is no confusion here. At this point I force MetaTrader 5 to make sure that there is only one indicator on the chart. How do I do this? In simple terms, I check whether the indicator is on the chart or not.

If desired, you can force MetaTrader 5 to keep only one indicator on the chart using the code above. But to understand how it works, you need to break it down by listing and explaining the key lines to make it easier to follow the entire explanation.

In line 7, we define what name our indicator will have. This name will be stored in MetaTrader 5 as long as the indicator is on the chart.

To determine this name, we use the method in line 24. Here we set the name that the indicator will use. The reason we define line 8 is because of the way MetaTrader 5 works.

MetaTrader 5 is an event-based platform. This means that when any event occurs, be it a price movement, a change in time on the chart, a mouse movement, a key press, or the addition or removal of something from the chart, MetaTrader 5 triggers some type of event. Every type of event has a purpose and consequence.

When MetaTrader 5 triggers an update event, all objects, indicators, Expert Advisors and other elements that may be present on the chart must be updated in some way. In the case of scripts, they are simply reset from the chart, and in the case of indicators and EAs, they have a new call to the OnInit function. If something happens and the indicator has to be updated from scratch, MetaTrader 5 will force line 14 to be called. What's the problem? The problem occurs because we need the indicator to have a short name that is recognized by MetaTrader 5, otherwise we will not be able to perform the check shown in line 18.

You might think that we could just move line 24 before line 18, but therein lies the problem. If we add line 24 before line 18, when the indicator marked in line 18 is already present on the chart, we will get a positive result. This will cause MetaTrader 5 to reset the indicator from the chart, when in fact we want it to remain when line 20 is executed. But we want only one instance of it to be present.

I hope you're following the explanation. For the reason described above, we need a temporary name, defined in line 8, where we make a small change to the short name of the indicator before checking in line 18, if the indicator is already on the chart. We use line 17 to set the temporary name of the indicator. NOTE: This name must be unique, otherwise there will be problems.

If the indicator is already present on the chart, line 18 will allow line 20 to be executed, thereby removing the indicator that is trying to get on the chart. To notify the user about the error, we have line 21 which displays a message in the MetaTrader 5 message box. The indicator will return in line 22, indicating that it was not possible to display it on the chart.

If the indicator gets on the chart, line 24 will correct the indicator name, so it will be basically impossible to place a new indicator on the chart. But there is a "loophole" that is not actually a MetaTrader 5 gap. Fortunately, MetaTrader 5 can differentiate between things when we are adding a new indicator to the chart. To understand this, there is code in line 10.

If the user DOES NOT MODIFY the value of the variable when placing the indicator on the chart (the variable is declared on line 10), MetaTrader 5 will understand that the indicator is the same as the one that is already on the chart, if there is already one running there. If the user modifies the value, MetaTrader 5 can make two completely different decisions based on whether it is the same indicator or a new one is being added.

- The first case is when there is no indicator, it will be placed on the chart.
- The second is when the indicator already exists. In this case, if the value specified in the variable differs from the indicator already displayed on the chart, then MetaTrader 5 will perceive it as a different indicator. If the value is identical, then MetaTrader 5 will recognize them as the same indicator.

This approach allows you to limit the number of indicators with the same name on the chart to only one. If desired, you can allow a maximum number of indicators with the same name. All you need to do is change the check in line 18. Thus, it is possible to configure MetaTrader 5 so that it accepts, for example, 3 indicators with the same name. However, if the user tries to place a fourth indicator with the same name, MetaTrader 5 will prohibit this. More precisely, this will be done by the indicator code, which will prevent this attempt to place a fourth indicator.

As you can see, we can customize and limit the system to such an extent that we do not allow duplication of indicators on the chart. This is very important and will be discussed in more detail later in this series on the replay/simulation system.

There are several points in the code of this indicator that I will explain later so that the interaction between the EA and the indicator is clear without using global terminal variables. I am talking about lines 5 and 10. It is important to understand the logic behind this type of programming.

In order to understand what is actually happening, you can create the code described in detail above and run it on the MetaTrader 5 platform.

### Conclusion

In this article, you showed you how to block or restrict the placement of more than one indicator on a chart by creating and executing the described code in the MetaTrader 5 platform. You can use this knowledge to make many other tasks easier, which will make your life much easier as a trader and user of the MetaTrader 5 platform, providing you with a better experience with it.

Although the reason for this may not be clear from this article, the knowledge presented, if used correctly, will allow us to do much more and avoid a number of problems associated with the presence of duplicate or even unnecessary indicators on the chart.

It is not uncommon for a user with little experience with MetaTrader 5 to place the same indicator on the chart multiple times, which makes using and configuring these indicators extremely frustrating. And all this can be avoided using fairly simple code (as you saw for yourself). It is quite capable of doing what it is designed to do.

I hope this knowledge will be useful to you. In the next article we will look at how we can take the next step in terms of creating direct communication between the Expert Advisor and the indicator, which is necessary to further implement our replay/simulation system.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11585](https://www.mql5.com/pt/articles/11585)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11585.zip "Download all attachments in the single ZIP archive")

[swap.mq5](https://www.mql5.com/en/articles/download/11585/swap.mq5 "Download swap.mq5")(1.24 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/466367)**

![The Group Method of Data Handling: Implementing the Combinatorial Algorithm in MQL5](https://c.mql5.com/2/76/The_Group_Method_of_Data_Handling___LOGO.png)[The Group Method of Data Handling: Implementing the Combinatorial Algorithm in MQL5](https://www.mql5.com/en/articles/14804)

In this article we continue our exploration of the Group Method of Data Handling family of algorithms, with the implementation of the Combinatorial Algorithm along with its refined incarnation, the Combinatorial Selective Algorithm in MQL5.

![Neural networks made easy (Part 68): Offline Preference-guided Policy Optimization](https://c.mql5.com/2/62/midjourney_image_13912_49_444__1-logo.png)[Neural networks made easy (Part 68): Offline Preference-guided Policy Optimization](https://www.mql5.com/en/articles/13912)

Since the first articles devoted to reinforcement learning, we have in one way or another touched upon 2 problems: exploring the environment and determining the reward function. Recent articles have been devoted to the problem of exploration in offline learning. In this article, I would like to introduce you to an algorithm whose authors completely eliminated the reward function.

![MQL5 Wizard Techniques you should know (Part 17): Multicurrency Trading](https://c.mql5.com/2/76/MQL5_Wizard_Techniques_you_should_know_wPart_17m_Multicurrency_Trading___LOGO.png)[MQL5 Wizard Techniques you should know (Part 17): Multicurrency Trading](https://www.mql5.com/en/articles/14806)

Trading across multiple currencies is not available by default when an expert advisor is assembled via the wizard. We examine 2 possible hacks traders can make when looking to test their ideas off more than one symbol at a time.

![Developing a Replay System (Part 36): Making Adjustments (II)](https://c.mql5.com/2/60/Replay_1Parte_36q_Ajeitando_as_coisas_LOGO.png)[Developing a Replay System (Part 36): Making Adjustments (II)](https://www.mql5.com/en/articles/11510)

One of the things that can make our lives as programmers difficult is assumptions. In this article, I will show you how dangerous it is to make assumptions: both in MQL5 programming, where you assume that the type will have a certain value, and in MetaTrader 5, where you assume that different servers work the same.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/11585&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070147986345889986)

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