---
title: Developing a Replay System (Part 48): Understanding the concept of a service
url: https://www.mql5.com/en/articles/11781
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:08:13.686115
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11781&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070038391665397484)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 47): Chart Trade Project (VI)](https://www.mql5.com/en/articles/11760), we managed to make the Chart Trade indicator functional. Now we can focus again on what we actually need to develop.

At the beginning of this series of articles about the replay/simulator system, I spent some time trying to get the service to be able to place a control indicator on the chart. Although I didn't succeed at first, I didn't give up and kept trying. Despite numerous unsuccessful attempts, I was never able to succeed in this matter. But since the project could no longer be stopped, at that moment I decided to go a different way.

It really bothered me that I could do something with a script, but when I tried to do the same thing with a service, I couldn't get it to work properly.

You might think: "So what? The fact that you can do something with a script means nothing." However, if you think so, I'm afraid this is because of the lack of knowledge in MQL5 programming. Any script created in MQL5 can be converted into a service. Basically, there are two differences between a service and a script. Well, there are, of course, more, but these two are the most obvious and can be noticed by everyone.

The first difference is that the script is always linked to a specific chart and remains there until the chart is closed. It is noteworthy that when changing the timeframe, MetaTrader 5 actually sends a command to redraw the chart. To speed up this process, it closes the graphic object (not the window, but the object inside the window) and creates a new one. This allows it to quickly redraw the chart. However, the script is not relaunched on the chart because it does not have this function (since it doesn't have certain events).

So if you want to use a script to monitor a chart, you won't be able to use it directly on the chart. This can be done with the help of something that is outside the chart but able to observe it. This seems very complicated and confusing. However, in most cases this can be achieved by changing some details of the code and converting the script into a service. Then, already being a service, the script will no longer be linked to a specific chart but will be able to continue monitoring the asset chart.

You may find it difficult to understand the logic and reasons why this should happen. But without this it will be impossible to move on to further development phases. So creating a service that can monitor what's happening on the chart is critical to what we are going to do. Not because we will have to create something, but because we will need to observe.

You can use this article as a valuable research resource since the content it features will be essential to the replay/simulator system. In addition, it will be very useful for much more complex and detailed projects where we need to analyze the chart without being linked to it.

### Implementing the example script

To really understand how this works and the level of challenges we will face, let's start with a very simple example. Simple, but functional enough to serve as a basis for something even more complex to be implemented later.

Let's start by understanding things at the most basic level. So, first we will use a very good yet very simple script shown below:

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property version   "1.00"
04. //+------------------------------------------------------------------+
05. void OnStart()
06. {
07.     long id;
08.     string szSymbol;
09.     int handle;
10.     bool bRet;
11.     ENUM_TIMEFRAMES tf;
12.
13.     id = ChartID();
14.     szSymbol = ChartSymbol(id);
15.     tf = ChartPeriod(id);
16.     handle = iMA(szSymbol, tf, 9, 0, MODE_EMA, PRICE_CLOSE);
17.     bRet = ChartIndicatorAdd(id, 0, handle);
18.
19.     Print(ChartIndicatorName(id, 0, 0), " ", szSymbol, " ", handle, "  ", bRet, " >> ",id);
20.
21.     Print("*******");
22. }
23. //+------------------------------------------------------------------+
```

Example of script source code

Although the script is very simple, it will help me explain what we are actually going to do. In lines 02 and 03, we have data about the properties of the code. They are not that important for our explanation. Line 05 is where we actually start our code. Since this is a very simple script, the only function really needed is OnStart.

In lines 07 to 11 are our local variables. There is something important to note here. We don't actually need all these variables in one script. At least for what we are going to do. In fact, we won't need any of these variables. But because of line 19, we will use them here in the script.

Actually, the following happens in the script: in line 13 we get the identifier of the chart on which the script will be placed. Then in line 14, we take the name of the asset found on this chart. In line 15 we get the timeframe at which our script will be executed. All these three steps are optional. Although the step in line 13 is required, the steps presented in lines 14 and 15 are not mandatory for the script. However, it will soon become clear why they are here.

In line 16, we instruct MetaTrader 5 to place one of the standard technical indicators on the chart. In this case, it is a moving average. Now, please note the following: the added moving average will be of exponential 9-period MA calculated based on the closing price. If everything is done correctly, this indicator will return a handle.

In line 17, we use the handle created in line 16 and notify MetaTrader 5 that the indicator to which the handle belongs should be launched on the chart. In line 19, we print out all the values that were captured and used by the script. This will be a form of script debugging, which also serves another purpose that we will learn about shortly.

Please note that everything we did could have been done in many other ways. You're probably thinking, "What a useless code! Why would anyone want to create something like this?" I don't blame you for thinking like that. But when it comes to programming, nothing is useless, except when you create something completely pointless.

If you compile this script and place it on a chart, you will see that it will display a 9-period exponential moving average based on the closing price. Since the script code executes and completes very quickly, and is mainly intended to test something else, it does not contain any error checking. Any errors that occur will be visible in the MetaTrader 5 log.

Now the fun part begins. Let's think about this: if we replace line 16 with another one, for example:

```
handle = iCustom(szSymbol, tf, "Replay\\Chart Trade.ex5", id);
```

We can make our Chart Trade indicator be placed on the chart using a script. But what's the real benefit of this? As mentioned at the beginning of this article, things get interesting when we start using this script not only as a script, but also as a service.

### Converting a script to a service

Converting a script into a service is not the hardest thing. At least at this phase. This is because, essentially, the only difference in the code is the use of a property that indicates that the script is actually a service. And that's basically it. But despite the apparent simplicity, there are some details that need to be understood in order to do it right.

Let's return to our simple example. If you try to convert this script into a service, keeping the code unchanged and just adding a property indicating that it is a service, nothing will work. Why?

The reason is that, unlike a script, the service is not actually linked to a chart. And this circumstance somewhat complicates the situation.

To make things easier to understand, let's first understand how MetaTrader 5 works. Let's start by launching the platform without open charts. We can add services, but we can't do anything else. As soon as the first chart opens, we can place indicators, scripts and Expert Advisors on it. But we can also perform some manipulations with the chart using the service for this.

You've probably never tried this or seen anyone else do it. Up to this moment. But yes, you can make the service place certain elements on the chart. We'll get there, but let's not rush. Without properly understanding this basis that I am trying to show, you will not understand my new articles, because I am going to actively use a similar approach to speed up and simplify other things.

When adding charts in MetaTrader 5, if you do not close previous charts, a whole series of charts is created. But the order of access to them differs from the one in which they are organized in the MetaTrader 5 terminal. The order of access depends on the order in which they are opened. This is the first point. Luckily, MQL5 gives us the ability to navigate through this list of charts, and this will be important for us.

So, if you want to develop a way to add specific indicators to specific assets instead of creating a template, you will need to use a service. Seems complicated? It's actually not that difficult if you really understand what's going on.

Let's get back to the code from the previous topic where we added a 9-period exponential moving average using a script. Now let's do the following: when the first chart is opened, it should get this moving average. No matter what the asset will be. What is important is that this is the first chart that is opened. In this case, we will need to transform the same script into the service code shown below.

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property copyright "Daniel Jose"
04. #property version   "1.00"
05. //+------------------------------------------------------------------+
06. void OnStart()
07. {
08.     long id;
09.     string szSymbol;
10.     int handle;
11.     bool bRet;
12.     ENUM_TIMEFRAMES tf;
13.
14.     Print("Waiting for the chart of the first asset to be opened...");
15.     while ((id = ChartFirst()) < 0);
16.     szSymbol = ChartSymbol(id);
17.     tf = ChartPeriod(id);
18.     handle = iMA(szSymbol, tf, 9, 0, MODE_EMA, PRICE_CLOSE);
19.     bRet = ChartIndicatorAdd(id, 0, handle);
20.
21.     Print(ChartIndicatorName(id, 0, 0), " ", szSymbol, " ", handle, "  ", bRet, " >> ",id);
22.
23.     Print("*******");
24. }
25. //+------------------------------------------------------------------+
```

Source code of the service

Look at the code above against the code from the previous topic. What is the difference between them? You'll probably say that it's line 02 where there is now a property that defines this code as a service. This is true, but there is something else. We also have two different lines. Line 14 is where we output a message that the service is waiting for a chart to be placed in the MetaTrader 5 terminal, and we expect that the chart will open in line 15.

This is where things get a lot more interesting. To understand what is actually happening, write the code above and compile it using MetaEditor. Open MetaTrader 5, then close all open charts, absolutely all of them. After that, run the service we just created. Open the log window and you will see the message shown in line 14. If this happens, then you did everything right. Now open a chart of any symbol. You will immediately see the message from line 21 and the service will be closed. But looking at the chart, you will notice that there is a 9 period exponential moving average as expected. That's because the service put it there.

I hope you, dear reader, are following along carefully because now we're going to make things a little more complicated.

### Ensuring a standard chart

This is where the really fun part of using the services begins. Many people like to use a standardized graphical model for which they create templates. The preset template helps us to some extent in this regard. It ensures that all charts are up to standard at the time they are created. But after placing the template, we can accidentally delete some indicator or change the settings of one of them. This is not a problem in many cases. You can simply restore an indicator and that's it. But is it possible to use MetaTrader 5 avoiding this issue? So that this platform would always be something very close to the Bloomberg terminal.

Yes, it is possible. In fact, we can prevent the user or ourselves from deconfiguring anything in the terminal. In this particular case I am talking about indicators. In previous articles, I showed how you can force the system to support a particular indicator on the chart. This has been done for quite some time in this replay/simulator system to avoid removing the control indicator from the chart. But there is a more universal solution that allows you to keep any things on the chart, or prevent others from appearing.

As far as prevention goes, I don't think there's much point in showing how it's done, but as far as keeping goes, I think that's really interesting. So, changing the code shown in the previous topic again, we get the code shown below:

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property copyright "Daniel Jose"
04. #property version   "1.00"
05. //+------------------------------------------------------------------+
06. void OnStart()
07. {
08.     long id;
09.     string szSymbol;
10.     int handle;
11.
12.     Print("Waiting for the chart of the first asset to be opened...");
13.     while (!_StopFlag)
14.     {
15.             while (((id = ChartFirst()) < 0) && (!_StopFlag));
16.             while ((id > 0) && (!_StopFlag))
17.             {
18.                     if (ChartIndicatorName(id, 0, 0) != "MA(9)")
19.                     {
20.                             handle = iMA(szSymbol = ChartSymbol(id), ChartPeriod(id), 9, 0, MODE_EMA, PRICE_CLOSE);
21.                             ChartIndicatorAdd(id, 0, handle);
22.                             IndicatorRelease(handle);
23.                     }
24.                     id = ChartNext(id);
25.             }
26.             Sleep(250);
27.     }
28.     Print("Service finish...");
29. }
30. //+------------------------------------------------------------------+
```

Improved service code

You may notice that this code contains elements that seem strange. However, each of the elements has its reason to be presence in the code. The code itself is not much different from what was discussed in the previous topic, but it is capable of doing something quite interesting. Let's see what this code does, then you can adapt it to your own needs and interests.

In line 12 we output a message that the service is waiting. Just like in line 28, we are notifying that the service has been removed and is no longer running. Everything that happens between these two lines is what really interests us. To make sure that the service closes correctly without causing problems in MetaTrader 5, we check the **\_StopFlag** constant, which will be true until we request the service to be closed.

So, in line 13 we enter an infinite loop. This loop is not actually infinite, because the moment we signal that we want to close the service, this loop will be terminated.

In line 15 we have everything the same as in the previous code. But now we are adding a check to avoid problems when closing the service. Please pay attention: When **ChartFirst** returns a value, this will be the value of the first window you opened in the MetaTrader 5 terminal. The value will be in the "id" variable. Remember this fact, because from now on it is important to take into account the sequence of actions.

In line 18 we check if the window contains the **iMA** indicator, which is the 9-period moving average. If this condition is not met, that is, the MA is not found on the chart, we will add it. To do this, we first create a handler in line 20, and then in line 21 we add the MA to the chart. Once this is done, we delete the handle as we no longer need it. Thus, if you remove the MA from the chart, it will be added back. This will be done automatically while the service is running.

Line 24 will look for the next window. Now notice that we need the index of the current window so that MetaTrader 5 can know what the next index will be. **Don't confuse things**. If another window is found, the loop will repeat. It will repeat until **ChartNext** returns a value of **-1**. When the loop that started in line 16 will terminate.

Of course, we don't want the service to act like a mindless lunatic. So, in line 26 we generate a small delay, about 250 milliseconds, before the next iteration of the loop. This way the service will run 4 times per second, ensuring that the specified moving average will always be present on any of the charts that may be open in the MetaTrader 5 terminal. Things like this are very interesting, aren't they?

But we can do a little more. We can force elements to be placed on a specific chart for a specific symbol. This is exactly what I have tried to achieve many times since the beginning of this series of articles, but for one reason or another it did not work out. Now let's say you, a trader on the B3 (Brazilian Stock Exchange), decide that you need to use a certain indicator only for a specific symbol. And always only for that symbol. One of the ways to do this is to use a template. But it is very likely that in the daily hustle and bustle you will end up closing this symbol, and when you open the chart again you will not notice the much needed indicator is missing until it is too late.

When using the service, this will not happen, because as soon as the symbol chart is opened in the MetaTrader 5 terminal, the service will take care of adding the indicator to it. If you want to use a template to add additional elements, you can do that too. But using the template will be covered another time, as it involves some things we will cover in future articles within this series.

So, let's see how to do this. And to avoid confusion, let's consider this in a new topic.

### Ensuring terminal quality

If you found what you saw in the previous topic interesting, get ready because this one is going to get even more interesting. Because we will be able to do exactly what we want, that is, we will be able to make it so that the symbol chart will always correspond to the pre-configured terminal. It's almost as if we could make MetaTrader 5 look like the **Bloomberg Terminal**, but with the advantage that the configuration will never be changed accidentally, since it will not be associated with a template, but will be part of the service configuration.

To illustrate, let's look at how the service code discussed in the previous topic has been modified. The complete code is shown below：

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property copyright "Daniel Jose"
04. #property version   "1.00"
05. //+------------------------------------------------------------------+
06. void OnStart()
07. {
08.     long id;
09.     string szSymbol;
10.     int handle;
11.     ENUM_TIMEFRAMES tf;
12.
13.     Print("Waiting for the chart of the first asset to be opened...");
14.     while (!_StopFlag)
15.     {
16.             while (((id = ChartFirst()) < 0) && (!_StopFlag));
17.             while ((id > 0) && (!_StopFlag))
18.             {
19.                     szSymbol = ChartSymbol(id);
20.                     tf = ChartPeriod(id);
21.                     if ((StringSubstr(szSymbol, 0, 3) == "WDO") && (ChartWindowFind(id, "Stoch(8,3,3)") < 0))
22.                     {
23.                             handle = iStochastic(szSymbol, tf, 8, 3, 3,MODE_SMA, STO_CLOSECLOSE);
24.                             ChartIndicatorAdd(id, (int)ChartGetInteger(id, CHART_WINDOWS_TOTAL), handle);
25.                             IndicatorRelease(handle);
26.                     }
27.                     if (ChartIndicatorName(id, 0, 0) != "MA(9)")
28.                     {
29.                             handle = iMA(szSymbol, tf, 9, 0, MODE_EMA, PRICE_CLOSE);
30.                             ChartIndicatorAdd(id, 0, handle);
31.                             IndicatorRelease(handle);
32.                     }
33.                     id = ChartNext(id);
34.             }
35.             Sleep(250);
36.     }
37.     Print("Service finish...");
38. }
39. //+------------------------------------------------------------------+
```

Service code

Note that the changes are small, but the results are striking. In our case, for demonstration, we will use a symbol from B3 (Brazilian Stock Exchange). It is a mini-dollar futures contract. This contract expires every month, but that's not a problem. To understand this, let's move on to an explanation of the code itself. It's not difficult, but you need to pay attention to details, otherwise you won't get what you expect.

I won't cover what I've already covered, so let's focus on the new parts of the code. In lines 19 and 20, we enter the data of the chart that the service will check. We need the name of the symbol, as well as the chart timeframe, to avoid mistakes in the future. We then put these values into two variables.

In line 21, we make two checks:

- the first one concerns the name of the symbol we want to configure
- the second is the presence or absence of a specific indicator

If both checks are true, we will execute further code. Now, notice that we only use the first three letters of the symbol name because this is a futures contract with an expiration date. Since we don't want to constantly change the name of the symbol, we use the common part of the symbol name regardless of the current contract.

The second point to pay attention to is related to the indicator. We must specify its short name. If you don't know this name, just place this indicator on the chart and use a simple script to print its name to the terminal. This way you will know exactly what name should be used there. In this case, we use a stochastic indicator with 8 periods, an MA of 3 and a constant of 3. But if the indicator is different, change the **string** to the correct name. Otherwise, we will have problems in the next step.

So, let's return to the situation when the indicator is not on the chart of the specified symbol. In this case, we will first execute line 23, where we set up the desired indicator. Then, in line 24, we add the indicator to the new subwindow. Be careful at this point: if you specify the subwindow number, MetaTrader 5 will place the indicator in the specified window. This can lead to confusion. But according to the existing code, MetaTrader 5 will create a new subwindow and add the indicator to it.

Line 25 frees the handle because we don't need it anymore. Everything that was described will happen only if the indicator is not present on the chart. If you try to delete it, the service will force MetaTrader 5 to place it on the chart again. But if it is already present on the chart, the service will not be activated and will simply watch what is happening.

**Important note:** Although the code works and has adequate performance, for practical reasons it is recommended to change the check shown in line 27 to something more efficient. When using multiple indicators on a single chart, it is best to test by name rather than by index as is done in the code. So, simply replace line 27 with the following code:

```
27.     if (ChartIndicatorGet(id, 0, "MA(9)") == INVALID_HANDLE)
```

Thus, no matter what index the indicator will have, only one indicator will be present on the chart – the one placed that will be placed on the chart by the service.

### Conclusion

In this article, I have shown you something that few people actually know how to do. If this is the case for you, I was very happy to show you that MetaTrader 5 can be a much more powerful tool than it seems. This is perhaps the best platform out there for those who really know what they are doing and are able to use all of its capabilities.

In the video you can see how MetaTrader 5 behaves when using the service from this article. But beyond what has been shown here, and this is just an introduction to what we still have to do, I am very glad that you, dear readers, were curious after reading the article and tried to learn a little more. However, I would like to emphasize that learning consists not only of reading educational articles, but also of independently finding solutions.

Try to get the most out of everything. And when you reach the limit of what is possible, look for more qualified people to find a solution, while always trying to surpass what has already been done before. This is the essence of the evolutionary path, and this is how you become a real professional.

YouTube

Demo video

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11781](https://www.mql5.com/pt/articles/11781)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11781.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11781/anexo.zip "Download Anexo.zip")(420.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/474870)**
(1)


![Sergei Lebedev](https://c.mql5.com/avatar/2019/7/5D3E1392-839A.jpg)

**[Sergei Lebedev](https://www.mql5.com/en/users/salebedev)**
\|
4 May 2024 at 13:34

Highly interesting [project](https://www.mql5.com/en/articles/7863 "Article: Projects let you create profitable trading robots! But it's not exactly"), but reading all 48 articles to get the clue for every component is really hard work.

I kindly ask you to write next one 49 article "User Guide" and lay out in it on each component of this project.

Also - if this project is ready for practical implementation, please add practical example of using it. like "Replaying and re-trading Brexit case on GBPUSD in educational purpose". It is expected that in this practical example, this project will be used to retrieve historical data of 'GBPUSD' from real symbol (of any connected forex account) plus/minus 1 day from Brexit, using extracted data to feed custom symbol like 'repGBPUSD' in replay mode, adding some generic indicators (RSI(14), MA(50) etc) and providing user real-time experience of re-trading this historical event.

This User Guide with practical example of real time re-trading Brexit will be really great finalisation of this project!

![Creating a Trading Administrator Panel in MQL5 (Part IV): Login Security Layer](https://c.mql5.com/2/98/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_IV__Logo.png)[Creating a Trading Administrator Panel in MQL5 (Part IV): Login Security Layer](https://www.mql5.com/en/articles/16079)

Imagine a malicious actor infiltrating the Trading Administrator room, gaining access to the computers and the Admin Panel used to communicate valuable insights to millions of traders worldwide. Such an intrusion could lead to disastrous consequences, such as the unauthorized sending of misleading messages or random clicks on buttons that trigger unintended actions. In this discussion, we will explore the security measures in MQL5 and the new security features we have implemented in our Admin Panel to safeguard against these threats. By enhancing our security protocols, we aim to protect our communication channels and maintain the trust of our global trading community. Find more insights in this article discussion.

![MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://c.mql5.com/2/98/MQL5_Trading_Toolkit_Part_3___LOGO.png)[MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)

Learn how to develop and implement a comprehensive pending orders EX5 library in your MQL5 code or projects. This article will show you how to create an extensive pending orders management EX5 library and guide you through importing and implementing it by building a trading panel or graphical user interface (GUI). The expert advisor orders panel will allow users to open, monitor, and delete pending orders associated with a specified magic number directly from the graphical interface on the chart window.

![MQL5 Wizard Techniques you should know (Part 43): Reinforcement Learning with SARSA](https://c.mql5.com/2/98/MQL5_Wizard_Techniques_you_should_know_Part_43___LOGO.png)[MQL5 Wizard Techniques you should know (Part 43): Reinforcement Learning with SARSA](https://www.mql5.com/en/articles/16143)

SARSA, which is an abbreviation for State-Action-Reward-State-Action is another algorithm that can be used when implementing reinforcement learning. So, as we saw with Q-Learning and DQN, we look into how this could be explored and implemented as an independent model rather than just a training mechanism, in wizard assembled Expert Advisors.

![Visualizing deals on a chart (Part 1): Selecting a period for analysis](https://c.mql5.com/2/79/Visualization_of_trades_on_a_chart_Part_1_____LOGO.png)[Visualizing deals on a chart (Part 1): Selecting a period for analysis](https://www.mql5.com/en/articles/14903)

Here we are going to develop a script from scratch that simplifies unloading print screens of deals for analyzing trading entries. All the necessary information on a single deal is to be conveniently displayed on one chart with the ability to draw different timeframes.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/11781&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070038391665397484)

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