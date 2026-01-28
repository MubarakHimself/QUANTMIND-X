---
title: Developing a Replay System (Part 57): Understanding a Test Service
url: https://www.mql5.com/en/articles/12005
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:42:07.092350
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/12005&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069669359485389005)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 56): Adapting the Modules](https://www.mql5.com/en/articles/12000), we made some changes to both the control indicator module and, most importantly, the mouse indicator module.

Since the article already contained quite a lot of information, I decided not to add any more new data, as it would most likely only confuse you, dear reader, instead of helping to clarify and explain how things actually work.

The attachment to the previous article contains the mouse indicator and the service. When you run the service, it will create a custom symbol and will add a panel with a mouse indicator and a control indicator. Both modules are placed on the custom symbol chart, and although they do not perform any service related actions, some interaction activity can be seen between the user and these two modules.

You may not have any idea how to do this, especially if you are new to MQL5. If you look at the code of these two indicators, you may not see any activity that would prevent the loss of data present in the indicator, or more precisely, in the control module.

So, this article will focus on explaining how the service actually works. This explanation is of paramount importance because a proper understanding of how the service works is essential to understanding how the replay/simulator system works. This is because it is always easier to create and explain code with fewer components than to try to understand code with a much more complex structure right away.

So, even though we will not actually use the code that will be explained next, it is very important to understand it in detail. The whole basis of how the control module, mouse module and service interact will be much better understood if you understand this simpler code well.

So, without further ado, let's take a look at the source code of the service that was implemented and shown in the previous article. Let's try to understand the video that was provided at the end of that article.

### Let's analyze the service code

The entire source code for the service is as follows:

```
01. //+------------------------------------------------------------------+
02. #property service
03. #property copyright "Daniel Jose"
04. #property description "Data synchronization demo service."
05. #property link "https://www.mql5.com/en/articles/12000"
06. #property version   "1.00"
07. //+------------------------------------------------------------------+
08. #include <Market Replay\Defines.mqh>
09. //+------------------------------------------------------------------+
10. #define def_IndicatorControl   "Indicators\\Market Replay.ex5"
11. #resource "\\" + def_IndicatorControl
12. //+------------------------------------------------------------------+
13. #define def_Loop ((!_StopFlag) && (ChartSymbol(id) != ""))
14. //+------------------------------------------------------------------+
15. void OnStart()
16. {
17.    uCast_Double info;
18.    long id;
19.    int handle;
20.    short iPos, iMode;
21.    double Buff[];
22.    MqlRates Rate[1];
23.
24.    SymbolSelect(def_SymbolReplay, false);
25.    CustomSymbolDelete(def_SymbolReplay);
26.    CustomSymbolCreate(def_SymbolReplay, StringFormat("Custom\\%s", def_SymbolReplay));
27.    CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_SIZE, 0.5);
28.    CustomSymbolSetDouble(def_SymbolReplay, SYMBOL_TRADE_TICK_VALUE, 5);
29.    Rate[0].close = 110;
30.    Rate[0].open = 100;
31.    Rate[0].high = 120;
32.    Rate[0].low = 90;
33.    Rate[0].tick_volume = 5;
34.    Rate[0].time = D'06.01.2023 09:00';
35.    CustomRatesUpdate(def_SymbolReplay, Rate, 1);
36.    SymbolSelect(def_SymbolReplay, true);
37.    id = ChartOpen(def_SymbolReplay, PERIOD_M30);
38.    if ((handle = iCustom(ChartSymbol(id), ChartPeriod(id), "\\Indicators\\Mouse Study.ex5", id)) != INVALID_HANDLE)
39.       ChartIndicatorAdd(id, 0, handle);
40.    IndicatorRelease(handle);
41.    if ((handle = iCustom(ChartSymbol(id), ChartPeriod(id), "::" + def_IndicatorControl, id)) != INVALID_HANDLE)
42.       ChartIndicatorAdd(id, 0, handle);
43.    IndicatorRelease(handle);
44.    Print("Service maintaining sync state. Version Demo...");
45.    iPos = 0;
46.    iMode = SHORT_MIN;
47.    while (def_Loop)
48.    {
49.       while (def_Loop && ((handle = ChartIndicatorGet(id, 0, "Market Replay Control")) == INVALID_HANDLE)) Sleep(50);
50.       info.dValue = 0;
51.       if (CopyBuffer(handle, 0, 0, 1, Buff) == 1) info.dValue = Buff[0];
52.       IndicatorRelease(handle);
53.       if ((short)(info._16b[0]) == SHORT_MIN)
54.       {
55.          info._16b[0] = (ushort)iPos;
56.          info._16b[1] = (ushort)iMode;
57.          EventChartCustom(id, evCtrlReplayInit, 0, info.dValue, "");
58.       }else if (info._16b[1] != 0)
59.       {
60.          iPos = (short)info._16b[0];
61.          iMode = (short)info._16b[1];
62.       }
63.       Sleep(250);
64.    }
65.    ChartClose(id);
66.    SymbolSelect(def_SymbolReplay, false);
67.    CustomSymbolDelete(def_SymbolReplay);
68.    Print("Finished service...");
69. }
70. //+------------------------------------------------------------------+
```

Source code of the test service

To make sure everyone can really understand what's going on, even those who don't have much experience with MQL, and especially those who aren't very familiar with how to program services for MetaTrader 5 yet, let's take a closer look at the code, starting from line 2.

When you come across a property declared in the second line in MQL5 code, you should understand that it is a service. If this property were missing, the code would have to be treated as a script. The main differences between a service and a script are, first of all, the presence or absence of this property in the code. But from the point of view of execution, the main difference is that the script will always be linked to a chart, while the service does not depend on any.

The remaining lines, between 3 and 6, are well known to those who have at least minimal knowledge of MQL5 programming. So, we can skip their description.

In line 8, we added the include directive to include the header file in this code. Note that in this case the header file will be located in the "Include" folder inside the "Market Replay" folder, named "Defines.mqh". The "Include" folder is located in the root directory of the main MQL5 directory, which will be referenced by MetaEditor during compilation.

Now we need to pay special attention to lines 10 and 11. These lines ensure that the indicator, or rather the control module, becomes an internal resource of the compiled service code. What does this mean? This means that when you migrate already compiled code, you don't need to also migrate the control module code, since it will already be embedded as a service resource. For this reason, the attachment to the previous article only contained two executable files, although at the time the service was launched, three files were actually running.

But why didn't we also include the indicator, or rather the mouse module, as a service resource, just like we did with the control module? The reason is simple: to allow the user to use the mouse module and make it easier to access. If this module were embedded in the service executable, it would be difficult for the user to access the mouse module to place it on a different chart than the one created by the service.

For this reason, we often have to make decisions about which elements, and especially why, should be made internal resources of a particular executable.

Next look at line 13. In this line, we declare a definition in order to simplify, or better yet, standardize, some of the tests we will be performing. In complex programs, it is quite common to run the same type of test in different places. Creating a definition to standardize such tests, in addition to making the code simpler and easier to maintain, ensures that we always perform the same type of testing. In many cases, this is highly desirable for most programmers, as it is possible to forget to modify some test point, and then for some reason the code performs well a particular test but fails at other points. This usually causes a lot of headaches.

All right. In line 15, we actually get into the executable part of the code. Note that OnStart is the same entry point for both scripts and services in the MQL5 code. However, from the declaration in line 2 we know that we are dealing with a service. MetaTrader 5 will only generate one OnStart event call each time the code is executed, so some maneuvering will be required to ensure the code runs for the required period of time. This will only happen in a specific case, which will be in line 47. But first we need to do a few more things.

Namely, we need to initialize and adapt the service so that it can do something useful for us. Among the necessary steps is the declaration and initialization of variables and conditions that will be imposed by our service in MetaTrader 5. All this will be done after the service is launched. With this in mind, in lines 17 to 22 we declare the variables we will use and move on to properly initializing the required rules. Much of what will be discussed from this point on may seem strange.

These are seemingly simple things that could originally be part of another type of application, but because we want to centralize everything in one code, we have to work this way.

So, in line 24, we tell MetaTrader 5 that the symbol we are referring to should be removed from the Market Watch window. Immediately after that, in line 25, we remove it from the list of symbols. This list contains all the symbols that we can access. But what we're really interested in is line 26. In this line, we tell MetaTrader 5 that we want to create a custom symbol and specify where it should be created. You should pay attention to this because if you are not careful, you can overwrite the market symbol during this action and later, while trying to access the real instrument, you will actually be accessing a custom one. But usually we always take some precautions to prevent this from happening.

Lines 26 and 27 are mandatory; without them, the mouse module would not be able to correctly set the price line to the correct position as you move the mouse over the chart.

Now, between lines 29 and 34, we define the bar that will be the first bar visible on the chart. For reasons I don't fully understand, this bar is always positioned in a way that prevents you from seeing the highs and lows. But since our goal here is to simply display the bar on the chart and prevent the modules from generating any range errors, we don't really care whether the entire bar is visible or not.

Up until this point, we have absolutely nothing on the chart, not even the chart itself. In line 35, we tell MetaTrader 5 that the values defined in the previous lines where we describe the bar should be placed as the first bar of the asset we are creating. In line 36, we tell the platform that the asset should be placed in the Market Watch window. If the code ended at this point, we could manually open the custom symbol and display the bar placed inside it. But since we want even more automation, we have line 37.

At the moment of execution of line 37, MetaTrader 5 will open a chart with the specified symbol and timeframe. At some point in this series I already explained the reason for informing modules about the Chart ID. It is very important to remember that we now have a chart open, and since we do not specify which template should be used, the chart will open using the default template. This idea is necessary for understanding the following lines.

Let's start with this: notice that in line 38 we are trying to generate a handle to try and place the mouse module on the chart. Especially note the location and name of the executable file specified. If the attempt to create a handle fails for any reason, line 39 will not be executed. Usually the reason is that the executable file is not in the specified location, I will explain this in more detail later. But if the handle is successfully created, then in line 39 we will add the module to the chart. This way the indicator will be available, not necessarily visible, but will be shown in the list of indicators running on the chart.

So, in line 40 we tell MetaTrader 5 that the handle is no longer needed, so the memory allocated for it can be returned to the system. However, since the module is already on the chart, MetaTrader 5 will not remove it from the chart unless you specifically tell the terminal to do so.

This same script could be removed and the effect would be the same, i.e. the mouse module would be placed on the chart. This would happen if the chart template contained the module. To do this, you could open any chart, add a mouse indicator to it, and then save this template as Default.tpl. Or, to make it clearer, you can now create a default template that already includes a mouse indicator. This would eliminate the need for having commands shown on lines 38 to 40, while still allowing us to place the mouse indicator in a location that is most convenient for us.

In the case of the control module, the situation is a little different. This is due to the fact that the control module is integrated into the service executable file. The simple fact that this happens makes it easier for the service to place the module on the chart. This is done in lines 41 and 42. In line 41, we generate a handle to access the module, and in line 42 we add it to the chart. Note that if we don't use line 42, the module won't be placed on the chart, it will only be loaded into memory, but MetaTrader 5 won't run it on the chart we need.

In line 43, we remove the handle since we don't need it anymore, just like we did in line 40.

Until this point, the service behaves like a program that will soon terminate. But before that, we use line 44 to print a message to the message box so we know how far we have progressed in execution. Then, between lines 45 and 46, we initialize the last variables we're actually going to use. Note that we initialize these variables with values that will be used to start the control module. It has just been launched on the chart but has not yet been initialized, and for this reason it is not displayed on the chart.

Finally, in line 47 we enter a loop. This loop will terminate if any of the conditions defined in line 13 are met, which will cause the condition to become false and the loop will terminate. From this moment on, we will no longer do things in any random way. From this point on, the service will stop performing actions and will be responsible for managing what is already running. It is important to keep this concept in mind and know how to make this distinction. Otherwise, you could be trying to do something that should not be in the loop, which will again make everything very unstable and problematic.

Then we see a new loop in line 49. This loop from line 49 is a dangerous type of loop if not planned properly. The reason is that if we didn't use the definition in line 13, we could get stuck in this loop indefinitely. This is because the control module may not be present on the chart and the service will wait for MetaTrader 5 to tell it the handle value before it can access the indicator or control module.

However, deleting the chart control module will cause MetaTrader 5 to close the chart. This makes the definition in line 13 false, so the loops in line 49 and line 47 will terminate.

That's why I explain how it all works using a simpler system as an example. It would be much more difficult to notice and understand such subtleties in a more complex system. So whenever you want to test something, do it with some simple program that follows the same logic of operation as the system that will be designed later.

So, assuming that MetaTrader 5 returns us a valid handle, in line 50 we zero out the value that we will define if the reading of the control indicator buffer in line 51 is successful.

If reading the buffer fails for any reason, the variable will be zeroed. But if reading is successful, we will have data that is in the control indicator buffer.

Once we receive the data, we can remove the handle. This is done in line 52. The reason is that anything can happen in the next steps, and we don't want to get false when the loop repeats. It may seem like this will slow down the overall performance of the system, but it's better to lose a little performance than to analyze data that may be invalid or unnecessary.

Now pay attention to the following: the indicator, or rather the control module, has not yet been initialized. However, its buffer contains the values that were placed there during the first stage of the indicator operation. To understand this, please refer to the previous articles in this series. So you shouldn't expect null values to be returned when reading the buffer, this won't happen.

We have two conditional tests for this. One allows us to initialize the control module so that it knows which controls to display. The second test allows us to store in the service all the information about what is happening in the control module. This is done so that when we need to report the last working state of the module again, we know what values to pass to it.

So, in line 53 we check if the module has been added to the chart. This can happen at two points. The first is when we are still in the first execution of the loop started at line 47. In this case, the values to be reported to the control module come from lines 45 and 46. The second point is when MetaTrader 5 has to reset and re-place the control module on the chart, because the chart timeframe has changed for some reason. In this case, the last values that were in the control indicator before MetaTrader 5 placed it back on the chart will be used.

But anyway, at the end we will execute line 57, which will cause MetaTrader 5 to generate a custom event on the chart monitored by the service. This way the control indicator will get new values or old values, giving the impression that somehow nothing was magically lost. This will only happen because in the indicator, in the control module, we have placed a certain value in the buffer indicating that the indicator wants and should receive updated values from an external source. In this case, it is the service that operates in the MetaTrader 5 platform.

Now, if it is not the content coming from the indicator buffer, we have a new check performed in line 58 where we check if the value is non-zero. This allows us to understand if the control indicator is in the play mode or in the pause mode. In the demo service, both situations indicate that the service should capture the content of the indicator buffer and save the value that will be used if MetaTrader 5 decides to remove and re-place the indicator back on the chart due to a chart timeframe change.

Since execution can happen very quickly and we don't need a super-fast update, we use line 63 to give the service some rest during which it won't do anything important. This line also tells us that the loop that started in line 47 has ended.

Additionally, when the user closes the service or the chart is closed for any reason, we exit the loop that started in line 47 and execute line 65 first. In this line, we close the chart, because if the user terminates the service, the chart will still remain open. To avoid this, we tell MetaTrader 5 that it can now close the chart we are using.

After closing the chart, we try to remove the created custom symbol from the market watch window. This is done in line 66. If there are no other open charts with this symbol, we can force MetaTrader 5 to remove it from the Market Watch. However, even after deletion, it will still appear in the location where it was created, on line 26.

If the symbol has indeed been removed from the market watch, we will use line 67 to remove it from the list of symbols available in the system. And finally, in line 68, if everything went well and all procedures were performed as expected, we will print a message in the terminal message window, informing that the service has terminated and is no longer running.

### Conclusion

This type of situation, although it may seem unimportant or useless in the long run, actually provides useful and necessary knowledge for what we will really need later. Often, when developing a solution or application, we have to deal with unknown or unfavorable scenarios. So whenever you need to design, program, or develop something that might be much more complicated than originally thought, create a small program that is easy to understand and learn, but which nevertheless does some of the work you need to do in the long run.

If you do everything right, you will be able to take part of the small program you created and use it in your real big and complex project. For this reason, I decided to write this article. So that you understand that a program, sometimes extremely complex and requiring different levels of knowledge, is not just created out of thin air.

It is always created in stages, in functional parts, and then part of the idea is used in a larger solution. And that's exactly what we'll be doing with our replay/simulator system. The way the service described here works gives us exactly what we need to implement our major system.

Please note that without the proper analysis and research that this article has allowed us to do, it is unlikely that we would have been able to get the modules to work properly in the replay/simulator system with minimal effort. We would have to make a lot of adjustments and changes to the code, but they would all focus on one point: the C\_Replay class. But without knowing where, how, and in what way the C\_Replay class should be adapted, we would end up in a dead end, making the implementation of what we see here extremely difficult.

Now that we know how the system will actually behave, all we need to do is make a few small changes (I hope) to the C\_Replay class so that the loop at lines 47 to 64 can become part of our system code. But there's a small point: unlike what we see here, in the replay/simulator system we will perform calculations to follow what has already been displayed in the chart. But this will be discussed in the following articles. Take your time to study what is shown in this article, understand and study how the system works in order to better understand everything that will be presented further. We will make the replay/simulator service use the modules that were shown in the video in the previous article.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12005](https://www.mql5.com/pt/articles/12005)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12005.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12005/anexo.zip "Download Anexo.zip")(420.65 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/480877)**
(3)


![Levison Da Silva Barbosa](https://c.mql5.com/avatar/avatar_na2.png)

**[Levison Da Silva Barbosa](https://www.mql5.com/en/users/1226819)**
\|
20 Jul 2024 at 14:07

very good lessons.


![Alexey Viktorov](https://c.mql5.com/avatar/2017/4/58E3DFDD-D3B2.jpg)

**[Alexey Viktorov](https://www.mql5.com/en/users/alexeyvik)**
\|
20 Dec 2024 at 18:10

Very interesting how it could happen that the article is published today

![](https://c.mql5.com/3/450/2720137046472.png)

And the remarks were already in July

![](https://c.mql5.com/3/450/1839299278884.png)

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
22 Dec 2024 at 10:21

**Alexey Viktorov [#](https://www.mql5.com/ru/forum/478587#comment_55435375):**

Very interesting how it could happen that the article is published today

And the remarks were already in July

This is a translation of the article and the comments are synchronised.


![Build Self Optimizing Expert Advisors in MQL5 (Part 5): Self Adapting Trading Rules](https://c.mql5.com/2/116/Automating_Trading_Strategies_in_MQL5_Part_5___LOGO2.png)[Build Self Optimizing Expert Advisors in MQL5 (Part 5): Self Adapting Trading Rules](https://www.mql5.com/en/articles/17049)

The best practices, defining how to safely us an indicator, are not always easy to follow. Quiet market conditions may surprisingly produce readings on the indicator that do not qualify as a trading signal, leading to missed opportunities for algorithmic traders. This article will suggest a potential solution to this problem, as we discuss how to build trading applications capable of adapting their trading rules to the available market data.

![Automating Trading Strategies in MQL5 (Part 5): Developing the Adaptive Crossover RSI Trading Suite Strategy](https://c.mql5.com/2/115/Automating_Trading_Strategies_in_MQL5_Part_5___LOGO.png)[Automating Trading Strategies in MQL5 (Part 5): Developing the Adaptive Crossover RSI Trading Suite Strategy](https://www.mql5.com/en/articles/17040)

In this article, we develop the Adaptive Crossover RSI Trading Suite System, which uses 14- and 50-period moving average crossovers for signals, confirmed by a 14-period RSI filter. The system includes a trading day filter, signal arrows with annotations, and a real-time dashboard for monitoring. This approach ensures precision and adaptability in automated trading.

![Neural Networks in Trading: Reducing Memory Consumption with Adam-mini Optimization](https://c.mql5.com/2/85/Reducing_memory_consumption_using_the_Adam_optimization_method___LOGO.png)[Neural Networks in Trading: Reducing Memory Consumption with Adam-mini Optimization](https://www.mql5.com/en/articles/15352)

One of the directions for increasing the efficiency of the model training and convergence process is the improvement of optimization methods. Adam-mini is an adaptive optimization method designed to improve on the basic Adam algorithm.

![From Basic to Intermediate: Variables (II)](https://c.mql5.com/2/85/Do_b8sico_ao_intermedixrio__Varipveis_II___LOGO.png)[From Basic to Intermediate: Variables (II)](https://www.mql5.com/en/articles/15302)

Today we will look at how to work with static variables. This question often confuses many programmers, both beginners and those with some experience, because there are several recommendations that must be followed when using this mechanism. The materials presented here are intended for didactic purposes only. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/12005&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069669359485389005)

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