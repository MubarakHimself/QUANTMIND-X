---
title: Developing a Replay System (Part 38): Paving the Path (II)
url: https://www.mql5.com/en/articles/11591
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:15:53.586156
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/11591&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070145795912569016)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 37): Paving the Path (I)](https://www.mql5.com/en/articles/11585), we have seen a simple way to prevent the user from duplicating an indicator on the chart. In the article, we verified that with a very simple code consisting of just a few lines, the MetaTrader 5 platform will help us avoid the appearance of a second indicator on the chart if we need it in a single copy. It is important that the indicator is not duplicated.

I hope you understand the idea and how the desired result was achieved. Under no circumstances should the indicator be duplicated.

While confirming the principle of no duplication in MetaTrader 5, please note that two-way communication between the indicator and the Expert Advisor is still a long way off. We are still very far from this. However, the guarantee that the indicator will not be duplicated on the chart at least gives peace of mind. Because when the indicator and the EA interact, we will know that we are dealing with the right indicator.

In this article we will start to look at things through the eyes of the EA. But we will also have to make some changes to the indicator, because the indicator that was in the previous article, unfortunately, cannot react to the EA at all. This happens when the EA wants to know something.

### Creating the first interaction between processes

To provide a good explanation of what's going on, we need to create some control points. Let's start with the indicator code. Don't worry, it's all easy to understand.

The full indicator code is shown below:

```
01. #property copyright "Daniel Jose"
02. #property link      ""
03. #property version   "1.00"
04. #property indicator_chart_window
05. #property indicator_plots 0
06. //+------------------------------------------------------------------+
07. #define def_ShortName       "SWAP MSG"
08. #define def_ShortNameTmp    def_ShortName + "_Tmp"
09. //+------------------------------------------------------------------+
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
20.             ChartIndicatorDelete(m_id, 0, def_ShortNameTmp);
21.             Print("Only one instance is allowed...");
22.             return INIT_FAILED;
23.     }
24.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
25.     Print("Indicator configured with the following value:", user00);
26.
27.     return INIT_SUCCEEDED;
28. }
29. //+------------------------------------------------------------------+
30. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
31. {
32.     return rates_total;
33. }
34. //+------------------------------------------------------------------+
```

Note that the only addition to the code is line 25. This will be our verification line, at least for the first time. At the moment we are interested in how the EA interacts with the indicator and whether we can send information or data to it.

Note that in line 10, the type of information expected by the indicator is _**double**_. This is done on purpose because the goal is to create a mechanism that can replace the global terminal variable system for communication between the EA and the indicator. If the means developed or used turn out to be unsuitable, then we can easily replace this model with a model of terminal global variables.

You can also use other types of data to provide communication, at least for the transfer of data from the EA to the indicator.

Now that we have changed the indicator code, we can move on to the EA code. It is shown in full below:

```
01. #property copyright "Daniel Jose"
02. #property link      ""
03. #property version   "1.00"
04. //+------------------------------------------------------------------+
05. input double user00 = 2.0;
06. //+------------------------------------------------------------------+
07. int m_handle;
08. long m_id;
09. //+------------------------------------------------------------------+
10. int OnInit()
11. {
12.     m_id = ChartID();
13.     if ((m_handle = ChartIndicatorGet(m_id, 0, "SWAP MSG")) == INVALID_HANDLE)
14.     {
15.             m_handle = iCustom(NULL, PERIOD_CURRENT, "Mode Swap\\Swap MSG.ex5", user00);
16.             ChartIndicatorAdd(m_id, 0, m_handle);
17.     }
18.     Print("Indicator loading result:", m_handle != INVALID_HANDLE ? "Success" : "Failed");
19.
20.     return INIT_SUCCEEDED;
21. }
22. //+------------------------------------------------------------------+
23. void OnDeinit(const int reason)
24. {
25.     ChartIndicatorDelete(m_id, 0, "SWAP MSG");
26.     IndicatorRelease(m_handle);
27. }
28. //+------------------------------------------------------------------+
29. void OnTick()
30. {
31. }
32. //+------------------------------------------------------------------+
```

For many, this EA code may seem quite strange. Let's see how it works. This is the first step of what we really need to do.

On line 05 we give the user the opportunity to set the value passed to the indicator. Remember: we want to test and understand how the communication will happen. Lines 07 and 08 contain two internal global variables in the EA code. I usually don't like using global variables, but in this case you can make an exception. The contents of these variables will be determined in the code that handles the **_OnInit_** command which starts on line 10.

This may seem a bit complicated. This is because there are a few things we need to understand at this stage.

On line 12, we initialize a variable that will indicate the index of the timeframe in the EA is running. Then comes line 13, where beginners often encounter problems. Many people simply don't add this line to their code. The absence of this line will not immediately break your code, but its inclusion prevents certain types of errors from occurring. Many of these errors are RUN-TIME errors that can occur at any time, making them difficult to resolve.

Now that we have added code on line 13, if MetaTrader 5 already has the indicator that we need on the chart, it will return the _**handle**_ of this indicator." Thus, we do not need to bother placing the indicator on the chart.

But if line 13 shows that the desired indicator is not on the chart, we will run the code that will add the indicator to the chart. Remember the following: the indicator that we need and will use is a custom one. Therefore, we call [iCustom](https://www.mql5.com/en/docs/indicators/icustom), which can be seen on line 15.

Now comes the tricky part: why is iCustom, present on line 15, declared this way? Do you have any idea why? To understand this, let's look at what the documentation says:

```
int  iCustom(
                 string            symbol,   // symbol name
                 ENUM_TIMEFRAMES   period,   // period
                 string            name      // folder/custom indicator name
                 ...                         // list of indicator parameters
             );
```

From the table above you can see that the first parameter of the function is the name of the symbol. But in line 15 of the code the first parameter is **_NULL_**. Why is this? Why didn't we use a the _**\_Symbol**_ constant? The reason is that for the compiler, **_NULL_** and **_\_Symbol_** are one and the same. This way, the indicator will be created using the same symbol that the EA chart has.

The next parameter in the list is the chart period. Again we have something different from what many expect. On line 15 we use the value _**PERIOD\_CURRENT**_, but why? The reason is that we want the indicator to remain synchronized with the same period as the Expert Advisor. One detail: if you want the indicator to analyze at a different chart period, simply specify the desired period in this parameter. Thus, the indicator will be fixed at a certain period, and the EA will be able to move through different periods.

When it comes to custom indicators, I believe that it is the third parameter that can lead to serious consequences. Many people don't know what to put here and sometimes leave it empty or specify the wrong location because they don't understand the meaning. If we specify the wrong location, MetaTrader 5 will not be able to find the indicator we need.

Look at what's declared on line 15, in that third parameter. We specify not just a name, but a name and a path. Where does this information come from? To find out, we'll have to go back to the previous article and find out where this information came from. See Figure 01.

![Figure 01](https://c.mql5.com/2/49/003__1.png)

Figure 01 - Creating an indicator

In Figure 01, you can see where the information for the third parameter comes from. These are practically the same things. With the following changes: first we delete the root directory, in this case it will be _**indicators**_. This is because MetaTrader 5 will first look for the indicator in this directory. The second point is that we have added the _**.ex5**_ file. This way we will get the correct location of the executable file. When MetaTrader 5 places the EA on the chart and executes line 15, it will know where and which indicator to use.

Below we'll look at some of the issues and problems that may arise. But first let's get the basics out of the way.

This third parameter and the following ones are optional. If we want to provide any value to the indicator, we must do so from the fourth parameter. But you must ensure that this happens in the same sequence in which the parameters appear on the indicator. You also need to make sure that they are of the correct type. You cannot input **double** if the indicator expects to receive a **float** or **long** value. In this case, MetaTrader 5 will generate an error immediately after launching the indicator.

There are other ways to pass parameters to the indicator, but for now, for simplicity, we will use iCustom. If you look at the indicator code, you will see that in line 10 a **double** value is expected. This value is provided to the indicator through the EA, in the fourth parameter on line 15 of the EA code.

Now comes an important question that many people neglect, which leads to problems in the entire system. It is line 16 in the EA code. Why is this line so important?

There are cases where this line can be ignored when it comes to custom indicators. This is especially true for those indicators that we need when we want MetaTrader 5 to support only one instance placed on the chart. We cannot ignore line 16 in any way or form and when line 16 is present, line 25 must also be added.

These two lines avoid problems. Line 16 will run the indicator on the chart. Thus, if the user tries to run another instance of the indicator on the same chart, MetaTrader 5 along with the indicator code will prevent the new instance from running. If you remove line 16 from the EA code, the user may accidentally place a new instance of the indicator on the chart, causing the user, not the EA, to become confused about the calculations that the indicator performs.

With calculated indicators we have a certain problem, but in the case of an indicator like ChartTrader, the problem is even more complex. Since in this case we are dealing with graphic objects. It is important to understand how to do everything right. Then it will be easier to understand how to do more complex things.

I think it's clear now how important line 16 is. What does line 25 do? It will remove the specified indicator from the chart. Please note one thing: we need to specify the name of the indicator to be deleted. It is placed in the indicator on line 24 of the indicator code. The names must be the same, otherwise MetaTrader 5 will not understand which indicator needs to be deleted. Additionally, in some cases we also have to tell which subwindow is used. Since we are not using a subwindow, we leave this value equal to **ZERO**.

Then, on line 26, we remove the indicator handler because we no longer need it.

Let's now see how MetaTrader 5 works with this scheme of interaction between the EA and the indicator. To understand this, you need to do a few things. Therefore, pay close attention to the details of the process. Also, you should test each modification and try the generated code, because each of the changes will bring MetaTrader 5 its own way of handling the situation. It is important not just to read the above explanations but to understand what is happening. And to understand what happens when an EA is written in a certain way, or when the same EA is programmed in a different way, it is necessary to consider three situations. The code used in the EA and indicator is the same as the one we have explained so far, the only difference is in the presence or absence of the EA code line.

Now do the following:

- First, compile the code with all the lines and see how the system works in MetaTrader 5.
- Then delete lines 16 and 25 and retest the system in MetaTrader 5.
- Finally, delete only line 16 and leave line 25. After that test the system again.

You might think that all this is nonsense and that you, as an experienced programmer, would never make such mistakes. But knowing what each line of code does and in what it results when running MetaTrader 5 is perhaps more important than it seems.

If you have reached this stage, then you understand the second phase, in which we can transfer data from the EA to the indicator without using global terminal variables. You have taken a big step towards becoming a quality programmer. But if you still don't understand, don't be discouraged. Go back to the beginning of this or the previous article and try to understand this step. Now things get very complicated, and I'm not kidding. We need to make the indicator send data to the EA.

Here the indicator sends data to the Expert Advisor. In some cases, this can be quite simple. But anyone who follows my articles has probably noticed that I like to take things to the extreme, make the computer sweat and ask it to do things differently. This is because I really want both the language and the platform to work at the limit of what is possible.

### IndicatorCreate or iCustom - which one to use?

Many beginners may wonder how to call custom indicators. This can be done in two ways. We covered the first one in the previous topic, where we explained the use of iCustom. But there is another way, which can be more attractive in some types of modeling, while for others, using iCustom will be sufficient.

This is not about having to use one method or another. My purpose here is to show how you can use the [IndicatorCreate](https://www.mql5.com/en/docs/series/indicatorcreate) call to be able to load your own indicators.

Before we begin, let's understand one thing. The IndicatorCreate function is not a special function. This is actually a basic function. What does this mean? A basic function is a function that essentially serves to promote another function, but with one detail: "derived" functions use the basic function but make it easier to use.

This is because there is no need to model a call, as we would expect from a basic function. We can use simpler modeling. This is how other functions arise. You can see them in the documentation as [Technical indicators](https://www.mql5.com/en/docs/indicators). Among these indicators is iCustom, which is essentially a derived function that simplifies modeling.

So, what would be the source code of the Expert Advisor discussed at the beginning of the article if we used IndicatorCreate instead of iCustom? Its code can be seen below:

```
01. #property copyright "Daniel Jose"
02. #property link      ""
03. #property version   "1.00"
04. //+------------------------------------------------------------------+
05. input double user00 = 2.2;
06. //+------------------------------------------------------------------+
07. int m_handle;
08. long m_id;
09. //+------------------------------------------------------------------+
10. int OnInit()
11. {
12.     MqlParam params[];
13.
14.     m_id = ChartID();
15.     if ((m_handle = ChartIndicatorGet(m_id, 0, "SWAP MSG")) == INVALID_HANDLE)
16.     {
17.             ArrayResize(params, 2);
18.             params[0].type = TYPE_STRING;
19.             params[0].string_value = "Mode Swap\\SWAP MSG";
20.             params[1].type = TYPE_DOUBLE;
21.             params[1].double_value = user00;
22.             m_handle = IndicatorCreate(NULL, PERIOD_CURRENT, IND_CUSTOM, ArraySize(params), params);
23.             ChartIndicatorAdd(m_id, 0, m_handle);
24.     }
25.     Print("Indicator loading result:", m_handle != INVALID_HANDLE ? "Success" : "Failed");
26.
27.     return INIT_SUCCEEDED;
28. }
29. //+------------------------------------------------------------------+
30. void OnDeinit(const int reason)
31. {
32.     ChartIndicatorDelete(m_id, 0, "SWAP MSG");
33.     IndicatorRelease(m_handle);
34. }
35. //+------------------------------------------------------------------+
36. void OnTick()
37. {
38. }
39. //+------------------------------------------------------------------+
```

Compare the differences between the code above and the one we showed at the beginning of the article. Both do the same thing. The main question is: why does this code that uses the IndicatorCreate call need to be executed this way?

The iCustom feature actually hides exactly what we need to do, but let's figure out what's going on.

First, we need to declare a variable, which is declared on line 12 as an array. This should be done exactly this way. If the indicator is not present on the chart, we will need to create it. To do this, we must tell the IndicatorCreate function what to do and how to do it.

In addition to the number of parameters that we use in the indicator, we must also tell the IndicatorCreate function the name of the indicator. Thus, if we are going to enter just the name of the indicator, the ArrayResize function in line 17 should select not two, but one element. If we passed 5 parameters to the indicator, we would have to assign 6 elements, and so on. Let's see how many parameters we need to send and assign an additional position.

So, now we need to configure this array. Here line 22 with IndicatorCreate function indicates that we're going to use a custom indicator. This is done using the [IND\_CUSTOM](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_indicator) enumeration. There are several things that we need to do. On line 18 we see how to proceed for the first position of the array. Remember that MQL5 is close to C++, so we start counting from zero. The information type in the first position must be STRING, so the declaration is written as shown on line 18.

On line 19 we enter the name of the custom indicator used. Note that the indicator name is the same as what would be used if we were using the iCustom call. The only difference is that we do not specify the file extension. This must be an executable file with an extension .ex5, so specifying the extension is optional.

IMPORTANT: What is done in lines 18 and 19 must be done every time the custom indicator is used. If you use any other indicator, whatever it may be, then as soon as you start setting up the array data, you will have to specify the parameters as you will see it further, that is, starting from line 20 onwards.

Now that we have told the IndicatorCreate function which custom indicator to use, let's start filling out the parameters that will be passed to it. These parameters must be specified in the correct order. No matter which indicator you use, declaration of incorrect types will result in runtime errors. Therefore, be very careful when filling out this information.

On line 20 we specify that the first parameter will be of type double. But it can be any of the types defined in the [ENUM\_DATATYPE](https://www.mql5.com/en/docs/constants/indicatorconstants/enum_datatype) enumeration. You can use any of them, but there are some things to pay attention to.

Types **TYPE\_DOUBLE** and **TYPE\_FLOAT** will be used for _**double\_value**_, in this case on line 21.

If line 20 had used the type **TYPE\_STRING**, then in line 21 the value would be cast to the _**string\_value**_ variable. For any of the other possible types declared on line 20, we enter the value on line 21 using an _**integer\_value**_ variable.

This is very important, so let's make this information clear. For a better understanding, take a look at the table below:

| Identifier used | Data type | Where to put the value |  |  |
| --- | --- | --- | --- | --- |
| TYPE\_BOOL | Boolean | integer\_value |  |  |
| TYPE\_CHAR | Char | integer\_value |  |  |
| TYPE\_UCHAR | Unsigned Char | integer\_value |  |  |
| TYPE\_SHORT | Short | integer\_value |  |  |
| TYPE\_USHORT | Unsigned Short | integer\_value |  |  |
| TYPE\_COLOUR | Color | integer\_value |  |  |
| TYPE\_INT | Integer | integer\_value |  |  |
| TYPE\_UINT | Unsigned Integer | integer\_value |  |  |
| TYPE\_DATETIME | DateTime | integer\_value |  |  |
| TYPE\_LONG | Long | integer\_value |  |  |
| TYPE\_ULONG | Unsigned Long | integer\_value |  |  |
| TYPE\_FLOAT | Float | double\_value |  |  |
| TYPE\_DOUBLE | Double | double\_value |  |  |
| TYPE\_STRING | String | string\_value |  |  |

Table or correspondence.

This table clearly shows what we will use on line 20 (the identifier used) and which variable will receive the value on line 21. We must do this for each of the parameters that will be passed to our custom indicator. Since we're only using one parameter, we'll only work with that one.

Note that using the IndicatorCreate call is much more labor intensive than its derivative, iCustom, in our specific case.

After we have set all the parameters that we want to define, we go to line 22, where we actually call the function. This is done so that we can increase or reduce things easily. All fields are filled in so that you won't have to edit line 22 again if you need to change the indicator or number of parameters.

The idea is to always simplify things rather than complicate them.

Because of this extra work to set up the same type of execution in MetaTrader 5, we won't often see the IndicatorCreate function used in current code. But nothing prevents you from using it.

Before concluding this article, I would like to briefly talk about one more function: [IndicatorParameters](https://www.mql5.com/en/docs/series/indicatorparameters). This function allows checking something about an unknown indicator. Assume that you have several different indicators on your chart, each of which has been initialized in a specific way. You may want to automate some indicator-based strategy, but since the market changes so suddenly, it might take a few minutes to get all the indicators set up correctly.

To speed up the process a little, you can use the IndicatorParameters function. Once this function is called, MetaTrader 5 will populate it to tell us exactly how a particular indicator is configured. Then another function, usually IndicatorCreate, is used to change the configuration of this indicator. If the EA uses this indicator to buy or sell, it will immediately understand what to do because we have a trigger.

This issue has been discussed in detail and demonstrated in a series on trading automation. In the article " [Creating an EA that works automatically (Part 15): Automation (VII)](https://www.mql5.com/en/articles/11438)", we looked at how to use the indicator as a trigger to buy or sell.

But as I just mentioned, we can use the IndicatorParameters function to make the same EA even more interesting. However, we will not explore the use of the IndicatorParameters function, at least not in this series on replay/simulation. I just wanted to mention how useful this function is.

### Conclusion

In this article, we have seen how to send data to the indicator. Although we did this using an Expert Advisor, you can use other types of processes, such as scripts. Unfortunately, at least at the time of writing this article, it is not possible to use services as a way to do this very job due to the nature of services.

But it's all right. We have to work with what we have. However, this question is not yet completely closed. We still need to understand how to transfer data from the indicator to the Expert Advisor. We need a way to use Chat Trader in the replay/simulation system, but we don't want to use global terminal variables to do this, and we don't want to compile multiple programs and risk forgetting to compile any of them.

To find out how close we are to what we need for Chart Trader, we'll need another article. So, in the next article in the series, we will understand how to approach what we need to create a Chart Trader.

Don't miss the next article: the topic will be very interesting and exciting.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11591](https://www.mql5.com/pt/articles/11591)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11591.zip "Download all attachments in the single ZIP archive")

[EA.mq5](https://www.mql5.com/en/articles/download/11591/ea.mq5 "Download EA.mq5")(1.36 KB)

[swap.mq5](https://www.mql5.com/en/articles/download/11591/swap.mq5 "Download swap.mq5")(1.31 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/466516)**

![Data Science and Machine Learning (Part 22): Leveraging Autoencoders Neural Networks for Smarter Trades by Moving from Noise to Signal](https://c.mql5.com/2/77/Data_Science_and_ML_gPart_22k_____LOGO.png)[Data Science and Machine Learning (Part 22): Leveraging Autoencoders Neural Networks for Smarter Trades by Moving from Noise to Signal](https://www.mql5.com/en/articles/14760)

In the fast-paced world of financial markets, separating meaningful signals from the noise is crucial for successful trading. By employing sophisticated neural network architectures, autoencoders excel at uncovering hidden patterns within market data, transforming noisy input into actionable insights. In this article, we explore how autoencoders are revolutionizing trading practices, offering traders a powerful tool to enhance decision-making and gain a competitive edge in today's dynamic markets.

![Custom Indicators (Part 1): A Step-by-Step Introductory Guide to Developing Simple Custom Indicators in MQL5](https://c.mql5.com/2/76/Indicators_Article_Thumbnail_Artwork.png)[Custom Indicators (Part 1): A Step-by-Step Introductory Guide to Developing Simple Custom Indicators in MQL5](https://www.mql5.com/en/articles/14481)

Learn how to create custom indicators using MQL5. This introductory article will guide you through the fundamentals of building simple custom indicators and demonstrate a hands-on approach to coding different custom indicators for any MQL5 programmer new to this interesting topic.

![MQL5 Wizard Techniques you should know (Part 18): Neural Architecture Search with Eigen Vectors](https://c.mql5.com/2/77/MQL5_Wizard_Techniques_you_should_know_fPart_18j___LOGO.png)[MQL5 Wizard Techniques you should know (Part 18): Neural Architecture Search with Eigen Vectors](https://www.mql5.com/en/articles/14845)

Neural Architecture Search, an automated approach at determining the ideal neural network settings can be a plus when facing many options and large test data sets. We examine how when paired Eigen Vectors this process can be made even more efficient.

![Developing an MQL5 RL agent with RestAPI integration (Part 2): MQL5 functions for HTTP interaction with the tic-tac-toe game REST API](https://c.mql5.com/2/61/DALLvE_2023-11-26_00.52.08_-_A_digital_artwork_illustrating_the_integration_of_MQL55_Pythono_and_Fas.png)[Developing an MQL5 RL agent with RestAPI integration (Part 2): MQL5 functions for HTTP interaction with the tic-tac-toe game REST API](https://www.mql5.com/en/articles/13714)

In this article we will talk about how MQL5 can interact with Python and FastAPI, using HTTP calls in MQL5 to interact with the tic-tac-toe game in Python. The article discusses the creation of an API using FastAPI for this integration and provides a test script in MQL5, highlighting the versatility of MQL5, the simplicity of Python, and the effectiveness of FastAPI in connecting different technologies to create innovative solutions.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/11591&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070145795912569016)

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