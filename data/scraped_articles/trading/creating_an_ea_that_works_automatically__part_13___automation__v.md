---
title: Creating an EA that works automatically (Part 13): Automation (V)
url: https://www.mql5.com/en/articles/11310
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:07:24.379967
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/11310&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069129683959742633)

MetaTrader 5 / Trading


### Introduction

Now that we have finished creating the basic skeleton, we can finally automate the EA to make it operate 100% automatically while following the operational rules that we have defined. The purpose of the article is not to build an operational model, but to show and prepare you to use the system proposed here, by turning a manual Expert Advisor into an automated one.

I think the big problem of the most traders who want to use a 100% automated EA (not necessarily your case, dear reader) is that they know absolutely nothing about programming, or even worse, they know nothing about the financial market and do not want to explore this area.

Many people just want to run a computer program that will magically generate money, without actually understanding what is going on behind the scenes. I see a lot of people talking good or bad about a particular platform, system or EA. But the vast majority do not have the slightest idea of the risk level they are putting themselves in, imagining that a simple computer program written in MQL5 to run on MetaTrader 5 or developed in any other programming language will become an ATM.

Many people have this "dream", or "illusion". They think that some programmer or even themselves can develop a method, program or algorithm that will magically generate money in their brokerage accounts. Many of them even need to learn the basics of the financial market, but they don't like anyone telling them that they are wrong.

Everything, absolutely everything that can be used, created or developed to somehow try to turn a program into a money factory in the financial market, has already been created, tested and developed. Some things have proven to be unsuitable, while others having good risk control, manage to generate some profit over some period.

So, if you are one of those who hope to find a magic system that can eventually make you rich, stop reading this article immediately because this information is not aimed at you. The same applies to anyone who expects to see a magic formula in this article: please, do not continue reading.

Here, I'll show you how to develop a trigger to automate the system that already use.

Let's figure out how to automate an Expert Advisor. Let's start by planning what we actually need to do.

### Planning the automation

First, it doesn't matter what you are going to trade. It doesn't matter which indicators, chart timeframes or signals you are going to use. The automation process will always follow the same type of planning. The only difference is how the trigger will be built, but the planning will always be the same.

To plan how to automate the system, you must first have a trading model that has been in use for some time. This is paramount. Do not try to automate an unknown model. You need to have a habit of using that specific model.

So, this is the first requirement for automation: You must know how to use the model. Now you need to clearly define the rules for performing trades. This should be done in a clear and objective way. There should not be things like: If this is like this, when maybe this here is that other way, I hope this works out. This is wrong. The rule should be something like: If this happens, sell; if that happens - buy. The rule must be something objective. Remember, we are going to teach a robot that has no emotions or feelings. It feels no fear or remorse. So, be objective.

After determining clear rules, it's time to start creating the programming logic. Many get lost at this point , since by the time of programming they try to invent new rules trying to make the system better or more profitable. Again, this is wrong. The EA must do the same what your model does. Exactly the same.

As an example, let us look at how to create rules for the Larry Williams model, known as the 9-period Exponential Moving Average. There are many variants of the model, but this original one is quite simple. Please note that this is only an example, which I do not recommend using if you do not actually understand how it works. So, let's see the planning of the rules for this model.

_**If the bar closes above the 9-period Exponential Moving Average, BUY; if it closes lower, then SELL.**_

This is the key rule for programming the model: everything is clear and objective. If you want to add a trailing stop, the approach must be as simple as this example:

**_The trade stop will always be one tick above the low of the closed bar for BUY trades and one tick below the high of the closed bar for SELL trades._**

Again, the rules are simple. This is how you should create your own rules. To make it very clear, let us look at another example which uses the RSI indicator. The logic is applicable to any other indicators. The purpose here is to demonstrate the correct approach:

_**If the indicator is above the level of 80, then wait for a SELL entry. As soon as it falls below 80, SELL. If the indicator is below 20, get ready to enter into a BUY trade. As soon as it rises above 20, BUY.**_

This rule is simple and effective, but it still needs stop level or trailing stop rules. For this, we can use another indicator or even the same indicator. Again, this doesn't matter as long as the rule is clear. Let's then define a rule based on the 20-period arithmetic average. The rule may look like this:

**_Whether bought or sold, the STOP level is placed at the price of the 20-period arithmetic moving average._**

But you can also change this rule as follows:

**_When in long, if the price closes below the 20-period arithmetic MA, close the trade. For a sell position, if the price closes above the 20-period arithmetic MA, close the position._**

Is it clear now how things should actually be done? There is no middle ground, **everything has to be black and white**, without inventing or imagining possibilities.

Because of this, it is necessary that you have the habit of using the desired mode before automating it. There is no room for uncertainty here. The rules are strict and must be strictly followed. Remember: you are telling the robot what to calculate. There's no way to tell it how to be biased, trying to play by the rules. Even adaptive systems still follow strict and very well defined rules. The fact that they can adapt to the market is only due to the fact that within these strict rules, we place some kind of error that it can accept.

Adding an acceptable level of error into systems can make them more subjective, so you start telling them "maybe buy" "maybe sell". Although this approach seems interesting, it complicates control. Not from the points of programming or mathematics, but the difficulty for the trader who controls the EA: this requires deep knowledge of the market in order to analyze the EA actions and determine if it does something wrong.

To understand how to make the system adaptive, let's use the example of a 9-period Exponential Moving Average. This example can be applied to any model by adding rules to fit it. The buy rule will look like this:

**_When the bar closes above the 9-period Exponential Moving Average, analyze the previous 5 bars: if they form lower highs, do not enter the trade._**

We have added a level of mathematical subjectivity to the system. Previously, we bought if the bar closed above the average. But now buying additionally depends on previous highs. To analyze all this in real time, you must be an experienced trader.

Note that even with some subjectivity, the rule remains strict. If you look at how the EA works, it may seem that it adapts to the market. But this is only an illusion since its operation is still governed by a strict rule.

Some people often use data analysis programming in order to generate these same rules. With this type of programming, some people may say that the system started to use artificial intelligence.

However, based on my programming experience, I can say that this is nonsense. We actually only use algorithms to create mathematical curves and opportunity graphs based on observed patterns.

These opportunity curves lead to strict rules that are based on mathematical analysis rather than market observation and experience. The problem is that such a system cannot quickly adapt to market changes, because we are dealing with machines that do not reason. They are mere calculators and do nothing until they are programmed to do so.

Now you probably understand how far we can go. There is no point in looking for a magical system, and we must clearly and objectively determine what needs to be done. Now, we can move on to the next step.

### Turning the idea into code

Many may be wary, thinking that turning an idea into code is something very complicated, accessible only to experts, masters or PhDs in programming science. But this is not true. Anyone who has common sense, caution, discipline, curiosity, and interest can actually turn their idea into a code, if this idea is clearly and objectively stated. If you could not clearly define the idea that you are going to implement, go back to the previous step and write down your idea on a paper in a clear and simple way.

Now let's see how to turn the idea into code. If the idea is too short, give a detailed description. Describe the step-by-step process and how events will unfold. The more steps we include in the description, the better.

To know if the description is definitely adequate, make sure you are able to complete the following flowchart:

![Figure 01](https://c.mql5.com/2/48/003__4.png)

Figure 01 - Implementation block diagram

I know that many will look and find this image quite strange. Probably, the have never seen anyone using such things, especially programmers.

Figure 01 is a flowchart widely used in programming and other fields. But since we are dealing with programming, let's focus on this point. What we have in Figure 01 is the way the mechanism should work, no matter if it's manual or automatic. Each and every program has its own map shown in figure 01. If you can correctly fill in each points, you will actually be able to create the program, so that it will work perfectly. Such a flowchart works based on fractals.

But how is this possible? How can a flowchart be a fractal and represent how we should program things? 😵😱 Let's figure out how Figure 01 becomes a fractal able to build anything. No matter how complex the thing is, if it can be represented within the framework of this chart, then it can be built.

Each process starts with **P1**. Next, we go through some kind of decision at point **P2**. It doesn't matter what the decision is. We will always take one of the paths: the one that goes to point **P3** or the one that goes to **P4**. The path to be taken is determined at **P2**.

Both **P3** and **P4** can represent an image similar to Figure 01, where the process is repeated until it finally ends. However, to simplify the explanation, let's assume that **P3** and **P4** are some simple calculations or calls that will return shortly. In the end both will converge to **P5**.

Here we make a new decision based on what **P3** or **P4** informed us. Depending on what **P5** decides, we can either return to **P2** or go to **P6**. If we return to **P2**, the process will repeat, creating a feedback system. If we go to **P6**, the process will end.

Thanks to this dynamic, leaving **P1** and arriving at **P6**, we have a message or execution flow. The same flow can also occur in **P3** and **P4**. If this happens, the fractal will appear in **P3** or **P4**, but not elsewhere in the flowchart.

Let's get back to programming. In MQL5 **P2** and **P5** can be replaced by **IF ELSE** or **SWITCH CASE**. Thus, it is possible to create anything without programming knowledge, as long as you can represent your idea in this flowchart, explaining and describing it properly.

**P1** can be a set of arguments that a function or procedure receives, or the point where you start to execute the code when it is placed on the chart. It is important to understand that we place all initial and necessary values in **P1**.

**P6** can be the **RETURN** command or something else that will provide a response based on the values entered in **P1**.

As for **P3** and **P4**, they can be either another flow chart showing how to decide or a simple factorization in which **P1** values are processed. Our purpose is to describe the system in Figure 01 so that it can be automated.

To make it easier to understand, let's view some examples. Let's start with a 9-period Exponential Moving Average/ We will convert the description of the procedure into a programmable form and put it into the flowchart. Consider the following:

**Example 01 - Buying or selling using the 9-period Exponential Moving Average:**

- In P1, we enter the MA value and the bar closing price.
- In P2, we check if the price closed above or below the given average.
- If the price closes higher, go to P4 (buy call).
- if it closes lower, go to P3 (sell call).
- After buying or selling, go to point P5.
- If the EA is automated, we return to P2 to check the procedure on the next bar. If the EA is manual, we go to P6 as there are no other actions to perform.

**Example 02 - Trailing stop and breakeven:**

- In P1, indicate the desired profit and the current bar price.
- In P2, check if the position has reached the desired profit: if yes, then execute P3; if not, P4.
- In P3 we start a new flowchart (P1 - we specify the value of the stop price).
- In this new flowchart in P2, we check if breakeven has already occurred (if not, execute P3; if yes, P4).
- If the breakeven was not executed, then proceed further. Move on to P5 and P6, returning to the previous flowchart and waiting for a new trailing stop and breakeven procedure call.
- Returning to P2 of the nested flowchart, we check if breakeven has occurred. If the answer is yes, execute P4 (trailing stop).
- When P4 is called, we open a new flowchart to check if the price can be modified, which can be done in several ways using other flowcharts. At the end, we return to the main flowchart and exit the procedure, waiting for a new call.

**Example 03 - Working on RSI or IFR:**

The flowchart is similar to Example 01. Here it is:

- In P1 we get the indicator value.
- In P2 check if the value is greater than 80 or less than 20 (Note: arbitrary values are shown for demonstration).
- If it is above 80, execute P3 (sell).
- If less than 20, then P4 (buy).
- Regardless of the result, move on to P5; if the system is automatic, return to P2; otherwise, go to P6 as no further action is required.

Is it clear now how it works 😁? If you can do this, then you will be on your way to creating an automated EA.

I understand that you are probably impatient to see the code. Don't worry, as the code is the easiest part. The main thing is to understand how to achieve this and how to create your own automated EA. I want to demonstrate that anyone, whether a professional programmer or not, can create an EA and use the most appropriate model without relying on confusing and insecure programming.

So be patient, we'll get to the code soon. I will show you how to create an automated EA, but first you need to know how to use the system. Otherwise, you will be limited to the presented code, while the system can offer much more possibilities. In this short series I share some of my knowledge.

### Understanding and implementing automation

If you understood the previous topic, then you can move on to the code. However, this will not be covered in this article: I intend to explain in detail how to build the code later. This part will require a very detailed explanation as I am not suggesting the use of a specific model.

The purpose of this article series is to teach how to convert an already tested and manually operated model into an automated system, rather than creating a specific operating model. In this article, I'll show you how to work with the class system which is used to create an Expert Advisor. However, we will leave the actual implementation for the next article, for the reasons stated above.

The EA presented in this series has a concise list of functions which however is able to cover almost 100% of cases. Well, there can be unpredictable situations which may require additions to the code, but they are quite rare. Since most users will work with indicators, the proposed class system will make them easier to use, preventing the user from writing larger code. You may need to turn on a few alerts to inform you of what the EA is doing, since the indicator does not need to be on the chart to receive data and values. The indicator may be running the chart so that we can observe the trades, but this does not matter to the EA.

Another important case is when we need to add additional code to cover a particular model. This should be done within a class system, preserving the concepts of encapsulation and inheritance so as not to compromise code security or create compilation-related problems.

As I already mentioned in previous articles, pay attention to the warning messages generated by the compiler. They should not be ignored, except during the testing phase when the solution is not yet complete. However, once the code becomes more specific, do not ignore compiler messages, even if the code has already been compiled. The presence of warnings indicates that the code may not be completely secure or may not have sufficient stability and accuracy for 100% automated use. Keep this in mind when editing the system we are creating here.

Let's see the system structure for manual operation shown in Figure 02.

![Figure 02](https://c.mql5.com/2/48/004__2.png)

Figure 02 - Manual operation

In this figure, it may seem that we are dealing directly with the EA or with one of its classes. However, this is not true. The trader is actually dealing with the platform, while the EA responds to the commands generated by the platform but not the user's commands.

It is very important to keep this in mind because many people think that when they work with an EA, they are interacting directly with it, when in fact the interaction is with the platform that sends commands to the EA. This has already been discussed and explained in previous articles in this series. Before automating an EA, you should understand Figure 02. Although the form may differ depending on the programmer, the principle is the same: you interact with the platform, not with the EA.

Based on this, we can think where to place a trigger system. Pay attention that Figure 02 shows the structure of files, not the internal structure of classes. Keep this in mind if you are going to modify the system in any way. I'm just trying to explain that any modifications must be made with care in order not to break the system stability and or, even worse, not to make it outwardly reliable, while in fact it is a time bomb, ready to explode at any moment.

Let's think about how the system works: you, as a trader, inform the MetaTrader 5 platform about your intention to buy or sell at a certain price by interacting with the C\_Mouse class.

This class is connected with the EA that receives mouse movement events and informs the C\_Mouse class where to place the price line. Upon the mouse click, the point is passed from C\_Mouse to the EA that analyzes whether the trader is buying or selling. The EA then sends a request to the C\_Manager class which checks conditions for the C\_ControlOfTime time.

Depending on the creation parameters in the C\_Manager class, the request can be accepted or rejected. If it is accepted, the corresponding order is sent to the C\_Orders class, which, in turn, sends the request to the trade server. When the server responds to the request, a new event is fired in the MetaTrader 5 platform, informing the EA about what happened on the server via the OnTradeTransaction event.

The EA analyzes this information and, depending on the context, informs the C\_Manager and C\_Orders classes about the result of the trader's request. Thus, a position is opened, modified or closed by interacting with the code present in the EA. However, a trader can use the toolbar to close or modify an order or position opened by the EA on the server. If this happens, the MetaTrader 5 platform will generate an event using OnTradeTransaction, notifying the trader of the change. The EA should then pass the information to the C\_Manager class, if applicable.

Understanding how this works, you can see that the automation system should replace the C\_Mouse class, which initializes the entire trigger system. Although the C\_Mouse class will be removed when automating the system, automation will not happen instead, because C\_Mouse will continue to interact with the EA, which will generate calls to the C\_Manager class.

The C\_Mouse class can even establish some connection between the EA and the C\_Manager class. But since we knew in advance that the C\_Mouse class would be removed from the system, the entire system is designed to easily remove it. Automation occurs in this step.

Still, we have to deal with the issues related to triggering the events caused by a trader's interaction with the MetaTrader 5 platform. In this case, the EA will simply ignore any requests, it will only inform the C\_Manager class about events related to the order system so that it is aware of the trader's actions. This process will become clearer in the next article, where we will look at the automation in more detail.

### Conclusion

In this article, we looked at how to use a flowchart to set the operation rules. Such questions often arise when discussing artificial intelligence. But the idea itself is not new. However, I often see people speaking of it as something recent. In any case, the material presented here is valuable, although it is only a small part of the existing potential.

I invite the reader to take the time to study this article and related articles in order to understand each step mentioned. In the next article, we will see how the system works automatically, as if by magic, based on what we have explained here, so a good understanding of this material will help you understand the result of the entire article series.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11310](https://www.mql5.com/pt/articles/11310)

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

**[Go to discussion](https://www.mql5.com/en/forum/449127)**

![Automated exchange grid trading using stop pending orders on Moscow Exchange (MOEX)](https://c.mql5.com/2/54/moex-mesh-trading-avatar.png)[Automated exchange grid trading using stop pending orders on Moscow Exchange (MOEX)](https://www.mql5.com/en/articles/10671)

The article considers the grid trading approach based on stop pending orders and implemented in an MQL5 Expert Advisor on the Moscow Exchange (MOEX). When trading in the market, one of the simplest strategies is a grid of orders designed to "catch" the market price.

![Rebuy algorithm: Math model for increasing efficiency](https://c.mql5.com/2/54/mathematical_model_to_increase_efficiency_Avatar.png)[Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)

In this article, we will use the rebuy algorithm for a deeper understanding of the efficiency of trading systems and start working on the general principles of improving trading efficiency using mathematics and logic, as well as apply the most non-standard methods of increasing efficiency in terms of using absolutely any trading system.

![Improve Your Trading Charts With Interactive GUI's in MQL5 (Part I): Movable GUI (I)](https://c.mql5.com/2/55/Revolutionize_Your_Trading_Charts_Part_I_avatar.png)[Improve Your Trading Charts With Interactive GUI's in MQL5 (Part I): Movable GUI (I)](https://www.mql5.com/en/articles/12751)

Unleash the power of dynamic data representation in your trading strategies or utilities with our comprehensive guide on creating movable GUI in MQL5. Dive into the core concept of chart events and learn how to design and implement simple and multiple movable GUI on the same chart. This article also explores the process of adding elements to your GUI, enhancing their functionality and aesthetic appeal.

![Category Theory (Part 9): Monoid-Actions](https://c.mql5.com/2/55/category_theory_p9_avatar.png)[Category Theory (Part 9): Monoid-Actions](https://www.mql5.com/en/articles/12739)

This article continues the series on category theory implementation in MQL5. Here we continue monoid-actions as a means of transforming monoids, covered in the previous article, leading to increased applications.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/11310&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069129683959742633)

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