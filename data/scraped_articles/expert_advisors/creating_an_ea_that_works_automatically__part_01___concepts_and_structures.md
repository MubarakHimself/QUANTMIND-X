---
title: Creating an EA that works automatically (Part 01): Concepts and structures
url: https://www.mql5.com/en/articles/11216
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:27:15.337532
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/11216&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071843398916190082)

MetaTrader 5 / Expert Advisors


### Introduction

During this time, several people, who have read my articles about how to create an EA from scratch, contacted me requesting to create an EA for them or give them some guidelines regarding how to do it. To all such requests I replied that programming an EA that operates automatically is not by far the most complicated programming task. I always motivate and give proper guidelines to those who actually want to learn and deepen their knowledge, whether enthusiasts or people who want to create their own EA that would follows a certain operation process in a fully automated mode.

I have also noticed that many of these people have basically the same doubts about how to get started with programming. But what amazes me the most is that many of them, for one reason or another, have a completely wrong idea about how to implement this type of EA. Some users even do not have a proper understanding of the risks associated with the process of creating and using an automated system. In this case, by saying an automated EA system I mean a system that works without any control and that can buy and sell assets without direct or indirect intervention of the trader.

Therefore, I decided to explain these points so that everyone can understand them or argue on this topic. Step by step, I will show how you, an amateur or even a beginner in this topic, can step by step program an _automatically trading_ EA from scratch in a simple but safe way, basically using only and exclusively native MQL5. There will be no miracle solutions or anything like that.

At this early stage, I won't go into complicated details and will not use beautiful and flashy charts. I will not try to deceive you, who may not have sufficient knowledge of statistical mathematics, and I will not tell that the EAs presented here will serve as slave trader and earn you for living. It is quite the opposite: creating an EA is a very long task that requires a lot of study and dedication, if you want to create something really valuable. Not because of programming but because of operational and other questions that we will consider here.

The big problem is not the programming part itself as many may mistakenly think. The problem is that in many cases we will have more loss than the real profit. I repeat again: the problem is not in the programming but in the lack of experience of the person who designs what type of operating system will be executed by the EA. The fault is never in the programmer, although if he thinks that he is a great programmer and people trust him blindly, they will sure have problems. However if the person knows what he is doing, he will be completely calm about his work. Given this fact, the problem will not be in the programming but it can be in the trading system designed by the person who requested the development of the automated Expert Advisor.

### How do you split an automated EA? How does it actually work?

Before we start anything directly related to writing the code, we need to make things clear for everyone who will be reading this short series of articles. If you already know how an automated EA actually works, most likely the series will not add anything to your knowledge. But if you have no idea how it works or what it takes to make the EA work, follow these articles for at least the basic knowledge. Because without this knowledge you will be completely lost. Come with me through this series, if you want to take the first steps or to know what to study and analyze.

To begin the explanation, let's take a look at figure 01 below:

![Figure 1](https://c.mql5.com/2/48/001.png)

Figure 01. Schematic diagram of the automated EA operation

This image shows in a very simple and generalized form how the automated EA actually works. Please note that the EA has two internal structures: the order system, which will always be the same for all EA, and the second structure, the trigger system, which is responsible for making the EA work automatically.

As mentioned above, the order system will always be the same. Therefore, once the system creation is finished and it operates stably, and you completely trust it — since it is the part of the EA that you created — you can move on to the trigger system where the most problems lie.

But before we start to considering the systems that we want to build, you need to understand how to separate them. Please pay attention to this, because many people confuse things and get completely lost when trying to create an automated EA. Especially when it comes to beginners, as they don't understand the most basic concepts.

### Understanding how to create the order system

Here I would like you to do the following: read the series of articles [Developing a trading EA from scratch](https://www.mql5.com/en/articles/10678), because in this series I showed every step required in developing this order system. In this series, I didn't explain some of the things that can (but not necessarily will) exist in the order system.

Two of the details are _breakeven_ which is makes the EA automatically move the _Stop Loss_ value to a position where you no longer risk to receive profit, and _trailing stop_ which will first create the _breakeven_ and then will move it to secure profits...

While many don't know it, the _Trailing Stop_ mechanism exists by default in the MetaTrader 5 platform. But to use it, you have to enable it for each position one by one, adjusting the value accordingly... I know that it doesn't seem very promising and not at all rewarding to do this, but in the absence of an order system with the _trailing stop_ not really developed and tested, the best choice is to use the mechanism present in the MetaTrader 5 platform, as it is stable and reliable.

So, absolutely everything that is part of the order system, starting from the lines that appear on the charts and allow us to analyze what is happening, to the _trailing stop_ mechanism, all this is part of the order system, which you should create completely separately and test manually using it for a period of time that you think appropriate.

During his period, you will have to adjust, correct and test all possibilities of failures and errors that can happen. Believe me, you will not want the order system to crash or generate any information on the chart which does not really represent what the EA is doing. In this case, you will want everything that the EA does to be undone and modified by you.

These tests should be performed over a long period on a demo account where you do not have the risk of losing or earning real money. The tests must be very intensive.

Test everything and absolutely everything, until you believe that the system is reliable, comfortable and easy to use, i.e. that you will be able to easily utilize it. Do not move on to the next step, but stay here adjusting and testing the order system. And again, if you don't have the slightest idea how to create a minimally decent order system, read the article series I mentioned above. Study the code, understand how all the things work and how you can modify and adjust them to make the system comfortable to you so that you will be able to automate it.

Take your time. Believe me, you don't want the order system to break in the middle of real trading. This wouldn't be funny at all. You won't blame the platform or the market, because the only person actually responsible for the problem will be solely you.

Once you have a finished and working order system which operates the way you want and expect it to work, you can move on to the next step. It is considerably simpler, but that doesn't mean that you shouldn't neglect the necessary precautions when working on it. I'm talking about the trigger system.

### Understanding the trigger system

The trigger system is perhaps the part that many, who have no idea of ​​programming, or mathematics, are most excited about when starting to study and follow the financial market. The trigger system is the part that, once connected to an order system, makes the "magic" happen, giving the impression that the EA is a true trader and can stay there operating 24 hours a day, 7 days a week, 52 weeks a year, without getting tired or suffering any stress, being afraid to enter or exit a trade.

In fact, the trigger system is something very interesting, not because of the programming itself, but because of what it can represent mathematically. Everyone who follows the market must have heard of traders who have super cool trading systems, and that are extremely profitable. For example, who doesn't know the story of [Larry Richard Williams](https://www.mql5.com/go?link=https://www.senhormercado.com.br/a-louca-historia-do-alavancado-trader-larry-williams/ "https://www.senhormercado.com.br/a-louca-historia-do-alavancado-trader-larry-williams/") and wouldn't like to become trader too?

I'm sorry to inform you that today the market has many varieties of algorithms that are more efficient than the one that Larry Williams used to make money and get recognized. This doesn't mean that the trigger system he used is no longer viable. On the contrary, it is still profitable and if you have adequate risk control, you will be able to earn a lot of money with the same algorithm. The trigger system should be simple so that the EA can take the advantage of it. Furthermore, it is easier to test a simpler system.

There is no point in creating a trigger system full of complicated and bizarre rules and extravagant analysis, if you can't put it in a simple mathematical formula that you can program. So forget those bizarre things like looking at the indicator Z linked to the asset X, in order to trade Y at the moment K, while monitoring the level of aggression P, analyzing Depth of Market, looking for the incidence of intention W, in order to be able to enter a sell or buy position. Forget this. This sort of thing won't work, or at the very least it will make the thing so bizarrely complicated that you won't be able to test the algorithm.

Remember: always choose simplicity. Always try to keep things as simple as possible. Make it so that the number of possible failures drops to a level, make sure that the trigger system will not generate a trigger effect, where it starts sending a huge series of orders to the system of buy or sell calls, which ends up breaking not only the account, but destroying all your equity.

I would like to mention here one warning. Avoid, as far as possible, creating a trigger system in which the EA starts buying or selling with an increase in position, i.e. averaging the price, up or down, because this kind of trigger often causes a lot of problems. Ideally an EA should enter and exit a position before trying to open a new one, whether it is buying or selling.

Furthermore, there are several ways to block the sending of orders for a period of time, or to make it so that the EA will not send an order if there is already an open position. Think about this when you are creating and designing the trigger system, as many EAs end up entering an endless loop — and when this happens, all your money is lost in a matter of minutes or even seconds.

Another important thing to mention about the trigger system is the importance of background. If you haven't studied, I advise you to so. **You have probably heard about the 1987 crash, also known as "Black Monday".**

If you haven't heard, then before ever proceeding to creating the EA, find out about this crash. because it was caused exactly by the trigger system. So study first, because you are no better than anyone who came before you. At least you might not have all the required knowledge. Don't be deceived by people who talk about the wonders of quantitative trading and how EAs will dominate the market in the future. Be careful before relying on any information or opinion.

Stop and think for a second: if quantitative trading really were as simple and effective as many claim, wouldn't large companies, with all their economic power and interest in even greater profitability, use such mechanisms? Why would they hire the best programmers, mathematicians, and statisticians when they could pay for a trigger system that would never lose and add a contractual clause to prevent anyone from replicating it? So study well, before you decide you've discovered the holy grail, because everyone who does serious studies knows that it simply doesn't exist.

Now that I have given these warnings and shown you how and what it takes to generate an automated EA, let's move on to the most interesting part: **THE CODE**.

### Initial planning

Now that you are familiar with the ideas and concepts that you should remember throughout your life, whether you are a programmer, a beginner, or even a professional trader, let's move on to the next step: this is where programming skills really become important. This is where many people make mistakes because they think that programming is writing a series of commands apparently without any logical sense. They think everything will work just because they want things to work.

Programming is not something complicated or mystical, where only grand masters, or a PHD in computer science, with a broad and immense academic background will actually manage to generate something. This is all nonsense from people with a low level of knowledge, who want to appear smarter than others.

To program, and especially in a language like MQL5, which is very similar to C and C++, you don't need a lot of knowledge. In fact, you only need to know 4 basic things and have 2 essential qualities. The 4 things you need to know are:

- There is no program that cannot be executed with simple commands, so learn and master simple commands before using more complex ones.
- Learn how to use the following commands: IF, FOR and the basic operators: Equality (=), Addition (+), Subtraction (-), Division (/), Multiplication (\*) and Modulo (%).
- Understand the difference between working with Bits and Bytes and what Boolean operators do: Not (!), Or (\|), And (&) and Xor (^).
- Understand the difference between function and procedure, global variable and local variable, event handling and procedural call.

Everything in programming boils down to these 4 things. No matter how complex a program is or could become, if you can't boil your code down to these 4 things you are trying to build something the wrong way. Now the 2 essential qualities are:

- Do read the language documentation (in this case the MQL5 documentation) — it is there to help you.
- Learn to divide tasks, be organized, comment the code parts that you still don't understand and always test, no matter the result. Test anyway and write down the test results.

Based on these premises, you will certainly become an excellent programmer over time, but you need to dedicate yourself and study. Don't wait for someone to simply tell you how to do something, always seek knowledge and learn — things will become simpler and clearer for you over time. But the main thing is:

Be humble, you don't know everything and you never will. Accept that there will always be someone who knows more than you.

Now that you know what you need to do to create your first automated EA, let's start writing its code. Remember what I said above: first we need to create an order system that best adapts to our needs and is safe, stable and robust. Create something that works and not something beautiful but meaningless.

### True planning

Let's finally proceed to the code. Right? No. Not yet. Don't rush. Now you can open the code editor MetaEditor. The first thing you need to do is create an EA from scratch. This should be clean code, not the one you borrowed from a friend or downloaded from the web. Let's really start from scratch.

For those who don't know how to do it: after opening the MetaEditor, in the left corner you will see a window called browser. There you will have a structure of directories each starting with MQL5. Treat this window as your file explorer — this way it will be easier for you to follow the explanation.

In the same window there is a folder called Experts. This is where we will place codes for any EA we are creating. There are also other folders for other purposes, but now we will focus only on Experts. By clicking on this folder, you can open it and see codes contained in it. Most likely existing codes are organized in relevant orders, each one representing a different EA. Get used to this structure - it's important to maintain a certain organization, otherwise you will get totally lost among all the files present there, and won't be able to fine the required code when you need it.

So, let's make the first steps. They are very simple, so just follow the images shown below:

![Figure 1](https://c.mql5.com/2/49/002_1.png)

Figure 02. Choose what to create

![Figure 03](https://c.mql5.com/2/49/003_1.png)

Figure 03. Specify the directory where the code will be located

![Figure 04](https://c.mql5.com/2/49/004_1.png)

Figure 4. Give a name to the EA (the extension is optional).

![Figure 05](https://c.mql5.com/2/49/005_1.png)

Figure 05. Click Next

![Figure 06](https://c.mql5.com/2/49/006_1.png)

Figure 06. Click Finish

When you complete steps shown in the figures above, Meta Editor will create an absolutely clear and safe code. You will see something similar to the below code:

```
#property copyright "Daniel Jose"
#property link      "https://www.mql5.com/pt/articles/11216"
#property version   "1.00"
//+------------------------------------------------------------------+
int OnInit()
{
        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
```

Don't worry about the **#property** compilation directives. They will not affect what we are going to do here. They are here to configure the window which will be opened when you launch the EA on the chart. So don't worry about them. But you should worry about other things.

As you can see, MetaEditor has created three procedures: **OnInit**, **OnDeInit** and **OnTick**. These procedures are actually event management functions. So, MQL5 programming differs from other languages. MQL5 is an event handling language, unlike other languages where we work procedurally. Event-based languages can be confusing at first, but over time you come to understand how to do things in them.

Here, without going into too much detail, you will work as follows: The MetaTrader 5 platform generates an event, and you react to this event, i.e. when you put the EA on the chart, MetaTrader 5 fires the Init event, which will cause the platform to look for the OnInit function in its code so that the EA (in this case) is loaded and launched. Once this is done, this function will not be called again. This is a general description, since the function will be called each tome MetaTrader 5 will place the EA on the chart - remember that.

Here we also have the OnDeInit function. The MetaTrader 5 platform will call it when the DeInit is formed, which tells all elements (EAs and indicators), that something happened and they will be removed from the chart. To find out the reason for the program removal, check the reason value returned by the MetaTrader 5 platform.

And finally, we have the OnTick function, which the MetaTrader 5 platform will be called every time the Tick event happens. This event happens every time a new tick arrives. In other words, when a new operation, no matter what, is performed on the trade server, the Tick event is generated, so the OnTick function is triggered, and the code goes into it.

As you can imagine, the OnTick function is very important for our system, but it is often the Achilles heel of many Expert Advisors. This is due to the fact that many programmers, due to lack of experience or negligence, put all the code inside this function, which can overload it and make the EA less stable and more prone to crashes and critical errors. One of the most difficult errors to fix is the RUN TIME error, which can be caused by complex and hard to reproduce interactions. But we will see this in more detail in one of the further articles.

So because of this, the platform and its MQL5 language, promotes other ways to work with functions, in order to keep things within a minimum level of security and reliability We have other types of events as well, which we can use to improve things in general. But we'll see this later, since I don't want to create any big confusion in the head, especially if you're seeing this for the first time and had no idea that MQL5 works by handling events, which are triggered by the MetaTrader 5 platform, without creating a procedural code, where we have an entry point and an exit point, and all things between these two points must be managed by the program and handled by the programmer.

Here, in MQL5, we will process only those events that we need. To be considered an Expert Advisor, the program must contain two events: OnInit, which handles the initialization of the Expert Advisor, and OnTick, which handles the tick event. The OnDeInit event may or may not be present, but it can be useful in some special cases.

To learn how to read the documentation and become interested in studying what the documentation explains, let's see what it says about the events that can be triggered, especially about the [three events](https://www.mql5.com/en/docs/runtime/event_fire), which are automatically created by the MetaEditor when we start building an EA: [OnInit](https://www.mql5.com/en/docs/basis/function/events#oninit), [OnDeInit](https://www.mql5.com/en/docs/basis/function/events#ondeinit) and [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick). Don't forget to also take a look at the compilation directive [property](https://www.mql5.com/en/docs/basis/preprosessor/compilation).

It is very important that you understand the first step described in this article, otherwise you will be completely lost in the explanations of other articles in this series.

In the next article, I will show you how to create a simple and clear order system that will allow you to submit buy and sell orders directly from the chart. Therefore, stay tuned for the next article, because the topic will be very interesting for those who want to create an EA that works in automatic mode.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11216](https://www.mql5.com/pt/articles/11216)

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
**[Go to discussion](https://www.mql5.com/en/forum/441790)**
(7)


![91976shungu](https://c.mql5.com/avatar/avatar_na2.png)

**[91976shungu](https://www.mql5.com/en/users/91976shungu)**
\|
17 Feb 2023 at 02:08

Let me [check](https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordercalcmargin_py "MQL5 Documentation: order_calc_margin function") on it


![Roman Andrzej Kaminski](https://c.mql5.com/avatar/avatar_na2.png)

**[Roman Andrzej Kaminski](https://www.mql5.com/en/users/romekk)**
\|
23 Feb 2023 at 21:08

What is  "the difference between working with Bits and Bytes".

Could you explain it ?

![Shahzad Iqbal](https://c.mql5.com/avatar/avatar_na2.png)

**[Shahzad Iqbal](https://www.mql5.com/en/users/shahzadiqbal)**
\|
16 Mar 2023 at 18:08

Hi

Can we Apply these EA Steps in MT4 as well

Not only EA Steps in this article but other parts too.

Thanks

![Khanh Nguyen](https://c.mql5.com/avatar/2023/5/64537576-48a1.jpg)

**[Khanh Nguyen](https://www.mql5.com/en/users/danielkhanhnguyen)**
\|
4 May 2023 at 09:02

Thanks for your guide!


![Peter Huang](https://c.mql5.com/avatar/2022/9/6312ab42-c29f.png)

**[Peter Huang](https://www.mql5.com/en/users/fx53168)**
\|
20 Jul 2023 at 19:14

Thanks for sharing, great and heartfelt post.


![Creating an EA that works automatically (Part 02): Getting started with the code](https://c.mql5.com/2/50/Aprendendo-a-construindo_part_II_avatar.png)[Creating an EA that works automatically (Part 02): Getting started with the code](https://www.mql5.com/en/articles/11223)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. In the previous article, we discussed the first steps that anyone needs to understand before proceeding to creating an Expert Advisor that trades automatically. We considered the concepts and the structure.

![How to choose an Expert Advisor: Twenty strong criteria to reject a trading bot](https://c.mql5.com/2/0/Avatar_Twenty_strong_criteria_to_reject_a_trading_bot.png)[How to choose an Expert Advisor: Twenty strong criteria to reject a trading bot](https://www.mql5.com/en/articles/11933)

This article tries to answer the question: how can we choose the right expert advisors? Which are the best for our portfolio, and how can we filter the large trading bots list available on the market? This article will present twenty clear and strong criteria to reject an expert advisor. Each criterion will be presented and well explained to help you make a more sustained decision and build a more profitable expert advisor collection for your profits.

![Population optimization algorithms: Bat algorithm (BA)](https://c.mql5.com/2/51/Bat-algorithm-avatar.png)[Population optimization algorithms: Bat algorithm (BA)](https://www.mql5.com/en/articles/11915)

In this article, I will consider the Bat Algorithm (BA), which shows good convergence on smooth functions.

![DoEasy. Controls (Part 31): Scrolling the contents of the ScrollBar control](https://c.mql5.com/2/51/MQL5-avatar-doeasy-library-2.png)[DoEasy. Controls (Part 31): Scrolling the contents of the ScrollBar control](https://www.mql5.com/en/articles/11926)

In this article, I will implement the functionality of scrolling the contents of the container using the buttons of the horizontal scrollbar.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=jpptjishuadynsatjlyxjlsvtanayfgm&ssn=1769192834404479655&ssn_dr=0&ssn_sr=0&fv_date=1769192834&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11216&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20an%20EA%20that%20works%20automatically%20(Part%2001)%3A%20Concepts%20and%20structures%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919283426893580&fz_uniq=5071843398916190082&sv=2552)

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