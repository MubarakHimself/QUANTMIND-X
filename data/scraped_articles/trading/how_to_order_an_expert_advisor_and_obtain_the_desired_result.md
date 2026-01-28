---
title: How to Order an Expert Advisor and Obtain the Desired Result
url: https://www.mql5.com/en/articles/235
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:22:47.409326
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/235&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069389675510039620)

MetaTrader 5 / Trading


### Introduction

Automated trading is gaining a new momentum - the release of [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/"), with the new [MQL5](https://www.mql5.com/en/docs/basis), is complete. It has successfully passed the [Automated Trading Championship 2010](https://www.mql5.com/ "https://www.mql5.com/"), and the new version of the trading platform is being actively promoted by brokers. The predecessor of the MetaTrader 5, - MetaTrader 4 - is still actively used by hundreds of brokers and by millions of traders around the world.

Despite such popularity (or rather, because of it), the professional level of the average trader is becoming lower - as in any other area, the quantity rarely turns into quality. If we consider automated trading (the connection of trading and programming), the situation is even worse - very few traders have a computer programming degree, and for the majority of people, even for those with a technical mind, the mastering of programming can be very challenging. Also, we must not forget about those who are simply not interested in programming. "We must do what we do best" - they say, and I can not disagree with them.

All it brings us gradually to the topic of our article. There is a demand for MetaTrader programming services, and this demand continues to grow. Where there is demand, there is a supply - this is the law of the market. Indeed, there are enough traders who want to automate their strategy and programmers who want to earn the money. But, sadly, their communication do not always result in a mutual benefit - there is a lot of unsatisfied customers, as well as of programmers who are tired of explaining the concepts.

This article deals with the problems that may arise during the communication of the "Customer - Programmer". First and foremost, it is intended for traders - they often lack the experience in dealing with people of a different mental structure. But there is no doubt that for programmers, this article will be very useful - a relationship always has two sides, and the success of the joint venture equally depends on both of them.

### Contents

1. [Verify the Idea](https://www.mql5.com/en/articles/235#1)
2. [Getting Rid of Illusions](https://www.mql5.com/en/articles/235#2)
3. [Determine the Goals](https://www.mql5.com/en/articles/235#3)
4. [Synchronize the Vocabularies](https://www.mql5.com/en/articles/235#4)
5. [Preparing the Requirement Specifications](https://www.mql5.com/en/articles/235#5)
   - [Where shall we start?](https://www.mql5.com/en/articles/235#5.1)
   - [How do we formulate the task?](https://www.mql5.com/en/articles/235#5.2)
   - [What must not be forgotten?](https://www.mql5.com/en/articles/235#5.3)
   - [How can we simplify the understanding?](https://www.mql5.com/en/articles/235#5.4)
   - [Text, voice or video?](https://www.mql5.com/en/articles/235#5.5)
6. [Selecting the Appliciant/Developer](https://www.mql5.com/en/articles/235#6)
7. [Protect Yourself from Getting Cheated](https://www.mql5.com/en/articles/235#7)
8. [Check the Results](https://www.mql5.com/en/articles/235#8)
9. [Provide Feedback](https://www.mql5.com/en/articles/235#9)

### 1\. Verify the Idea

![Verify the idea](https://c.mql5.com/2/2/idea5.jpg)The most common cause of frustration of the customer is **the loss of his strategy**. When it comes to a full trading system, rather than a semiautomated Expert Advisor or indicator, the trader expects the only one thing from the Expert Advisor - a profitable trade.

And so he gets the long-awaited letter from the programmer, launched the client terminal, starts testing... and sees how his brilliant idea leads to the loss of the deposit. He again verifies the parameters, updates the historical data... and once again sees the loss of deposit. Some traders at once begins to write an angry letter to the programmer (of course he is to blame!), and the more patient ones test the results and try to figure out what the problem is. But it doesn't change the fact - the idea that they believed is turned out to be unprofitable.

Next there are a number of different possible options. Some blame the programmer of being clumsy, and without paying for the work,  began to find another programmer. Others start feverishly trying to figure out how to fix the situation, and ask to make "some small changes" (of course, free of charge because they are so small!). Fairly, I should note that there are also customers who accept their mistake as granted and do not shift the responsibility to the programmer. But this chapter was not written for them, they have a good sense of self-criticism.

This is just the consequences, there is no point of demounting them, it makes much more sense to find and eliminate the causes of the problem. The problem is that the trader is too lazy to check his idea. To do it, one needs to choose an arbitrary interval of historical data, and carefully, day after day, observe how the strategy would work and what will the result be. It is done easier and faster in the visual mode of testing (I'm sure this will soon be available in MetaTrader 5), but the test can also be done in real time - a week or two of work on a demo account.

Really, not all strategies can be tested on the historical data. I know from personal experience, that the checking of some ideas can be really time consuming. I understand that sometimes it is easier to pay for Expert Advisor than to sit for hours with a pencil over the charts. But be aware that the result can be a very unpleasant surprise you when sending an untested idea for implementation!

**Remember!** The programmer is not responsible for the profitability of your strategy, his task is to write a program that will work with the algorithm that you approved.

Sometimes the programmer can tell you the weaknesses of your system (you are not the first one to have done this), but it is only based on his good will. He is not obliged to protect you from mistakes or consult you, and certainly, is not responsible for checking your idea free of charge. When ordering an EA, you should either be fully confident in your strategy, or understand that this is just an attempt, and the result is unlikely to stun you with its exclusiveness.

So our intermediate conclusion is that it is best to check the strategy for a number of times, before ordering its implementation.

### 2\. Getting Rid of Illusions

![Escher's impossible cube](https://c.mql5.com/2/2/illusion2.jpg)We see only what we want to see. And we want a yacht, an island in the ocean, and a suitcase full of money. We look at the charts, and see only the successful indicator signals. We close our eyes and mentally count the number of zeros in the balance of our account ... And the program just works based on the algorithm, and exposes our illusions.

The causes of the second largest and most frequent disappointment - is **confusion and self-deception**. If you do decide to take the first step and check your idea, take the process critically. Many disappointments can be avoided at this stage - just try to remove the rose-colored glasses and look at the strategy sensibly.

A very common situation: you have read in the Internet (on a forum, blog, an online book) about a strategy and decided to make an Expert Advisor for it. Looked at the charts, estimated the correctness of calculations, and thought - "Well, finally, here it is - the Grail!". The situation is worsened by the downloading (or, God forbid, the purchase) of this super-indicator by which the strategy is works - now you have the illusion that 90% of the task has already been completed, and there is just a bit left to do.

If half of you at this point asked yourselves one simple question - "Why this strategy is located on a public domain?" (or, in the case of a purchase, "why is the author selling instead of using it?") - I would have exactly half the number of customers that I have now ... But we believe in miracles, and do not persuade us otherwise! Alright, I will not. But what prevents you from checking of that, on which you're going to spend your personal money?

Here are a few rules that will help you avoid falling into the trap of your own illusions, or someone's bad intentions. They may not solve all of the possible problems, but will relieve you of the most common mistakes.

1. Be very skeptical of indicators without a source code!

   - You will only be able to slightly understand how they work, and nobody will be able to guarantee the immutability of their behavior in the future.
   - They can contain anything - from the coding errors that lead to re-drawing of the old signals, or the inappropriate behavior when working on certain instruments or account types, to a deliberate deception (drawing of known in advance successful signals on the historical data), or a simple restriction on the work time or an account number.


Even if you absolutely do not understand programming, you can always ask a programmer you know to "examine" the program - just to be sure, or directly before ordering the Expert Advisor.

2. Observe the work of the indicators in real time!


   - Without knowing how the indicator works online, it makes no sense to consider its signals in the history - it may simply be a pretty (but, unfortunately, useless) picture.

   - Virtually all of the indicators may change the value on the last (uncompleted) bar - this is normal. It should be understood that on the history (on the formed bars), these changes are not visible, the indicator data plotted at completed bars. It means that the signals (arrows, crossover of indicator/price lines) may be used in the system (to open a position) only at the next bar after its appearance. If we try to use the signals (without waiting for the closure of the bar), during the testing of the Expert Advisor, it will turn out that there were a lot of arrows and crossovers (and therefore open positions), but at the time of bar closing, most of them have disappeared. You'll see the position on the chart, but not the corresponding signal - it is visible only when tracking the bar formation.

   - Some indicators may change the value, not only for the last, but also for several previous bars. Sometimes it caused the essence of the indicator (eg, for the formation of a Fractal, there must be 2 bars on its right, and the last section of the ZigZag can be re-drawn for a long time, until the conditions of a new section are met), but often, such behavior is evidence of a deliberate deception - many indicators are created only for the beautiful drawing on the historical data (in order to sell them), and it is practically impossible to use them in trade.


     If the indicator, on the history, shows the signals to buy at the lowest bottoms, and to sell- at the very tops, it is not an indication that this is a very good indicator, but on the contrary - that it is "predicting the past".

   - It is not necessary to sit down and watch for hours the online chart - many indicators can be tested in a visual testing mode. But anyway, the most reliable way of testing - is running it on a demo account and monitoring it. You can attach the indicator to the chart, wait a bit, and attach it again (with the same parameters) - if it is "real" (does not use the re-drawing of the previous values), then the picture on both copies of the indicator should be identical. If the new copy of the indicator shows different signals, it is useless to analyze its signals on the history.
3. Check the strategy on different intervals of the history!
   - It often happens that you unconsciously choose a very convenient (for your system) interval of history for testing. Whether it's the entrance by the signal of the indicator, a grid of orders, or a pattern recognizer - there will always be an interval, on which the strategy will make money. But when you will test the Expert Advisor on all of the available historical data, you will right away see the most unsuccessful intervals. Try to find them yourself.

   - Select a few random intervals of the history. For example, take January 2008, October 2009, and August 2010 - scan them with your eyes and calculate the ratio of profitable and losing signals. Try to look at the quality of the signals on other instruments and time-frames. If the idea has a core, it is most likely will work in other conditions (possibly with slightly modified indicator parameters or stop levels).
   - Do not change your system parameters for a single check. If while looking at August, you will use certain indicator parameters, and, having moved to September, take other parameters (which fit better) - you are simply fooling yourself. Checking should be done under the same conditions, otherwise you will only see what you want to see. I am not saying that the system parameters must always remain the same, but if you want to change them "on the fly", you must think of the criteria for their change. It is always simple to draw conclusions on the history, but what will happen in the future?

   - Find the most complicated interval of the history for the strategy and test the system on it. Very common tactics - the use of the channels and waiting for the backward movement, often backed-up by the doubling of the lot at the opening of next positions - works great at some intervals of history. But the market has a long-term movements in the channel and strong trends, and both systems periodically lead to the lose of the deposit. Find the interval of history that will be most dangerous for your system (a long flat - for a breakdown tactic, and a strong one-way movement - for tactics, based on backward movement), and watch the problems when trading on this interval.




      And do not forget - in the future you may see an even longer flat, and even more extreme trend movement, the market is constantly changing. Always reserve a "margin of safety".
4. Consider the overheadcost (spreads, swaps, commissions, margin requirements, minimum clearances for installation of orders)!
   - For most long-term strategies, all of these expenses are not important, since an error by a few points is irrelevant for them. But the development trend of automated trading has shown that more and more strategies are being developed with small profits and a high frequency trading, and the increase of the spread by 2 points or the Stops Level by 10 points, can become quite noticeable.
   - Remember that all of the MetaTrader charts (the 4th and 5th versions) are created using the Bid price, while the opening of long position and the closing of the short positions are executed by the Ask price. When you trade in your mind, taking the spread into account is inconvenient, but we can always take the total result of the trade and subtract from it the number of trades, multiplied by the spread - it will bring the results closer to the reality.
   - When analyzing the history, do not forget that a few years ago trading conditions were very different. The "fluffy" chart of 1999 is ideally suited for scalping strategies, if we trade using the current spread. But in 1999 the spread was 3-4 times larger! From the profits that you have virtually gained, there will be nothing left, if we subtract the real spread of that time period from each transaction. The situation is analogous even with other conditions - a few years ago, the maximum available leverage was 1:100, and the distance for placing a pending order was measured in tens of points.
   - Remember that trading conditions may change (of course, for the worse one) at news releases, and the execution of orders can be delayed and have slippage. Do not invent "a brilliant news system" on the history, try to test them at least on the micro-real account. Your view of the system greatly changes if you are estimating a 2-point spread and instant execution, but the position is opened after 5 seconds, with a slippage of 10 points, and the spread widens so much that it is difficult to close the position at least without losing.
   - Always have a "margin of safety" in relation to trading conditions. If your system will "break" from the slightest change in the spread or from a delay of the opening of a position by a few seconds, it is unlikely to survive in the "real fight".

      Remember that spread is the reason of lose deposit for majority of the strategies, ie the choose of deal direction with correctness probability close to 50%, so try to improve the strategy as much as it possible, so the spread would not greatly affect the result.

If you still firmly believe in your system after all the checks, and you still have the desire to automate it, then we can move on further.

### 3\. Determine the Goals

![Determine the goals](https://c.mql5.com/2/2/target2_small.jpg)Have you ever wondered why you need an Expert Advisor ? Do you just want to check your idea in the [Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester")? Or, maybe, you want try a ready strategy on the micro-real account? No, I know - you have been trading on your system for 2 years and you will launch the Expert Advisor immediately on a real account with a deposit of $100 000.

These seemingly useless question are very rarely asked. Generally, it is understandable - the answers affect only some small things: the details of the technical requirements, the choice of the Applicant/Developer, and, perhaps, the cost of the work. But if you think about it, the difference between the different approaches can be very substantial.

In most cases, the programmer doesn't care whether to write the Expert Advisor for the Strategy Tester (with a minimum of checks and without handling of exceptions) or for the real trade. If he is writing in MQL for a long time, and is not receiving any complaints about the stability of his programs, then he most certainly had a basis for the implementation of any algorithm, and there will not be any differences in the approaches to writing the Strategy Tester version or version for a real account.

But even if you are lucky enough to work with such a person, you will feel the difference when drawing up the technical requirements.

Let me explain on an example:

1. Situation number one - you simply need to test an idea(you have found/heard/read it somewhere). It means that:


   - We are not planning to run it on a real account - all sorts of checks and the handling of exceptions do not need to be done.

   - The Expert Advisor will actively tested and optimized - we need to achieve the maximum performance (perhaps at the expense of robustness).

   - Most likely, after checking the first version, it will need the refinement - an Expert Advisor should be easily extended and expandable.
   - The "bottlenecks" of the algorithm (setting of orders at the minimum allowed distances, the sequence of opening and closing positions, the bypassing of the maximum lot limitations, the restart of the work after disconnections, etc.) do not need to be processed, because we don't know which ones of them will maintain their relevance in the final version.
2. Situation number two - the strategy is ready, you need a valid Expert Advisorfor the real account. From the previous version it will differ in the following way:
   - All trading operations should be as correct as possible, not only the values of the user-defined parameters need to be tested, but all of the values are calculated during the working process (lot size, the levels of stop orders, etc.).


     The "quality" of the trade requests directly affects the "relation" of the server to your account - if the server is bombarded with incorrect trade orders, it could easily block the account, and you will lose control of the situation.


      In the case of a critical error, you must let the the user know (through a message on the screen, a email, Skype, ICQ, or via SMS).
   - The robustness of the work, as opposed to the speed of its performance, is brought to the forefront - if some additional checking will help to prevent a possible error, it should be included in the Expert Advisor. Even if it slows down the testing and optimization (when working in real-time, the execution speed is most often not palpable).
   - The future improvements, even if they are needed, are minor. Therefore, no specific requirements for the ease of extensibility are specified.
   - All of the algorithm "bottlenecks" should be exhaustively thought out and accurately processed. There should be a minimum of situations, in which an error can occur (we won't be able to make provision for all of them, unfortunately):
     - The Expert Advisor should be able to restore its normal work after a temporary disconnection, reconnection to the account, or a reset of the client terminal.
     - If you can not set the pending or stop orders, they need to be tracked virtually, and when the price of their levels is reached, executed on the market price.
     - If you can not open position on the market prices within N attempts, the maximum allowable slippage should be increased.
     - At the triggering of a pending or stop order, by a level, not provided by the algorithm, and with slippage (eg, with a price gap), the levels of all of the dependent orders should be adjusted (and, perhaps, even their volumes).
     - All functions that operate on the size of the deposit (for example, the calculation of the lot), should normally accept non-trade operations with a balance (balance or withdrawal). The list can go on for a long time. There are an infinite amount of such nuances and each strategy also has its details.
   - Also, there are additional requirements put forward - the Expert Advisor should work fine with different brokers, take into account the list of available instruments, their specifications, and other server settings (the maximum number of pending orders, the Stop Out levels, the opportunity to open the positions/orders in opposite directions (for MetaTrader 4 only), and so on) .
3. And situation number three - you really are ready to launch the Expert Advisor on a real account with a deposit of $100 000. Will something change, as compared with the previous versions for real account? Everything depends on your paranoia (sorry, I meant to say on your foresight):


   - There will be absolutely no harm in creating of more log files, and a regular saving of screen-shots of the graphs, they may become very useful in analyzing a problem or dispute.
   - If you plan to run the Expert Advisor on the dedicated server, you can envisage the possibility of control/correct of its work with another (parallel running) copy. For example, you can simultaneously run the Expert Advisor on the dedicated server and at home, and the "home" copy should be able to take control of the situation, in case of a disconnection of the "server" copy for 5 or more minutes.
   - If you can not constantly monitor the work of the EA, you can implement an hourly (or daily) reporting about the current state of account and trade situation. The notifications of critical errors should be present in any case.
   - If you want to be able to control the trade process, even when you only have a phone, you can implement a "feedback" - the ability to send to the Expert Advisor the commands via SMS, email, Skype, ICQ, or by setting pending orders (with description of commands in the comments).

Well, do you feel the difference? And this is just a glance of the method, each of these elements can think through out and extended to infinity. And there are so many details that we have not yet mentioned!

After exploring the details, it becomes clear that thinking through the algorithm details really depends on the purpose of the Expert Advisor and the customer's requirements.

You can not expect from the programmer an equal quality of implementation of all of the nuances, such standards of quality, unfortunately, do not exist yet. Therefore, determine what you need the Expert Advisor for, think about how it should behave in different situations, and be sure to mention this in the technical requirements.

The standard handling of simple errors (requotes, invalid stops, etc.) will be added by any self-respecting EA developer. But it's not guaranteed that this "standard behavior" will be suitable to your specific strategy.

For example, it is not always best way to set the Stop Loss level to the minimum allowable distance, if it is impossible to set it to the calculated level - the other orders, or the maximum loss of the ongoing series of positions may be dependent on it. So, again, if you're on your way to creating an EA for a real account, consider a maximum amount of scenarios at the preparation stage of your work.

You can always ask the programmer to prepare a list of possible situations and to develop an algorithm of their handling. But do not forget that the analysis and refinement of your strategy, as well as the creation of the algorithm, are not directly related to writing the Expert Advisor. It is a separate part of work, and its result is not the Expert Advisor code, but the text of the algorithm. Some may quiet reasonably require extra payment for this work - since he is investing his time, and you are getting, although intermediate, but still a result, and can go with this algorithm to another programmer.

And do not forget that not all programmers have experience with working on real accounts. Basically, they may simply not be aware of all of the possible surprises. Well, probably there isn't a single person who has worked with absolutely all of the companies, with all types of accounts, with all possible instruments, and at all different market conditions. Share your experiences (if any), ask the programmer to share his experience, but do not think that you are insured against all surprises. This, unfortunately, is unrealistic.

We have come close to the most critical part - the preparation of the technical requirements, but not before we make one small deviation.

### 4\. Synchronize the Vocabularies

Before you begin a dialog, try to understand who you are going to communicate with. "On the other end of the line" there is a completely different person, and his knowledge may be quite different from yours.

Until you "synchronize the vocabularies" (find a common language, define the terms), the movement towards a common goal can be unpleasant and difficult. Elementary, in your opinion, things, can cause so much confusion that the development of a simple Expert Advisor can easily turn into many days (or even weeks) of ascertaining the relationship.

![](https://c.mql5.com/2/2/santehnik.jpg)

Let's take a little detour.

Imagine that you need to change your faucet. You call a plumber, and say: "I broke a thing with which I wash my face!" I want a new one, that I will be able to turn on with one hand". Sounds silly, doesn't it? But believe me, some of the tasks, sent by traders, look even sillier!

Now try to put yourself in the place of this plumber. Yes, he understood approximately what was going on, but he can not do anything for this "task". And only a telepath will really understand what kind of faucet you want.

Let's try this is a different way. You call and say: "I need a new faucet in my bathroom. Diameter of pipes for hot and cold water is 13 mm, the distance from the pipe to the base of the sink is 20 cm. The adjustment of the pressure and water temperature must be made with one handle. It is desirable to have a choice amongst several models.

Bingo! Now the plumber knows that: a) you need a faucet handle; b) it must be with two valves, and a single handle control; c) he will need to connect a hose with a length of 20-25 cm and a diameter of 13 mm. When he arrives at your home, he will be able to provide you with a few solutions to choose from and quickly do his job.

Many programmers who write in MQL, are really knowledgeable in trading - the more they communicate with customers and program Expert Advisors, the better they understand the diversity of approaches to the analysis. In addition, many of them develop strategies for themselves, and so they have also studied trading literature and participated in thematic forums.

But don't demand too much from them! The phrase "stretch the fibo-grid onto the last 2 fractals", which you may use in your every day life, may lead an inexperienced technician into a deadlock. Finish him of with the commonly used phrase "after the triggering of the lock..." or "at the opening of London, set the stops to break-even level", and the client is ready - you are guaranteed with long hours of questioning and clarification.

I'm not saying that the common "MA" should be called "the technical indicator with a moving average with a smoothing period of 36" - no fanaticism is required! But always try to remember that the person you are talking to has a different baggage of knowledge and a different vision.

As an intermediate result - a few tips:

1. Explain yourself with simple and understandable words, without using abbreviations and jargon.
2. Use common terminology. If you don't know how to call some tool or an event, do not hesitate to use the wording from a reference or a textbooks.
3. Explain things thoroughly and in detail. Talk even about what you consider obvious. Very often, it turns out that it is only obvious to you!
4. Finally, make sure that the other person understands you. Ask leading questions, or ask him to formulate the task in his own words - make sure that you are talking about the same things.


Dear Colleagues, EA writers! If you have mistakenly thought that the foregoing relates only to traders, I will hasten to disappoint you! We, as the technically competent people, are responsible for the correctness of the entire work process. And we, to a greater extent, are responsible for its success.

Very few traders are able to formulate the task accurately and correctly - this is a fact. In the six years of my EA writing experience, I remember only two clients, whose tasks were really stunning - these were ready-made programs, which simply needed to be transferred into MQL. Several more people were tagged as "they understand what they want" and "are able to express their thoughts" - after reading their algorithms, I clearly understood how the future programs will run. But the majority of the customers - are beginner traders, they are frightened not only by the need to clearly describe the actions of the EA, but of the word "algorithm".

Be a little bit of a psychologist - determine the level of your partner, and use the corresponding to his (level) terms and concepts. Do not crush your intellect upon him, be lenient. If possible, get rid of all of the programming nuances in the discussion, the person absolutely does not need to know what a cycle is, where the numbering of bars starts from, and under what conditions the function, that saves the ordering information in the file, is executed.

This does not mean that the customer must be "baby-talked with", and that at the slightest sign of disagreement, the EA should be rewritten. No way! But try to become to him not only a good technician, but also a pleasant conversationalist - he will definitely appreciate it.

We have finally reached the most critical part - the making of the algorithm. Well, let's get started!

### 5\. Preparing the Requirement Specifications

![Preparing the task](https://c.mql5.com/2/2/slovar1_small.jpg)

Your cousin, having nothing to do with trading, should be able to trade on your system, having only your prepared algorithm.

This classic phrase, changed only by a bit, very clearly demonstrates the main qualities of good technical specifications:

1. This task should be fully self-sufficient (understandable to a totally unfamiliar with your system person).
2. The task should be as detailed as possible (understandable even to an inexperienced, in matters of trading, individual).

How categorical and exaggerated is the phrase "it has nothing to do with trading". Indeed, it will be difficult to explain to a person, in addition to the essence of the system, the basics of trading and using the trading terminal. I will allow myself to turn once again to a classics phrase:

Your cousin, who has installed MetaTrader only two month ago, and who has traded only on demo accounts all this time .... the rest we know.

But this doesn't change the essence! If you give your algorithm to ten different people and ask them to trade on your system, they must obtain absolutely identical results. How many of you can boast of such experience?

**5.1. Where shall we start?**

We'll start from the beginning. Describe the general idea, tell what it is that you want to get. Do not forget about your own goals and requirements for the program, they can be formulated now.

Having a general idea about you and your system, the programmer can assess the seriousness of your intentions, and assume the approximate amount of the work.

A few examples of such entry:

We need an indicator for MetaTrader 4. The task - to draw an inverted chart of an arbitrary currency pair in the main chart sub-window. The indicator should work on 4-and 5-digit quotes and with non-standard symbol names (eg, EURUSDFXF).

We need an Expert Advisor for MetaTrader 5. Trading on a single currency, the entry signal is based on the custom indicator (code is attached). Closing of positions - by the SL, TP, and by a reverse signal. All positions are accompanied by a trailing stop, based on another indicator (the code is also available). The lot is calculated as a percentage of the balance.

It is necessary to finalize the Expert Advisor (MetaTrader 4) to work on a real account - place all of the necessary checks, restore normal operation after a connection failure, add a few attempts to open a position with an error, and maybe something else - at your discretion.

We need an EA to test a strategy for breaching a channel. Determine the borders of the channel by the indicator; the entrance to the market by the pending orders, after a failed deal - an increase of the lot with specified coefficients. You should also specify the work time during the day, when the setting of the first orders is allowed. And so we have the code for MetaTrader 4.

This part of the task is the least difficult. But, unfortunately, often this part is where it all ends ... Do not forget it - it is only an introduction, the description of the general idea. To write a program we need a lot more.

**5.2. How do we formulate the task?**

In fact, the algorithm is the most important part. It is most difficult to formulate (especially for a non-programmer), and it really requires a lot of hard work.

Try to immediately divide the algorithm into logical sub-sections, do not try to convey all of the subtleties of the system in a single illegible sentence.

When it comes to writing the Expert Advisor, we can distinguish the following logical sections (their number and content may vary depending on the strategy):

1. The general conditions: the work time (within days, on certain days of the week), the order of the execution (for example, the beginning of trading with the pressing of the button), the necessary for analysis depth of the history, and other conditions, related to the entire task in general (not to its individual points).

2. The market entry signal (the opening of the first position or the setting of the first orders) is based on indicators, certain price patterns, simply on time, or on the user's command. This may also include limitations on the first entrance (filters) - by time, by another indicator, after too long a series of losses, during unacceptable trading conditions, (too large spreads or a StopLevel), or during a lack of available funds. The method of the lot calculation and the level of Stop Loss and Take Profit should be specified separately (if the rules for calculation for all of the positions are the same, we can separate them into an independent clause).

3. The processing of pending order triggering or SL/TP of a position (if necessary). For example, removing a Sell Stop order when Buy Stop order triggered, the setting of an additional Sell Limit order, when a previously set Sell Limit order triggered, the opening of a position to sell a doubled volume after the Stop Loss triggering, and so on.

4. The signals for the opening of additional positions (if there are open positions) or for the setting of additional orders (if necessary). For example it can be done by the signal of other indicators or when reaching of certain profit of loss of the opened position. Here, there also must be the rules of calculation of the lot, of Stop Loss and Take Profit (if they differ from those that were described above).

5. The trailing of positions and orders (separately - the first in series, separately - the additional ones, or all together, if the rules are the same). For example, the pulling up of a pending Stop-orders to the market (if price movement in the opposite direction), a Trailing Stop of the position (conventional, by the indicator, or some other), the partial closing of the position when reaching a certain profit, and so on.

6. Signals for closing (full, partial) or a position reversal. The rules for pending orders removing.

7. The general conditions, such as those associated with the account state - the closure of all positions and the stopping of work when specified drawdown, a reduce of the deposit percentage use, with the increasing of the balance to a certain level, and the like. These conditions apply to the first point, but it is easier to describe them in the end (as in the order of usage).

8. And in the end, maybe some additional information on the chart is needed, the drawing of arrows of position opening/closing price levels, the detailed information in the journal, the sending of emails when triggering of pending orders, and everything else that is not related to trade, but is related the interface.


If we are talking about an indicator, then on the one hand, everything is much easier - the logic is more primitive and less complex, but on the other hand, there are some subtleties.

The compendium is something like this:

1. The required data: a list of the analyzed instruments (if several), the depth of the history for all of the used instruments and time-frames, the time zone of trade server.
2. The drawing type (lines, signal arrows, candlestick chart, sections like in ZigZag, geometric shaped figures, etc.).
3. The algorithm for calculating the first value (the value on the left bar), if it is differs from the main algorithm.
4. The basic algorithm for the calculation of a single bar or the description of the calculation process, if it is difficult to derive a formula for an individual bar (as in the case of a Zig-Zag, for example).

5. And the nicely-comfortable stuff, if needed: audio signals, saving of screenshots, sending notifications to the email, etc.

Let's try to move from theory to practice, and finish some of our sample-tasks:

We need an indicator for MetaTrader 4. The task: to draw an inverted chart of an arbitrary currency pair in the main window of the chart. The indicator should work on 4-and 5-digit quotes and with non-standard symbol names (eg, EURUSDFXF).

1. In the settings, specify the name of the instrument (the _**symbol**_  parameter), which needs to be displayed (for example, _"GBPUSD\_m"_). If the name is not specified, use the symbol of the chart, on which the indicator is attached. If there isn't such a symbol in the "Market Watch" window, it is necessary to display a window with an error message.

2. The indicator should be plotted as Japanese candlesticks. The colors of the growing and falling candlesticks, and the shadows (separately - the upper and the lower) must be designed as input parameters.
3. The calculation of the OHLC values for each bar is made by the formulas:
   - _Open (indicator) = 1 / Open (_ _**symbol**_ _)_;
   - _Close (indicator) = 1 / Close ( **symbol**);_
   - _Low (indicator) = 1 /_ _High_ _( **symbol**);_
   - _High (indicator) = 1 /_ _Low_ _( **symbol**)._
4. At all of the "round" price levels (1.3200, 1.3300, 1.3400, 1.3500, ...), that is, levels that have a 4-digit multiples of 100 points, you must draw a horizontal line (style and color should be designed as input parameters).


This will be a little harder with the Expert Advisor:

I need an Expert Advisor for MetaTrader 5. Trading on a single currency, the market entry signal is based on custom indicator (code is attached). Closing of positions - by the SL, TP, and by a reverse signal. All positions are accompanied by a trailing stop, based on another indicator (the code is also available). The lot is calculated as a percentage of the balance.

1. Signal to open a position - the arrow of the _iSignalArrow_  indicator (all of the indicator parameters should be adjustable):
   - A **long-position** is opened if the indicator's arrow is pointing **up**(below the chart) on the last closed bar;
   - A **short-position** is opened if the indicator's arrow is pointing **down**(above the chart) on the last closed bar;
   - Arrows on the current (uncompleted) bar are ignored, only the completed bars are analyzed.
2. The position volume is calculated as a percentage of the current balance: _Lot = Balance/MarginRequired \* **LotPercent** / 100_  where:
   - _Balance_ \- current account balance;
   - _MarginRequired_  \- the margin required to open a position with a size of 1 lot;
   - _**LotPercent**_ \- an input parameter (the percentage for the calculation of the lot).
      For example, when the _**LotPercent**_ = 5, with a 1:100 leverage, the lot for EURUSD (at current price of 1.3900) will be: 10 000 / 1 390 \* 5 / 100 = 0.3597

      The obtained result is rounded off by the usual rules to the nearest correct value (up to 0.36 - if the DC allows lots with a precision of up to 0.01, or up to 0.4 - if the lot step = 0.1).
3. StopLoss (SL) and TakeProfit (TP) - are fixed, adjustable by the parameters _**StopLoss**_ and _**TakeProfit**_:
   - The levels are specified in points of 4-digit quotes;
   - The levels are calculated relative to the price of opened position (**the Ask price** \- for the long positions and **the Bid price** \- for short positions);
   - If the value is too low, the stop should be set at a minimum allowed distance;
   - If the 0 values specified, the stops are not used.
4. All of the open positions are accompanied by trailing stop by the _iTrailingLine_ indicator(all indicator parameters should be adjustable):
   - If a **long-position** is opened,  and the indicator line is **below** the current price, the stop is moved to the level of the indicator line;
   - If a **short position** is opened,  and the indicator line is **above** the current price, the stop is moved to the level of the indicator line;
   - The indicator values are taken from the completed (formed) bar, the current bar (uncompleted) is not used. That is, the modification should occur no more frequently than once in a bar;
   - Moving the SL is permitted only in the direction of profit of the position - **up** for long position and **down**for short position;
   - If you can not set the SL at the level of the line, it should be set at the minimum allowable distance (but only if it complied with the previous rule of moving in the direction of the profit);
   - The trailing stop feature should be configurable ( _**AllowTrailing**_ = true/false parameter).
5. If there is an opened position, and there is a opposite signal, the opened position must be closed and a new one must be opened (in the opposite direction).

    For a new position the calculation of the lot must be called **after closing** the opened position.
6. Miscellaneous:
   - When you run the Expert Advisor, it should attach the used indicators with the specified parameters;
   - The information on the opening/closing of positions and of modifications of the SL should be stored in the Journal;
   - If there are any errors, it must print a message describing the error.

In this form, the algorithm can be sent to the programmer - it contains enough details about the system and can easily be "translated" to MQL. But do not rush with the order, think it through to the end.

**5.3. What must not be forgotten?**

![What must not be forgotten?](https://c.mql5.com/2/2/reminder_1_small.jpg)A program, written based on one of these algorithms, will work fine under ideal conditions - in a separate client terminal, with a single access to the account, without the intervention of the user or other programs.

An example of such an environment - a Strategy Tester, there are no connection lost errors, accidentally closed positions, and other trading Expert Advisors. But in everyday life, such conditions are extremely rare, most likely the program will operate with the "outside world".

You may want to run multiple copies of the program on different symbols or with different settings, you will restart the terminal, trade on the account manually or with other Expert Advisors, connect to different accounts from one terminal - all of which can affect the program, if the processing of these situations is clearly not provided by the algorithm.

If your goal has a larger scale than just testing the program in the Strategy Tester, describe the rules of interaction with the outside world right away:

1. How the Expert Advisor should react to positions, opened manuallyor other Expert Advisors?

    Usually people choose one of the three options:

   - Completely ignore all of the "other" positions. If the system is self-sufficient, then the trading actions of other Expert Advisors or the user, should not influence it.
   - Work only with "manual" positions. If the Expert Advisor is designed to accompany the manually opened positions, it should not interfere with the positions of other Expert Advisors, and it usually does not open position of its own. Its task is to help with the manual trading (move the Stop Loss, close the position by the signal, and so on).
   - A more universal solution is to give the user a choice: to work only with its own positions or to accompany positions that meet specific conditions (for a particular symbol or with a specific Magic Number).


In MetaTrader 5, the separation of trades into "their own" and "other" is particularly relevant - the terminal displays only the total position of the symbol, even if it was "collected" from the deals of several different Expert Advisors. The implementing of a complete accounting of the deals (for the normal operation of multiple EAs on a single symbol) is more difficult in its implementation, and thus may be more expensive. Check with the Appliciant/Developer, whether the Expert Advisor will normally work with other Expert Advisors, working on the same symbol.

2. How should the Expert Advisor respond to a connection to another trading account? Is there a need in any special procedure to run it on the real account?

    I think that many traders can "boast" about their losses, caused by simple lack of attention - accidental connection to a real account, launching a terminal with a running Expert Advisor, or a change of Expert Advisor parameters in the presence of open positions. These absurdities can be avoided simply by considering them in the algorithm.

    For example:
   - When you run the EA on a real account, it should create a button on the chart, which permits trading. The work should begin only after the user clicks on it.
   - When you change an account, the EA should notify the user about this and stop its work until a new launch is executed (an alternative - to ask the user whether it should continue its work).

   - If there are positions, opened by the EA (or set orders), when there is a change in the external parameters, the EA should modify the positions (orders) in accordance with the designed algorithms - for example, if there is a change of the value _**StopLoss**_, we need to modify the Stop Loss of all open positions, but only if it has not yet been moved by the trailing stop. Here it is impossible to give a universal recipe, each parameter should be described separately. In addition, for different strategies there can be different reactions to the changes in the same parameter.
3. Is there a need to run several copies of the programwith similar (or identical) parameters?

    If the Expert Advisor places the graphic object on the chart, by running multiple copies of the indicator, and changing the value of only one parameter, then all of the objects, created by it, must have names containing the value of this parameter - otherwise, every next run will distort the results of the previous ones.

    In the case of the Expert Advisor, usually a special parameter is added - ExpertId or MagicNumber, which allows you to run any number of copies of the EA with any set of other parameters. Specify in the task, for which settings and for what combinations there is a capability of simultaneously running the program, not all situations will be provided by the programmer himself.
4. How should the migration of the EA into another terminal, connected to the same account, be implemented? Can the Expert Advisor store some data in the files or in global variables of the terminal?

    For most programs there is absolutely no need to store intermediate information, their algorithms are based on historical data of currency pairs and on trading history of accounts (this data can be obtained from any terminal, connected to your account). But it is often needed to store some information in a file, and to retrieve it at the next launch - sometimes this allows to accelerate the speed of execution, and sometimes, it is simply impossible to create a viable program without it. Notify the programmer of any special requirements for the process of transferring the EA, or just ask him to describe this process specifically for your case.

All of the subtleties, unfortunately, are impossible to be provided for. For example, if there is not enough margin for opening a position (because of positions, opened by other EAs), computed in the EAs lot, then you will either have to skip the signal or open with a smaller volume.

If other EAs take up the trading context (in MetaTrader 4 only), your EA will not be able to trade. And if there is a limit on the maximum number of pending orders, new order will not be able to be set. However, because of the fact that most of these instances are provided for in your algorithm, it will not worsen the situation. Gain experience, and each new version of the EA will be better and more reliable.

**5.4. How can we simplify the understanding?**

Information is digested much easier if it is well illustrated.

To understand a simple strategy, it is sufficient enough to have a text description, but if your system is unusual and complicated, take a few steps to help out the programmer:

1. Attach to the task a few screenshots, Illustrating the different points of the algorithm (the time of the occurrence of a signal, a demonstration of the work of a trailing stop, the sequence of setting pending orders, etc.). Do not hesitate to supply the graphs with brief comments, even if they are partially duplicated in the text of the algorithm.
2. Format the textof the task with tastefulness: use different colors for **Long** and **Short**positions, highlight the _**external variables**_ (the parameters that you want to be able to configure), mark the **important points** and _formulas_. In addition to the fact that the text will be easier to read, it will be much easier to navigate through it.
3. Give examples. Any _formula_, Illustrated by concrete numbers, becomes much clearer.
4. Number theparagraphs and sub-paragraphs of the algorithm - so they can always be referred to during discussion. "Error in a position 2.1.4 - is much shorter and more accurate than "an error in the place where the Stop Loss level for the second long position of the series is calculated.

**5.5. Text, voice or video?**

![Text, voice or video?](https://c.mql5.com/2/2/text_voice_video_1.png)I always find it hilarious when a client, instead of sending me the task, sends me a link to a 120-page discussion of the strategy on a forum, a 70-page book, or a one and half hour video lecture. Indeed, programmers have all of the time in the world, nothing to take care of - they'll take the time to study this ... The fact that the useful part of this information will fit a half of a page algorithm, or the fact that it is simply impossible to formalize this description, does not seem to concern anyone.

If you have already researched this material, if you understand what it's all about, if you have a good idea of how the strategy will work - just formalize the algorithm! Remove the "waste" (which usually composes 80% of the info), the awkward pauses, the distracting discussions, the stories about the bright future, the observations of testing results - and the Appliciant/Developer will receive only what he really needs to write the program.

But if you still don't know what the book or lecture is about, if you're not sure that this is enough to make a fully automated trading system, then formulate the question differently! You can only ask "how much will it cost to write an Expert Advisor based on this strategy?" when you have a strategy.

And in our case there is only a certain amount of information. It's quality (whether it is amenable to formalization, detailed enough, etc.) - is still unknown. So feel free to ask whether the programmer is interested in studying this material "for the idea", and if not, how much he will take for writing full-fledged rules for a trading system out of this "lengthy discussion". Believe me, even the form of the question itself will illustrate your relationship to the Appliciant/Developer.

Not all programmers are interested in reading multi-page papers or discussions of some strategy - they have enough of their ideas that are pending verification, and new information is simply of no need.

Not everyone will want to watch a lecturer talk about the construction of trend lines and their role in his system, instead of their favorite movie. This is often quite boring, and what's worse, difficult to formalize. Some moments will have to be literally invent yourself (find the most logical explanation), some things will have to be guessed or selected through experimentation, for some issues, you will need to seek out and research additional information. In general, the process is quite laborious and creative, do not underestimate it.

I want to separately mention those who like to communicate via Skype or phone. Most often, the desire to describe your strategy orally is due not only to the reluctance to take extra actions (type on the keyboard), but to what's worse, the lack of understanding of the strategyby the author.

It is impossible to construct a set of trading rules, if they are based on guesses and intuitions of the author, it is very difficult to structure an emotional and messy story. And, as is the case of the long video lecture, the programmer is not always interested in listening to these "revelations" - since in order to write the program, he needs an algorithm, and someone will still have to write it.

It is difficult to overestimate the importance of modern technologies - it is much easier to find a common language when communicating by voice or video, and by showing an image from your monitor to your companion. But the conversation will be much more productive if there is something to discuss - formulate your thoughts on paper, and you can prepare a full-fledged task right in the course of the discussion, simply by making minor clarifications to an already prepared text.

The conclusions you should make from the above are:

If you can draw a clear strategy algorithm from the description - draw it yourself and submit to the programmer only the necessary information.

If the process of formalization is very complex and requires a lot of work - do not expect that the programmer will do it for free.

I hope that now you have a good understanding of what a task is, and we can talk about choosing a particular executor.

### 6\. Selecting the Applicant/Developer

![Selecting the executor](https://c.mql5.com/2/2/Vibor.jpg)The question of choosing a programmer became vital as soon as this choice came into existence. Every customer wants to pay as little as possible and get as the highest quality results. Ideally, the software should written by the best professionals, and at the same time, be free of charge. This is an ideal to strive for, maneuvering through the expensive professionals and the beginners.

When selecting the developer for the implementation of your first [job](https://www.mql5.com/en/job), I advise you to evaluate these criteria:

1. Experience **of public** program writingin MQL4/MQL5.

    If the EA developer entered the market a week ago, he also might leave in a week. A "pro" with two years of experience, of course, may also suddenly disappear, but the probability of this is much lower. Losing communication with the developer threatens, not only the ability to make a new order, but also:


   - The lack of support (there isn't any software without errors, there are only badly-tested programs);
   - The complexity of making even minimal improvements (other people's code is always harder to work with, so another developer may request a decent amount, even for minor changes.)
2. Feedbacks of real customers.

    The practice is the criterion of the truth. If a person wrote programs that are used by real people, it means that they work. Otherwise, you would always have stumbled upon feedbacks, exposing his lack of professionalism.

    If you have trader friends who have already used the services of a programmer, ask them for advice - at least you will then get the expected result.

3. Online availability.

    No one likes to wait for an answer to a letter for a number of days. And for some, even 2 hours is too long.

    Observe the person - whether he is frequently "online", whether or not he is quick to respond to messages. In the future, this can save you a tremendous amount of time.

4. Communication methods.

    Today there are a variety of ways to communicate via the Internet: e-mail, the messengers, programs for voice and video chat, private messages at Internet resources. Someone are accustomed to one method, while others to another. For good communication, you will need to select a method that is comfortable to both of you. There is no problem in installing one more program, but some people may not want to do this, because they don't not see a need for this.

    If you need live communication (ie via Skype), check with your developer whether he is ready for it. Especially, arrange in advance if you want to talk on the phone or meet in person - not everyone will agree to carry out their work "off-line".

5. Terms of cooperation.

    Before making an order it is necessary to find out all of the conditions for cooperation:



   - Is it possible to work through the ["Jobs" service](https://www.mql5.com/en/job)? A positive answer to this question will nullify most of the others.

   - Is prepayment required, and if so, how much is it?
   - How the program will be checked? Will you get a demo version?

   - Will you get the source code of the program, and if so, when - after the complete payment or right away?
   - Who will own the rights for distributing the program?

      Checking the integrity of the programmer is unlikely to succeed, but it is still worthwhile to formally clarify this point: if he is repeatedly suspected of selling client EAs or algorithms, his reputation will go a long way ahead of him.

   - How long will be the technical support and on what terms? Will the fix of errors (discrepancies from the algorithm) be done free of charge? How much will small improvements cost?

All of these nuances should be clarified before the beginning of the financial relationship, since it will be harder to do afterwards.

6. Payment Methods.

    What electronic payment systems does the developer use? Can the payment be done by a bank transfer or a credit card?

    Be sure to check the requirements for the currency of payment - the Internet is international, and so not everyone will need the Russian Rubles.

7. Character.

    If you are not looking for a single co-operation, but a constant partner, try to find out whether you suit each other in your characters. Socialize with each other, indicate your requirements from the other party, describe your world-view, your principles, and weaknesses (to the extent that your self-criticism will allow).

    An alternative - try to order at first pick, and sort everything out in the process. But then it may happen that this partner will be found far from the first attempt.

8. Cost of the work.

    Last but not least. I would not trust to write a serious program to a man who values his work at $10, but I am also not willing to pay $1000 - so I have to choose something in between. The price, on the one hand, indicates the professionalism of the executor, and on the other, depends on his interest and work-load. Don't expect to get a "Mercedes" for $5 but do not overpay for a "Buick".


We have repeatedly raised the topic of "a list of programmers" - the list of some ready-to-work specialists with their contact information and customer reviews. There have been several attempts to create such a list - by myself in the ["An Expert Advisor Made to Order. Manual for a Trader "](https://www.mql5.com/en/articles/1460)  article at mql4.com, by independent forum users, and just by people indifferent to this subject. And it really could facilitate the selection for the first [order](https://www.mql5.com/en/job).

But just like a few years ago, there is no comprehensive and constantly updated list. We can work together to make another attempt to create it, but I think that the discussion of this idea is beyond the scope of this article.

### 7\. Protected Yourself from Getting Cheated

There are different people and there are different situations. Even the most trustworthy EA developers may disappear without completing the work, and even the most responsible person can violate the terms of agreement under the pressure of circumstances. Do not take chances where they are not necessary - use the ["Jobs"](https://www.mql5.com/en/job) service!

Steadily following the development of the strategy of automated trading, in mid-2010 the [MetaQuotes Software Corp.](https://www.metaquotes.net/ "https://www.metaquotes.net/") released a [new service](https://www.mql5.com/en/forum/1211), the main purpose of which is the organization of relationships between the customer and the developer. Even now, after only six months, the service is deservedly popular, and is daily used by many traders and programmers.

More information about this service can be gained from the [official announcement](https://www.mql5.com/en/forum/1211) on the forum and from the [article on its use](https://www.mql5.com/en/articles/117):

The main difference between the " [Jobs](https://www.mql5.com/en/job)" service at [MQL5.community](https://www.mql5.com/en) and the majority of similar resources and services on other websites, is the security. The customer and the programmer are secured from each other's negligent actions throughout the whole period of joint work. In the event of a dispute, the [MetaQuotes Software Corp.](https://www.metaquotes.net/ "https://www.metaquotes.net/") is ready to assume the role of the [Arbitrator](https://www.mql5.com/en/job/rules).

Despite all of the thoroughness and formality of the service, problems still arise when using it. The majority of them can be avoided by following a few simple rules:

1. Before processing a new order (if you are the trader), or making proposals on implementation (if you are the programmer), be sure to review the [terms of the service](https://www.mql5.com/en/job/rules). Many disputes arise due to inattentive reading or lack of understanding of the rules. If some points are unclear, or, in your opinion, can be interpreted ambiguously, then ask some clarifying questions in the [special branch](https://www.mql5.com/en/forum/1211) \- perhaps, your question will cause the rules to be more simple and straightforward.

2. When ordering an Expert Advisor, prepare a clear algorithm.

    There are several chapter of this article devoted to this aspect, so I will not repeat it here.

3. Choose an adequate Appliciant/Developer with an adequate price and an adequate timing for the task, do not be tempted by a "free cheese" - there is no such a thing.

    Look at the developer's portfolio and read feedbacks of his completed jobs. Make sure that he is not too busy with other jobs - perhaps this will prevent him from finishing your order on time.

    And remember that the time of implementation is counted after completion of the [2nd step (Requirements Negotiation)](https://www.mql5.com/en/articles/117) \- the process details are not regulated, and job schedule depends only on you and on the programmer.

4. Have all discussions, using the messages in ["Jobs"](https://www.mql5.com/en/job) service - only then they can be used in [Arbitrage](https://www.mql5.com/en/job/rules) cases.

    Even if you are communicating through ICQ or Skype, try to "document" all of the key moments in the comments of the Job service.

5. Track the updates of the works, which are related (doesn't matter - as a customer or as an executor): regularly look through your personal messages, allow the sending of notifications to your email, or add your mobile phone number to your profile in order to receive SMS notifications.

    If you do not keep track of the workflow, it may get completed without you, and not in your favor - at the [expiry date](https://www.mql5.com/en/job/rules) it may be forcibly closed by the other party.


Otherwise, the use of the service is no different from the work without intermediaries, except that the Developer will pay a small commission to the ["Jobs"](https://www.mql5.com/en/job) service for organizing the process.

### 8\. Check the Results

The last stage of our journey is to check the completed job. To make sure that the program works according to the approved algorithm, you must carefully and thoroughly test it.

1. Test it in different conditions: on different types of accounts, currency pairs, time frames, with different combinations of parameters - the program should work the same and correctly in every situation (if the "indulgence" is not clearly specified in the algorithm).

2. Check the program not only in the [Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester"), but on a demo account. The Strategy Tester will help you to quickly find the obvious errors, and allow you to check the strategy at different intervals of history, and the online testing will show how the program will work in conditions close to real ones. You can create "distractions" for the program - restart the client terminal, connect to different accounts, run other Expert Advisors or indicators, change the settings during the work - it is better to learn about the features of its behavior in different situations at this stage.

3. Compare the work of the Expert Advisor with the approved algorithm, rather than to your expectations of the system profitability. If it turns out that the algorithm contains an error during the checking process, please make the necessary changes and ask the developer to modify the Expert Advisor. But do not expect that he will do this for free (especially if the improvement is significant), this error is not his fault.


If you find a problem, report it to the Appliciant/Developer.

1. Indicate the part of the algorithm, which is processed by the program incorrectly (or at which the incorrectly actions begin).

    If it is difficult to find the specific place where the logic is violated, then explain the problem in your own words, but still try not to deviate too far from the algorithm.

2. Describe the conditions under which the testing was performed:
   - Attach a set-file with parameters of the program (the "Save" button in the "Options" window of the Expert Advisor);
   - Specify the used currency pair and time-frame of the chart;
   - Specify the server address to which the terminal was connected, and the type of account (demo, real, contest, or otherwise);
   - Specify the build version of the client terminal ("Help" menu - "About");
   - If you checked it in the Strategy Tester, specify the Strategy Tester's settings (testing period, execution type and mode, initial deposit, leverage).
3. Attach a screenshot, illustrating the problem.

4. If the problem is related to opening or closing a position, copy the excerpt from the Strategy Tester Report or a few lines from the account history.

5. Attach the log files of the Expert Advisor (select "Open" in the context menu of the "Experts" tab of the client terminal or in the "Journal" tab in the Strategy Tester).


The more information the programmer has, the easier it will be to find and fix the problem. I hope that after all these step descriptions you will get exactly what you wanted. But do not hurry back to the busy weekdays of a trader, share your experiences with others.

### 9\. Provide Feedback

![Provide feedback](https://c.mql5.com/2/2/feed_back_300.jpg)

[MQL](https://www.mql5.com/) is a growing community, and you are a part of it.

Remember how you chose a programmer and prepared your first task specifications - what might have helped you in the process? Share your experiences!

Write a clear algorithm, and let it serve an example to follow. Help the newbies to formalize their strategies or show that it is impossible - save another grail seeker from a collapse of hopes.

Tell about your experiences with programmers, indicate their strengths and weaknesses, describe - what you liked the most and what was a problem. This will take you 10 minutes, but it will save a lot of nerves and money to your fellow traders.

I have a positive attitude to any constructive comments, and would be grateful for any criticism of this article. If you believe that some sections need to be worked on, that something was missing, or, conversely, there is something extra - tell me about it!

This article was conceived as a tool for writing a [Job description](https://www.mql5.com/en/job), but in fact, has covered a lot of related topics. I'd really like it to be easy to read, make all necessary disclosures, and be really helpful. If you have already read this far, please spend just a few minutes and leave your feedback. Thanks to your comments this article may become even better.

Anticipating some skepticism from my fellow Expert Advisor developers, I want to inform you that this article was written by the request of MetaQuotes Software Corp.  Its purpose is not to advertise my services, but help with the culture relationship between the customer and the programmer.

I hope that you, as a true professional, will support this initiative and help raise our overall business to the next level. I am waiting for comments and observations from you.

### Conclusion

Automated trading continues to gain new momentum. How and where it will move depends on us.

Let's create a culture of relationships now, and very soon, you will reap the rewards in the form of high-quality Expert Advisors.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/235](https://www.mql5.com/ru/articles/235)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Equivolume Charting Revisited](https://www.mql5.com/en/articles/1504)
- [Testing Visualization: Account State Charts](https://www.mql5.com/en/articles/1487)
- [An Expert Advisor Made to Order. Manual for a Trader](https://www.mql5.com/en/articles/1460)
- [Testing Visualization: Trade History](https://www.mql5.com/en/articles/1452)
- [Sound Alerts in Indicators](https://www.mql5.com/en/articles/1448)
- [Filtering by History](https://www.mql5.com/en/articles/1441)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3670)**
(75)


![Keith Watford](https://c.mql5.com/avatar/avatar_na2.png)

**[Keith Watford](https://www.mql5.com/en/users/forexample)**
\|
6 May 2019 at 15:04

**Cazz223:**

Thanks for the tipping.

About being unable to see the developer's profile before confirming the job.... I am not sure it's a sound policy though.

I totally agree with you.

Unfortunately it is Metaquotes policy and so we are stuck with it.

![jquintin](https://c.mql5.com/avatar/2020/5/5EB72706-0AF2.jpg)

**[jquintin](https://www.mql5.com/en/users/jquintin)**
\|
9 May 2020 at 21:50

Excellent!!! Thank you for the complete and comprehensive text... It helped me a lot in the preparation of my order!


![dmitriu1979](https://c.mql5.com/avatar/avatar_na2.png)

**[dmitriu1979](https://www.mql5.com/en/users/dmitriu1979)**
\|
10 Mar 2022 at 22:28

Здравствуйте скажите пожалуйста, как с вами связаться, моя платформа находится в рабочем состоянии, но брокер изчез,, т, е слился как с вами связаться,,,!!!


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
11 Mar 2022 at 04:53

**dmitriu1979 [#](https://www.mql5.com/ru/forum/3537/page6#comment_28258223):**

Hello, please tell me how to contact you, my platform is in working condition, but the broker disappeared.

Any discussion of brokers and trading organisations is prohibited in this forum. Here is the MQL5 community. MetaQuotes Company is the ONLY and ONLY developer of MetaTrader terminals and has nothing to do with brokers. MetaQuotes Company does not provide real trading accounts - only demo accounts....

I recommend you:

Before transferring funds do a little analysis about the company where you are going to transfer funds.

Find a specialised forum on the Internet and there look for ways to get your funds back.

And questions about the funds in the trading account you should address to your BROKER - that is, to whom you gave your funds.

![Hall Dave](https://c.mql5.com/avatar/2025/6/683EA076-2835.png)

**[Hall Dave](https://www.mql5.com/en/users/halldave)**
\|
23 Jun 2025 at 14:06

**MetaQuotes:**

New article [How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235) is published:

Author: [Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter "https://www.mql5.com/en/users/komposter")

My odle


![Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://c.mql5.com/2/0/brain.png)[Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://www.mql5.com/en/articles/252)

This article presents connecting MetaTrader 5 to ENCOG - Advanced Neural Network and Machine Learning Framework. It contains description and implementation of a simple neural network indicator based on a standard technical indicators and an Expert Advisor based on a neural indicator. All source code, compiled binaries, DLLs and an exemplary trained network are attached to the article.

![Tracing, Debugging and Structural Analysis of Source Code](https://c.mql5.com/2/0/Trace_program.png)[Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)

The entire complex of problems of creating a structure of an executed code and its tracing can be solved without serious difficulties. This possibility has appeared in MetaTrader 5 due to the new feature of the MQL5 language - automatic creation of variables of complex type of data (structures and classes) and their elimination when going out of local scope. The article contains the description of the methodology and the ready-made tool.

![MQL5 Wizard: New Version](https://c.mql5.com/2/0/New_Master_MQL5.png)[MQL5 Wizard: New Version](https://www.mql5.com/en/articles/275)

The article contains descriptions of the new features available in the updated MQL5 Wizard. The modified architecture of signals allow creating trading robots based on the combination of various market patterns. The example contained in the article explains the procedure of interactive creation of an Expert Advisor.

![The Indicators of the Micro, Middle and Main Trends](https://c.mql5.com/2/0/three_degrees_of_trend.png)[The Indicators of the Micro, Middle and Main Trends](https://www.mql5.com/en/articles/219)

The aim of this article is to investigate the possibilities of trade automation and the analysis, on the basis of some ideas from a book by James Hyerczyk "Pattern, Price & Time: Using Gann Theory in Trading Systems" in the form of indicators and Expert Advisor. Without claiming to be exhaustive, here we investigate only the Model - the first part of the Gann theory.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/235&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069389675510039620)

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