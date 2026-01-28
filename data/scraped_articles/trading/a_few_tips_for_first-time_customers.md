---
title: A Few Tips for First-Time Customers
url: https://www.mql5.com/en/articles/361
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:39:16.572735
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/361&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083015523681178450)

MetaTrader 5 / Trading


### Introduction

A proverbial wisdom often attributed to various famous people says: "He who makes no mistakes never makes anything." Unless you consider idleness itself a mistake, this statement is hard to argue with. But you can always analyze the past mistakes (your own and of others) to minimize the number of your future mistakes. We are going to attempt to review possible situations arising when executing jobs in the same-name service.

It will be our aim to make it as objective as possible. Whatever the case, it will represent the point of view of the executing party - a developer. Before we proceed to actual situations, I need to point out the following:

Any similarities to actual names, nicknames, images and other visual material are coincidental. All assertions and statistical calculations set out below represent personal opinion of the author of this article and do not claim to be absolute truth.

### 1\. Jobs Service - Why Do We Even Need It?

To use or not to use the [Jobs](https://www.mql5.com/en/job) service - a decision that everybody takes for themselves. My say is a definite yes, to use. And below are a few thoughts on this. Considering the main problems that may arise in the interaction between the customer and the executing party, we can highlight the following:

- Unclear or ambiguous requirements specification (RS) following which the executing party outputs something totally different from what the customer expects;
- Fears of the customer to be cheated by the executing party who can get the prepayment without doing the job or doing only a part of it;
- Fears of the executing party to be cheated by the customer who can get the finished work without paying the remaining part of the agreed amount;
- The emergence of disputable (often dead-end) situations where both are 100% sure they are right and are in no way willing to seek a compromise.

The last three of the above situations can be fully solved; the problem of unclear RS can also be addressed. There is basically only one weak point - the need for the customer/executing party to make some extra moves - place an order, transfer the payment, go through intermediate progress stages, etc. But I believe, this is a small price to pay for peace of mind and assurance that the money will not be wasted and the work will be done.

So you have finally decided to use the Jobs service and place an order for development or modification of an indicator, EA or a script. It is expected that you started off by getting yourself familiar with: [Rules of Using the Jobs Service.](https://www.mql5.com/en/job/rules) If you haven't done it yet, now it is time to fill this small knowledge gap.

It is further recommended to carefully read the articles ["How to Order an Expert Advisor and Obtain the Desired Result"](https://www.mql5.com/en/articles/235) and ["How to Order a Trading Robot in MQL5 and MQL4"](https://www.mql5.com/en/articles/117) \- all of these will help you get into the mood to do serious and efficient business, after all you are a serious person, future or well established trader. Because this profession requires a thoughtful and scrupulous approach.

### 2\. Common Types of Orders

Let us try to classify the most common types of orders that are very often associated with problematic situations. Without claiming to be comprehensive, the list comprises the following types of orders.

**2.1. Forex is very easy**

A customer who doesn't really know what he wants and has a very little understanding of how it all should work. As a rule, this would be a person new to Forex who picked up a smattering of information and thought that Forex is very easy. Spotting a favorable section of the chart with two crossing [МА's](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") and adding [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") to top it off (without even realizing that those are in fact the same two [МА's](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma")), he believes that he has found a good trading system that only needs to be coded in the form of an Expert Advisor to start raking in the dough.

Orders placed by such customers seem to be roughly as follows: "Simple Expert Advisor", "Simple indicator" and other variations including the words "simple" and "uncomplicated". Accordingly, the payment offered is 10-20 credits. The Jobs service simply does not allow to offer less.

When talking to such a customer, he would rarely admit that he is a beginner and that his order could well be nonsense which is not even properly priced. For some reasons, English speakers tend to admit it right away and even ask for a piece of advice regarding their system operation.

The only advice that can be given here is to manually test your system on a longer interval (at least on a 1-year time frame) before you start using the Jobs service. Alternatively, it can be tested in the tester's visual mode using existing Expert Advisors for opening/closing trades, e.g. [Trading SIMULATOR 2](https://www.mql5.com/en/code/9220) or in special programs for testing.

**2.2. In the wilds of indicators**

Another extreme so characteristic of beginners is the use of a whole lot of various indicators that sometimes produce mutually exclusive signals. It is also popular to use a combination of a number of indicators at different scales (sometimes being several orders different) in one indicator window and to watch for signals indicating crossings between them. And Forex beginners are not the only ones who are given to this habit.

All attempts to explain that it is not feasible are set back by a persistent lack of understanding and abstract references to a programmer who did it as easy as ABC, etc. That said, the less time spent using the terminal, the more persistent is the lack of understanding - there seems to exist an inverse proportionality.

Let us give an example of such a combination of indicators. Take [RSI](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") with a normalized range from 0 to 100 and [ATR](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/atr") with unnormalized range (from 0 to unknown value considerably smaller than unity). We will enter into BUY when ATR crosses RSI from the bottom up on the next bar. A gray rectangle in the image below (Fig.1) represents an emerging buy signal.

We will mark it with a vertical dotted line and see what happens next.

![Fig. 1. BUY signal emerged](https://c.mql5.com/2/3/cross-001__1.gif)

Fig. 1. BUY signal emerged

The next image (Fig. 2) suggests that the buy signal is still there and has even got stronger but for some reason it has shifted to the left (the position of the vertical dotted line on the time axis remains unchanged). It appears that we should have entered earlier. Something is wrong here.

![Fig. 2. BUY signal is still there](https://c.mql5.com/2/3/cross-002__1.gif)

Fig. 2. BUY signal is still there

The third image (Fig. 3) is completely confusing - the signal is gone, it has disappeared as if it never existed. These are the results of scaling (when several indicators with different value ranges need to be displayed in the same chart so that all of them are visible at the same time).

![Fig. 3 The signal disappeared](https://c.mql5.com/2/3/cross-003__1.gif)

Fig. 3. The signal disappeared

Another common situation is getting trapped in the illusion when looking at performance results of an indicator over the history. There is no clear understanding of the fact that readings of a great number of indicators change depending on the changing price and their final values are formed upon completion of a few bars in the chart when it might actually be too late to open a position. This is what "redrawing" is all about. The most graphic example of such an indicator would probably be Zig Zag.

Discussions with such a customer in an attempt to find a missing black cat in a dark room may take a very long time, lasting for as long as patience of the both parties permits, and can become a real psychological test.

There is a very simple tip - monitor indicators in dynamics. Check whether signals disappear when changing the chart scale along the time and/or price axis. Does the curve or histogram of an indicator look different on the previous bars when the price changes? You can easily check it in the tester's visual mode by dropping your set of indicators on the chart. Any Expert Advisor, e.g. provided in the terminal, can be used here.

**2.3. Manual system automation**

This is one of the most complex types. One can be an established and successful trader but gets puzzled when trying to put his system in the form of an algorithm or ends up describing every bar of the chart with the specification of price and indicator values. This happens because his trading is based on intuition, i.e. processing of trading information is hidden in the subconscious and only the end result is generated as output. Such a trader doesn't really know how the result was derived.

If the system formalization failed, the best option would be to gradually automate the trading process. Any system is made up of standard elements (opening/closing positions/orders, risk control, position management, etc.). Break your work down into stages and deal with every stage separately. Gradually, step by step, the trading process may get almost fully automated.

The second option is to immediately put away the idea of a fully automated trading system and focus on developing a semi-automated system. It can roughly be generalized as follows: the trader decides where and when positions should be opened while the remaining work is done by the Expert Advisor. Alternatively, you can set the time and/or price range where the entry is allowed, subject to certain conditions, and the Expert Advisor will then monitor these conditions to open positions within the ranges specified.

Thus, it would be optimal to apply two approaches. The first one is to start with developing a semi-automated system that would handle the majority of routine trading operations. And the second approach consists in breaking your work down into several stages. Each of them should represent a separate activity resulting in the final version of the Expert Advisor. A combination of the above two approaches is also possible.

**2.4. Getting the Expert Advisor to make tea**

Every now and then, there appear orders where the list of functions directly related to trading forms just a small part of the RS. Whereas the main part includes monitoring computer condition and network, sending SMS and e-mails, tracking various malfunctions, including the slightest of them, and changes in the environment around the Expert Advisor. Top it off with sound (or voice) notification of all the above available to the owner.

This all is, of course, feasible provided that both the customer and executing party are eager to go for it. However customers are as a rule quite reluctant when it comes to putting up money for implementation of these functions. And does the Expert Advisor really need to cover it all? I believe, it doesn't. Such an Expert Advisor will anyway not be able to operate absolutely independently - there is such a thing as fatal error (e.g. prolonged power failure).

Therefore, functions assigned to the Expert Advisor should be reasonably limited. Not only will it require less money for its development, but also ensure a more reliable operation of the Expert Advisor itself since fewer functions will result in less errors.

**2.5. Do that, don't know what**

The first part of this well-known phrase, "go there, don't know where", is usually used by the customer and executing party towards each other at the end of the work. Due to the fact that the trading system as such was not thought out well enough, the customer starts coming up with changes, additions to and removals from the RS in the course of development of an Expert Advisor (or an indicator).

In my opinion, this is quite acceptable if such a change does not require much effort from the programming point of view. A very small detail could be overlooked or simply forgotten. However it often concerns changes that do not seem very considerable to the customer but in fact require a modification of a substantial part of the Expert Advisor. Most importantly, such changes require alteration of the program logic. One of the common examples is a replacement whereby instead of opening market orders, the Expert Advisor should open pending orders or vice versa.

The only advice that can be given here is to seek full clarity on the RS for the executing party, as well as the customer (odd as it may sound). Modification of the RS can sometimes take more time than the work itself. And I believe, this is normal because identification of objectives is also a part of work.

### 3\. The Ideal RS, What Is It Like?

Speaking of the structure of the requirements specification for some abstract development, the ideal RS could be just about as set forth below:

**_3.1._** **_Operating environment_**

The terminal (MetaTrader 4/MetaTrader 5) for a required development. This most important thing is usually neglected in the RS and even in the subject of an order. This should be specified in the order subject due to the fact that not every developer has a fair knowledge of the MQL5 language or, on the contrary, may have got down to MQL5 without first learning MQL4.

**_3.2._ _Dealing Center/Broker where the development is going to be applied_**

This information is mainly necessary when developing EA's (or trading scripts). There may be various peculiarities: differences in execution of orders, minimum/maximum lot size, overnight positions, number of digits in quotes, account currency, quotes over the weekend and whether they need processing, fixed or floating spread, work based on new bars opening or based on ticks, etc.

It would be best if a demo account was opened by the customer with his DC/Broker in the expected account currency to be further provided to the developer. This is necessary because even in order to open a demo account with some DC's (e.g. Oanda), you are required to be registered on their website with certain forms to be filled up, etc. Save the developer's time. It may be needed in debugging and testing work.

**_3.3. Whether the development is intended for a real account or only for a testing/demo account_**

The difference is in fact crucial. Whereas operating speed is of utmost importance in testing (it affects the optimization time), error handling and recovery mechanism rebuilding the correct state of the Expert Advisor in different connection failures, reboots of the terminal, etc. are a priority when dealing with a real account. That said, the speed of testing the Expert Advisor for a real account may get several times if not several dozen times slower. Again, this would greatly affect the cost of development either up or down.

This is when you should ask yourself - do I really need an Expert Advisor for a real account? If the development is experimental and the algorithm has not been finally set and will likely be changed and revised, there is no point in ordering with a real account in view. Moreover, it is also adverse causing the optimization to run considerably slower. A follow-on development for a real account can further be ordered after the algorithm has been worked out. It will not be pricey.

**_3.4. Operation algorithm description_**

It is also advisable to provide a provisional list of external parameters (if any is expected to be used) specifying preferred names and default values. All this significantly speeds up the routine working stages for a developer and helps avoid asking unnecessary questions and seeking clarifications from the customer.

**_3.5. Type of information displayed_**

This should be specified since not everything the customer wants can be displayed. And it is better to sort it out when finalizing the RS rather than later. If something is expected to be displayed during the operation of an Expert Advisor or indicator, it is advisable to provide an approximate form and specify the chart background color expected for such operation (black, light, white or any other).

If illustrations are required for better explanation of the operation algorithm, they should be provided. An illustration supported by relevant comments can give a simpler and clearer idea of an algorithm than a few pages of plain text.

Properly prepared requirements specification is undoubtedly pivotal to the successful completion of the development and obtaining the expected result. So do not try to save time when detailing the points in your RS. And your efforts will be repaid in development.

**_3.6. Miscellaneous_**

Here you can usually find general terms and requirements. It may be a requirement to provide a source text (code) upon completion of the last working stage, extended comments in the source text, error correction warranty period and other requirements.

### 4\. A Few Words About Debugging And Testing

This is the stage where the functions actually performed by an indicator or Expert Advisor are checked against the functions listed in the RS. As you may know, one of Murphy's Laws of Computer Programming says: every program has at least one bug. There is also a corollary to this point that states: every bug detected and fixed is last but one. This is, of course, a joke but the above "law" is not far from the truth when it comes to complex programs (both Expert Advisors and indicators).

Hence another conclusion may be deduced: you should not suspect a developer of malicious intents and incompetence as soon as one or maybe more bugs are detected after delivery and acceptance of work. I believe, the quality of service provided by the executing party is primarily determined by prompt and efficient correction of errors detected in the course of work.

In my opinion, incremental development/testing is optimal when testing complex solutions that implement algorithms with multiple logical conditions and branching. The problem should be divided into several logically independent parts to be gradually solved/tested by the customer and executing party together. At the final stage, testing concerns the interaction between the parts tested earlier and work as a whole.

The notion of complexity/simplicity is certainly subjective and depends largely on experience of the executing party and understanding of all nuances of work by the customer. Testing methods and techniques provided by MetaTrader 4 and MetaTrader 5 are quite diverse, insomuch that some customers have no idea (or forget - I actually came across this in the past) of a possibility to run an Expert Advisor on historical data in the tester. But what is most important is that the tester is available in [visual testing mode](https://www.metatrader5.com/en/terminal/help/algotrading/visualization "https://www.metatrader5.com/en/terminal/help/algotrading/visualization").

It is one of the most powerful and convenient testing methods whereby an indicator or an Expert advisor can be tested on historical quotes received in the fast mode in just a few minutes by visually monitoring their performance results represented by executed trades and information displayed. The appearance of quotes can be slowed down or even temporarily suspended to give you time to analyze the details of a certain trade.

Following the testing in the tester, you can verify Expert Advisor's working efficiency by running it on a demo account. One and a half days of real time work on a demo account is usually enough to make completely sure of it. Provided, always, that very long time frames (a week or more) are not used.

### 5\. Describing Detected Errors

I would like to make a few points regarding the optimal way of describing errors detected by the customer. Speaking of extreme examples of how errors should not be described, it would be something like this: "Your Expert Advisor doesn't work; it should have opened a lot of positions but there was not even one."

Without having any information whatsoever on the instrument, time frame, parameters set, etc., the developer can only guess what was happening in the customer's terminal.

In my opinion, errors/deficiencies detected are best described using graphical images. MetaTrader 4 and MetaTrader 5 give an option to save the chart as an image file - as GIF and BMP in MT4 and as GIF, BMP and PNG in МТ5. PNG is a preferred format as it fully preserves sharpness of the image.

The image sharpness is very important to correctly assess the position of lines, bars and numerical values of the saved chart.

In the МТ4 terminal, it appears just about as shown below (Fig. 4). To do this, right-click on the required chart in the terminal and a pop-up menu will be displayed.

![Fig. 4. Saving the chart image in a file](https://c.mql5.com/2/3/Fig4_MT4-save-picture.jpg)

Fig. 4. Saving the chart image in a file

In МetaТrader 5, there is a similar procedure - the same command, _Save as a picture_ is selected in the menu which is however somewhat different in the commands it contains.

Once the menu command is selected, an additional dialog box will appear to set the resolution and area of the picture to be saved. This is shown below in Fig. 5:

![Fig. 5. Setting parameters of the picture to be saved](https://c.mql5.com/2/3/Fig5_MT4-save-picture2__1.jpg)

Fig. 5. Setting parameters of the picture to be saved

To save only the image of a window with a chart, select _Active chart (as is)_. The option _Active work area_ selected will result in saving the image of the whole terminal window which may sometimes prove useful when several charts need to be shown simultaneously.

It is not recommended to change resolution of the active chart since it may impair the quality of the saved picture.

The image can also be saved by using the Alt+PrtScr shortcut key or PrtScr alone that allows to save a capture of the current window or the entire screenshot in the clipboard to be further dealt with in a graphics editor.

You can then open the resulting graphical file in a graphics editor and edit it (if necessary) - e.g. add a text note regarding a probable error in the relevant places of the chart. The easiest way to edit is to use Paint graphics editor bundled with Windows. It is however a matter of taste and habit.

### 6\. Hidden Traps in RS

As a rule, they crop up for various reasons: lack of understanding of all the nuances of pricing in stock exchanges and Forex, lack of consideration of peculiarities of scaling along the time axis and price axis, etc. Whether the developer is going to pay attention to these points when discussing RS or choose a formal approach depends largely on his qualifications and knowledge of peculiarities of the market trade. And, of course, on honesty.

Using figurative expressions of folk art, when writing an RS the customer often considers the behavior of "a spherical cow in a vacuum" rather than a real foreign exchange or financial market thus extremely idealizing the price series behavior.

**_6.1. Discreteness of price changes_**

Let us have a look at the following practical example. A sentence in the requirements specification is substantially to the following effect: _when the price reaches the value set by the external parameter, open a buy position_. One would think, no question would ever be raised on that - there is a price, the value is set, just write a trigger condition and that's it.

The only "trifle" overlooked here is that the price moves in the chart discretely (from one point to another) and not continuously. Enabling of the terminal option _Show chart as a broken line_ so that we can see it on the screen will not make the price movement continuous. The actual price series in the chart will still consist of discrete points.

It can be illustrated by the following tick chart example (in a rough form).

![Fig. 6. Discrete price change](https://c.mql5.com/2/3/PriceDiscr__2.png)

Fig. 6. Discrete price change

The specified price value is shown as a horizontal solid line while red and blue dots represent the actual price values observed. Having a close look at the chart, you can see that technically **_the price_** **_never_** **_hit the level as specified_**. That is, following the logic of the original RS, the condition has never been met.

An important note - all price comparisons shall be kept accurate to one point. For example, if the instrument quotes have five digits after the decimal point, then one point is equal to 0.00001 - this will be the accuracy for price comparisons to be made. You should also keep in mind that there are instruments (e.g. mini futures contracts) where the discreteness of price change is more than one point.

So, even though we can see that the price seems to have reached the specified level, the condition, as set out in the provided RS, has never been met. In order to bring the idealistic notion of the price introduced in the RS in line with the reality, you should do as follows: add two limits to the existing level, upper and lower. They are shown as a dashed line in the above Figure 6. The range of these limits depends on a certain trading system. In the majority of cases, 2-3 units of average spread should suffice.

Then all the points making it into the area between the dashed lines will be considered as having reached the specified value. In the above Figure they are encircled in green.

Let us get back to our RS and amend it as follows: _when the price reaches the value set by the external parameter plus/minus delta, open a buy position._ It is much better now.

There is another point to consider which is related to the fact that the connection with the broker's server may get lost at any moment. Until the connection is back, the price may go any distance up or down from the specified level. This area (of lost connection to the broker's server) is outlined in pink rectangle in the above Figure. Obviously, a situation of this kind should also be considered in the RS.

Let us amend our RS once again: _when the previous price value is lower than the one set by the external parameter minus delta, and the current value is within the range of the_ _value set by the external parameter plus/minus delta, provided that the time interval between them is not greater than the one specified, open a buy position._ Now everything is correct.

**_6.2. Triggering at an exact specified time_**

A RS says: _to place two pending buy and sell orders at a given distance from the price at the time set by the external parameter_. Here, you need to distinguish between two types of terminals - МТ4 and МТ5. Whereas MetaTrader 5 allows to use the timer with minimum discreteness of one second and the same time countdown accuracy, the MQL4 language of MetaTrader 4 does not provide for a timer.

Therefore, for a standard time countdown in МТ4 we usually use the time of incoming price ticks of the instrument on which the Expert Advisor or indicator is used. Since ticks do not appear regularly and the time intervals between successive ticks may be quite large making a few dozen seconds, sometimes even minutes in an inactive market, it turns out that we will not be able to guarantee an absolute accuracy in triggering orders at a specified time. Another standard method of time countdown in МТ4 is to use a looped Expert Advisor which is quite inconvenient for development and has a number of downsides.

So this part of the RS should better be amended as follows: _to place two pending buy and sell orders at a given distance from the price_ _at the time_ _set by the external parameter_ _with the specified error_.

**_6.3. Use of general, multidimensional terms_**

Here is a typical example: _when the market is flat, do so-and-so. When the trend is clearly in place, do so-and-so_. In so doing, the customer believes that no further detail is required in such a simple straightforward task as programming a trend/flat. I guess no comments are needed here.

There is another example: _to place a pending order on the extremum_. The customer's interpretation of the extremum is of course missing in the RS. Indeed, why give an explanation if it can be derived from the chart. The fact that the extremum can be obtained using a number of different algorithms every one of which has its advantages and disadvantages may come as a surprise to the customer.

**_6.4. Use of common words with general meaning_**

This refers to the terms that seem to be clear when used in everyday life while being absolutely not suitable for a requirements specification. Here are some examples. A RS says: _if the indicator line has formed two bulges following one after the other, the second one being lower than the first one..._ and so on, thus giving a description of the divergence used.

A whole lot of questions arises after reading the above - what can be considered a bulge; maximum/minimum distance at which such "bulges" can be situated from each other; how much lower the second bulge should be from the first one, etc.

The words like a lot/a little, small/large and so forth are also very frequently used. You should understand that programming is based on mathematics which requires precise quantitative definitions. There is no other way.

### 7\. When to Resort to Arbitration?

Obviously, you should file for arbitration before the work is completed, i.e. before being through the last stage - Delivery of work/Payment. Otherwise there will be no financial factor to bind the executing party and the customer. This condition is essential.

It appears that the only reason to resort to arbitration can be the occurrence of a problem that cannot be resolved between the customer and executing party in private. For example, the customer is convinced of the incorrect operation (in disagreement with the RS) of the work delivered to him, while the executing party is trying to prove the contrary.

Another possible option is "disappearance" of the customer or executing party for a long time during which they do not reply to any private messages or mails, etc. Meanwhile, the customer's money is blocked in the account and the work is not done. Or, on the contrary, the work has long been done but there is no way to get to the payment. There can be different circumstances involved but the situation should and can, nevertheless, be resolved through arbitration.

The situation where the parties are in a hurry to finalize the RS and start working is very common. It later appears that the RS in the form it was provided cannot be used and requires further details and clarification of some points. Consequently, the order value may need to be revised. The work can be rolled back from this stage only through arbitration. You should therefore better not rush to start work and rather take your time to go through the RS once again.

### Conclusion

Hopefully, this article has been helpful to users new to the [Jobs](https://www.mql5.com/en/job) service, mainly to first-time customers. It is likely that this article has not covered all issues but a journey of a thousand miles begins with a single step. We should be easier on each other and not vent so many negative emotions in the world around us - they are not going to make anybody's life easier - be it the life of a customer or executing party.

### A list of favorite questions asked by customers

**_1\. Why are some developers reluctant to talk on Skype, ICQ, phone, e-mail, etc?_**

I believe there are two main reasons for this:

First, one developer may have quite a few customers and if he starts to talk on-line with all of them (good, if he does not have to do it simultaneously), he will have almost no time for programming itself. Thinking that one can explain things faster using Skype or ICQ is one of the common myths.

Second, all of these options lack one important feature - the ability to document what has been said. The point is that when it comes to disputes and arbitration, written communication related to the relevant order can prove fault or innocence of either party.

**_2\. Why does the starting of a compiled EA/indicator (.ex4 or .ex5 file) cause the terminal to shut down?_**

It is very likely that the version of the terminal you have installed is old compared to the one used in the process of compiling. You should update the terminal to the [latest available version](https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe").

**_3\. What are these errors that occur when compiling the EA/indicator file - Function 'xxxxxx' is not referenced and will be removed from exp-file?_**

This is not an error. The message says that the function 'xxxxxx' is not used ('xxxxxx' is replaced by a specific function name) and will not be present in the compiled file. You can disregard this message as the presence of this "extra" function does not affect the operation of an EA or indicator in any way.

**_4\. What is this error that occurs when compiling the EA/indicator file - can't open "хххххх" include file?_**

The message says that the file "хххххх" was either not copied when transferring files to your computer or was copied into the wrong directory. Usually, .mqh files should be placed in the include directory of the terminal.

**_5\. Why are there no marks to show where a position was closed at Take Profit/Stop Loss level, during the operation of the Expert Advisor?_**

This feature is only available when testing an Expert Advisor in the Strategy Tester. It is not available when working with a real or demo account. A transaction of interest can be dragged from the account history to the chart, keeping the left mouse button pressed. The marks indicating opening and closing positions will then appear in the chart, being connected by a dashed line as they are during testing in the Strategy Tester.

**_6\. A position has not been opened (an order has not been placed) even though the historical chart suggests that all conditions for entry are present._**

What you have faced is most likely a result of redrawing of an indicator whose readings are used in the Expert Advisor. The conditions (values of the indicator) present at the moment of the assumed entry did not match the signal for opening a position. Then, after a few bars the values of indicators changed and took their final form which is what we can see in the history. This can easily be checked using the tester's visual mode.

**_7\. How to cancel work at the Prototype/Pilot Model - Demonstration stage?_**

It can only be canceled through arbitration.

### Recommended links

1. [Rules of Using the Jobs Service](https://www.mql5.com/en/job/rules)
2. [How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235)
3. [How to Order a Trading Robot in MQL5 and MQL4](https://www.mql5.com/en/articles/117)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/361](https://www.mql5.com/ru/articles/361)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Creating Custom Criteria of Optimization of Expert Advisors](https://www.mql5.com/en/articles/286)
- [The Indicators of the Micro, Middle and Main Trends](https://www.mql5.com/en/articles/219)
- [Drawing Channels - Inside and Outside View](https://www.mql5.com/en/articles/200)
- [Create your own Market Watch using the Standard Library Classes](https://www.mql5.com/en/articles/179)
- [Controlling the Slope of Balance Curve During Work of an Expert Advisor](https://www.mql5.com/en/articles/145)
- [Several Ways of Finding a Trend in MQL5](https://www.mql5.com/en/articles/136)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6769)**
(33)


![Alexander Puzikov](https://c.mql5.com/avatar/avatar_na2.png)

**[Alexander Puzikov](https://www.mql5.com/en/users/im_hungry)**
\|
5 Jun 2012 at 10:42

**FAQ:**

Delusional is an understatement.

Well, life happens. :-)

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
12 Jun 2012 at 07:47

Here's another [Sad story](https://www.mql5.com/ru/forum/6920)

![Andrey F. Zelinsky](https://c.mql5.com/avatar/2016/3/56DBD57F-0B0E.jpg)

**[Andrey F. Zelinsky](https://www.mql5.com/en/users/abolk)**
\|
12 Jun 2012 at 11:37

**Rosh:**

Here's another Sad story

I guess this link is [https://www.mql5.com/ru/forum/6920](https://www.mql5.com/ru/forum/6920)

![880012856](https://c.mql5.com/avatar/avatar_na2.png)

**[880012856](https://www.mql5.com/en/users/880012856)**
\|
27 Aug 2022 at 20:13

**Jose Martin [#](https://www.mql5.com/en/forum/6769#comment_202940):**

Hi, really good article...Thanks.

Hi

How I can get EA in mt4 or mt5

![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
27 Aug 2022 at 20:19

**880012856 [#](https://www.mql5.com/en/forum/6769#comment_41672288):**

Hi

How I can get EA in mt4 or mt5

Search the [Market](https://www.mql5.com/en/market).

![MetaTrader 5 - More Than You Can Imagine!](https://c.mql5.com/2/0/diamond_MetaTrader_5.png)[MetaTrader 5 - More Than You Can Imagine!](https://www.mql5.com/en/articles/384)

The MetaTrader 5 client terminal has been developed from scratch and far surpasses its predecessor, of course. The new trading platform provides unlimited opportunities for trading in any financial market. Moreover, its functionality keeps expanding to offer even more useful features and convenience. So, it is now quite difficult to list all the numerous advantages of MetaTrader 5. We have tried to briefly describe them in one article, and we got surprised with the result' the article is far from brief!

![Trader's Kit: Drag Trade Library](https://c.mql5.com/2/17/902_26.png)[Trader's Kit: Drag Trade Library](https://www.mql5.com/en/articles/1354)

The article describes Drag Trade Library that provides functionality for visual trading. The library can easily be integrated into virtually any Expert Advisor. Your Expert Advisor can be transformed from an automat into an automated trading and information system almost effortless on your side by just adding a few lines of code.

![OpenCL: The Bridge to Parallel Worlds](https://c.mql5.com/2/0/OpenCL_Logo.png)[OpenCL: The Bridge to Parallel Worlds](https://www.mql5.com/en/articles/405)

In late January 2012, the software development company that stands behind the development of MetaTrader 5 announced native support for OpenCL in MQL5. Using an illustrative example, the article sets forth the programming basics in OpenCL in the MQL5 environment and provides a few examples of the naive optimization of the program for the increase of operating speed.

![Synthetic Bars - A New Dimension to Displaying Graphical Information on Prices](https://c.mql5.com/2/17/995_16.png)[Synthetic Bars - A New Dimension to Displaying Graphical Information on Prices](https://www.mql5.com/en/articles/1353)

The main drawback of traditional methods for displaying price information using bars and Japanese candlesticks is that they are bound to the time period. It was perhaps optimal at the time when these methods were created but today when the market movements are sometimes too rapid, prices displayed in a chart in this way do not contribute to a prompt response to the new movement. The proposed price chart display method does not have this drawback and provides a quite familiar layout.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/361&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083015523681178450)

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