---
title: Brute force approach to patterns search (Part V): Fresh angle
url: https://www.mql5.com/en/articles/12446
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:33:38.752923
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/12446&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071926334734676224)

MetaTrader 5 / Trading


### Contents

- [Introduction](https://www.mql5.com/en/articles/12446#para1)
- [Ways to complete the objective](https://www.mql5.com/en/articles/12446#para2)
- [Obtaining a stable income based on automated trading systems](https://www.mql5.com/en/articles/12446#para3)
- [EAs and patterns](https://www.mql5.com/en/articles/12446#para4)
- [Maximum automation of the work chain](https://www.mql5.com/en/articles/12446#para5)
- [Universal receiver EAs](https://www.mql5.com/en/articles/12446#para6)
- [EA for dynamic collection of quotes](https://www.mql5.com/en/articles/12446#para7)
- [Conclusion](https://www.mql5.com/en/articles/12446#para8)
- [List of references](https://www.mql5.com/en/articles/12446#para9)

### Introduction

It has been quite some time since I published the last article on the topic. Since then I have had to rethink a lot of what I was doing before. This made it possible to look at the problem of profitable algorithmic trading from a completely different angle, while taking into account all the little things that had not been possible to consider before. Instead of standard and colorless math and code, I offer my readers to approach the problem in a completely different way. This article can be both the beginning of something new and a reboot of the old. I am tired of being clever and throwing unnecessary equations and code into the dustbin of history, so this article will be as simple and understandable as possible for any reader.

### Ways to complete the objective

I started thinking about the variety of paths that lead people to success or drive people into dead ends while they try to make money using algorithmic trading. In theory, it turns out that there are several paths:

1. Head-on approach.
2. Beautiful picture.
3. Ready-made trading systems.
4. Modernization and hybridization of publicly available algorithms.
5. Team approach.

The first approach is the most common among stubborn people. In fact, it is useful for people like me in terms of letting go of ambitions and false hopes. It does not sound like much, but it is actually very beneficial for your future. This approach takes a lot of effort and time, and if you do not stop at some point, you may become a Ph.D. in forum sciences with all the ensuing consequences. I think, no clarifications are needed here. Everyone understands my sarcasm perfectly well. Nevertheless, this approach allows you to learn the theoretical information that I learned in my time, and its value is absolute. The main thing is to stop in time. Of course, if we weigh the time spent and the results obtained, the outcome will be far from perfect.

The second approach is much simpler and, in fact, much more efficient in terms of time spent, because there is much less effort involved and all you have to do is convince people that you are successful. Everything is in our heads, and at some point I realized that this works great. People tend to trust beautiful wrappers. There is no place for morality or anything else here. The result is all that matters. This may seem cynical, but the whole world lives like this. All you need is to create a certain image. You may use martingale, averaging or other trading techniques. They are quite sufficient to create such an image.

I believe, the third approach is wise because in this case the effort spent will be minimal, but your picture is real. With proper implementation of this approach, there will be no negative aspects, although there will be some disadvantages. The most important thing needed to implement this approach is knowledge. If I had not had the experience that I have now, then I would not have been able to take advantage of it even with the right attitude and a rational and balanced approach to achieving my goals.

As for the fourth approach, I do not know if anyone practices it. In theory, it should take less time, but I cannot say anything about its efficiency. In general, everything is possible, but I do not think this approach is the most effective to put it mildly. Rather, it is better to use it in combination with the previous one, as this will increase the variability of your trading and increase the chances of receiving more consistent trading signals.

The fifth approach can only be effective if you have a lot of ideas and constantly work on them, however, it is much better than the first one, even if each team member uses a head-on approach. But it just so happens that most developers of trading systems are narcissistic loners, and only a few can assemble such a team and, most importantly, arrange its work. I know there are such teams, and they are quite successful. It is good if you find yourself in such a team working together to develop profitable algorithms. The advantage is that in the end, the total quality and quantity of developments can play a decisive role and make it possible to create a competitive product.

Of course, these are rather idealized scenarios, and everyone's path is a little unique, but despite this, I can definitely say that no matter what path you choose, the result will always be preceded by the acquisition of some kind of knowledge. It is not necessarily just technical or philosophical, but in my case it is both. It seems to me that this is exactly how it should be, because a problem must always be looked at from all sides so that it can be solved.

### Obtaining a stable income based on automated trading systems

Before finding a harmonious approach to automated trading, we first need to structure the entire process from beginning to end - from the moment the idea is conceived to its implementation:

1. Idea.
2. Arranging the implementation plan.
3. Development.
4. Fixing errors.
5. Improvements and modernization.
6. Extensive tests.
7. Optimization and determining applicability limits
8. Preparation for trading (resources, demo account)
9. Trading on a real account

If you are a novice, then you will be almost one hundred percent sure that your system should work either because you read it somewhere, or made it up yourself and convinced yourself that it will work. The reality is that you do not have a mathematical model of the market, and it is so complex that, even assuming you have one, you will not be able to use it due to its incredible complexity and the irrationality of using it in an EA. So what can we do? The answer is not as simple as it seems. This is exactly why I came up with my own brute force algorithm.

It is obvious that if you take on the task of building a super EA, then you will go through a lot of such stages before you arrive at the desired result, which is actually extremely doubtful, to put it mildly. I know that from my own experience. The most annoying thing after yet another unsuccessful attempt to develop an EA is the fact that it will have to be thrown away, which means that despite all the usefulness of the experience gained, this does not reduce the disappointment from the time spent. When you develop EAs yourself, this is inevitable. If we are talking about a Freelance order, then everything is even sadder, since you will get your EA, which will most likely be of no use.

In this regard, I want to make it clear that this is primarily about your time. Successful people have the ability to correctly assess the value of time. If the time spent does not bring the desired result, then it is hardly worth carrying on. Here is the diagram of the standard approach:

**Diagram 1.**

![standard approach](https://c.mql5.com/2/56/niy6sdqs4_1.jpg)

Each action in the diagram takes its own time, and the overall result directly depends on what knowledge and resources you have. Resources will not necessarily mean your available funds for investments, but rather the availability of computers for constant testing of trading systems or funds to purchase the necessary equipment. Your desire to pursue your goals and the availability of free time are important as well. The thing is that finding or creating a good trading system is only half the battle, the second half concerns the aspect of how to properly manage it considering that you simply have little free time.

If you have at least the slightest understanding of the issue, then you can see how the diagram will change if we use ready-made trading systems from the MQL5 market or other sources. There is no need to redraw it all, but only indicate the appropriate replacements:

**Diagram 2.**

![replacing development with search](https://c.mql5.com/2/56/yfqgi31ys_2.jpg)

The meaning of the diagram does not change, but searching for and choosing something ready-made is much easier and, I would say, much more pleasant than writing tons of code. Fortunately, I can do both. Of course, this requires knowledge and experience. Among other things, the idea behind this diagram is that EAs may lose their relevance over time, and the majority will certainly end up scrapped. After you throw out another EA, you do not suspect that it can be used again after some time, do you? Digging through the pile of rubbish in search of long-forgotten algorithms while thinking about how to apply them will also take a lot of time.

Nevertheless, it would be good to accumulate a certain database of EAs and continue to trade successfully, while changing them wisely. In this case, our process is simplified even more, because there is no need to look for new EAs. Is it possible? Yes, it is. Ideally, this collection of EAs should have the following qualities:

- Algorithm flexibility.
- Possibility of signal inversion.
- Performance (minimal resource consumption).
- Order magic numbers.

Based on this data, it is even possible to enter a mathematical definition of the prospects of a selected collection of EAs. We can even try to find such expressions to make it clearer how and what the characteristics of these EAs and their number influence. Alternatively, we can simply make a simple and understandable list:

1. The more EAs, the better our collection (simply because the more EAs, the more of them will meet the required trading criteria in the selected trading area).
2. The more inputs an EA has, the more effectively it can be optimized.
3. EAs based on bars are better (they are easier to use and test, as well as optimize, and we do not have to worry about ping, slippage, and other issues).
4. If it is possible to invert the trading signal, then the weight of the EA doubles.

I will not dwell on over-optimization and fitting to history here since this is a separate issue. I assume that you know how to do all this correctly. If everything is done correctly, then our diagram transforms into a very simple design:

**Diagram 3.**

![trading based on the EA database](https://c.mql5.com/2/56/31lzaogr7_3.jpg)

Obviously, the more EAs you have, the better you can sort the robots. But here we face several unpleasant moments. The better selection quality we want to achieve, the more time it will take us to make this selection. In addition, we will have to make a selection many times. You need to do this regularly. So, it will turn into just another job, unless someone will do everything for you. In this regard, the question arises: "Why do I need it when there are already tested examples of regular business with absolutely no risks?"

In addition, the more successful you want to be in your trading activities, the more simultaneously operating terminals you need. This means that you should constantly monitor each terminal, add and remove new EAs from it, as well as configure and monitor their work. As you understand, this is all a wagonload of work. Despite the fact that we have freed ourselves from the need to constantly develop new EAs, we still have not got rid of the main routine. Let's list the main labor-intensive points:

- Selecting EAs using optimization.
- Preliminary forward testing on a demo account.
- Selecting the most durable trading signals.
- Real trading using the most durable combinations.
- Constant control (shutdown, pause, replacing robots, etc.) (working with terminals).

All this is possible, but only if you have the optimal workflow paradigm. But of course there is a limit. Given my experience, I think, that you work alone. It is impossible to jump above your head, because everything takes time.

Initially, I developed my brute force algorithm for research purposes, in order to understand whether it is possible to achieve profitable trading using simple algorithms. I realized that this can be done. Given the capabilities it had at the time, it was only able to provide additional EAs in order to expand their total number. To better understand how simple EAs can help and how to use them correctly, we need to understand a little deeper how a particular algorithm is capable of solving the profitable trading problem and how to properly treat certain EAs.

### EAs and patterns

It is not enough to have a collection of algorithms and constantly optimize them. Optimization is a separate skill and its mastery determines the result of using the EA on a trading account. Each EA is unique in its own way and has its own nuances both in optimization and in use. An important option that I think should be included in any EA is the ability to invert trade. This means that any trading action is replaced by the opposite, that is, buying is replaced by selling, and selling by buying. Initially, this option seems completely unnecessary because there is a false belief that everything should work as intended.

To understand this fact, we should first understand what a pattern is. In popular understanding, a pattern is the difference between some statistical characteristics and a random distribution. When considering this issue superficially, one might think about the inertia of patterns. But this is only one of the possible future scenarios for this pattern. Only a small part of the patterns have inertia. Let's consider the found pattern within a backtest or a trading signal.

Suppose that we have a very large database of robots, from which we can create separate groups according to some characteristics that we consider appropriate for one reason or another. The characteristics are not as important as grouping itself:

**Diagram 4.**

![grouping robots](https://c.mql5.com/2/56/6ilo2mk6y_7_jhivcuv21e8_3wz0ub77w.jpg)

Here, my brute force program appears for the first time as an element of the flowchart. In fact, the program carries out this grouping thanks to the different settings of each of its copies. In essence, each copy of the program, configured differently, is a completely independent group of robots selection can be made from. The generated robots can be used for trading, which is shown at the very bottom of the flowchart. The most important thing here is that all these groups of robots subsequently, after a while, are divided into three groups:

- Profitable ones with a direct signal (based on the pattern inertia).
- Profitable with an inverted signal (based on an immediate pattern change).
- Chaotic (simple fitting to history).

It is impossible to know in advance which set will provide a certain group of signals, but after some time filtering by exclusion is possible. In this regard, the more such independent groups and signals, the better. It is better to make at least two signals for each group of EAs:

1. Direct signal.
2. Inverted signal.

Additionally, we can add mixed signals from different groups. All this will maximize the chances of effectively finding harmoniously composed groups of EAs who can work for a long time. Two facts can lead to finding good portfolios as quickly as possible:

1. The highest quality and effective grouping of available EAs (as many independent groups as possible).
2. Direct and inverted signal for each group + mixed ones.

All these factors ultimately provide the most numerous and varied signals eventually providing us with the best possible sample for subsequent use in real trading. The most important factors influencing the ability to make such groupings are the size and diversity of the EA collection. In case of my program, the thoughtfulness of its settings and the variety of internal algorithms, such as analysis methods, clustering and others, are paramount. One of the universal methods of increasing variability is clustering:

**Diagram 5.**

![work limitations](https://c.mql5.com/2/56/uzqi4hfuq_6.jpg)

In this case, clustering is represented by the possibility of dividing subgroups of robots by day of the week and time windows within a day, which, in itself, already provides the broadest opportunities for grouping EAs into portfolios. Of course, you can come up with a lot of clustering options, but I believe, this is the simplest and most effective. In addition, it can be used to configure the program itself to work on certain days and hours. This allows for the highest possible optimization of the computing resources consumption and setting the correct weights for each copy of the program. Each setting has its own calculation complexity, so this technique is necessary.

Additionally, I would like to say a few words about complex and simple EAs. In theory, it is possible to create an EA that will be as flexible and variable as possible in relation to almost any pattern, but I think it is obvious to everyone that the more complex the system, the easier it is to break it. It is also possible, of course, that the system will turn out to be super-successful and fault-tolerant, but let's look at things realistically. Let's say that a few have such a system (although, I think this is a utopia), and they will definitely never share it with others. But what should other people have to do?

Any successful algorithmic trader should have an understanding of the simple truths that I present, and to a greater extent, the profit of such people is based on this understanding. Profitable trading is, first of all, the ability to correctly use the tools at your disposal. Waiting for magical strategies is not the best solution. Any idea can be implemented with the right set of accompanying solutions. This is what I am trying to show in this article without focusing on specific algorithms.

### Maximum automation of the work chain

The idea developed gradually from a simple research to the full automation of finding stable trading signals. It came exactly to what it should have come to. At the moment, my system performs a whole range of tasks, ranging from simple generation of trading EAs to trading in MetaTrader 4 and 5 terminals. This is roughly what the current version of what I am doing now with my program looks like:

**Diagram 6.**

![current structure of using brute force](https://c.mql5.com/2/56/sjavokyo9_4_o7zey1n1z.jpg)

This structure completely relieves me of routine operations, such as:

- Selecting and grouping EAs.
- Enabling/disabling EAs on charts and their subsequent configuration.
- Optimizing and selecting settings.
- Search for new EAs.
- Creation new EAs.

One of the tricks of this structure is the fact that in addition to generating simple EAs that end up in my Telegram channel, at the same time there is a universal EA inside the terminals. There is no need to constantly remove it from the charts and install it every time the program finds a new working EA. Instead, it creates the EA itself and a separate text file that is the equivalent of the EA. The file ends up in the terminal shared folder, in which there is a corresponding directory the universal EA reads the settings from. This all happens automatically on the fly and does not require my control, and the EA itself ends up in the Telegram channel.

So, the whole system automates both creating EAs and auto-posting them to my channel. All I have to do is scale the system, purchase equipment and monitor trading signals. Now, of course, all this works at 1% of its capabilities, but I consider it quite suitable as a demonstration option.

At the moment, I have only one computer at my disposal. It is quite old, but it is enough to ensure the alternate operation of two independent workers (brute force programs). Based on these two settings, I created two signals: direct and inverted for each, and additionally direct and inverted mixed signals. Based on the results of testing for two months, you can see what I said above:

**Figure 1.**

![direct profitable and inverted profitable](https://c.mql5.com/2/56/yi6f0no.png)

According to the testing results, only two clearly positive signals remained over two months of continuous trading at minimum capacity. There is another one there (mixed) but it is almost identical to the inverted one shown here. They refer to two completely different trading algorithms and their settings. Here we can see that positive trading can be obtained both on a direct signal and on an inverted one. You can find the EAs generated during the entire testing process in my Telegram channel. Find the link in my profile, as well as at the end of the article.

### Universal receiver EAs

Of course, more thoughtful and high-quality EAs are much more effective. They can be used for automated trading. However, most EAs can be used repeatedly. Any algorithm has limits of applicability, and many EAs that seemingly did not meet the expectations of a developer or a customer have their own unused resource. If we estimate the approximate ratio of how many trading systems never reach the testing stage (at least on a demo account), we will see that there are simply tons of them.

The truth is that if you do not know how to properly optimize an EA, it will most certainly be discarded. I threw out quite a few interesting EAs because of this. I just did not realize that they could be used a little differently. Unfortunately, this requires some experience. I will not touch on this topic here but I will write a separate article on advanced optimization a bit later.

Deciding on such an adventure is not easy, because on an instinctive level you always want to take one super-EA, put it in the terminal, press one button and forget about it for at least a few weeks. But we still need to find it and confirm that it really has the characteristics that we require. But think about this: while you are looking for it, you can just take everything you have and run as many different configurations as possible.

Of course, you will need to spend a lot of time and effort in order to competently control the trading process of such EAs. In my system, I bypassed this problem using a universal EA, which is a convenient optional add-on to the generated advisor (setting). The first and simplest version of such an EA contains the following important control variables:

```
input int DaysToFuture=50;//Days to future
input LOT_VARIATION LotMethod=SIMPLE_LOT;//Lot Style
input bool bInitLotControl=false;//Auto lot bruteforce
input double MinLotE=0.01;//Min Lot
input double LotDE=0.01;//Lot (without variation)
input double MaxLotE=1.0;//Max Lot
input bool CommonE=true;//Common Folder
input string SubfolderE="T_TEYLOR_DIRECT";
input int MinutesAwaitE=2;//Minutes To Check New Settings
input bool bBruteforceInvertTrade=false;//Invert Bruteforce Trade
```

Of course, these are not all the variables that are there, but these are the variables that are necessary to provide automation of the following important actions:

- Disabling trading after the specified allowable time for trading has expired (if a new one has not been generated during the current setting, therefore the old one loses its relevance).
- Trading style (simple lot / gradual increase in lot from minimum to maximum within a given trading window / gradual decrease in lot from maximum to minimum within a given trading window).
- Reading settings from the current terminal folder / terminals' shared folder.
- Ability to select a subdirectory.
- Interval for updating settings.
- Inverting the signal.

All this allows us to flexibly configure the interaction between terminals and the brute force program, and also launches an arbitrarily large number of terminals and brute force machines simultaneously on one machine, as far as its capabilities allow. The only thing is that now I have to assign a separate EA to each chart, because I have not yet made a universal multi-receiver. It will be added later. We will need it to implement better and more thoughtful trading. It will be very convenient to see all the pros and cons in the diagram:

**Diagram 8.**

![advantages and disadvantages](https://c.mql5.com/2/56/8mecuwfpr_8.jpg)

As can be seen from the diagram, two systems are recommended for implementation in both the simple and advanced receiver EA:

- Parallel trading synchronization system.
- Parallel trading optimization system.

These are very important add-ons to all algorithms, which can both improve the quality of parallel trading and reduce trading costs. I am planning to put them into operation a little later, but for now there are no necessary resources for this.

When we are talking about trading with only one EA, then all these things are redundant, but when we are talking about the parallel operation of many EAs, the need for such an add-on inevitably arises. The advantage of my approach is that I can implement these add-ons more efficiently based on the uniformity of all EAs. This applies to both external and internal systems.

I would also like to say that the multi-receiver is designed to work on one chart without the need to attach it to each instrument. The only disadvantage of such an EA is that it is more difficult to customize for each specific instrument, but nevertheless, it has many more advantages than disadvantages. Perhaps, in one of the following articles I will dwell in more detail on these systems, while describing the technical details of the innovations that I was able to introduce during my pause in writing articles.

### EA for dynamic collection of quotes

Previously, I had a simple EA that created text files that had to be manually opened in the program. Of course, this data loses its relevance after some time. To ensure the operation of all the structures described above we need access to fresh quotes. To do this, all I had to do was make an EA that constantly writes data to the shared terminal folder. In order to compile sets of tradable instruments and periods, I decided to add the corresponding lists to the program settings:

**Diagram 9.**

![updating fresh quotes](https://c.mql5.com/2/56/rt18bt6we_9.jpg)

Using this technique, we can configure each browser to have its own unique set of period tools without the need to duplicate data in several folders. In this case, only one terminal is needed to operate any number of trading terminals. If we set analysis intervals to years, then possible pauses and delays in updating data are insignificant. In addition, the entire system is based on the bar-by-bar paradigm, so the whole thing is as reliable as possible and resistant to almost any emergency situations. This bot has only a few settings:

```
input bool CommonE=true;//Common Folder/Terminal Folder
input double YearsE=10.0;//Years Of History Data
input int MinutesForNew=2;//Rewrite Timeout In Minutes
```

The EA writes either to the common folder of all terminals, or to the current terminal’s own folder. It writes the last years of history, which we indicate starting from the current terminal time and going back in history. The EA writes after a specified timeout specified in minutes. This was the last element of all logic. The hardest part is over. All that remains is to implement a receiver EA that works on the same chart. I have already prepared half of the functionality for its implementation.

### Conclusion

In this article, I got away from the usual technical part and tried to look at the problem of profitable trading from a slightly different angle using my own experience. As you can see, in addition to the EAs themselves (which are only a small part of making a profit), there are a lot of subtleties and nuances that can either help you achieve profit or hinder your efforts. Quite interestingly, the result is far from what was intended at the beginning.

Along the way, I had to gradually adapt the original idea to reality, which was a completely uncontrollable process, but rather a spontaneous and inevitable one. As a result, I found the right way out of the theoretical impasse, which inevitably can only be found in a practical way. That means more robots, more signals, more computers and more automation. I would like this article to serve as some kind of impetus for many theorists and Grail seekers revealing that it is always possible to find an alternative way out. In the next article, I will show in detail what improvements my system has acquired. However, there is still a lot of work to be done.

All currently available signals can be found [here](https://www.mql5.com/en/signals/author/w.hudson). I have shown only the most important ones in the article. There will be more of them, and their quality will increase as I get additional power and improvements. We will see if I can end up with a group of stable and versatile signals. As for automatically generated EAs, you can find them in my [public Telegram channel](https://www.mql5.com/go?link=https://t.me/Centropolis_Lights "https://t.me/Centropolis_Lights"). For now, their quality matches my minimum available capacity, but later, as capacity expands, you will see an increase in both variety and quality.

### Links

- [Brute force approach to pattern search (Part IV): Minimal functionality](https://www.mql5.com/en/articles/8845)
- [Brute force approach to patterns search (Part III): New horizons](https://www.mql5.com/en/articles/8661)

- [Brute force approach to patterns search (Part II): Immersion](https://www.mql5.com/en/articles/8660)

- [Brute force approach to pattern search](https://www.mql5.com/en/articles/8311)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12446](https://www.mql5.com/ru/articles/12446)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Multibot in MetaTrader (Part II): Improved dynamic template](https://www.mql5.com/en/articles/14251)
- [Brute force approach to patterns search (Part VI): Cyclic optimization](https://www.mql5.com/en/articles/9305)
- [OpenAI's ChatGPT features within the framework of MQL4 and MQL5 development](https://www.mql5.com/en/articles/12475)
- [Rebuy algorithm: Multicurrency trading simulation](https://www.mql5.com/en/articles/12579)
- [Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)
- [Multibot in MetaTrader: Launching multiple robots from a single chart](https://www.mql5.com/en/articles/12434)

**[Go to discussion](https://www.mql5.com/en/forum/457514)**

![The case for using Hospital-Performance Data with Perceptrons, this Q4, in weighing SPDR XLV's next Performance](https://c.mql5.com/2/60/Insurance_Claims_Data_with_Perceptrons__Logo.png)[The case for using Hospital-Performance Data with Perceptrons, this Q4, in weighing SPDR XLV's next Performance](https://www.mql5.com/en/articles/13715)

XLV is SPDR healthcare ETF and in an age where it is common to be bombarded by a wide array of traditional news items plus social media feeds, it can be pressing to select a data set for use with a model. We try to tackle this problem for this ETF by sizing up some of its critical data sets in MQL5.

![The price movement model and its main provisions. (Part 3): Calculating optimal parameters of stock exchange speculations](https://c.mql5.com/2/57/Avatar_The_price_movement_model_and_its_main_points_Part_3.png)[The price movement model and its main provisions. (Part 3): Calculating optimal parameters of stock exchange speculations](https://www.mql5.com/en/articles/12891)

Within the framework of the engineering approach developed by the author based on the probability theory, the conditions for opening a profitable position are found and the optimal (profit-maximizing) take profit and stop loss values are calculated.

![Neural networks made easy (Part 50): Soft Actor-Critic (model optimization)](https://c.mql5.com/2/57/NN_50_Soft_Actor-Critic_Avatar.png)[Neural networks made easy (Part 50): Soft Actor-Critic (model optimization)](https://www.mql5.com/en/articles/12998)

In the previous article, we implemented the Soft Actor-Critic algorithm, but were unable to train a profitable model. Here we will optimize the previously created model to obtain the desired results.

![Data Science and Machine Learning (Part 15): SVM, A Must-Have Tool in Every Trader's Toolbox](https://c.mql5.com/2/60/Data_Science_and_Machine_LearningdPart_15g__Logo.png)[Data Science and Machine Learning (Part 15): SVM, A Must-Have Tool in Every Trader's Toolbox](https://www.mql5.com/en/articles/13395)

Discover the indispensable role of Support Vector Machines (SVM) in shaping the future of trading. This comprehensive guide explores how SVM can elevate your trading strategies, enhance decision-making, and unlock new opportunities in the financial markets. Dive into the world of SVM with real-world applications, step-by-step tutorials, and expert insights. Equip yourself with the essential tool that can help you navigate the complexities of modern trading. Elevate your trading game with SVM—a must-have for every trader's toolbox.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/12446&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071926334734676224)

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