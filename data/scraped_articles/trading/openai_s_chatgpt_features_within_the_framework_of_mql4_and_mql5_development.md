---
title: OpenAI's ChatGPT features within the framework of MQL4 and MQL5 development
url: https://www.mql5.com/en/articles/12475
categories: Trading, Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:06:14.552750
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/12475&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069090260454932580)

MetaTrader 5 / Trading


### Contents

- [Introduction](https://www.mql5.com/en/articles/12475#para2)
- [Prospects for using ChatGPT within MQL5](https://www.mql5.com/en/articles/12475#para3)
- [Potential pitfalls associated with ChatGPT](https://www.mql5.com/en/articles/12475#para4)
- [Opportunities for using ChatGPT to solve mathematical problems and develop mathematical models for use in a code](https://www.mql5.com/en/articles/12475#para5)
- [Correct approach to code generation in MQL4 and MQL5](https://www.mql5.com/en/articles/12475#para6)
- [Developing a trading system using ChatGPT](https://www.mql5.com/en/articles/12475#para7)
- [Evaluation of functionality and analyzing results](https://www.mql5.com/en/articles/12475#para8)
- [Conclusion](https://www.mql5.com/en/articles/12475#para9)

### Introduction

So, let's start from the basics. This technology is one of the variations of artificial intelligence from OpenAI, designed to help people in solving various tasks. This tool looks like a regular messenger chat. However, at the other end there is a so-called artificial intelligence that answers you in text form.

Of course, the communication channel is limited only to a text, but it is still sufficient for solving various problems or learn a lot of new things. This text channel is suitable for solving completely different problems, such as programming, mathematics, physics, chemistry, not to mention skillful translation and other abilities.

We are interested in this model only in terms of the development of profitable trading systems. I became interested in how to optimally and correctly use this technology for faster and easier development of trading systems. Ultimately, the one who first begins to correctly apply a technology for its intended purpose reduces both the cost of any development and labor costs, which gives obvious competitive advantages.

### Prospects for using ChatGPT within MQL5

Let's dwell in more detail on the technology prospects. After a fairly detailed study on my own examples, I realized that such technology is only the beginning of something truly great. But even now I can highlight the following features:

- Generation of any MQL4 and MQL5 codes
- Working code refactoring and optimization
- Cleaning up a code
- Adding comments to a code
- Fixing errors
- Implementing mathematical models
- Subsequent code building based on mathematical models
- Modernization of any known algorithms and mathematical models
- Speeding up the Expert Advisor (EA) development process
- A huge amount of information

This list is by no means final, and you can add something of your own here. I think that when people learn about this kind of technology, they all start to fall into roughly three subgroups:

1. "Now we will make a super algorithm"
2. Those who are wary of AI and questioning its usefulness
3. Machine cannot be better than a human. It's all just another hype

I began to get acquainted with this technology a long time ago and at the beginning I belonged to the third category. In the first two days of dealing with this AI, I abruptly moved from the third to the first category, after which a more interesting and rather unpleasant process of adjusting my own beliefs began, which is more like a rollback to the "1.5" category, which is already a more realistic assessment of this technology.

This technology is useful, but not as much as you might initially think. In this regard, it is worth paying tribute to the developers and marketers of this technology, if only for the fact that it creates an incredible "wow" effect from use in the first couple of days, which is enough for a chain reaction of self-promotion.

To understand this, you need a lot of practice in dealing with this AI. Personally, I had about a hundred different dialogues with it on various topics. I can say that I have gained enough practice in using this technology to start using it for MQL5. Before moving on to practical applications, I just need to tell you some very important information, and for this we will need to look deeper under the hood of this technology.

### Potential pitfalls associated with ChatGPT

The so-called "wow" effect you will experience in the early days of using this technology is due to the fact that this is primarily a text model that is designed to turn your question into an answer. The reason why you will like its answers is because the creators taught this model to lie beautifully. Yes, I am not kidding! It lies so beautifully that you will want to believe it yourself, and even after many attempts to expose it, the model will often provide you with something like this:

- Sorry! Yes, you are right. I have made a small mistake. I will take this into account in the future. (In fact, it will not. This is just an empty promise)
- Sorry for misunderstanding, I have made a mistake. Here is the corrected version. (Containing even more errors)
- You might find out that the model made a mistake in calculations. (In fact, the model made no calculations at all, it simply found an approximate number somewhere)
- You have detected an error in a code. (The model will again make excuses and try to deceive you)
- The model imitates execution of a task and tries to convince you that it did what you asked for. (Actually, it finds something similar somewhere and just provides it to you. )
- Sorry! I was unable to help you. (This happens when the model understands that you have detected its lies or mistakes)
- The model adds a lot of excessive words into its answer to make an impression of a harmonious and coherent answer. (I think, this is somehow connected to the optimization of resource costs)

In short, the model will try to skip out on the provided task in every possible way, while taking advantage of the imperfection of your "prompt". In case of mistakes, it will try to justify itself in such a way that you are not be able to catch it on the same thing, and if it understands that you have got it pegged, it will try to soften your negative reaction with certain replies and psychological tricks. In general, I think that the algorithm is tailored for optimal resource consumption with an acceptable level of user satisfaction. In other words, the goal in this case is not to give you the most high-quality solution to the problem, but to provide you with a solution that you consider as such, due to completely different possible reasons. It turns out that the goal is a positive reaction from a user, and it is not so important whether the task is solved correctly. This is what it strives for. From the marketing point of view, this is correct, and in this regard, the AI can be as persuasive as possible.

Considering all this, I realized that in order to minimize its freedom in this aspect, we should first of all accumulate experience in communicating with it and draw conclusions about what type of tasks does not require the use of this persuasiveness simply because it will be easy for it to fulfill our request since it is most easily fulfilled given its functionality and it is much easier to give a direct answer than to enable imagination. Because it will not be efficient in terms of resource costs, in other words, it will simply not be profitable for it to deceive you, but it will be much easier to give you a true answer.

In order to begin to understand how to ask such questions, we need to understand the following:

01. The structure and type of a "prompt" are not that important, the details and the quality of the question are of greater importance. (The model understands everything and there is no need to read Internet guides on how to compose prompts. Write as you wish, but without slang.)
02. Divide complex questions or requests into subparagraphs. (Simplify and divide questions to get more precise, simple and clear answers.)
03. The smarter and more punctual your question is, the more useful the answer.
04. It will not give you a super idea or an algorithm. (The model does not have the breadth of thinking, planning and intellectual creativity. But at first it will seem that it is capable of creating it simply because, as I said, it lies very well.)
05. We should think about the use of this technology solely in the context of speeding up our work and reducing our labor costs.
06. Each new request should be the least dependent on the entire dialog. (The model is unable to remember the entire dialog and old messages are often not taken into account in replies. Just try it and you will understand what I mean.)
07. The more complex the question and the more details it contains, the higher the probability to get complete nonsense. (This is an explanation of the second subparagraph.)
08. The model has no Internet access, and it generates all answers only based on its knowledge base. (If you urge it to get sone data from the Internet, it will give you the old data from its database or adjust the answer based on your context passing it off as a new one. Keep this in mind. The model will do this simply because it realizes that it is useless to argue with you and it is better to convince you that it did everything.)
09. ChatGPT 3.5 was trained till 2019. (Which means that he has no information about the events after 2019 until the next training session sanctioned by the developers.)
10. ChatGPT 4.0 was trained till 2021. (It is better because it lies very little and tries to always answer correctly. If you try to ask questions and compare, you will see that 3.5 lies blatantly.)

In fact, there are a lot of other smaller unpleasant moments that spoil the impression of this technology. But nevertheless, I would not write about it if this technology was not useful. This all boils down to the fact that this is by no means an artificial intelligence. But if we remain pragmatic and think whether everything is so bad and what routine operations we can perform faster and better with the help of this technology, then I am sure we will not be too harsh in our judgment. We just need to think about how to use all this in the development of trading systems.

Concluding this section, I want to focus your attention on the most important and basic point that you need to remember at all times:

- **Always double check ChatGPT answers and especially numbers, equations and generated code**

Thanks to my knowledge of both math and programming, I happened to see many examples of its mistakes and shortcomings and I can say that they are pretty common. It would seem insignificant from the point of view of the generated text, but when we are dealing with math or code, even the most insignificant mistake makes the whole solution useless. Therefore, always double-check answers, correct mistakes and call the model's attention to that. Sometimes it answers correctly for a while. This subparagraph, among other things, will be extremely useful when developing your EAs.

### Opportunities for using ChatGPT to solve mathematical problems and develop mathematical models for use in a code

Since this model is textual, it is easy to guess that if we write equations in the correct format, then it will understand them and, moreover, perform mathematical transformations and solve problems. When developing many trading systems, you will need help in creating mathematical equations and expressions. It is possible to solve some possible mathematical problems with their subsequent implementation in a code. There is the following format for writing mathematical equations, which ChatGPT understands and uses to give the answer:

- LaTeX

An example of using the latex format for writing equations:

Code:

```
E &=& mc^2\\
m &=& \frac{m_0}{\sqrt{1-\frac{v^2}{c^2}}}
```

Paste this into any [free latex converter](https://www.mql5.com/go?link=https://latexeditor.lagrida.com/ "https://latexeditor.lagrida.com/") and get a data visualization of all familiar expressions:

![Enstein's energy and relativistic mass expansion equations](https://c.mql5.com/2/55/1__1.png)

I think, now it is clear how to visually interpret the model answers in LaTeX format. Most importantly, do not forget to ask the model to generate equations in this format if the answer contains mathematical expressions. There are also neural networks capable of converting equations in pictures or other formats back into the format we need. I think, you will find these tools if you need them. My task is to show you the existence of such a possibility.

There are Telegram bots that combine many neural networks, including ChatGPT, and the function of converting images back to LaTeX format. If you wish, you can find one of them in [my profile](https://www.mql5.com/en/users/w.hudson). This bot was made by my friend and tested by me personally.

You can ask ChatGPT, for example, to solve an inequality or an equation, both numerically and explicitly. You can also ask to solve systems of equations or inequalities, as well as differential equations or integrals, or perform any required mathematical transformations. However, as a mathematician, I can say that it does not always do that efficiently and rationally and sometimes leaves the task unfinished. So double checks are necessary.

Of course, this feature can be useful to non-mathematicians. In case of a prolonged use, you will make more mistakes, and your solution will be quite irrational and clumsy. However, you will cover some of your math problems for your code, given that the code usually uses only numerical methods, while the applied math is not that complicated. There is no differential math here.

### Correct approach to code generation in MQL4 and MQL5

This is where things get interesting. Given that the sizes of codes of all sufficiently high-quality and more or less decent trading systems are quite large, it is worth thinking about how to approach the process of generating such codes. Here the main obstacle is that the size of the answer to your question is limited to a certain number of characters, and after numerous attempts to generate large and complex codes, I came to the conclusion that each code output should be short enough. This means that the code should be displayed in parts. How can we achieve this? The answer is very simple - we need to make a plan for developing an EA, indicator or script.

The plan should be drawn up with the condition that each sub-item should be a separate sub-task that can be solved independently. Then we can simply solve each sub-task sequentially, and then combine all the code together. Another advantage of this approach is that each sub-task can then be finalized separately, and since each sub-task is simpler than all of them combined together, finalization is performed faster and in a more comfortable manner. In addition, we will avoid more errors.

As a developer, it is much more comfortable for me to independently think over the main architecture of my EA without allowing AI to interfere. Instead, let it implement separate procedures in my EA. Let all the main logic be contained in the procedures. We will only have to implement an approximate code template with empty functions, and then ask it to implement each function individually. We can also ask it to implement function prototypes or other structures.

An additional and important advantage is that you can ask it to prepare a plan for an EA or other code, indicating your requirements, and then simply implement its plan piece by piece. It is good when you have a rough or precise idea of what your EA's algorithm will be like and what indicators or other approaches it will use. But if there is no such idea, you can first talk to it and ask it to help you choose a trading strategy for your EA. It will offer you some options. Let's consider this course of actions here as an example.

Let's now summarize the above and form brief sub-points, symbolizing the possible paths we will take when building a new EA from scratch. To begin with, there are several obvious possible scenarios for getting started:

1. We have not decided on the architecture of the future code and do not know where to start, and we also do not know at all which of the trading approaches to choose.
2. We have not decided on the architecture of the future code and do not know where to start, but we know a rough picture of the main working code and what we want from the EA.
3. We have a ready-made architecture that is comfortable for us, but we absolutely do not know which trading approach to choose.
4. We know the architecture we want to use and also have a clear idea of the future trading logic of the EA.

As a rule, everything will be reduced to similar constructions. All four points can be applied to the general scheme of building any trading system if we do the following:

- **If we do not know the architecture (main code or framework), then first of all we need to implement it, and then implement everything that ensures the functioning of this framework.**

This may mean, for example, that we can ask to implement classes, input variables, fields and method prototypes, interfaces, as well as the main trading functionality of the EA, which will use the entities described by us. With proper handling of ChatGPT, the code can be implemented in such a way that it takes, say, no more than 5-10% of the total number of characters. In this case, we can quickly implement it and then move on to the implementation of the procedures, which will contain about 90% of the entire code. These procedures will be implemented in the same simple manner, because there will be a lot of them, and they will turn out to be quite small and easily executable. Of course, it is much easier when you have a ready-made template and you do not have to implement all this, but this requires knowledge and experience.

### Developing a trading system using ChatGPT

I believe I have provided you enough theoretical information. It is time to apply it. In my case, I use a ready-made template to base my EAs on. I described such a pattern in one of the previous articles. Its peculiarity is that it provides parallel trading of many instruments, being activated on a single chart. It already features all the necessary trading functionality and the main architecture. I will build a trading system strictly following recommendations of ChatGPT. I will implement the main trading logic of the EA myself, as well as the visual component, because this will require less efforts from me.

When you start interacting with ChatGPT, you will realize that you will spend much more efforts trying to explain to it what needs to be done, and you will correct its answers a hundred times when implementing some requests and tasks. After a while, you will simply begin to feel which questions are worth asking and which are not. You will begin to set only those tasks that will ultimately save your time instead of wasting it. There is a rather thin line here, which you have to feel for yourself - there is no other way. Everything is learned in practice. My approach to the EA design has been formed fully due to these considerations.

First of all, I asked to describe the basis of the EA - what its working principle is and what methods or indicators it is to use (I allowed the model to use any available information at its discretion). At the same time, I stated that I only need logical conditions from it in a readable form, which I can implement on my own in the following four predicates:

1. Opening a buy position.
2. Closing a buy position.
3. Opening a sell position.
4. Closing a sell position.

To implement these predicates, the model offered me the following conditional logic:

1. The current price is locked above the EMA, the difference between the current price and the EMA is less than the ATR \* ratio, and the RSI is less than 30.
2. The current price is closed below SMA, or the current price is closed above the upper band of the Bollinger Bands indicator.
3. The current price is locked below the EMA, the difference between the current price and the EMA is less than the ATR \* ratio, and the RSI is greater than 70.
4. The current price is locked above the SMA, or the current price is locked below the lower band of the Bollinger Bands indicator.

Obviously, these boolean conditions return 'true' on success and 'false' on failure. These signal values are quite sufficient for trading with market orders. Here I want to draw your attention to the obvious possibility of modernizing this logic. To do this, we can do the following:

- \[ **K1**\] — zone of the lower RSI value
- \[ **K2** = 100 - **K1**\] — zone of the upper RSI value

These expressions can be used to expand the flexibility of the algorithm, which will subsequently have a positive effect on the efficiency of the EA optimization:

1. The current price is locked above the EMA, the difference between the current price and the EMA is less than the ATR \* ratio, and the RSI is less than **K1**
2. The current price is locked below the SMA or the current price is closed above the upper band of the Bollinger Bands indicator
3. The current price is closed below the EMA, the difference between the current price and the EMA is less than the ATR \* ratio, and the RSI is greater than **K2**
4. the current price is locked above the SMA or the current price is locked below the lower band of the Bollinger Bands indicator

I provided this example because you should not hesitate to extend the model if it is obvious that the solution in question is only a special case of a different, more extended algorithm. Even if you do not know what such an extension can give you, by doing this you at least increase the flexibility of your algorithm, and hence the likelihood of its finer tuning and, as a result, a possible increase in its efficiency.

Considering what conditions need to be implemented, we will need one of two options for implementing the following indicators:

1. SMA — standard moving average (one line)
2. EMA — exponential moving average (one line)
3. Bollinger Bands — Bollinger bands (a set of three lines)
4. RSI — relative strength index (one line in a separate window)
5. ATR — average true range (one line in a separate window)

Indicators can be implemented using special predefined MQL5 functions, but I do not like this approach, because the implemented code will be more difficult to convert to the MQL4 version. Besides, it will be more difficult for me to integrate it, for example, in my projects in other languages, which I do very often. I have long been in the habit of doing everything as simply as possible and with the future use in mind. I think, this is a very good habit.

The second important point is that, as a rule, such indicators drag along unnecessary and redundant calculations and functionality. In addition, it is impossible to refine such indicators, since their functions are rigidly set at the code level. In order to make changes, you will need to create your own version of the indicator in any case. I think it is obvious that it is better to have a custom implementation inside an EA or a script. In order to implement such indicators, I came up with the following trick:

1. Creation of arrays for storing the values of the indicator lines (limited to the last N bars).
2. Implementation of shifting array values when a new bar appears.
3. Implementing clearing the array of indicator values in case of errors or a long disconnect.
4. Implementing the calculation of the indicator value for the last bar when it closes.

In this approach, the first three paragraphs create common array blocks and functions that provide the listed actions. Let's see how this looks using our task as an example. Let's start with the first point:

```
   double SMA1Values[]; // Array for storing SMA values
   double EMAValues[];  // Array for storing EMA values (exponential)
   double RSIValues[];  // Array for storing RSI values

   double BollingerBandsUpperValues[];  // Array for storing BollingerBands values, upper
   double BollingerBandsMiddleValues[]; // Array for storing BollingerBands values, middle
   double BollingerBandsLowerValues[];  // Array for storing BollingerBands values, lower

   double ATRValues[];// array for storing Average True Range values
```

These arrays are initialized at the start of the EA with the given length limits:

```
   //Prepare indicator arrays
   void PrepareArrays()
   {
      ArrayResize(SMA1Values, LastBars);
      ArrayResize(EMAValues, LastBars);
      ArrayResize(RSIValues, LastBars);
      ArrayResize(BollingerBandsUpperValues, LastBars);
      ArrayResize(BollingerBandsMiddleValues, LastBars);
      ArrayResize(BollingerBandsLowerValues, LastBars);
      ArrayResize(ATRValues, LastBars);
   }
```

Unlike conventional indicators, we do not need to drag all the previous values with us for this strategy. This is definitely an advantage. I like this implementation paradigm, because it ensures the simplicity of the code and the equivalence of both the starting values of the indicator and those obtained using the previous ones. Now let's see how the value shift looks like:

```
   //shift of indicator values
   void ShiftValues()
   {
      int shift = 1;
      for (int i = LastBars - 1; i >= shift; i--)
      {
         SMA1Values[i] = SMA1Values[i - shift];
         EMAValues[i] = EMAValues[i - shift];
         RSIValues[i] = RSIValues[i - shift];
         BollingerBandsUpperValues[i] = BollingerBandsUpperValues[i - shift];
         BollingerBandsMiddleValues[i] = BollingerBandsMiddleValues[i - shift];
         BollingerBandsLowerValues[i] = BollingerBandsLowerValues[i - shift];
         ATRValues[i] = ATRValues[i - shift];
      }
   }
```

As you can see, everything is extremely simple. The same will apply to clearing arrays:

```
   //reset all indicator arrays if connection fails [can also be used when initializing an EA]
   void EraseValues()
   {
      for (int i = 0; i < LastBars; i++)
      {
         SMA1Values[i] = -1.0;
         EMAValues[i] = -1.0;
         RSIValues[i] = -1.0;
         BollingerBandsUpperValues[i] = -1.0;
         BollingerBandsMiddleValues[i] = -1.0;
         BollingerBandsLowerValues[i] = -1.0;
         ATRValues[i] = -1.0;
      }
   }
```

I think, it is pretty clear where this functionality will be used. Now let's move on to the implementation of the indicators themselves. To do this, I asked ChatGPT to implement the appropriate function, which would be suitable based on my code building paradigm. I have started with the SMA indicator:

```
   //1 Function that calculates the indicator value to bar "1"
   double calculateMA(int PeriodMA,int Shift=0)
   {
      int barIndex=Shift+1;//bar index SMA is calculated for (with a shift)
      int StartIndex=barIndex + PeriodMA-1;//starting bar index for calculating SMA
      if (StartIndex >= LastBars) return -1.0; // Check for the availability of the bars for calculating SMA (if not valid, then the value is -1)
      double sum = 0.0;

      for (int i = StartIndex; i >= barIndex; i--)
      {
         sum += Charts[chartindex].CloseI[i];
      }
      LastUpdateDateTime=TimeCurrent();
      return sum / PeriodMA;
   }
```

As you can see, the function turned out to be very simple and short. Initially, the appearance of this function was a bit different. During the first generation, I found a lot of errors in it, for example, related to the fact that it misunderstood the direction of the numbering of bars relative to time, and so on. But after some manipulations, I fixed all this and additionally added the Shift parameter, which was not in the original implementation. After implementing some visual improvements, I asked to implement the rest of the indicators in a similar fashion. After that, there were fewer errors in its implementations. I just sent the following requests to implement a similar function for another indicator, including examples of previous implementations in the context of the question. This saved a lot of time. Let's now look at its subsequent implementations of all the remaining indicators. Let's start with EMA:

```
   //2 Function that calculates the value of the exponential moving average to bar "1"
   double calculateEMA(int PeriodEMA,double Flatness=2.0,int Shift=0)
   {
      int barIndex = Shift+1; // bar index EMA is calculated for (with a shift)
      int StartIndex=barIndex + PeriodEMA-1;//index of the starting bar for calculating the first SMA, for starting the recurrent calculation of EMA
      if (StartIndex >= LastBars) return -1.0; // Check for the availability of the bars for calculating EMA (if not valid, then the value is -1)

      double sum = 0.0;
      double multiplier = Flatness / (PeriodEMA + 1); // Weight multiplier
      double prevEMA;

      // Calculate the initial value for the EMA (the first value is considered as a normal SMA)
      for (int i = StartIndex; i >= barIndex; i--)
      {
         sum += Charts[chartindex].CloseI[i];
      }
      prevEMA = sum / PeriodEMA;//this is the starting value for the bar (StartIndex-1)

      // Apply the EMA formula for the remaining values
      for (int i = StartIndex; i >= barIndex; i--)
      {
         prevEMA = (Charts[chartindex].CloseI[i] - prevEMA) * multiplier + prevEMA;
      }

      LastUpdateDateTime = TimeCurrent();
      return prevEMA;
   }
```

By the way, when researching ChatGPT generations, we have to turn to various Internet resources in order to understand which indicator is based on which idea. This makes us more competent. This is especially evident with EMA. If we look closely at the code, we will see that it does not differ much from the simpler implementation of SMA and rather looks like an add-on to the usual moving average. There is no exponent here, although for some reason it is present in the indicator name. The next is RSI indicator:

```
   //3 Function for calculating RSI to bar "1"
   double calculateRSI(int PeriodRSI,int Shift=0)
   {
       int barIndex = Shift+1; // bar index RSI is calculated for (with a shift)
       int StartIndex = barIndex + PeriodRSI - 1; // starting bar index for calculating RSI
       if (StartIndex >= LastBars) return -1.0; // Check for the availability of the bars for calculating RSI (if not valid, then the value is -1)

       double avgGain = 0.0;
       double avgLoss = 0.0;
       double change;

       // Calculate initial values for avgGain and avgLoss
       for (int i = StartIndex; i > barIndex; i--)
       {
           change = Charts[chartindex].CloseI[i]-Charts[chartindex].OpenI[i];
           if (change > 0)
           {
               avgGain += change;
           }
           else if (change < 0)
           {
               avgLoss -= change;
           }
       }

       avgGain /= PeriodRSI;
       avgLoss /= PeriodRSI;

       // RSI calculation
       double RS = 0.0;
       if (avgLoss != 0)
       {
           RS = avgGain / avgLoss;
       }

       double RSI = 100 - (100 / (1 + RS));

       LastUpdateDateTime = TimeCurrent();
       return RSI;
   }
```

Now we need to do a similar procedure for the Bollinger Bands indicator. This procedure should return the values of three curves, which can be put into the preliminarily prepared structure:

```
   //structure to return all three bollinger band lines
   struct BollingerBands
   {
       double upper;
       double middle;
       double lower;
   };
```

Now, having added this structure to the context of the question, we can ask to implement the function, taking into account the fact that it should return the BollingerBands type. The model will cope with such a task without problems:

```
   //4 Function for calculating the Bollinger Bands of the indicator
   BollingerBands calculateBollingerBands(int PeriodBB, double DeviationBB,int Shift=0)
   {
       int barIndex = Shift+1; // bar index Bollinger Bands is calculated for (with a shift)
       int StartIndex = barIndex + PeriodBB - 1; // index of the starting bar for calculating the first SMA, for starting the recurrent calculation of EMA
       BollingerBands rez;
       rez.lower=-1.0;
       rez.middle=-1.0;
       rez.upper=-1.0;
       if (StartIndex >= LastBars) return rez; // Check for the availability of the bars for calculating BB (if not valid, then the value is -1)

       double sum = 0.0;
       double prevBB;
       double sumSquares = 0.0;

       // Calculate the initial value for BB (the first value is considered as a normal SMA)
       for (int i = StartIndex; i >= barIndex; i--) {
           double closePrice = Charts[chartindex].CloseI[i];
           sum += closePrice;
       }
       prevBB = sum / PeriodBB; //this is the starting value for the bar (StartIndex-1)

       // Calculation of standard deviation
       for (int i = StartIndex; i >= barIndex; i--) {
           double closePrice = Charts[chartindex].CloseI[i];
           sumSquares += pow(closePrice - prevBB, 2);
       }
       double standardDeviation = sqrt(sumSquares / PeriodBB);

       // Calculate Bollinger Bands
       double upperBand = prevEMA + DeviationBB * standardDeviation;
       double lowerBand = prevEMA - DeviationBB * standardDeviation;

       rez.upper = upperBand;
       rez.middle = prevEMA;
       rez.lower = lowerBand;

       LastUpdateDateTime = TimeCurrent();
       return rez;
   }
```

Now it remains to implement the version of the function for calculating ATR:

```
   //5 Function for calculating Average True Range (Relative)
   double calculateRelativeATR(int PeriodATR,int Shift=0)
   {
       int barIndex = Shift+1; // bar index ATR is calculated for (with a shift)
       int StartIndex = barIndex + PeriodATR - 1; // starting bar index for calculating the first ATR
       if (StartIndex >= LastBars) return -1.0; // Check for the availability of the bars for calculating ATR and True Range (if not valid, then the value is -1)

       double sumPrice=0.0;
       double sumTrueRange = 0.0;
       double ATR;

       // Calculating True Range for bars and the sum of values for calculating the first ATR
       for (int i = StartIndex; i >= barIndex; i--)
       {
           sumPrice+=Charts[chartindex].HighI[i]+Charts[chartindex].LowI[i]+Charts[chartindex].CloseI[i]+Charts[chartindex].OpenI[i];//added by me
           double high = Charts[chartindex].HighI[i];
           double low = Charts[chartindex].LowI[i];
           double trueRange = high - low;
           sumTrueRange += trueRange;
       }

       // ATR calculation
       //ATR = sumTrueRange / PeriodATR; - conventional calculation
       ATR = 100.0 * (sumTrueRange / PeriodATR)/(sumPrice/(PeriodATR*4.0));//calculation of relative ATR in %

       LastUpdateDateTime = TimeCurrent();
       return ATR;
   }
```

Here pay attention to the commented line at the end. I slightly modified this indicator so that it operates relative values. This is necessary so that we do not have to set our own weights for each trading instrument. Instead, this will happen automatically based on the current price. This will allow for more efficient multi-currency optimization. We will need it to prove the fact that even such a simple algorithm, if used correctly, can give us a small but sufficient forward period for trading. In combination with other methods of efficiency, this trade can be made quite acceptable, even at the level currently allowed by EA.

I implemented the predicates myself. It was very easy. Let's look at one of them, let's say the first one:

```
   //to open buy positions
   bool bBuy()
      {
      //determine if an open position is already present
      bool ord;
      ulong ticket;
      bool bOpened=false;
      for ( int i=0; i<PositionsTotal(); i++ )
         {
         ticket=PositionGetTicket(i);
         ord=PositionSelectByTicket(ticket);
         if ( ord && PositionGetInteger(POSITION_MAGIC) == MagicF)
            {
            bOpened=true;
            return false;
            }
         }

      if (!bOpened && EMAValues[1] > 0.0)//only if nothing is open and the indicator has been calculated
         {
         //K - control ratio
         //RSIPercentBorder - control RSI
         double Val1=Charts[chartindex].CloseI[1]-EMAValues[1];
         double Val2=ATRValues[1]*(1.0/K);
         if (Val1 > 0 && Val1 < Val2 && RSIValues[1] < RSIPercentBorder) return true;
         }
      return false;
      }
```

The predicate for opening a sell position is similar with minor exceptions. The closure predicate is even simpler:

```
   //to close a buy position
   bool bCloseBuy()
      {
      if (SMA1Values[1] > 0.0)
         {
         if (Charts[chartindex].CloseI[1] < SMA1Values[1] || Charts[chartindex].CloseI[1] > BollingerBandsUpperValues[1] )
            {
            return true;
            }
         }
      return false;
      }
```

All this will work in a very simple manner:

```
   IndicatorDataRecalculate();//recalculate indicators

   if ( bCloseBuy() )
      {
         CloseBuyF();
      }
   if ( bCloseSell() )
      {
         CloseSellF();
      }
   if ( bBuy() )
      {
         BuyF();
      }
   if ( bSell() )
      {
         SellF();
      }
```

I think, it is as simple as possible, and there is no need for it to be more complicated. All this code should be executed when a new bar appears. I implemented the visualization of indicators separately. The only thing I do not like is that indicators like ATR and RSI are designed to be drawn in a separate window. I made my version of rendering for them so that they are also tied to the price, since a separate window cannot be created artificially, and, frankly, they are not really needed. To achieve this, I created a certain paradigm for drawing window indicators.

1. Entering the Percent control value to create three corridors out of one.
2. Determining the maximum (Max) and minimum (Min) indicator values for the entire array of stored values.
3. Calculating the delta of the given corridor (Delta = Max - Min).
4. Calculating the upper corridor of increased values (HighBorder = Max - Delta \* Percent / 100).
5. Calculating the lower corridor of increased values (LowBorder = Min + Delta \* Percent / 100).
6. The middle corridor has already been defined, since both the upper and lower corridors have been defined.

If the current value of the indicator lies in one of the corridors, then its points acquire the appropriate color corresponding to the corridor. All is simple. This is how we can bind values to the chart bars and, for example, just change their color accordingly, or just create objects linked to the bar and having the corresponding color. As many have probably noticed, I was inspired by the RSI indicator, because this particular structure is usually used for trading. These areas are called overbought and oversold zones there.

I think that this code is not so important here because it has the least relation to the implementation of our task, but only helps in correcting possible errors and improvements. If you wish, you can even implement this rendering using ChatGPT. However, I think it is worth showing how this rendering works:

![Visualizing the indicators](https://c.mql5.com/2/55/indicators.png)

Here everything is done as simply as possible with the sole help of Line object type. If, when creating a line, the start and end points of the line are tied to the same time and price, then the line turns into a dot. By adjusting the line thickness, we thereby adjust the weight of the corresponding dot. These are just some life hacks from me that I have been using for a very long time.

### Evaluation of functionality and analyzing results

Although ChatGPT considers this algorithm to be optimal, we do not know what these decisions are based on. A good backtest or real trading may serve as a measure of efficiency. I hope everyone understands that real trading should be preceded by the optimal setting, which can be done using the MetaTrader 5 optimizer. Due to the fact that this terminal has the possibility of multi-currency optimization and given the capabilities of my template, which fully uses the effectiveness of this tool, we can safely optimize the EA for all "28" currency pairs. Perhaps, it is not worth listing them. But it is worth noting the obvious advantages of this approach:

1. Automatic search for multicurrency patterns
2. Multi-currency patterns have more weight and further adaptation to market changes
3. It turns out more trades, since each trading instrument provides something of its own
4. Saving time (you do not have to optimize each tool individually)

There are, of course, downsides. In this case, the most important one is the inability of fine tuning for each instrument. Of course, this problem can be solved by introducing additional functionality, but this is not the topic of this article. But it is not enough to perform this optimization, it is also important to correctly select the results from the entire scope it offers you. I chose the optimal one:

![Optimal optimization option](https://c.mql5.com/2/55/2sz87ik_1.png)

As you can see, I optimized from "2018" to "2023" using the "H4" timeframe, and I left all six months from "2023" for the forward. As we can see, despite the far-from-perfect profit curve on the optimization section, we got a couple more months for profitable trading, and this fact means that the strategy is not without meaning and has the potential to be successfully used for trading. Many optimizable trading systems would most likely not even come close to this result. Sometimes you can test incredibly cool systems in terms of code while not getting such results.

I would add a lot more to this algorithm. I consider it a playground. But it is worth saying that it is not at all important, but the expansion potential of the resulting EA. It is not the proposed algorithm that plays a role here, but your competence and creativity. You will need important qualities for successful integration into the MQL5 development process:

- Programming skills (mandatory)
- Math skills (desirable)
- Block thinking
- Ability to simplify tasks and break them down to separate stages
- Understanding the fact that ChatGPT is just a tool, so there is no point in blaming it for things that do not work (it is up to you to fix anything that does not work)
- Correct interpretation of obtained trading results
- Realization that the model makes mistakes anyway, but this should not bother you (the main thing is that you have reduced development labor cost)
- Developing your own application structure (this is not necessary, of course, you can use my approach)
- ChatGPT is your companion whose strength depends on you. The more patient, smarter and more resourceful you are, the more effective this companion is.

You have a tool that you can either use or not, at your discretion, but I can definitely say that this tool was very useful to me in many aspects, including programming.

### Conclusion

In fact, it is not necessary to adhere to my model of using this tool. I tried a lot of things with it, and I can say that possibilities are almost limitless. Of course, before you succeed, you will definitely spend some amount of time and effort, like me.

But I still advise you to try this tool and do not spare your time. At the very least, using ChatGPT will allow you to get the right attitude towards artificial intelligence. It is potentially possible to integrate this technology into almost any development chain. I am sure that this is just the beginning, and changes are just around the corner. I advise you to start familiarizing yourself with this tool as soon as possible and try to apply it in the widest range of tasks.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12475](https://www.mql5.com/ru/articles/12475)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12475.zip "Download all attachments in the single ZIP archive")

[ChatGPT\_bot.set](https://www.mql5.com/en/articles/download/12475/chatgpt_bot.set "Download ChatGPT_bot.set")(2.99 KB)

[ChatGPT\_bot.mq5](https://www.mql5.com/en/articles/download/12475/chatgpt_bot.mq5 "Download ChatGPT_bot.mq5")(145.44 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Multibot in MetaTrader (Part II): Improved dynamic template](https://www.mql5.com/en/articles/14251)
- [Brute force approach to patterns search (Part VI): Cyclic optimization](https://www.mql5.com/en/articles/9305)
- [Brute force approach to patterns search (Part V): Fresh angle](https://www.mql5.com/en/articles/12446)
- [Rebuy algorithm: Multicurrency trading simulation](https://www.mql5.com/en/articles/12579)
- [Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)
- [Multibot in MetaTrader: Launching multiple robots from a single chart](https://www.mql5.com/en/articles/12434)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/452888)**
(34)


![Longsen Chen](https://c.mql5.com/avatar/2021/4/6066B2E5-2923.jpg)

**[Longsen Chen](https://www.mql5.com/en/users/gchen2101)**
\|
15 Jan 2024 at 08:39

I tried to get MT5 code but failed. It seems ChatGPT can only provide MT4 code


![Yauheni Shauchenka](https://c.mql5.com/avatar/2024/3/65E432BB-4AD4.png)

**[Yauheni Shauchenka](https://www.mql5.com/en/users/merc1305)**
\|
2 Apr 2024 at 15:23

The essence of the article can be described by your phrase " everything that doesn't work is your part of the work you have to do" :)

Have you tried working with copilot, [Evgeniy @Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)? [https://www.metatrader5.com/ru/metaeditor/help/development/copilot](https://www.metatrader5.com/ru/metaeditor/help/development/copilot "https://www.metatrader5.com/ru/metaeditor/help/development/copilot")

![Evgeniy Ilin](https://c.mql5.com/avatar/2021/1/60171FA1-EC8E.png)

**[Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)**
\|
2 Apr 2024 at 19:13

**Yauheni Shauchenka [#](https://www.mql5.com/ru/forum/449719/page3#comment_52913846):**

The gist of the article can be described by your phrase " anything that doesn't work is your part of the work you have to do" :)

Have you tried working with copilot, [Evgeniy @Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)? [https://www.metatrader5.com/ru/metaeditor/help/development/copilot](https://www.metatrader5.com/ru/metaeditor/help/development/copilot "https://www.metatrader5.com/ru/metaeditor/help/development/copilot")

What's the point ) it's GPT 3.5 which is free. It's practically rubbish now. And nobody will give you access to GPT 4.0 for free, because it costs money at least. And GPT 4.0 I will tell you so I do not really know mql5. The saving grace is that this language is a spin-off from C++ and has a similar syntax, so it can give something out. At least some sketch, which can be further improved. I have access to GPT 4.0, I don't use Copilot. And there are better neurons than GPT 4.0, in fact, for working with code... well, that's just the way it is.

![Rajesh Kumar Nait](https://c.mql5.com/avatar/2025/11/69247847-e34b.png)

**[Rajesh Kumar Nait](https://www.mql5.com/en/users/rajeshnait)**
\|
1 Aug 2024 at 21:57

Block thinking which you mentioned is a most important part to make Chat **Generative Pre-Trained Transformers work for you**

Create a template, break logics in parts, and ask GPT to write e.g. "Write function void findSMA(int index) {} to find sma value" which may be more useful instead asking something like "Build a MA Crossover EA for me with [bollinger bands](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/bb "MetaTrader 5 Help: Bollinger Bands indicator")"

![Evgeniy Ilin](https://c.mql5.com/avatar/2021/1/60171FA1-EC8E.png)

**[Evgeniy Ilin](https://www.mql5.com/en/users/w.hudson)**
\|
1 Aug 2024 at 23:01

**Rajesh Kumar Nait [#](https://www.mql5.com/en/forum/452888#comment_54174079):**

Block thinking which you mentioned is a most important part to make Chat **Generative Pre-Trained Transformers work for you**

Create a template, break logics in parts, and ask GPT to write e.g. "Write function void findSMA(int index) {} to find sma value" which may be more useful instead asking something like "Build a MA Crossover EA for me with [bollinger bands](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/bb "MetaTrader 5 Help: Bollinger Bands indicator")"

Everything is correct. You just need experience working with such models and understanding how they can help in creating code. From the very beginning of using the model, I wasn't even interested in what prompts could be there, I just wrote "do this or that" and eventually understood how to work with this tool. It's a tool that speeds up development by leaps and bounds. It has its drawbacks, but it only seems so at first, while you're thinking in a style like "write me a grail in 2 lines". To this day, I use it and it already seems like I won't be able to do without it. The effect was like with google back in the day..

![Wrapping ONNX models in classes](https://c.mql5.com/2/54/ONNX_Models_in_the_Class_Avatar.png)[Wrapping ONNX models in classes](https://www.mql5.com/en/articles/12484)

Object-oriented programming enables creation of a more compact code that is easy to read and modify. Here we will have a look at the example for three ONNX models.

![Testing different Moving Average types to see how insightful they are](https://c.mql5.com/2/57/moving_average_types_avatar.png)[Testing different Moving Average types to see how insightful they are](https://www.mql5.com/en/articles/13130)

We all know the importance of the Moving Average indicator for a lot of traders. There are other Moving average types that can be useful in trading, we will identify these types in this article and make a simple comparison between each one of them and the most popular simple Moving average type to see which one can show the best results.

![Developing a Replay System — Market simulation (Part 05): Adding Previews](https://c.mql5.com/2/53/replay-p5-avatar.png)[Developing a Replay System — Market simulation (Part 05): Adding Previews](https://www.mql5.com/en/articles/10704)

We have managed to develop a way to implement the market replay system in a realistic and accessible way. Now let's continue our project and add data to improve the replay behavior.

![Category Theory in MQL5 (Part 17): Functors and Monoids](https://c.mql5.com/2/57/Category-Theory-p17-avatar.png)[Category Theory in MQL5 (Part 17): Functors and Monoids](https://www.mql5.com/en/articles/13156)

This article, the final in our series to tackle functors as a subject, revisits monoids as a category. Monoids which we have already introduced in these series are used here to aid in position sizing, together with multi-layer perceptrons.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/12475&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069090260454932580)

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