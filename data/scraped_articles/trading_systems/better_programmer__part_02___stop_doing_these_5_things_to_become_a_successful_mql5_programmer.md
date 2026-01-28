---
title: Better Programmer (Part 02): Stop doing these 5 things to become a successful MQL5 programmer
url: https://www.mql5.com/en/articles/9711
categories: Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:32:17.362014
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/9711&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070366574411453584)

MetaTrader 5 / Trading systems


### Introduction

I use the word [noob](https://www.mql5.com/go?link=https://www.google.com/search?q=noob+meaning "https://www.google.com/search?q=noob+meaning") a lot in this article series to describe someone with less experience in MQL5 programming (don't get offended by that). It makes it easy to relate to. This doesn't mean that this article is only for noobs. Absolutely not! This is for everyone, regardless of the programming experience you have, because what makes you a noob or not is your coding habits and behaviors... not years of coding.

> ![noobs_vs_professional coder](https://c.mql5.com/2/43/noobVs_Pro.png)

The first article is [here](https://www.mql5.com/en/articles/9643) for those who haven't read it, so let me continue this mission to turn noobs to professionals.

### 01: Stop thinking you are better than others

So you've been coding for a while now and managed to create several great EAs and indicators for yourself and your customers on the freelance, codebase or something, so now you are good. It's good if you notice your progress and use that to increase your confidence to tackle huge challenges to improve more and not use it in an arrogance manner. Keep in mind that:

_"An arrogant person considers himself perfect. This is the chief harm of ignorance. It interferes with a person's main task in life — becoming a better person."_

— Leo Tolstoy

It's so easy to feel super smart and to criticize someone else's code without fully understanding it, or rushing to respond to their questions in the forum in offensive way to show how you are a badass coder and that person who just asked the question on the forum sucks at coding. I have seen a lot of times where people ask a question on the forum and they post their code and describe what they have tried before that is not working. Then suddenly a noob pops up with an explanation that this person is terribly wrong, without offering a solution or offering a solution ( _code in this case_) less accurate than the code that was provided by the topic starter in the first place. **OMG!**

![noobs answer](https://c.mql5.com/2/43/noobs_answer.png)

Instead of jumping straight to describing that the topic starter is terribly wrong then providing code that's not even in the right format, **why** not start with explanations on why there is a problem than explain what needs to be done in a detailed and clear way?

There is no fun being in the forum starting topics. I'm sure anybody who starts a topic on the forum already knows that they are wrong in someway and that's why they are not getting the desired results. Chances are high that the topic starter is a noob and they have been struggling with a problem for hours if not days _(I know how it feels) ._

Even jumping straight and dragging down non-explained or unclear code won't fix their problem in the long run. It's like giving pain killer to someone who needs a surgery. By doing this you will be teaching noobs the worst habit that I first described in the [first article](https://www.mql5.com/en/articles/9643) of this series—the **Copy and Paste Habit**—by making them think like " _so its working this way here, I have a problem I post my code in the forum and I'll be given the code to replace the bad one … this is cool!"_

Always make sure that you explain the solution in a clear, detailed and beginner friendly way because most of them are. Also keep in mind there is a lot of noobs who will stumble upon the same problem in the future so they will probably need a solution from that topic.

Apart from that example, thinking we are better than other causes wrong answers in the forum and false reviews on the codebase. By thinking you are better than the author of that particular system just because you see the code is not much complicated the way you think it should be or the way that you like and familiar with, we tend to refuse to fully understand how and why the code was written in the first place, hence leading to ignorance.

### 02: Avoid having the fixed mindset

Stop thinking that there is no way that you can become good in a particular area.

_"Insanity is doing the same thing over and over again and expecting different results."_

—Albert Einstein

Don't be stuck into making the same type of Expert Advisors, indicators, scripts or whatever else. This destroys your creativity and keeps stuck in the same place. Thinking you were not born to be good in a particular programming area just because its too complex or tough is self limiting. Just know that your brain is flexible. It's designed to adapt and change to new stimuli. The more you challenge it the better you will become at that particular subject. You can become good at:

- [Math](https://www.mql5.com/en/articles/1365)
- Complex [Expert Advisors](https://www.mql5.com/en/articles/100) and [Indicators](https://www.mql5.com/en/articles/10)
- [Machine Learning](https://www.mql5.com/en/articles/7447) and self adapting algorithms

and everything you can think of. But you have to put enough work to make it happen.

Yes, of course, there is no holy grail EA ( _I haven't seen it myself_) or indicator so far, but that should not stop you from improving your systems to the best they can be by changing your approaches.

On my early days of MQL5 programming, I was so scared when I downloaded something like simple moving average from the C [odeBase](https://www.mql5.com/en/code) and found out the code was over 1000 lines or more _because that was the first thing that I was looking in the code_. I thought that was too complex. I think its the same issue that noobs are facing.

I'm a bit experienced now. All I can say is that it might be hard now but that's ok. Everything in life is like that in the beginning but it will be easy, only time and effort will tell.

### 03: Stop writing code that will be used someday

So you have been creating your EA(s) or indicator(s) on your project. You now have too many lines of code right. Without much attention and careful re-reading your code, there is a big chance that there is unwanted and unused code.

_"Those who have knowledge don't predict. Those who predict don't have knowledge."_

— Lao Tzu

Always less code means

- Fewer bugs
- Less time to read
- Easy to compile
- Fast to review
- Fast shipping
- Maintainable
- Easy to debug

Don't spend much time writing code that is not needed right now.

I hope that at least once in MQL5 programming you have seen something like this, from someone's else code

```
int MathRandInt(int mini, int maxi)
  {
   double f   = (MathRand() / 32768.0);
   return mini + (int)(f * (maxi - mini));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void chart_random()
 {
//---
  string background[]={"clrBlack"};
  string bulls[]={"clrGreen","clrDeepPink","clrYellow","clrCrimson","clrMediumBlue","clrKhaki","clrYellow"};
  string bears[]={"clrDarkViolet","clrDarkOrange","clrIndigo","clrWhite","clrSilver","clrMediumPurple","clrBlue"};
//---
   ChartSetInteger(ChartID(), CHART_MODE, CHART_CANDLES);
   ChartSetInteger(ChartID(), CHART_SHOW_GRID, false);
   ChartSetInteger(ChartID(), CHART_AUTOSCROLL, true);
   ChartSetInteger(ChartID(), CHART_SHOW_PERIOD_SEP, true);
   ChartSetInteger(ChartID(), CHART_SHIFT, true);
   ChartSetInteger(ChartID(), CHART_SHOW_ASK_LINE, true);
   ChartSetInteger(ChartID(), CHART_SHOW_BID_LINE, true);
   ChartSetInteger(ChartID(), CHART_COLOR_ASK, clrBlue);
   ChartSetInteger(ChartID(), CHART_COLOR_BID, clrCrimson);
   ChartSetInteger(ChartID(), CHART_SCALE, 2);
   ChartSetInteger(ChartID(), CHART_FOREGROUND, true);
   ChartSetInteger(ChartID(), CHART_COLOR_FOREGROUND,clrWhite);
//---
/*
 for (int i=0; i<ArraySize(background); i++){
     int maxi=ArraySize(background);
     int random_=MathRandInt(0,maxi);
    ChartSetInteger(0,CHART_COLOR_BACKGROUND,StringToColor(background[random_]));
    }
  for (int i=0; i<ArraySize(bulls); i++){
    ChartSetInteger(0,CHART_COLOR_CANDLE_BULL,StringToColor(bulls[MathRandInt(0,ArraySize(bulls))]));
    ChartSetInteger(0,CHART_COLOR_CHART_UP,StringToColor(bulls[MathRandInt(0,ArraySize(bulls))]));
   }
  for (int i=0; i<ArraySize(bears); i++){
    ChartSetInteger(0,CHART_COLOR_CANDLE_BEAR,StringToColor(bears[MathRandInt(0,ArraySize(bears))]));
    ChartSetInteger(0,CHART_COLOR_CHART_DOWN,StringToColor(bears[MathRandInt(0,ArraySize(bears))]));
   } */
//---
   int y_distance=70;
   int font_size=30;
   int width=(int)ChartGetInteger(ChartID(),CHART_WIDTH_IN_PIXELS);
   int padding=70;
   int x=width/3;
//---
    string label_name= MQLInfoString(MQL_PROGRAM_NAME);
    ObjectCreate(0, "name", OBJ_LABEL, 0, 0, 0);
    ObjectSetString(0,"name",OBJPROP_TEXT,label_name);
    ObjectSetString(0,"name",OBJPROP_FONT,"Lucida Console");
    ObjectSetInteger(0,"name",OBJPROP_CORNER,CORNER_RIGHT_LOWER);
    ObjectSetInteger(0,"name",OBJPROP_XDISTANCE,x+padding);
    ObjectSetInteger(0,"name",OBJPROP_YDISTANCE,y_distance);
    ObjectSetInteger(0,"name",OBJPROP_FONTSIZE,font_size);
    ObjectSetInteger(0,"name", OBJPROP_SELECTABLE, false);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
```

The example image is too small—I have seen over 200 lines of code that was commented (hidden) and was unnecessary inside the program.

[MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor") detects unnecessary variables at a function level by warning you of unused variables so that you can choose between keep them and their warning at the same time if it pleases you or to get rid of them.

But for the rest of code we have to check that out for ourselves to make sure that we don't have these unwanted code.

_If you don't use it - you don't need it - then delete it._

Unused code causes confusion and not to mention they increase your scrolling time to necessary code when they are in the middle of your program's code.

### 04: You don't have to always be right.

_"A man who has committed a mistake and doesn't correct it, is making another mistake."_

— Confucius

The main reason to why noobs run immediately to the [Forum](https://www.mql5.com/en/forum) to drag down their code and provide fewer non clear description to what and why they were coding that way and what they wanted, is because they are too afraid to expose themselves to more problems. This is not wise, my dear noobs!

The coding career is full of problems and never-ending challenges, and there is no better way to become great at it if not to solve as many problem as you can get exposed to. I don't mean you should not post in the forum your problems ( _don't get me wrong_). But make sure you've tried a bit solving it on your own that's why its a good practice before you start a topic you describe things you've tried. The reason behind this is to get your mind focused to it and by challenging yourself to different methods on your own not only might solve the puzzle but also you might discover things that you did not know the first time and learn more.

Once the mistake is fixed, make sure you understand it and how to avoid it the next time.

### 05: Give up the idea of overnight success

> _"If you look closely most overnight success took a long time."_
>
> — Steve Jobs

Ok I understand that you have been coding for a while now but you feel like you are not making any progress, you don't feel like you are going no where. Right?

This feeling happens a lot of times to noobs, because most of them expected the results to be immediately but the world doesn't work that way.

Sit back, relax, and enjoy the process. You are actually making a progress; your doubts are just part of the journey but they should not stop you. I would recommend to work on your favorite projects so that you can have a lot of fun while still learning but don't get stuck there. Make sure you stay open to new ideas and possibilities as I have explained on the second point.

Put the work in, and the progress will come automatically as an end product.

- Read as many source codes you can get your eyes on, on the [Codebase](https://www.mql5.com/en/code).
- Read as many articles as you can in [Articles](https://www.mql5.com/en/articles).
- Spend more hours in the [Forum](https://www.mql5.com/en/forum) helping others.
- Experiment with a lot of stuff to find the one that works best for you.

Remember

_Success doesn't come from what you do occasionally. It comes from what you do consistently._

### Conclusion

To be honest I'm not this magnificent, not perfect in MQL5 programming or in my web development career. I'm still a student in this Coding thing and there is no ending to learning it, because it never stops teaching. I always strive to improve and to become better at it everyday. The reason I have the guts to share this with you ( _to teach while being a student_) is to help people with the same passion as me to **Become a Better Programmer**.

> > Let's walk this journey together!

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 04): Tester 101](https://www.mql5.com/en/articles/20917)
- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/375048)**
(18)


![Alexey Viktorov](https://c.mql5.com/avatar/2017/4/58E3DFDD-D3B2.jpg)

**[Alexey Viktorov](https://www.mql5.com/en/users/alexeyvik)**
\|
30 Sep 2021 at 07:40

Oh, how gentle we all are... Does it really sting?


![Aliaksandr Hryshyn](https://c.mql5.com/avatar/2016/2/56CF9FD9-71DB.jpg)

**[Aliaksandr Hryshyn](https://www.mql5.com/en/users/greshnik1)**
\|
30 Sep 2021 at 08:36

"'A proud man thinks he's always quite good. This is why pride is especially harmful. It prevents a person in the main business of people's life, in becoming better" (Leo Tolstoy)." - don't insult Tolstoy, he did nothing to you, it's necessary to take the text from the original and not through third hands. Loss of style, with mistakes, horror


![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
5 Oct 2021 at 12:24

In any case, what the author writes about is important. Maybe everything is more colourful on the hub. Maybe the style isn't the same here. And mistakes, oh horror. We should not dwell on our own subjective opinion and take a closer look at ourselves. Maybe we have intolerance to other opinions, pride and conceit. That's where the horror really lies.


![avavolcano](https://c.mql5.com/avatar/avatar_na2.png)

**[avavolcano](https://www.mql5.com/en/users/avavolcano)**
\|
8 Oct 2021 at 05:03

lit. not knowing what's going on (idiom); ignorant and strict


![BinkoBinev](https://c.mql5.com/avatar/avatar_na2.png)

**[BinkoBinev](https://www.mql5.com/en/users/binkobinev)**
\|
17 Oct 2021 at 12:52

**You dsYoug Chee [#](https://www.mql5.com/en/forum/375048#comment_24004460):**

How can I make a function that only allow 1 trade or open position per candle not only at the beginning of the candle but until new candle forms. The newbar function only allows at beginning of new candle only

You should check open times of your positions and whether one already exists in the specified period of the candle. If it exists, then do nothing. If it does not exist,then open a new position.

![Patterns with Examples (Part I): Multiple Top](https://c.mql5.com/2/42/gdij.png)[Patterns with Examples (Part I): Multiple Top](https://www.mql5.com/en/articles/9394)

This is the first article in a series related to reversal patterns in the framework of algorithmic trading. We will begin with the most interesting pattern family, which originate from the Double Top and Double Bottom patterns.

![Graphics in DoEasy library (Part 78): Animation principles in the library. Image slicing](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library3-2__5.png)[Graphics in DoEasy library (Part 78): Animation principles in the library. Image slicing](https://www.mql5.com/en/articles/9612)

In this article, I will define the animation principles to be used in some parts of the library. I will also develop a class for copying a part of the image and pasting it to a specified form object location while preserving and restoring the part of the form background the image is to be superimposed on.

![Better Programmer (Part 03): Give Up doing these 5 things to become a successful MQL5 Programmer](https://c.mql5.com/2/43/Article_image__1.png)[Better Programmer (Part 03): Give Up doing these 5 things to become a successful MQL5 Programmer](https://www.mql5.com/en/articles/9746)

This is the must-read article for anyone wanting to improve their programming career. This article series is aimed at making you the best programmer you can possibly be, no matter how experienced you are. The discussed ideas work for MQL5 programming newbies as well as professionals.

![Better Programmer (Part 01): You must stop doing these 5 things to become a successful MQL5 programmer](https://c.mql5.com/2/42/Article_image.png)[Better Programmer (Part 01): You must stop doing these 5 things to become a successful MQL5 programmer](https://www.mql5.com/en/articles/9643)

There are a lot of bad habits that newbies and even advanced programmers are doing that are keeping them from becoming the best they can be to their coding career. We are going to discuss and address them in this article. This article is a must read for everyone who wants to become successful developer in MQL5.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/9711&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070366574411453584)

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