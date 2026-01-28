---
title: Developing a Replay System (Part 28): Expert Advisor project — C_Mouse class (II)
url: https://www.mql5.com/en/articles/11349
categories: Trading, Trading Systems
relevance_score: 6
scraped_at: 2026-01-22T18:02:07.511470
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/11349&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049576506042199533)

MetaTrader 5 / Tester


### Introduction

In the previous article " [Developing a Replay System (Part 27): Expert Advisor project (II)](https://www.mql5.com/en/articles/11337)", we started developing a new class. However, towards the end of the article, I became convinced of the importance of presenting a different approach to programming. This will be only for curiosity, in order to approach natural language.

For those who have been programming for a long time, what we show below may not make much sense. Why go to all this trouble to bring programming closer to natural language? The answer is simple: **you are not programming for the machine, but for other programmers**. At the time of the first systems capable of factoring something everything depended on how well the engineers knew the project. This was the reality at the dawn of computer technology, when there were no terminals for programming.

As it developed and more people got interested in being able to create something, new ideas and ways of programming emerged which replaced the previous-style changing of connector positions. This is when the first terminals appeared. Over time, programming, which was originally done exclusively in binary format, was no longer the norm. This happened because programs evolved very quickly, which led to the need to find a more efficient way to read what was programmed. At that time the assembly language emerged. This powerful framework transformed complex binary code work into something more readable, in the form of OpCodes or mnemonic code. Programs became more and more complex, requiring more and more code, and the first higher-level languages appeared.

There was no longer a need to deal directly with OpCodes as it became possible to use a language closer to natural. At first, these languages were developed primarily to create and describe mathematical concepts, that is, they mainly served to facilitate the translation of formulas into computer-readable forms. This process no longer had to be done manually by a person. This gave birth to a new era - the era of the compiler, which translated **human language** into a language that a machine can understand. I've been programming this way for years, trying to explain how programs are made and getting more people to learn and translate their ideas into something that a computer can understand. However, I realized that many people have difficulty understanding some concepts since programming mostly involves combining and using symbols to express what we want to create.

But considering that the MQL5 language is similar to C/C++ and has the ability to do things in a way that makes the code more readable, it becomes ideal for demonstrating something different. Then, after analyzing things, I realized that I could help enthusiasts understand what was being programmed even without fully understanding the code. So, I decided to change the way the code was expressed. In the end, everything will be understood by the compiler, so it doesn't matter to the compiler. But this is of great importance for enthusiasts, since the language will be much closer to natural. Although the code may seem strange and unusual at first glance, it will be much easier for a beginner to understand.

I invite you to join me in these short moments when we use the MQL5 language in a way that is much closer to natural language.

### Creating the Defines.mqh file

Two articles ago we looked at using the compilation directive **#define**. I mentioned that there is a special case where the definition is not removed at the end of the file, although at that time I did not show the practical application of this use of the directive because there was no correct way to do this. Therefore, we left this question open. The key point here is that if you understand some things and concepts about MQL5, knowing that it is derived from C/C++, then you might want to start performing certain operations in the language without much trouble. This would make the code more readable for both non-programmers who cannot understand all these symbols and for programmers who needs to understand what is being done.

One of the ways is to make codes more readable using the **#define** directive. However, there are some limitations. At first glance it may seem quite strange, but it all comes down to knowing how to correctly and without exaggeration define some symbols or combinations of symbols present in the syntax of the MQL5 language. We're not creating a new language. We are just replacing some existing symbols in a fairly consistent manner. This is how the **Defines.mqh** file is born. It will contain definitions with which the syntax, which was previously symbolic, turns into some word or definition that is more expressive for the human reader.

Here is the full content of this file:

```
//+------------------------------------------------------------------+
#property copyright "Daniel Jose"
/*+------------------------------------------------------------------+
Definitions for increasing the level of the coded language in MQL5, for
more details see the article at the following link:
https://www.mql5.com/en/articles/11349
+------------------------------------------------------------------+*/
#define equal =
#define different !=
#define identical ==
#define and &&
//+------------------------------------------------------------------+
```

What does this little piece of code that performs virtually no function actually do? To understand this, you need to use it in practice. Each of the code lines represents an addition that seeks to make the code more readable. Even a person with little experience will be able to understand, if not all, then some aspects of programming. The code readability is something we always need to improve. It's better to have more readable code, even if it means a little more work up front. But in the end it will be worth it, because no one wants to deal with code that looks like a hieroglyph, inaccessible even to the author. And once this way of code writing is lost, either because the language or the person ceases to exist, all the knowledge contained in it will also be lost.

One of the ways to make code more readable is by using comments. However, if you look at the codes in my articles, you'll notice that I don't comment. This is because, in my opinion, such codes are quite simple and perfectly readable. Could you understand these codes without the descriptions in the articles? This is exactly the point. At some point, the code can become so complex that without a good structure it becomes completely unreadable, and even I will not be able to maintain and improve it.

Once the Defines.mqh file is created, we need to somehow force the compiler to use it. To do this, we include this file in one of the most basic files for building the entire system - C\_Terminal.mqh. The C\_Terminal.mqh part with includes has the following line:

```
#include "Macros.mqh"
#include "..\..\Defines.mqh"
```

Be careful because this is very important. We declare the Macros.mqh file in double quotes, which tells the compiler that this file is in the same folder as the C\_Terminal.mqh file. However, when we include the Defines.mqh file, it is also enclosed in double quotes, but with one specific feature. This difference is in **..\**, which tells the compiler how many levels, starting from the directory in which C\_Terminal.mqh is located, we will need to go up in the directory structure to find the Defines.mqh file. In this case, we need to go up two levels because the directory structure contains different code levels. So, the Defines.mqh file is located at the root of the project's directory structure. If for some reason the project root changes, it will not affect the compiler, which will always look for the Defines.mqh file in the same location.

This kind of organization is very interesting, especially when you start organizing your header base to make it easy to find and plan things. This will allow us to easily distribute our code, or part of it, without worrying about missing a particular file. Having everything well organized and ready to distribute code makes our lives a lot easier. Now that we have explained how to include the Defines.mqh file in the system, we can start using it. You can increase the number of definitions to make everything more readable. Well, the current definitions are sufficient for our purposes. But to really understand how much this helps make the code more readable and easier to solve problems that arise, I suggest you look at the example below:

```
if (m_Info.Study == eStudyExecute) ExecuteStudy(memPrice);
if (m_Info.Study identical eStudyExecute) ExecuteStudy(memPrice);
if_case m_Info.Study identical eStudyExecute then ExecuteStudy(memPrice);
```

These three lines mean the same, and the compiler will interpret them the same way, generating exactly the same code. However, in this case, the Defines.mqh file will not be able to tell the compiler what to do. We need to add two new definitions:

```
#define if_case if(
#define then )
```

If you add these two lines to the Defines.mqh file, the compiler will be able to correctly understand the three lines in the example. Pay attention to the highlighted line. Here we have a language very similar to the natural language. Of the three lines shown, this one is the most similar to natural language and thus it can be named the highest level among the three. This is what I meant when I said that code is either high-level or low-level. Nothing changes for the compiler, but for a human eye the third line is much simpler. This is a simple case, but let's look at a more complex case where all the code will be written as shown in the figure, not forgetting that even the codes shown in previous articles have undergone similar changes to make them more readable. This is for now.

Let's go back to where we stopped in the previous article. We were going to look at the last function present in the C\_Mouse class.

### DispatchMessage: Communication with the external world

All classes that one way or another must receive events from the MetaTrader 5 platform will have this DispatchMessage function in their portfolio. With this the class will be able to receive and respond to generated events. The most important thing is to understand that MetaTrader 5 is an event-based program, that is, a program of **REAL TIME** type. Working with this is, to put it mildly, quite difficult. Therefore, any code must be very specific when dealing with such events. But before we look at the DispatchMessage function, we need to become familiar with the other code that will appear in the C\_Terminal class. This code can be seen below:

```
const double AdjustPrice(const double arg) const { return NormalizeDouble(round(arg / m_Infos.PointPerTick) * m_Infos.PointPerTick, m_Infos.nDigits); }
```

This code could have been placed in the C\_Mouse class, but due to other factors I decided to place it in the C\_Terminal class. This code is simply a factorization that seeks to adjust the price so that it is always equal to the value expected by the trading server. Often orders are rejected by the server due to the fact that the specified price is incorrect. Many people refuse to consider creating an EA just because when they try to send an order to the trading server they receive an error in response. Some asset classes have a simpler factorization for price adjustment, while others have a much more complex one involving multiple issues. However, in practice, the above function manages all of these factors, ensuring that the price is always appropriate, regardless of the type of asset used. This is very important to us because in order to create and use an EA in the replay/simulation system, it is necessary that regardless of the asset, the price is correct, be it on a DEMO account or on a REAL account. Therefore, you can use and even abuse this feature. If you do this, you can see how the system adapts to any type of market and asset. So use and study this capability.

Now that we've seen a function that will adjust the price as needed, we can start looking at the DispatchMessage function. Its full code is shown below:

```
virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
   {
      int w equal 0;
      static double memPrice equal 0;

      C_Terminal::DispatchMessage(id, lparam, dparam, sparam);
      switch (id)
      {
         case (CHARTEVENT_CUSTOM + ev_HideMouse):
            ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
            break;
         case (CHARTEVENT_CUSTOM + ev_ShowMouse):
            ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, m_Info.corLineH);
            break;
         case CHARTEVENT_MOUSE_MOVE:
            ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X equal (int)lparam, m_Info.Data.Position.Y equal (int)dparam, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
            ObjectMove(GetInfoTerminal().ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price equal AdjustPrice(m_Info.Data.Position.Price));
            if (m_Info.Study different eStudyNull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineV, 0, m_Info.Data.Position.dt, 0);
            m_Info.Data.ButtonStatus equal (uint) sparam;
            if (CheckClick(eClickMiddle) and ((color)ObjectGetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR) different clrNONE)) CreateStudy();
            if (CheckClick(eClickLeft) and (m_Info.Study identical eStudyCreate))
            {
               ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
               ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 0, m_Info.Data.Position.dt, memPrice equal m_Info.Data.Position.Price);
               m_Info.Study equal eStudyExecute;
            }
            if (m_Info.Study identical eStudyExecute) ExecuteStudy(memPrice);
            break;
         case CHARTEVENT_OBJECT_DELETE:
            if (sparam identical def_NameObjectLineH) CreateLineH();
            break;
      }
   }
```

Much of what's going on in the code above can be understood, even without much programming experience, just by looking at the code. How can you understand the code without having a lot of programming knowledge? When you program in a language so that your code begins to approximate natural language, then the code becomes easier to understand. This way, even people who do not understand programming will be able to understand what is being programmed there. In fact, a lot of what's going on there is very easy to understand, it's so simple that it doesn't even need a detailed explanation. Now, please take a look at the following lines.

```
int w equal 0;
static double memPrice equal 0;
```

Although the code is declared this way, you should read in the same way as it is written. The compiler will interpret this code as follows:

```
int w = 0;
static double memPrice = 0;
```

As you can see, there is no difference. In both cases, anyone could understand the code. In the first case we have a "literal" format. But don't worry, we're just getting started. This example does not fully reflect all of our options to make the code more readable.

Let's look at another code. This part is not directly related to the readability of the code but still requires clarification.

```
case (CHARTEVENT_CUSTOM + ev_HideMouse):
   ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, clrNONE);
   break;
```

Sometimes we need to hide the price line on the chart. Those who followed the order system shown in the series on how to develop an EA from scratch saw that at some points the price line was hidden. In this particular code, this was done by calling a specific method that caused the line to be hidden. However, for now we will use a message that will be sent to the C\_Mouse class to hide the price line. I decided to sue a message instead of a method is because I want to build a more modular and easily portable system, among other things. So we have another message that meets the same criteria. It is shown below:

```
case (CHARTEVENT_CUSTOM + ev_ShowMouse):
   ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR, m_Info.corLineH);
   break;
```

Don't worry about how to use these messages just yet. We will consider this in the future. In fact, it is easier to understand them if you think that in both message could be methods or functions. However, rather than adding a public method or function for this purpose, we will concentrate everything into one central point: the message processing function. Only in special cases will we use a different method.

These events occur as a result of a special call, which will be discussed in more detail later, since I think many have no idea how this is done. These are not events fired by the MetaTrader 5 platform, but events fired by our code at a very specific time in order to perform some action. But we also need to work with events coming from the MetaTrader 5 platform. This is done as follows:

```
C_Terminal::DispatchMessage(id, lparam, dparam, sparam);
```

This line of code from the DispatchMessage function of the C\_Mouse class forwards calls to the C\_Terminal class, so that we do not have to do this in other parts of the code. This helps avoid programming oversights and standardizes code, making it quicker to program, analyze, create, and fix. However, not all events will be handled in the C\_Terminal class: some of them will be resolved locally, that is, within the class with which we are working. An example of such an event is shown below:

```
case CHARTEVENT_OBJECT_DELETE:
   if (sparam identical def_NameObjectLineH) CreateLineH();
   break;
```

This is the code for the event intended for the reader. I think you can understand that it is interpreted by the compiler as follows:

```
case CHARTEVENT_OBJECT_DELETE:
   if (sparam == def_NameObjectLineH) CreateLineH();
   break;
```

Regardless of how the code is presented, its result is the following: when an object is removed from the symbol chart, the platform triggers a message, or rather an event. It notifies the program that requested this type of message that the object has been removed from the chart. Meeting the **CHART\_EVENT\_OBJECT\_DELETE** handler, the program must execute the code present in it. If the name of the object specified in the **param** constant matches the one being tested, then the executed code will recreate the price line.

Now we have the **CHART\_EVENT\_MOUSE\_MOVE** event which is slightly expanded. Despite being more extensive, it is not very complicated. You can understand most of the code below, even without programming knowledge, just by trying to read literally every line. Try it and tell me if it's easier or harder to understand what we're doing. It doesn't matter that you can't understand all the code, as long as you try to understand as much as possible without extra effort.

```
case CHARTEVENT_MOUSE_MOVE:
   ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X equal (int)lparam, m_Info.Data.Position.Y equal (int)dparam, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
   ObjectMove(GetInfoTerminal().ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price equal AdjustPrice(m_Info.Data.Position.Price));
   if (m_Info.Study different eStudyNull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineV, 0, m_Info.Data.Position.dt, 0);
   m_Info.Data.ButtonStatus equal (uint) sparam;
   if (CheckClick(eClickMiddle) and ((color)ObjectGetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR) different clrNONE)) CreateStudy();
   if (CheckClick(eClickLeft) and (m_Info.Study identical eStudyCreate))
   {
      ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
      ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 0, m_Info.Data.Position.dt, memPrice equal m_Info.Data.Position.Price);
      m_Info.Study equal eStudyExecute;
   }
   if (m_Info.Study identical eStudyExecute) ExecuteStudy(memPrice);
   break;
```

Except for those moments where we call MQL5 functions, I think that you were able to read the entire code and it was not difficult for you to understand some points. For example:

- If **m\_Info.Study** is different from **eStudyNull**, something should be executed.
- **m\_Info.Data.ButtonStatus** is equal to **sparam**.
- If the middle button was pressed and anything (price line color) is different from **clrNONE**, then do the following.
- If it was pressed with the left mouse button and **m\_Info.Study** equals **eStudyCreate**, then this action will be performed.
- Set eStudyExecute to m\_Info.Study.
- If **m\_Info.Study** is equal to **eStudyExecute**, run this.

You can see that even if you read all the above points that have been demonstrated, it still shows that we can add even more things to our Defines.mqh file to make the language even more readable than what I am demonstrating. We simply can add more elements and make the program more readable. This is a quality of a good programming language, and it will always be present in programs of good technical quality. Another way to make the code fairly readable is to always add comments to the most important functions or points, and in this regard MQL5 is significantly superior to C/C++. Try placing comments on variables and procedures. When using MetaEditor, it displays these comments as a tooltip, which is very helpful.

How would the above code actually be programmed? Or more precisely, how would the compiler actually see the above code? This is shown below:

```
case CHARTEVENT_MOUSE_MOVE:
   ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X = (int)lparam, m_Info.Data.Position.Y = (int)dparam, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
   ObjectMove(GetInfoTerminal().ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price = AdjustPrice(m_Info.Data.Position.Price));
   if (m_Info.Study != eStudyNull) ObjectMove(GetInfoTerminal().ID, def_NameObjectLineV, 0, m_Info.Data.Position.dt, 0);
   m_Info.Data.ButtonStatus = (uint) sparam;
   if (CheckClick(eClickMiddle) && ((color)ObjectGetInteger(GetInfoTerminal().ID, def_NameObjectLineH, OBJPROP_COLOR) != clrNONE)) CreateStudy();
   if (CheckClick(eClickLeft) && (m_Info.Study == eStudyCreate))
   {
      ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
      ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 0, m_Info.Data.Position.dt, memPrice = m_Info.Data.Position.Price);
      m_Info.Study = eStudyExecute;
   }
   if (m_Info.Study == eStudyExecute) ExecuteStudy(memPrice);
   break;
```

Both codes do the same. Therefore, for those who do not have a complete understanding of the functions present in the MQL5 language, I will explain what we are doing here. The first thing we do is convert the graphical coordinates reported by the platform into price and time coordinates. This way we can know where the mouse pointer is in relation to the chart. Next, we adjust the price to match the expectations of the trading server. Similarly, we can place the price line in the desired place on the chart. If we are doing analysis, we need to move the timeline correctly. We save the state of the mouse buttons and check to see if the middle button was pressed to explore. However, analysis will only be performed if the price line is visible on the chart. As soon as the left button is pressed, the analysis will begin. Therefore, we need to tell the platform not to move the chart so that we can easily drag the mouse with the left button pressed. While the button is held down, the analysis will be made using all the objects that we want to place on the chart.

And before finishing the article, let's briefly look at the EA code at this stage of development. It's full code is provided below:

```
#property copyright "Daniel Jose"
#property description "Generic EA for use on Demo account, replay system/simulator and Real account."
#property description "This system has means of sending orders using the mouse and keyboard combination."
#property description "For more information see the article about the system."
#property version   "1.28"
#property link "https://www.mql5.com/en/articles/11349"
//+------------------------------------------------------------------+
#include <Market Replay\System EA\Auxiliar\C_Mouse.mqh>
//+------------------------------------------------------------------+
input group "Mouse";
input color     user00 equal clrBlack;          //Price Line
input color     user01 equal clrDarkGreen;      //Positive Study
input color     user02 equal clrMaroon;         //Negative Study
//+------------------------------------------------------------------+
C_Mouse *mouse;
//+------------------------------------------------------------------+
int OnInit()
{
   mouse equal new C_Mouse(user00, user01, user02);

   return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   delete mouse;
}
//+------------------------------------------------------------------+
void OnTick() {}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   (*mouse).DispatchMessage(id, lparam, dparam, sparam);
   ChartRedraw();
}
//+------------------------------------------------------------------+
```

Note that the code is quite simple. Therefore, at this stage I will not explain the process in detail, since this is not particularly necessary.

### Conclusion

From now on, we can monitor the operation of the system using the EA included in the system. However, this EA only operates with the mouse. But now we can use it not only in the replay/simulation system, but also on demo and real accounts. Even though the EA is not yet very useful in such situations, it is very interesting because it can be used on any market, symbol or in any situation.

Another important detail: in the attachment, you can find the last two classes (C\_Terminal and C\_Mouse) that use the contents of the Defines.mqh file. This makes the code more readable. However, this is different from what we said at the beginning of the article, where we implied that all code would follow this formatting. Actually, you can use this technique if you wish. Early in my career as a C/C++ programmer, I used this approach for a while to better understand the syntax of the language. I know it can be quite confusing at first, but you can get used to it over time. This technique can be useful, especially in complex projects that require the analysis of extensive Boolean and logical combinations. At this point, the presence of double symbols can complicate things for beginners. As an example, see the following fact:

> _Who has never, even among experienced programmers, confused the use of LOGICAL AND (&) and BOOLEAN AND (&&)? They are almost the same. But in the first case, the operation is performed bit by bit, and in the second, the entire variable is parsed and true or false is returned. This catches a lot of people when we need to create programs very quickly._

For this reason, do not underestimate the knowledge of certain techniques. Although they may seem simple, they actually make a significant contribution to making a program more readable and therefore speeding up its development. I think I've demonstrated in a pretty interesting way how you can perform tasks much faster while still ensuring that the code is always correct without having to waste time trying to figure out why a particular part isn't working as it should.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11349](https://www.mql5.com/pt/articles/11349)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11349.zip "Download all attachments in the single ZIP archive")

[Files\_-\_BOLSA.zip](https://www.mql5.com/en/articles/download/11349/files_-_bolsa.zip "Download Files_-_BOLSA.zip")(1358.24 KB)

[Files\_-\_FOREX.zip](https://www.mql5.com/en/articles/download/11349/files_-_forex.zip "Download Files_-_FOREX.zip")(3743.96 KB)

[Files\_-\_FUTUROS.zip](https://www.mql5.com/en/articles/download/11349/files_-_futuros.zip "Download Files_-_FUTUROS.zip")(11397.51 KB)

[Market\_Replay\_-\_28.zip](https://www.mql5.com/en/articles/download/11349/market_replay_-_28.zip "Download Market_Replay_-_28.zip")(53.71 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/463119)**

![Neural networks made easy (Part 60): Online Decision Transformer (ODT)](https://c.mql5.com/2/59/Online_Decision_Transformer_logo_up.png)[Neural networks made easy (Part 60): Online Decision Transformer (ODT)](https://www.mql5.com/en/articles/13596)

The last two articles were devoted to the Decision Transformer method, which models action sequences in the context of an autoregressive model of desired rewards. In this article, we will look at another optimization algorithm for this method.

![Developing a Replay System (Part 27): Expert Advisor project — C_Mouse class (I)](https://c.mql5.com/2/58/Projeto_Expert_AdvisoraClasse_C_Mous_Avatar.png)[Developing a Replay System (Part 27): Expert Advisor project — C\_Mouse class (I)](https://www.mql5.com/en/articles/11337)

In this article we will implement the C\_Mouse class. It provides the ability to program at the highest level. However, talking about high-level or low-level programming languages is not about including obscene words or jargon in the code. It's the other way around. When we talk about high-level or low-level programming, we mean how easy or difficult the code is for other programmers to understand.

![Experiments with neural networks (Part 7): Passing indicators](https://c.mql5.com/2/59/Experiments_with__networks_logoup.png)[Experiments with neural networks (Part 7): Passing indicators](https://www.mql5.com/en/articles/13598)

Examples of passing indicators to a perceptron. The article describes general concepts and showcases the simplest ready-made Expert Advisor followed by the results of its optimization and forward test.

![Developing a Replay System (Part 26): Expert Advisor project — C_Terminal class](https://c.mql5.com/2/58/replay-p26-avatar.png)[Developing a Replay System (Part 26): Expert Advisor project — C\_Terminal class](https://www.mql5.com/en/articles/11328)

We can now start creating an Expert Advisor for use in the replay/simulation system. However, we need something improved, not a random solution. Despite this, we should not be intimidated by the initial complexity. It's important to start somewhere, otherwise we end up ruminating about the difficulty of a task without even trying to overcome it. That's what programming is all about: overcoming obstacles through learning, testing, and extensive research.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/11349&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049576506042199533)

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