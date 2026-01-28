---
title: Developing a Replay System (Part 30): Expert Advisor project — C_Mouse class (IV)
url: https://www.mql5.com/en/articles/11372
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:42:13.613078
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/11372&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062654501069956707)

MetaTrader 5 / Tester


### Introduction

In the previous article [Developing a Replay System (Part 29): Expert Advisor project — C\_Mouse class (III)](https://www.mql5.com/en/articles/11355) We designed the C\_Mouse class in such a way that we can extend the research functionality without breaking any part of our code. Since we rely on programming techniques that allow the creation of code in parallel, we will have one main project that will develop in an organized manner. At the same time, if desired, additional functionality can be added to the system. This way, when implementing these functions, our main code will not change at all and will not freeze due to excessive use of inheritance.

While object-oriented programming (OOP) is a great way to program, it is much more suitable for production projects where we want to have tight control over what happens and avoid weird bugs as the system grows. Sometimes we need to develop part of a project in parallel. Although it sounds strange, when we add some function or method to a program, it is not very appropriate to place the function, which is it its early stage (at the testing stage) in the middle of the code, which is already at an advanced stage. I mean, you should not add an untested function to your already tested and working code, as this can lead to unpredictable failures.

Because of such flaws, you often have to get the entire project back into the early stages of development. Sometimes, the new function can be embedded in the code in such a way that removing it would be much more difficult than returning the project to an earlier stage. Although many people, especially new programmers, do not actually use directory structures to have a point of return to earlier stages of development, we can, even without using such directory structures, use a certain technique that allows us to return the system to the point where the added resource is not actually part of the completed project.

In this way, we can develop in parallel, while the final project progresses without any problems. As part of this, in the previous article, we have seen how to use pointers. Now let's take it one step further and generate a more complex study based on the basic model. If the study or resource proves suitable for the final project, once it passes a more advanced stage of testing and confirms to be sufficiently stable and reliable, it can be included in the main class system. Thus, what was previously considered a minor project becomes part of the final project, inheriting and being inherited in the class system.

To demonstrate this, we will create a modification of the C\_Mouse class, but without using inheritance and polymorphism. We will get a completely different analytical model, different from the original system that is present in the C\_Mouse class. To do this, we'll create a new class that may (or may not) inherit from the C\_Studies class that we looked at in the previous article. Whether or not to inherit the C\_Studys class is more a personal question than a practical one. In fact, one way or another, one project will have nothing to do with the other, since they can work in parallel. Despite this, any code that belongs to the main system will inherit the C\_Mouse class until the code that extends this class is considered stable and interesting enough for us to use it in the final project.

Before moving on to programming, it is important to know that the system can progress in two different ways. The path we choose depends on what we want to do and how far we want to go. Since we have two paths and the difference between them is very small, let's look at both. In the attached code, you will have access to one of two paths. But if you want, you can make the necessary changes to take a different path.

_My idea here is to demonstrate what we can do within the platform, not how we should do it._

### Additions to the C\_Terminal class

The system we are programming for demonstration purposes does not require or need any additions to the main code, but for practical reasons and to begin testing the creation of objects that we will use frequently as we develop the code, we will add some general code that creates objects on the symbol chart. This way we can start testing and improving it from the very beginning. The code for this has been developed long time ago, and we have already considered choosing a more suitable location. Until we find a better location, the create function will be in the C\_Terminal class, as shown below:

```
inline void CreateObjectGraphics(const string szName, const ENUM_OBJECT obj, const color cor, const int zOrder = -1)
   {
      ObjectCreate(m_Infos.ID, szName, obj, 0, 0, 0);
      ObjectSetString(m_Infos.ID, szName, OBJPROP_TOOLTIP, "\n");
      ObjectSetInteger(m_Infos.ID, szName, OBJPROP_BACK, false);
      ObjectSetInteger(m_Infos.ID, szName, OBJPROP_COLOR, cor);
      ObjectSetInteger(m_Infos.ID, szName, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(m_Infos.ID, szName, OBJPROP_SELECTED, false);
      ObjectSetInteger(m_Infos.ID, szName, OBJPROP_ZORDER, zOrder);
   }
```

This function will become a general function for creating objects that will be displayed on the chart. In several cases we will see it appear in code with only two elements declared. This is because the element is declared with a default value, so it does not need to be declared at call time unless its value turns out to be different for some reason. Additionally, we will always see the function being called with two elements in its declaration. The point is that this property of the **OBJPROP\_ZORDER** object is used to solve a problem that we will see at some points. If we do not define this property correctly, we will have serious problems with objects placed on the symbol chart when working with the program in any mode: replay/simulation, trading on a demo or live account. We've seen how the main code has changed, and now we understand how to use the system in a different way than just using the original code. Let's see further details in separate topics.

### First way: Using inheritance

IN the first path, we use inheritance, not from the C\_Mouse class, but from the C\_Study class, which we looked at in the previous article. Our header file C\_StudyS2.mqh will contain the following code:

```
//+------------------------------------------------------------------+
#include "C_StudyS1.mqh"
//+------------------------------------------------------------------+

// ... Local definitions ....

//+------------------------------------------------------------------+

// ... Local alias ...

//+------------------------------------------------------------------+
class C_StudyS2 : public C_StudyS1
{
   protected:
   private :

// ... Code and internal functions ...

//+------------------------------------------------------------------+
   public  :
//+------------------------------------------------------------------+
      C_StudyS2(C_Mouse *arg, color corP, color corN)
         :C_StudyS1(arg, corP, corN)
      {
// ... Internal code ....

      }
//+------------------------------------------------------------------+
virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
      {
         double v1, v2;
         int w, h;
         string sz1;

         C_StudyS1::DispatchMessage(id, lparam, dparam, sparam);

// ... Internal code ...

      }
//+------------------------------------------------------------------+
};
```

Here we see that the class from the previous article is used according to the principles of inheritance, where we inherit the class and add properties. In many cases this will be the best option, but not always. It is important to know how these paths differ and complement each other. Thus, by using this method, we will be able to get the most out of it. Please note all the points highlighted in the above excerpt. Don't worry, we'll explain how to work with it to create something interesting.

It is logical that since we are using inheritance, the EA, indicator or script code for this case will be slightly different from the case when we do not use inheritance. To understand these differences, let's first look at the EA code for this first case. The full code can be seen below:

```
#property copyright "Daniel Jose"
#property description "Generic EA for use on Demo account, replay system/simulator and Real account."
#property description "This system has means of sending orders using the mouse and keyboard combination."
#property description "For more information see the article about the system."
#property version   "1.30"
#property icon "../../Images/Icons/Replay - EA.ico"
#property link "https://www.mql5.com/en/articles/11372"
//+------------------------------------------------------------------+
#include <Market Replay\System EA\Auxiliar\C_Mouse.mqh>
#include <Market Replay\System EA\Auxiliar\Study\C_StudyS2.mqh>
//+------------------------------------------------------------------+
input group "Mouse";
input color     user00 = clrBlack;      //Price Line
input color     user01 = clrPaleGreen;  //Positive Study
input color     user02 = clrLightCoral; //Negative Study
//+------------------------------------------------------------------+
C_Mouse *mouse = NULL;
C_StudyS2 *extra = NULL;
//+------------------------------------------------------------------+
int OnInit()
{
   mouse = new C_Mouse(user00, user01, user02);
   extra = new C_StudyS2(mouse, user01, user02);

   MarketBookAdd((*mouse).GetInfoTerminal().szSymbol);
   OnBookEvent((*mouse).GetInfoTerminal().szSymbol);
   EventSetMillisecondTimer(500);

   return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   MarketBookRelease((*mouse).GetInfoTerminal().szSymbol);
   EventKillTimer();

   delete extra;
   delete mouse;
}
//+------------------------------------------------------------------+
void OnTick() {}
//+------------------------------------------------------------------+
void OnTimer()
{
   (*extra).Update();
}
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
{
   MqlBookInfo book[];

   if (mouse.GetInfoTerminal().szSymbol == def_SymbolReplay) ArrayResize(book, 1, 0); else
   {
      if (symbol != (*mouse).GetInfoTerminal().szSymbol) return;
      MarketBookGet((*mouse).GetInfoTerminal().szSymbol, book);
   }
   (*extra).Update(book);
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   (*mouse).DispatchMessage(id, lparam, dparam, sparam);
   (*extra).DispatchMessage(id, lparam, dparam, sparam);

   ChartRedraw();
}
//+------------------------------------------------------------------+
```

The only differences from the code in the previous article are the highlighted parts. This is because we use an inheritance system. As we already know, the inheritance system works very well when we want the system to develop smoothly and without many unexpected events. But we may end up encountering other problems that make our lives more difficult. Sometimes we need to use a slightly different method, which will be available in the application. If we want to use the system as an inheritance-based model, that's fine. Just remember to make the changes that have been noted, and everything will go smoothly.

### Second way: Using pointers

In this second path we will see a detailed explanation of the class code. First, let's look at what the EA code looks like. At this stage, we will have large differences in the class code, which will practically only be supplemented by what is shown in the previous topic. Here is the full code of the EA that follows the second path:

```
#property copyright "Daniel Jose"
#property description "Generic EA for use on Demo account, replay system/simulator and Real account."
#property description "This system has means of sending orders using the mouse and keyboard combination."
#property description "For more information see the article about the system."
#property version   "1.30"
#property icon "../../Images/Icons/Replay - EA.ico"
#property link "https://www.mql5.com/en/articles/11372"
//+------------------------------------------------------------------+
#include <Market Replay\System EA\Auxiliar\C_Mouse.mqh>
#include <Market Replay\System EA\Auxiliar\Study\C_StudyS1.mqh>
#include <Market Replay\System EA\Auxiliar\Study\C_StudyS2.mqh>
//+------------------------------------------------------------------+
input group "Mouse";
input color     user00 = clrBlack;      //Price Line
input color     user01 = clrPaleGreen;  //Positive Study
input color     user02 = clrLightCoral; //Negative Study
//+------------------------------------------------------------------+
C_Mouse *mouse = NULL;
C_StudyS1 *extra1 = NULL;
C_StudyS2 *extra2 = NULL;
//+------------------------------------------------------------------+
int OnInit()
{
   mouse = new C_Mouse(user00, user01, user02);
   extra1 = new C_StudyS1(mouse, user01, user02);
   extra2 = new C_StudyS2(mouse, user01, user02);

   MarketBookAdd((*mouse).GetInfoTerminal().szSymbol);
   OnBookEvent((*mouse).GetInfoTerminal().szSymbol);
   EventSetMillisecondTimer(500);

   return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   MarketBookRelease((*mouse).GetInfoTerminal().szSymbol);
   EventKillTimer();

   delete extra1;
   delete extra2;
   delete mouse;
}
//+------------------------------------------------------------------+
void OnTick() {}
//+------------------------------------------------------------------+
void OnTimer()
{
   (*extra1).Update();
}
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
{
   MqlBookInfo book[];

   if ((*mouse).GetInfoTerminal().szSymbol == def_SymbolReplay) ArrayResize(book, 1, 0); else
   {
      if (symbol != (*mouse).GetInfoTerminal().szSymbol) return;
      MarketBookGet((*mouse).GetInfoTerminal().szSymbol, book);
   }
   (*extra1).Update(book);
}
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   (*mouse).DispatchMessage(id, lparam, dparam, sparam);
   (*extra1).DispatchMessage(id, lparam, dparam, sparam);
   (*extra2).DispatchMessage(id, lparam, dparam, sparam);

   ChartRedraw();
}
//+------------------------------------------------------------------+
```

Here I am showing where we use the main class system, marked in yellow. The extension class system we showed in the previous article is marked in green. The system that will perform a different type of analysis, but could be something else, is shown in orange. Note that since we are not using inheritance, we have to declare more code in the EA. At the same time, this allows us to release more parallel code to test what kinds of things we'll have in the final version. The best thing is that if this parallel code starts showing any malfunctions or errors, we can remove it from the code without too much trouble. However, there is another problem here: both orange and green code can be polymorphic. This allows us to test more aspects of the system, which is being developed in parallel. We'll leave the topic of polymorphism for another time. If we talk about it now, we will overcomplicate the explanation, so that enthusiasts may not actually follow all the reasoning behind the use of polymorphism.

After these explanations, you can move on to the class code. Remember that the code for the first and second paths is almost identical, except, of course, for the points indicated in the topic about the first path.

### Let's analyze the code of the C\_StudyS2 class

In a sense, all the codes associated with the analytics system will be very similar to each other, with a few exceptions. However, there are a number of things that make the analytics generation code interesting to analyze and understand. Let's take a closer look at this. Please keep in mind that the code provided here is for demonstration purposes only, it is by no means a complete method. Here is how C\_StudyS2.mqh begins:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "..\C_Mouse.mqh"
#include "..\..\..\Service Graphics\Support\Interprocess.mqh"
//+------------------------------------------------------------------+
#define def_ExpansionPrefix "Expansion2_"
#define def_ExpansionBtn1 def_ExpansionPrefix + "B1"
#define def_ExpansionFibo def_ExpansionPrefix + "FB"
//+------------------------------------------------------------------+
#define def_InfoTerminal (*mouse).GetInfoTerminal()
#define def_InfoMousePos (*mouse).GetInfoMouse().Position
//+------------------------------------------------------------------+
class C_StudyS2
{
   protected:
   private :
//+------------------------------------------------------------------+
      C_Mouse *mouse;
//+------------------------------------------------------------------+
      struct st00
      {
         bool            ExecStudy,
                         ClearStudy;
         double          MemPrice;
         datetime        MemDT;
         color           corP,
                         corN;
       }m_Info;
//+------------------------------------------------------------------+
```

Here we declare the files that need to be included in the system. Please note that the paths are shown relative to the path where this C\_StudyS2.mqh file is located. This will make it easier to move the project to other directories while maintaining its structure. Next, we will define some object names that we will use in the research process. There are also alias declarations to make the programming process easier since they will be used in many places while writing code. And the last thing we see in this fragment is the structure, which will be accessed through a private global variable.

Here is the next code part:

```
#define def_FontName "Lucida Console"
#define def_FontSize 10
       void GetDimensionText(const string szArg, int &w, int &h)
          {
             TextSetFont(def_FontName, -10 * def_FontSize, FW_NORMAL);
             TextGetSize(szArg, w, h);
             h += 5;
             w += 5;
          }
//+------------------------------------------------------------------+
       void CreateBTNInfo(int x, int w, int h, string szName, color backColor)
          {
             (*mouse).CreateObjectGraphics(szName, OBJ_BUTTON, clrNONE);
             ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_STATE, true);
             ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_BORDER_COLOR, clrBlack);
             ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_COLOR, clrBlack);
             ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_BGCOLOR, backColor);
             ObjectSetString(def_InfoTerminal.ID, szName, OBJPROP_FONT, def_FontName);
             ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_FONTSIZE, def_FontSize);
             ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
             ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_XDISTANCE, x);
          }
#undef def_FontSize
#undef def_FontName
```

Here we have two declarations that will only be used in this location, so we define and remove the definition as soon as it is no longer needed by the rest of the code. At this point we are already calling a function that creates objects to be used on the chart. The task is to create an object and then adjust some of its properties so that it builds the way you want. However, if you notice, we are using the button as if it were a window that will have read-only text. Perhaps it would be more appropriate to use **OBJ\_LABEL** or **OBJ\_EDIT** here. However, here we are talking only about demonstrating one of the ways to achieve the most suitable result. Therefore, we can use another object to put the data on the chart.

The great thing of this class is the two functions it contains. The first one is shown below. The other one will be discussed at the end of the article. Let's now see how this class creates the analysis presented in video 01, which uses the Fibonacci object. The code to create this object is shown below:

```
void CreateStudy(void)
   {
      const double FiboLevels[] = {0, 0.236, 0.382, 0.50, 0.618, 1, 1.618, 2};
      ENUM_LINE_STYLE ls;
      color cor;

      ObjectDelete(def_InfoTerminal.ID, def_ExpansionFibo);
      ObjectDelete(def_InfoTerminal.ID, "MOUSE_TB");
      ObjectDelete(def_InfoTerminal.ID, "MOUSE_TI");
      ObjectDelete(def_InfoTerminal.ID, "MOUSE_TT");
      (*mouse).CreateObjectGraphics(def_ExpansionFibo, OBJ_FIBO, clrNONE);
      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_HIDDEN, false);
      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_RAY_LEFT, false);
      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_LEVELS, ArraySize(FiboLevels));
      for (int c0 = 0, c1 = ArraySize(FiboLevels); c0 < c1; c0++)
      {
         ls = ((FiboLevels[c0] == 0) || (FiboLevels[c0] == 1) || (FiboLevels[c0] == 2)  ? STYLE_SOLID : STYLE_DASHDOT);
         ls = (FiboLevels[c0] == 0.5 ? STYLE_DOT : ls);
         switch (ls)
         {
            case STYLE_DOT    : cor = clrBlueViolet;  break;
            case STYLE_DASHDOT: cor = clrViolet;      break;
            default           : cor = clrIndigo;
         }
         ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_LEVELSTYLE, c0, ls);
         ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_LEVELCOLOR, c0, cor);
         ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_LEVELWIDTH, c0, 1);
         ObjectSetString(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_LEVELTEXT, c0, (string)NormalizeDouble(FiboLevels[c0] * 100, 2));
      }
      ObjectSetDouble(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_PRICE, 1, m_Info.MemPrice = def_InfoMousePos.Price);
      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_TIME, 1, m_Info.MemDT = def_InfoMousePos.dt);
      CreateBTNInfo(def_InfoMousePos.X, 50, 18, def_ExpansionBtn1, clrNONE);
      m_Info.ExecStudy = true;
      m_Info.ClearStudy = false;
   }
```

Although this code may seem complicated at first glance, it actually consists of three parts. In each of these parts we do something specific analyze using **OBJ\_FIBO**.

1. In the first part, we removed the "unwanted" objects created by the C\_Mouse class when receiving an event from the platform indicating that the user has started analyzing the symbol chart. When deleting these objects, be careful not to remove anything that is truly essential. So we can create a very special analyzoz by removing everything that we don't need to see in the study that we're going to create here. Please note that we have removed the object of the old analysis. This was done so that we could conduct the analysis based on specific criteria. The reason for this could also be that we want to use a key combination to create analytics based on **OBJ\_FIBO** variations. These variations can be **OBJ\_FIBOTIMES**, **OBJ\_FIBOFAN**, **OBJ\_FIBOARC**, **OBJ\_FIBOCHANNEL** and **OBJ\_EXPANSION**. They all follow the same principles, which are shown here.
2. In the second part we create and define the properties of the object. Here are some interesting points: It is at this stage that we tell the platform that the object will be visible in the list of objects. Here we specify which levels the object will contain. Well, we used static levels here, however you can use dynamic or other levels in your system. In this section. I will tell you what all the levels will be like, both in color and in the form of lines that will be used to construct them. We can make changes at our discretion to get the appropriate representation because when we do analytics, we want it to be understood quickly so that we can benefit from it.
3. And in the third and last part, we begin to construct the object directly on the chart, that is, we begin to plot. We'll also see what happens with variables. This is necessary so that during the function, which we will see later, everything is done correctly.

Basically, in this way we are going to create a study based on the already created and tested system, which is in the C\_Mouse class. In other words, we're not going to build something from scratch, but rather reuse and adapt what we already have in order to achieve something different. Everything will become clearer when we study the second procedure. Let's move on and look now at the class constructor and destructor. You can see them below:

```
C_StudyS2(C_Mouse *arg, color corP, color corN)
   {
      mouse = arg;
      ZeroMemory(m_Info);
      m_Info.corP = corP;
      m_Info.corN = corN;
   }
//+------------------------------------------------------------------+
~C_StudyS2()
   {
      ObjectsDeleteAll(def_InfoTerminal.ID, def_ExpansionPrefix);
   }
```

Look closely at these two functions. The goal is to use a system based on the second way. To use the inheritance model, you need to add the lines used in the first path topic. The same thing needs to be done in the last function in the class. This part enables interaction with the platform. Below is the complete code of the function, which actually allows you to create the analytics:

```
virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
   {
      double v1, v2;
      int w, h;
      string sz1;

      switch (id)
      {
         case CHARTEVENT_KEYDOWN:
            if (TerminalInfoInteger(TERMINAL_KEYSTATE_ESCAPE) && (m_Info.ExecStudy)) m_Info.ClearStudy = true;
            break;
         case CHARTEVENT_MOUSE_MOVE:
            if (mouse.GetInfoMouse().ExecStudy)
            {
               if (!m_Info.ExecStudy) CreateStudy();
               v1 = def_InfoMousePos.Price - m_Info.MemPrice;
               v2 = MathAbs(100.0 - ((m_Info.MemPrice / def_InfoMousePos.Price) * 100.0));
               sz1 = StringFormat(" %." + (string)def_InfoTerminal.nDigits + "f [ %d ] %02.02f%% ", MathAbs(v1), Bars(def_InfoTerminal.szSymbol, PERIOD_CURRENT, m_Info.MemDT, def_InfoMousePos.dt) - 1, v2);
               GetDimensionText(sz1, w, h);
               ObjectSetDouble(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_PRICE, 0, def_InfoMousePos.Price);
               ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_TIME, 0, def_InfoMousePos.dt);
               ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_COLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
               ObjectSetString(def_InfoTerminal.ID, def_ExpansionBtn1, OBJPROP_TEXT, sz1);
               ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn1, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
               ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn1, OBJPROP_XSIZE, w);
               ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn1, OBJPROP_YSIZE, h);
               ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn1, OBJPROP_XDISTANCE, def_InfoMousePos.X - w);
               ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn1, OBJPROP_YDISTANCE, def_InfoMousePos.Y - (v1 < 0 ? 1 : h));
            }else if (m_Info.ExecStudy)
            {
               ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionFibo, OBJPROP_COLOR, clrNONE);
               ObjectDelete(def_InfoTerminal.ID, def_ExpansionBtn1);
               if (m_Info.ClearStudy) ObjectDelete(def_InfoTerminal.ID, def_ExpansionFibo);
               m_Info.ExecStudy = false;
            }
            break;
         }
      }
```

This code is very interesting, isn't it? Pay attention that we have not considered mouse actions. We're only looking at what the C\_Mouse class is doing. While the C\_Mouse class indicates that we are doing analysis, this class will follow that direction, thus performing the analysis as directed by the C\_Mouse class. Once the C\_Mouse class is no longer used in the analysis, we will delete the object that was used to host the information text. If you press the ESC key during analysis, the analytical object will also be deleted. The size of the object used to display the text is calculated dynamically, meaning it can be larger or smaller depending on the case, and all this is controlled in this code. Here we also control colors and the placement of objects.

This code has an interesting part that deserves more explanation. Let's see it next. We need to understand what we represent and why these values are used.

```
v1 = def_InfoMousePos.Price - m_Info.MemPrice;
v2 = MathAbs(100.0 - ((m_Info.MemPrice / def_InfoMousePos.Price) * 100.0));
sz1 = StringFormat(" %." + (string)def_InfoTerminal.nDigits + "f [ %d ] %02.02f%% ", MathAbs(v1), Bars(def_InfoTerminal.szSymbol, PERIOD_CURRENT, m_Info.MemDT, def_InfoMousePos.dt) - 1, v2);
```

To understand these 3 lines that factor and format information for presentation, you need to see that the system dynamically adapts to the symbol on which the code is running. Some symbols may require 4 characters to represent values, while others may require 5. In some stock instruments we have only 2 characters. To make the system easily adaptable, we use the above code. It sounds strange, but in fact it is simply unusual.

First, we take into account the difference between the price at which the analysis began and the price line. This provides the value in points or in financial value; this is the offset value between the position where we started analysis and the current position where the mouse is. To represent it correctly, we need to know how many characters are needed. To do this, we use the following method. Using the percent symbol ( **%**), you can define the type of information that will be converted to a string. With the following format < **%.2f** \> we get a value containing two decimal places, if its is < **%.4f** >, we get a value containing 4 decimal places, and so on. But we need this to be defined in **runtime**.

The StringFormat function will create the appropriate format itself. I know this seems confusing, but once the value we calculated using the difference is placed, it will be placed exactly according to the format we created. This will give us the number of decimal places corresponding to the displayed value. To understand how this works in practice, you will need to use the same code on assets with a different number of characters to make it clearer. Another question is to find out in a simple way the number of bars from a given point.

Some platforms provide an indicator that counts the bars and shows it on the chart. We can easily create such an indicator. However, at the same time, more and more information will appear on the chart, which often makes it difficult to read, since it ends up being filled with a huge amount of information, most of which is often unnecessary. Using the MQL5 language in a slightly more exotic way, we can calculate how many bars are in the area of analysis and display this value in real time directly on the chart. Once the analysis is compete, only the information we need will remain on the chart.

In order to conduct such analysis, we use this function with these parameters. But be careful, if the analysis is performed in the area where there are no bars, then the indicated value will be -1, and if the analysis is made on one bar, the value will be zero. To change this, since the number of bars will refer to the number of bars present in the area under analysis, we simply delete this value of **-1** from factorization. Thus, the value will always correspond to the actual number of bars, including the bar from which the analysis began. This is why sometimes analysis comes up with a value of -1.

Since we also want to report the display a percentage of deviation, we use this calculation to obtain this percentage. To make the visualization more understandable, we use this format so that the percent symbol can be displayed along with the rest of the information.

### Conclusion

Here I have demonstrated a technique that can be very useful at different points in your professional programming life. I have shown that, contrary to what many think, it is not the platform itself that is limited, but the knowledge of those who say that the platform or language does not allow you to create different things. What I have explained here proves that, with common sense and creativity, the MetaTrader 5 platform can be made much more interesting and versatile. And you don't have to create crazy programs or anything like that. You can create simple, safe and reliable code. Use your creativity to modify existing code without deleting or adding a single line to the source code. So if at some point your code that you've been using for a while really comes in handy, you can continually and effortlessly add it to your robust code. This is why the concept of classes is used, where a code inheritance hierarchy is simply created.

There is no work that cannot be done. There is work that some people can't do. But this does not mean that the task cannot be performed.

In the attachment you will find the complete code that we have created throughout these articles. In the next part we will look at this system in more detail, but without the analytics code. Perhaps part of it will be built into the C\_Mouse class. And if this happens, I won't go into detail because all the code has been explained in previous articles. See you soon.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11372](https://www.mql5.com/pt/articles/11372)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11372.zip "Download all attachments in the single ZIP archive")

[Files\_-\_BOLSA.zip](https://www.mql5.com/en/articles/download/11372/files_-_bolsa.zip "Download Files_-_BOLSA.zip")(1358.24 KB)

[Files\_-\_FOREX.zip](https://www.mql5.com/en/articles/download/11372/files_-_forex.zip "Download Files_-_FOREX.zip")(3743.96 KB)

[Files\_-\_FUTUROS.zip](https://www.mql5.com/en/articles/download/11372/files_-_futuros.zip "Download Files_-_FUTUROS.zip")(11397.51 KB)

[Market\_Replay\_-\_30.zip](https://www.mql5.com/en/articles/download/11372/market_replay_-_30.zip "Download Market_Replay_-_30.zip")(60.95 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/463824)**
(1)


![Hilario Miguel Ofarril Gonzalez](https://c.mql5.com/avatar/avatar_na2.png)

**[Hilario Miguel Ofarril Gonzalez](https://www.mql5.com/en/users/hilariomiguelofarrilgonzalez)**
\|
4 Mar 2024 at 05:53

**MetaQuotes:**

Published article [Development of a repetition system (Part 30): Expert Advisor Project - Class C\_Mouse (IV)](https://www.mql5.com/es/articles/11372):

Author: [Daniel Jose](https://www.mql5.com/es/users/DJ_TLoG_831 "DJ_TLoG_831")

Very accurate and concrete


![Developing a Replay System (Part 31): Expert Advisor project — C_Mouse class (V)](https://c.mql5.com/2/59/sistema_de_Replay_logo.png)[Developing a Replay System (Part 31): Expert Advisor project — C\_Mouse class (V)](https://www.mql5.com/en/articles/11378)

We need a timer that can show how much time is left till the end of the replay/simulation run. This may seem at first glance to be a simple and quick solution. Many simply try to adapt and use the same system that the trading server uses. But there's one thing that many people don't consider when thinking about this solution: with replay, and even m ore with simulation, the clock works differently. All this complicates the creation of such a system.

![The Disagreement Problem: Diving Deeper into The Complexity Explainability in AI](https://c.mql5.com/2/72/The_Disagreement_Problem_Diving_Deeper_into_The_Complexity_Explainability_in_AI____LOGO.png)[The Disagreement Problem: Diving Deeper into The Complexity Explainability in AI](https://www.mql5.com/en/articles/13729)

In this article, we explore the challenge of understanding how AI works. AI models often make decisions in ways that are hard to explain, leading to what's known as the "disagreement problem". This issue is key to making AI more transparent and trustworthy.

![Neural networks made easy (Part 62): Using Decision Transformer in hierarchical models](https://c.mql5.com/2/59/Neural_networks_are_easy_0Part_62s_logo.png)[Neural networks made easy (Part 62): Using Decision Transformer in hierarchical models](https://www.mql5.com/en/articles/13674)

In recent articles, we have seen several options for using the Decision Transformer method. The method allows analyzing not only the current state, but also the trajectory of previous states and actions performed in them. In this article, we will focus on using this method in hierarchical models.

![Modified Grid-Hedge EA in MQL5 (Part III): Optimizing Simple Hedge Strategy (I)](https://c.mql5.com/2/72/Modified_Grid-Hedge_EA_in_MQL5_Part_III____LOGO.png)[Modified Grid-Hedge EA in MQL5 (Part III): Optimizing Simple Hedge Strategy (I)](https://www.mql5.com/en/articles/13972)

In this third part, we revisit the Simple Hedge and Simple Grid Expert Advisors (EAs) developed earlier. Our focus shifts to refining the Simple Hedge EA through mathematical analysis and a brute force approach, aiming for optimal strategy usage. This article delves deep into the mathematical optimization of the strategy, setting the stage for future exploration of coding-based optimization in later installments.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=orhcnltdfzyymolugtozscxfbpjaillm&ssn=1769157731307393320&ssn_dr=1&ssn_sr=0&fv_date=1769157731&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11372&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2030)%3A%20Expert%20Advisor%20project%20%E2%80%94%20C_Mouse%20class%20(IV)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915773201796317&fz_uniq=5062654501069956707&sv=2552)

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