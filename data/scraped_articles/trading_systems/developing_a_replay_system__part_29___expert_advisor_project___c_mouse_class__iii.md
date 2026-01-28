---
title: Developing a Replay System (Part 29): Expert Advisor project — C_Mouse class (III)
url: https://www.mql5.com/en/articles/11355
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:42:23.500573
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/11355&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062656536884455021)

MetaTrader 5 / Tester


### Introduction

In the previous article [Developing a Replay System (Part 28): Expert Advisor project — C\_Mouse class (II)](https://www.mql5.com/en/articles/11349), we looked at how to create more readable code. While this model is quite interesting for making a program more understandable, I think you'll notice that programming using this approach can take longer. And this is not because programming becomes confusing. Quite the opposite. The difficulty is that this method of making a program more readable has its limitations. One such limitation is the syntax of any programming language. Although syntax is designed to have a specific format and structure, the use of definitions, while helpful, limits us in other ways. However, I think it's appropriate to show this in real code from my point of view.

We'll keep the original syntax, but if you want to explore the code the way we show it, feel free to create any definitions you feel necessary and adapt the code to make it easier for you to understand. This will help you learn some pretty interesting techniques. This is how I learned to program in other languages. This is a labor-intensive, but useful process, since there are methods and even algorithms that are easier to implement in one language than in another. However, if you can read the code in the original language, then you you can do the things that others can't. You can think of it as a translation work and you are an interpreter between two different worlds. To be successful, our understanding must be much broader than that of the person who always communicates using the same signs, symbols and terms.

> _Broaden your mind, see outside the box, and the entire universe will open for you._

But let's get to what brought us to this article. Here we will look at how, **without changing the class and without using the inheritance system**, to extend the capabilities of the system in a controlled, safe and reliable way, regardless of ability. The task may seem simple at first, but it will provide a deeper understanding of how things work, far beyond what we get when we build the same method every time.

In today's article we will look at a way to expand the financial instrument analysis system. We'll use the C\_Mouse class along with what it inherits from the C\_Terminal class to create another analysis niche. However, we will do this in a rather interesting way: we will create a new class that will use the contents of the C\_Mouse class, but without directly inheriting from it. This new class may or may not then be added to the final code, depending on our goals. But regardless of this, we will learn how to create our own training model without violating the integrity of the previously created and tested code. This is the real purpose of this article.

### Setting the stage for expansion

Before we start programming the class that will not inherit the C\_Mouse class but will extend or rather modify its functionality, we need to adjust some details of the original C\_Mouse class. Not because there is any problem, but because we need to make some additions and a small change that will make any kind of expansion easier. This way, any change in the class functionality will be quite practical, because if something goes wrong, we can simply revert to using the original class without any problems. The changes will be few and simple yet important. First, we will add a new variable to the C\_Mouse class code.

```
class C_Mouse : public C_Terminal
{
   protected:
      enum eEventsMouse {ev_HideMouse, ev_ShowMouse};
      enum eBtnMouse {eKeyNull = 0x00, eClickLeft = 0x01, eClickRight = 0x02, eSHIFT_Press = 0x04, eCTRL_Press = 0x08, eClickMiddle = 0x10};
      struct st_Mouse
      {
         struct st00
         {
            int     X,
                    Y;
            double  Price;
            datetime dt;
         }Position;
         uint    ButtonStatus;
         bool    ExecStudy;
      };
```

This variable will allow us to extend or change the behavior of the C\_Mouse class without the need for inheritance or polymorphism. Although these are the most common methods, we will take a different approach. In fact, the method we will demonstrate allows us to apply this strategy in any class. The important part here is that everything s done without changing even a line of the source code of the class. Before implementing the mentioned changes aimed at enabling us to extend the capabilities, we need to add a simple line of code to the C\_Mouse class. Something simple.

```
inline void CreateObjectBase(const string szName, const ENUM_OBJECT obj, const color cor)
   {
      ObjectCreate(GetInfoTerminal().ID, szName, obj, 0, 0, 0);
      ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_TOOLTIP, "\n");
      ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BACK, false);
      ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_COLOR, cor);
      ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_ZORDER, -1);
   }
```

This particular line ensures that even if the mouse price line turns out to be a foreground line, it **will not receive** any event in case it overlaps any object that we are trying to click on. The price line will only capture click events if there are no objects in the mouse cursor's focus. It is important to note that adding this line does not block the generation of analytics even if the object is clicked, since this line of code does not allow the object to directly receive a click, but does not prevent the **CHARTEVENT\_MOUSE\_MOVE** event from being fired activated and captured by the C\_Mouse class.

**Important Note:** Previously, I had certain problems precisely because of the absence of this line of code. In the article [Making charts more interesting: Adding a background](https://www.mql5.com/en/articles/10215), there is a flaw which at that time I could not solve. No matter how much I tried, the problem persisted. We could solve the error that prevented access to objects present on the chart by simply adding the highlighted line to the object that was used to insert the background into the chart. I could have shared this advice earlier, but I wanted to somehow reward those who actually read the articles. Now you know how to solve the problem mentioned in the article.

Next we will make small changes to the following function:

```
virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
      {
         int w = 0;
         static double memPrice = 0;

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
               ChartXYToTimePrice(GetInfoTerminal().ID, m_Info.Data.Position.X = (int)lparam, m_Info.Data.Position.Y = (int)dparam, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
               ObjectMove(GetInfoTerminal().ID, def_NameObjectLineH, 0, 0, m_Info.Data.Position.Price = AdjustPrice(m_Info.Data.Position.Price));
               m_Info.Data.Position.dt = AdjustTime(m_Info.Data.Position.dt);
               ChartTimePriceToXY(GetInfoTerminal().ID, w, m_Info.Data.Position.dt, m_Info.Data.Position.Price, m_Info.Data.Position.X, m_Info.Data.Position.Y);
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
               m_Info.Data.ExecStudy = m_Info.Study == eStudyExecute;
               break;
            case CHARTEVENT_OBJECT_DELETE:
               if (sparam == def_NameObjectLineH) CreateLineH();
               break;
         }
      }
```

These changes do not benefit the C\_Mouse class specifically, but rather the entire program that will be built based on the C\_Mouse class. We may not notice any difference in the code between the previous and current articles, which is due to the subtlety and specificity of the change. Actually, nothing is changed in the code. These changes bring various benefits in terms of usability and customization options. It may not be easy to notice the difference, but the three new lines that we have added to the code help a lot. Let's see what each of them does:

1. This line will adjust the time value so that we can actually use any object on the chart using not only screen coordinates (X and Y) but also financial instrument coordinates (Price and Time). I have been looking for the answer to this question for a very long time. Consider also that working in asset coordinates is much more interesting than sometimes using screen coordinates. The level of freedom this gives us is obvious. You can also see that we are making the call, and for practical reasons it is in the C\_Terminal class.
2. The added call is ChartTimePriceToXY, to convert price coordinates to screen coordinates.
3. And the last point was just this. We indicate whether the C\_Mouse class is in the study mode. **_To avoid confusion, pay attention to the syntax_**.

These are all changes made to the C\_Mouse class. However, as already mentioned, the C\_Terminal class now includes a new function. Let's look at this function to understand what it is. The added feature is shown below:

```
inline datetime AdjustTime(const datetime arg) const
   {
      int nSeconds= PeriodSeconds();
      datetime dt = iTime(m_Infos.szSymbol, PERIOD_CURRENT, 0);

      return (dt < arg ? ((datetime)(arg / nSeconds) * nSeconds) : iTime(m_Infos.szSymbol, PERIOD_CURRENT, Bars(m_Infos.szSymbol, PERIOD_CURRENT, arg, dt)));
   }
```

If this function seems very strange and confusing to you, don't worry. Although it may seem confusing and pointless at first, it actually offers something wonderful and interesting. To understand the magic of this function, we need to understand one thing about type conversion, as well as a few additional details. First, let's look at type conversion.

When this function is called, it receives **date and time value as parameter.** It is important to consider this value not only as a date and time, but as a **ulong** value, because that's what it really is: an 8-byte value. This is the first thing. The second one is as follows: since datetime is a ulong value, time and date information in a variable is compressed in a rather specific way. What we're interested in are the least significant bits ( **LSB**), where the values of second, minute, hour, day, month and year are ordered from least significant bit to most significant bit.

Note that **nSeconds** contains the value in seconds of the period used in the chart. This is determined by the [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds) function, which provides this information. Now the **dt** variable contains the creation value of the last bar present on the chart. This fact is very important, since if the **dt** value is less that the value of the call, this will indicate at what point of time we are, in this case we will be in the future. Thus, the position of time relative to the screen coordinates (X and Y) indicates where the future bar is or will be created. At the moment it is not possible, using the [iTime](https://www.mql5.com/en/docs/series/itime) function, to find out where it will be, since the chart to this point has not yet been built. However, even then, we need to know where it will actually be, especially if we are doing chart analysis related to the future.

To find out the screen coordinates (X and Y), we will use the fact that **datetime** is an **ulong** value. By dividing this value by the number of seconds **nSeconds**, we will get a **double** value. Next comes an important point: if we multiply this fractional value by **nSeconds**, we will get the original value. For example, if we divide 10 by 3 and multiply the result of the division by 3, we get 10 again. However, when we do a type conversion (and this is the key), we convert the **double** value into **ulong** or, better yet, into a value of the **datetime** type. There is no fractional part here. Therefore, multiplying the value by **nSeconds**, we will get the future value already corrected. This is the most interesting part. However, there is a problem.

This problem appears when we look into the past, especially for continuous series, that is, when there are no gaps. This approach is not appropriate for analyzing the past, mainly for assets that have such gaps, which is typical for assets traded during certain hours or days. This refers to stock market instruments where bars only form during a certain time window and the market closes outside that period. To deal with this situation, we use a slightly different setup for an attempt to determine which column the specified value is in. We do this by calculating the number of bars between the current bar and the bar at a specified time, which gives us a value that we can use as an offset to capture the exact point in time. This way we will also be able to make adjustments in the past regardless of what happened.

You may think that this function is completely unnecessary. Why bother developing it? However, with this function you can convert symbol coordinates (Price and Time) into screen coordinates (X and Y). Therefore, we can use any kind of graphical object, not limited to screen coordinates or resource objects. We will be able to convert one type of coordinates to another, and for this we will use calls: [ChartXYToTimePrice](https://www.mql5.com/en/docs/chart_operations/chartxytotimeprice) (to convert screen coordinates to asset coordinates) and [ChartTimePriceToXY](https://www.mql5.com/en/docs/chart_operations/charttimepricetoxy) (to convert asset coordinates to screen coordinates). However, in some types of analysis we need to have as accurate information as possible. When we want a particular bar to be used as an indicator point for something, this conversion becomes necessary. In addition, this gives us very interesting information, which we will look at later.

### Create the C\_Studys class

After improving the C\_Mouse class, we can focus on creating a class designed to create a completely new framework fr our analysis. We will not use inheritance or polymorphism to create this new class. Instead, we will change, or better said, add new objects to the price line. That's what we will do in this article. In the next one, we will look at how to change the analysis. But we will do all this without changing the code of the C\_Mouse class. Well, actually, it would be easier to achieve this using inheritance or polymorphism. There are other methods that can achieve the same result because they provide flexibility without causing significant disruption if there are defects in the new code.

This approach allows us to remove problematic code, fix bugs, and introduce it again without requiring changes to previously tested code. Remember: bug fixes can often make code incompatible with existing class structure, making it difficult to use inheritance or polymorphism to enable new additions. Therefore, mastering these alternative techniques is important, especially if we want to implement new functionality in an already completed and tested program without the need for significant restructuring of existing code.

The first step is to create a new file, although it is also possible to create several different files with their own hierarchy between the classes involved. This will not be a problem since this file and files will not integrate the parent class hierarchy. In this way, we can establish an independent hierarchy that can be changed or improved in various ways as needed. This flexibility comes from the fact that it is not directly related to the main development hierarchy and acts almost like a separate project.

We'll start with a simpler system and will explore more complex approaches in the future. The file runs like this:

```
#include "..\C_Mouse.mqh"
#include "..\..\..\Service Graphics\Support\Interprocess.mqh"
//+------------------------------------------------------------------+
#define def_ExpansionPrefix "Expansion1_"
#define def_ExpansionBtn1 def_ExpansionPrefix + "B1"
#define def_ExpansionBtn2 def_ExpansionPrefix + "B2"
#define def_ExpansionBtn3 def_ExpansionPrefix + "B3"
//+------------------------------------------------------------------+
#define def_InfoTerminal (*mouse).GetInfoTerminal()
#define def_InfoMousePos (*mouse).GetInfoMouse().Position
//+------------------------------------------------------------------+
```

From this introduction to the code, we understand that the process will be intensive. Note the inclusion of two header files located in different places in this file. This part has already been discussed previously. A little later we will mention some of the objects that we will use. This is important to avoid confusion when further accessing the desired objects. We also defined a kind of **alias** to facilitate coding since we will use something very similar to pointers, one of the most powerful and at the same time riskiest resources available. But because of the way we plan to program, the risk associated with this resource will be quite manageable, which will allow us to use it in very interesting ways.

This type of definition ( **alias**) is quite common when we want to access certain things but don't want to risk typing something wrong while writing code. This is always a very interesting resource.

Next comes the following class code:

```
class C_Studys
{
   protected:
   private :
//+------------------------------------------------------------------+
      enum eStatusMarket {eCloseMarket, eAuction, eInTrading, eInReplay};
//+------------------------------------------------------------------+
      C_Mouse *mouse;
//+------------------------------------------------------------------+
      struct st00
      {
         eStatusMarket   Status;
         MqlRates        Rate;
         string          szInfo;
         color           corP,
                         corN;
         int             HeightText;
      }m_Info;
```

At this stage we focus solely on the private variables of the class. Look at one of them, which acts as a sort of pointer to the C\_Mouse class. Pointers are one of the most powerful resources in programming, but they require special attention because of the problems that can arise when using them. Therefore, it is very important to use pointers carefully, even if in MQL5 they do not have the same characteristics as in C/C++. Anyway, you should be careful when using them. **Never underestimate pointers**. The rest of the code is nothing special yet.

The first of the class functions is presented below:

```
const datetime GetBarTime(void)
   {
      datetime dt = TimeCurrent();

      if (m_Info.Rate.time <= dt)
         m_Info.Rate.time = iTime(def_InfoTerminal.szSymbol, PERIOD_CURRENT, 0) + PeriodSeconds();

      return m_Info.Rate.time - dt;
   }
```

It is a fairly simple function that calculates the time remaining until a new bar appears. The calculation itself only occurs at that specific point, however each time the test detects that a bar's time limit has been reached, we proceed with the necessary reading and adjustment to move the calculation to the next bar. Thus, this iTime function call occurs only once for each bar created, and not with each interaction or method call. The next function creates objects in a standard way.

```
inline void CreateObjectBase(const string szName, const ENUM_OBJECT obj)
   {
      ObjectCreate(def_InfoTerminal.ID, szName, obj, 0, 0, 0);
      ObjectSetString(def_InfoTerminal.ID, szName, OBJPROP_TOOLTIP, "\n");
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_BACK, false);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_ZORDER, -1);
   }
```

We're essentially going through the same process as the function that's in C\_Mouse, which suggests that we might soon consider combining the two into an even more general function. Immediately after this, we present a procedure that may be of interest to some to test.

```
int CreateBTNInfo(const string szExample, int x, string szName, color backColor, string szFontName, int FontSize)
   {
      int w;

      CreateObjectBase(szName, OBJ_BUTTON);
      TextGetSize(szExample, w, m_Info.HeightText);
      m_Info.HeightText += 5;
      w += 5;
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_STATE, true);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_BORDER_COLOR, clrBlack);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_COLOR, clrBlack);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_BGCOLOR, backColor);
      ObjectSetString(def_InfoTerminal.ID, szName, OBJPROP_FONT, szFontName);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_FONTSIZE, FontSize);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_XSIZE, w);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_YSIZE, m_Info.HeightText);
      ObjectSetInteger(def_InfoTerminal.ID, szName, OBJPROP_XDISTANCE, x);

      return w;
   }
```

This method allows you to adapt the size of an object depending on the size and type of font used, as well as the longest text that will be printed inside the object. This is achieved by using the [TextGetSize](https://www.mql5.com/en/docs/objects/textgetsize) function, which, based on the information provided, gives us an estimate of the size of the text. However, to make the text look better inside the object, we will slightly increase its size. This creates a small space between the text and the object's boundaries.

Next we have a function that displays information on our chart.

```
void Draw(void)
   {
      double v1;

      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn1, OBJPROP_YDISTANCE, def_InfoMousePos.Y <= 0 ? UINT_MAX : def_InfoMousePos.Y - m_Info.HeightText);
      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn2, OBJPROP_YDISTANCE, def_InfoMousePos.Y <= 0 ? UINT_MAX : def_InfoMousePos.Y - 1);
      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn3, OBJPROP_YDISTANCE, def_InfoMousePos.Y <= 0 ? UINT_MAX : def_InfoMousePos.Y - 1);
      ObjectSetString(def_InfoTerminal.ID, def_ExpansionBtn1, OBJPROP_TEXT, m_Info.szInfo);
      v1 = NormalizeDouble(100.0 - ((m_Info.Rate.close / def_InfoMousePos.Price) * 100.0), 2);
      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn2, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
      ObjectSetString(def_InfoTerminal.ID, def_ExpansionBtn2, OBJPROP_TEXT, (string)MathAbs(v1) + "%");
      v1 = NormalizeDouble(100.0 - ((m_Info.Rate.close / iClose(def_InfoTerminal.szSymbol, PERIOD_D1, 0)) * 100.0), 2);
      ObjectSetInteger(def_InfoTerminal.ID, def_ExpansionBtn3, OBJPROP_BGCOLOR, (v1 < 0 ? m_Info.corN : m_Info.corP));
      ObjectSetString(def_InfoTerminal.ID, def_ExpansionBtn3, OBJPROP_TEXT, (string)MathAbs(v1) + "%");
   }
```

This is precisely generates a type of study, which will always be present and will accompany the price line. In this way, we will be able to easily understand certain things. The type of information and what will be shown depends on what you need. Here for demonstration purposes we will have three types of information. So let's see what will be provided.

We can receive different information depending on how the program interacts with the platform and the trading server. However, we receive the following information:

- Remaining time until the next bar appears on the chart;
- Information that the market is closed;
- Information that we are dealing with replay;
- Information that the asset is at auction;
- And (in extremely rare cases) an error message.

Please note that what we are doing is not unusual. This only serves to demonstrate the working technique that we will use. Next comes the destuctor.

```
~C_Studys()
   {
      ObjectsDeleteAll(def_InfoTerminal.ID, def_ExpansionPrefix);
   }
```

The purpose of the above code is to indicate to the platform that we want to remove elements created by the class. Note that we don't care if this triggers events in the platform, since we don't intend to recreate such objects if they are deleted.

Here comes the next function:

```
void Update(void)
   {
      switch (m_Info.Status)
      {
         case eCloseMarket: m_Info.szInfo = "Closed Market";                         break;
         case eAuction   : m_Info.szInfo = "Auction";                                break;
         case eInTrading : m_Info.szInfo = TimeToString(GetBarTime(), TIME_SECONDS); break;
         case eInReplay  : m_Info.szInfo = "In Replay";                              break;
         default         : m_Info.szInfo = "ERROR";
      }
      Draw();
   }
```

The important thing about this function is that it does not work in isolation. We have another function with a very similar name, presented just below:

```
void Update(const MqlBookInfo &book[])
   {
      m_Info.Status = (ArraySize(book) == 0 ? eCloseMarket : (def_InfoTerminal.szSymbol == def_SymbolReplay ? eInReplay : eInTrading));
      for (int c0 = 0; (c0 < ArraySize(book)) && (m_Info.Status != eAuction); c0++)
         if ((book[c0].type == BOOK_TYPE_BUY_MARKET) || (book[c0].type == BOOK_TYPE_SELL_MARKET)) m_Info.Status = eAuction;
      this.Update();
   }
```

Why do we have two functions named Update? Is it possible? Yes, we can declare functions with the same name. This phenomenon is known as overload. Although the names are identical, to the compiler the names of both functions are different. This is because of parameters. In this way, you can overload functions and procedures, but the only rule is that the parameters must be different. We must not forget that in the second function we call the first one using a method whose parameters are not required. This practice is quite common when programming methods that become overloaded and want to create something that will make debugging easier.

Now we come to an interesting point: how to find out whether an asset is at auction or not? Pricing information is usually provided in the order book. However, it may happen that the asset's order book will show one of these specific values, and when this happens, it will mean that the asset is at auction. Pay attention to this as it could be an interesting resource to add to your automated EA. I already mentioned this in another series of article: [Creating an EA that trades automatically (Part 14): Automation (VI)](https://www.mql5.com/en/articles/11318). But I did not go into detail about how to understand whether an asset is up for auction or not. Our goal was not to provide detailed coverage of all aspects of the trading process using an automated EA.

The following function is used to respond to some platform events:

```
virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
   {
      switch (id)
      {
         case CHARTEVENT_MOUSE_MOVE:
            Draw();
            break;
      }
   }
```

Simple but effective because the objects must follow the mouse and the mouse analysis is handled by the C\_Mouse class. Again, we modify the work of the C\_Mouse class without using inheritance or polymorphism. So the job of adjusting and correcting the positioning of the mouse is the responsibility of the C\_Mouse class. We will only use the data.

We have seen almost the entire code. But we need to look at where the magic actually happens. Let's start with the EA code, which is shown in full below:

```
//+------------------------------------------------------------------+
#include <Market Replay\System EA\Auxiliar\C_Mouse.mqh>
#include <Market Replay\System EA\Auxiliar\Study\C_Studys.mqh>
//+------------------------------------------------------------------+
input group "Mouse";
input color     user00 = clrBlack;      //Price Line
input color     user01 = clrPaleGreen;  //Positive Study
input color     user02 = clrLightCoral; //Negative Study
//+------------------------------------------------------------------+
C_Mouse *mouse = NULL;
C_Studys *extra = NULL;
//+------------------------------------------------------------------+
int OnInit()
{
   mouse = new C_Mouse(user00, user01, user02);
   extra = new C_Studys(mouse, user01, user02);

   OnBookEvent(_Symbol);
   EventSetMillisecondTimer(500);

   return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   MarketBookRelease(_Symbol);
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

Please note that we provide this EA with the ability to trade both real and simulated assets using the same tools. This is possible thanks to this type of verification. Currently and at this stage of development we will have different information depending on what the EA recognizes on the chart.

We pass to the class constructor the pointer created during initialization of the C\_Mouse class. This way, the C\_Studys class does not need to directly inherit from the C\_Mouse class to use its contents, eliminating the need to initialize C\_Mouse inside C\_Studys. This modeling technique is useful when we want to exchange information or interact with elements without resorting to inheritance or polymorphism. If we need to remove, adapt, or create new functionality of the C\_Studys class, which is somehow related to the C\_Mouse class, this can be done easily, without the need to change the C\_Mouse class.

The big advantage of this model is the possibility of parallel development of components. In the event that the C\_Studys class is not the part of the final project, it you simply remove the code from it. This C\_Studys becomes a parallel class, independent of the main class system of the final code.

Now let's analyze the constructor code to understand how this information transfer is implemented.

```
C_Studys(C_Mouse *arg, color corP, color corN)
   {
#define def_FontName "Lucida Console"
#define def_FontSize 10
      int x;

      mouse = arg;
      ZeroMemory(m_Info);
      m_Info.Status = eCloseMarket;
      m_Info.Rate.close = iClose(def_InfoTerminal.szSymbol, PERIOD_D1, ((def_InfoTerminal.szSymbol == def_SymbolReplay) || (macroGetDate(TimeCurrent()) != macroGetDate(iTime(def_InfoTerminal.szSymbol, PERIOD_D1, 0))) ? 0 : 1));
      m_Info.corP = corP;
      m_Info.corN = corN;
      TextSetFont(def_FontName, -10 * def_FontSize, FW_NORMAL);
      CreateBTNInfo("Closed Market", 2, def_ExpansionBtn1, clrPaleTurquoise, def_FontName, def_FontSize);
      x = CreateBTNInfo("99.99%", 2, def_ExpansionBtn2, clrNONE, def_FontName, def_FontSize);
      CreateBTNInfo("99.99%", x + 5, def_ExpansionBtn3, clrNONE, def_FontName, def_FontSize);
      Draw();
#undef def_FontSize
#undef def_FontName
   }
```

As you can see, this code has some quirks, including a somewhat unusual parameter: a pointer to a class. This demonstrates how the elements come together to form the system we need to design. However, there is another interesting feature that may seem strange at first glance: the [TextSetFont](https://www.mql5.com/en/docs/objects/textsetfont) function. It is very important to adjust the sizes of objects according to the type of information we are going to display. Note that we're doing factorization here. Why does it use a negative number? To be clear, let's look at the explanation given in the documentation:

_Font size is determined using positive or negative values. This fact determines the dependence of the text size on the operating system settings (size scale)._

- _**If the size is positive**, it is converted into device physical units (pixels) when displaying the logical font as a physical one. The size corresponds to the height of symbol cells from available fonts. It is not recommended in cases of shared usage of texts displayed using the [TextOut()](https://www.mql5.com/en/docs/objects/textout) function and texts displayed using the [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) ("Text label") graphical object._
- _**If the size is negative**, it is assumed to be set in tenths of a logical point (the value -350 is equal to 35 logical points) and is divided by 10. The resulting value is converted into physical units of the device (pixels) and corresponds to the absolute value of the character height from available fonts. Multiply the font size defined in the object properties by -10 to make the screen text size similar to the [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) object size._

For this reason, factorization is performed this way. If we don't set this parameter here, we will have problems using the TextGetSize function, which is used in the object creation function. This happens because the font used or its dimensions may not exactly match what we intend to use.

### Conclusion

Be sure to test the app attached. It is advisable to conduct experiments both in replay/simulation mode and on an account running on the market (DEMO or REAL),

to gain a broad understanding. But I will **NOT** modify a single line of code in the main class. I promise.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11355](https://www.mql5.com/pt/articles/11355)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11355.zip "Download all attachments in the single ZIP archive")

[Files\_-\_FUTUROS.zip](https://www.mql5.com/en/articles/download/11355/files_-_futuros.zip "Download Files_-_FUTUROS.zip")(11397.51 KB)

[Files\_-\_FOREX.zip](https://www.mql5.com/en/articles/download/11355/files_-_forex.zip "Download Files_-_FOREX.zip")(3743.96 KB)

[Files\_-\_BOLSA.zip](https://www.mql5.com/en/articles/download/11355/files_-_bolsa.zip "Download Files_-_BOLSA.zip")(1358.24 KB)

[Market\_Replay\_-\_29.zip](https://www.mql5.com/en/articles/download/11355/market_replay_-_29.zip "Download Market_Replay_-_29.zip")(56.76 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/463673)**

![Understanding Programming Paradigms (Part 2): An Object-Oriented Approach to Developing a Price Action Expert Advisor](https://c.mql5.com/2/71/MQL5_Article-02_Artwork_thumbnail_WhiteBG.png)[Understanding Programming Paradigms (Part 2): An Object-Oriented Approach to Developing a Price Action Expert Advisor](https://www.mql5.com/en/articles/14161)

Learn about the object-oriented programming paradigm and its application in MQL5 code. This second article goes deeper into the specifics of object-oriented programming, offering hands-on experience through a practical example. You'll learn how to convert our earlier developed procedural price action expert advisor using the EMA indicator and candlestick price data to object-oriented code.

![Population optimization algorithms: Charged System Search (CSS) algorithm](https://c.mql5.com/2/59/Charged_System_Search_CSS__logo.png)[Population optimization algorithms: Charged System Search (CSS) algorithm](https://www.mql5.com/en/articles/13662)

In this article, we will consider another optimization algorithm inspired by inanimate nature - Charged System Search (CSS) algorithm. The purpose of this article is to present a new optimization algorithm based on the principles of physics and mechanics.

![Modified Grid-Hedge EA in MQL5 (Part III): Optimizing Simple Hedge Strategy (I)](https://c.mql5.com/2/72/Modified_Grid-Hedge_EA_in_MQL5_Part_III____LOGO.png)[Modified Grid-Hedge EA in MQL5 (Part III): Optimizing Simple Hedge Strategy (I)](https://www.mql5.com/en/articles/13972)

In this third part, we revisit the Simple Hedge and Simple Grid Expert Advisors (EAs) developed earlier. Our focus shifts to refining the Simple Hedge EA through mathematical analysis and a brute force approach, aiming for optimal strategy usage. This article delves deep into the mathematical optimization of the strategy, setting the stage for future exploration of coding-based optimization in later installments.

![Deep Learning GRU model with Python to ONNX  with EA, and GRU vs LSTM models](https://c.mql5.com/2/70/Deep_Learning_Forecast_and_ordering_with_Python_and_MetaTrader5_python_packag___LOGOe.png)[Deep Learning GRU model with Python to ONNX with EA, and GRU vs LSTM models](https://www.mql5.com/en/articles/14113)

We will guide you through the entire process of DL with python to make a GRU ONNX model, culminating in the creation of an Expert Advisor (EA) designed for trading, and subsequently comparing GRU model with LSTM model.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/11355&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062656536884455021)

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