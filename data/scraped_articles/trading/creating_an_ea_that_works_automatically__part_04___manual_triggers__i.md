---
title: Creating an EA that works automatically (Part 04): Manual triggers (I)
url: https://www.mql5.com/en/articles/11232
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:09:42.517143
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/11232&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069177297967186228)

MetaTrader 5 / Trading


### Introduction

In the previous article " [Creating an EA that works automatically (Part 03): New functions](https://www.mql5.com/en/articles/11226)" we finished covering the orders system. If you haven't read it or do not completely understand its contents, I suggest that you go back to that article. Here we will no longer discuss the order system. We will proceed to other things, in particular, to **triggers**.

The trigger system is probably the most difficult part. It creates confusion, doubts and problems, because there is no 100% error-free method. All methods have their own drawbacks and issues. Some of them have a higher probability of errors, while others have a lower probability. But always remember the following rule: **don't rely on triggers whatever they are, since they can fail when you least expect**. If the trigger system fails, you can miss a good opportunity or have big losses, therefore you should always be very careful with triggers.

But let's look at some details of this system. First, what is the purpose of the system? The trigger system serves a number of purposes, including indicating when to breakeven or when to trigger a trailing stop move. These are the simplest examples. The trigger system can also indicate when the EA should open a position or when it should close it.

All these events happen completely automatically: the trader does not need to do anything. We just instruct the EA which trigger should activate the event. Then human participation is not required, except for the case when the EA enters a continuous loop, because then the trader will have to deactivate it immediately.

To better understand these idea and concepts, we will need to program a manual EA. But here we will do something completely different from what we usually do as a manual EA. In the EA which we are using as an example in this series, we will add a possibility to place pending orders or to send order to open a market position. Since the EA is for demonstration and learning purposes, I advise everyone who thinks about using it to do so on a DEMO account. Do not use the EA on a real account since there is a risk that it will get stuck or will run in a crazy way.

### How to send orders and open market positions

In my opinion, the best way to trade on a chart is to use the mouse in combination with the keyboard. This is good for placing pending orders, while market orders can be placed using only the keyboard. But the only question is how to implement the code that would not be so complicated that it would be hard to explain, since the system for submitting orders using the mouse and keyboard can become quite difficult to develop.

After thinking about it for a while, I decided to come up with an ideal system to use in the article, something quite simple. So, don't use this system on a real account. It's just to help me explain a few things about triggers so you can understand how a trigger is implemented and how it should be handled.

The code is the simplest one you can build, but still it is sufficient for the rest of this series of articles. So, let's see how the code is implemented.

First, I have added a header file named C\_Mouse.mqh. The code in the file starts as follows:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_MouseName "MOUSE_H"
//+------------------------------------------------------------------+
#define def_BtnLeftClick(A)     ((A & 0x01) == 0x01)
#define def_SHIFT_Press(A)      ((A & 0x04) == 0x04)
#define def_CTRL_Press(A)       ((A & 0x08) == 0x08)
//+------------------------------------------------------------------+
```

It sets the name of the graphical object we are creating - the horizontal line to indicate the price position where the mouse pointer is located on the chart. We have also created three definitions to have an easier access to some information that a mouse event will generate. This is the way to test click and keypress events. We will use these access definitions later in the WA, while the definition of the object name will be completed at the end of the file using the following line:

```
//+------------------------------------------------------------------+
#undef def_MouseName
//+------------------------------------------------------------------+
```

Adding this line to the end of the header file ensures that this definition does not leak or appear anywhere else in the code. If you try to use this definition without re-declaring it elsewhere than in this header file, the compiler will return a relevant warning.

Now let's start writing the code of the class responsible for operations with the mouse. It starts as follows:

```
class C_Mouse
{
        private :
                struct st00
                {
                        long    Id;
                        color   Cor;
                        double  PointPerTick,
                                Price;
                        uint    BtnStatus;
                }m_Infos;
//+------------------------------------------------------------------+
                void CreateLineH(void)
                        {
                                ObjectCreate(m_Infos.Id, def_MouseName, OBJ_HLINE, 0, 0, 0);
                                ObjectSetString(m_Infos.Id, def_MouseName, OBJPROP_TOOLTIP, "\n");
                                ObjectSetInteger(m_Infos.Id, def_MouseName, OBJPROP_BACK, false);
                                ObjectSetInteger(m_Infos.Id, def_MouseName, OBJPROP_COLOR, m_Infos.Cor);
                        }
//+------------------------------------------------------------------+
inline double AdjustPrice(const double value)
                        {
                                return MathRound(value / m_Infos.PointPerTick) * m_Infos.PointPerTick;
                        }
//+------------------------------------------------------------------+
```

Don't be scared by this code, this part is just a small example. All we do is declare a data structure, which will contain some global variables within the class, but they will be private to the class. After that we use the procedure that will create the chart object in the MetaTrader 5 platform. This object will be a horizontal line, and we will set up some of its properties and move on..

As for this function, you have already seen it in the previous article. If not, read the previous articles in the series first, as they are also important for what we are going to do.

Now let's look at the procedures that will be public in our C\_Mouse class. Let's start with the class constructor:

```
                C_Mouse(const color cor)
                        {
                                m_Infos.Id = ChartID();
                                m_Infos.Cor = cor;
                                m_Infos.PointPerTick = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
                                ChartSetInteger(m_Infos.Id, CHART_EVENT_MOUSE_MOVE, true);
                                ChartSetInteger(m_Infos.Id, CHART_EVENT_OBJECT_DELETE, true);
                                CreateLineH();
                        }
```

It has two lines which may seem strange to you if you are not very familiar with MQL5. So, I will explain them briefly. This line tells the MetaTrader 5 platform that our code, the EA, wants to receive mouse events. There is another type of events also generated by the mouse, which occurs when using the scroll. But we won't use it because this event type if quite enough.

This line will inform MetaTrader 5 that we want to know of any object has been removed from the chart. If this happens, this will generate the event that informs us about which element has been removed from the chart. This type of events can be useful when we want to save certain critical elements for the perfect execution of the task. Then, if the user accidentally deletes any critical element, MetaTrader 5 informs the code (in this case the EA) that the element has been removed so we can recreate it. Now let's look at the class destructor code:

```
                ~C_Mouse()
                        {
                                ChartSetInteger(m_Infos.Id, CHART_EVENT_OBJECT_DELETE, false);
                                ObjectDelete(m_Infos.Id, def_MouseName);
                        }
```

We do two things in this code. First, we inform MetaTrader 5 that we no longer want to receive notifications if any element is deleted from the chart. It is important to do this before we continue, as if we try to remove an element, MetaTrader 5 will generate an event informing that something has been removed from the chart. This is exactly what we are going to do next: we delete the line we created to show what price range the mouse pointer is in.

The next thing we need to do is create a way to read the data inside the class so that we know the mouse position and the state of its buttons. For this purpose, we will use the following function:

```
const void GetStatus(double &Price, uint &BtnStatus) const
                        {
                                Price = m_Infos.Price;
                                BtnStatus = m_Infos.BtnStatus;
                        }
```

I would like to explain something which is very common among people with less experience in object oriented programming. When using classes, we must not allow any code outside the class to have direct access to the class variables or procedures and we must always check what will be entering the class.

Using and allowing variables or procedures in a class to get a value that has not been properly checked is a serious mistake, as it will lead to problems in the long run. At some point, you can modify in in the code the value of a critical variable used in the class, and when it comes to using it, it can put the hole code at risk. Besides it is extremely complicated to deal with this type of situation and to fix the problem later.

So, always follow this practice to be a good programmer: If you're going to read the contents of a class, allow it to be read, not modified; this happens very often, especially when using a pointer. You request to read a variable and return a pointer. At this moment your code is at risk because when you use a pointer, you will know where to write in memory, and this is very dangerous. See how game _cheats_ are created (little programs to cheat in electronic games). You simply write to a memory location using a pointer.

Be very careful when using pointers. When in doubt, always prefer to use a function rather than a procedure. The difference between the two is that the function usually returns a value, but still be careful not to return a pointer.

But if you have no other option, then do the same as I showed above. This may look like excessive care, but when starting the declaration of a procedure or a function with the reserved word **const** guarantees that the caller can never change the value in the variables, since the compiler will treat any value as a constant which cannot be changed.

Completing the declaration of a procedure or function with the reserved word **const** guarantees that you as a programmer will not accidentally or unknowingly change any value within the procedure or the function. This is probably the best programming practice in existence, although it may seem strange to many. It avoids some programming mistakes, especially when we do it in OOP (Object Oriented Programming), but we will come back to this a little later with a clearer example that will help you to understand this problem of completing a declaration with the word **const**.

The last procedure we need is shown below:

```
                void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
                        {
                                int w;
                                datetime dt;

                                switch (id)
                                        {
                                                case CHARTEVENT_OBJECT_DELETE:
                                                        if (sparam == def_MouseName) CreateLineH();
                                                        break;
                                                case CHARTEVENT_MOUSE_MOVE:
                                                        ChartXYToTimePrice(m_Infos.Id, (int)lparam, (int)dparam, w, dt, m_Infos.Price);
                                                        ObjectMove(m_Infos.Id, def_MouseName, 0, 0, m_Infos.Price = AdjustPrice(m_Infos.Price));
                                                        m_Infos.BtnStatus = (uint)sparam;
                                                        ChartRedraw();
                                                        break;
                                        }
                        }
```

It's very interesting because many programmers will want to put the code inside the [OnChartEvent](https://www.mql5.com/en/docs/event_handlers/onchartevent) message handling event. But I like to handle events inside the object class. Often, this helps not to forget to handle something in particular or to handle it in a special way, which often will make you check whether the code is working or not. By doing this handling inside the class, we have more confidence in the long run. Because when we use the class, we know that the class's message handling system is working as expected. This greatly speeds up the creation of new programs. This is also the path to reuse: program once and use always.

Let's see what happens in this messaging system. Here we are replicating exactly the data that will be received by the OnChartEvent event handler, which is done on purpose, as we want to handle these events here, within the class.

Once this is defined, let's determine which events we will handle. The first is the object removal event, which is generated because we previously told MetaTrader 5 that we want to be informed in case any object is removed from the chart. If the removed object was the mouse line, the object will be recreated immediately.

The next event we want to deal with here, within the class, is the mouse movement events. Again, this event will be generated because we have told MetaTrader 5 that we want to know what is happening with the mouse. Here we simply convert the values provided by MetaTrader 5 into time and price values. After that we will position the horizontal line object at the price position so that it is at the correct point. Please note that the point it not in the screen coordinate system but in the price chart. From there we save the value of the buttons and force immediate redrawing of the chart.

It is important that this chart redrawing happens either here in the class event handler or in the OnChartEvent event handler. If this is not done, you may have the impression that the code is making the platform slow since the price line will move strangely.

This completes our C\_Mouse class and now we can initiate pending orders directly on the chart. Now let's return to our EA code and add the C\_Mouse class in order to be able to generate events and send pending orders. Here is how it is done:

```
#include <Generic Auto Trader\C_Orders.mqh>
#include <Generic Auto Trader\C_Mouse.mqh>
//+------------------------------------------------------------------+
C_Orders *manager;
C_Mouse *mouse;
//+------------------------------------------------------------------+
input int       user01   = 1;           //Lot increase
input double    user02   = 100;         //Take Profit ( FINANCEIRO )
input double    user03   = 75;          //Stop Loss ( FINANCEIRO )
input bool      user04   = true;        //Day Trade ?
input color     user05  = clrBlack;     //Color Mouse
//+------------------------------------------------------------------+
#define def_MAGIC_NUMBER 987654321
//+------------------------------------------------------------------+
int OnInit()
{
        manager = new C_Orders(def_MAGIC_NUMBER);
        mouse = new C_Mouse(user05);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        delete mouse;
        delete manager;
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
```

The above snippet describes the steps of how we are going to produce things. Everything is done in such a way that the C\_Mouse class can be easily removed. Since for an automated EA it has no use, but now we need a way to test our EA, and the best way to do this, without adding triggers, is using the C\_Mouse class that we have just created. So, we include this class so that the compiler can add the code for us. We define a color to be used for the horizontal price line.

We initialize the C\_Order class so that all orders will have the same magic number for this EA. Please note that you should always set different magic number for each EA. The reason is that if you are going to use more than one EA at the same time and on the same asset, the way to separate them so that each EA manages its own orders is through this magic number. If two EAs have the same magic number, you will not be able to separate them, and one of them may interfere with the other's orders. Thus, each EA must have its own magic number.

After that we initialize the C\_Mouse class and thus the EA will be ready to work. Do not forget to call the destructors at the end of the EA. However, if you forget to do it yourself, the compiler usually does it for you. Still, it's good practice to do this in code so everyone knows where to stop referring to a class.

Although I said that the EA is already ready for use, there is one thing missing: the chart event handling system. It will have the following format:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        uint    BtnStatus;
        double  Price;
        static double mem = 0;

        (*mouse).DispatchMessage(id, lparam, dparam, sparam);
        (*mouse).GetStatus(Price, BtnStatus);
        if (def_SHIFT_Press(BtnStatus) != def_CTRL_Press(BtnStatus))
        {
                if (def_BtnLeftClick(BtnStatus) && (mem == 0)) (*manager).CreateOrder(def_SHIFT_Press(BtnStatus) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, mem = Price, user03, user02, user01, user04);
        }else mem = 0;
}
```

Here it is, very simple. I don't want to turn an EA, which we are building to run automatically, into a manual EA. For this reason, the EA will not create price lines that define the limits of take profit and stop loss. But let's see how this code works. There's still something missing here.

When MetaTrader 5 notices that an event has happened on the chart, whatever it may be, it will generate a call to the code above. Once this code is activated and receives the processor to run, it will call the message handling procedure from the **C\_Mouse** class. This way, the events linked to the mouse will be handled properly and always in the same way.

When the **C\_Mouse** class message handler is returned, we will capture the mouse status, to be able to use it here, in this function.

Now let's check the state of the **Shift** and **Ctrl** keys using the definitions in the C\_Mouse.mqh header file. They will serve to indicate whether we are submitting a buy or a sell order. For this reason, they must have different values. If this happens, we will do a new test.

But this time we check if the left mouse button was clicked or not. If this is true and the memory price is zero, send a pending order to the server using the data specified by the user. Buying or selling is determined by the **Shift** and **Ctrl** keys: Shift indicates buying, while Ctrl indicates selling. Thus, we can place as many orders as we need to test the system, but as mentioned above, there is still something missing here.

In the previous article, I offered you a practical task: try to generate a code for making a trade in the market. If you were unable to fulfill it, it is ok. But it would be great if you could handle the task yourself and not look at my solution. The solution is just below, in the updated OnChartEvent handler code.

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        uint    BtnStatus;
        double  Price;
        static double mem = 0;

        (*mouse).DispatchMessage(id, lparam, dparam, sparam);
        (*mouse).GetStatus(Price, BtnStatus);
        if (TerminalInfoInteger(TERMINAL_KEYSTATE_CONTROL))
        {
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_UP))  (*manager).ToMarket(ORDER_TYPE_BUY, user03, user02, user01, user04);
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_DOWN) (*manager).ToMarket(ORDER_TYPE_SELL, user03, user02, user01, user04);
        }
        if (def_SHIFT_Press(BtnStatus) != def_CTRL_Press(BtnStatus))
        {
                if (def_BtnLeftClick(BtnStatus) && (mem == 0)) (*manager).CreateOrder(def_SHIFT_Press(BtnStatus) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, mem = Price, user03, user02, user01, user04);
        }else mem = 0;
}
```

This is a way to send market orders to a trade server. We have the following behavior: When CTRL is pressed, the condition is true here, and now we can check the second condition. This second condition indicates whether we are buying at the market price or selling at the market price. For this, we will use arrow keys. If CTRL + Up Arrow are pressed, we are performing a market buy operation following the parameters specified in the user interaction area. If CTRL + Down arrow are pressed, this is a market sell operation also based on the parameters.

These guidelines are shown in Figure 01:

![Figure 1](https://c.mql5.com/2/48/001__2.png)

Figure 01. EA settings.

### Conclusion

What I just explained about the event system and how it was implemented is sort of a trigger mechanism. But in this case, the mechanism is manual, i.e. it requires the intervention of a human trader: an order to open a position or placing a pending order is performed through the interaction of the EA user with the EA itself. In this case, there is no automatic mechanism for placing an order in the order book or opening a position, it all depends on the trader.

But once the order is placed in the order book, the system will no longer depend on the trader, the order will become a position as soon as the price reaches the point specified in the order, and the position will be closed if one of the limit prices is reached. However, you need to be careful, as there may be a moment of high volatility, which can lead to a price jump. In this case, the position will remain open until it is forcibly closed by the broker or until the time limit expires in case of day trades due to market close. So, you should always be careful with the open position.

The attached file contains the code in its current form. In the next article, we will modify this code to make it easier to use the EA in manual mode, since we will need to make some changes. However, these changes will be interesting.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11232](https://www.mql5.com/pt/articles/11232)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11232.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_04.zip](https://www.mql5.com/en/articles/download/11232/ea_automatico_-_04.zip "Download EA_Automatico_-_04.zip")(4.83 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/442590)**

![Creating an EA that works automatically (Part 05): Manual triggers (II)](https://c.mql5.com/2/50/Aprendendo_construindo_005_avatar.png)[Creating an EA that works automatically (Part 05): Manual triggers (II)](https://www.mql5.com/en/articles/11237)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. At the end of the previous article, I suggested that it would be appropriate to allow manual use of the EA, at least for a while.

![Population optimization algorithms: Invasive Weed Optimization (IWO)](https://c.mql5.com/2/51/invasive-weed-avatar.png)[Population optimization algorithms: Invasive Weed Optimization (IWO)](https://www.mql5.com/en/articles/11990)

The amazing ability of weeds to survive in a wide variety of conditions has become the idea for a powerful optimization algorithm. IWO is one of the best algorithms among the previously reviewed ones.

![Revisiting Murray system](https://c.mql5.com/2/51/murrey_system_avatar.png)[Revisiting Murray system](https://www.mql5.com/en/articles/11998)

Graphical price analysis systems are deservedly popular among traders. In this article, I am going to describe the complete Murray system, including its famous levels, as well as some other useful techniques for assessing the current price position and making a trading decision.

![Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation](https://c.mql5.com/2/52/Recreating-built-in-OpenCL-API-002-avatar.png)[Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation](https://www.mql5.com/en/articles/12108)

Bulit-in OpenCL support in MetaTrader 5 still has a major problem especially the one about device selection error 5114 resulting from unable to create an OpenCL context using CL\_USE\_GPU\_ONLY, or CL\_USE\_GPU\_DOUBLE\_ONLY although it properly detects GPU. It works fine with directly using of ordinal number of GPU device we found in Journal tab, but that's still considered a bug, and users should not hard-code a device. We will solve it by recreating an OpenCL support as DLL with C++ on Linux. Along the journey, we will get to know OpenCL from concept to best practices in its API usage just enough for us to put into great use later when we deal with DLL implementation in C++ and consume it with MQL5.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=omzpaljjtamxncvwwjubnfjfcbsjpttp&ssn=1769180981491811133&ssn_dr=0&ssn_sr=0&fv_date=1769180981&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11232&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20an%20EA%20that%20works%20automatically%20(Part%2004)%3A%20Manual%20triggers%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918098113914465&fz_uniq=5069177297967186228&sv=2552)

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