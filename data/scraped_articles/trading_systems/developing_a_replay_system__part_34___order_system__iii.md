---
title: Developing a Replay System (Part 34): Order System (III)
url: https://www.mql5.com/en/articles/11484
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:16:27.342502
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/11484&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6387814424858530009)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 33): Order System (II)](https://www.mql5.com/en/articles/11482), I explained how we would construct our order system. In that article, we looked at most of the code and touched upon the complexities and problems that we have to solve. Although the code may seem simple and easy to use, this is far from true. So, in this article we'll dive a little deeper into what we actually have and still need to do in terms of implementation. We still have to discuss and explain the last method in the C\_Manager class, as well as comment on the EA code. In the case of the EA code, we will mainly focus on modified parts. This way you will be able to get an idea of how it will behave without resorting to testing in the MetaTrader 5 platform. Well, you can test it as much as you want, since the code will be included in the appendix to this article.

Many may find it unnecessary to test the system as it is shown, hoping to have a more compatible system before actually experimenting with it. This is not a good idea, because if you don't understand how the system works now, you will have big problems understanding how it will work in the future. Worse, you won't be able to adapt it in such a way as that it can generate something you'd like to see, if this is not shown here for one reason or another and you are not interested in development. As the system gradually evolves, you can pay more attention to some details, test others, or wait until the system has matured to the point where you actually feel like it's time to put it on the platform and analyze how it behaves.

I know that many people like to pick up and use a more sophisticated system, while others like to watch it grow and develop. Well, we will start by considering a method that was previously missing. Since this is quite a large topic, we will devote a separate section to it.

### Main function of the C\_Manager class: DispatchMessage

This function is undoubtedly the core of the entire class we are creating. It processes events that are generated and sent to our program from the MetaTrader 5 platform. These are the events that we want the platform to send to us. For example, **CHARTEVENT\_MOUSE\_MOVE**. There are also other sent events, which our program may ignore as they are not very useful for the project we are creating. An example is **CHARTEVENT\_OBJECT\_CLICK**.

The fact that all the event handling is concentrated in the classes makes it much easier to run the project in modules. While this may seem like a lot of work, you will see soon that it enables much easier transfer of code from one project to another, thereby speeding up the development of new projects.

There are two moments here:

- First, by concentrating event processing in one place, as well as making it easier to port and move code between projects, we reduce the amount of code that we need to place in the main code, in this case the EA. This makes debugging much easier because it reduces the number of errors that can occur both when reusing classes and when handling events.
- The second moment is a little more complex. It concerns how each program will work. Some types of code should happen in a certain sequence. Very often, people write code that must be executed in a specific sequence, otherwise it will not work or will produce erroneous results.

Given these two points, we can look at the code of the method to find out and understand why it is executed:

```
void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
   {
      static double price = 0;
      bool bBuy, bSell;

      def_AcessTerminal.DispatchMessage(id, lparam, dparam, sparam);
      def_AcessMouse.DispatchMessage(id, lparam, dparam, sparam);
      switch (id)
      {
         case CHARTEVENT_KEYDOWN:
            if (TerminalInfoInteger(TERMINAL_KEYSTATE_CONTROL))
            {
               if (TerminalInfoInteger(TERMINAL_KEYSTATE_UP))  ToMarket(ORDER_TYPE_BUY);
               if (TerminalInfoInteger(TERMINAL_KEYSTATE_DOWN))ToMarket(ORDER_TYPE_SELL);
            }
            break;
         case CHARTEVENT_MOUSE_MOVE:
            bBuy = def_AcessMouse.CheckClick(C_Mouse::eSHIFT_Press);
            bSell = def_AcessMouse.CheckClick(C_Mouse::eCTRL_Press);
            if (bBuy != bSell)
            {
               if (!m_Objects.bCreate)
               {
                  def_AcessTerminal.CreateObjectGraphics(def_LINE_PRICE, OBJ_HLINE, m_Objects.corPrice, 0);
                  def_AcessTerminal.CreateObjectGraphics(def_LINE_STOP, OBJ_HLINE, m_Objects.corStop, 0);
                  def_AcessTerminal.CreateObjectGraphics(def_LINE_TAKE, OBJ_HLINE, m_Objects.corTake, 0);
                  EventChartCustom(def_InfoTerminal.ID, C_Mouse::ev_HideMouse, 0, 0, "");
                  m_Objects.bCreate = true;
               }
               ObjectMove(def_InfoTerminal.ID, def_LINE_PRICE, 0, 0, def_InfoMouse.Position.Price);
               ObjectMove(def_InfoTerminal.ID, def_LINE_TAKE, 0, 0, def_InfoMouse.Position.Price + (Terminal.FinanceToPoints(m_Infos.FinanceTake, m_Infos.Leverage) * (bBuy ? 1 : -1)));
               ObjectMove(def_InfoTerminal.ID, def_LINE_STOP, 0, 0, def_InfoMouse.Position.Price + (Terminal.FinanceToPoints(m_Infos.FinanceStop, m_Infos.Leverage) * (bSell ? 1 : -1)));
               if ((def_AcessMouse.CheckClick(C_Mouse::eClickLeft)) && (price == 0)) CreateOrder((bBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL), price = def_InfoMouse.Position.Price);
            }else if (m_Objects.bCreate)
            {
               EventChartCustom(def_InfoTerminal.ID, C_Mouse::ev_ShowMouse, 0, 0, "");
               ObjectsDeleteAll(def_InfoTerminal.ID, def_Prefix);
               m_Objects.bCreate = false;
               price = 0;
            }
            break;
         }
      }
```

Don't be afraid of this code above. Although this code may seem complicated at first glance, you really shouldn't be afraid of it. Everything we do is quite simple and relatively common. The appearance of the code gives us the first impression of being very complex and difficult to understand. So, let's take it step by step. First, let's look at the first calls. To explain them better, I will divide it into parts. I believe such an explanation will be easier to understand. We will focus on just a few points, so you won't have to scroll the page to find what we are discussing.

```
def_AcessTerminal.DispatchMessage(id, lparam, dparam, sparam);
def_AcessMouse.DispatchMessage(id, lparam, dparam, sparam);
```

Remember we mentioned that at some points we need everything to happen in a certain sequence? These two lines above are exactly about this. They will prevent you from forgetting or (even worse) placing event processing in the wrong sequence in the EA code. It is not that this has any impact now, at this stage of writing the code, but the less code we have to place in the EA, the better it will be in the future. By missing a certain event, we can cause the entire project to work in a completely unexpected way or not the way we want it to work.

This is all clear. Now we can start by looking at the **CHARTEVENT\_KEYDOWN** event. It will take care of the triggers that occur when you press a key.

```
case CHARTEVENT_KEYDOWN:
        if (TerminalInfoInteger(TERMINAL_KEYSTATE_CONTROL))
        {
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_UP))  ToMarket(ORDER_TYPE_BUY);
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_DOWN))ToMarket(ORDER_TYPE_SELL);
        }
        break;
```

Here we have a situation that may seem a little confusing: according to the documentation, the **lparam** variable will contain the code of the key pressed. This does happen, but the problem is that we need to do things in a slightly different way. When a key is pressed, the operating system generates a specific event. If MetaTrader 5 receives focus from the operating system, it will pass the generated event. Since our code has a handler for the keypress action, it isolates keypress handling from other kinds of events.

**Note:** The code shown above doesn't actually need to be placed in the CHARTEVENT\_KEYDOWN event. It could be placed outside of this event. However, by placing it in CHARTEVENT\_KEYDOWN, we prevent some embarrassing situation. This situation occurs when we perform the analysis of a key condition, that is, the CHARTEVENT\_KEYDOWN event, when the platform alerts us to some other type of event that was fired for some reason.

Think about handling the keyboard state when an event like **CHARTEVENT\_CHART\_CHANGE** is fired, which is actually activated when some changes occur on the chart. And at the same time, our program checks the state of the keyboard. Such things have no practical meaning, and also require a lot of time to implement. This is why I isolate the parsing of the keyboard state in the CHARTEVENT\_KEYDOWN event.

But let's get back to the code. You'll notice that I'm using the [TerminalInfoInteger](https://www.mql5.com/en/docs/check/terminalinfointeger) function to recognize and isolate specific keyboard code. If it is not done, we would have to make an extra effort to check whether the CTRL key was pressed at the same time as another key, in this case UP ARROW or DOWN ARROW. This is exactly what we do. We need a keyboard shortcut so that our program, in this case the EA, knows what to do in terms of programming. If you press the combination CTRL + UP ARROW, the EA should understand this we want to buy at the market price. If the combination CTRL + DOWN ARROW is pressed, the EA should proceed to sell at the market price. Note that although the **lparam** variable specifies the key values individually, this does not help us work with keyboard shortcuts. But if you do it the way you do now, by pressing only one of the key combinations, the EA will not receive any instructions to trade at the market price.

If you think that this combination may somehow conflict with what you are using, simply change it. But be careful and keep the code inside the **CHARTEVENT\_KEYDOWN** event to take advantage of the fact that it will only be executed when MetaTrader 5 fires a key event, thus preventing unnecessary code execution. Another thing is that the key code displayed in the **lparam** variable follows a table that varies from region to region, which makes things much more complicated. The way I am showing here will not actually use such a table.

Now let's look at the next event handler, CHARTEVENT\_MOUSE\_MOVE. For ease of explanation, I'll break it down into small parts, in the order they appear in the class code.

```
case CHARTEVENT_MOUSE_MOVE:
        bBuy = def_AcessMouse.CheckClick(C_Mouse::eSHIFT_Press);
        bSell = def_AcessMouse.CheckClick(C_Mouse::eCTRL_Press);
```

Pay attention to one thing. Here we use the C\_Study class to access the C\_Mouse class. Don't forget this, and note that unlike the **CHARTEVENT\_KEYDOWN** event handler discussed above, here we capture the state of the buttons. This now refers to the mouse. Doesn't this bother you? In fact, these buttons belong to the mouse, not the alphanumeric keyboard. Why? Are you trying to confuse me? Nothing like that, my dear reader. The fact that we can press SHIFT and CTRL on an alphanumeric keyboard and still manage to do it inside the C\_Mouse class is because it doesn't quite work that way. These SHIFT and CTRL keys actually belong to the mouse. But not just any mouse. I'm talking about a very specific kind of mouse, more or less similar to the one shown in Figure 01:

![Figure 01](https://c.mql5.com/2/48/001__13.png)

Figure 01:

This type of mouse has additional buttons on its body. For the operating system and therefore for the platform and our program, the **SHIFT** and **CTRL** keys, which we are talking about, are actually part of the mouse. However, since a mouse may not have such additional buttons, the operating system allows the use of the keyboard, and thanks to this, the platform and the program will ensure that the code is interpreted in the correct way. Therefore, the SHIFT and CTRL keys from the CHARTEVENT\_KEYDOWN event should not be confused with those used here in the CHARTEVENT\_MOUSE\_MOVE event.

Now that we know the state of the SHIFT and CTRL keys, we can look at the rest of the event code. This can be judged from the fragment below.

```
        if (bBuy != bSell)
        {
                if (!m_Objects.bCreate)
                {
                        def_AcessTerminal.CreateObjectGraphics(def_LINE_PRICE, OBJ_HLINE, m_Objects.corPrice, 0);
                        def_AcessTerminal.CreateObjectGraphics(def_LINE_STOP, OBJ_HLINE, m_Objects.corStop, 0);
                        def_AcessTerminal.CreateObjectGraphics(def_LINE_TAKE, OBJ_HLINE, m_Objects.corTake, 0);
                        EventChartCustom(def_InfoTerminal.ID, C_Mouse::ev_HideMouse, 0, 0, "");
                        m_Objects.bCreate = true;
                }
                ObjectMove(def_InfoTerminal.ID, def_LINE_PRICE, 0, 0, def_InfoMouse.Position.Price);
                ObjectMove(def_InfoTerminal.ID, def_LINE_TAKE, 0, 0, def_InfoMouse.Position.Price + (Terminal.FinanceToPoints(m_Infos.FinanceTake, m_Infos.Leverage) * (bBuy ? 1 : -1)));
                ObjectMove(def_InfoTerminal.ID, def_LINE_STOP, 0, 0, def_InfoMouse.Position.Price + (Terminal.FinanceToPoints(m_Infos.FinanceStop, m_Infos.Leverage) * (bSell ? 1 : -1)));
                if ((def_AcessMouse.CheckClick(C_Mouse::eClickLeft)) && (price == 0)) CreateOrder((bBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL), price = def_InfoMouse.Position.Price);
        }else if (m_Objects.bCreate)
        {
                EventChartCustom(def_InfoTerminal.ID, C_Mouse::ev_ShowMouse, 0, 0, "");
                ObjectsDeleteAll(def_InfoTerminal.ID, def_Prefix);
                m_Objects.bCreate = false;
                price = 0;
        }
```

Although this code looks complicated, it actually does three fairly simple things. They can be separated into a separate function or method within a class. But for now, they will remain here to make things a little easier, at least as far as calls are concerned. The first thing to do is to inform the platform through the EA that we want to place a pending order. But before we do that, we need to see where the limit levels and the order will be placed. To do this, we use three objects that are created at this point in the code. Note that we also send an event to the C\_Mouse class system to tell the C\_Mouse class that the mouse should be hidden in the coming moments. This is the first phase.

Once we have the required objects on the chart, we move them in a specific way. To do this, we use this set of functions. But if the user tells us that the desired moment to submit the order is the one shown on the chart through the created objects, we will fulfill the request to place a pending order. Notice how the checks are done so we can know what's happening with the mouse and on the chart.

As for the third and last point, events develop as follows. First, an event is sent to the C\_Mouse class system so that the mouse becomes visible again on the graph. Immediately after this we will delete the created objects.

Now there is something important in this code. You should always pay attention to this when programming your own codes. If you are just starting to program, you may not have noticed a very interesting and at the same time dangerous point in the code above. It can be dangerous if you don't do it right. I'm talking about the **RECURSION**. In this code above, we use recursion. If this is not planned correctly, we will end up in an infinite loop. Once a system enters the code part that uses recursion, it may never leave it again.

To understand how this recursion occurs, take a look at Figure 02, located just below:

![Figure 02](https://c.mql5.com/2/48/002__6.png)

Figure 02: Internal message flow.

The green arrow in Figure 02 is exactly where recursion takes place. But how does this happen in code? To see this, take a look at the code below showing the DispatchMessage method present in the C\_Manager class.

```
void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
   {

// ...

      def_AcessMouse.DispatchMessage(id, lparam, dparam, sparam);
      switch (id)
      {

// ...

         case CHARTEVENT_MOUSE_MOVE:
// ...
            if (bBuy != bSell)
            {
               if (!m_Objects.bCreate)
               {
// ...
                  EventChartCustom(def_InfoTerminal.ID, C_Mouse::ev_HideMouse, 0, 0, "");
               }

// ...

            }else if (m_Objects.bCreate)
            {
               EventChartCustom(def_InfoTerminal.ID, C_Mouse::ev_ShowMouse, 0, 0, "");

// ...
            }
            break;
         }
      }
```

The recursion may not be very clear from Figure 02. If you just look at the code above, it may not be clear enough either. But if you combine Figure 02 with the code above, you can see how the recursion works. If this is not planned correctly, we will have serious problems. Let's get into the explanation so that you, as a new programmer, can understand both the power of recursion and its dangers. When we need to do something very complex, no matter what it is, we need to break the process down into smaller or simpler tasks. These tasks can be repeated a certain number of times to end up with something much more complex, while following a simple concept. This is the power of recursion. However, it is not only used in this scenario. Although it is most often associated with this, we can use recursion to solve other cases.

The above fragment is one of such cases. To understand the explanations, please once again take a look at Figure 02. It starts when the user generates an event, which causes the MetaTrader 5 platform to call the OnChartEvent function. In this case, this function calls the DispatchMessage method in the C\_Manager class. At this point, we call the event handler which is present by inheritance in the C\_Mouse class, specifically in the DispatchMessage method present in this class. When the method returns, the program continues where it left off. We enter an event handler to check whether the user wants to create or delete the helper lines. So, at some point we have the EventChartCustom call. At this point, the recursion is activated.

What actually happens is that MetaTrader 5 makes a new call to the OnChartEvent function, due to which the function is executed again and the DispatchMessage method from the C\_Manager class is called. This, in turn, again calls the C\_Mouse class by inheritance to execute a custom method that will cause the mouse cursor to appear or disappear, depending on the situation. However, because of the recursion, the code will not return as many might think. In fact, it will return, but this will again trigger the execution of code in the DispatchMessage method of the C\_Manager class. Herein lies the danger: if you place the call so that a custom event that was handled in the C\_Mouse class appears in the C\_Manager class, it will also be processed in the C\_Manager class. And if we accidentally handle this event again in the C\_Mouse class using the EventChartCustom function, we will find ourselves in an endless loop.

However, since the C\_Manager class does not have such an event, which is handled by the C\_Mouse class, we will fall back to the OnChartEvent function, which will be executed entirely. When the OnChartEvent function is executed, it will return to the point where the EventChartCustom call was made. This is shown in the code part above. This will cause the execution of all the remaining code in the DispatchMessage method in the C\_Manager class. When it completes, we will return to the OnChartEvent function, where it will be executed completely, freeing the platform to execute other type of events.

Thus, when calling EventChartCustom, due to recursion, at least twice as much code will be executed as in the OnChartEvent function. This seems inefficient, but the point is that the code is simple and does not greatly affect the overall performance of the platform. However, it is good that we are always aware of what is really happening. The cost of recursion in this case is quite low compared to more modular code. But in some situations, these costs may not be covered and may make the code too slow. In such cases, we would have to take some other action, but this is not our case at the moment.

I think I have explained in detail the DispatchMessage method used in the C\_Manager class. Although this may seem quite complicated, in fact we are far from something truly complex, since the system does not yet know how to work with the cross-order model. To do this, the DispatchMessage method required significant modifications. But we will leave this for the future.

Let's now look at further changes in the EA code.

### Analyzing updates in the Expert Advisor

Although the EA can now place orders and trade within the configured trading time, its code has not undergone any major changes. There is one part of the code that deserves special attention. I will explain what's going on there. This important point concerns the user interaction and code in the OnInit event. Let's start with the user interaction. It is shown in the code below:

```
input group "Mouse";
input color     user00 = clrBlack;      //Price Line
input color     user01 = clrPaleGreen;  //Positive Study
input color     user02 = clrLightCoral; //Negative Study
input group "Trade";
input uint      user10 = 1;             //Leverage
input double    user11 = 100;           //Take Profit ( Finance )
input double    user12 = 75;            //Stop Loss ( Finance )
input bool      user13 = true;          //Is Day Trade
//+------------------------------------------------------------------+
input group "Control of Time"
input string    user20  = "00:00 - 00:00";      //Sunday
input string    user21  = "09:05 - 17:35";      //Monday
input string    user22  = "10:05 - 16:50";      //Tuesday
input string    user23  = "09:45 - 13:38";      //Wednesday
input string    user24  = "11:07 - 15:00";      //Thursday
input string    user25  = "12:55 - 18:25";      //Friday
input string    user26  = "00:00 - 00:00";      //Saturday
```

The code is responsible for user interaction. Here we have two new groups of information which can be accessed and configured by the user. In the first group, the user selects how the trading operation will be performed - by market or as a pending order. The settings are quite straightforward. The value representing leverage (user 10) should represent the number of times that you will use the minimum volume to trade the asset. In the case of Forex, you will most likely use a value of 100 or something similar to find a good margin to work with. Otherwise, you will be working with the order of cents, which will make the limit lines far from where you would expect them to be. If you trade on the exchange, you must report the number of shares to be used. Otherwise, specify the number of lots. For futures trades, specify the number of contracts. So, this is all pretty straightforward. As for the take profit (user 11) and stop loss (user 12), you should indicate not the number of points, but the financial value that will be used. This value must be adjusted accordingly by the code to reflect the correct position in the asset price. The last variable (user 13) only serves to indicate whether we are taking a long position or a short position.

**Important Note:** This mechanism should be tested with caution as brokers may have very specific trading conditions. Please check this with your broker in advance.

Now, in the second group, there are some things to check before setting them properly. This is not because they are complicated or difficult to understand, but because you should understand that these variables will determine when the EA allows us to send orders or place pending orders. The question of managing, terminating or even modifying orders will no longer depend on EA. The MetaTrader 5 platform will be responsible for this, at least for now.

You can then set a 1-hour window in which the EA is allowed to work using the resources it has. This configuration is done for a week rather than a specific day or special date.

To understand this, look at the OnInit code below:

```
int OnInit()
{
        string szInfo;

        terminal = new C_Terminal();
        study    = new C_Study(terminal, user00, user01, user02);
        manager  = new C_Manager(terminal, study, user00, user02, user01, def_MagicNumber, user12, user11, user10, user13);

        if (_LastError != ERR_SUCCESS) return INIT_FAILED;

        for (ENUM_DAY_OF_WEEK c0 = SUNDAY; c0 <= SATURDAY; c0++)
        {
                switch (c0)
                {
                        case SUNDAY     : szInfo = user20; break;
                        case MONDAY     : szInfo = user21; break;
                        case TUESDAY    : szInfo = user22; break;
                        case WEDNESDAY  : szInfo = user23; break;
                        case THURSDAY   : szInfo = user24; break;
                        case FRIDAY     : szInfo = user25; break;
                        case SATURDAY   : szInfo = user26; break;
                }
                (*manager).SetInfoCtrl(c0, szInfo);
        }

        MarketBookAdd(def_InfoTerminal.szSymbol);
        OnBookEvent(def_InfoTerminal.szSymbol);
        EventSetMillisecondTimer(500);

        return INIT_SUCCEEDED;
}
```

Pay attention to the OnInit code above. It represents a complete picture of how the EA should behave during the week, not only on a specific day. Throughout the entire week. There are assets or markets, in this case the Forex market, where trading runs almost continuously and does not stop at any time of the day. If we needed to configure our individual trading schedule for the EA were that can run 24 hours a day, we would have problems during the day change period. That is, as soon as it is 23:59:59, we would need to stop the EA and set it back the next second to find out the new trading interval. But if you use the above method, the EA can run 24 hours a day, 7 days a week, 52 weeks a year without getting lost or knowing which time schedule to use. I know just by looking at this code many may not understand how this actually happens. Therefore, you should test the EA to understand how this system works. But this system is not new. We already discussed this in one of the earlier articles: [Creating an EA that works automatically (Part 10): Automation (II)](https://www.mql5.com/en/articles/11286).

### Conclusion

Although the system seems to be quite stable and versatile, it began to suffer from one error. This is quite strange and puzzling, since the appearance of this error makes no sense. The reason is that the error appeared in a place that was not modified in any way. Anyway, we will get back to this error in the next article. The attachment contains the entire code at the current stage of development, which you can study and analyze in detail. In the previous article, I did not attach any code.

Please note that in the last article I talked about the system of selecting objects with one click, as opposed to the platform-standard double-click mode. The best option is to test the system on your platform and draw your own conclusions after seeing it in action. So, download and run it to see how the system works.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11484](https://www.mql5.com/pt/articles/11484)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11484.zip "Download all attachments in the single ZIP archive")

[Files\_-\_BOLSA.zip](https://www.mql5.com/en/articles/download/11484/files_-_bolsa.zip "Download Files_-_BOLSA.zip")(1358.24 KB)

[Files\_-\_FOREX.zip](https://www.mql5.com/en/articles/download/11484/files_-_forex.zip "Download Files_-_FOREX.zip")(3743.96 KB)

[Files\_-\_FUTUROS.zip](https://www.mql5.com/en/articles/download/11484/files_-_futuros.zip "Download Files_-_FUTUROS.zip")(11397.51 KB)

[Market\_Replay\_-\_34.zip](https://www.mql5.com/en/articles/download/11484/market_replay_-_34.zip "Download Market_Replay_-_34.zip")(130.63 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/466110)**

![Population optimization algorithms: Bacterial Foraging Optimization - Genetic Algorithm (BFO-GA)](https://c.mql5.com/2/64/Bacterial_Foraging_Optimization_-_Genetic_Algorithmz_BFO-GA____LOGO.png)[Population optimization algorithms: Bacterial Foraging Optimization - Genetic Algorithm (BFO-GA)](https://www.mql5.com/en/articles/14011)

The article presents a new approach to solving optimization problems by combining ideas from bacterial foraging optimization (BFO) algorithms and techniques used in the genetic algorithm (GA) into a hybrid BFO-GA algorithm. It uses bacterial swarming to globally search for an optimal solution and genetic operators to refine local optima. Unlike the original BFO, bacteria can now mutate and inherit genes.

![Developing an MQL5 Reinforcement Learning agent with RestAPI integration (Part 1): How to use RestAPIs in MQL5](https://c.mql5.com/2/60/RestAPIs_em_MQL5_Logo.png)[Developing an MQL5 Reinforcement Learning agent with RestAPI integration (Part 1): How to use RestAPIs in MQL5](https://www.mql5.com/en/articles/13661)

In this article we will talk about the importance of APIs (Application Programming Interface) for interaction between different applications and software systems. We will see the role of APIs in simplifying interactions between applications, allowing them to efficiently share data and functionality.

![A Generic Optimization Formulation (GOF) to Implement Custom Max with Constraints](https://c.mql5.com/2/76/A_Generic_Optimization_Formulation_2GOFt_to_Implement_Custom_Max_with_Constraints____LOGO.png)[A Generic Optimization Formulation (GOF) to Implement Custom Max with Constraints](https://www.mql5.com/en/articles/14365)

In this article we will present a way to implement optimization problems with multiple objectives and constraints when selecting "Custom Max" in the Setting tab of the MetaTrader 5 terminal. As an example, the optimization problem could be: Maximize Profit Factor, Net Profit, and Recovery Factor, such that the Draw Down is less than 10%, the number of consecutive losses is less than 5, and the number of trades per week is more than 5.

![Population optimization algorithms: Evolution Strategies, (μ,λ)-ES and (μ+λ)-ES](https://c.mql5.com/2/63/midjourney_image_13923_53_472__2-logo.png)[Population optimization algorithms: Evolution Strategies, (μ,λ)-ES and (μ+λ)-ES](https://www.mql5.com/en/articles/13923)

The article considers a group of optimization algorithms known as Evolution Strategies (ES). They are among the very first population algorithms to use evolutionary principles for finding optimal solutions. We will implement changes to the conventional ES variants and revise the test function and test stand methodology for the algorithms.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/11484&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6387814424858530009)

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