---
title: Developing a Replay System (Part 76): New Chart Trade (III)
url: https://www.mql5.com/en/articles/12443
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:33:18.174651
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/12443&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069541927805716156)

MetaTrader 5 / Examples


### Introduction

In the previous article, [Developing a Replay System (Part 75): New Chart Trade (II)](https://www.mql5.com/en/articles/12442), I explained several aspects of the C\_ChartFloatingRAD class. However, because the material was quite dense, I aimed to provide the most detailed explanation possible whenever relevant. One procedure remained to be discussed. Even if I had left it in the header file C\_ChartFloatingRAD.mqh and tried to explain it in the previous article, it would not have been adequate. This is because, to fully understand how the DispatchMessage procedure works, another topic needs to be explained alongside it.

In this article, I will be able to go into detail about how the DispatchMessage procedure truly operates. This is the most important procedure of the C\_ChartFloatingRAD class, as it is responsible for generating and responding to the events that MetaTrader 5 sends to the Chart Trade.

The content here should be read in conjunction with the previous article. I do not recommend taking on this article without first thoroughly grasping what was covered before. Together, the last two articles and this one represent the full conceptual foundation behind the Chart Trade indicator. Therefore, it is important to understand each of them.

So, let's move on to the explanation of DispatchMessage.

### Understanding How DispatchMessage Works

If you still expect to see numerous objects being coded here in Chart Trade, then you haven't yet fully understood how things work. Before we continue, I strongly recommend revisiting the previous article. This time we are looking at what makes Chart Trade functional, purely by responding to and generating events. We will not be creating any objects here. Our task is simply to make the events reported by MetaTrader 5 operational.

In the last article, you may have noticed that the header file C\_ChartFloatingRAD.mqh contained a region where code was missing. The missing code is shown in the snippet below:

```
259. //+------------------------------------------------------------------+
260.       void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
261.          {
262. #define macro_AdjustMinX(A, B)    {                          \
263.             B = (A + m_Info.Regions[MSG_TITLE_IDE].w) > x;   \
264.             mx = x - m_Info.Regions[MSG_TITLE_IDE].w;        \
265.             A = (B ? (mx > 0 ? mx : 0) : A);                 \
266.                                  }
267. #define macro_AdjustMinY(A, B)   {                           \
268.             B = (A + m_Info.Regions[MSG_TITLE_IDE].h) > y;   \
269.             my = y - m_Info.Regions[MSG_TITLE_IDE].h;        \
270.             A = (B ? (my > 0 ? my : 0) : A);                 \
271.                                  }
272.
273.             static short sx = -1, sy = -1, sz = -1;
274.             static eObjectsIDE obj = MSG_NULL;
275.             short   x, y, mx, my;
276.             double dvalue;
277.             bool b1, b2, b3, b4;
278.             ushort ev = evChartTradeCloseAll;
279.
280.             switch (id)
281.             {
282.                case CHARTEVENT_CHART_CHANGE:
283.                   x = (short)ChartGetInteger(GetInfoTerminal().ID, CHART_WIDTH_IN_PIXELS);
284.                   y = (short)ChartGetInteger(GetInfoTerminal().ID, CHART_HEIGHT_IN_PIXELS);
285.                   macro_AdjustMinX(m_Info.x, b1);
286.                   macro_AdjustMinY(m_Info.y, b2);
287.                   macro_AdjustMinX(m_Info.minx, b3);
288.                   macro_AdjustMinY(m_Info.miny, b4);
289.                   if (b1 || b2 || b3 || b4) AdjustTemplate();
290.                   break;
291.                case CHARTEVENT_MOUSE_MOVE:
292.                   if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft)) switch (CheckMousePosition(x = (short)lparam, y = (short)dparam))
293.                   {
294.                      case MSG_TITLE_IDE:
295.                         if (sx < 0)
296.                         {
297.                            DeleteObjectEdit();
298.                            ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
299.                            sx = x - (m_Info.IsMaximized ? m_Info.x : m_Info.minx);
300.                            sy = y - (m_Info.IsMaximized ? m_Info.y : m_Info.miny);
301.                         }
302.                         if ((mx = x - sx) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, mx);
303.                         if ((my = y - sy) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, my);
304.                         if (m_Info.IsMaximized)
305.                         {
306.                            m_Info.x = (mx > 0 ? mx : m_Info.x);
307.                            m_Info.y = (my > 0 ? my : m_Info.y);
308.                         }else
309.                         {
310.                            m_Info.minx = (mx > 0 ? mx : m_Info.minx);
311.                            m_Info.miny = (my > 0 ? my : m_Info.miny);
312.                         }
313.                         break;
314.                      case MSG_BUY_MARKET:
315.                         ev = evChartTradeBuy;
316.                      case MSG_SELL_MARKET:
317.                         ev = (ev != evChartTradeBuy ? evChartTradeSell : ev);
318.                      case MSG_CLOSE_POSITION:
319.                         if ((m_Info.IsMaximized) && (sz < 0))
320.                         {
321.                            string szTmp = StringFormat("%d?%s?%c?%d?%.2f?%.2f", ev, _Symbol, (m_Info.IsDayTrade ? 'D' : 'S'), m_Info.Leverage,
322.                                                 FinanceToPoints(m_Info.FinanceTake, m_Info.Leverage), FinanceToPoints(m_Info.FinanceStop, m_Info.Leverage));
323.                            PrintFormat("Send %s - Args ( %s )", EnumToString((EnumEvents) ev), szTmp);
324.                            sz = x;
325.                            EventChartCustom(GetInfoTerminal().ID, ev, 0, 0, szTmp);
326.                            DeleteObjectEdit();
327.                         }
328.                         break;
329.                   }else
330.                   {
331.                      sz = -1;
332.                      if (sx > 0)
333.                      {
334.                         ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
335.                         sx = sy = -1;
336.                      }
337.                   }
338.                   break;
339.                case CHARTEVENT_OBJECT_ENDEDIT:
340.                   switch (obj)
341.                   {
342.                      case MSG_LEVERAGE_VALUE:
343.                      case MSG_TAKE_VALUE:
344.                      case MSG_STOP_VALUE:
345.                         dvalue = StringToDouble(ObjectGetString(GetInfoTerminal().ID, m_Info.szObj_Editable, OBJPROP_TEXT));
346.                         if (obj == MSG_TAKE_VALUE)
347.                            m_Info.FinanceTake = (dvalue <= 0 ? m_Info.FinanceTake : dvalue);
348.                         else if (obj == MSG_STOP_VALUE)
349.                            m_Info.FinanceStop = (dvalue <= 0 ? m_Info.FinanceStop : dvalue);
350.                         else
351.                            m_Info.Leverage = (dvalue <= 0 ? m_Info.Leverage : (short)MathFloor(dvalue));
352.                         AdjustTemplate();
353.                         obj = MSG_NULL;
354.                         ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
355.                         break;
356.                   }
357.                   break;
358.                case CHARTEVENT_OBJECT_CLICK:
359.                   if (sparam == m_Info.szObj_Chart) if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft)) switch (obj = CheckMousePosition(x = (short)lparam, y = (short)dparam))
360.                   {
361.                      case MSG_DAY_TRADE:
362.                         m_Info.IsDayTrade = (m_Info.IsDayTrade ? false : true);
363.                         DeleteObjectEdit();
364.                         break;
365.                      case MSG_MAX_MIN:
366.                         m_Info.IsMaximized = (m_Info.IsMaximized ? false : true);
367.                         DeleteObjectEdit();
368.                         break;
369.                      case MSG_LEVERAGE_VALUE:
370.                         CreateObjectEditable(obj, m_Info.Leverage);
371.                         break;
372.                      case MSG_TAKE_VALUE:
373.                         CreateObjectEditable(obj, m_Info.FinanceTake);
374.                         break;
375.                      case MSG_STOP_VALUE:
376.                         CreateObjectEditable(obj, m_Info.FinanceStop);
377.                         break;
378.                   }
379.                   if (obj != MSG_NULL) AdjustTemplate();
380.                   break;
381.                case CHARTEVENT_OBJECT_DELETE:
382.                   if (sparam == m_Info.szObj_Chart) macro_CloseIndicator(C_Terminal::ERR_Unknown);
383.                   break;
384.             }
385.             ChartRedraw();
386.          }
387. //+------------------------------------------------------------------+
388. };
389. //+------------------------------------------------------------------+
```

Missing code from C\_ChartFloatingRAD.mqh

The missing code fits perfectly into that gap, as shown in the highlighted snippet below. In order to complete the C\_ChartFloatingRAD class code, this fragment should be inserted exactly in that location, following the indicated line numbers. This should be quite easy to do. Let's now see how it works.

The first thing to notice is that this fragment contains no calls to create objects. As I mentioned, we are not programming objects here. Chart Trade is created and stored as a template. This is done entirely within MetaTrader 5. In the previous article, I left some references at the end to help you understand this concept.

This type of programming is not commonly used by most developers. But in my opinion, it is far more practical and simpler in many cases. You don't have to manually code and position every object. Instead, you can simply reference them in the template. And we only create them programmatically if MetaTrader 5 cannot configure them directly. As discussed previously, creating elements and saving them as a template is a rather rare activity. The only object that could not be handled this way was **OBJ\_EDIT**. This is because creating the logic for text entry and manipulation is more cumbersome than simply creating the **OBJ\_EDIT** object itself. Therefore, this is the only object that will actually be created.

Looking at the snippet, you'll see that it deals solely with event handling. This might seem surprising if you've seen the video in the previous article or run the attached executables. After all, the code doesn't seem to contain many objects, yet it works exactly as shown. You might have assumed that other objects appear inside DispatchMessage. But as you can see, there are none. **THERE ARE NO OTHER OBJECTS**. Everything is just handling events from MetaTrader 5. So how does this work?

Let's break it down step by step. We handle five types of events sent by MetaTrader 5 here. Before looking at them individually, notice that lines 273–278 declare some variables. Variables declared as static are those that can be located in the global scope of the class. Those marked as static retain their values between calls and could have been class-level variables, but they are only needed inside this procedure. If you look closely, you will see that the variable declared on line 278 is not static. It is initialized at the moment of its declaration. This avoids problems associated with incorrect use of this variable. The assigned value corresponds to the event declared in the Defines.mqh file.

Let's start with the event **CHARTEVENT\_CHART\_CHANGE** that appears in lines 282–290. It triggers whenever the chart changes. It performs simple calculations to position and maintains the **OBJ\_CHART** object correctly within the chart window.

The second event, **CHARTEVENT\_MOUSE\_MOVE** shown in lines 291–338, is the longest handler in DispatchMessage. This event is fired whenever the mouse moves or its buttons are pressed. But by default, MetaTrader 5 does not send it. We must explicitly specify that we want to receive mouse events. These events are requested from MetaTrader 5 via the mouse indicator. This indicator was described in detail in previous articles. Understanding the mouse indicator is crucial, as it enables user interaction with the elements that we are creating.

Notably, we don't directly process mouse events here. Instead, on line 292, the first step is to check for a left-click. But we also need to confirm its validity with the mouse indicator. Why are we doing this? Why don't we just check whether the click is valid or not right in the handler? Why do we have to ask the mouse indicator?

Because the mouse indicator may be running in “the "research" mode. When this mode is on at the user's request, no clicks should be processed. We cannot reliably detect when and how long the mouse indicator will be in the "research" mode. So, the best way is to query the mouse indicator instead. If the click is valid, we convert the position data from MetaTrader 5 and check which object was clicked. This is done via CheckMousePosition. We could also use the coordinates provided by the mouse indicator. But in this case, we would need to read the indicator buffer. Since the values will be the same as those provided directly by MetaTrader 5, there is no point in reading the buffer to get this information.

Before we analyze how the actual click is handled, let's look at what happens in line 329, which handles the case where the **CHARTEVENT\_MOUSE\_MOVE** event did not receive a valid click or was caused by any movement made by the user interacting with the mouse. In this case, in line 331, the **sz** variable is assigned a negative value. Then, in line 332, we check whether the variable **sx** has a positive value. If it does, we proceed to handle the mouse event, but we will leave that for now. So, in line 334, we inform MetaTrader 5 that the mouse can move the chart. Then, in line 335, we assign negative values to variables **sx** and **sy**, which causes the check in line 332 to fail and prevents line 334 from being executed unnecessarily.

It should be noted that mouse events **CHARTEVENT\_MOUSE\_MOVE** fire extremely frequently. In some cases, **CHARTEVENT\_MOUSE\_MOVE** can happen even more frequently than OnTick. So, careless handling of **these** frequent events can severely slow down MetaTrader 5. This is why they are disabled by default.

Okay, we got that sorted out. Lines 293–328 contain the logic for interacting with specific Chart Trade elements. When CheckMousePosition returns a value, it can point to one of the objects mentioned in this part of the code, such as the title bar, Market Buy, Market Sell, and Close Position buttons. Other elements are handled in other places and will be described later, but each of these four elements requires a different approach. Let's start with the hardest part: the title bar.

When the title bar is clicked, the user may want to move the Chart Trade. So, lines 295 to 313 contain the required logic. We won't go into detail now, as this process has already been described in another article within this series, where we created the first versions of Chart Trade. If you have questions, please read the previous articles, they are also important for what we are developing here. Remember that code does not come out of nowhere: it was gradually created and implemented until it took the form you see now.

The part we're interested in is between lines 314 and 328, so pay attention to the value defined on line 278. This is important because here we will be testing two of the three conditions used. While we handle three conditions, the first one we check is whether the Market Buy button was clicked. If this is true, a certain value is assigned to the 'ev' variable to indicate the Market Buy event. This is done in line 315.

Now please note: In line 316 we check whether the Market Sell button was clicked. However, we also need to check if the Market Buy button was clicked in order to correctly assign a value to the 'ev' variable. Why do we need this check if we are handling another event? True, this is a separate event.

But the point is that all three events (Market Buy, Market Sell and Close Position) perform the same type of action. There is no pause between events, on the contrary, they are arranged in a string. Therefore, in line 317 we need to check whether the previous event has changed the value of the ev variable. If this is true, then we should not perform this action for the next event in the row. Otherwise, only the last event is triggered, ignoring the previous ones.

In any case, we will always come to line 318, where the Close Position event is handled. For this event, there are checks required for MetaTrader 5 to run the custom event. Now we will look at how to get to this point to avoid possible failures.

Line 319 checks if Chart Trade is maximized. This check is important because the CheckMousePosition function does not take into account whether the window is maximized. Please note that if we don't do this check and click where one of the buttons should be, the custom event will trigger even if the window is not maximized. To avoid this problem and make sure that the event is triggered only when the objects are visible on the chart, we check this condition.

Here an important point arises: the window must be maximized, that is, the buttons must be visible. However, there is a small problem in this check. It is the Chart Trade code that performs this check, not MetaTrader 5. Why is this important? The problem occurs when the buttons are covered by another object. If we place something covering the buttons and the Chart Trade is maximized, then when we click on the area where the buttons are located, it is our code, not MetaTrader 5, that will trigger the custom event. This is a bug that I plan to fix in the future. However, since this is not so critical, we can put up with it for now. But you should take this issue into account if you use Chart Trade on a real account. **DO NOT PLACE ANYTHING IN FRONT OF THE BUTTONS WHEN CHART TRADE IS MAXIMIZED**.

There's another check in the same line 319, which should prevent multiple events from being triggered by a click followed by a mouse movement. This is a simple check in which the value of **sz** must be negative. If it is positive, it means that the event has already fired and the mouse button is still pressed. When the button is released and the mouse moves, line 331 will ensure that a new user event is fired. Although this is a simple check, it can avoid a lot of problems.

All the "magic" happens in line 321. To prevent it from being too long, I divided it into two parts. So, consider line 322 as part of line 321. Before we explain this line, let's look at what the next four lines do. Line 323 logs the event to the MetaTrader 5 Toolbox. You may think that this is not necessary, but believe me, it will help you avoid many problems. This is an invaluable debugging practice. Therefore, never ignore the messages that appear in the MetaTrader 5 Toolbox, many of them are very important.

Line 324 simply performs the lock to prevent the condition previously described, since the variable **sz** is used in the check in line 319. The custom event is actually triggered in line 325. Note that, unlike what was done in the control indicator, here the data is sent in the string field. If that were not possible, we would have problems transmitting a large amount of information between applications.

Now, here's an important detail. Strings in MQL5 follow the same principle as in C/C++ - they end with a **NULL** character. And why am I mentioning this? Because it explains why we need to use the [StringFormat](https://www.mql5.com/en/docs/convert/stringformat) function here. If you think it's unnecessary to print what is being sent via the custom event, you could simply take the content from line 321 and place it here on line 325, replacing the variable **szTmp** with the StringFormat function and using exactly the same expression as in line 321.

Returning to the issue of strings: the problem is that we need to pass numeric values inside the string. At this point, you might be wondering: why not use the lparam and dparam fields of the message? Why must we use sparam? The reason is simply the amount of data we need to pass. If we were passing only a few parameters, we could use lparam and dparam. But since the amount of information is large, we must use the sparam field. And this is where many people, especially beginners or aspiring programmers, make mistakes.

When you look at a number — for example, the value 65 — one application may interpret it as the number 65, while another may interpret it as the letter "A". Many readers may find this confusing, but the problem is that you are looking at 65 literally, instead of in binary form. In binary, it is represented as _**0100 0001**_ in 8 bits. This is the correct way to think when dealing with programming. So, when you look at a string, you should not think of it as readable characters or human-comprehensible text. Think of a string as one very large number - not just 1 byte, 256 bytes, or any other fixed size, but potentially an enormous sequence of bytes.

Thinking about it this way makes it easier to understand something else: how do you mark the end of a string? Some programming languages store a value at the start of the string - a "dead" value indicating the number of characters or bytes the string contains. BASIC and the old Pascal language are examples of this approach. In Pascal, the length is stored at the beginning of the string.

In such cases, strings - or more accurately, arrays - can contain any byte value from 0 to 255. While this approach works well in many scenarios, it is poor in others because fixed-size arrays waste memory. For example, you might only need 3 bytes, but if the compiler detects that sometimes you need 50 bytes in the same array, it will allocate the maximum - 50 bytes - even when you only need 3.

Because of this inefficiency, other languages do not use this string format. Instead, they define a specific code or value to mark the end of the string. The most common method is to use the NULL character. This is 0000 0000 in binary. The choice of this particular character comes from compiler implementation decisions. Any character could be used in theory. What matters here is how this affects our message when we need to send it inside a string.

Let’s go back to numeric values. Our Chart Trade must transmit mainly two types: short and double. A short value uses 16 bits or 2 bytes. I won't even go into the 'double' type; I'll keep the example with 'short'. Within Chart Trade, a short is used for the leverage level. The smallest leverage value is 1. In binary, that is: **_0000 0000 0000 0001_** (written here in 16-bit form).

However, even though short uses 2 bytes, a string is actually composed of 1-byte characters. This means that the first byte of the value 1 in the string represents the **NULL** character. That's right. Before any information can be read, the string would already be terminated. For this reason, we must convert the values into printable characters. In other words, regardless of the binary form of the data, we convert it into a human-readable (printable) form for transmission and then convert it back to its binary equivalent afterward. This is precisely why we need to use the StringFormat function here.

### Final Thoughts

Although this article might seem to end abruptly, I do not feel comfortable explaining the second necessary topic - the communication protocol between applications - in such a short space. This is a complex subject that deserves its own detailed treatment. I can simply say that line 321 creates the protocol we will use.

However, with this approach, the whole question of communication between programs remains completely vague and will not help you if you try to understand how to use this knowledge in your programs. Although MetaTrader 5 has been around for quite some time and many programmers are interested in it, I have not seen anyone using what I show here. Most people who want to achieve something start from scratch or create huge programs that, in most cases, are extremely difficult to maintain, adjust or improve. However, with the knowledge we share here, you will be able to develop smaller, much simpler programs that are also much easier to debug and improve.

That's why I wanted to take this opportunity from Chart Trade to explain in detail how the messaging process works. I hope that as a beginner programmer you will realize that we can do much more than we are usually expected to do.

As for the remaining events ( **CHARTEVENT\_OBJECT\_ENDEDIT**, **CHARTEVENT\_OBJECT\_CLICK**, and **CHARTEVENT\_OBJECT\_DELETE**) I am confident you will be able to understand them on your own, as their code is much simpler than what we've discussed here.

In the next article, I will explain exactly why line 321 has its specific format and how it affects the Expert Advisor code that will receive Chart Trade's events, enabling it to execute orders based on user interaction with its buttons.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12443](https://www.mql5.com/pt/articles/12443)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12443.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12443/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/493250)**

![Price Action Analysis Toolkit Development (Part 36): Unlocking Direct Python Access to MetaTrader 5 Market Streams](https://c.mql5.com/2/162/19065-price-action-analysis-toolkit-logo__1.png)[Price Action Analysis Toolkit Development (Part 36): Unlocking Direct Python Access to MetaTrader 5 Market Streams](https://www.mql5.com/en/articles/19065)

Harness the full potential of your MetaTrader 5 terminal by leveraging Python’s data-science ecosystem and the official MetaTrader 5 client library. This article demonstrates how to authenticate and stream live tick and minute-bar data directly into Parquet storage, apply sophisticated feature engineering with Ta and Prophet, and train a time-aware Gradient Boosting model. We then deploy a lightweight Flask service to serve trade signals in real time. Whether you’re building a hybrid quant framework or enhancing your EA with machine learning, you’ll walk away with a robust, end-to-end pipeline for data-driven algorithmic trading.

![Neural Networks in Trading: A Parameter-Efficient Transformer with Segmented Attention (PSformer)](https://c.mql5.com/2/102/Parameter-efficient_Transformer_with_segmented_attention_PSformer____LOGO.png)[Neural Networks in Trading: A Parameter-Efficient Transformer with Segmented Attention (PSformer)](https://www.mql5.com/en/articles/16439)

This article introduces the new PSformer framework, which adapts the architecture of the vanilla Transformer to solving problems related to multivariate time series forecasting. The framework is based on two key innovations: the Parameter Sharing (PS) mechanism and the Segment Attention (SegAtt).

![From Basic to Intermediate: Definitions (I)](https://c.mql5.com/2/103/Do_bcsico_ao_intermediurio_Defini3oes_I___LOGO.png)[From Basic to Intermediate: Definitions (I)](https://www.mql5.com/en/articles/15573)

In this article we will do things that many will find strange and completely out of context, but which, if used correctly, will make your learning much more fun and interesting: we will be able to build quite interesting things based on what is shown here. This will allow you to better understand the syntax of the MQL5 language. The materials provided here are for educational purposes only. It should not be considered in any way as a final application. Its purpose is not to explore the concepts presented.

![Automating Trading Strategies in MQL5 (Part 25): Trendline Trader with Least Squares Fit and Dynamic Signal Generation](https://c.mql5.com/2/162/19077-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 25): Trendline Trader with Least Squares Fit and Dynamic Signal Generation](https://www.mql5.com/en/articles/19077)

In this article, we develop a trendline trader program that uses least squares fit to detect support and resistance trendlines, generating dynamic buy and sell signals based on price touches and open positions based on generated signals.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/12443&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069541927805716156)

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