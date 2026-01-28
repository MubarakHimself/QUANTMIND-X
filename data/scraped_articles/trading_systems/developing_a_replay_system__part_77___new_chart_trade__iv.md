---
title: Developing a Replay System (Part 77): New Chart Trade (IV)
url: https://www.mql5.com/en/articles/12476
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:32:38.218245
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/12476&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069531572639565459)

MetaTrader 5 / Examples


### Introduction

In the previous article " [Developing a Replay System (Part 76): New Chart Trade (III)](https://www.mql5.com/en/articles/12443)", I explained the most critical part of the DispatchMessage code and began discussing how the communication process - or, more precisely, the communication protocol - should be designed.

Before we dive into the main topic of this article, we need to make a small adjustment to the code presented earlier. Everything previously explained remains valid. But this change is necessary in order to stabilize the system. After that, we can move on to the true focus of this article.

### Further Stabilizing the DispatchMessage Code

Due to a certain flaw in the interaction between the mouse indicator and Chart Trade, a small modification is required. I cannot fully explain why the interaction sometimes fails. But with the code adjustment shown below, the problem disappears.

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
292.                   if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft))
293.                   {
294.                      switch (CheckMousePosition(x = (short)lparam, y = (short)dparam))
295.                      {
296.                         case MSG_MAX_MIN:
297.                            if (sz < 0) m_Info.IsMaximized = (m_Info.IsMaximized ? false : true);
298.                            break;
299.                         case MSG_DAY_TRADE:
300.                            if ((m_Info.IsMaximized) && (sz < 0)) m_Info.IsDayTrade = (m_Info.IsDayTrade ? false : true);
301.                            break;
302.                         case MSG_LEVERAGE_VALUE:
303.                            if ((m_Info.IsMaximized) && (sz < 0)) CreateObjectEditable(obj = MSG_LEVERAGE_VALUE, m_Info.Leverage);
304.                            break;
305.                         case MSG_TAKE_VALUE:
306.                            if ((m_Info.IsMaximized) && (sz < 0)) CreateObjectEditable(obj = MSG_TAKE_VALUE, m_Info.FinanceTake);
307.                            break;
308.                         case MSG_STOP_VALUE:
309.                            if ((m_Info.IsMaximized) && (sz < 0)) CreateObjectEditable(obj = MSG_STOP_VALUE, m_Info.FinanceStop);
310.                            break;
311.                         case MSG_TITLE_IDE:
312.                            if (sx < 0)
313.                            {
314.                               DeleteObjectEdit();
315.                               ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
316.                               sx = x - (m_Info.IsMaximized ? m_Info.x : m_Info.minx);
317.                               sy = y - (m_Info.IsMaximized ? m_Info.y : m_Info.miny);
318.                            }
319.                            if ((mx = x - sx) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, mx);
320.                            if ((my = y - sy) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, my);
321.                            if (m_Info.IsMaximized)
322.                            {
323.                               m_Info.x = (mx > 0 ? mx : m_Info.x);
324.                               m_Info.y = (my > 0 ? my : m_Info.y);
325.                            }else
326.                            {
327.                               m_Info.minx = (mx > 0 ? mx : m_Info.minx);
328.                               m_Info.miny = (my > 0 ? my : m_Info.miny);
329.                            }
330.                            break;
331.                         case MSG_BUY_MARKET:
332.                            ev = evChartTradeBuy;
333.                         case MSG_SELL_MARKET:
334.                            ev = (ev != evChartTradeBuy ? evChartTradeSell : ev);
335.                         case MSG_CLOSE_POSITION:
336.                            if ((m_Info.IsMaximized) && (sz < 0))
337.                            {
338.                               string szTmp = StringFormat("%d?%s?%c?%d?%.2f?%.2f", ev, _Symbol, (m_Info.IsDayTrade ? 'D' : 'S'), m_Info.Leverage,
339.                                                          FinanceToPoints(m_Info.FinanceTake, m_Info.Leverage), FinanceToPoints(m_Info.FinanceStop, m_Info.Leverage));
340.                               PrintFormat("Send %s - Args ( %s )", EnumToString((EnumEvents) ev), szTmp);
341.                               sz = x;
342.                               EventChartCustom(GetInfoTerminal().ID, ev, 0, 0, szTmp);
343.                            }
344.                            break;
345.                      }
346.                      if (sz < 0)
347.                      {
348.                         sz = x;
349.                         AdjustTemplate();
350.                         if (obj == MSG_NULL) DeleteObjectEdit();
351.                      }
352.                   }else
353.                   {
354.                      sz = -1;
355.                      if (sx > 0)
356.                      {
357.                         ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
358.                         sx = sy = -1;
359.                      }
360.                   }
361.                   break;
362.                case CHARTEVENT_OBJECT_ENDEDIT:
363.                   switch (obj)
364.                   {
365.                      case MSG_LEVERAGE_VALUE:
366.                      case MSG_TAKE_VALUE:
367.                      case MSG_STOP_VALUE:
368.                         dvalue = StringToDouble(ObjectGetString(GetInfoTerminal().ID, m_Info.szObj_Editable, OBJPROP_TEXT));
369.                         if (obj == MSG_TAKE_VALUE)
370.                            m_Info.FinanceTake = (dvalue <= 0 ? m_Info.FinanceTake : dvalue);
371.                         else if (obj == MSG_STOP_VALUE)
372.                            m_Info.FinanceStop = (dvalue <= 0 ? m_Info.FinanceStop : dvalue);
373.                         else
374.                            m_Info.Leverage = (dvalue <= 0 ? m_Info.Leverage : (short)MathFloor(dvalue));
375.                         AdjustTemplate();
376.                         obj = MSG_NULL;
377.                         ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Editable);
378.                         break;
379.                   }
380.                   break;
381.                case CHARTEVENT_OBJECT_CLICK:
382.                   if (sparam == m_Info.szObj_Chart) if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft)) switch (obj = CheckMousePosition(x = (short)lparam, y = (short)dparam))
383.                   {
384.                      case MSG_DAY_TRADE:
385.                         m_Info.IsDayTrade = (m_Info.IsDayTrade ? false : true);
386.                         DeleteObjectEdit();
387.                         break;
388.                      case MSG_MAX_MIN:
389.                         m_Info.IsMaximized = (m_Info.IsMaximized ? false : true);
390.                         DeleteObjectEdit();
391.                         break;
392.                      case MSG_LEVERAGE_VALUE:
393.                         CreateObjectEditable(obj, m_Info.Leverage);
394.                         break;
395.                      case MSG_TAKE_VALUE:
396.                         CreateObjectEditable(obj, m_Info.FinanceTake);
397.                         break;
398.                      case MSG_STOP_VALUE:
399.                         CreateObjectEditable(obj, m_Info.FinanceStop);
400.                         break;
401.                   }
402.                   if (obj != MSG_NULL) AdjustTemplate();
403.                   break;
404.                case CHARTEVENT_OBJECT_DELETE:
405.                   if (sparam == m_Info.szObj_Chart) macro_CloseIndicator(C_Terminal::ERR_Unknown);
406.                   break;
407.             }
408.             ChartRedraw();
409.          }
410. //+------------------------------------------------------------------+
```

Fragment of the file C\_ChartFloatingRAD.mqh

Notice that some lines have been struck through, for example, lines 314 and 341. Both were moved inside the test located at line 346. This adjustment addresses a stability issue that occurred when clicking certain controls. The variable **sz** is used in each of the objects. This can be seen in lines 297, 300, 303, 306, and 309, as well as in the conditional tests at lines 312 and 336.

Compared with the previous version, this modification specifically stabilizes the interaction between the mouse indicator and Chart Trade. Previously, if the mouse indicator was loaded first, some Chart Trade controls would not respond correctly. The only workaround was to remove the indicator from the chart and reinsert it. Only then would the controls behave properly. Strange, to say the least.

Because of this, the **CHARTEVENT\_OBJECT\_CLICK** event should be removed from the processing code. All lines between 381 and 403 must therefore be deleted from the code shown in the last article. Since these changes don't alter any of the explanations previously given, we can now move on to the main subject of this article.

### Understanding the Planning Behind the Message Protocol

I cannot assume what you, dear reader, already know about computer communication systems. To avoid leaving anyone behind, I'll explain this from the ground up. If you already have experience with such protocols, much of this may be obvious, and you can skip ahead to the next section.

In the previous article, I explained why numeric values must be converted into their literal (string) equivalents. For example, if you need to transmit the binary value 0001 0011, you must convert it into the string "19". That means transmitting two bytes instead of one. While this may seem inefficient, the goal is not efficiency but clarity: ensuring that the information is correctly understood on the receiving side. Efficiency is desirable, of course, but accuracy takes priority.

As explained earlier, we'll use the 'sparam' field. The challenge now is this: if the information must be transmitted inside a single string, how do we determine where one piece of data ends and another begins? After all, this is the key question.

Designing a strategy for this is essential. You need a format that allows you to extract each piece of data later. Several approaches are possible, each with its own advantages and drawbacks. One option is to use fixed-length arrays for each field. Concatenated, they form the transmitted string. This approach has its advantages and disadvantages.

This makes indexing simple, since each block always has the same size. However, it wastes memory and bandwidth if fields don't fully use their allocated space. In fact, empty positions will take up bandwidth, in our case, memory.

To make it clearer, take a look at the image below:

![Figure 1](https://c.mql5.com/2/141/Image_01__1.png)

The blue block represents the NULL character that marks the end of the string. Notice how one of the arrays has empty positions, wasting space. It is situations like these that make using a fixed size array difficult.

An alternative is to use variable-length arrays sized exactly to the data. This avoids wasted space and makes extraction easier. The drawback is added complexity. We'll need to write more code. As we create more code, we need to test it. Additionally, in many cases, there's the risk of data loss if the information exceeds the expected size. Figure 2 shows an idealized version of this approach:

![Figure 2](https://c.mql5.com/2/141/Image_02__1.png)

As you can see in the first image, the blue block indicates where the character that marks the end of the line is. Note that here we have an idealized state where each color represents information that will be compressed into a row. Although in some cases this may complicate the extraction of the string, for a complete and correct recovery of the original information this case seems quite adequate. However, in this case, each set will have a fixed size. Then we get a more advanced approach than the one shown in the first figure. But problems arise if, for example, a field expected to be two bytes suddenly needs three.

That would be the worst case scenario. One of the bytes will be lost during the assembly of the green set. What if the green set could grow to accommodate this extra byte? Is it possible to do this? No. If we do this, all information after the green set will be compromised, and the receiver will not be able to understand that there are three bytes in the green set, not two as he expected.

This demonstrates that protocol design is not trivial. If the receiver expects fixed sizes and the sender changes them, communication fails. A different approach is needed.

One possibility is variable-length blocks, where each field can be as large as needed. That would be much better, wouldn't it? But then a new problem arises: how does the receiver know where one field ends and the next begins? This requires delimiters or markers. But here you need to think carefully, otherwise you may end up sending a message that will not be understood by the recipient, even if it is right for you. But how is it possible that for the recipient everything is not so clear?

Think about it: each section of a message can be any size, allowing us to convey virtually any type of information. The information must be arranged in a certain sequence, whatever it may be, but the sequence must always be observed. So far so good. Again, how do we indicate that one section is finished and another is beginning?

This i sthe hardest part. It all depends on the type of data we put into the string. If we use all 255 possible values in a byte - again, we must avoid the null character, so 255 rather than 256 - we have a big problem: how to indicate that we are providing different information in a block of strings. If we reduce the values to characters between 32 and 127, we will have to do some more assembly work. But this allows us to use any value between 128 and 255 as a marking symbol.

However, we can limit things even further. We can use only alphanumeric characters to convey the required information. Thus, we can reserve punctuation marks as delimiters. Why can we do this here? This works because the data we need to transmit is relatively simple: asset names and numeric values such as leverage, take-profit, and stop-loss levels. These are configured in Chart Trade but must be transmitted to the Expert Advisor.

But in addition to these values, which are quite simple, we need to pass one more. Event though MetaTrader 5 already handles this, we will include the operation type, to ensure communication remains under control. This is not a necessary thing.

Keep in mind: communication doesn't have to occur only inside one terminal. With proper network protocols, one computer could run the terminal while another manages orders and positions. Even a modest machine could then handle trading as if it were much more powerful.

We won't go into further details. The idea is to show how the connection will actually work.

### Combining the Best of Both Worlds

Our chosen solution combines the strengths of both fixed and variable-length approaches. We will transmit data as alphanumeric strings, use delimiters for clarity, and still allow indexing. Each field can occupy as many characters as necessary, ensuring nothing is lost.

Refer to line 338 of the earlier code fragment. To make this even clearer, let's look at a practical sending example.

![Figure 3](https://c.mql5.com/2/141/Image_03__1.png)

Figure 3 shows a real example of a Chart Trade transmission. At first glance the message looks confusing. To understand it, we need to look at what is happening in line 338 of the fragment. The message still follows a well-defined protocol. Essentially, we are using the best of both worlds. It allows blocks in a row to be of any size, while at the same time indexing the information in the row in a specific way.

You may not understand how indexing works, and it's not really that obvious, but it does exist. Notice the D symbol in the message. Observe that it is both preceded and followed by the same character. At this point, we have one and only one block, which is filled by the character D. This situation does not occur anywhere else in the message. That indicates that this single character can be indexed in some way. However, this will become clearer later. For now, let's focus on understanding what is happening here.

The first block, which in this particular message contains a single character, indicates the type of operation to be executed. Once again, MetaTrader 5 will provide this information to our Expert Advisor, as will be seen later. Here, however, I am considering the fact that Chart Trade will indeed communicate with the Expert Advisor - whether over the network, by email, or by some other means. For this reason, I am explicitly specifying the type of operation to be executed.

A detail worth noting: the value 9 corresponds to a market buy event. But this value may differ. For example, if the block contained the value 11, this would indeed indicate the closing of all positions. In that case, the block would contain two characters rather than the single one shown above. But why is 9 used for buy and 11 for close-all? Where do these values come from? This is a perfectly valid question. Looking at line 338, the first value placed in the string is a ushort. But that alone does not explain why 9 indicates buy and 11 indicates close-all. Indeed, it does not.

Now look at line 278. Where does this value come from? It comes from the header file Defines.mqh. Pay close attention: inside Defines.mqh there is an enumeration called EnumEvents. This enumeration starts at zero, and with each new element, the compiler increments the value by one. Counting from the first event, evHideMouse, the ninth event is evChartTradeBuy, and the eleventh is evChartTradeCloseAll. So now you know where these values that appear at the start of the string come from: They derive from the EnumEvents enumeration.

Let's continue Notice that all the question-mark characters are highlighted in violet. The NULL character, which terminates the string, is marked in blue. Since each block is delimited by a question mark, we can insert as many alphanumeric characters as needed to send the message. But there are important details. This message must be constructed in a specific sequence. Remember: the receiver expects to receive the data in a given order. Even though, in theory, the information could be placed in a random order, the receiver - in the version I will demonstrate - does not expect that.

The next block of characters provides the name of the asset for which the order is being placed. Again, this would not be necessary if the Expert Advisor and Chart Trade were operating on the same chart. But I am considering the case where they are not.

Moreover, in the future you will see that these message details can be used for other purposes. In the example, the asset is BOVA11, which is an ETF. The asset name complicates matters if no delimiter is used. Because in some markets the asset symbol consists of four alphanumeric characters, while in others it may have five. In this case we have five. Even on B3 (the Brazilian stock exchange), many assets use four-character symbols.

There is another point. Remember that the goal here is to design a Chart Trade that can also be used in replay/simulation In such cases, the symbol name could have any number of alphanumeric characters For this reason, a dynamically sized block is highly desirable.

Now we return to our D character. At this stage, I want you to look again at line 338. If the operation is not of the type that closes within the same day (that is, a day trade) the character in this block will be different. It will be replaced by S. You could choose any other character if you prefer, but remember to also update the receiver; otherwise communication will break down, as the receiver may not interpret the letter or character sequence correctly.

Immediately after this, we have a literal value: 250. What does this represent? Again, look at line 338: this value is the desired leverage level. Here lies an interesting point. We are using three numeric characters to represent the leverage value.

Could we instead use a binary value? This might seem appropriate, since leverage cannot be zero. But there is a condition that prevents this. Not because we cannot format the string with a character corresponding to 250, but because of the delimiter character itself. Take an ASCII table and check the value of the question mark. It is 63.

To make this easier, I have included below an ASCII table from character 0 through character 128.

![Figure 4](https://c.mql5.com/2/141/Image_04__2.png)

Why is 63 important to us? Because any leverage value that includes 63 will be interpreted by the receiver (which I will show shortly) as a delimiter. In other words, the receiver will not recognize that the fourth block represents the leverage level.

You might think: What if I offset the leverage values by adding 63? Since leverage will never be zero, the first valid value would become 64. Problem solved? I wish it were that simple. But no: by adding 63 to the leverage, you are only postponing the problem.

Here is why. You may be wondering: What's the issue? If I add 63, all values will be greater than 63? And that's the whole point. The issue is that in programming no value is infinite. Every value is limited to a maximum, which depends on the word size being used. Even on a 64-bit processor (such as the one MetaTrader 5 currently runs on), the system still relies on 8-bit concepts for character handling.

This means that even with a 64-bit processor, you cannot actually count up to 2^64 (18,446,744,073,709,551,615) in terms of characters. At most, you can count to 255, which corresponds to 2^8. Why? Could this be worked around? Yes. One approach is to use a character set other than ASCII. Unicode, for example.

But there is another problem. [StringFormat](https://www.mql5.com/en/docs/convert/stringformat) does not use Unicode, at least at the time of writing. String functions in MQL5 generally follow C/C++ principles and therefore use ASCII. Even though C/C++ can handle Unicode, this was not originally the case.

Thus, even if you add 63 to the leverage, every 255 positions you will still generate a composite value. This will be a combination of a factor and the current count. The factor shows how many times the counting cycle to 255 occurred. The value 575 is factor 2 plus 63. And so on.

To represent this correctly, you would need two bytes: the second byte always being 63 at some stage. And the first byte indicating the factor, that is, how many times the maximum count has been reached (2^8, as explained above). This introduces various mathematical implications that I will not expand on here, as they are beyond the scope of this article.

To conclude the explanation of how this message protocol is being built: note that we also have two values that could be represented as doubles or floats. For the same reasons discussed in relation to leverage, these must be written as literal values. That is why they appear as they do in the image.

But you may now be asking: Why do these values look this way? What do they represent? They may seem strange because you may be forgetting to check line 338 of the code snippet. There, a monetary value is being converted into points. Thus, the value 3.60 corresponds to $900, and 3.02 corresponds to $755.

Why not use monetary values directly instead of points? The reason is simplicity. It is far simpler to implement an Expert Advisor using already-converted values than to perform the conversions internally. This may not be entirely clear now, but later you will see the benefits. We will explore this in more depth in the future, as more explanations are needed to fully understand the advantages of sending pre-converted values directly to the Expert Advisor. But, as I already said, this is for the future.

### Conclusion

In this article, I have tried to explain in as much detail as possible how to create a communication protocol. The topic is not yet complete, since we still need to look at the part responsible for receiving these messages. However, I believe the main objective has been achieved: to show why you must take care when designing your communication protocol, particularly if you choose to use something different from what I present here.

This is something you must plan now. If you leave it until later, you will end up struggling to make your protocol properly transfer the information. And this information is essential for the Expert Advisor to know what to do and how to do it. Do not postpone it. Study now, and start making the necessary adjustments you think should be implemented. In the next article, we will finally look at the receiving side - that is, the Expert Advisor itself.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12476](https://www.mql5.com/pt/articles/12476)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12476.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12476/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/494117)**

![Chart Synchronization for Easier Technical Analysis](https://c.mql5.com/2/165/18937-chart-synchronization-for-easier-logo.png)[Chart Synchronization for Easier Technical Analysis](https://www.mql5.com/en/articles/18937)

Chart Synchronization for Easier Technical Analysis is a tool that ensures all chart timeframes display consistent graphical objects like trendlines, rectangles, or indicators across different timeframes for a single symbol. Actions such as panning, zooming, or symbol changes are mirrored across all synced charts, allowing traders to seamlessly view and compare the same price action context in multiple timeframes.

![Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://c.mql5.com/2/165/19285-simplifying-databases-in-mql5-logo__2.png)[Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://www.mql5.com/en/articles/19285)

We explore how to manipulate databases in MQL5 using the language's native functions. We cover everything from table creation, insertion, updating, and deletion to data import and export, all with sample code. The content serves as a solid foundation for understanding the internal mechanics of data access, paving the way for the discussion of ORM, where we'll build one in MQL5.

![From Novice to Expert: Mastering Detailed Trading Reports with Reporting EA](https://c.mql5.com/2/165/19006-from-novice-to-expert-mastering-logo.png)[From Novice to Expert: Mastering Detailed Trading Reports with Reporting EA](https://www.mql5.com/en/articles/19006)

In this article, we delve into enhancing the details of trading reports and delivering the final document via email in PDF format. This marks a progression from our previous work, as we continue exploring how to harness the power of MQL5 and Python to generate and schedule trading reports in the most convenient and professional formats. Join us in this discussion to learn more about optimizing trading report generation within the MQL5 ecosystem.

![Trend criteria in trading](https://c.mql5.com/2/106/Trend_Criteria_in_Trading_LOGO.png)[Trend criteria in trading](https://www.mql5.com/en/articles/16678)

Trends are an important part of many trading strategies. In this article, we will look at some of the tools used to identify trends and their characteristics. Understanding and correctly interpreting trends can significantly improve trading efficiency and minimize risks.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lfgtgsjfhbcviybwxzuxqaijwvyvgxsc&ssn=1769182356405449673&ssn_dr=0&ssn_sr=0&fv_date=1769182356&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12476&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2077)%3A%20New%20Chart%20Trade%20(IV)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918235622176094&fz_uniq=5069531572639565459&sv=2552)

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