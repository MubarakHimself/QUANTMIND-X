---
title: Developing a Replay System (Part 39): Paving the Path (III)
url: https://www.mql5.com/en/articles/11599
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:13:57.871976
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/11599&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070118939482067012)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 38): Paving the Path (II)](https://www.mql5.com/en/articles/11591), I explained and demonstrated how to send data, in our case from the Expert Advisor to the indicator. This allowed us to configure the indicator and to make it return some plausible information.

So, it's time to move in the opposite direction, that is, to make the indicator tell the caller, which in our case is the Expert Advisor, something that has some kind of meaning. We need to know how we should proceed.

There are several things here. Some of which are quite easy to explain, others not so much. Accordingly, some of them will be easy to demonstrate, while others will require more time. But one way or another, in this article we will create the necessary foundation for building our Chart Trader. The Chat Trader that will be developed here in the replay/simulation system will be very similar to the one presented in the last article.

But it will be different from what we saw in the article: [Developing a trading Expert Advisor from scratch (Part 30): CHART TRADE as an indicator?!](https://www.mql5.com/en/articles/10653). What we will do here may seem like child's play, thanks to the chosen method of creating the Chart Trader. But let's not spoil the surprise. Otherwise it won't be interesting. To make it easier to explain what we will be doing, we first need to understand a few more things, which will be covered in this article.

### Initial construction of the model

Despite the fact that everything described in the last two articles is to some extent fair and works quite well, which is what I basically wanted to show, now we have another problem. We only send data from the EA to the indicator. This means there is communication. However, we are still not reading the indicator data. In this case, we need to implement the ability to read data from the indicator.

To make our lives easier, we need to prepare our code.

The first thing we do is create a header file shown below.

**Header file**:

```
1. #property copyright "Daniel Jose"
2. #property link      ""
3. //+------------------------------------------------------------------+
4. #define def_ShortName        "SWAP MSG"
5. //+------------------------------------------------------------------+
```

This header file is called **Defines.mqh**. It should be saved in the **Includes** directory, in the **Mode Swap** subfolder. The only line that is interesting to us is line 4. Here we set the name that will be used both in the EA and in the indicator, just for ease of use. Because you can specify a name in the indicator and forget to specify the same name in the EA. I'm not talking about the file name. I mean the name by which the indicator will be recognized in MetaTrader 5.

To make it clearer, let's look at the indicator code below.

Indicator source code:

```
01. #property copyright "Daniel Jose"
02. #property link      ""
03. #property version   "1.00"
04. #property indicator_chart_window
05. #property indicator_plots 0
06. #property indicator_buffers 0
07. //+------------------------------------------------------------------+
08. #include <Mode Swap\Defines.mqh>
09. //+------------------------------------------------------------------+
10. #define def_ShortNameTmp    def_ShortName + "_Tmp"
11. //+------------------------------------------------------------------+
12. input double user00 = 0.0;
13. //+------------------------------------------------------------------+
14. long m_id;
15. //+------------------------------------------------------------------+
16. int OnInit()
17. {
18.     m_id = ChartID();
19.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortNameTmp);
20.     if (ChartWindowFind(m_id, def_ShortName) != -1)
21.     {
22.             ChartIndicatorDelete(m_id, 0, def_ShortNameTmp);
23.             Print("Only one instance is allowed...");
24.             return INIT_FAILED;
25.     }
26.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
27.     Print("Indicator configured with the following value:", user00);
28.
29.     return INIT_SUCCEEDED;
30. }
31. //+------------------------------------------------------------------+
32. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
33. {
34.     return rates_total;
35. }
36. //+------------------------------------------------------------------+
```

Pay attention to the code present on line 08. On this line, we ask the compiler to include the code from the above header file. The code remains the same, except that now we no longer have this definition in the indicator code.

Thus, every explanation previously made remains valid. Now let's look at the EA code below.

EA source code:

```
01. #property copyright "Daniel Jose"
02. #property link      ""
03. #property version   "1.00"
04. //+------------------------------------------------------------------+
05. #include <Mode Swap\Defines.mqh>
06. //+------------------------------------------------------------------+
07. #define def_SWAP "Mode Swap\\Swap MSG.ex5"
08. #resource "\\Indicators\\" + def_SWAP
09. //+------------------------------------------------------------------+
10. input double user00 = 2.2;
11. //+------------------------------------------------------------------+
12. int m_handle;
13. long m_id;
14. //+------------------------------------------------------------------+
15. int OnInit()
16. {
17.     m_id = ChartID();
18.
19.     EraseIndicator();
20.     m_handle = iCustom(NULL, PERIOD_CURRENT, "::" + def_SWAP, user00);
21.     ChartIndicatorAdd(m_id, 0, m_handle);
22.
23.     Print("Indicator loading result:", m_handle != INVALID_HANDLE ? "Success" : "Failed");
24.
25.     return INIT_SUCCEEDED;
26. }
27. //+------------------------------------------------------------------+
28. void OnDeinit(const int reason)
29. {
30.     EraseIndicator();
31. }
32. //+------------------------------------------------------------------+
33. void OnTick()
34. {
35. }
36. //+------------------------------------------------------------------+
37. void EraseIndicator(void)
38. {
39.     if ((m_handle = ChartIndicatorGet(m_id, 0, def_ShortName)) == INVALID_HANDLE) return;
40.     ChartIndicatorDelete(m_id, 0, def_ShortName);
41.     IndicatorRelease(m_handle);
42. }
43. //+------------------------------------------------------------------+
```

Here we have something that is **_very different_**, but upon closer examination the difference is not that big. It seems the only real difference is the presence of a function called **_EraseIndicator_**. It is on line 37. This function is called at two points: the first is on line 19, and the second is on line 30. I don't think anyone has any questions about why it is called on line 30. But what about line 19? Do you know the reason for this call?

The reason for the call from line 19 is precisely to ensure synchronization between the value reported in the EA and the value that will be sent to the indicator. Also, there is something that requires attentive consideration. Look at line 39 of the EA code. Please note that when calling the function, both on line 19 and 30, we check whether the indicator is on the chart. If the indicator is not there, the code will terminate. In this case, both line 40, which removes the indicator from the chart, and line 41, which releases the handle, will not be executed.

But why do we need to remove the indicator from the chart using line 40? The reason for this is not in the EA code, but in the indicator code.

Looking at the indicator code, you will notice that if it is on the chart, then the moment the EA tries to change its value, line 20 will block the indicator. This will prevent it from updating because the system will detect that you are trying to place a new indicator on the chart. Although the new indicator is not actually added, line 20 will perceive it that way. This is why we need to remove the indicator from the chart.

But let's get back to the EA code. Any idea what lines 07 and 08 represent? And why is line 20 different from the one discussed in the previous article?

This is the very point where the transition from the main to another area begins. Line 07 creates a definition, but that definition specifies where the indicator code is located. Thus, when the compiler creates the EA executable and finds line 08, it will look for the executable specified on line 07. If the executable is not there, the compiler will start compiling the code responsible for creating such an executable file. That is, in line 08 we make the indicator an internal resource of the EA.

This is quite interesting. But there are pros and cons. One of the advantages that is worth noting is that after compiling the EA, you can delete the executable file of the indicator. However, you don't need to do this yet. There is a detail that needs to be taken into account to make this possible. A disadvantage that is also worth mentioning is that after compiling the EA, in case of problems with the indicator, you will have to recompile the entire EA code. Even if the problem is solely in the indicator. Please be calm. There is one detail that must be taken into account.

What kind of detail is this? This is a rather subtle point that is almost unnoticeable. This is why every good programmer is also a very good observer. So, here is the detail. Look at line 20 of the EA code. Notice that in the declaration of the third parameter a small chain appears ( **::**). This little chain that comes before the definition is the [scope resolution operator](https://www.mql5.com/en/docs/basis/operations/other#context_allow).

The very fact that the operator is here indicates that we will be using something in a different way than we might imagine. Every time this operator appears, we are somehow explicitly telling the compiler what should be done. Usually there is some decision that the compiler should make, but we need it to explicitly know what it is.

In the specific case shown in the code, we tell the compiler to use the indicator found as a resource in the EA's executable file. This means that we can simply delete the indicator executable file after successful compilation of the EA. And the adviser will still be able to use the correct indicator.

If this scope resolution operator is not added as shown in the example below, then even though the compiler includes the indicator as an EA resource, the EA will actually use the executable file that is located in the directory specified in the definition. In this case we would need to port the indicator together with the EA in a separate file.

```
20.     m_handle = iCustom(NULL, PERIOD_CURRENT, def_SWAP, user00);
```

This simple detail is crucial. In the case of line 20, in the full code, after compiling the EA, you can delete the executable file of the indicator. There is one more detail: if an indicator is added and used as the EA resource, then before compiling the latter it is advisable to delete the executable file of the indicator. This will guarantee that the latest code present in the indicator will actually be compiled. Typically, in systems where multiple compilations are used we have a helper - a file called **MAKE**.

If used, each time the code is compiled again, MAKE will adequately remove executable files. Absolutely automatically. It compares the last compilation time with the last modification time of some files used in the executable's source code. If any changes are detected, MAKE will remove the executable, forcing the compiler to create a new one. At the moment, MQL5 does not have such an opportunity, but who knows, perhaps in the future the developers will decide to add it.

All of the above explanations demonstrate what we can do, how we should act, and how we will act in the future. But we still don't have the ability to read any data from the indicator.

Let's implement one very simple version of communication. To explain this in great detail and consistently, let's move on to a new topic.

### Reading data from the indicator

In fact, there is only one way to read any information presented in the indicator which is to read its buffer. But there is a small nuance here: the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) function does not allow us to work with any type of data present in the buffer. Not in its natural form. If you look at the declaration of this function, you'll see what I mean.

CopyBuffer function declaration:

```
int  CopyBuffer(
   int       indicator_handle,     // indicator handle
   int       buffer_num,           // number of indicator buffer
   int       start_pos,            // starting position
   int       count,                // amount to copy
   double    buffer[]              // destination array to copy
   );
```

To make it clearer, you can see above how one of the variants of the CopyBuffer function is declared. I say one of the variants because there are three, but what we really care about is the last variable present in the declaration. Pay attention to its type – double, that is, we can only return values of double type. This is only in theory, because in practice we can return anything. And in the article [Developing a trading Expert Advisor from scratch (Part 17): Accessing data on the web (III)](https://www.mql5.com/en/articles/10447) I showed a way to get around this limitation.

In this sequence we will also use something similar, but on a different level. Perhaps this will be a little larger and more complex for beginners. But the idea will be the same as in the mentioned article. If you need more details, please refer to the mentioned article to understand exactly what we will do. Here we will simply do this, and I will not go into the details.

So, this is something already familiar to us. Let's change the indicator source code as shown below. Note: I will move gradually so that you can follow and understand what will be done.

Indicator source code:

```
01. #property copyright "Daniel Jose"
02. #property link      ""
03. #property version   "1.00"
04. #property indicator_chart_window
05. #property indicator_plots 0
06. #property indicator_buffers 1
07. //+------------------------------------------------------------------+
08. #include <Mode Swap\Defines.mqh>
09. //+------------------------------------------------------------------+
10. #define def_ShortNameTmp    def_ShortName + "_Tmp"
11. //+------------------------------------------------------------------+
12. input double user00 = 0.0;
13. //+------------------------------------------------------------------+
14. long m_id;
15. double m_Buff[];
16. //+------------------------------------------------------------------+
17. int OnInit()
18. {
19.     m_id = ChartID();
20.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortNameTmp);
21.     if (ChartWindowFind(m_id, def_ShortName) != -1)
22.     {
23.             ChartIndicatorDelete(m_id, 0, def_ShortNameTmp);
24.             Print("Only one instance is allowed...");
25.             return INIT_FAILED;
26.     }
27.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
28.     Print("Indicator configured with the following value:", user00);
29.
30.     SetIndexBuffer(0, m_Buff, INDICATOR_CALCULATIONS);
31.     ArrayInitialize(m_Buff, EMPTY_VALUE);
32.
33.     return INIT_SUCCEEDED;
34. }
35. //+------------------------------------------------------------------+
36. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
37. {
38.     m_Buff[rates_total - 1] = user00 * 2.0;
39.
40.     return rates_total;
41. }
42. //+------------------------------------------------------------------+
```

If you compare the source code shown above with the source code presented at the beginning of the article, you will notice only a few small differences. Like I said, we'll move slowly. Actually, we have a change on line 06 which states that the indicator will have a buffer present. This buffer will be visible outside the indicator. We are going to talk about it later. First, let's figure out what happens in the indicator.

Next, on line 15 we have a declaration of a global variable. However, this variable will only be visible to the indicator code or those parts of it that relate to it. This variable, which is a buffer, requires initialization. This is done on lines 30 and 31. Now we have our buffer initialized and available, with some type of information.

However, if you read it, you will only get useless data. We need some efficient way to load data into the buffer. This can be done in different ways, but technically, the best and safest way is the one that is already in use. For this reason, we have line 38.

You may notice that on this line, I'm not pointing to any area in the buffer. Why?

The reason is that we need to somehow standardize the process. If we put information, in this case the result of multiplying the value reported by the EA by 2.0, it can be difficult to know where to look for really useful information. We try to understand things calmly. But in the future everything will become more complicated.

Therefore, we use our own indicator calculation system so that the information is always placed in the same position.

To make it clearer, let's look at the EA source code below.

EA source code:

```
01. #property copyright "Daniel Jose"
02. #property link      ""
03. #property version   "1.00"
04. //+------------------------------------------------------------------+
05. #include <Mode Swap\Defines.mqh>
06. //+------------------------------------------------------------------+
07. #define def_SWAP "Indicators\\Mode Swap\\Swap MSG.ex5"
08. #resource "\\" + def_SWAP
09. //+------------------------------------------------------------------+
10. input double user00 = 2.2;
11. //+------------------------------------------------------------------+
12. int m_handle;
13. long m_id;
14. //+------------------------------------------------------------------+
15. int OnInit()
16. {
17.     double Buff[];
18.
19.     m_id = ChartID();
20.
21.     if ((m_handle = ChartIndicatorGet(m_id, 0, def_ShortName)) == INVALID_HANDLE)
22.     {
23.             m_handle = iCustom(NULL, PERIOD_CURRENT, "::" + def_SWAP, user00);
24.             ChartIndicatorAdd(m_id, 0, m_handle);
25.     }
26.
27.     Print ("Buffer reading:", (m_handle == INVALID_HANDLE ? "Error..." : CopyBuffer(m_handle, 0, 0, 1, Buff) > 0 ?  (string)Buff[0] : " Failed."));
28.
29.     return INIT_SUCCEEDED;
30. }
31. //+------------------------------------------------------------------+
32. void OnDeinit(const int reason)
33. {
34.     ChartIndicatorDelete(m_id, 0, def_ShortName);
35.     IndicatorRelease(m_handle);
36. }
37. //+------------------------------------------------------------------+
38. void OnTick()
39. {
40. }
41. //+------------------------------------------------------------------+
```

Please note that the code has not changed much from what was presented earlier. We remember that here the indicator will be used as a resource, so after compiling the EA we can delete its executable file. Let's see what has been added to the code.

First, we now have a variable declaration on line 17. This will be our return buffer or indicator reading buffer. In addition to this line 17, we also have a new line 27.

Many may find this line 27 a real chaos, but it is only two nested ternary operators. First we check whether the indicator handle is valid; if not, a corresponding warning will appear in the terminal message window. If the handle is valid, we read one position from the indicator buffer. Which position will be read? The first one. Not so clear? Calm down, relax, I'll explain everything later. If buffer reading is successful, we will print the value it contains. If not, another error message will be displayed. It will be different from the first one, indicating a different reason for the failure.

Now let's see why when we read the buffer as shown on line 27, we read the first position and not the last.

### Understanding the buffer writing and reading process

This is an image from the MQL5 documentation. It clearly illustrates what I am about to explain. See below:

![Figure 01](https://c.mql5.com/2/49/001__2.png)

Figure 01 – Writing and reading the indicator buffer

This image clearly explains what is happening. Anyway, let's see in detail how data is written to the indicator buffer and how it is read from it.

When writing to the indicator buffer, it is important to always write to a specific position. As new bars appear, new positions automatically appear in the buffer. The problem is that at each timeframe you will have a different buffer size.

You may wonder where the problem is. There is not just one problem, there are several. To begin with, suppose, in the indicator, we decide to write to the zero position using the following code.

**Code snippet - model 01:**

```
35. //+------------------------------------------------------------------+
36. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
37. {
38.     m_Buff[0] = user00 * 3.0;
39.
40.     return rates_total;
41. }
42. //+------------------------------------------------------------------+
```

If you change the lines in the indicator code discussed in the previous topic to this code, you will literally write to the zero position of the buffer, as seen in line 38. But there is a problem with this approach.

The problem appears not when writing, but when trying to read the buffer outside the indicator code, for example, in an EA.

When reading a buffer, MetaTrader 5 typically returns a maximum of 2000 positions. We have what is shown in Figure 01. If by chance the amount of data present in the indicator exceeds these 2000 positions, problems will arise. Remember that we wrote to the zero position, but the zero position **IS NOT** the same zero position referenced by CopyBuffer. For CopyBuffer, this zero position, which is in the indicator code, may actually be position 2001, and if this is the case, it will not be possible to read the zero position from the indicator buffer. That value will be fixed in the indicator.

For this reason, we do not write to zero position and start writing from the position that is at the end of the buffer. Still not clear?

The zero position in the indicator buffer should always be considered as the position "rates\_total - 1". This is the position to which we write in the indicator code discussed in the previous topic. Precisely because of this, when we read the indicator buffer via CopyBuffer, when outputting the value, we actually use the zero index.

Perhaps this is still not entirely clear. To clarify, let's look at another code example, where we pass data to the indicator, but return not just one value, as before, but several. One of them will be a simple string.

It is noteworthy that we can write to the buffer either in reverse order or in the usual way, or rather, following the order: first information - first value, second information - second value. If the reverse is done, then it will be like this: first information - last value, second information - last value minus one position, and so on. To make it easier to interpret, not when writing but when reading, let's do things in the usual way.

Let's first change the header file, which can be seen below.

**Defines.mqh file source code**:

```
01. #property copyright "Daniel Jose"
02. //+------------------------------------------------------------------+
03. #define def_ShortName       "SWAP MSG"
04. //+------------------------------------------------------------------+
05. union uCharDouble
06. {
07.     double  dValue;
08.     char    cInfo[sizeof(double)];
09. };
10. //+------------------------------------------------------------------+
```

As you can see, between lines 05 and 09 we have a union. This union will allow us to pass text-type data using a double-type value. If you are seeing this for the first time, it may seem strange. But we've done this before. An example of this can be seen in the article: [Developing a trading Expert Advisor from scratch (Part 17): Accessing data on the web (III)](https://www.mql5.com/en/articles/10447). But let's return to our question. Now we have a way to send a small string from the indicator to the EA. The reason for using the double value is that we cannot send a value of another type through CopyBuffer. We must use the double type.

After making this change in the Defines.mqh file, we can move on to the indicator source code.

**Updated indicator source code allowing to write to more than one position**:

```
01. #property copyright "Daniel Jose"
02. #property version   "1.00"
03. #property indicator_chart_window
04. #property indicator_plots 0
05. #property indicator_buffers 1
06. //+------------------------------------------------------------------+
07. #include <Mode Swap\Defines.mqh>
08. //+------------------------------------------------------------------+
09. #define def_ShortNameTmp    def_ShortName + "_Tmp"
10. //+------------------------------------------------------------------+
11. input double user00 = 0.0;
12. //+------------------------------------------------------------------+
13. long m_id;
14. double m_Buff[];
15. //+------------------------------------------------------------------+
16. int OnInit()
17. {
18.     m_id = ChartID();
19.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortNameTmp);
20.     if (ChartWindowFind(m_id, def_ShortName) != -1)
21.     {
22.             ChartIndicatorDelete(m_id, 0, def_ShortNameTmp);
23.             Print("Only one instance is allowed...");
24.             return INIT_FAILED;
25.     }
26.     IndicatorSetString(INDICATOR_SHORTNAME, def_ShortName);
27.     Print("Indicator configured with the following value:", user00);
28.
29.     SetIndexBuffer(0, m_Buff, INDICATOR_CALCULATIONS);
30.     ArrayInitialize(m_Buff, EMPTY_VALUE);
31.
32.     return INIT_SUCCEEDED;
33. }
34. //+------------------------------------------------------------------+
35. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
36. {
37.     uCharDouble info;
38.     int pos = rates_total - 3;
39.
40.     StringToCharArray("Config", info.cInfo);
41.     m_Buff[pos + 0] = info.dValue;
42.     m_Buff[pos + 1] = user00 * 2.0;
43.     m_Buff[pos + 2] = user00 * 2.5;
44.
45.     return rates_total;
46. }
47. //+------------------------------------------------------------------+
```

Let's take a closer look at the above code. Only starting from line 37 there are some changes in relation to the code presented at the beginning of the article. Why is that? Because we will now assemble broader information returned through the CopyBuffer. No other part of the code needs to be modified, only the code starting at line 37.

Now let's figure out what's going on. On line 37 we declare a variable that will be used to convert the string to a double value. Note that the string length is limited to 8 characters. If the information contains more characters, an array must be provided for this, always considering the information in blocks of 8.

On line 38 we declare a variable that will be used to write information in the normal way, that is, as if we were writing text from left to right. In the case of Arabic, writing will be done from right to left. I think the idea is clear. On the same line 38 we indicate the number of double values that we will post. In our case we have 3 values.

On line 40, we convert string to an array of characters. This will give us the value that will be used in our first position. Then, on line 41, we put this value into the buffer.

On lines 42 and 43 we perform a simple calculation to have some data to read. Thus, you can show how reading will happen later when you need to access more positions in the buffer.

Basically, that's all on the issue of the indicator. At this particular moment there is nothing more to discuss about communication. Let's see how to proceed to recognize and read the buffer created in the indicator. To do this, let's turn to the already updated EA code presented below.

**EA source code updated to read more than one position**:

```
01. #property copyright "Daniel Jose"
02. #property version   "1.00"
03. //+------------------------------------------------------------------+
04. #include <Mode Swap\Defines.mqh>
05. //+------------------------------------------------------------------+
06. #define def_SWAP "Indicators\\Mode Swap\\Swap MSG.ex5"
07. #resource "\\" + def_SWAP
08. //+------------------------------------------------------------------+
09. input double user00 = 2.2;
10. //+------------------------------------------------------------------+
11. int m_handle;
12. long m_id;
13. //+------------------------------------------------------------------+
14. int OnInit()
15. {
16.     double Buff[];
17.     uCharDouble Info;
18.     int iRet;
19.     string szInfo;
20.
21.     m_id = ChartID();
22.     if ((m_handle = ChartIndicatorGet(m_id, 0, def_ShortName)) == INVALID_HANDLE)
23.     {
24.             m_handle = iCustom(NULL, PERIOD_CURRENT, "::" + def_SWAP, user00);
25.             ChartIndicatorAdd(m_id, 0, m_handle);
26.     }
27.     ArraySetAsSeries(Buff, false);
28.     if (m_handle == INVALID_HANDLE) szInfo = "Invalid handler to read the buffer.";
29.     else
30.     {
31.             if ((iRet = CopyBuffer(m_handle, 0, 0, 3, Buff)) < 3) szInfo = "Buffer reading failed.";
32.             else
33.             {
34.                     Info.dValue = Buff[0];
35.                     szInfo = CharArrayToString(Info.cInfo) + " [ " + (string)Buff[1] + " ] [ " + (string)Buff[2] + " ]";
36.             }
37.     }
38.     Print("Return => ", szInfo);
39.
40.     return INIT_SUCCEEDED;
41. }
42. //+------------------------------------------------------------------+
43. void OnDeinit(const int reason)
44. {
45.     ChartIndicatorDelete(m_id, 0, def_ShortName);
46.     IndicatorRelease(m_handle);
47. }
48. //+------------------------------------------------------------------+
49. void OnTick()
50. {
51. }
52. //+------------------------------------------------------------------+
```

This updated EA code above doesn't have much difference from what we saw at the beginning. But there are still a few new elements here that deserve explanation.

Between lines 17 and 19 there is a declaration of new variables that will be used to decrypt the information contained in the buffer. This may seem strange, but in fact the information present in the buffer is encoded, since we pass a string on one of the positions.

What's really interesting is between lines 27 and 38. Here the buffer is read. So, let's start with this part.

Line 27 contains code that can sometimes be seen when reading, even when we are reading multiple buffer positions. This is because by default reading will be done directly. You can see in Figure 01 that the array is in direct sequence. But there are moments when reading is done in the reverse order. Then, instead of using reverse indexing to access the array, we use the function defined on line 27 to specify that the reading will occur in reverse order. In this case, contrary to what we see, the value passed to the function will be **false**. This way we can use access indexing as if it were direct access.

Although line 27 may not make much sense in our current code since we are writing directly, it was added to explain what was said earlier.

There is no need to discuss in detail most of the other lines since many of them are self-explanatory. However, line 31 has something that requires some thought.

When we write data to a buffer inside the indicator, we know how many positions of information we are sending. Line 31 does just that: reads the expected number of positions. It would be more suitable to make a better combination using a header file. So, if the number of positions containing information increases, both the EA and the indicator will always be aware of this. But since these are only examples to explain a concept that will be used later, this can be ignored for now.

If the number of positions read is less than expected, this will be an error, and the relevant information will appear in the terminal. Now if the number is as expected, we go to line 34 and convert the information provided as a double into a string. So, we recover back the information the indicator placed in the buffer.

Note that the index in this case is zero, just as specified in the indicator. If reverse writing was done in the indicator, we could still use the same indexing in the EA. Simply change the value in line 27, and the EA will understand the information using the same indexing method.

Try to do this to understand how things really happen. Understanding these details is important for understanding the next articles in this series.

Once the conversion is complete, we use line 35 to generate a message that will be displayed in the terminal. Simple as that.

### Conclusion

In this short digression that we had in our replay/simulation system project, I have shown the basics of what will be covered in future articles.

This is just the simplest part of everything that awaits us. There are some things we haven't yet added or implemented that will be part of the replay/simulator system. All the content presented here is of great importance for future articles. Without understanding the basics of data transfer between indicators and other processes, you can get lost. I have not yet shown how to hide the indicator from the indicator window to prevent the user from deleting it. Also, we haven't added some elements that significantly complicate understanding in various possible combinations.

I really hope you understand the concepts presented and the real significance of the last three articles. Then everything will become much more complicated.

Since I want you to develop a kind of mechanical memory, there will be no attached files to this article. If you want to experiment and learn the basics shown here, you will have to write the code. But there shouldn't be any problem with this as the codes are very short.

See you in the next article where we will start integrating Chart Trader for use in a replay/simulator system.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11599](https://www.mql5.com/pt/articles/11599)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11599.zip "Download all attachments in the single ZIP archive")

[EA.mq5](https://www.mql5.com/en/articles/download/11599/ea.mq5 "Download EA.mq5")(1.81 KB)

[swap.mq5](https://www.mql5.com/en/articles/download/11599/swap.mq5 "Download swap.mq5")(1.72 KB)

[Defines.mqh](https://www.mql5.com/en/articles/download/11599/defines.mqh "Download Defines.mqh")(0.24 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/469271)**
(1)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
1 Aug 2024 at 05:17

**MetaQuotes:**

A new article [Developing a Playback System (Part 39): Paving the Way (III)](https://www.mql5.com/en/articles/11599) has been released:

By [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831 "DJ_TLoG_831")

Hard work, learnt a lot


![Creating Time Series Predictions using LSTM Neural Networks: Normalizing Price and Tokenizing Time](https://c.mql5.com/2/82/Creating_Time_Series_Predictions_using_LSTM_Neural_Networks___LOGO.png)[Creating Time Series Predictions using LSTM Neural Networks: Normalizing Price and Tokenizing Time](https://www.mql5.com/en/articles/15063)

This article outlines a simple strategy for normalizing the market data using the daily range and training a neural network to enhance market predictions. The developed models may be used in conjunction with an existing technical analysis frameworks or on a standalone basis to assist in predicting the overall market direction. The framework outlined in this article may be further refined by any technical analyst to develop models suitable for both manual and automated trading strategies.

![Population optimization algorithms: Resistance to getting stuck in local extrema (Part I)](https://c.mql5.com/2/72/Population_optimization_algorithms__Resistance_to_getting_stuck_in_local_extrema__LOGO.png)[Population optimization algorithms: Resistance to getting stuck in local extrema (Part I)](https://www.mql5.com/en/articles/14352)

This article presents a unique experiment that aims to examine the behavior of population optimization algorithms in the context of their ability to efficiently escape local minima when population diversity is low and reach global maxima. Working in this direction will provide further insight into which specific algorithms can successfully continue their search using coordinates set by the user as a starting point, and what factors influence their success.

![Mastering Market Dynamics: Creating a Support and Resistance Strategy Expert Advisor (EA)](https://c.mql5.com/2/82/Creating_a_Support_and_Resistance_Strategy_Expert_Advisor__LOGO_2.png)[Mastering Market Dynamics: Creating a Support and Resistance Strategy Expert Advisor (EA)](https://www.mql5.com/en/articles/15107)

A comprehensive guide to developing an automated trading algorithm based on the Support and Resistance strategy. Detailed information on all aspects of creating an expert advisor in MQL5 and testing it in MetaTrader 5 – from analyzing price range behaviors to risk management.

![Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status](https://c.mql5.com/2/71/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status](https://www.mql5.com/en/articles/14246)

Having started developing a multi-currency EA, we have already achieved some results and managed to carry out several code improvement iterations. However, our EA was unable to work with pending orders and resume operation after the terminal restart. Let's add these features.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kdgliqseilebkfmokqaybjswmstiznmf&ssn=1769184836379453543&ssn_dr=0&ssn_sr=0&fv_date=1769184836&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11599&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2039)%3A%20Paving%20the%20Path%20(III)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918483659081926&fz_uniq=5070118939482067012&sv=2552)

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