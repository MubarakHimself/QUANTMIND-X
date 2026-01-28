---
title: Guide to Testing and Optimizing of Expert Advisors in MQL5
url: https://www.mql5.com/en/articles/156
categories: Trading, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:30:20.896302
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/156&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062515485863486363)

MetaTrader 5 / Tester


### Introduction

Most of the time when a developer has written an Expert Advisor, making sure that the Expert Advisor achieves its aim of good profitability is always a very tasking process. In this article, we will look at some of the major steps needed in testing and optimizing an Expert Advisor so that we will be able to achieve close to the desired goal of writing the Expert Advisor.

### 1\. Identifying and Correcting Code Errors

We will begin this article by looking at some common code errors normally encountered in the process of writing an Expert Advisor code. Most of the time beginners face some tough time identifying and correcting code errors when writing their codes or when modifying a code written by another developer. In this section we will look at how easy it is to use the MQL5 editor to identify and correct some of such errors.

You have just completed writing the code and it seems everything should just work because you are almost certain that the code is error free. Or, it was a code that was written by someone else and you made a few changes and, alas! when you hit the **Compile** button (or press **F7**), you were presented by series of errors in the code as shown in the **Error** tab of the MetaEditor Toolbox window.

![Compilation errors in an Expert Advisor code](https://c.mql5.com/2/2/compilation-errors-1__1.png)

Figure 1. Compilation errors in an Expert Advisor code

Wow! **38 errors and 1 warning**, your code may not have as much errors as shown here, all we want to look at are the various types of errors that are likely to show up when compiling our code and how we can resolve them. Let us describe the diagram above.

- The Section marked **1** shows the **_description of the error in the code_**. This is what gives us the idea of what the error looks like.
- The section marked **2**shows us in _**which file we have the error**_. This is very important if we have included files that have errors. With this we will be able to know which file we are to check for the error described.
- The section marked **3** shows us **_which line and column (on the line) in our code the error is located_**. This enables us to know which particular line to check for the error described.
- The section marked **4** shows the **_summary of the compilation errors and warnings._**

Let us now begin to resolve the errors one after the other. Let us scroll up to the first line in the Error tab so that we can start from the beginning.

![Identifying and resolving code errors-1](https://c.mql5.com/2/2/compilation-errors-2__1.png)

Figure 2. Identifying and resolving code errors

The first issue is described as : " **truncation of constant value**" and is discovered on **line 16 column 20**, to locate the exact line in our code, from the **Edit** menu of the MetaEditor, select **Go to Line** or press **CTRL G** on your keyboard.

![Figure 3. Locating the error code line number](https://c.mql5.com/2/2/Image03__1.png)

Figure 3. Locating the error code line number

A dialog box will be displayed.

![Figure 4. Locating error line number dialog](https://c.mql5.com/2/2/Image04__1.png)

Figure 4. Locating error line number dialog

The range of number as shown on the dialog box is the total number of lines in the code. In this case **(1-354)** shows that our code contains 354 lines of code.

Type the line number you want to check in the box and click the **OK** button. You will be taken straight to the line number in the code. _**You will see the mouse cursor blinking on that particular line.**_

_**![Figure 5. Cursor showing the error line number](https://c.mql5.com/2/2/Image05__1.png)**_

Figure 5. Cursor showing the error line number

The problem here is that we declare **Lot** as an **integer**( **int**) variable but initialize it with a **double** value ( **0.1**). To correct this error we will change the **int** to **double**, save the file and then click **COMPILE** button again to see if that has been corrected.

![Figure 6. Compile and save code after correction is made ](https://c.mql5.com/2/2/Image06__1.png)

Figure 6. Compile and save code after correction is made

On compiling again, the first issue has been resolved, but we still have more issues as shown below:

![More errors in the code to resolve](https://c.mql5.com/2/2/compilation-errors-3__1.png)

Figure 7. More errors shows up in code after compilation

We will now follow the same procedure as above and go to line **31**. However, this time we will **right-click on the error** on the **Errors** tab and **select Go to line**

![Another way of locating code error line](https://c.mql5.com/2/2/gotoline_menu2.png)

Figure 8. Another way of locating code error line

Or simply **select the error and hit the Enter button on your keyboard**. Immediately, you will be taken to the code line number **31**.

You will see the mouse cursor blinking and also a small round red button (error icon) on that particular code line 31.

![Locating the code error line](https://c.mql5.com/2/2/codeline-31__1.png)

Figure 9a. Locating the code error line

However, if it is a warning message like the first one on line 16 that we corrected earlier, it will show a triangular yellow button (warning icon):

![Warning sign](https://c.mql5.com/2/2/warning_sign.png)

Figure 9b. Locating the code error line

It is very obvious that we don’t have any problem on line **31**, but the error description says: " **'STP' - unexpected token**" .

We then must check the previous code line (that is line 30) to see what may be wrong. On close examination, semicolon is missing after " **doubleccminb = -95.0000**" on line 30, that is why we have that error on line 31. We will now fix this error by typing the semicolon after -**95.0000** and compile the code again.

Now the line **31** errors are gone. Next is line **100** as shown below:

![More errors still exist in code](https://c.mql5.com/2/2/compilation-errors-4__1.png)

Figure 10. More errors still exist in code

Hey Olowsam, must we be compiling after each correction, why don’t we just go to through all the lines at the same time and after we have done all the corrections then we compile the code once instead of compiling after each correction?

**Did you just ask this question?**

You may be right in a way, but I will not advise that. Problems are always resolved one after the other – Step by Step. Any attempt to lump problem together and solve them at once may lead to many headaches. You will soon understand why… just be patient with me.

Back to our problem, we are to check line 100 for the next error. The error states : " **'if' - expressions are not allowed on a global scope**" And I am sure that the if expression in line 100 is  not on a global scope, but why are we having this error. Please let us go to line 100.

![Figure 11.  Locating the error in the code](https://c.mql5.com/2/2/codeline-100__1.png)

Figure 11.  Locating the error in the code

We can't find any problem on line 100 and because we just finished correction line 31, we are sure that the problem now is between line 32 and line 99. So let us move upward to line 99 (we have a comment , so it can't be the error). If we also look upwards to the declarations ( [MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick), [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderesult) and [MqlTradeResult](https://www.mql5.com/en/docs/constants/structures/mqltraderesult)), they  are correctly declared and punctuated.

Next let us look at the code for the **if** expression before these declaration code lines and see if the expression is okay.On very close study, we discover that the if expression has a closing brace, but no opening brace.

![Figure 12. Looking above the error line number to identify error](https://c.mql5.com/2/2/if-code-error.png)

Figure 12. Looking above the error line number to identify error

Add the opening brace and compile the code again.

```
//--- Do we have enough bars to work with
   int Mybars=Bars(_Symbol,_Period);
   if(Mybars<60) // if total bars is less than 60 bars
    {
      Alert("We have less than 60 bars, EA will now exit!!");
      return;
     }
```

Once the code was compiled; errors on line **100, 107, 121, 126, 129,** etc were completely cleared and new ones show up. See why it is good to follow step by step?

![More errors show up in code](https://c.mql5.com/2/2/compilation-errors-5__3.png)

Figure 13. More errors still exist in code

Next we move to line **56** with two errors : " **'cciVal1' - parameter conversion is not allowed**" and " **'cciVal1' - array is required**"

On closer look at line **56**, **cciVal1** is supposed to have been declared as an array. _**Could it be that we did not declare it as an array but now trying to use it as an array?**_ Let us check the declaration section to confirm this before we can know what next to do.

```
//--- Other parameters
int maHandle;               // handle for our Moving Average indicator
int cciHandle1,cciHandle2;  // handle for our CCI indicator
double maVal[];             // Dynamic array to hold the values of Moving Average for each bars
double cciVal1,cciVal2[];   // Dynamic array to hold the values of CCI for each bars
double p1_close,p2_close;   // Variable to store the close value of Bar 1 and Bar 2 respectively
```

From, here, we can see that we mistakenly declare **cciVal1** as a double rather than as a dynamic array because we omitted the square brackets ( **\[\]**). Let us add the square brackets (just as we have for **cciVal2\[\]**) and then compile the code.

```
//--- Other parameters
int maHandle;               // handle for our Moving Average indicator
int cciHandle1,cciHandle2;  // handle for our CCI indicator
double maVal[];             // Dynamic array to hold the values of Moving Average for each bars
double cciVal1[],cciVal2[]; // Dynamic array to hold the values of CCI for each bars
double p1_close,p2_close;   // Variable to store the close value of Bar 1 and Bar 2 respectively
```

![Errors in code reduced considerably](https://c.mql5.com/2/2/compilation-errors-6__1.png)

Figure 14. Errors in code has been reduced considerably

What! So many errors have disappeared. We only corrected error reported on line **56** and some other errors were resolved automatically. This is because, the error reported on line **56** was responsible for those other errors. **This is why it is good to follow a step by step process in resolving errors in your code.**

We will now move to the next reported error on line **103** : " **'GetLastError' - undeclared identifier**"Wait a minute, **GetLastError** is supposed to be a function… Let go to line **103** to see what the problem is.

```
//--- Get the last price quote using the MQL5 MqlTick Structure
   if(!SymbolInfoTick(_Symbol,latest_price))
     {
      Alert("Error getting the latest price quote - error:",GetLastError,"!!");    // line 103
      return;
     }
```

The problem is actually on line **103**. **GetLastError** is a function and every function needs a pair of parenthesis for input parameters. Let us type an empty pair of parenthesis and then compile the code. The empty pair of parenthesis indicates that the function takes no arguments or parameters.

```
//--- Get the last price quote using the MQL5 MqlTick Structure
   if(!SymbolInfoTick(_Symbol,latest_price))
     {
      Alert("Error getting the latest price quote - error:",GetLastError(),"!!");  // line 103
      return;
     }
```

Next, we move to line **159** : " **'=' \- l-value required**" anda warning : " **expression is not Boolean**" Let us go to line **159** and see what this error means.

```
      else if(PositionGetInteger(POSITION_TYPE) = POSITION_TYPE_SELL) // line 159
       {
            Sell_opened = true; // It is a Sell
```

It can be seen here that we assigned the value of [POSITION\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer) to [PositionGetInteger](https://www.mql5.com/en/docs/trading/positiongetinteger) [(POSITION\_TYPE)](https://www.mql5.com/en/docs/trading/positiongetinteger) in the if statement and this is not what we intend to do. We wanted to make comparison instead. We will now change the expression to use **[equal operator](https://www.mql5.com/en/docs/basis/operations/relation)** rather than using an **[assignment operator](https://www.mql5.com/en/docs/basis/operations/assign)**. (that is ‘==’ instead of ‘=’). Make the correction and compile the code.

```
      else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) // line 159
       {
            Sell_opened = true; // It is a Sell
```

Good! Now we have one more to go. Let us go to line **292** to see why it says " **'PositionsTotal' - undeclared identifier**" . Wait a minute, can you remember that we have seen an error like this before? ‘ **GetlastError**’ line **103**. Possibly, we forget to add the pair of parenthesis for PositionsTotaltoo, since it is a function. Let us go to line **292** to confirm.

```
bool CheckTrade(string otyp)
{
   bool marker = false;
   for (int i=1; i<=PositionsTotal;i++)  // line 292
   {
```

Just like we suspected, it is because we forgot to add the pair of parenthesis for the function **PositionsTotal**. Now add the pair of parenthesis ( **PositionsTotal()**) and compile the code. Let me also state that, it is possible to get this error if we actually use a variable which we did not declare anywhere in the code.

![Figure 15. All compilation Errors has been completely resolved](https://c.mql5.com/2/2/Image15__1.png)

Figure 15. All compilation Errors has been completely resolved

Wonderful! Now we have been able to correct all the compilation errors. It is now time to debug our code and see if there are run-time errors.  Here, we will not be going into the details of how to debug our code as it has already been explained in **[this article.](https://www.mql5.com/en/articles/100)**

As the debug session begins, we notice another error :

![Figure 16. Runtime error observed during debugging of code](https://c.mql5.com/2/2/Image16__1.png)

Figure 16. Runtime error observed during debugging of code

Click the **OK** button and you will be taken to the line of code that generates the error.

![Identifying the line of code that generates run-time error](https://c.mql5.com/2/2/debug-error-1-location__1.png)

Figure 17. Identifying the line of code that generates run-time error

The error is generated by this code on line **172** as you can see from the figure above. Since the error is an " **Array out of range**" error, it means that the value we intend to get from the array is out of the range of the array values available. So we will now go to the line where we copy the indicator buffers into arrays to see what the problem is.

```
//--- Copy the new values of our indicators to buffers (arrays) using the handle
   if(CopyBuffer(maHandle,0,0,3,maVal)<0)
     {
      Alert("Error copying MA indicator Buffers - error:",GetLastError(),"!!");
      return;
     }
   if(CopyBuffer(cciHandle1,0,0,3,cciVal1)<0 || CopyBuffer(cciHandle2,0,0,3,cciVal2)<0)
     {
      Alert("Error copying CCI indicator buffer - error:",GetLastError());
      return;
     }
```

We can observe from the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer) functions that we have only copied three values (Bar 0, 1, and 2) which means that we can only access array values of **maVal\[0\]**, **maVal\[1\]**, and **maVal\[2\]** and also **cciVal1\[0\]** , **cciVal1\[1\]** and **cciVal1\[2\]**, etc. But in our code on line **172**, we were trying to get the array value for **cciVal1\[3\]**. This is why the error was generated. Now, stop the debugger so that we can fix the error:

![Figure 18. Stop debugger to correct error in code](https://c.mql5.com/2/2/Image18__1.png)

Figure 18. Stop debugger to correct error in code

To fix this we need to increase the number of records to be copied from Indicator buffers to **5** so that we will be able to obtain array values of **cciVal1\[0\], cciVal1\[1\], cciVal1\[2\], cciVal1\[3\],** and **cciVal1\[4\]** if need be.

```
//--- Copy the new values of our indicators to buffers (arrays) using the handle
   if(CopyBuffer(maHandle,0,0,5,maVal)<0)
     {
      Alert("Error copying MA indicator Buffers - error:",GetLastError(),"!!");
      return;
     }
   if(CopyBuffer(cciHandle1,0,0,5,cciVal1)<0 || CopyBuffer(cciHandle2,0,0,5,cciVal2)<0)
     {
      Alert("Error copying CCI indicator buffer - error:",GetLastError());
      return;
     }
```

Correct the code as shown and then start the debugger again. This time, no more errors as we notice our Expert Advisor performing trade actions

![Figure 19. All errors corrected, Expert Advisor performs trade during debugging](https://c.mql5.com/2/2/Image19.png)

Figure 19. All errors corrected, Expert Advisor performs trade during debugging

### 2\. Testing the Expert Advisor

Once we are sure that our code is error free, it is now time to test the Expert Advisor to be able to get the best settings that will give us the best results. In order to carry out the test, we will use the Strategy Tester, a program which is built into the MetaTraderterminal. To launch the Strategy Tester, Go to **View** menu on the Terminal and select **Strategy Tester**.

![Figure 20. Launching the Strategy Tester](https://c.mql5.com/2/2/Image20__1.png)

Figure 20. Launching the Strategy Tester

**2.1. Preliminary Testing of our Expert Advisor**

At this point, we want to test our Expert using the available symbols in the Market Window. With this result we will be able to guess which currency pairs we can better optimize our Expert for. Make sure the Market Window contains most of the currencies you are targeting for the Expert.

Select the Expert on the Strategy Tester **Settings** Tab, select the period/timeframe you have in mind (and of course you can also test it for different timeframes) and then Select ' **All Symbols Selected in MARKET Watch**' in the optimization field. Directly in-front is the Optimization results parameter, select **Balance + max Profit Factor**.

![Figure 34. Preliminary test of Expert Advisor with all symbols in the Market Watch window ](https://c.mql5.com/2/2/Image34__1.png)

Figure 21. Preliminary test of Expert Advisor with all symbols in the Market Watch window

**1\. Select the tick generation mode –(Every Tick)**

**2\. Select Optimization Type –(All Symbols Selected in MARKET Watch)**

**3\. Select type of expected Result from optimization**

You can get the details of the various optimization types from the terminal help documentation. We are not forward testing, so leave **Forward** as **No**.

For this test, the main values/parameters (highlighted in green) in the **Inputs** tab will be used.

![Figure 35. Preliminary test input parameters](https://c.mql5.com/2/2/Image35__1.png)

Figure 22. Preliminary test input parameters

Once you are done, switch over to the **Settings** tab and click the **Start** button. On completion of the test, you will see a message in the Journal Tab similar to the following:

![Figure 36. Preliminary test completed](https://c.mql5.com/2/2/Image36__1.png)

Figure 23. Preliminary test completed

Once the test is completed, go to the **Optimization Results** Tab to see the results.

![Figure 37. Preliminary test optimization results](https://c.mql5.com/2/2/Image37__1.png)

Figure 24. Preliminary test optimization results

Our interest is in the symbol that gives the highest Result based on our setting – ( **Balance + max Profit Factor**). To get this, let us sort the result by clicking on the **Result** title such that the symbol with the highest result is listed at the top.

![Figure 38. Preliminary optimization result analysis](https://c.mql5.com/2/2/Image38__1.png)

Figure 25. Preliminary optimization result analysis

From this result, we can see that our Expert Advisor can be profitable for the following symbols ( **EURUSD, EURJPY, AUDUSD**) in the timeframe we have selected. You can further perform this test for another timeframe, say, 30mins and see what you have. This should be taken as an assignment and please share the result so that we can all learn too.

From the result of our preliminary test, we will now decide which symbol(s) and timeframe(s) we are going to optimize our Expert Advisor for.

In this example, we will optimize our Expert Advisor for the **EURUSD** and **1 Hour** timeframe.What are the things that motivate the choice we just made:

- **Profit Factor:**

The Profit factor is the ratio of the total profit to that total loss for that test. The higher the Profit factor the more profitable your trading strategy is.

- **Drawdown %:**

This refers to the relative drawdown of the equity or the largest loss (in percent) from the maximal value of equity. The lower the Drawdown (in percent), the better the strategy.

- **Recovery Factor:**

This is the ratio of the profit to the maximal drawdown. It reflects the riskiness of the trading strategy.

Having decided on the symbol and timeframe to use, it is now time to optimize our Expert Advisor.

**2.2. Optimizing the Expert Advisor**

Optimization is simply a process of fine-tuning the performance of our EA by testing with various factors (parameters) that determines the effectiveness or profitability of our Strategy coded in the EA. It is a similar procedure to testing, but instead of testing the EA only once, it will be tested many times depending on the parameters selected in the Input tab.

To begin, we go to the settings tab and enable optimization and then select the type of result we want from our optimization.

![Figure 39. Optimization settings for Expert Advisor](https://c.mql5.com/2/2/Image39__1.png)

Figure 26. Optimization settings for Expert Advisor

**1\. Select the tick generation mode –(Every Tick)**

**2\. Select Optimization Type –(Fast Genetic Based Algorithm)**

**3\. Select type of expected Result from optimization**(here we select **Balance + Max Profit Factor**)

You can get the details of the various optimization types from the terminal help documentation. We are not forward testing, so leave Forward as No. Having set the optimization properties, let us set the parameters to be used for the optimization in the **Inputs** tab.

![Figure 40. Optimization Input parameters](https://c.mql5.com/2/2/Image40__1.png)

Figure 27. Optimization Input parameters

Since we are optimizing, we will only concentrate on the areas highlighted in yellow. First of all, any parameter we do not want to use in the optimization must be unchecked. In order words, we will only check the parameters we want to use in the optimization of the EA. Here, I have checked five parameters, but you may decide to check only one or two depending on the parameters that the effectiveness of your strategy is based on. For example, you may check only the Moving Average and CCI periods such that the optimization result will let you know the best value for each of the Indicators that give your EA the best performance. This is the main essence of optimizing.

Also, the number of parameters checked will determine the total number of tests that your EA will go through. You will soon see what I am talking about.

**Setting The Values**

**_Start:_**

This is the starting value to be used for the selected variable for optimization.Let us use the Stop Loss variable to explain how to set the values. For the Stop Loss, we have asked the tester to start with a value of **30**. This will be the minimum value that will be used for Stop Loss during the optimization.

**_Step:_**

This is the incremental value for the Stop Loss. If we set an increment of 2; it means that, if in the first test, it uses **30** for Stop Loss it will use either **32, 36, 34** etc. in the second… It does not mean that it will use **30**, then followed by **32**, **34** etc. No, it selects the values at random but they will always be multiples of two (2) between the Start value and the Stop value.

**_Stop:_**

This is the maximum or highest value that will be used for the optimization. Here we specified **38**. This means that the values that will be used for the testing will be between **30** and **38** but will be values which are multiples of **2**. It will not use **40** or any value greater.

The total number of tests that will be carried out depends on the settings of these three sections. In our example, the tester will combine a total of **5** possibilities alone for the Stop Loss as shown in the **_Steps_** column on the **Inputs** Tab, it will combine a total of 8 possibilities for the Take Profit, etc. By the time you consider all the other variables, it will be getting to hundreds or thousands of possibilities (tests/passes). If you don't want to wait for ages in order to optimize a single EA, make sure you don't include or check too many variables; maybe just two or three that the performance of your EA really depends on (most especially, the indicator periods, if you use them in your own code). Also you must make sure your step value will not result in having too many possibilities (tests). For example, if we use **1** as the step value, then we have increased the number of attempts for the Stop Loss alone to **10**. Well, as said earlier, the total time required to complete an optimization session depends on the total number of available agents you have setup on your system.

I believe the explanation is sufficient.

Once we have finished setting the inputs, we now go back to the Settings tab and click the **Start** Button.

Once the optimization is completed, we can see the details on the journal tab.

![Figure 43. Optimization completed as shown in the Journal tab](https://c.mql5.com/2/2/Image43__1.png)

Figure 28. Optimization completed as shown in the Journal tab

To view the results as each test is passed or completed, we go to the **Optimization Results** tab. And it is always good to sort the output by the Results so that we can easily identify the settings that gives us the best result based on our optimization setting. Clicking on the **Result** heading within the **optimization Results** tab will arrange the results in either ascending or descending order

![Figure 44. Optimization report](https://c.mql5.com/2/2/Image44__1.png)

Figure 29. Optimization report

Switch over to the **Optimization Graph** tab to see how the graph looks like.

![Figure 45. Optimization graph](https://c.mql5.com/2/2/Image45__1.png)

Figure 30. Optimization graph

Don’t understand what you see? Don’t worry; the dots you see is a plot of the number of tests your EA passed against the optimization result based on the optimization result type you selected. In our case we selected Balance + max Profit factor.

**2.3. Interpreting the result**

To successfully interpret the optimization report, go to the Optimization Results tab. You will discover that you cannot see some fields like, Profit factor, Expected Payoff, Drawdown %, etc . To see them, right-click anywhere in the **Optimization Results** tab and select the additional information you wish to see as shown below:

![Figure 46. Selecting Drawdown% in optimization result](https://c.mql5.com/2/2/Image46__1.png)

Figure 31. Selecting Drawdown% in optimization result

![Figure 47. Selecting Profit Factor in optimization result](https://c.mql5.com/2/2/Image47__2.png)

Figure 32. Selecting Profit Factor in optimization result

Having added these additional records, we will now analyze the Optimization result to decide the best settings for our Expert Advisor.

![Optimization report Analysis](https://c.mql5.com/2/2/optimization-result-analysis-1.png)

Figure 33. Analyzing the optimization result

From the above figure, the highlighted sections labeled A and B indicates the best results for our Expert Advisor. Now the choice you make is completely yours, It all depends on what you are looking for. However, here we are interested not only in the settings that give the highest profit, but also have a lower **drawdown%.**

As you can see, the section **A** (highlighted in yellow) has the best result ( **Balance + max Profit Factor**) of **22381.71** with a profit of **924.10** while the section **B** (highlighted in green) has the second best result of **22159.25** but with a higher profit of **936.55**. Section **A** had a lower **Drawdown%** of **1.78** while B has a higher drawdown of **1.95**.

The Strategy Tester saves the optimization results to the"<Client terminal data folder>\\Tester\\cache" folder. In your case all the optimization data will be saved to the cci\_ma\_ea.EURUSD.H1.0.xml file,

The file name has the following form: ExpertName.SYMBOL.PERIOD.GenerationMode.xml, where:

- ExpertName - Expert Advisor Name;

- Symbol - symbol;

- Period - timeframe (M1,H1,...)
- GenerationMode - tick generation mode (0-every tick, 1 - one minute OHLC, 2 - open prices only).


The XML files can be opened by MS Excel.

**2.4. Choosing the Best Result**

To finally obtain the best result, we need to look at the Optimization graph again. Switch back to the Optimization graph.Right-click anywhere within the graph and select **1D Graph.**

![Select 1-D graph form optimization graph](https://c.mql5.com/2/2/optimization-graph-select1d.png)

Figure 34. Select 1-dimensional (1 D) graph for result analysis

With this we can easily see the values of each of the input parameters that give the best result. You can now begin to choose each parameter to be able to see the best value. Right-click on the graph and select **X-Axis** and then select the parameter you want to check. It will look like below (for Stop loss)

![Figure 50. Getting the best StopLoss value from the optimization result](https://c.mql5.com/2/2/Image50__1.png)

Figure 35. Getting the best StopLoss value from the optimization result

Actually, from the optimization result, it is very clear that the best Stoploss is 34, the best TakeProfit is 78, and the best CCI\_Period1 is 62. To obtain the best values for the MAPeriod and CCI\_Period2, select each of them as above

![Figure 51. Getting the best Moving Average Period value from the optimization result](https://c.mql5.com/2/2/Image51__1.png)

Figure 36. Getting the best Moving Average Period value from the optimization result

This graph shows a value of **26** as the MA\_Period with the best result.

![Figure 52. Getting the best CCI_Period1 value from the optimization result](https://c.mql5.com/2/2/Image52__1.png)

Figure 37. Getting the best CCI\_Period1 value from the optimization result

This graph shows a value of **62** as the CCI\_Period1 with the best result.

![Figure 53. Getting the best CCI_Period2 value from the optimization result](https://c.mql5.com/2/2/Image53__1.png)

Figure 38. Getting the best CCI\_Period2 value from the optimization result

This graph shows values of **28** or **30** as the CCI\_Period2 with the best results.

Having obtained the best values for each parameter, it is now time for the final testing of our Expert Advisor.

**2.5. The Final Test**

The final test involves putting together the best parameters for the testing of the Expert Advisor. In this case, we will use the best values we discovered in the INPUT section of the Strategy Tester as shown below.

![Input values for the final test](https://c.mql5.com/2/2/final-result-inputs.png)

Figure 39. The final test input parameters

In the **SETTINGS** tab of the Strategy Tester, we will disable Optimization as shown below

![The settings for the final test](https://c.mql5.com/2/2/final-result-settings.png)

Figure 40. The final test settings

We will now click the **START** button to begin the test. Once the test is completed, we have the results on the **RESULTS** tab as shown below

![Figure 56. The final test results](https://c.mql5.com/2/2/Image56__1.png)

Figure 41. The final test results

And likewise, we have the graph for the test on the **GRAPH** tab

![Figure 57. The final test graph result](https://c.mql5.com/2/2/Image57__1.png)

Figure 42. The final test graph result

### **Conclusion**

In this article, we have discussed the ways to identify and correct code errors and we have also discussed how to test and optimize an Expert Advisor for the best symbol from the market watch.

With this article, I believe checking code for errors using the editor and optimizing and testing of Expert Advisors using the Strategy Tester makes writing a better and profitable Expert Advisor possible.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/156.zip "Download all attachments in the single ZIP archive")

[cci\_ma\_ea\_corrected.mq5](https://www.mql5.com/en/articles/download/156/cci_ma_ea_corrected.mq5 "Download cci_ma_ea_corrected.mq5")(14.3 KB)

[cci\_ma\_ea.mq5](https://www.mql5.com/en/articles/download/156/cci_ma_ea.mq5 "Download cci_ma_ea.mq5")(14.23 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [The Use of the MQL5 Standard Trade Class libraries in writing an Expert Advisor](https://www.mql5.com/en/articles/138)
- [Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://www.mql5.com/en/articles/116)
- [Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners](https://www.mql5.com/en/articles/100)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/2309)**
(11)


![fenix74](https://c.mql5.com/avatar/2014/1/52D6AA06-084C.jpg)

**[fenix74](https://www.mql5.com/en/users/fenix74)**
\|
15 Jan 2014 at 15:45

Can you please tell me if there is a similar article on testing and [optimising Expert Advisors](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization "Help: Optimising Expert Advisors in the MetaTrader 5 trading terminal") for MQL 4. I can't find at least reference books of error codes, examples of frequent errors, I have to look for a description of each error through a search in the internet.


![Joao Carlos Marcuschi](https://c.mql5.com/avatar/avatar_na2.png)

**[Joao Carlos Marcuschi](https://www.mql5.com/en/users/marcuschi)**
\|
22 Apr 2015 at 21:20

Excellent article Samuel. It really helped me clarify my doubts about optimising my AE. Thank you for your willingness to help and clarify.


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
9 Oct 2015 at 08:15

**fenix74:**

Can you please tell me if there is a similar article about testing and [optimising Expert Advisors](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization "Help: Optimising Expert Advisors in the MetaTrader 5 trading terminal") for MQL 4.

Have you looked at the articles ? [https://www.mql5.com/en/articles/mt4/strategy\_tester](https://www.mql5.com/ru/articles/mt4/strategy_tester)

![Riko Dewaner](https://c.mql5.com/avatar/avatar_na2.png)

**[Riko Dewaner](https://www.mql5.com/en/users/riko_dewaner)**
\|
10 May 2020 at 06:48

i still don't understand about "pass"

what does it mean?

does it mean higher value is better?

i can not find any guide about this in manual, at least it's not clear what it really means

![DoctorTanzanite](https://c.mql5.com/avatar/avatar_na2.png)

**[DoctorTanzanite](https://www.mql5.com/en/users/doctortanzanite)**
\|
8 Sep 2021 at 02:29

Great article excellent work my guy✌


![Interview with Valery Mazurenko (ATC 2010)](https://c.mql5.com/2/0/notused_ava.png)[Interview with Valery Mazurenko (ATC 2010)](https://www.mql5.com/en/articles/531)

By the end of the first trading week, Valery Mazurenrk (notused) with his multicurrency Expert Advisor ch2010 appeared on the top position. Having treated trading as a hobby, Valery is now trying to monetize this hobby and write a stable-operating Expert Advisor for real trading. In this interview he shares his opinion about the role of mathematics in trading and explains why object-oriented approach suits best to writing multicurrency EAs.

![The "New Bar" Event Handler](https://c.mql5.com/2/0/new_bar_born.png)[The "New Bar" Event Handler](https://www.mql5.com/en/articles/159)

MQL5 programming language is capable of solving problems on a brand new level. Even those tasks, that already have such solutions, thanks to object oriented programming can rise to a higher level. In this article we take a specially simple example of checking new bar on a chart, that was transformed into rather powerful and versatile tool. What tool? Find out in this article.

![Interview with Boris Odintsov (ATC 2010)](https://c.mql5.com/2/0/bobsley_ava.png)[Interview with Boris Odintsov (ATC 2010)](https://www.mql5.com/en/articles/532)

Boris Odintsov is one of the most impressive participants of the Championship who managed to go beyond $100,000 on the third week of the competition. Boris explains the rapid rise of his expert Advisor as a favorable combination of circumstances. In this interview he tells about what is important in trading, and what market would be unfavorable for his EA.

![Technical Analysis: What Do We Analyze?](https://c.mql5.com/2/0/tech_analysis_MQL5__1.png)[Technical Analysis: What Do We Analyze?](https://www.mql5.com/en/articles/173)

This article tries to analyze several peculiarities of representation of quotes available in the MetaTrader client terminal. The article is general, it doesn't concern programming.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=obzonxurkmuxpgsohvqiwamlnulxqiiy&ssn=1769157019199418560&ssn_dr=0&ssn_sr=0&fv_date=1769157019&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F156&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Guide%20to%20Testing%20and%20Optimizing%20of%20Expert%20Advisors%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915701983927401&fz_uniq=5062515485863486363&sv=2552)

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