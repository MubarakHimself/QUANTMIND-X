---
title: Understand and Use MQL5 Strategy Tester Effectively
url: https://www.mql5.com/en/articles/12635
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:07:56.430064
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/12635&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069142719185486026)

MetaTrader 5 / Tester


## Introduction

As MQL5 programmers or developers, we find that we need to well understand and use the strategy tester to get effective results. By using this valuable tool we can get valuable insights in terms of the performance of our created MQL5 programs and this will affect in getting better trading results. So, we find that we need to well understand important topics before we do our testing like dealing with all error types as it is very normal as developers that we make mistakes that result in errors. Another topic that we need to understand well also is Debugging which let us execute our created programs in an interactive way. Then, we come to the most important and valuable tool which is the strategy tester to test and evaluate our created programs with the most interesting features than MetaTrader 4. So, it will be an amazing learning journey in this article to learn the most important aspects of using the MQL5 Strategy Tester we will cover it through the following topics:

- [Errors](https://www.mql5.com/en/articles/12635#errors)
- [Debugging](https://www.mql5.com/en/articles/12635#debugging)
- [The Strategy Tester](https://www.mql5.com/en/articles/12635#tester)
- [Conclusion](https://www.mql5.com/en/articles/12635#conclusion)

We will try to cover the most popular points about these previous topics to well understand what we need to deal with as programmers or developers. What I need to mention here is that when we deal with this topic, it is normal that all readers already know the MQL5 programming language and they are know how to code their programs as this is a pre-requested topic that you need to be understandable to get the most benefits from reading this article. If you need to learn about the language first, you can find the [MQL5 documentation](https://www.mql5.com/en/docs). Also, you can read my other articles about learning the basics of mql5 programming and how to create trading systems based on the most popular technical indicators and I hope you find them useful.

## Errors

In this topic, we will learn about errors that we can find when creating, executing, and running MQL5 programs. What makes this topic very important to understand is that the MQL5 will report these errors correctly but if we do not know what these errors mean or at which stage of our program we will take much more time to handle or resolve them than if we already know what they mean.

We will present this topic based on the stage of our working on the MQL5 program, while we write our code there are errors and warnings that can be faced which are Compilation errors and warnings. We may face also errors when executing the MQL5 program which are the runtime errors and we may face also other types of errors when our MQL5 program is trying to trade which are the trade server errors. So, we will cover the following types of errors:

- Compilation Errors and Warnings
- Runtime Errors
- Trade Server Errors

Before covering these errors there is an important thing you need to know the place of errors appearing. In the MQL5 IDE, there is a toolbox window in the lowest part if you do find it by default, you can view it by one of the following methods:

1- Clicking View ==> Choosing Toolbox

![Toolbox](https://c.mql5.com/2/54/Toolbox.png)

2- Pressing Ctrl+T from keyboard

3- Clicking the Toolbox button from the main toolbox

![Toolbox2](https://c.mql5.com/2/54/Toolbox2.png)

After that, we can find it the same as the following picture

![Toolbox3](https://c.mql5.com/2/54/Toolbox3.png)

### **Compilation Errors and Warnings:**

When we write our code for a specific program we may make a mistake and this is very normal by writing a wrong syntax or having typos for example which leads to errors when compiling the code. In this part, we will share the most popular and commonly faced this type of error. The most important thing also that you need to know is that the program cannot be compiled until the errors are eliminated or solved. I will present the most popular errors that we can face as per this type of errors:

#### semicolon expected error:

This error we will face when forgetting to write a semicolon at the end of the line, So we have a missing semicolon, or if we forgot a left bracket also. We have to use these types of symbols correctly to avoid this type of error. the following code is an example of this error.

**Wrong code with error:**

```
int a=(2+1)
```

The error will be the same as the following:

![semicolon expected](https://c.mql5.com/2/54/semicolon_expected.png)

**Correct code without error:**

After correcting the code by adding the semicolon at the end of the code line we can find the correct code the same as the following:

```
int a=(2+1);
```

After compiling the code we find the error is eliminated and the code is compiled without any errors the same as the following:

![semicolon expected resolved](https://c.mql5.com/2/54/semicolon_expected_resolved.png)

#### Unexpected token error:

This is another type of code that we commonly face and the reason for it is forgetting a right parenthesis in the last line of code or we may add extra left parenthesis in the present line of code. We can see an example of this type of error the same as the following:

**Wrong code with error:**

```
int a=(2+1;
```

We can see the error as we know in the Errors tab in the Toolbox the same as the following:

![Unexpected token](https://c.mql5.com/2/54/Unexpected_token.png)

**Correct code without error:**

```
int a=(2+1);
```

We can see the Error tab with any errors after correcting the code by adding the right parenthesis the same as the following:

![ Unexpected token resolved](https://c.mql5.com/2/54/Unexpected_token_resolved.png)

#### Undeclared identifier error:

This type of error occurs when we use a variable without declaring it first as we must declare any new variable before using it or assigning any value to it. Declaring it by choosing the right and suitable data type that we need to be returned to this variable like integer or string for example. The following is an example of this type of error by using a new variable without declaring it.

**Wrong code with error:**

```
a=5;
```

In the previous code, we used the variable (a) by assigning 5 to it without declaring it. So, we will find the  undeclared identifier error when we compile this code the same as the following:

![Undeclared identifier](https://c.mql5.com/2/54/Undeclared_identifier.png)

As we can see the accuracy of the error description can be very helpful as it not only reports the error but specified the variable 'a' that is the reason for this error.

**Correct code without error:**

```
int a=5;
```

After compiling this correct code we will not find any errors in the same picture that we mentioned before from the Errors tab of the Toolbox.

#### Unbalanced left parenthesis error:

This type of error occurs when we miss the right parenthesis or we use an extra right one. We can find this error through the following example.

**Wrong code with error:**

```
   bool a=7;
   if (a=5
   a=5;
```

In the previous code, we find that by forgetting the right parenthesis the error of  "unbalanced left parenthesis' in addition to the error of "some operator expected" the same as in the following picture

![Unbalanced left parenthesis](https://c.mql5.com/2/54/Unbalanced_left_parenthesis.png)

**Correct code without error:**

```
   bool a=7;
   if (a=5)
   a=5;
```

After correcting the code we will find that it compiled without errors.

#### Unexpected end of program error:

Sometimes we miss a closing bracket in our code and this occurs unexpected end-of-program error we have to check our code to see what needs to add or we need to make sure that every opening bracket has a closing one to solve this error. The following is an example of this code.

**Wrong code with error:**

```
void OnStart()
  {
   bool a=7;
   if (a=5)
   a=5;
```

By writing the code the same as the previous block of code by forgetting the closing bracket after the last line (a=5) we can find the "unexpected end of program" error in addition to another error of "unbalanced parentheses" the same as the following picture

![unexpected end of program](https://c.mql5.com/2/54/unexpected_end_of_program.png)

**Correct code without error:**

```
void OnStart()
  {
   bool a=7;
   if (a=5)
   a=5;
   }
```

After correcting the code by adding the closing bracket we will find that the code compiled without errors.

#### Expressions are not allowed on a global scope error:

This error occurs when we miss a left bracket in a compound operator or it may happen when we write a statement or expression outside the scope of a specific function as we have to use only expressions within the scope of the function. The following example is for this type of error.

**Wrong code with error:**

```
   int a=(7+5);
   if (a<7)

   a=7;
   }
```

In the previous code, we missed the (}) opening that changed the scope and reported the error "Expressions are not allowed on a global scope". The following is the error from the Errors tab.

![Expressions are not allowed on a global scope](https://c.mql5.com/2/54/Expressions_are_not_allowed_on_a_global_scope.png)

**Correct code without error:**

```
   int a=(7+5);
   if (a<7)
   {
   a=7;
   }
```

After correcting the code the same as the previous block of code we will find that the code compiled without any errors.

#### Wrong parameters count error:

We experience this type of code when we use a specific function with specific parameters and we do not specify these parameters properly by specifying too many or not enough parameters. This error will appear if we use the predefined function or our created functions. The following is an example of this error.

**Code with error:**

If we created a function of "myVal" to return an integer value of the result of the summation of two integer values what means that we have two parameters in this function? So, when we call this function by using parameters different than these two parameters we will get this type of error the same as the following code

```
void OnStart()
  {
   int example=myVal(10);
  }
//+------------------------------------------------------------------+
int myVal(int a, int b)
  {
   return a+b;
  }
```

By compiling the previous wring code we will get the error the same as the following:

![Wrong parameters count](https://c.mql5.com/2/54/Wrong_parameters_count.png)

**Code without error:**

To write this code correctly we need to use the specified parameters of this created function the same as the following

```
void OnStart()
  {
   int example=myVal(10,20);
  }
//+------------------------------------------------------------------+
int myVal(int a, int b)
  {
   return a+b;
  }
```

After compiling this correct code we will not find any errors.

#### Some operator expected error:

This type of error that we find when we miss an operator in a specific location by missing it completely or misplacing it. The following is an example of this type of error.

**Code with error:**

```
int a= 7 10;
```

In the previous code, we missed an operator in between the two numbers and this is the reason that we will get an error of "some operator expected" the same as in the following picture:

![Some operator expected](https://c.mql5.com/2/54/Some_operator_expected.png)

**Code without error:**

```
int a= 7+10;
```

After correcting the code by adding the (+) we will find the code will be compiled without errors.

The mentioned errors are the most popular errors in this type as compilation errors but we said that there are compilation warnings we need also to learn what they are, they are shown for informational objectives and they are not errors. The following is an example code that we find that we will get a warning in the Errors tab

```
int a;
```

By compiling the previous code we will get a warning of the variable 'a' not used as we declared a new 'a' variable but we do not use it or assign value to it. The following is the warning as a picture

![ warning](https://c.mql5.com/2/54/warning.png)

You can check out all MQL5 The following link from documentation for more information about:

- [Compilation Errors](https://www.mql5.com/en/docs/constants/errorswarnings/errorscompile)
- [Compilation Warnings](https://www.mql5.com/en/docs/constants/errorswarnings/warningscompile)

**Some tips that can be very useful and helpful to deal with these errors when occurs:**

- Fix them from the start of errors as it may solve all errors based on the reason for this error.
- Double-clicking the error line in the Errors tab of Toolbox will lead to the line of code that has the problem.

Now we learned about the errors that we can get in the first stage when writing the code of the MQL5 program. We will move to the second stage which is errors that occur when we execute the program which are Runtime errors.

### **Runtime Errors:**

As we mentioned before Runtime errors are the types of errors that occur during the execution of a MQL5 program. These errors will not prevent the code from compiling without errors but it will not be executed or operated as we need or expect. We can find these errors in the log as printed messages with the reason for the error and the line that has the error in the source code also.

Examples of these errors:

| Code of error | Reason of error |
| --- | --- |
| 4007 | Not enough memory for the relocation of an array, or an attempt to change the size of a static array |
| 4101 | Wrong chart ID |
| 4301 | Unknown symbol as a market info |
| 4754 | Order not found |
| 5120 | Internal database error |

If you noticed that we will have a code of error as MQL5 defined all Runtime errors by codes to be easy to identify and you can find these codes by using GetLastError() function as adding error handling method to your program, we can see these errors reporting in the Journal tab while testing or in the expert tab while attached the program to the chart to be executed in the current time, and you can see all of Runtime Errors with all codes in the MQL5 documentation through the following link:

[https://www.mql5.com/en/docs/constants/errorswarnings/errorcodes](https://www.mql5.com/en/docs/constants/errorswarnings/errorcodes)

I need to mention here that the program will not end the execution after the runtime error except for a few critical errors that will end the execution like A array out-of-range error for example.

### **Trade Server Errors:**

These types of errors occur in the stage of execution of a trade as a structure of a trade request MqlTradeRequest using the OrderSend() function, we can see also these errors reported in the Journal tab while testing or in the expert tab while attaching the program to the chart to be executed in the current time. If you need to see examples of these errors the following are for that:

| Code of error | Reason of error |
| --- | --- |
| 10006 | Request rejected |
| 10032 | Operation is allowed only for live accounts |
| 10033 | The number of pending orders has reached the limit |
| 10034 | The volume of orders and positions for the symbol has reached the limit |
| 10041 | The pending order activation request is rejected, the order is canceled |

You can see all codes of this type of error you can find them in the MQL5 documentation through the following link:

[https://www.mql5.com/en/docs/constants/errorswarnings/enum\_trade\_return\_codes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes)

## Debugging

In this topic, we will learn how Debugging can be used through the MQL5 debugger tool as this can be used to execute our created program through historical or live data and we can do that by clicking the start/ resume debugging on real or history data the same as the following picture of Debugger buttons from the toolbox

![ Debugger buttons](https://c.mql5.com/2/54/Debugger_buttons.png)

1- Start/Resume debugging on history data

2- Start/Resume debugging on real data

3- Pause debugging

4- Stop debugging

Once we start the start button to debug on history data we will find the program will be executed on the historical data by opening a chart to debug the program on the historical data or if we chose the debugging on real data the program will be attached the chart on real data and beside the program you will find the (Debugging).

## The Strategy Tester

In this part or topic, we learn about the most important and valuable tool in the MetaTrader 5 which is the Strategy Tester with new features that are better than the tester of MetaTrader 4 like for example the multi-currency testing and others. How we can get this Strategy Tester we can do that through one of the following methods:

1- While opening the MetaTrader 5 trading terminal, press Ctrl+R from the Keyboard.

2- From the menu of View ==> choose Strategy Tester.

![Tester](https://c.mql5.com/2/54/Tester.png)

After that, we will find the Strategy Tester the same as the following of the Overview tab to choose what you need or the type of testing

![Tester1](https://c.mql5.com/2/54/Tester1.png)

Simply, through the Overview tab, we will choose what we need to test, and once we do that the tab will be changed to the Settings tab with predefined settings based on what you chose to test.

- Signal: for testing Expert advisors with signals without visualizing mode.
- Indicator: for testing an indicator.
- Visualize: for testing with visualizing mode during the testing on the chart to see how the program will react to the data based on its instructions.
- Stress & Delays: to test stress and delays as it is clear from its name.
- Optimization options (Complete, Genetic, Forward).
- Market scanner: for testing scanning for the market.
- Math calculations: for testing math calculations also.

You can also see previous tested results through the last option View previous results. You can also search for a specific previous test through the search below the options of testing.

If we move to the setting tab we will it the same as the following:

![ Tester2](https://c.mql5.com/2/54/Tester2.png)

-   1- Expert: choose the program file that we need to test.
-   2- IDE: to open the source code of the selected program.

-   3- to save or load specific settings.

-   4- Symbol: choose what symbol or symbol we need to test.

-   5- choose the timeframe of the selected symbol to test.

-   6- for the specifications of the selected symbol.

-   7- Date: choose the period that we need to test.

-   8- choose the tested starting period.

-   9- choose the tested ending period.

-  10- Forward: is Selecting the needed fraction of the optimization period that will set aside for forward testing.

-  11- select from or start the date of forwarding.

-  12- Delays, tune to get data close to actual.

-  13- to try to be close to real data and real executions.
-  13/2- Modeling: To select the model that you need to test is it by every tick or something else?

-  14- tick if you need profits in pips.

-  15- choose the deposit amount that we need to start with the testing.

-  16- choose the currency of the deposit.

-  17- to choose the leverage.

-  18- to choose if we want an optimization or not, we will take about it later.
-  19- tick if we need to visualize trades or executions during the testing.

We have also the Inputs tab to check if the tested program has inputs that can be inserted or edited by the user.

After determining our setting and pressing the Start button, the program will be tested and we can monitor that testing through the chart that will appear if we do visualize testing and tabs in this chart which Trade to see executed trades if our program does that, History to monitor closed or canceled orders, Operations to see all Operations, and Journal to check printed messages of the program and others. We can also monitor it through the Graph tab in the Strategy Tester window to monitor the performance of the program.

After the test is finished we can check also the Backtest tab that will be appeared as it will have all testing statistics and insights the same as the following example

![Tester3](https://c.mql5.com/2/54/Tester3.png)

![Tester4](https://c.mql5.com/2/54/Tester4.png)

As per the program you will test the data in this Backtest tab will appear like or different than what you see in the previous example.

There is one more thing I need to mention here as it is very important in the testing process which is the Optimization that includes selecting a wide range of different parameters and inputs and checking every possible combination of these inputs or parameters to check and test which parameters will generate the best results. Here we will choose one of the Optimization models from the Overview tab and we will select the suitable setting of Optimization the same as the following pictures

![Tester5](https://c.mql5.com/2/54/Tester5.png)

![Tester6](https://c.mql5.com/2/54/Tester6.png)

The Most important elements that we need to check and evaluate after testing:

- Net Profit: This is calculated by subtracting the gross loss from the gross profit.
- Draw Down: this is the maximum loss that the account experience during trades.
- Profit Factor: This is the ratio of gross profit to gross loss.
- Excepted Payoff: which is the average profit or loss of a trade.
- Recovery factor: which measures how well the tested strategy will recover after experiencing losses.

## Conclusion

We learned some essential topics about one of the most important and valuable tools that we all need to master when developing any MQL5 program which is the Testing process through the Strategy Tester of the MetaTrader 5 trading terminal and we learned some related topics that will help to understand and use this tool effectively. We learned about Errors that we face when developing MQL5 programs as per the types of these errors, which are:

- Compilation Errors and Warnings.
- Runtime Errors.
- Trade Server Errors.

Then we learned about Debugging through learning how to use the debugger of the MQL5 IDE and we learned how to use the Strategy Tester of the MetaTrader 5 through learning its different model, we can set settings of the testing based on the Model, how we can monitor the testing process through a different tab that reports everything when testing any MQL5 program, and after testing how we can read the report of the test and identified the most important elements that we need to check and evaluate and how we can optimize our strategy through testing to get the best result from the test MQL5 program.

I hope that you find this article and its information useful for you to help in improving your results through making effective testing and making your MQL5 programs. If you find the article useful and you need to read more articles for me you can check my other articles like sharing how we can create trading systems based on the most popular technical indicators.

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/447832)**
(2)


![Vitaliy Davydov](https://c.mql5.com/avatar/2019/3/5C98AE5B-7DD0.jpg)

**[Vitaliy Davydov](https://www.mql5.com/en/users/viteck116)**
\|
24 Sep 2023 at 09:30

Everything is good in MT5, except the strategy tester.

In my opinion, it loses to MT4 tester in terms of visual testing.

A simple addition of an indicator to a chart during [visual testing of](https://www.mql5.com/en/articles/2661 "Article: How to quickly develop and debug a trading strategy in MetaTrader 5") an Expert Advisor turns into some tambourine dancing.

And sometimes it is impossible at all.

Using templates, as advised, does not always work, and if it does, the Expert Advisor often stops working normally.

In MT4, you just throw an indicator on the chart  while testing an Expert Advisor and there are no problems at all.

I don't understand why it was necessary to spoil everything in the MT5 tester?

I created an Expert Advisor that creates a subwindow at startup and adds MACD to it using ChartIndicatorAdd.

Everything works fine in real life, but in the tester during visual testing it does not work at all.

And there are no errors in the logs. The indicator handle is created, the subwindow is created, but the indicator is not.

And there is no clue, the function returns "true".

![Rajesh Kumar Nait](https://c.mql5.com/avatar/2025/11/69247847-e34b.png)

**[Rajesh Kumar Nait](https://www.mql5.com/en/users/rajeshnait)**
\|
7 Aug 2024 at 10:13

**MetaQuotes:**

New article [Understand and Use MQL5 Strategy Tester Effectively](https://www.mql5.com/en/articles/12635) has been published:

Author: [Mohamed Abdelmaaboud](https://www.mql5.com/en/users/M.Aboud "M.Aboud")

Just want to update about Word error in article

Signal: for testing Expert advisors with signals without visualizing mode.

It should be "Single"

![Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://c.mql5.com/2/53/neural_network_experiments_p5_avatar.png)[Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://www.mql5.com/en/articles/12459)

Neural networks are an ultimate tool in traders' toolkit. Let's check if this assumption is true. MetaTrader 5 is approached as a self-sufficient medium for using neural networks in trading. A simple explanation is provided.

![Multibot in MetaTrader: Launching multiple robots from a single chart](https://c.mql5.com/2/53/launching_multiple_robots_avatar.png)[Multibot in MetaTrader: Launching multiple robots from a single chart](https://www.mql5.com/en/articles/12434)

In this article, I will consider a simple template for creating a universal MetaTrader robot that can be used on multiple charts while being attached to only one chart, without the need to configure each instance of the robot on each individual chart.

![Category Theory in MQL5 (Part 8): Monoids](https://c.mql5.com/2/54/Category-Theory-p8-avatar.png)[Category Theory in MQL5 (Part 8): Monoids](https://www.mql5.com/en/articles/12634)

This article continues the series on category theory implementation in MQL5. Here we introduce monoids as domain (set) that sets category theory apart from other data classification methods by including rules and an identity element.

![Implementing an ARIMA training algorithm in MQL5](https://c.mql5.com/2/54/Implementing_an_ARIMA_training_algorithm_in_MQL5_Avatar.png)[Implementing an ARIMA training algorithm in MQL5](https://www.mql5.com/en/articles/12583)

In this article we will implement an algorithm that applies the Box and Jenkins Autoregressive Integrated Moving Average model by using Powells method of function minimization. Box and Jenkins stated that most time series could be modeled by one or both of two frameworks.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zilyodntvhbdkhnqqbwmkpfqybxnkqop&ssn=1769180874114694756&ssn_dr=0&ssn_sr=0&fv_date=1769180874&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12635&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Understand%20and%20Use%20MQL5%20Strategy%20Tester%20Effectively%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918087482318731&fz_uniq=5069142719185486026&sv=2552)

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