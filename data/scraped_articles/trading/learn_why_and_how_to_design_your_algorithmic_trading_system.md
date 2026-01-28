---
title: Learn Why and How to Design Your Algorithmic Trading System
url: https://www.mql5.com/en/articles/10293
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:30:46.464621
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/10293&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049183314671150740)

MetaTrader 5 / Trading


### Introduction

We can say without any doubt that the importance of programming or coding increases every day in all fields of our world. So, we can observe how programming or coding can contribute to ease our life and not only make our life easy but guarantee accurate outputs according to what you want and what you determined before.

When it comes to the world of trading—this magnificent field and career—we can say that programming or coding makes our trading easy and systematic through created programs that contribute to give us that ease and automation once we completed an accurate and good program with what we expect and want to get from this program. So, the world of coding can give us a lot of benefits, but in my opinion the most important benefit for coding to trade is helping you to be disciplined as like we all know as traders there is always a challenge to be disciplined with our trading decision. Discipline is a great and important trait which affects our trading and investing results, so it is not an option to be disciplined at your trading. As I always like to say, Discipline is the key to your success in trading and whole life. To identify discipline in a simple way, it means to do what you have to do at the right time whatever surrounded circumstances. So, when we find a tool which helps to achieve this, we have to be attentive and learn what it is. And the right tool for this is coding.

As we all know, the most popular thing which prevents us from being disciplined while trading is emotions, and we have to avoid these emotions or avoid the effect of these emotions on our trading decisions in a negative way. And I want you to imagine if you have a system which works for you with your predetermined parameters without human interfering. So, in this case the negative effect of emotions on our trading decisions will be avoided. The good news is that we have a tool which helps us do that. Here I’ll write about MQL (MetaQuotes Language) for the MetaTrader platform. This great programming language or tool will help us to design our trading system with our specific parameters which will guarantee specific actions or specific trading decisions.

If you want to understand more about this concept, let us mention this example. If we have two investors (A&B) and their trading strategy is the same, it’s buying and holding during an uptrend and selling when reversing of the trend, but each one of them behave differently—investor A is disciplined but investor B is not. Look at the below figures:

![](https://c.mql5.com/2/44/1-_Investor_A.png)

![2- Investor_B](https://c.mql5.com/2/44/2-_Investor_B.png)

So, according to the previous figures, it’s obvious that discipline is essential for the good result, but the lack of discipline will drive to poor results.

And through this article I’ll share with you simple trading system Simple Moving Averages crossover to learn how to design your own one through sharing some of the basics of MQL5 coding with examples to practice these basics and to understand them deeply. The purpose is to give you an overview about what you can do using this magnificent tool.

The main objective of this article is to guide beginners to learn how to design their algorithmic trading system in MQL5 through learning some of the basics of MQL5 for a simple trading system idea which will be coded step by step through this article after explaining some of the MQL5 basics. We'll code them by scripts then we'll present the result after code execution. To enhance your understanding, I advise you to apply and code what you’ll read here by yourself as this will help you to understand concepts of mentioned codes deeply. And be noted that all created codes, programs, and trading strategies in this article are designed for educational purposes only, not for anything else. And be noted that we'll use MQL5 to write codes.

### What we need in order to design our algorithmic trading

In this part, I’ll mention what we’ll want to have as tools and what we want to know about these tools:

- The MetaTrader 5 platform, a.k.a. MetaTrader 5 Terminal. To execute orders and test our codes through the terminal. And MetaTrader is the most popular trading platform.

![3- MT5 platform](https://c.mql5.com/2/44/3-_MT5_platform.png)

A Demo Account. You can open a demo account with your broker to have virtual money to test your trading strategies risk-free but in the same market environment. Please be sure to use this demo account not your real account for coding as you will need to create and execute programs which will carry out transactions on your account.

MetaQuotes Language Editor, where we’ll write our codes or programs. The following screenshots will show how to open it as it is installed with MetaTrader. There are three ways to open it.

- Click Tools menu then MetaQuotes Language Editor:

![4- MetaEditor opening](https://c.mql5.com/2/44/4-_Metaeditor_opening_1.png)[https://c.mql5.com/2/44/Screen_Shot_2022-01-02_at_10.44.54_PM.png](https://c.mql5.com/2/44/Screen_Shot_2022-01-02_at_10.44.54_PM.png "https://c.mql5.com/2/44/Screen_Shot_2022-01-02_at_10.44.54_PM.png")

Or click MetaQuotes Editor Icon:

[![5- MetaEditor opening](https://c.mql5.com/2/44/5-_Metaeditor_opening_2.png)](https://c.mql5.com/2/44/Screen_Shot_2022-01-02_at_11.19.26_PM.png "https://c.mql5.com/2/44/Screen_Shot_2022-01-02_at_11.19.26_PM.png")

Or press F4 button from Keyboard while in the open terminal.

The following screenshot shows how it looks like, and here will be the most of our work to write our programs and design our trading systems.

[![6- MetaEditor window](https://c.mql5.com/2/44/6-_Metaeditor_window.png)](https://c.mql5.com/2/44/Screen_Shot_2022-01-02_at_11.26.00_PM.png "https://c.mql5.com/2/44/Screen_Shot_2022-01-02_at_11.26.00_PM.png")

Now we need to use this editor to write our first code, so follow the following steps to know how to do that.

Click on New then you will find more than one type of programs to choose from:

![7- MetaEditor window](https://c.mql5.com/2/44/7-_Metaeditor_window.png)

![8- MetaEditor - New](https://c.mql5.com/2/44/8-_Metaeditor_-_New_file.png)

What we need to mention here in this article are: Expert Advisor, Custom Indicator, and Script.

- Expert Advisor: EA is a program in the terminal that is developed and used for automation of analytical and trading processes according to the parameters you set.
- Custom Indicator: is a program coded; it is basically intended for graphical displaying of preliminarily calculated dependencies.
- Script: is a program intended for a single performing of any action, it can fulfill both analytical and trading functions, and unlike Expert Advisors, it is executed on request, not by ticks.

### The Hello World! program

In this part, we will learn how to write our first program and our first code in MQL5. All beginners in programming or coding start their journey by coding the “Hello World” code. So, we’ll start by writing a program that prints on the terminal screen “Hello World”. Let’s do it…

Open the MetaEditor as shown above, then click new, then select from the options (Script), then click next.

![9- MetaEditor - New file - Script](https://c.mql5.com/2/44/9-_Metaeditor_-_New_file_-_Script.png)

After clicking next, the following window will appear. Fill the data of the script that you want:

- **Name** is the name of the script
- **Author** is the script creator
- **Link** is your link
- **Parameter** is the parameters that you need to set for the script but we will not set any parameter for this script, so we will skip it.


Then click Finish.

![10- MetaEditor - Script info](https://c.mql5.com/2/44/10-_Metaeditor_-_Script_info.png)

After clicking Finish the following window will be opened. And in this window, our program will be coded. Here as I mention, we need to design a program that show “Hello World!” in the terminal. So, we’ll start wring our code between the two curly brackets.

![ 11 - Codes place](https://c.mql5.com/2/44/11_-_Codes_place.png)

Here we will use:

- Alert: it prints what I determine or displays a predetermined message in the program which is here in our example “Hello World!”
- ( “ ”): to write in between what I want or the predetermined message "Hello World!" or anything else.
- ; - to separate sentences.

Our code will be the same as:

```
//+----------------------------------------------------------+
//|                                         Hello World!.mq5 |
//|                          Copyright 2022, MetaQuotes Ltd. |
//|                                     https://www.mql5.com |
//+----------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+----------------------------------------------------------+
//| Script program start function                            |
//+----------------------------------------------------------+
void OnStart()
  {
   Alert("Hello World!");
  }
//+----------------------------------------------------------+
```

After writing our code then click Compile, then make sure that there are no errors and warnings. The Errors tab must be without errors after compiling your code to make sure that your program will run the same as what I mention in the following picture. Then go to the terminal by pressing F4 to test our code.

![12- Hello World code](https://c.mql5.com/2/44/12-_Hello_World_code.png)

After that in the Navigator window of the terminal, under scripts, you will find your program with the name which you determined before (Hello World!), drag it and drop on the chart or double click on it. Then you will see that an Alert message appears with what you determined at your code. It should be the same as in the following screenshot.

[https://c.mql5.com/2/44/5360096006160__1.png](https://c.mql5.com/2/44/5360096006160__1.png "https://c.mql5.com/2/44/5360096006160__1.png")![13- Hello World alert](https://c.mql5.com/2/44/13-_Hello_World_alert.png)

[https://c.mql5.com/2/44/5360096006160__2.png](https://c.mql5.com/2/44/5360096006160__2.png "https://c.mql5.com/2/44/5360096006160__2.png")

### **Trading strategy idea (Two Simple Moving Average crossover****)**

It this part, I’ll present a trading strategy idea for educational purposes only. It is aimed at helping you learn  the basics of MQL5  and how to program in MQL5.

**Disclaimer**

Any information is provided ‘as is’ solely for informational purposes and is not intended for trading purposes or advice. Past performance is no guarantee of future results. If you choose to use these materials on any trading accounts, you are doing so at your own risk.

Here, the strategy idea is trying to trade with the trend using a confirmation from the signal of two simple moving average indicator:

- Simple Moving Average is a lagging technical indicator which calculates the average closing price of a specific period, and it is a lagging as its signal comes after the signal of price action.
- The Strategy is as follows:

  - If shorter simple moving average (its period is 20) crossovers the longer simple moving average (its period is 50) to be above it, so the signal is Buy.
  - If shorter simple moving average (its period is 20) crossovers the longer simple moving average (its period is 50) to be below it, the signal is Sell.

Here, we need to design a program to do that.

![15- Trading Idea1](https://c.mql5.com/2/44/15-_Trading_Idea1.png)

![14- Trading Idea](https://c.mql5.com/2/44/14-_Trading_Idea.png)

### Algorithmic trading system blueprint

In this part I’ll talk about a very important step which you have to do if you want to code your system easily and smoothly. This step is to create a blueprint for your strategy and trading idea in manner of sequence of steps of what you want your system to do exactly, and you can do that through a diagram for example which will gives you a clear blueprint. And here is an example for our (Two Simple Moving Average Crossover) system blueprint to see in a clear way what we need to code and what we expect to get from this system.

![16- Simple 2MA-Blueprint-–-MQL5](https://c.mql5.com/2/44/16-_Simple_2MA-Blueprint-3-MQL5.png)

Now, let us first understand some of the basics of MQL5, then use what we need to design.

### Variables and types of them and how can we use

In this part, we will identify and understand:

- What are Variables?
- Types of Variables.
- How can we use them?

Generally speaking, in a program, a data values can be constant or variable. If values are variable they can be changed by the program and the user. A variable is a memory location. It has a name that is associated with that location. So, the memory location is used to hold data. A program in MQL5 can contain tens and hundreds of variables. A very important property of each variable is the possibility to use its value in a program. The limitation of this possibility is connected with the variable scope which is a location in a program where the value of the variable is available. Every variable has its scope.

So, according to the scope there are two types of variables in MQL5, local and global. A local variable is a variable declared within a function. The scope of local variables is the body of the function, in which the variable is declared. A local variable can be initialized by a constant or an expression corresponding to its type. Global variables are declared beyond all functions. The scope of global variables is the entire program. A global variable can be initialized only by a constant corresponding to its type (and not expression). Global variables are initialized only once before stating the execution of special functions.

Regardless of the scope of variables, now we will look at the following types of variables:

- int is a numerical type; there are different types of integers to store numerical values of different length.
- double. Building a program that will handle numbers used in trading requires a data type capable of managing floating-point numbers. So MetaTrader offers the following data types to deal with these data: float and double. The difference between them is the bytes allocated in memory, 4 for float and 8 for double, resulting in the following minimal and maximum values:

  - float - minimum 1.175494351e-38, maximum 3.402823466e+38

  - double - minimum 2.2250738585072014e-308, maximum 1.7976931348623158e+308

    To declare these types of data, use float and double keywords.

- string is a very important data type, which is widely used as well in MQL5 coding. String allows to store and manipulate any alphanumerical sequence of characters.
- bool - Boolean is a logical type that can assume value of either _true_ or _false_.

Let us take an example for using variable:

```
//+------------------------------------------------------------------+
//|                                                    Variables.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
    int myInteger = 5;
    double myDouble = 10.56;
    string myString = "My name is Mohamed";
    bool myBoolean = true;

    Alert(myInteger);
    Alert(myDouble);
    Alert(myString);
    Alert(myBoolean);
  }
//+------------------------------------------------------------------+
```

After compiling the previous code if you write like what I mention above, you will find no errors or warnings and you must see the Alert window like below screen shot.

![18- Variables](https://c.mql5.com/2/44/18-_Variables.png)

Another example for using variables.

Here we need to store or memorize variables and the value of them, A and its value 10, B = 10, C = 10 + 5, var1 = 2.5, var2 = 4, result = 2.5/4, message1 = Hello Mohamed, and message2 = Value of A is: 10. So, when we execute this code, the alert message will contain 4 elements:

- The equivalent value of message1
- The equivalent value of C
- The equivalent value of result
- The equivalent value of message2

```
//+------------------------------------------------------------------+
//|                                                  Variables 2.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   int A = 10;
   int B = 5;
   int C;

   double var1 = 2.5;
   double var2 = 4;
   double result = var1 / var2;

   string greeting = "Hello";
   string space = " ";
   string name = "Mohamed";
   string message1;
   string message2;

   C = A + B;
   message1 = greeting + space + name;
   message2 = "Value of A is: " + string(A);

   Alert(message1);
   Alert(C);
   Alert(result);
   Alert(message2);
  }
//+------------------------------------------------------------------+
```

### ![ 19- Variables 2](https://c.mql5.com/2/44/19-_Variables_2.png)

### Boolean operations

Boolean: simply it returns true or false for a logical operation.

- == means equal
- != means not equal
- < means lesser than
- <= means lesser than or equal
- \> means greater than
- >= means greater than or equal

```
//+------------------------------------------------------------------+
//|                                           Boolean Operations.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   bool result = 4 < 5;
   Alert (result);     //true
  }
//+------------------------------------------------------------------+
```

Here it must return true at alert message as 4 < 5.

![ 20- Boolean Operations](https://c.mql5.com/2/44/20-_Boolean_Operations.png)

### The While loop

The while operator consists of a checked expression and the operator, which must be fulfilled. If the expression is true, the operator is executed until the expression becomes false. In the following example, we can understand that first alerting at the start to know the start and end of the loop then the program will check the value of the counter and show its value if it is less than 3. Then it will add one to previous result till became false in this case = or > 3, then give the last alert when the loop has finished. So, we’ll see the following messages from this code in the alert message: Start of script, 1, 2, Loop has finished.

```
//+------------------------------------------------------------------+
//|                                                   While Loop.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   //While (Loop)
   Alert("Start of script");

   int counter = 1;

   while(counter < 3) //true?
    {
      Alert(counter);
      counter = counter + 1;
    }

    Alert("Loop has finished");
  }
```

![ 21- While Loop](https://c.mql5.com/2/44/21-_While_Loop.png)

### The For loop

The for operator consists of three expressions and an executable operator:

          for(expression1; expression2; expression3)

          operator;

Expression1 describes the loop initialization. Expression2 checks the conditions of the loop termination. If it is true, the loop body for is executed. The loop repeats expression2 until it becomes false. If it is false, the loop is terminated, and control is given to the next operator. Expression3 is calculated after each iteration.

So according to the following loop example of for, we can expect that the execution of the code will result five messages of Hello, (I =0, I = 2, …………,I = 4), then it will = 5, so the loop will be terminated.

```
//+------------------------------------------------------------------+
//|                                                     For Loop.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   for(int i=0; i<5 ; i++)
   {
   Alert("Hello");
   }
  }
//+------------------------------------------------------------------+
```

![22- For Loop](https://c.mql5.com/2/44/22-_For_Loop.png)

### The IF, Else statement

The IF - ELSE operator is used when a choice must be made. Formally, the syntax is as follows:

> if (expression)
>
> operator1
>
> else
>
> operator2

If the expression is true, operator1 is executed and control is given to the operator that follows operator2 (operator2 is not executed). If the expression is false, operator2 is executed.

Example: according to the following example, we need to receive an alert with the Bid value first, then we determine if Bid value > 1.146600, then we will be alerted with “The price is above 1.146600 -> BUY”. If it is not, then we will be alerted with “The price is below 1.146600 -> SELL”. And the following is the code and its execution result.

```
//+------------------------------------------------------------------+
//|                                            If-else statement.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   double level = 1.146600;
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
   Alert("Bid Price = " + string(Bid));

  if(Bid > level)
  {
  Alert ("The price is above " + string(level) + " -> BUY");
  }
// What if the condition is false and we need to take an action also here we can use else function instead of writing another if code
   else
   {
   Alert ("The price is below " + string(level) + " -> SELL");
   }

  }
//+------------------------------------------------------------------+
```

![ 23- If-Else Statement](https://c.mql5.com/2/44/23-_If-Else_Statement.png)

### Trader inputs

In this part we will learn how to determine our inputs or preferred parameters for the program instead of adjusting the code.

**Properties (#property):** Every MQL5-program allows to specify additional specific parameters named #property that help client terminal in proper servicing for programs without the necessity to launch them explicitly.

**script\_show\_inputs**: Display a window with the properties before running the script and disable this confirmation window

**Input variables:** Theinputstorage class defines the external variable.Theinputmodifier is indicated before the data type.

```
//+------------------------------------------------------------------+
//|                                                Trader Inputs.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
input int TakeProfit = 10;
input int StopLoss = 10;

void OnStart()
  {
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
   double TakeProfitLevel = Bid + TakeProfit * 0.00001; // 0.00001 (5 digits broker, so multiplied by 10)
   double StopLossLevel = Bid - StopLoss * 0.00001;

   Alert("Price now = " + string(Bid));
   Alert ("TakeProfitLevel = ", TakeProfitLevel);
   Alert ("StopLossLevel = ", StopLossLevel);
  }
//+------------------------------------------------------------------+
```

![24- Trader Input 1](https://c.mql5.com/2/44/24-_Trader_Input_1.png)

![24- Trader Input 2](https://c.mql5.com/2/44/24-_Trader_Input_2.png)

![ 24- Trader Input 3](https://c.mql5.com/2/44/24-_Trader_Input_3.png)

![24- Trader Input 4](https://c.mql5.com/2/44/24-_Trader_Input_4.png)

### Opening orders

In this part, I’ll present how to write your code to open an order:

```
//+------------------------------------------------------------------+
//|                                                         TEST.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
#include <Trade\Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
input int TakeProfit = 150;
input int StopLoss = 100;

void OnStart()
  {
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
   double Balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double Equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double TakeProfitLevel = (Ask+TakeProfit*0.00001);
   double StopLossLevel = (Ask-StopLoss*0.00001);


   if(Equity >= Balance)
   trade.Buy(0.01,NULL,Ask,StopLossLevel,TakeProfitLevel,NULL);

   for (int i=PositionsTotal()-1; i>=0; i--)
   {
     ulong ticket = PositionGetTicket(i);
     ENUM_POSITION_TYPE position = ENUM_POSITION_TYPE(PositionGetInteger(POSITION_TYPE));


     Alert (" Order Ticket # ", ticket);
     Alert("TakeProfit = ", TakeProfitLevel);
     Alert("StopLoss = ", StopLossLevel);
   }
  }
//+------------------------------------------------------------------+
```

After the execution of the script, the result will be as follows:

![25- Opening orders](https://c.mql5.com/2/44/25-_Opening_orders.png)

### Errors handling technique

When the program runs, anyone can find crashes, and something went wrong. So, we have to include this technique to our code to check every executed order or code and warns if something went wrong. In other words, it is a protection technique for the trader to protect him and his fund from any inappropriate situation.

```
//+------------------------------------------------------------------+
//|                                                         TEST.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
#include <Trade\Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
input int TakeProfit = 150;
input int StopLoss = 100;

void OnStart()
  {
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
   double Balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double Equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double TakeProfitLevel = (Ask+TakeProfit*0.00001);
   double StopLossLevel = (Ask-StopLoss*0.00001);


   if(Equity >= Balance)
   trade.Buy(0.01,NULL,Ask,StopLossLevel,TakeProfitLevel,NULL);

      for (int i=PositionsTotal()-1; i>=0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         ENUM_POSITION_TYPE position = ENUM_POSITION_TYPE(PositionGetInteger(POSITION_TYPE));

       if (ticket <= 0)
        {
         Alert("Error!");  //in Case of error and the order did not open, appears "Error!"

        }
      else
        {
         Alert("Your ticket # is: " + string(ticket));
         Alert("TakeProfit = ", TakeProfitLevel);
         Alert("StopLoss = ", StopLossLevel);
        }
      }
  }
//+------------------------------------------------------------------+
```

### ![26- Errors programming techniques 1](https://c.mql5.com/2/44/26-_Errors_programming_techniques_1.png)

### ![27- Errors programming techniques 2](https://c.mql5.com/2/44/27-_Errors_programming_techniques_2.png)

### Closing orders

In this part, we’ll create code to open and close an order to learn how to close an order.

```
//+------------------------------------------------------------------+
//|                                                         TEST.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
#include <Trade\Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
input int TakeProfit = 150;
input int StopLoss = 100;

void OnStart()
  {
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
   double Balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double Equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double TakeProfitLevel = (Ask+TakeProfit*0.00001);
   double StopLossLevel = (Ask-StopLoss*0.00001);



   trade.Buy(0.01,NULL,Ask,StopLossLevel,TakeProfitLevel,NULL);

   for (int i=PositionsTotal()-1; i>=0; i--)
   {
     ulong ticket = PositionGetTicket(i);
     ENUM_POSITION_TYPE position = ENUM_POSITION_TYPE(PositionGetInteger(POSITION_TYPE));


     Alert (" Order Ticket # ", ticket);
     Alert("TakeProfit = ", TakeProfitLevel);
     Alert("StopLoss = ", StopLossLevel);

     Sleep(2000);

     trade.PositionClose(ticket,-1);
     Alert("Order Closed...");
   }

  }
//+------------------------------------------------------------------+
```

![ 28- Closing orders](https://c.mql5.com/2/44/28-_Closing_orders.png)

### Adjusting orders by OrderModify

In this part, we will learn how to create code which help to modify order. It will modify the characteristics of the previously opened or pending orders.

```
//+------------------------------------------------------------------+
//|                                                         TEST.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
#include <Trade\Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
input int TakeProfit = 150;
input int StopLoss = 100;

void OnStart()
  {
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
   double Balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double Equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double TakeProfitLevel = (Ask+TakeProfit*0.00001);
   double StopLossLevel = (Ask-StopLoss*0.00001);
   double TakeProfitLevel2 = (TakeProfitLevel+0.00100);
   double StopLossLevel2 = (StopLossLevel-0.00050);


   trade.Buy(0.01,NULL,Ask,StopLossLevel,TakeProfitLevel,NULL);

   for (int i=PositionsTotal()-1; i>=0; i--)
   {
     ulong ticket = PositionGetTicket(i);
     ENUM_POSITION_TYPE position = ENUM_POSITION_TYPE(PositionGetInteger(POSITION_TYPE));


     Alert (" Order Ticket # ", ticket);
     Alert("TakeProfit = ", TakeProfitLevel);
     Alert("StopLoss = ", StopLossLevel);

     Sleep(5000);

     trade.PositionModify(ticket,StopLossLevel2,TakeProfitLevel2);
     Alert("Order Modified...");
     Alert("Modified TakeProfit = ", TakeProfitLevel2);
     Alert("Modified StopLoss = ", StopLossLevel2);
   }

  }
//+------------------------------------------------------------------+
```

![29- Modifying orders](https://c.mql5.com/2/44/29-_Modifying_orders.png)

### The Two Simple Moving Average Crossover system

In this part, we’ll put all together to code two simple moving averages crossover. All will do everything required according to the above-mentioned blueprint.

Here, we’ll choose an Expert Advisor when starting a new project by clicking new from MetaEditor.

To remember everything, here is our detailed trading system blueprint:

![16- Simple 2MA-Blueprint-–-MQL5](https://c.mql5.com/2/44/16-_Simple_2MA-Blueprint-l-MQL5.png)

What we need to do now is to code this strategy:

- If shorter Simple Moving Average (its period is 20) crosses the longer simple moving average(its period is 50)to be above it, the signal is Buy.
- If shorter Simple Moving Average (its period is 20) crosses the longer simple moving average (its period is 50) to be below it, the signal is Sell.

And the following is the code to this program and how to execute it.

```
//+------------------------------------------------------------------+
//|                                                SMA crossover.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+

void OnTick()
  {
   //create an array for several prices
   double myMovingAverageArray1[], myMovingAverageArray2[];

   //define the properties of  MAs - simple MA, 1st 20 / 2nd 50
   int movingAverage1 = iMA(_Symbol, _Period, 20, 0, MODE_SMA, PRICE_CLOSE);
   int movingAverage2 = iMA(_Symbol,_Period,50,0,MODE_SMA,PRICE_CLOSE);

   //sort the price arrays 1, 2 from current candle
   ArraySetAsSeries(myMovingAverageArray1,true);
   ArraySetAsSeries(myMovingAverageArray2,true);

   //Defined MA1, MA2 - one line - currentcandle, 3 candles - store result
   CopyBuffer(movingAverage1,0,0,3,myMovingAverageArray1);
   CopyBuffer(movingAverage2,0,0,3,myMovingAverageArray2);

   //Check if we have a buy entry signal
   if (
      (myMovingAverageArray1[0]>myMovingAverageArray2[0])
   && (myMovingAverageArray1[1]<myMovingAverageArray2[1])
      )
         {
         Comment("BUY");
         }

   //check if we have a sell entry signal
   if (
      (myMovingAverageArray1[0]<myMovingAverageArray2[0])
   && (myMovingAverageArray1[1]>myMovingAverageArray2[1])
      )
         {
         Comment("SELL");
         }
  }
//+------------------------------------------------------------------+
```

After executing the program, it will display comments on the chart with the current signal (buy or sell) according to this strategy. This is shown in the following picture:

![30- SMA - comment](https://c.mql5.com/2/44/30-_SMA_-_comment.png)

![ 31- SMA - Sell comment](https://c.mql5.com/2/44/31-_SMA_-_Sell_comment.png)

Now the following pictures will explain how we can find our designed system from the terminal and how to execute it:

![32- Simple MA program place](https://c.mql5.com/2/44/32-_Simple_MA_program_place.png)

After drag and drop or double click on the program, the following window will pop up:

![33- Simple MA program interface](https://c.mql5.com/2/44/33-_Simple_MA_program_interface.png)

Enable the 'Allow Algo Trading' option, then click Ok. After that the Expert Advisor will be attached to the chart and a message will appear in the Journal tab as it is loaded successfully:

![ 34- Simple MA program activated1](https://c.mql5.com/2/44/34-_Simple_MA_program_activated1.png)

![35- Simple MA program activated2](https://c.mql5.com/2/44/35-_Simple_MA_program_activated2.png)

So, we have created and executed our automation Two Simple Moving Averages Crossover program. And what I want to mention here is that this is a simple example for what we can code for our trading. However, we can code and automate any other trading strategy using this magnificent tool MQL5.

Again, the main objective from this article is not to use this strategy but the main objective is to learn some of the basics of MQL5 and to learn what we can create to imagine what we can do and to make our trading easier and profitable. So, this strategy needs a lot of enhancements and if you use it, it will be your only responsibility.

### Conclusion

MQL5 is a good and magnificent tool to help us make our trading easier and profitable through designing good and profitable algorithmic trading systems. So, your investment in learning and using this magnificent tool will be a highly rewardable investment.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10293.zip "Download all attachments in the single ZIP archive")

[Boolean\_Operations.mq5](https://www.mql5.com/en/articles/download/10293/boolean_operations.mq5 "Download Boolean_Operations.mq5")(0.84 KB)

[Closing\_Orders.mq5](https://www.mql5.com/en/articles/download/10293/closing_orders.mq5 "Download Closing_Orders.mq5")(1.79 KB)

[Errors\_programming\_technique.mq5](https://www.mql5.com/en/articles/download/10293/errors_programming_technique.mq5 "Download Errors_programming_technique.mq5")(1.92 KB)

[For\_Loop.mq5](https://www.mql5.com/en/articles/download/10293/for_loop.mq5 "Download For_Loop.mq5")(0.85 KB)

[Hello\_Worldb.mq5](https://www.mql5.com/en/articles/download/10293/hello_worldb.mq5 "Download Hello_Worldb.mq5")(0.76 KB)

[If-else\_statement.mq5](https://www.mql5.com/en/articles/download/10293/if-else_statement.mq5 "Download If-else_statement.mq5")(1.25 KB)

[Modifying\_Orders.mq5](https://www.mql5.com/en/articles/download/10293/modifying_orders.mq5 "Download Modifying_Orders.mq5")(2.04 KB)

[Opening\_orders.mq5](https://www.mql5.com/en/articles/download/10293/opening_orders.mq5 "Download Opening_orders.mq5")(1.71 KB)

[Trader\_Inputs.mq5](https://www.mql5.com/en/articles/download/10293/trader_inputs.mq5 "Download Trader_Inputs.mq5")(1.25 KB)

[Variables\_2.mq5](https://www.mql5.com/en/articles/download/10293/variables_2.mq5 "Download Variables_2.mq5")(1.22 KB)

[Variables.mq5](https://www.mql5.com/en/articles/download/10293/variables.mq5 "Download Variables.mq5")(1.01 KB)

[While\_Loop.mq5](https://www.mql5.com/en/articles/download/10293/while_loop.mq5 "Download While_Loop.mq5")(1 KB)

[Simple\_MA\_crossover.mq5](https://www.mql5.com/en/articles/download/10293/simple_ma_crossover.mq5 "Download Simple_MA_crossover.mq5")(1.92 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/387124)**
(19)


![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
8 May 2022 at 19:45

**paulus itumeleng [#](https://www.mql5.com/en/forum/387124/page2#comment_39458815):**

I'm new in trading I need a scalping robot

Don't we all?

Such discussions are not allowed in the forum, you should make your own search.

![76877374](https://c.mql5.com/avatar/avatar_na2.png)

**[76877374](https://www.mql5.com/en/users/76877374)**
\|
18 Jul 2022 at 13:22

I don't know how this game is played


![Florian Silver Grunert](https://c.mql5.com/avatar/avatar_na2.png)

**[Florian Silver Grunert](https://www.mql5.com/en/users/bnd)**
\|
8 Aug 2022 at 14:09

I really like your work and ideas. If you don't mind could you consider the Ultimate Oscillator indicator. I would connect it with Mass Indicator (MI). What do you think?


![Johnson Mukiri Mbuthia](https://c.mql5.com/avatar/2020/12/5FCA224E-8F57.jpg)

**[Johnson Mukiri Mbuthia](https://www.mql5.com/en/users/mukiri.johnson)**
\|
8 Sep 2022 at 09:17

I like the simplicity of how you explain things in all your articles Mohammed. I develop in MQL4 mainly. What's the quickest way to transition to MQL5 and what are the benefits in your opinion?

Thanks

![pierluigimaps](https://c.mql5.com/avatar/2026/1/6967ca9b-057e.jpg)

**[pierluigimaps](https://www.mql5.com/en/users/pierluigimaps)**
\|
11 Nov 2024 at 18:38

thank you


![Graphics in DoEasy library (Part 89): Programming standard graphical objects. Basic functionality](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__1.png)[Graphics in DoEasy library (Part 89): Programming standard graphical objects. Basic functionality](https://www.mql5.com/en/articles/10119)

Currently, the library is able to track standard graphical objects on the client terminal chart, including their removal and modification of some of their parameters. At the moment, it lacks the ability to create standard graphical objects from custom programs.

![WebSockets for MetaTrader 5 — Using the Windows API](https://c.mql5.com/2/44/huge81r.png)[WebSockets for MetaTrader 5 — Using the Windows API](https://www.mql5.com/en/articles/10275)

In this article, we will use the WinHttp.dll to create a WebSocket client for MetaTrader 5 programs. The client will ultimately be implemented as a class and also tested against the Deriv.com WebSocket API.

![Universal regression model for market price prediction (Part 2): Natural, technological and social transient functions](https://c.mql5.com/2/43/universal_regression__1.png)[Universal regression model for market price prediction (Part 2): Natural, technological and social transient functions](https://www.mql5.com/en/articles/9868)

This article is a logical continuation of the previous one. It highlights the facts that confirm the conclusions made in the first article. These facts were revealed within ten years after its publication. They are centered around three detected dynamic transient functions describing the patterns in market price changes.

![Manual charting and trading toolkit (Part III). Optimization and new tools](https://c.mql5.com/2/43/MQL5-set_of_tools.png)[Manual charting and trading toolkit (Part III). Optimization and new tools](https://www.mql5.com/en/articles/9914)

In this article, we will further develop the idea of drawing graphical objects on charts using keyboard shortcuts. New tools have been added to the library, including a straight line plotted through arbitrary vertices, and a set of rectangles that enable the evaluation of the reversal time and level. Also, the article shows the possibility to optimize code for improved performance. The implementation example has been rewritten, allowing the use of Shortcuts alongside other trading programs. Required code knowledge level: slightly higher than a beginner.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/10293&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049183314671150740)

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