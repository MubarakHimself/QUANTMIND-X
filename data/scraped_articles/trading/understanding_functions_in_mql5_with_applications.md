---
title: Understanding functions in MQL5 with applications
url: https://www.mql5.com/en/articles/12970
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:06:44.378673
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/12970&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069108608555221122)

MetaTrader 5 / Trading


### Introduction

In the programming world there is a very popular term that we use and hear a lot as per its importance in any software which is the Function. In this article, we will learn about it in detail to learn how to create very functional and high-quality software. We will try to cover this important topic to be aware of what are functions, why we need to use functions, and how to use them in our applications. After that, we will share some simple functions that can be used in any trading system as examples to apply what we will learn through this article. The following points are the main ideas that we will share in this article to cover this interesting topic:

- [Function definition](https://www.mql5.com/en/articles/12970#definition)
- [Function structure](https://www.mql5.com/en/articles/12970#structure)

  - [Function with arguments](https://www.mql5.com/en/articles/12970#with)
  - [Function without arguments](https://www.mql5.com/en/articles/12970#without)
  - [Function with default values](https://www.mql5.com/en/articles/12970#default)
  - [Passing parameters](https://www.mql5.com/en/articles/12970#passing)
  - [Return operator](https://www.mql5.com/en/articles/12970#return)
  - [Void type function](https://www.mql5.com/en/articles/12970#void)
  - [Function overloading](https://www.mql5.com/en/articles/12970#overloading)

- [Function applications](https://www.mql5.com/en/articles/12970#applications)

  - [News alert App](https://www.mql5.com/en/articles/12970#news)
  - [Lot size Calc App](https://www.mql5.com/en/articles/12970#lotsize)
  - [Close All App](https://www.mql5.com/en/articles/12970#close)

- [Conclusion](https://www.mql5.com/en/articles/12970#conclusion)

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Function definition

In this part, we will identify the function in programming, the types of them, and why we need to use them. The function is a block of code declared by an expressed and meaningful name that can be used in any other part of the software by calling it over and over to perform a specific task. If we have a specific task that we need the software to perform in many parts of our software or in many software, we create a function or block of code to perform this task and then call it only in these parts without rewriting the full code again so, we can say that function is a method to abstract our code without having a lot of messy code the same as we will see through this article. We have two main types of these functions, we have built-in functions and user-defined functions. The built-in function is the ready-made function by the programming language itself while the user-defined function is the function that can be created by the user as per his needs or tasks that he needs the software to perform. We will focus in this article on user-defined functions. So, we will learn about this type in detail to see why we need to use this type of function and how much its importance or features of using them.

Let's say that we need the software to perform the task of closing all open orders if the equity reached a maximum drawdown and we need to perform this task in many parts of the software, it will be better here to create a function and include all needed code or logic to perform this task then call this function in other parts but it will be not good or overwhelmed to write and repeat the same code in many parts to do the task.

If you are asking why we need to use this type of function, the answer to this question will lead us to learn the features of using user-defined functions and the following is for that:

- It helps to apply the concept of DRY (do not repeat yourself): By using user-defined functions will help us to not repeat the same code over and over but we will create a function that can perform our task one time and then call it in any suitable part in software.
- Reusability:After creating our function we can reuse it at any time.
- It helps to apply the concept of divide and conquer: When we create software the code can be complex to solve a problem but if we divide the big problem into small ones and solve each one through functions this can be very helpful to achieve our objective in solving the big problem.
- It helps that the code is more readable and understandable: When we use functions, it helps that it makes our code readable as it became more organized as it has functions and everyone handles a specific problem and has a specific task.
- It helps to apply the concept of abstraction: Using functions gives a method to abstract our code because if we do not use them we may find that we need to write more lines of code than using functions.
- It helps to apply the concept of encapsulation: When we use functions, it allows us to secure and manage our code and data more than if we did not use them.
- It improves the debugging process: When we use functions, it helps to improve errors exploring and solving them much easier.

According to what we mentioned about features of using functions, we can easily find how much will be beneficial when we use these user-defined functions in our software.

### Function structure

In this part, we will learn more details about functions and the structure of functions and we will learn about that through the following two steps:

- Function declaration or definition
- Function call

First thing, we need to define or declare a new function so we need to do something similar to the following structure:

```
returnedDataType functionName(param1, param2)
{
        bodyOfFunction
}
```

- the (returnedDataType): is the type of data that we need the function to return after execution.
- the (functionName): is the name of the function and we give this name as per the task that the function is performing.
- the (param1, param2): here we add variables or placeholders that we need if essential as we may not specify any of them.
- the (bodyOfFunction): we specify all code that will perform our task.

Let us apply that to a simple example if we need to create a simple function to perform an addition task of two values, we can do that through the following block of code:

```
//addition function
// returned data type is an integer - the name of the function is add - parameters or arguments are two int variables val1 and val2
int add(int val1, int val2)
  {
   //body of function that we need the function to perform when calling it
   //create a result new variable to be assigned by the result of val1 and val2 addition
   int result = val1+val2;
   //Print result in the experts tab
   Print(result);
   //returning value
   return result;
  }
```

After defining our function we need to do the second step which is calling the function and we can do that by calling the name of the function and specifying desired parameters as per our function at the desired part of software code. Back to our example if we want to call our function it will be the same as the following:

```
   //calling our defined function by its name and specifying arguments
   add(5,15);
```

When we call it we can get the result of the 20 value in the experts' tab as per our function and specified arguments the same as the following picture.

![add function result](https://c.mql5.com/2/181/add_function_result__2.png)

The previous example is just a sample of what we can do by using function but there are many available characteristics that we can use at functions and the following are some of them.

#### Function with arguments

We see in the last add function example that we used two integer variables which were val1 and val2, these variables are considered as arguments in our function. These arguments can be the type of data like integer, string, ..etc. In our last example, they were integer variables the same as we saw and we can see another example of string arguments the same as the following:

```
//sayHello function
// returned data type is string - name of function is sayHello - parameters or arguments are two string variables greeting and name
string sayHello(string greeting, string name)
  {
   //body of function that we need the function to perform when calling it
   //create a result new variable to be assigned by the result of greeting and name addition
   string result = greeting+name;
   //Print result in the experts tab
   Print(result);
   //returning value
   return result;
  }
```

We can call this function the same as we mentioned and we can find its result after execution the same as the following:

![sayhello function result](https://c.mql5.com/2/181/sayhello_function_result__2.png)

They can be also a mix of these data types as per what we need in our function as parameters or arguments to execute the body of the function of them. These arguments also can be the number that we need as per our needs.

#### Function without arguments

The function can be declared or defined without specifying parameters or arguments also just give a meaningful name for the function and then leave the arguments empty then complete by filling the body of the function to perform the task then call the function without specifying arguments the same as the following:

```
//sayHello function
// returned data type is a string - the name of the function is sayHello - no parameters
string sayHello()
  {
   //body of the function that we need the function to perform when calling it
   //create a result new variable to be assigned by the result of greeting and name addition
   string greeting= "Hello, ";
   string name= "World!";
   string result = greeting+name;
   //Print the result in the experts' tab
   Print(result);
   //returning value
   return result;
  }
```

When we call the function it will be the same as the following:

```
sayHello();
```

The result will be the same as what we mentioned before in the function with arguments as the body of the function is the same.

#### Function with default values

We define a function also and give initial or default values for parameters but we are still able to change or update them with desired values and this can be the same as the following when applying the same example

```
//defining function with default values
string sayHello(string greeting= "Hello, ", string name="World!")
  {
   string result = greeting+name;
   Print(result);
   return result;
  }
```

Then we can call the function twice to identify the difference between default values and if we update them but I need also to mention here we can here to specify parameters if we will update them but if we do not specify them the function will return default values the same as the following:

```
   sayHello();
   sayHello("Hi, ", "Developer!");
```

The result will be the same as the following:

![sayhello function def values result](https://c.mql5.com/2/181/sayhello_function_def_values_result__2.png)

#### Passing parameters

We can pass to the function values and these values can be any type of data as we mentioned int, string, array,.. etc. By passing parameters by values, the original variables in the function will remain the same or unchanged as we passed the value of the parameters to the function. We can also pass parameters to the function by reference if we need to update the original variables.

The following is a simple example to understand what we mentioned about passing by reference.

```
//passing by reference
void updateNums(int &val1, int &val2)
  {
   val1*=2;
   val2/=2;
  }
```

Then we will create new variables then print their values then call our function with these new variables as parameters and print its values after calling to release the difference:

```
//new variables
   int firstNum = 10;
   int secondNum = 20;

//before calling function
   Print("before calling: ");
   Print(firstNum, " - " ,secondNum, "\n");

// calling
   updateNums(firstNum, secondNum);

// after calling
   Print("after calling: ");
   Print(firstNum, " - " ,secondNum, "\n");
```

So, the result of two prints will be the first one will be the values of the new variables 10, and 20, and after calling we will find the values after updating by 20, and 10 as per the body of the function. We can see the result is the same as the following:

![passing by reference](https://c.mql5.com/2/181/pass_by_reference__2.png)

#### Return operator

If we have a function that returns a value, it must have a return operator. We may have more than one return operator in the function based on the function task but if the function returns a value, so, it must have at least one return in the last line of the function. This return operator may be any type but cannot be an array but we can return an element from the array. If we want the function to return an array we can pass the array to the function by reference the same as we mentioned before.

The following is an example from what we mentioned before that has a return operator

```
string sayHello(string greeting= "Hello, ", string name="World!")
  {
   string result = greeting+name;
   Print(result);
   return result;
  }
```

#### Void type function

If we have a function that will not return a value we use the void type function as this type does not return a value. We can pass parameters to this type of function normally but it does not need a return operator. The following is an example of this type of functions.

```
void add(int val1, int val2)
  {
   int result= val1+val2;
  }
```

#### Function overloading

There are some cases when defining functions we need to define many functions under the same name to perform the same task but on different parameters. For example, if we have an addition task, but we need to perform this addition to two values and we need to perform the same task to three values, in this case, we create two functions under the same name of the function but we will change parameters as per the task. This means that we have an overloading function, so the overloading function is a function that performs the same task but on different parameters.

These different parameters can be different data types of parameters, numbers of the same data type, or both of them. The following is an example of an overloading function with the same data type but the number of parameters is different:

```
void overloadingFun(int val1, int val2)
{
   int result=val1+val2;
}

void overloadingFun(int val1, int val2, int val3)
{
   int result=val1+val2+val3;
}
```

As we can see we have the same function but the parameters are different. When we call the function we find that these two function appears when typing the name of the function then we can choose what we need as per our task details. The following is an example of an overloading function with different parameters as per data type:

```
void overloadingFun(int val1, int val2)
{
   int result=val1+val2;
}

void overloadingFun(string message, int val1, int val2)
{
   int result=message+val1+val2;
}
```

We can choose also the function that we need as per parameters when calling the function.

### Function Applications

In this part, we will create simple applications that we can use the function for to be beneficial from the user-defined function and make the coding process easier. After creating these applications we can call them in different parts of the software or even in another software by including them.

#### News alert App

We all know that trading during economic news is very risky and there are a lot of professional pieces of advice that we do not trade during news. If you do not know what is the economic calendar, it is a ready-made calendar that has macroeconomic news and indicators with descriptions, their date, time, and degree of importance, and released values of these economic events and there are many sources that can be used to get these important values to be updated with what can affect the market and trade according to that. We have this calendar also in the MetaTrader 5 trading terminal as you can find its tab in the Toolbox window and you can control what you need to view in terms of importance, currencies, and countries. There are also built-in functions for working with the economic calendar and you can check them all in the MQL5 documentation through the following link:

[Economic Calendar Functions](https://www.mql5.com/en/docs/calendar)

So, we need to check the news economic calendar manually to avoid trading during news or we create an app to alert us when we are approaching news to stop trading this task is a permanent task so we will need it in any trading system or many parts of software. So, we can create a function for that and then call it easily and this is what we will do in this part through the following steps:

This app will be an EA, in the global scope we will create a bool type for the name of our function which is (isNewsComing) and then we will not add parameters ()

```
bool isNewsComing()
```

The body of the function is creating an array with values name and its type will be (MqlCalendarValue) which will be values of the news release like actual value for example

```
MqlCalendarValue values[];
```

We need to define the current day by defining the starting time of the day by using (iTime) to return the opening time of the bar after declaring a new datetime variable with the name of (startTime) and the ending time of the day to be equal to the defined start time and the number of second of the day by using the (PeriodSeconds) function after declaring a new datetime variable to the (endTime)

```
   datetime startTime=iTime(_Symbol,PERIOD_D1,0);
   datetime endTime=startTime+PeriodSeconds(PERIOD_D1);
```

Getting the array of values of all events in the day by using defined the start time and the end time to determine the time range and sorting by the current country and currency by using the CalendarValueHistory function and the parameters are the array for values, start time, end time, country, and currency

```
CalendarValueHistory(values,startTime,endTime,NULL,NULL);
```

We will create a loop and this loop will start with a value of (0) for the created (i) int variable and continue looping if the (i) is less than the array size of values array and increment the (i) by one value

```
for(int i=0; i<ArraySize(values); i++)
```

The body of the for loop is creating an event variable and its type is (MqlCalendarEvent) for the event descriptions and it can be used in the (CalendarEventById)

```
MqlCalendarEvent event;
```

Getting the event description by its ID by using the (CalendarEventById) and its parameters are event\_id and the event

```
CalendarEventById(values[i].event_id,event);
```

Create a country variable and its type will be (MqlCalendarCountry) for the country descriptions and it can be used with the (CalendarCountryById)

```
MqlCalendarCountry country;
```

Getting the country descriptions by its ID by using the (CalendarCountryById) function and its parameters are country\_id and the country

```
CalendarCountryById(event.country_id,country);
```

Setting conditions to filter events by the current symbol or currency news, the importance of news is medium or high, and if there is something else that continues

```
      if(StringFind(_Symbol,country.currency)<0)
         continue;

      if(event.importance==CALENDAR_IMPORTANCE_NONE)
         continue;
      if(event.importance==CALENDAR_IMPORTANCE_LOW)
         continue;
```

Setting a condition with the time range of alert that we need it will be 30 seconds before the time of news

```
      if(TimeCurrent()>=values[i].time-30*PeriodSeconds(PERIOD_M1) &&
         TimeCurrent()<values[i].time+30*PeriodSeconds(PERIOD_M1))
```

Then we need to get a message printed in the experts' tab with the event name and the text of ( is coming! Stop Trading...)

```
Print(event.name, " is coming! Stop Trading...");
```

The returned value will be true

```
return true;
```

#### If conditions are false finish the loop and return false to terminate the function

```
return false;
```

Then we can call the function in the OnTick function and if it returns true we need a printed message of (News is coming...!)

```
   if(isNewsComing())
     {
      Print("News is comming...!");
     }
```

Now, we created the function and called it and we can use it in any part of our software as per our needs the following is for the full code in one block to be easy to read it again and you will find the source code files of all applications are attached to the article

```
//+------------------------------------------------------------------+
//| News Alert Function                                              |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(isNewsComing())
     {
      Print("News is comming...!");
     }
  }
//+------------------------------------------------------------------+
bool isNewsComing()
  {
   MqlCalendarValue values[];
   datetime startTime=iTime(_Symbol,PERIOD_D1,0);
   datetime endTime=startTime+PeriodSeconds(PERIOD_D1);
   CalendarValueHistory(values,startTime,endTime,NULL,NULL);
   for(int i=0; i<ArraySize(values); i++)
     {
      MqlCalendarEvent event;
      CalendarEventById(values[i].event_id,event);
      MqlCalendarCountry country;
      CalendarCountryById(event.country_id,country);
      if(StringFind(_Symbol,country.currency)<0)
         continue;
      if(event.importance==CALENDAR_IMPORTANCE_NONE)
         continue;
      if(event.importance==CALENDAR_IMPORTANCE_LOW)
         continue;
      if(TimeCurrent()>=values[i].time-30*PeriodSeconds(PERIOD_M1) &&
         TimeCurrent()<values[i].time+30*PeriodSeconds(PERIOD_M1))
        {
         Print(event.name, " is coming! Stop Trading...");
         return true;
        }
     }
   return false;
  }
//+------------------------------------------------------------------+
```

#### Lotsize calc App

We need to create an app that can be able to calculate the optimal lot size after determining the risk percentage and maximum loss in pips, and we need to create an overloaded function to calculate the optimal lot size after determining the risk percentage and the entry price and the stop loss price. We will create this app as a script the same as the following steps:

Create the function with the name OptimalLotSize as a double and parameter will be two for the first function, the double variable of maximum risk percentage, and the double variable of maximum loss in pips the same as the following

```
double OptimalLotSize(double maxRiskPrc, double maxLossInPips)
```

Then we specify what we need to perform for these parameters, first, we will define the account equity value by using the (AccountInfoDouble) function that returns the value of the suitable account property which is here the identifier of the account equity as (ENUM\_ACCOUNT\_INFO\_DOUBLE) and create an alert with the value

```
   double accEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   Alert("accEquity: ", accEquity);
```

Define the symbol contract size by using the (SymbolInfoDouble) function that returns the corresponding property of a specified symbol with its variant of the name of the symbol which will be (\_Symbol) to return the current symbol and prop\_id which will be (SYMBOL\_TRADE\_CONTRACT\_SIZE) as one of the (ENUM\_SYMBOL\_INFO\_DOUBLE) values after that we need an alert with this returned value

```
   double lotSize = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_CONTRACT_SIZE);
   Alert("lotSize: ", lotSize);
```

Calculating the pip value and getting an alert with the value

```
   double tickValue = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   Alert("tickValue: ", tickValue);
```

Calculating the maximum value of the loss from the defined account equity and getting an alert with this value

```
   double maxLossDollar = accEquity * maxRiskPrc;
   Alert("maxLossDollar: ", maxLossDollar);
```

Calculating the maximum value in the quote currency based on the calculated maximum loss value and returning an alert with this value

```
   double maxLossInQuoteCurr = maxLossDollar / tickValue;
   Alert("maxLossInQuoteCurr: ", maxLossInQuoteCurr);
```

Calculating the optimal lot size and returning an alert with the value

```
   double OptimalLotSize = NormalizeDouble(maxLossInQuoteCurr / (maxLossInPips * 0.0001)/ lotSize,2);
   Alert("OptimalLotSize: ", OptimalLotSize);
```

The return operator will be returned OptimalLotSize as a double-type value

```
return OptimalLotSize;
```

After that, we will create the overloading function by passing three double-type parameters for the maximum risk percentage, the entry price, and the stop loss price

```
double OptimalLotSize(double maxRiskPrc, double entryPrice, double stopLoss)
```

Defining the maximum loss in pips as an absolute value based on input parameters of entry price and stop loss price then dividing by 0.0001

```
double maxLossInPips = MathAbs(entryPrice - stopLoss)/0.0001;
```

The return operator will be the created OptimalLotSize function with its parameters the maximum risk percentage and the maximum loss in pips

```
return OptimalLotSize(maxRiskPrc,maxLossInPips);
```

Then we can call any of two functions in the OnStart() part as per what we need for example the same as the following

```
OptimalLotSize(0.01, 1.12303, 1.11920);
```

The following is the full code to create this type of function to create this type of app

```
//+------------------------------------------------------------------+
//| lotSize Calc Function                                            |
//+------------------------------------------------------------------+
void OnStart()
  {
   OptimalLotSize(0.01, 1.12303, 1.11920);
  }
//+------------------------------------------------------------------+
double OptimalLotSize(double maxRiskPrc, double maxLossInPips)
  {
   double accEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   Alert("accEquity: ", accEquity);
   double lotSize = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_CONTRACT_SIZE);
   Alert("lotSize: ", lotSize);
   double tickValue = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   Alert("tickValue: ", tickValue);
   double maxLossDollar = accEquity * maxRiskPrc;
   Alert("maxLossDollar: ", maxLossDollar);
   double maxLossInQuoteCurr = maxLossDollar / tickValue;
   Alert("maxLossInQuoteCurr: ", maxLossInQuoteCurr);
   double OptimalLotSize = NormalizeDouble(maxLossInQuoteCurr / (maxLossInPips * 0.0001)/ lotSize,2);
   Alert("OptimalLotSize: ", OptimalLotSize);
   return OptimalLotSize;
  }
//+------------------------------------------------------------------+
double OptimalLotSize(double maxRiskPrc, double entryPrice, double stopLoss)
  {
   double maxLossInPips = MathAbs(entryPrice - stopLoss)/0.0001;
   return OptimalLotSize(maxRiskPrc,maxLossInPips);
  }
//+------------------------------------------------------------------+
```

By executing this script we will get the following alert

![ lotSize app](https://c.mql5.com/2/181/lotSize_app__2.png)

As per the previous mentioned application, we have the lotSize Calc function that we can use and call in different software parts easily to perform the task without rewriting the code again.

#### Close All App

We need here to create a script that can be close opened and pending orders by creating a function that can be used or called in any suitable part of the software to perform this task. We can do that through the following steps:

Including the Trade class to the code by using the preprocess or #include to include all trading functions in the (Trade.mqh) file

```
#include <Trade/Trade.mqh>
```

Creating an object with the type of CTrade class to be used in the software

```
CTrade trade;
```

In the global scope also, we need to create a void closeAll function without arguments

```
void closeAll()
```

The body of the function is creating a for loop to check for open orders

```
for(int i=PositionsTotal()-1; i>=0; i--)
```

The body of the loop, creating the ulong posTicket variable and assigning the ticket of opened orders to it

```
ulong posTicket=PositionGetTicket(i);
```

Close opened trade by using trade.PositionClose(posTicket)

```
trade.PositionClose(posTicket);
```

We will delete pending orders by creating another for loop to detect these orders, assigning their tickets to the ulong variable of posTicket, deleting the pending order by its detected ticket

```
   for(int i=OrdersTotal()-1; i>=0; i--)
     {
      ulong posTicket=OrderGetTicket(i);
      trade.OrderDelete(posTicket);
     }
```

After that, we can call this function in the OnStart() part

```
closeAll();
```

By executing this script, it will close and delete all orders. The following is the full code to create this closeAllApp in one block:

```
//+------------------------------------------------------------------+
//| closeAll Function                                                |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
void OnStart()
  {
   closeAll();
  }
//+------------------------------------------------------------------+
void closeAll()
  {
//close all open positions
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong posTicket=PositionGetTicket(i);
      trade.PositionClose(posTicket);
     }
//delete all pending orders
   for(int i=OrdersTotal()-1; i>=0; i--)
     {
      ulong posTicket=OrderGetTicket(i);
      trade.OrderDelete(posTicket);
     }
  }
//+------------------------------------------------------------------+
```

These mentioned applications are just examples of what we can create as user-defined functions and you can develop them or create any other applications or functions as per your needs like trailing stop and trade management application and draw down safety tool ..etc.

### Conclusion

According to what we mentioned in this article, it is supposed that you found that it is crucial to use functions in your software because of all features and benefit that you will get when doing that as it is supposed that you identified features of using them like:

- It helps apply the DRY (do not repeat yourself) concept in programming.
- It helps to minimize any big problem by dividing it into small ones and dealing with them.
- Making the code more readable.
- Reusability.
- Abstracting the code.
- Encapsulating the code.
- debugging improvement.

It is supposed also that you understood well what is the function, its types the built-in and user-defined function, and how you can create or define functions through learning functions' structure and all characteristics of them like function with arguments and how we can pass these arguments or parameters, without arguments, and functions with default values. It is supposed that you can define your function using any data type and deal with the return operator based on that and you know how to create many overloading functions with the same name but different arguments to perform the same task.

After sharing applications to create functions as examples, I believe that helped a lot to deepen your understanding of the topic as we create two different functions:

- New Alert App: to be used or called in any part of the software to get alerts when important news is coming. You will find the newsAlertApp source code attached.
- Lot size Calc App: to be used or called in any part of the software to return the optimal lot size to open a trade based on the defined risk percentage, entry price, and stop loss price. Or, based on the defined risk percentage and maximum loss in pips which means that we created an overloading function in this app. You will find the lotSizeCalcApp source code attached.
- Close All App: to be used or called to close all opened and pending orders. You will find the closeAllApp source code attached.

The world of functions is very interesting and it is very important to pay attention to it to be able to create useful pieces of code easily, smoothly, and effectively. I hope that you found this article useful for you to develop your coding skills and enhance your trading career by creating useful tools that can help you to trade well. If you want to read more articles about programming or about how to create trading systems based on the most popular technical indicators like moving average, RSI, MACD, Stochastic, Bollinger Bands, Parabolic Sar...etc. You can check my [publications](https://www.mql5.com/en/users/m.aboud/publications) and you find articles about that and I hope you find them useful as well.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12970.zip "Download all attachments in the single ZIP archive")

[newsAlertApp.mq5](https://www.mql5.com/en/articles/download/12970/newsalertapp.mq5 "Download newsAlertApp.mq5")(1.37 KB)

[lotSizeCalcApp.mq5](https://www.mql5.com/en/articles/download/12970/lotsizecalcapp.mq5 "Download lotSizeCalcApp.mq5")(1.45 KB)

[closeAllApp.mq5](https://www.mql5.com/en/articles/download/12970/closeallapp.mq5 "Download closeAllApp.mq5")(0.86 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/450916)**
(23)


![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
12 Oct 2023 at 19:09

**fxsaber [#](https://www.mql5.com/ru/forum/455535/page2#comment_49903057):**

They (Kim's functions) have long been ported from MQL4 to MQL5.

Excuse my ignorance and believe me, I am not being sarcastic, but what is ported? Are they rewritten to MQL5? If they are collected somewhere in one place, and if it is not difficult for you, please provide a link to Kim's functions ported to MQL5. I tried to search the site - Kim's ported functions, but found nothing.

Regards, Vladimir.

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
13 Oct 2023 at 15:41

**MrBrooklin [#](https://www.mql5.com/ru/forum/455535/page2#comment_49903145):**

provide a link to Kim's functions ported to MQL5.

[https://www.mql5.com/ru/forum/107476](https://www.mql5.com/ru/forum/107476 "https://www.mql5.com/ru/forum/107476")

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/ru/forum)

[Libraries: MT4Orders](https://www.mql5.com/ru/forum/93352/page30#comment_10240151)

[fxsaber](https://www.mql5.com/en/users/fxsaber), 2019.01.13 17:23 PM.

Kim's functions under MT4 are quite popular, so I downloaded all the sources from his site and wrote a simple "converter" for them under MT5.

```
#include <KimIVToMT5.mqh> // https://c.mql5.com/3/263/KimIVToMT5.mqh
```

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
13 Oct 2023 at 15:43

**fxsaber [#](https://www.mql5.com/ru/forum/455535/page2#comment_49921907):**

[https://www.mql5.com/ru/forum/107476](https://www.mql5.com/ru/forum/107476 "https://www.mql5.com/ru/forum/107476")

Thank you!

Regards, Vladimir.

![Hilario Miguel Ofarril Gonzalez](https://c.mql5.com/avatar/avatar_na2.png)

**[Hilario Miguel Ofarril Gonzalez](https://www.mql5.com/en/users/hilariomiguelofarrilgonzalez)**
\|
9 Dec 2023 at 16:43

Adding to the knowledge.to enceñar the point expesifico.donde we learn from others . strategies.very concrete .and continue in a precise way.es like being in a battle where the numbers are infinite .de ello es como digo yo sentirce satisfied .when we win the güera .el programa.de las defensas en el tradi.es tratar de cobertice en ganadores .sin afectar.nuestras inversiones.I dedicate myself to read every article of every master and I learn from it, putting into practice the desire to become a great master like them and to transmit the teachings so that the world of tradi is stronger, that is what defines us all...will and sacrifice....and I will sacrifice myself in order to make the world of tradi stronger.


![Javid Rezaei](https://c.mql5.com/avatar/2023/10/6529D5B2-3ACF.png)

**[Javid Rezaei](https://www.mql5.com/en/users/javid.rezaei)**
\|
15 Aug 2025 at 23:56

Many thanks for this useful article.


![Developing a Replay System — Market simulation (Part 03): Adjusting the settings (I)](https://c.mql5.com/2/52/replay-p3-avatar.png)[Developing a Replay System — Market simulation (Part 03): Adjusting the settings (I)](https://www.mql5.com/en/articles/10706)

Let's start by clarifying the current situation, because we didn't start in the best way. If we don't do it now, we'll be in trouble soon.

![Category Theory in MQL5 (Part 13): Calendar Events with Database Schemas](https://c.mql5.com/2/56/Category-Theory-p13-avatar.png)[Category Theory in MQL5 (Part 13): Calendar Events with Database Schemas](https://www.mql5.com/en/articles/12950)

This article, that follows Category Theory implementation of Orders in MQL5, considers how database schemas can be incorporated for classification in MQL5. We take an introductory look at how database schema concepts could be married with category theory when identifying trade relevant text(string) information. Calendar events are the focus.

![Cycle analysis using the Goertzel algorithm](https://c.mql5.com/2/57/cycle_analysis_goertzel_algorithm_avatar.png)[Cycle analysis using the Goertzel algorithm](https://www.mql5.com/en/articles/975)

In this article we present code utilities that implement the goertzel algorithm in Mql5 and explore two ways in which the technique can be used in the analysis of price quotes for possible strategy development.

![Developing a Replay System — Market simulation (Part 02): First experiments (II)](https://c.mql5.com/2/52/replay-p2-avatar.png)[Developing a Replay System — Market simulation (Part 02): First experiments (II)](https://www.mql5.com/en/articles/10551)

This time, let's try a different approach to achieve the 1 minute goal. However, this task is not as simple as one might think.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/12970&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069108608555221122)

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