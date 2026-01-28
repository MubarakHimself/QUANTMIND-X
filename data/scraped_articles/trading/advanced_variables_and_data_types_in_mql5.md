---
title: Advanced Variables and Data Types in MQL5
url: https://www.mql5.com/en/articles/14186
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:02:06.957638
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/14186&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068957773598752621)

MetaTrader 5 / Trading


### Introduction

MQL5 is the programming language of the MetaTrader 5 which is considered the most popular trading platform and it is very rich in tools and concepts that can be used to create any trading system from its simple to complex. Our objective as developers is to understand how to use these tools and concepts to achieve our development objective.

We have already mentioned the basics of simple variables and data types in MQL5 in our article [Learn why and how to design your algorithmic trading system](https://www.mql5.com/en/articles/10293) and learned more about how we can use MQL5 to code trading software, variables in terms of definition, their types and how we can use them. We also learned about simple data types such as integer, float, double, string, and bool, we also mentioned input variables and how they can be used to give the user the advantage of setting their preferences in the software.

In another article, [Learn how to deal with date and time in MQL5](https://www.mql5.com/en/articles/13466), we learned in detail about the Datetime data type, because we learned how to use this important data type that no software does not use this type of data, especially in trading software.

In this article, we will mention and dive deeper to learn more about variables and data types in MQL5 and how they can be useful when creating or building MQL5 trading software. We will learn more about some advanced concepts of variables and data types and we will cover them through the following topics:

- [Constants](https://www.mql5.com/en/articles/14186#constants): they are identifiers whose unchanged values.
- [Arrays](https://www.mql5.com/en/articles/14186#arrays): they are any type of variable with multiple values.
- [Enumerations](https://www.mql5.com/en/articles/14186#enumerations): they are integer lists of constants with integer values.
- [Structures](https://www.mql5.com/en/articles/14186#structures): they are a set of related variables with different types.
- [Typecasting](https://www.mql5.com/en/articles/14186#typecasting): it is the process of converting a type of value to another one.
- [Local variables](https://www.mql5.com/en/articles/14186#local): they are locally declared variables inside functions.
- [Global variables](https://www.mql5.com/en/articles/14186#global): they are globally declared variables outside functions.
- [Static variables](https://www.mql5.com/en/articles/14186#static): they are declared local variables that retain its values in the memory.
- [Predefined variables](https://www.mql5.com/en/articles/14186#predefined): they predefined variables by the originator of the programming language.

We will try to explain each topic in detail to understand how we can use it and see as many examples as possible to deepen our understanding and be able to use all the concepts mentioned effectively. Programming is the kind of science where practice is very important, so it is very important to try and apply what you learn to understand how you can use each concept as part of your overall code to create an effective trading system.

### Constants

In this part, we will dive deep into the concept of constants in programming and MQL5 to understand why we need to use this type of variable. The constant variable is also called read-only or named constant, it cannot be changed or we cannot assign a new value to it after the first initialization and if the code tries to do so it will produce an error. The concept of the constant is supported by most programming languages and MQL5 also supports this concept, we have two methods to define the constant in our software:

- **Global constants**
- **The specifier of const**

**Global constants:**

This global constant can be defined by using the #define preprocessor directive at the beginning of our software globally, then specifying the identifier, and after that we assign the constant value. The constant value can be any type of data such as the string type and the following is what we need to code in our software:

```
#define Identifier_Name constant_value
```

As we can see in the previous code, we have the #define preprocessor directive which specifies that we have a constant declaration, and we can see an example like the following to understand more about this method:

In the global scope, we first use the #define preprocessor directive to declare that we have a constant, the identifier is PLATFORM\_NAME, and the constant value of the identifier is the string data type of "MetaTrader 5"

```
#define PLATFORM_NAME "MetaTrader 5"
```

In the needed function, we can say that in the OnStart() for example we will print the the identifier

```
void OnStart()
  {
   Print("The trading platform name is: ", PLATFORM_NAME);
  }
```

When compiling and executing the software we can find the result as a message in the expert tab the same as the following screenshot

![constant_example](https://c.mql5.com/2/70/constant_example.png)

The constant value of MetaTrader 5 can not be changed as we mentioned.

**The specifier of const:**

According to this method, we can declare the constant by using the const specifier before the variable and its data type this specifier declares that this will be a constant and cannot be changed and the following is how we can do this

```
const int varName = assignedVal
```

The following is a simple example of declaring a constant by this method

```
const int val = 1;
```

If we needed to print the value of this variable

```
Print("constant val is: ", val);
```

We can find the result the same as the following:

![constant_example2](https://c.mql5.com/2/70/constant_example2.png)

As we mentioned, if we tried to update the variable (val) we will find that the following error will be produced because it is a constant and cannot be updated.

![constant_error](https://c.mql5.com/2/70/constant_error.png)

To illustrate the difference between a normal variable and a constant, we can use the following example, where we have two values, one is a constant and the other is a normal variable.

```
   const int val = 1;
   int val2 = 2;

   Print("constant val is: ", val);
   Print("val 2 is: ", val2);
```

The result will be the same as the following screenshot when executing this software

![diffExam](https://c.mql5.com/2/70/diffExam.png)

Now let us see if we try to update val2 with another value, which is 4, with this code

```
val2 = 4;
```

If you then print out these two values again, the result will be the same as in the following example:

![diffExam2](https://c.mql5.com/2/70/diffExam2.png)

As we can see, the value of val2 is updated with the value of 4 instead of the value of 2.

### Arrays

In this part, we will understand a basic concept in any programming language, which is the array. The array is a variable where we can store many values of any data type. We can think of the array as a list with indexes and corresponding values, when we need to access a specific value we can do so by indexing.

Array indexing starts at zero, so the maximum index is the same as the result of decreasing the size of the array by one value. If we have an array that has five values then we can find that its indexes are (0, 1, 2, 3, 4) and the maximum value as we can see is 4 which is the result of (5-1).

If we need to access the value of a particular index we refer to it with its index given in square brackets \[\] and this is what we mean by access by indexing as we mentioned earlier.

We also have a static array and a dynamic array and the difference between the two in terms of the size of the array is that the static array is an array with a fixed size and cannot be resized but the dynamic array has no size and it can be used if we need to resize the array instead of using the static one.

**Static array declaration**

If we need to declare a new static array, we can do so using the following example

```
int newArray[5];

newArray[0] = 1;
newArray[1] = 2;
newArray[2] = 3;
newArray[3] = 4;
newArray[4] = 5;

Print("newArray - Index 0 - Value: ", newArray[0]);
Print("newArray - Index 1 - Value: ", newArray[1]);
Print("newArray - Index 2 - Value: ", newArray[2]);
Print("newArray - Index 3 - Value: ", newArray[3]);
Print("newArray - Index 4 - Value: ", newArray[4]);
```

After running this software, we can see the same result as in the following screenshot

![staticArrayDeclaration](https://c.mql5.com/2/70/staticArrayDeclaration.png)

We can get the same result by declaring the array using the following abbreviated code for more efficiency, by assigning values to the array inside the brackets {} and separating them with a comma (,).

```
   int newArray[5] = {1, 2, 3, 4, 5};

   Print("newArray - Index 0 - Value: ", newArray[0]);
   Print("newArray - Index 1 - Value: ", newArray[1]);
   Print("newArray - Index 2 - Value: ", newArray[2]);
   Print("newArray - Index 3 - Value: ", newArray[3]);
   Print("newArray - Index 4 - Value: ", newArray[4]);
```

If we execute the previous code we will get the same result as printed messages as we mentioned.

**Dynamic array declaration**

The same as we mentioned before the dynamic array is the array without a fixed size and can be resized. What we need to know here is that this type of array we used when we code through the MQL5 to store data like price and indicator value because this data is dynamic so this type will be much suitable.

If we need to declare a new dynamic array we can do the same we can see in the following code

```
   double myDynArray[];
   ArrayResize(myDynArray,3);

   myDynArray[0] = 1.5;
   myDynArray[1] = 2.5;
   myDynArray[2] = 3.5;

   Print("Dynamic Array 0: ",myDynArray[0]);
   Print("Dynamic Array 1: ",myDynArray[1]);
   Print("Dynamic Array 2: ",myDynArray[2]);
```

As we can see in the previous code, we declared the array with an empty square bracket, then we used the ArrayResize function to resize the array, and we can see the result when executing the software, as shown in the following screenshot

![dynamicArrayDeclaration](https://c.mql5.com/2/70/dynamicArrayDeclaration.png)

As we can see in the previous screenshot, we have three printed messages for each index and the corresponding value. Now we have identified the one-dimensional arrays, but we can also use multi-dimensional arrays such as two, three, or four, but the most common that we often use is two or three.

**Multi-Dimensional Arrays**

Multi-dimensional arrays can be thought of as nested arrays, or arrays within arrays. As an example, we can say that we have two arrays and each one has 2 elements, we can see how we can code this type of array through the following:

Declaring two dimensional arrays with two elements

```
double newMultiArray[2][2];
```

Assigning values or elements to the first array with index 0

```
newMultiArray[0][0] = 1.5;
newMultiArray[0][1] = 2.5;
```

Assigning values or elements of the second array with the index 1

```
newMultiArray[1][0] = 3.5;
newMultiArray[1][1] = 4.5;
```

Printing the values of two arrays and their elements or values

```
   Print("Array1 - Index 0 - Value: ", newMultiArray[0][0]);
   Print("Array1 - Index 1 - Value: ", newMultiArray[0][1]);
   Print("Array2 - Index 0 - Value: ", newMultiArray[1][0]);
   Print("Array2 - Index 1 - Value: ", newMultiArray[1][1]);
```

After running this software, we can see the same output as the following screenshot

![multiDymArrays](https://c.mql5.com/2/70/multiDymArrays.png)

As we can see in the previous screenshot, we have Array 1 with two values 1.5 and 2.5, we also have Array 2 with two values 3.5 and 4.5.

It is also good to mention here that when we declare multi-dimensional arrays, we can only leave the first dimension empty, the same as in the following example

```
double newMultiArray[][2];
```

Then we can pass it by using a variable that is the same as the following one

```
int array1 = 2;
ArrayResize(newMultiArray, array1);
```

Once we have compiled and run this software, we will find the same result as we have found before.

### Enumerations

In this part, we will identify enumerations. We can think of enumerations as sets of data or lists of constants or items that can be used to describe related concepts and we have built-in enumerations and custom enumerations. Built-in enumerations are predefined in MQL5 and we can call and use them in our program, but custom enumerations are custom enumerations according to our needs.

Built-in enumeration, we can find them listed in the documentation or reference of MQL5 like for example [ENUM\_DAY\_OF\_WEEK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_day_of_week). But if we take a look at the custom enums we can find that we can set what we need to use them later in our created software.

The following is for how we can use these enums in our software, first, we use the enum keyword to define an enumeration, the name of the enumeration type, and a list of related data type values separated by commas.

```
enum name of enumerable type
  {
   value 1,
   value 2,
   value 3
  };
```

Let's see an example to create our named workingDays enum this workingDays will be a type that I will declare variables to it later

```
   enum workingDays
     {
      Monday,
      Tuesday,
      Wednesday,
      Thursday,
      Friday,
     };
```

Now we need to declare a new related variable of type workingDays, assign the value of today among those defined in the list, print this variable

```
     workingDays toDay;
     toDay = Wednesday;

     Print(toDay);
```

We can find the message of (2) for Wednesday as a working day from the list starting from 0 for Monday to 4 for Friday as shown in the following screenshot.

![enums](https://c.mql5.com/2/70/enums__1.png)

We can also determine another start number to be assigned instead of (0) by assigning the number we need to start with to the value, for example, by specifying that Monday = 1.

### Structures

We may find that we need to declare a number of different data types for related variables, in which case the structure will do this effectively because it can be used to do the same as we will find in this part. The members of this structure can be of any data type, unlike an enumeration because its members can only be of the same type. They are the same as enumerations there are predefined structures in the MQL5 like MqlTick for example and we can create our own.

Let's say we need to create a structure for tradeInfo, we can see that we have the symbol, price, stop loss, take profit, and time of trade as members and they have different data types because the symbol is a string and price, stop loss, take profit are double values. So in this case we need to use structures to do the same task as below:

We will use the struct keyword to declare our own structure.

```
   struct tradeInfo
     {
      string         symbol;
      double         price;
      double         stopLoss;
      double         takeProfit;
     };
```

We can then declare a new trade object of type tradeInfo and assign values to the members of this structure by accessing them using (.) after the object name, as follows

```
   tradeInfo trade;

   trade.symbol = "EURUSD";
   trade.price = 1.07550;
   trade.stopLoss = 1.07500;
   trade.takeProfit = 1.07700;
```

We can print members with their values like the following to see our work like the following

```
   Print("Symbol Trade Info: ", trade.symbol);
   Print("Price Trade Info: ", trade.price);
   Print("SL Trade Info: ", trade.stopLoss);
   Print("TP Trade Info: ", trade.takeProfit);
```

We can find the output of the printed message in the same way as in the following screenshot

![struct](https://c.mql5.com/2/70/struct__1.png)

As we can see in the previous screenshot, our assigned values are the same as we need them to be, so we can play around with them to achieve our objectives in the same context.

### Typecasting

In this section, we will understand the concept of typecasting and what it means. When we need to convert the value of a variable from one data type to another, we can call this process typecasting, but this process can produce unexpected results.

How does this happen? When we cast or copy a value with numbers from one variable to another, we can face a loss of data when this process happens between types.

The following are examples where no data is lost or lost some of data:

We copy an integer value to a long value, this case will not produce this type of data loss because the casting is done from smaller to larger variables, when we copy an integer to a double it is also copying the whole number and the value after the decimal point is zero so there is no data to lose. If we copy a double value to an integer variable, which is the opposite case, the value after the decimal will be lost or truncated.

Let's look at examples through the code to make it easier to understand. First, we will see the case without any problems:

```
   int price = 3 - 1;
   double newPrice = price + 1;

   Print ("The price is ", newPrice);
```

As we can see the type of the variable (price) is int, then we cast it to a double variable and then print it. It is assumed that there will be no problem or loss of data because we are casting from a smaller to a larger type. The following is the result of the printed message

![typeCasting1](https://c.mql5.com/2/70/typeCasting1.png)

But when we do the opposite we will see something different the same as the following

```
   double price = 3.50 - 1;
   int newPrice = price + 1;

   Print ("The price is ", newPrice);
```

As we can see the type is double and the value is 3.50 for the price variable and when we cast to a new int variable (newPrice) then we print it. So we will lose the data after the decimal point which is the value of (.50) and the result will be (3) which is the same as the following printed message

![typeCasting2](https://c.mql5.com/2/70/typeCasting2.png)

It is very important to mention here that the compiler will warn us if we cast a value of a larger type into a variable of a smaller type, so that we can decide whether to ignore this message or not, depending on our goal or the sensitivity of the lost data. Below is an example of this warning:

![typeCasting2](https://c.mql5.com/2/70/typeCasting2-warning.png)

We can also choose to ignore the warning if it is not harmful, or we can round it off by adding (int) in front of the double variable to display the warning from the compiler.

### Local variables

In this part, we will take a look at a new concept which is the local variable. Let's consider that we have scopes in our software global, and local scopes. The global scope is that we can access wherever the place in our software and we will take a look at the concept of global in the next topic in terms of variables but the local one can only be accessible in the same scope that it is declared in.

For further clarification, if we have a function and inside that function, we have variables that are declared only at the level of the function, those variables are considered local variables. So they can be accessible when the function is running then they will not be accessible after exiting the function.

Let's take an example to make this clearer:

```
void myFunc()
{
   int var = 5;
   Print(var);
}
```

As we can see in the previous code we have the function of myFunc and this function can be called anywhere. When we call this function we can find a local variable called var and this variable can be running only inside the function but after exiting from the function it can not be accessible.

So when we call this function we can find the output is 5 which is the same as the following

![localVar](https://c.mql5.com/2/70/localVar__1.png)

If we tried to access to the var variable which is the local one from outside the function the compiler will produce an error of (undeclared identifier) the same as the following:

![localVarError](https://c.mql5.com/2/70/localVarError.png)

The same applies if we have nested levels or scopes inside the function, each declared variable will only be accessible in its scope.

Let's take an example about this case as well:

```
void myFunc()
  {
   int var = 5;
   if(var == 5)
     {
      int newVar = 6 ;
      Print(newVar);
     }

  }
```

As we can see in the previous example we have nested if to compare var value to 5 and if it is true we will declare newVar with a value of 6 and print it. The output will be 6 because the condition is true as follows

![localVar2](https://c.mql5.com/2/70/localVar2.png)

This newVar is a new local variable inside the scope of the if operator and cannot be accessible outside it the same as any other local scope and we try the compiler will produce the error of undeclared variable. It is important to know that any declaration for a variable with the same name as the local variable, will override the previous one and it will be typed in a higher scope.

So the concept is very simple, any local variable will be accessible in its level or scope within the function. But what do we do if we need to access a variable anywhere in our software here the global variable comes into play the same we will see in the next global variable topic.

### Global variables

In the previous topic on local variables, we mentioned global variables, which are variables that are declared globally, which means that we can access them from anywhere in our software, unlike local variables. So when we need to declare global variables, we will be careful to declare them globally or in the global scope of the software, outside of any function, so that we can use or call them anywhere in the software. These types of variables can be used when we need to declare a variable that will be used many times by many functions.

Global variables are defined at the beginning or top of the software, after the input variables. Unlike local variables, if we try to declare global variables inside the block, it will produce a compilation error of "Declaration of variable hides global declaration", the same as the following:

![globalVar](https://c.mql5.com/2/70/globalVar.png)

We can update global variables at any point in the software, the same as in the following example, which shows how we can declare and update global variables:

```
int stopLoss = 100;

void OnStart()
  {
   Print("1st Value: ", stopLoss);
   addToSL();
  }

void addToSL()
  {
   stopLoss = stopLoss + 50;
   Print("Updated Value: ", stopLoss);
  }
```

As we can see from the previous block of code, we declared the stopLoss variable globally at the top of the software, printed the assigned value, called the addToSL() function which added 50 to the first assigned value, and then printed the updated value. After compiling and executing this software we can find printed messages the same as the following:

![globalVar2](https://c.mql5.com/2/70/globalVar2.png)

As we can see we have the first initial value of the variable is 100 and when updating it by adding 50 in the addToSL() function became 150.

### Static variables

In this part, we will identify another type of variable which is the static variable. Static variables are local variables but they retain their values in the memory even if software has exited from the scope of them and it can be declared in the block of functions or local variables.

We can do that by using the keyword of static before the variable name and assigning its value and we can see the following example for more clarification

```
void staticFunc()
{
   static int statVar = 5;
   statVar ++;
   Print(statVar);
}
```

Then we will call the function of staticFunc()

```
staticFunc();
```

Here, the variable will retain its value (5) in the memory and every time we call the function the output will be 6 which is (5+1). The following is a screenshot of the output:

![staticVar](https://c.mql5.com/2/70/staticVar.png)

As we can see the printed message of the 6 value in the expert tab.

### Predefined variables

In the programming field, we may find ourselves needing to write many lines again and again to do something commonly used. So, in a lot of programming languages, there are variables or even functions that are predefined which means that they are coded and we can use them easily without the need to rewrite all code again. Here, the role of predefined variables comes, all that we need to do is to remember or know what is the keyword to do something and use it.

MQL5 has many predefined variables and we can access their values in the same way as the examples below:

- \_Symbol: it refers to the current symbol on the chart.
- \_Point: it refers to the point value of the current symbol, it is 0.00001 in five digits of the current symbol and 0.001 in three digits of the current symbol.
- \_Period: it refers to the current period or timeframe of the symbol.
- \_Digits: it refers to the number of digits after the decimal point of the current symbol, if we have a five-digit symbol this means that the number after the decimal point is 5, and a three-digit symbol means that the symbol has three numbers after the decimal point.
- \_LastError: it refers to the value of the last error.
- \_RandomSeed: it refers to the current status of the generator of pseudo-random integers.
- \_AppliedTo: allows you to find out the type of data used to calculate the indicator.

These and other predefined variables can be found in the documentation under the [Predefined Variables](https://www.mql5.com/en/docs/predefined) section.

### Conclusion

This article explores the intricacies of MQL5, focusing on essential programming concepts such as data types, variables, and other key elements crucial for developing advanced software. Mastery of MQL5 requires understanding both basic and complex aspects, with practice being fundamental to becoming proficient. This guide aims to aid your learning journey in MQL5 for trading platform development. For more insights on programming or creating trading systems with popular technical indicators like Moving Averages, RSI, and others, check out my publications for detailed articles and strategy development tips.

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/463995)**
(3)


![Dominik Egert](https://c.mql5.com/avatar/2024/2/65db1363-e44b.jpg)

**[Dominik Egert](https://www.mql5.com/en/users/dominik_egert)**
\|
14 Mar 2024 at 19:28

**MetaQuotes:**

Check out the new article: [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186).

Author: [Mohamed Abdelmaaboud](https://www.mql5.com/en/users/M.Aboud "M.Aboud")

I disagree to some extend that a preprocessor directive can be mentioned within the context of const variables.

The keyword const has a different effect than a #define "constant".

const can be very useful in function signatures, as it sets the parameters value passes in to read only mode. const is a keyword especially designed for the coder. There is no representation in the executable binary produced by the compiler.

In contrast to a const variable, a #define statement is a so called r-value, which is actually in a read only memory region later on in the executable binary, and therefore actually cannot be changed what so ever.

Also, #define is a preprocessor stage directive, and actually just replaces all occurrences in the source file. This is done before the compiler even sees the code to be compiled.

In addition, the #define directive has no such "location" as global, it may appear anywhere in the source file. At the stage of the preprocessor, the concept of code-blocks "{...}" is not available, or evaluated.

Visibility of variables is defined by code blocks. You can at any level define code blocks as you like, except at global level. At global level you need some "entity" to which this block belongs. Function, class, struct or namespace.

Inside such a "named" block, you can cascade as many blocks as you like, while the visibility is always only inside the same level or included/child blocks.

static variables actually reside in the global space memory allocated for a program, the visibility of this variable is governed by the code block in which it was declared/defined. This can be also the global space. Actually, a variable declared/defined on global space has an implicit "static" keyword. Mentioning it explicit will not change any of the behaviour of that variable.

I am missing the keyword "extern" in your article, as well as "input". I think these should be part.

Anyways, I think it is a good article, especially for beginners, as the concept of variables is sometimes hard to grasp at the beginning, and lots of errors can be avoided, if properly implemented.

Maybe you continue with a follow-up, would be nice if you could include about defining, declaring and initializing variables, and memory. Maybe also point out the pitfalls if not done correctly.

And for some more insight, maybe some extended details about stack, order of variables, MSB vs LSB, memory addressing.... Ok, maybe this is to far of a fetch.

![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
27 Jul 2024 at 15:09

**Dominik Egert [#](https://www.mql5.com/ru/forum/470401#comment_54085098):**

I somewhat disagree that a preprocessor directive can be mentioned in the context of const variables.

The const keyword has a different effect than #define "constant".

const can be very useful in function signatures because it puts the value of parameters into read-only mode. const is a keyword specially created for the coder. It has no representation in the executable binary created by the compiler.

Unlike the const variable, the #define operator represents a so-called r-value, which is actually located in a read-only memory location in a further executable binary, and therefore cannot actually be changed in any way.

Also, #define is a preprocessor stage directive, and it actually just replaces all occurrences in the source file. This is done before the compiler even sees the compiled code.

Also, the #define directive does not have the same "location" as a global directive, it can appear anywhere in the source file. At the preprocessor stage, the concept of "{...}" code blocks is not available or evaluated.

The visibility of variables is determined by codeblocks. You can at any level define codeblocks any way you want, except at the global level. At the global level, you need some "entity" that the block belongs to. A function, class, structure, or namespace.

Within such a "named" block, you can cascade as many blocks as you want, and visibility will always be only within blocks of the same level or included/subsidiary blocks.

static variables actually reside in the global memory space allocated for the programme, the visibility of this variable is determined by the code block in which it was declared/defined. This can be global space as well. In fact, a variable declared/defined in global space has the implicit keyword "static". Its explicit mention will not change the behaviour of this variable in any way.

In your article, I am missing the keyword "extern" as well as "input". I think they should be part of the article.

Anyway, I think this is a good article, especially for beginners, as the concept of variables is sometimes hard to understand at the beginning and many mistakes can be avoided if implemented correctly.

Maybe you will continue the article, it would be nice if you talk about the definition, declaration and initialisation of variables as well as memory. Perhaps you will also point out the pitfalls if done incorrectly.

And for more understanding, maybe some advanced details about the stack, variable ordering, MSB vs LSB, memory addressing..... Okay, maybe that's too far.

Thanks for sharing the information. I will try to write about what you mentioned as much as I can.

![Vladislav Boyko](https://c.mql5.com/avatar/2025/12/692e1587-6181.png)

**[Vladislav Boyko](https://www.mql5.com/en/users/boyvlad)**
\|
27 Jul 2024 at 22:05

**Dominik Egert [#](https://www.mql5.com/en/forum/463995#comment_52722451):**

The keyword const has a different effect than a #define "constant".

By the way, I like the way constants are implemented in C# (the compiler replaces them with literal values).

[https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/classes-and-structs/constants](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/classes-and-structs/constants "https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/classes-and-structs/constants")

In fact, when the compiler encounters a constant identifier in C# source code, it substitutes the literal value directly into the intermediate language (IL) code that it produces. Because there is no variable address associated with a constant at run time, const fields cannot be passed by reference and cannot appear as an l-value in an expression.

Judging by the fact that in MQL I can pass a constant by reference, in MQL constants remain variables after compilation.


![Neural networks made easy (Part 63): Unsupervised Pretraining for Decision Transformer (PDT)](https://c.mql5.com/2/60/Neural_networks_are_easy_wPart_636_Logo.png)[Neural networks made easy (Part 63): Unsupervised Pretraining for Decision Transformer (PDT)](https://www.mql5.com/en/articles/13712)

We continue to discuss the family of Decision Transformer methods. From previous article, we have already noticed that training the transformer underlying the architecture of these methods is a rather complex task and requires a large labeled dataset for training. In this article we will look at an algorithm for using unlabeled trajectories for preliminary model training.

![Introduction to MQL5 (Part 5): A Beginner's Guide to Array Functions in MQL5](https://c.mql5.com/2/73/Introduction_to_MQL5_Part_5___LOGO.png)[Introduction to MQL5 (Part 5): A Beginner's Guide to Array Functions in MQL5](https://www.mql5.com/en/articles/14306)

Explore the world of MQL5 arrays in Part 5, designed for absolute beginners. Simplifying complex coding concepts, this article focuses on clarity and inclusivity. Join our community of learners, where questions are embraced, and knowledge is shared!

![Developing an MQTT client for Metatrader 5: a TDD approach — Part 6](https://c.mql5.com/2/73/Developing_an_MQTT_client_for_Metatrader_5_PArt_6____LOGO.png)[Developing an MQTT client for Metatrader 5: a TDD approach — Part 6](https://www.mql5.com/en/articles/14391)

This article is the sixth part of a series describing our development steps of a native MQL5 client for the MQTT 5.0 protocol. In this part we comment on the main changes in our first refactoring, how we arrived at a viable blueprint for our packet-building classes, how we are building PUBLISH and PUBACK packets, and the semantics behind the PUBACK Reason Codes.

![Quantization in machine learning (Part 2): Data preprocessing, table selection, training CatBoost models](https://c.mql5.com/2/59/Quantization_in_Machine_Learning_Logo_2___Logo.png)[Quantization in machine learning (Part 2): Data preprocessing, table selection, training CatBoost models](https://www.mql5.com/en/articles/13648)

The article considers the practical application of quantization in the construction of tree models. The methods for selecting quantum tables and data preprocessing are considered. No complex mathematical equations are used.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/14186&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068957773598752621)

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