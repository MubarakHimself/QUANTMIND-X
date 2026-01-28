---
title: Understanding MQL5 Object-Oriented Programming (OOP)
url: https://www.mql5.com/en/articles/12813
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:43:39.717196
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/12813&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051638829078664135)

MetaTrader 5 / Trading


### Introduction

In this article, we will share one of the most important topics in programming which lead to coding smoothly and easily, and help us to apply the concept of (DRY) which is the shortcut of (Do Not Repeat Yourself) as developers or programmers. In addition to raising the security of any created software, and other of features. We will talk about Object-Oriented Programming (OOP) and how we can use this concept and code it in MQL5 (MetaQuotes Language) by understanding the basics of OOP first, then understanding how to use it in MQL5 by seeing some applications.

So, we will cover this interesting and important topic through the following three points:

- [What is the OOP](https://www.mql5.com/en/articles/12813#oop)
- [OOP in MQL5](https://www.mql5.com/en/articles/12813#mql5)
- [OOP applications](https://www.mql5.com/en/articles/12813#application)
- [Conclusion](https://www.mql5.com/en/articles/12813#conclusion)

The main objective of this article is to understand basics of the Object-Oriented Programming (OOP) in general and how it can be useful in software creation. From there, we will learn how we can apply this approach in MQL5 to use what may help us to create more effective and secure software by applying this incredible approach.

It will be very useful for anyone to learn more about this OOP approach there are very important and informative resources about this approach in general especially since it is applicable in other programming languages like C++, Java, Python, and others. In the MQL5 language, there is a very important resource not only in the topic of the article but every single topic in the MQL5 language which is the MQL5 documentation you can check the topic of the OOP through the link of [MQL5 OOP Documentation](https://www.mql5.com/en/docs/basis/oop).

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### What is the OOP

In this topic we will learn the basics of OOP in general so, let us start with this interesting topic by defining the OOP. OOP is short for Object-Oriented Programming and it is a paradigm of computer programming, It helps to create and develop reusable software smoothly without duplicated work and code and this helps us to apply the DRY (Do not Repeat Yourself) concept.

As we will see the OOP helps us to get closer to the nature of the world because everything around us is created by objects and each one has its nature and behavior and this is what we can do with our data in the software. If we want to dive more into the OOP we need to know that we will deal with objects and classes, the object means an instance of a Class and the Class is a template for the object. Inside the template which is the class, we define the behavior of this object in detail.

**Basic concepts of OOP:**

When we apply the OOP in software, the following principles will be applied.

1. Encapsulation.
2. Abstraction.
3. Inheritance.
4. Polymorphism.

![oop](https://c.mql5.com/2/55/OOP.png)

**1- Encapsulation:**

Encapsulation is the method that enables to link of functions and data in one class, the data and functions in the class can be private which will be accessible only within the class or it can be public which will be accessible outside the class. The Encapsulation concept helps to hide the complexity of the class implementation and gives the developer full control of his data, helping to keep track of all values which rely on others without conflict.

So, we can say that Encapsulation helps to keep our system up and running and avoid a lot of possible errors in addition to giving a high level of control for the developer and helping in testing and processing classes data smoother and easier without affecting or changing the whole code of the software. According to what mentioned it helps also to solve errors and prevent to be coding complicated.

The following picture represents the Encapsulation concept:

![Encapsulation](https://c.mql5.com/2/55/Encapsulation__1.png)

**2- Abstraction:**

Abstraction is the method of hiding unnecessary details and presenting only the essential details. it is broader than the Encapsulation concept but it helps to achieve the same objective of it protecting data and implementing functions without knowing all things about the process of implementation of all classes but only knowing what I need to do to let the implementation be done.

To do that we need to include two important methods the Interface and the Implementation. The interface is the method that let classes interact and deal with each other but the Implementation is the method that has all details of the code or the logic of classes details. So, Abstraction helps to raise the security of the software and helps also to not repeat our coding process or recode from scratch but only develop and code more Apps based on the well create one.

**3- Inheritance:**

From its name the Inheritance concept means that we derived a new class from an old one and the new one inherits features of the old one in this case the old one is called a parent or superclass and the newly derived one is called the child class. This concept helps to apply the concept of DRY (Do not Repeat Yourself) but provides the main idea of reusability.

![Inheritance](https://c.mql5.com/2/55/Inheritance.png)

**4- Polymorphism:**

From the name of this concept if we divide the word two segments we will find (Poly) means many or multiple and (Morphism) means the form which means that the Polymorphism concept means that it helps an entity to be able to behave in many or multiple forms for example if we have a Sum method that can be able to behave differently like one that can be able to get the summation of (a) and (b) variables and another one that can be able to get the summation also but for different parameters like (a), (b), and (c).

Simply, Polymorphism means one interface and many or multiple methods.

We can summarize what we mentioned that the OOP is a computer programming model that focuses on organizing the design of the software through objects that can have unique behavior and characteristics and this is a helpful method in large and complex software especially if this software is updated and manipulated a lot.

**OOP attributes and features:**

- Software that applies the OOP approach includes classes and functions.
- The high level of security as data is hidden by applying the encapsulation and abstraction principles.
- It helps to work on complex projects easily as code can be divided into small blocks of code and this can decrease the complexity of the project.
- Update and development processes are easier.
- Reusability of code by applying the inheritance principle.
- Ability to create many instances of the same class without conflict.

We can find many programming languages that have the ability to apply the OOP approach and the most popular are for example C++, C#, Python, Java, JavaScript, PHP, and others. From these languages that we can apply the OOP approach is the MQL5 language and this is what we learn about through the following topic.

### OOP in MQL5

In this topic, we will learn about the OOP in MQL5 and how we can use it. The same as we mentioned as basics of the OOP that we create a class to be like a blueprint for an object and the class itself is a collection or contains variables and methods which the same as a function but it is called a method in the class to perform a specific set of tasks. It is good also to mention here that variables and methods or functions inside the class are called class members. What we need to say here is that it is a good and appreciated effort that applying the OOP concept is available in MQL5 as it will save a lot of effort and works as we mentioned in addition to the quality and security.

If we need to apply the OOP concept in MQL5 we need to understand how we can use the following in MQL5:

- Classes
- Access Modifier
- Constructors and Destructors
- Derived (Child) Classes
- Virtual Functions
- Objects

**Classes:**

In the MQL5, if we need to create a class to be a blueprint of an object we need to declare this class on the global scope the same as functions. We can create this class by using the class keyword followed by the desired unique then between or inside the two curly brackets we can place our variables and methods which are members of the class then after the second curly bracket we place a semicolon to terminate the class declaration. By the way, we can use this class declaration in the program or in the include file.

The following is an example of this class declaration:

```
class Cobject
{
   int var1;       // variable1
   double var2;    // variable1
   void method1(); // Method or function1
};
```

As we can see in the previous example we have three members of the class and they are two variables and one method or function.

**Access Modifiers:**

By these Access Modifiers, we can determine what variables and functions we can use outside the class and we have three access keywords which are public, private, and protected.

- Public: represents members that can be available for use outside the class.
- Private: represents members that cannot be available for use outside the class but are only available for use inside the class by functions. The child class of this class will not inherit these private members.
- Protected: represents members that will be inherited by child classes but they are private by nature.

If we need to see an example of that we can see the following one:

```
class Cobject
{
   private:
   int var1;       // variable1
   protected:
   double var2;    // variable1
   public:
   void method1(); // Method or function1
};
```

As we can see in the previous example we have three members in the class with two variables one is private and the other one is protected and the third member is public.

**Constructors and Destructors:**

If we need to initialize variables in the class we use the constructor. It will be created by default by the compiler if we did not but this default constructor will not be visible. It also must be public as per the accessibility. On the other hand, The Destructor is an automatically called function when a class object is destroyed. We can call the destructor the same as the class name with a tilde (~). Whether the destructor is exist or not, the string, dynamic array, and object are requiring deinitialization so they will be de-initialized anyway.

The following is an example of a Constructor:

```
class CPrices
  {
private:
   double               open;         // Open price
   double               high;         // High price
   double               low;          // Low price
   double               close;        // Close price
public:
   //--- Default constructor
                     CPrices(void);
   //--- Parametric constructor
                     CPrices(double o,double h,double l, double c);
  };
```

**Derived (Child) Classes:**

As we learned before that the inheritance concept is one of the most valuable and useful features of the OOP because we can create a child class from a super or parent class and this child class inherits all members of the parent class except private members. After that, we can add new variables and functions for that child class.

If we need to see an example we can see that through the following one, if we have a parent class for prices we can create a child one for daily prices the same as the following:

```
class CDailyPrices : public CPrices
{
public:
   double               open;          // Open price
   double               high;          // High price
   double               low;           // Low price
   double               close;         // Close price
};
```

As we can see the parent class name is CPrices and CDailyPrices is the child or derived one. Now, we find all of the public and protected members of CPrices are part of the CDailyPrices class and they are still public.

**Virtual Functions:**

If we want to update the way a method or function operates in a child class we can do that by using the (virtual) function in the parent class then we define the function in the child class. For example, if we have two different versions of a function based on the class. For the parent class, we define the function using the virtual keyword

```
class CVar
  {
public:
   virtual
void varCal();
  };
```

Then we will update the same function in the child class

```
class CVar1 : public CVar
{
public:
 int varCal(int x, int y);
};
```

**Objects:**

Objects are the unique identifier the same as we do when creating a variable, we will use the class name as a type before the object identifier. We can create many objects belonging to our classes as we need for the project all we need is to use a unique identifier for everyone. After declaring the object we can access any public member by using the (.) which is the dot.

Let's see an example to understand that clearly if we create a class that has an integer variable of the number of trades (num\_trades)

```
class CSystrades
{
public:
int num_trades;
};
```

Then we need to create an object belonging to this class called system1, we will do that by doing the same as the following

```
CSystrades system1;
```

Then we can define this object by the (3) value the same as the following

```
system1.num_trades=3;
```

Now, we understood how we can apply the OOP approach in MQL5 by learning some of the most important ideas around that in this topic.

### OOP applications

In this interesting part we will present some simple applications of applying the OOP approach in our software to well understand how we can use it and how much it can be useful.

**priceClass application:**

In this simple application, we need to check prices of multi-time frames and in this example, we will present three-time frames (daily, weekly, and monthly) here we need to see all prices also (open, high, low, close) we need to see them in one place let say in Experts tab. After that, we can develop more in this simple example to implement more advanced software.

First, we need to declare the class through the following step:

- We need to declare a class for prices in the global scope and include all common members as public by using the class keyword.
- Use the public keyword.
- Create five variables (timeframe, open, high, low, and close).
- Create a void function to print all price data.

```
class CPrices
  {
public:
   string            timeFrame;
   double            open;
   double            high;
   double            low;
   double            close;
   void              pricesPrint()
     {
      Print(timeFrame," Prices = Open: ",open," - ","High: ",high,"-","Low: ",low,"-","Close: ",close);
     }
  };
```

Create objects from the class for daily, weekly, and monthly prices

```
CPrices CDailyPrices;
CPrices CWeeklyPrices;
CPrices CMonthlyPrices;
```

Inside the OnInit function, we will define the following for the three-time frames:

- Defining the string time frame.
- Defining the open price using the iOpen function.
- Defining the high price using the iHigh function.
- Defining the low price using the iLow function.
- Defining the close price using the iClose function.
- Calling the printing function or method.

```
int OnInit()
  {
//--- Daily time frame
   CDailyPrices.timeFrame="Daily";
   CDailyPrices.open=(iOpen(Symbol(),PERIOD_D1,1));
   CDailyPrices.high=(iHigh(Symbol(),PERIOD_D1,1));
   CDailyPrices.low=(iLow(Symbol(),PERIOD_D1,1));
   CDailyPrices.close=(iClose(Symbol(),PERIOD_D1,1));
   CDailyPrices.pricesPrint();

//--- Weekly time frame
   CWeeklyPrices.timeFrame="Weekly";
   CWeeklyPrices.open=(iOpen(Symbol(),PERIOD_W1,1));
   CWeeklyPrices.high=(iHigh(Symbol(),PERIOD_W1,1));
   CWeeklyPrices.low=(iLow(Symbol(),PERIOD_W1,1));
   CWeeklyPrices.close=(iClose(Symbol(),PERIOD_W1,1));
   CWeeklyPrices.pricesPrint();

//--- Monthly time frame
   CMonthlyPrices.timeFrame="Monthly";
   CMonthlyPrices.open=(iOpen(Symbol(),PERIOD_MN1,1));
   CMonthlyPrices.high=(iHigh(Symbol(),PERIOD_MN1,1));
   CMonthlyPrices.low=(iLow(Symbol(),PERIOD_MN1,1));
   CMonthlyPrices.close=(iClose(Symbol(),PERIOD_MN1,1));
   CMonthlyPrices.pricesPrint();
   return(INIT_SUCCEEDED);
  }
```

After that, we can find the prices printed after executing this expert in the Experts tab from the toolbox the same as the following:

![ Prices printing](https://c.mql5.com/2/55/Prices_printing.png)

As we can see in the previous picture we have three printed lines:

- The first line prints the daily prices of open, high, low, and close.
- The second prints the same prices but belongs to the weekly data.
- The third line prints the same also but belongs to the monthly data.

**indicatorClass application:**

We need to create software that can be able to print values of four moving average types (Simple, Exponential, Smoothed, and Linear-weighted averages) using the OOP approach the following are simple steps to create this type of software:

Declare the indicator CiMA class by using the class keyword and create public members of this class they are four common variables of MAType to define the type of the moving average, MAArray to define the array of the moving average, MAHandle to define the handle of every type, MAValue to define the value of every moving average. Create a void method or function of valuePrint and the body of the function to print the value of every moving average type.

```
class CiMA
  {
public:
   string            MAType;
   double            MAArray[];
   int               MAHandle;
   double            MAValue;
   void              valuePrint()
     {
      Print(MAType," Current Value: ",MAValue);
     };
  };
```

Create the following objects of every moving average from the class:

- The name of average
- The handle of average
- The array of average

```
//--- SMA
CiMA CSma;
CiMA CSmaHandle;
CiMA CSmaArray;

//--- EMA
CiMA CEma;
CiMA CEmaHandle;
CiMA CEmaArray;

//--- SMMA
CiMA CSmma;
CiMA CSmmaHandle;
CiMA CSmmaArray;

//--- LWMA
CiMA CLwma;
CiMA CLwmaHandle;
CiMA CLwmaArray;
```

Inside the OnInit function we will do the following steps for every moving average type:

- Define the name of the average.
- Define the handle of the average.
- Setting the AS\_SERIES flag to the array by using the ArraySetAsSeries.
- Getting the data of the buffer pf the average indicator by using the CopyBuffer function.
- Define the value of the average and normalize it by using the NormalizeDouble function.
- Calling the Print created method or function

```
int OnInit()
  {
   //--- SMA
   CSma.MAType="Simple MA";
   CSmaHandle.MAHandle=iMA(_Symbol,PERIOD_CURRENT,10,0,MODE_SMA,PRICE_CLOSE);
   ArraySetAsSeries(CSmaArray.MAArray,true);
   CopyBuffer(CSmaHandle.MAHandle,0,0,3,CSmaArray.MAArray);
   CSma.MAValue=NormalizeDouble(CSmaArray.MAArray[1],_Digits);
   CSma.valuePrint();

   //--- EMA
   CEma.MAType="Exponential MA";
   CEmaHandle.MAHandle=iMA(_Symbol,PERIOD_CURRENT,10,0,MODE_EMA,PRICE_CLOSE);
   ArraySetAsSeries(CEmaArray.MAArray,true);
   CopyBuffer(CEmaHandle.MAHandle,0,0,3,CEmaArray.MAArray);
   CEma.MAValue=NormalizeDouble(CEmaArray.MAArray[1],_Digits);
   CEma.valuePrint();

   //--- SMMA
   CSmma.MAType="Smoothed MA";
   CSmmaHandle.MAHandle=iMA(_Symbol,PERIOD_CURRENT,10,0,MODE_SMMA,PRICE_CLOSE);
   ArraySetAsSeries(CSmmaArray.MAArray,true);
   CopyBuffer(CSmmaHandle.MAHandle,0,0,3,CSmmaArray.MAArray);
   CSmma.MAValue=NormalizeDouble(CSmmaArray.MAArray[1],_Digits);
   CSmma.valuePrint();

   //--- LWMA
   CLwma.MAType="Linear-weighted MA";
   CLwmaHandle.MAHandle=iMA(_Symbol,PERIOD_CURRENT,10,0,MODE_LWMA,PRICE_CLOSE);
   ArraySetAsSeries(CLwmaArray.MAArray,true);
   CopyBuffer(CLwmaHandle.MAHandle,0,0,3,CLwmaArray.MAArray);
   CLwma.MAValue=NormalizeDouble(CLwmaArray.MAArray[1],_Digits);
   CLwma.valuePrint();
   return(INIT_SUCCEEDED);
  }
```

After compiling and executing this code we can find four lines for every type of moving average each line printing the value of the average the same as the following:

![ Indicator printing](https://c.mql5.com/2/55/Indicator_printing.png)

As I mentioned before that we can develop these applications to implement more complicated and advanced tasks but here the objective is to learn and understand basics of the Object-Oriented Programming (OOP) and how we can apply this approach in MQL5 because there are many features that we can get when applying this useful approach.

### Conclusion

In this article, we learned the basics of a very important topic and approach in programming in general. We learned how much this approach is very useful when designing our software through learning how can help us to create safe and highly secured software, helps to decrease the complexity of creating the software by dividing the code into small blocks of code which makes the coding process easier, helps also in creating many instances from the same class without any conflict even if has different behavior and all of that give more flexibility and safety when updating our software.

We learned also how to apply this important approach in MQL5 to get all these incredible features then learned some simple applications that can be created by applying the OOP approach in MQL5 to be used in the MetaTrader 5.

I hope that you find this article useful for you and helped you to learn a very important and crucial topic in MQL5 programming as it may give you more insights about how many ideas can be easy when coding it after applying this OOP approach.

If you liked this article and you want to learn more about how you can create trading systems in MQL5 to be used in the MetaTrader 5 based on the most popular technical indicator and how you can create a custom technical indicator and use it in your Expert Advisor you can read my other articles about these topics through the publication on my profile and I hope that you find them useful as well.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12813.zip "Download all attachments in the single ZIP archive")

[indicatorClass.mq5](https://www.mql5.com/en/articles/download/12813/indicatorclass.mq5 "Download indicatorClass.mq5")(2.09 KB)

[priceClass.mq5](https://www.mql5.com/en/articles/download/12813/priceclass.mq5 "Download priceClass.mq5")(1.7 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/449917)**
(6)


![Alexey Viktorov](https://c.mql5.com/avatar/2017/4/58E3DFDD-D3B2.jpg)

**[Alexey Viktorov](https://www.mql5.com/en/users/alexeyvik)**
\|
22 Sep 2023 at 08:24

**Alexey Volchanskiy [#](https://www.mql5.com/ru/forum/454447#comment_49484647):**

Wow, how time flies )). About 10 years ago I tried to open a branch on OOP in Russian and English parts of this forum. In the Russian part I got hysterics that everything is complicated and we, proletarians, don't need it! Grandfathers ploughed with ploughshare and we will not break traditions! In the English part they simply killed the branch without explanation.

And now we are taught by Mohamed )). The article is a reprint from some boring academic textbook, I fell asleep at the first paragraph.

It was **probably** too early. Few people used OOP in their work back then. And those who knew and used it didn't want to waste their time discussing it.

Unlike you Alexey, I didn't fall asleep, I read to the end, but from the middle of the article I started skipping a few lines... In general, I didn't like the article. I don't see anything that is not in the documentation.

![Valeriy Yastremskiy](https://c.mql5.com/avatar/2019/1/5C4F743E-FA12.jpg)

**[Valeriy Yastremskiy](https://www.mql5.com/en/users/qstr)**
\|
22 Sep 2023 at 10:55

Let's start with the definition of OOP. OOP helps you create and develop reusable software without duplicating work and code by applying the DRY (don't repeat yourself) concept.

There's something to that, but where is the definition of OOP?

Encapsulation is an encapsulation that allows... It's hard to come by, the concept of visibility for OOPers is apparently private)))) And that [access modifier](https://www.mql5.com/en/docs/basis/types/classes "MQL5 Documentation: Structures and Classes") is encapsulation, readers should guess about it themselves)))).

It's a normal business, to make an extract from a textbook for 200 tugriks, I hope you wrote it yourself, without GPT)))).

![Viktor Vlasenko](https://c.mql5.com/avatar/2017/8/59927922-0364.jpg)

**[Viktor Vlasenko](https://www.mql5.com/en/users/vito333)**
\|
25 Sep 2023 at 02:26

From the following:

"Inside the [OnInit function](https://www.mql5.com/en/docs/basis/function/events#oninit "MQL5 Documentation: Event Handling Functions"), define the following for the three timeframes:

- String timeframe "

the quality of the article is imposed on the quality of the translation

![Fedor Arkhipov](https://c.mql5.com/avatar/2023/9/650183c2-8854.jpg)

**[Fedor Arkhipov](https://www.mql5.com/en/users/mtnet)**
\|
26 Sep 2023 at 21:24

Mahmoud was trying, and you just jumped on it :-)


![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
26 Sep 2023 at 22:39

**Fedor Arkhipov [#](https://www.mql5.com/ru/forum/454447#comment_49565901):**

Mahmud was trying, and you just jumped on him :-))

That's the easiest way to get attacked ))

Mahmud probably knows the saying "a dog barks and the caravan goes on its way".

![Rebuy algorithm: Multicurrency trading simulation](https://c.mql5.com/2/54/Multicurrency_Trading_Simulation_Avatar.png)[Rebuy algorithm: Multicurrency trading simulation](https://www.mql5.com/en/articles/12579)

In this article, we will create a mathematical model for simulating multicurrency pricing and complete the study of the diversification principle as part of the search for mechanisms to increase the trading efficiency, which I started in the previous article with theoretical calculations.

![Simple Mean Reversion Trading Strategy](https://c.mql5.com/2/55/Mean_reversion_avatar.png)[Simple Mean Reversion Trading Strategy](https://www.mql5.com/en/articles/12830)

Mean reversion is a type of contrarian trading where the trader expects the price to return to some form of equilibrium which is generally measured by a mean or another central tendency statistic.

![Category Theory in MQL5 (Part 12): Orders](https://c.mql5.com/2/56/Category-Theory-p12-avatar.png)[Category Theory in MQL5 (Part 12): Orders](https://www.mql5.com/en/articles/12873)

This article which is part of a series that follows Category Theory implementation of Graphs in MQL5, delves in Orders. We examine how concepts of Order-Theory can support monoid sets in informing trade decisions by considering two major ordering types.

![Category Theory in MQL5 (Part 11): Graphs](https://c.mql5.com/2/55/Category-Theory-p11-avatar.png)[Category Theory in MQL5 (Part 11): Graphs](https://www.mql5.com/en/articles/12844)

This article is a continuation in a series that look at Category Theory implementation in MQL5. In here we examine how Graph-Theory could be integrated with monoids and other data structures when developing a close-out strategy to a trading system.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/12813&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051638829078664135)

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