---
title: Developing a Replay System (Part 26): Expert Advisor project — C_Terminal class
url: https://www.mql5.com/en/articles/11328
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:42:53.955571
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/11328&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062662300730566284)

MetaTrader 5 / Tester


### Introduction

In the previous article " [Developing a Replay System — Market simulation (Part 25): Preparing for the next phase](https://www.mql5.com/en/articles/11203)", we prepared the replay/simulation system for basic use. However, we need more than just analysis of past movements or potential actions. Namely, we need a tool that will allow us to conduct targeted research, as if we were working on a real market. For this purpose, we need to create an Expert Advisor to conduct more in-depth research. In addition, we intend to develop a universal Expert Advisor that is applicable to various markets (stocks and forex), and is also compatible with our replay/simulation system.

Considering the scale of the project, the task is going to be very serious. However, the complexity of development is not as high as it seems, since we have already covered most of the process in previous articles. Among them are: " [Developing a trading Expert Advisor from scratch (Part 31): Towards the future (IV)](https://www.mql5.com/en/articles/10678)" and " [Creating an EA that works automatically (Part 15): Automation (VII)](https://www.mql5.com/en/articles/11438)". In these articles, I described in detail how to create a fully automated Expert Advisor. Despite these materials, here we have a unique and even more difficult challenge: making the MetaTrader 5 platform simulate a connection to a trading server, promoting realistic open market simulation. The task is undoubtedly rather complex.

Despite this, we should not be intimidated by the initial complexity. We need to start somewhere, otherwise we end up ruminating about the difficulty of a task without even trying to overcome it. That's what programming is all about: overcoming obstacles through learning, testing, and extensive research. Before we begin, I'd like to say that I have a lot of fun outlining and explaining how things actually come to existence. I believe that many have learned from this series of articles by going beyond the usual basics and testing that we can achieve much more than some think possible.

### Expert Advisor implementation concepts

You may have already noticed that I'm a big fan of object-oriented programming (OOP). This is due to the rich capabilities provided by OOP. It also provides a way of creating robust, secure and reliable code right from the start. To begin with, we need to get a preliminary idea of what we will need by organizing the structure of the project. Having experience as both a user and a programmer, I realized that for an Expert Advisor to be truly effective, it must use the resources that are always available to us: the keyboard and mouse. Given that the MetaTrader 5 platform is based on charts, using the mouse to interact with graphical elements is essential. But the keyboard also plays a key role in helping in various aspects. However, the discussion goes beyond the use of a mouse and keyboard, which will be covered in the automation series. In some cases, full automation can be accomplished without these tools, but when choosing to use them, it is important to consider the nature of the operation being performed. Thus, not all Expert Advisors are well suited for all types of assets.

This is because some of the assets have the price movement of 0.01. Others may have 0.5, while some may also have 5. In the case of Forex, these values differ significantly from the examples mentioned. Because of such diversity of values, some programmers choose to develop EAs specifically for specific assets. The reason is clear: the trading server does not accept arbitrary values; we need to adhere to the rules set by the server. The same principle applies to the replay/simulation system. We cannot allow the EA to execute orders with random values.

The introduction of this restriction is not only necessary, **it is extremely necessary**. There is no point in having functional replay/simulations for training if the system behaves completely differently when trading on a real account. It is therefore important that the system maintains a certain standardization and that it adapts as closely as possible to the reality of a real account. Therefore, it is necessary to develop an EA that will work as if it were directly interacting with the trading server, regardless of the circumstances.

### Let's start with the first class: the C\_Terminal class

While it is often possible to write all the code without specific guidance, this approach is not recommended for projects that have the potential to become very large and complex. We still don't know one exactly how the project will develop, but it is important to always start by focusing on best programming practices. Otherwise, without proper project planning, we get a huge amount of messy code. Therefore, it is important to think big from the very beginning, even if the project turns out to be not so grandiose or complex. Implementing best practices makes us more organized even in small projects and teaches us to follow a stable methodology. Let's start by developing the first class. To do this, we will create a new header file called C\_Terminal.mqh. It's good practice to give the file the same name as the class; this will make it easier to find when we need to work with it. The code starts as follows:

```
class C_Terminal
{

       protected:
   private  :
   public   :
};
```

In the article " [Creating an EA that works automatically (Part 05): Manual triggers (II)](https://www.mql5.com/en/articles/11237)", we considered some basic concepts about classes and reserved words. It's worth a look if you're unfamiliar with the series on creating an automated EA, as it contains a lot of the elements we'll be using here. Although that code is outdated and completely obsolete. Here we will see a code that will provide a new form of interaction because we need to address certain issues and make the system even more robust, reliable and efficient. The first thing that will actually appear in the class code after you start coding is the structure, which will be described in detail below:

```
class C_Terminal
{
   protected:
//+------------------------------------------------------------------+
      struct st_Terminal
      {
         long    ID;
         string  szSymbol;
         int     Width,
                 Height,
                 nDigits;
         double  PointPerTick,
                 ValuePerPoint,
                 VolumeMinimal,
                 AdjustToTrade;
      };
//+------------------------------------------------------------------+
```

It should be noted that this structure is declared inside the private part of the code, which is critical to what we are going to do. It's interesting to note that we are not declaring any variables here. In fact, any global variable within a class must be declared in a private part of the code, which provides the highest level of security and information encapsulation. We will return to this topic throughout the implementation for deeper understanding. For best programming practice, never allow internal class variables be accessed from other parts of the code that do not belong to the class.

Let's now see how class variables are declared:

```
   private :
      st_Terminal m_Infos;
```

For now, we have only one global and private variable in the C\_Terminal class, where we will store the corresponding data. We'll look at how to access this data from outside of the class later. At this point, it is very important to remember that no information will leak or enter the class without permission. Following this concept is critical. Many new programmers allow code outside the class to change the values of internal variables, which is an error, even if the compiler does not indicate it as an error. This practice compromises encapsulation, making the code significantly less safe and manageable, since changing a value without knowing the class can lead to errors and crashes that are difficult to detect and fix.

After this, you will need to create a new header file to keep the structure organized. This file, called Macros.mqh, will initially contain only one line.

```
#define macroGetDate(A) (A - (A % 86400))
```

This line will be used to highlight the date information. Choosing a macro instead of a function may seem unusual, but in many situations using macros makes more sense. This is because the macro will be inserted into the code as an inline function, allowing it to execute as quickly as possible. The use of macros is also justified by reducing the likelihood of making significant programming errors, especially when one or another refactoring needs to be repeated in the code several times.

**Note**: In this system, I will at certain points try to use a high-level programming language to make it easier to read and understand, especially for those starting to learn programming. Using a high-level language does not mean that the code will be slower, but rather that it will be easier to read. I'll show you how we can apply this to our codes.

Whenever possible, try to write code in a high-level language, as this makes debugging and improvement much easier. Also remember that code is written not only for the machine, as other programmers may also need to understand it.

After creating the Macros.mqh header file, which will define all the global macros, we will include this file in the C\_Terminal.mqh header file. It is included as follows:

```
#include "Macros.mqh"
```

Note that the header file name is enclosed in double quotes. Why is it indicated this way and not between the less and greater than signs ( **< >**)? Is there any special reason for this? Yes, there is. By using double quotes, we tell the compiler that the path to the header file should start in the directory where the current header file is located, in this case C\_Terminal.mqh. Since a specific path is not specified, the compiler will look for the Macros.mqh file in the same directory as the C\_Terminal.mqh file. Thus, if the project's directory structure has been changed, but we save the Macros.mqh file in the same directory as the C\_Terminal.mqh file, then we will not need to tell the compiler the new path.

Using a name enclosed in less than and greater than signs ( **< >**), we tell the compiler to start looking for the file in a predefined directory on the build system. For MQL5 this directory is **INCLUDE**. Therefore, any path to the Macros.mqh file must be specified from this directory **INCLUDE**, located in the MQL5 folder. If the project's directory structure changes, then it will be necessary to redefine all paths so that the compiler can find the header files. Although this may seem like a minor detail, the choice of a particular method can make a big difference.

Now that we understand this difference, let's look at the first code from the C\_Terminal class. This code is private to the class and therefore cannot be accessed from outside:

```
void CurrentSymbol(void)
   {
      MqlDateTime mdt1;
      string sz0, sz1;
      datetime dt = macroGetDate(TimeCurrent(mdt1));
      enum eTypeSymbol {WIN, IND, WDO, DOL, OTHER} eTS = OTHER;

      sz0 = StringSubstr(m_Infos.szSymbol = _Symbol, 0, 3);
      for (eTypeSymbol c0 = 0; (c0 < OTHER) && (eTS == OTHER); c0++) eTS = (EnumToString(c0) == sz0 ? c0 : eTS);
      switch (eTS)
      {
         case DOL:
         case WDO: sz1 = "FGHJKMNQUVXZ"; break;
         case IND:
         case WIN: sz1 = "GJMQVZ";       break;
         default : return;
      }
      for (int i0 = 0, i1 = mdt1.year - 2000, imax = StringLen(sz1);; i0 = ((++i0) < imax ? i0 : 0), i1 += (i0 == 0 ? 1 : 0))
      if (dt < macroGetDate(SymbolInfoInteger(m_Infos.szSymbol = StringFormat("%s%s%d", sz0, StringSubstr(sz1, i0, 1), i1), SYMBOL_EXPIRATION_TIME))) break;
   }
```

The code presented may seem complex and confusing, but it has a fairly specific function: to generate an asset name so that it can be used in the cross-order system. To understand how to do this, let's carefully analyze the process. The current focus is on creating names for index futures and dollar futures transactions according to the B3 (Brazilian Stock Exchange). By understanding the logic behind the creation of these names, the code can be adapted to generate the names of any future contract, allowing these contracts to operate through a cross-order system, as discussed earlier in the article " [Developing a trading EA from scratch (Part 11): Cross-order system](https://www.mql5.com/en/articles/10383)". However, the goal here is to extend this functionality so that the EA can adapt to different conditions, scenarios, assets or markets. This will require the EA to be able to determine what type of asset it will be dealing with, which may result in the need to include more different types of assets in the code. To explain it better, let's break it down into smaller parts.

```
MqlDateTime mdt1;
string sz0, sz1;
datetime dt = macroGetDate(TimeCurrent(mdt1));
```

These three lines are the variables that will be used in the code. The main problem here may be related to the way these variables are initialized. The [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) function is used to initialize two different variables in one step. The first variable **mdt1** is a structure of [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) type which stores detailed date and time information in the **mdt1** variable, while TimeCurrent also returns the value stored in **mdt1**. The second variable **dt** uses a macro to extract a date value and store it, allowing two variables to be fully initialized in one line of code.

```
enum eTypeSymbol {WIN, IND, WDO, DOL, OTHER} eTS = OTHER;
```

This line may seem unusual because we are simultaneously creating an enumeration, declaring a variable, and assigning an initial value to it. Understanding this line is key to understanding the other part of the function. Please pay attention to the following points: In MQL5, an enumeration cannot be created without a name, so we indicate the name at the very beginning. Inside the enumeration, we have elements that start with a null value by default. This can be changed, but we'll deal with it later. For now remember: by default, the enumeration starts from zero. So the value of WIN is zero, IND is one, and WDO is two, and so on. For a reason that will be explained later, the last element must be OTHER, regardless of the number of elements we want to include. After defining an enumeration, we declare a variable that will use the data from this enumeration, starting with the value of this variable as the value of the last element, i. e. OTHER.

**Important note**: Look at the declaration of the enumeration. Doesn't it seem familiar to you? Please note that names are also declared in capital letters, which is very important. What happens is: If we want to add additional assets for use in a future contract, we must do so by adding the first three characters of the contract name before the OTHER element so that the function can correctly generate the name of the current contract. For example, if we want to add a bull contract, we have to insert the BGI value into the enumeration and that is the first step. There is one more step that we will discuss later. Another example: if we want to add a corn futures contract, we add the CCM value, etc. always before OTHER. Otherwise the enumeration will not work as expected.

Now consider the following piece of code. Together with the enumeration described above, we will complete the first cycle of work.

```
   sz0 = StringSubstr(m_Infos.szSymbol = _Symbol, 0, 3);
   for (eTypeSymbol c0 = 0; (c0 < OTHER) && (eTS == OTHER); c0++) eTS = (EnumToString(c0) == sz0 ? c0 : eTS);
   switch (eTS)
   {
      case DOL:
      case WDO: sz1 = "FGHJKMNQUVXZ"; break;
      case IND:
      case WIN: sz1 = "GJMQVZ";       break;
      default : return;
   }
```

The first action is to save the asset name in a private global class variable. To simplify the process, we use the [StringSubstr](https://www.mql5.com/en/docs/strings/stringsubstr) function to capture the first three letters of the asset name in which the class code is executed and save them in the **sz0** variable. This is the easiest stage. Let's now do something very unusual, but possible: use an enumeration to determine which naming rule will be applied to the contract. To do this, we use the **for** loop. The expression used in this loop may seem quite strange, but what we are doing is iterating through the enumeration looking for the contract name originally defined in the enumeration, as I explained above. Since the default enumeration starts at zero, our local loop variable will also start at zero. Regardless of which element is first, the loop will start there and continue until the **OTHER** element is found or until the **eTS** variable is different from **OTHER**. At each iteration, we will increase the position inside the enumeration. Now the interesting part: In MQL5 we use the [EnumToString](https://www.mql5.com/en/docs/convert/enumtostring) function to convert the enumeration value into a string at each iteration of the loop and compare it with the value present in the **sz0** variable. When these values match, the position is saved in the **eTS** variable, as a result of which it becomes different from **OTHER**. This function is very interesting and shows that enumeration in MQL5 should not be treated in the same way as in other programming languages. Here you can think of an enumeration as an array of strings, offering much more functionality and practicality than in other languages.

After defining the desired value in the **eTS** variable, the next step is to define a specific naming rule for each contract, which requires the appropriate initialization of the variable **sz1** . Selecting the next letter in **sz1** depends on researching the specific contract we want to add to the naming rule, following the methodology presented here.

_If the asset is not included in the enumeration and the corresponding rule has not been found, the function will be completed. This is especially true when we are using an asset in the replay/simulation mode, as this type of asset is by nature personalized and special. For these cases, the function ends here._

Now we will study another loop, this is the stage in which " _everything is getting more complicated_". The complexity of this loop can confuse many programmers, making it difficult to understand its functionality. Therefore, it is important to pay even more attention to the following explanation. Here is the code of this loop:

```
for (int i0 = 0, i1 = mdt1.year - 2000, imax = StringLen(sz1);; i0 = ((++i0) < imax ? i0 : 0), i1 += (i0 == 0 ? 1 : 0))
	if (dt < macroGetDate(SymbolInfoInteger(m_Infos.szSymbol = StringFormat("%s%s%d", sz0, StringSubstr(sz1, i0, 1), i1), SYMBOL_EXPIRATION_TIME))) break;
```

Although the code may seem confusing and complex at first glance, it is actually simple. The code is shortened to make it more efficient. To make things simpler and avoid unnecessary complexity, we use the **IF** command, although this is not strictly necessary. Theoretically, the entire command can be included in the **FOR** loop, but that would complicate the explanation a little. That's why we use **IF** in this loop to check the match between the name of the generated contract and the name present on the trading server in order to determine which of the future contracts is the most relevant. To understand this process, it is important to know the naming convention used to create the contract name. As an example, let's look at what happens to a mini-dollar futures contract traded on the Brazilian Stock Exchange that follows a certain nomenclature rule:

- The first 3 characters of the contract name will be WDO. Regardless of the expiration date or whether it is a historical contract or not.
- Next we have a symbol indicating the month of expiration.
- After this we have a two-digit value indicating the year of expiration.

Thus, we mathematically construct the name of the contract, which is what this loop does. Using simple math rules and a loop, we create a contract name and check its validity. Therefore, it is important to follow the explanations to understand how it is done.

First, we initialize three local variables in the loop that will act as the accounting units we need. The loop executes its first iteration, which, what is especially interesting, occurs not inside the body of the loop, but in the **if** command However, the same code as in the **if** command can be placed between colons (;) in the for loop, and the loop will work identically. Let's understand what happens in this interaction. First we create the name of the contract, following specific rules for its formation. Using the [StringFormat](https://www.mql5.com/en/docs/convert/stringformat) function, we get the required name, which will be stored as a symbol name that we can access later. When we already have the contract name, we request from the trading server [one of the properties of the asset](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) \- contract expiration time using the **SYMBOL\_EXPIRATION\_TIME** enumeration. The [SymbolInfoInteger](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger) function will return a value, while we are only interested on the date. To extract exactly this value, we use a macro which allows us to compare the expiration date with the current date. If the return value is a future date, the loop will end because we have defined the most recent contract in the variable. However, at the initial stage this is unlikely to happen, since the year begins in 2000, that is, already in the past, so a new iteration will be required. Before repeating the entire process described, we need to increase the position to create a new contract name. Care must be taken here as this increase must be done first in the expiration code. Only if none of the expiration codes are satisfactory this year, we increase the year. This action is performed in three stages. In the code we use two [ternary](https://www.mql5.com/en/docs/basis/operators/ternary) operator to perform this increment.

Before the loop iterates again and even before ternary operators are executed, we increment the value indicating the expiration month symbol. After this increment, we check if the value is within acceptable limits using the first ternary operator. Thus, the index will always indicate one of the values valid for the month of expiration. The next step is to check the expiration month with the second ternary operator. If the expiration month index is zero, then all months have been checked. Then we increment the current year for a new attempt to find a valid contract, and this check will happen again in the **if** command. The process itself repeats until a valid contract is found, demonstrating how the system looks up the name of the current contract. **_This is not magic but a combination of mathematics and programming._**

I hope the explanations helped you understand how the code for this procedure works. Despite the complexity and length of the text, my goal was to explain it in an accessible way so that you can apply the same concept to implement functionality for other future contracts, allowing them to work through history. This is important because, whether contracts exist or not, the code will always use the correct contract.

Let's now analyze the following code, which references our class constructor:

```
C_Terminal()
{
   m_Infos.ID = ChartID();
   CurrentSymbol();
   ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, false);
   ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, 0, true);
   ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, false);
   m_Infos.nDigits = (int) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_DIGITS);
   m_Infos.Width   = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
   m_Infos.Height  = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
   m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
   m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
   m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
   m_Infos.AdjustToTrade = m_Infos.PointPerTick / m_Infos.ValuePerPoint;
}
```

This code ensures that the values in the global variable structure are correctly initialized. Pay attention to the parts that change the behavior of the MetaTrader 5 platform. Here is what happens. We ask the MetaTrader 5 platform not to generate descriptions of objects on the chart where this code is applied. In another line, we indicate that whenever an object is removed from the chart, the MetaTrader 5 platform should generate an event notifying which object was removed, and in this line we indicate the removal of the time scale. This is all we need at this stage. Further lines will collect information about the asset.

The next code is the class destructor:

```
~C_Terminal()
{
   ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, true);
   ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, true);
   ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, 0, false);
}
```

In this destructor code, we reset the conditions existing before the class constructor code was executed, returning the chart to its original state. Well, this may not be exactly the original state, but the time scale will again be visible on the chart. Now let's solve the case when the chart behavior can be changed by the class. We will create a small structure and change the constructor and destructor code to actually return the chart to the state it was in before the class changed it. This is done as follows:

```

   private :
      st_Terminal m_Infos;
      struct mem
      {
         long    Show_Descr,
                 Show_Date;
      }m_Mem;
//+------------------------------------------------------------------+
   public  :
//+------------------------------------------------------------------+
      C_Terminal()
      {
         m_Infos.ID = ChartID();
         CurrentSymbol();
         m_Mem.Show_Descr = ChartGetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR);
         m_Mem.Show_Date  = ChartGetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE);
         ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, false);
         ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, 0, true);
         ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, false);
         m_Infos.nDigits = (int) SymbolInfoInteger(m_Infos.szSymbol, SYMBOL_DIGITS);
         m_Infos.Width   = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
	 m_Infos.Height  = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
         m_Infos.PointPerTick  = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_SIZE);
         m_Infos.ValuePerPoint = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_TRADE_TICK_VALUE);
         m_Infos.VolumeMinimal = SymbolInfoDouble(m_Infos.szSymbol, SYMBOL_VOLUME_STEP);
         m_Infos.AdjustToTrade = m_Infos.PointPerTick / m_Infos.ValuePerPoint;
      }
//+------------------------------------------------------------------+
      ~C_Terminal()
      {
         ChartSetInteger(m_Infos.ID, CHART_SHOW_DATE_SCALE, m_Mem.Show_Date);
         ChartSetInteger(m_Infos.ID, CHART_SHOW_OBJECT_DESCR, m_Mem.Show_Descr);
         ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, 0, false);
      }
```

This global variable will represent the structure that will store the data for us. This way the class will be able to know what the chart was like before it was modified by the code. We capture here the data before changing and hen return the chart to its original state as of this point. Notice how a simple code change can make the system nicer and more convenient for us. It's worth noting that a global variable will store data for the entire life of the class. However, to understand this, we shouldn't think of a class as just a code set - it's very important to think of a class as if it were an object or a special variable. When it is created, the constructor code is executed, and when it is deleted or no longer needed, the destructor code is called. This is done automatically. If you don't fully understand how this works yet, don't worry, the concept will become clear as day later. For now, here's what you need to understand: a class is not just a bunch of code but is actually something special, and should be treated as such.

Before we close this topic, let's take a quick look at two other functions. We will look at them in detail in the next article, but for now we will see part of their code. Here is the code:

```
//+------------------------------------------------------------------+
inline const st_Terminal GetInfoTerminal(void) const
{

   return m_Infos;
}
//+------------------------------------------------------------------+
virtual void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
{

   switch (id)
   {
      case CHARTEVENT_CHART_CHANGE:
         m_Infos.Width = (int)ChartGetInteger(m_Infos.ID, CHART_WIDTH_IN_PIXELS);
         m_Infos.Height = (int)ChartGetInteger(m_Infos.ID, CHART_HEIGHT_IN_PIXELS);
         break;
   }
}
//+------------------------------------------------------------------+
```

These two functions are special in every way. I will add just a brief explanation of them as they will be better explained when we come to use them. This function allows any code outside the class body to access the data of a global variable contained in the class. This aspect will be covered extensively throughout the code we will develop. By taking this approach, we ensure that there is no risk of changing the values of a variable without the class knowing about it, since we will be using the compiler to help us avoid these types of problems. However, there is a problem here that we will solve in the future. This function is already used to update class data when the chart changes. These values will often be used in other parts of the code when we draw something on the chart. Again, we will see it in detail in the future.

### Conclusion

From the material covered in this article, we have already formed our base class C\_Terminal. However, there is still a function that we need to discuss. This is what we will do in the next article, in which we are going to create the C\_Mouse class. What we've covered here allows us to use the class to create something useful. I do not attach any related code here as our work is just beginning. Any code presented now will have no practical use. In the next article, we will create something really useful to get you started with your chart. We will develop indicators and other tools to support operations on demo and real accounts, and even in the replay/simulation system. See you in the next article!

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11328](https://www.mql5.com/pt/articles/11328)

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
**[Go to discussion](https://www.mql5.com/en/forum/462877)**
(1)


![edupoletto](https://c.mql5.com/avatar/avatar_na2.png)

**[edupoletto](https://www.mql5.com/en/users/edupoletto)**
\|
14 Sep 2023 at 00:02

Wow, what a cool [project](https://www.mql5.com/en/articles/7863 "Article: Projects let you create profitable trading robots! But it's not exactly")!

I'm curious because I'm not a programmer, and I'd like to know if you intend to sell it as a product.

Congratulations on your work.

Thanks.

![Developing a Replay System (Part 27): Expert Advisor project — C_Mouse class (I)](https://c.mql5.com/2/58/Projeto_Expert_AdvisoraClasse_C_Mous_Avatar.png)[Developing a Replay System (Part 27): Expert Advisor project — C\_Mouse class (I)](https://www.mql5.com/en/articles/11337)

In this article we will implement the C\_Mouse class. It provides the ability to program at the highest level. However, talking about high-level or low-level programming languages is not about including obscene words or jargon in the code. It's the other way around. When we talk about high-level or low-level programming, we mean how easy or difficult the code is for other programmers to understand.

![Data Science and Machine Learning (Part 20): Algorithmic Trading Insights, A Faceoff Between LDA and PCA in MQL5](https://c.mql5.com/2/70/Data_Science_and_Machine_Learning_Part_20__LOGO.png)[Data Science and Machine Learning (Part 20): Algorithmic Trading Insights, A Faceoff Between LDA and PCA in MQL5](https://www.mql5.com/en/articles/14128)

Uncover the secrets behind these powerful dimensionality reduction techniques as we dissect their applications within the MQL5 trading environment. Delve into the nuances of Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA), gaining a profound understanding of their impact on strategy development and market analysis.

![Developing a Replay System (Part 28): Expert Advisor project — C_Mouse class (II)](https://c.mql5.com/2/58/Replay-p28_II_avatar.png)[Developing a Replay System (Part 28): Expert Advisor project — C\_Mouse class (II)](https://www.mql5.com/en/articles/11349)

When people started creating the first systems capable of computing, everything required the participation of engineers, who had to know the project very well. We are talking about the dawn of computer technology, a time when there were not even terminals for programming. As it developed and more people got interested in being able to create something, new ideas and ways of programming emerged which replaced the previous-style changing of connector positions. This is when the first terminals appeared.

![MQL5 Wizard Techniques you should know (Part 12): Newton Polynomial](https://c.mql5.com/2/70/MQL5_Wizard_Techniques_you_should_know_Part_12_Newton_Polynomial___LOGO__1.png)[MQL5 Wizard Techniques you should know (Part 12): Newton Polynomial](https://www.mql5.com/en/articles/14273)

Newton’s polynomial, which creates quadratic equations from a set of a few points, is an archaic but interesting approach at looking at a time series. In this article we try to explore what aspects could be of use to traders from this approach as well as address its limitations.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/11328&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062662300730566284)

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