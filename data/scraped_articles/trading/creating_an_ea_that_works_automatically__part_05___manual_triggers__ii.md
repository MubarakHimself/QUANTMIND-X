---
title: Creating an EA that works automatically (Part 05): Manual triggers (II)
url: https://www.mql5.com/en/articles/11237
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:09:32.687257
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=fopivsirvlssgntsavlznoclpimiedei&ssn=1769180970524407550&ssn_dr=0&ssn_sr=0&fv_date=1769180970&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11237&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20an%20EA%20that%20works%20automatically%20(Part%2005)%3A%20Manual%20triggers%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918097056628672&fz_uniq=5069172977230086442&sv=2552)

MetaTrader 5 / Trading


### Introduction

In the previous article entitled [Creating an EA that works automatically (Part 04): Manual triggers (I)](https://www.mql5.com/en/articles/11232) I have shown how, with a bit of programming, to send market orders and to place pending orders using a combination of keys and mouse.

At the end of the previous article, I suggested that it would be appropriate to allow manual use of the EA, at least for a while. This turned out to be a lot more interesting than expected as the original idea was to release 3-4 articles showing how to actually get started with developing an EA that can trade automatically. Although this is fairly easy for programmers, this can be difficult for beginners who are just starting to learn programming, since there is quite little material available that clearly explains how to actually program certain things. Furthermore, being limited to one level of knowledge is not really something that one should do.

Since many may be using articles published in this community to start learning programming, I see this as an opportunity to share some of my experience, based on years of programming in C/C++ and to show how to implement some things in MQL5 which is very similar to C/C++. I want to show, that programming is nothing mythical, it's all real.

Well, to make the use of our EA in the manual mode more comfortable, we need to do a few things. This work is simple and easy for programmers, so we can get straight to the point. Namely, we will create lines indicating the location of order limits for the orders that we send to the trading server.

These limits are more appropriate to be viewed when we are using the mouse to place order, i.e., when we are creating a pending order. Once the order is already on the server, the indication is managed by the MetaTrader 5 platform. But before this actually happens, we need to show the user where the order limits are most likely to be placed. This is done by us, the programmers. The only support we receive from MetaTrader 5 is the possibility to use horizontal lines on the chart. Except for that, all the work must be implemented via EA programming.

To do this, we simply need to write the code that will place these lines at the right positions of the chart. But we don't want to do this in some random way. This should be properly controlled, since we don't want to compromise the code that we have already created, and we don't want to add work in case we have to remove the C\_Mouse class and the OnChartEvent event handler from the EA in the future. This is because an automated EA does not need these things, but a manual EA needs them. We need to make sure that these things are minimally usable.

### Creating the C\_Terminal class

For this purpose, we will create something convenient from manual operations. We will need to add lines indicating possible limits of an order or position which will be sent. In doing so, we will remove duplicate codes in the C\_Orders and C\_Mouse classes. Thus, we will have a new class: the C\_Terminal class which will help us build and isolate some things so that we can work comfortably. By using this class, we will be able to create both automated and manual EAs in the future, without running the risk of generating some kind of catastrophic failure in the new EA.

The biggest problem is that often, when creating a new automated EA, many do it from scratch. This approach often leads to numerous errors, as there are not enough checks.

True, it would be quite interesting to convert these classes into a private library. But since our intention is different at the mode, we will not think about it now. Perhaps, I will do it in the future. Let's see what we are actually going to do. We will start with the following: as usual we create a header file named C\_Terminal.mqh. This is the most basis code, which is always present in every class we are going to create. It is shown below:

```
class C_Terminal
{
        private :
        public  :
};
```

Always initialize your code in a class in such a way that you never forget that some points must be in the private section and others can be in the public one. Even if you don't have anything private in a class, it's always a good idea to clarify things. Mainly because you can show your code to other people.

The code that is well delimited and well written code, i.e. the code that is easy to read, is sure to attract other people's interest and encourage them to analyze it if you need help troubleshooting a problem. Code that is all messy, without any organization, without tabs and often without explanatory comments becomes uninteresting. Even if the idea is good, no one likes to spend time organizing someone else's code in order to understand what you are doing with it.

That is my advice. Of course, my own codes are not perfect, but: always clean up your codes, use tabs whenever you need to put multiple lines that are nested within one procedure, which helps a lot. Not just other people, but mostly you. Sometimes code is so badly organized that even its creator cannot figure it out. How will other programmers do it?

So, let's start writing the code, adding structure to our class. The first lines of the code are shown below.

```
class C_Terminal
{
        protected:
//+------------------------------------------------------------------+
                struct stTerminal
                {
                        ENUM_SYMBOL_CHART_MODE ChartMode;
                        int     nDigits;
                        double  VolMinimal,
                                VolStep,
                                PointPerTick,
                                ValuePerPoint,
                                AdjustToTrade;
                };
//+------------------------------------------------------------------+
        private :
        public  :
};
```

Here we have a new thing — the reserved word **protected**. But what does it tell us? We usually used only private and public declarations. So, what is this? Actually, it is somewhere in the middle, between public and private ones. To understand what is going on, we need to understand some basic concepts of object-oriented programming.

One of the concepts is **Inheritance**. But before we delve into the topic of inheritance, let's consider the class using the example of an individual person. So, to better understand the concept, think of each class as of an individual, unique and exclusive, living being. Now we can proceed to the explanation.

Some information is public, which allows anyone, in addition to the individual who maintains it, to benefit from its use and knowledge. This information is always placed in the public part of the code. Other information is private data of an individual, that is, it can only be accessed by this individual. When the person ceases to exist, this information will die with him; and he was the only one who could benefit from it. Think of information as a personal skill. The person cannot teach or pass it on to anyone else, and no one can take it away from him. This type of information is in the private part of the code.

But there is also information that does not fit into any of these concepts. It is located in the protected part, so the individual may or may not use it. The main thing is that it can be passed further down to lineage members. To understand how this happens, let's delve into the topic of inheritance.

When we come to the topic of inheritance, the easiest way to understand it is to think about the bloodline. There are three types of inheritance: public, private and protected. Here I am talking about inheritance, and not about individual issues of each member of the lineage.

In a public inheritance, the information, data and content of the parent are passed on to the children and to all their descendants, including grandchildren and beyond, and anyone outside of the bloodline can access, in theory, these things. Pay attention to the phrase **in theory**, as there are some nuances in such transfer. We'll look at this in more detail later. Let's focus on inheritance first. As for private inheritance, only the first generation will have access to the information, while subsequent generations will not be able to access it, even if they are part of the bloodline.

And the last thing we have is protected inheritance. It creates something very similar to private inheritance. But we have an aggravating factor, which makes many people not understand these concepts: the parent clause. This is because, there is a kind of rule for passing the information, even in cases of public inheritance. Some things cannot be accessed outside the bloodline. To understand this, see the table below, in which I briefly summarize this issue:

| Definition in parent class | Inheritance type | Access from child class | Access by calling child class |
| --- | --- | --- | --- |
| private | public: | Access is denied | Unable to access base class data or procedures |
| public: | public: | Access is allowed | Able to access base class data or procedures |
| protected | public: | Access is allowed | Unable to access base class data or procedures |
| private | private | Access is denied | Unable to access base class data or procedures |
| public: | private | Access is allowed | Unable to access base class data or procedures |
| protected | private | Access is allowed | Unable to access base class data or procedures |
| private | protected | Access is denied | Unable to access base class data or procedures |
| public: | protected | Access is allowed | Unable to access base class data or procedures |
| protected | protected | Access is allowed | Unable to access base class data or procedures |

Table 1) Inheritance system based on information definition

Please note that depending on the code part used in the data type definition during inheritance, the child may or may not have access to the data. But any call outside the bloodline will not have access, except in the unique case that occurs when the parent's data is declared public and the child inherits in the same public way. In addition, it is not possible to access any information outside of the lineage.

Without understanding the scheme shown in table 01, many less experienced programmers disrespect object-oriented programming. This is because they actually don't know how things really work. Those of you who have followed my articles and my code should have noticed that I use object-oriented programming a lot.

This is because it provides an ultimate level of security when implementing very complex things, which would be impossible to do otherwise. And I am not talking only about inheritance. Beyond that, we also have polymorphism and encapsulation, but those are topics for another time. Although, encapsulation is part of table 01, it deserves a more detailed explanation which is beyond the scope of this article.

And so, let's continue. if you look carefully, you can notice that the structure in the code above is the same as in the C\_Orders class. Pay attention to this, because the C\_Order class will lose the definition of this data and will start to inherit the data from the C\_Terminal class. But for now, let's stay inside the C\_Terminal class.

The next thing to add to the C\_Terminal class are functions that are common to both the C\_Mouse class and the C\_Orders class. These functions will be added to the protected part of the C\_Terminal class, so when the C\_Mouse and C\_Orders classes are inherited from C\_Terminal, these functions and procedures will follow table 01. So, we add the following code:

```
//+------------------------------------------------------------------+
inline double AdjustPrice(const double value)
                        {
                                return MathRound(value / m_TerminalInfo.PointPerTick) * m_TerminalInfo.PointPerTick;
                        }
//+------------------------------------------------------------------+
inline double FinanceToPoints(const double Finance, const uint Leverage)
                        {
                                double volume = m_TerminalInfo.VolMinimal + (m_TerminalInfo.VolStep * (Leverage - 1));

                                return AdjustPrice(MathAbs(((Finance / volume) / m_TerminalInfo.AdjustToTrade)));
                        };
//+------------------------------------------------------------------+
```

These codes are no longer duplicated in both classes. Now all the code is exclusively inside the C\_Terminal class, which facilitates its maintenance, testing and possible modifications. Thus, our code will become more reliable and attractive as we use and extend it.

There are a few more things to pay attention to inside the C\_Terminal class. But first let's take a look at the class constructor. It is shown below:

```
        public  :
//+------------------------------------------------------------------+
                C_Terminal()
                        {
                                m_TerminalInfo.nDigits          = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
                                m_TerminalInfo.VolMinimal       = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
                                m_TerminalInfo.VolStep          = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
                                m_TerminalInfo.PointPerTick     = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
                                m_TerminalInfo.ValuePerPoint    = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
                                m_TerminalInfo.AdjustToTrade    = m_TerminalInfo.ValuePerPoint / m_TerminalInfo.PointPerTick;
                                m_TerminalInfo.ChartMode        = (ENUM_SYMBOL_CHART_MODE) SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE);
                        }
//+------------------------------------------------------------------+
```

Note that it is almost identical to the one that existed in the C\_Orders class. Now, we can modify the C\_Orders class code so that it inherits what we are implementing in the C\_Terminal class. But there is one important thing. Look at the code part that contains the declaration of the structure which is initialized in the above constructor. You can see that it has no variable. Why?

The reason is encapsulation. You should not allow the code outside the class to access and modify the contents of internal class variables. This is a serious programming error, although the compiler won't complain, you should **NEVER** allow this. Absolutely all global variables of a class must always be declared inside the private part. The variable declaration looks like below.

```
//+------------------------------------------------------------------+
        private :
                stTerminal m_TerminalInfo;
        public  :
//+------------------------------------------------------------------+
```

Note that the global class variable is defined between the private and public parts. Thus, it will be unavailable for any class that inherits from C\_Terminal, in other words, we guarantee the encapsulation of information and at the same time we add inheritance to our code. In addition to expanding the code utility we grow the level of robustness exponentially.

But how will we access the required data in the above class? We need to give some level of access to the variables of the parent class which in this case is the C\_Terminal class. Yes, we need this. But we shouldn't do it by making these variables public or protected. This is a programming error. You must add some means, so that the derived classes can access the values of the parent class. But here lies the danger, which is important: YOU MUST NOT ALLOW ANY DERIVED CLASSES TO MODIFY THE VARIABLES OF THE PARENT CLASS.

To do this, we need to somehow convert the variable to a constant. That is, the parent class can modify the values of the variables as needed and when needed. If any child class wants to make any change to any parent class variable, then the child class must call some procedure provided by the parent class to inform the desired value to use for the certain variable present in the parent class. Such a procedure, which should be implemented in the parent class, will check whether the data passed by the child is valid. If it is valid, the procedure will push the changes requested by the child inside the parent class.

But absolutely never will a child element be able to change parent data without the parent class knowing about such a change. I have seen lots of potentially dangerous codes that do so. Some people say that calling a procedure within the parent class to check the data provided by the child slows down the code or causes the program to crash on execution. But this is an error. The cost and risk of throwing incorrect values inside the parent class is not worth the small increase in the speed if such an increase is achieved by omitting the call of data check procedures. So, don't worry that this will make the code slow.

Now we have come to another point. You as a programmer who is or wants to become a professional, should always give first preference to placing any procedure that will be inherited by other classes within a protected code part, and only as a last resort, transfer the procedure to the public area. We always prioritize encapsulation. Only if it is really necessary, we leave the encapsulation and allows the public use of functions and any procedures. But we never do it with variables, as they always should be private.

To create a procedure or a function that allows a child class to access parent class data, we will use the following function:

```
inline const stTerminal GetTerminalInfos(void) const
                        {
                                return m_TerminalInfo;
                        }
```

Now I want you to pay close attention to what I'm about to explain because it's extremely important and makes the difference between well-written code and a simply well-done code.

Throughout this article, I've said that we must somehow allow code outside the class, in which the variables are declared and used, access them. I said that the ideal case would the situation where the variable could be modified whenever necessary inside the class in which it is declared. But outside the class the variable should be treated as a constant, i.e. its value cannot be changed.

Despite being extremely simple, the code above does exactly this. In other words, you can guarantee that inside the C\_Terminal class we have an accessible variable and its value can be changed, but outside the class the same variable will be treated as a constant. How is it done and why do we have two reserved words const **here**?

Let's consider it one by one: the first word **const** informs the compiler that the return variable **m\_TerminalInfo** should be considered as a constant in the caller function. If the caller tries to modify the value of any of the structure members presented in the returned variable, the compiler will generate an error and will prevent the coed from being compiled. The second word **const** informs the compiler that if some value is modified here for any reason, the compiler should return the error. Thus, you will not be able to modify, even if you want to do this, any data inside this function, as it exists only to return a value.

Some programmers sometimes make this kind of error: they modify variable values within functions or procedures, where these variables should only be used for external access, but not for some sort of factoring. By using the code principle above you avoid this type of error.

While we haven't finished our base C\_Terminal class yet, we can remove duplicated parts in the code thus making the C\_Mouse class have the same type of code as the C\_Orders class. But since changing the C\_Mouse class is much easier, let's see how it will look now that it inherits the C\_Terminal class. This can be seen in the following code:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "C_Terminal.mqh"
//+------------------------------------------------------------------+
#define def_MouseName "MOUSE_H"
//+------------------------------------------------------------------+
#define def_BtnLeftClick(A)     ((A & 0x01) == 0x01)
#define def_SHIFT_Press(A)      ((A & 0x04) == 0x04)
#define def_CTRL_Press(A)       ((A & 0x08) == 0x08)
//+------------------------------------------------------------------+
class C_Mouse : private C_Terminal
{
// Inner class code ....
};
```

Here we are including the header file of the C\_Terminal class. Note that the filename is enclosed in double quotes here. This tells the compiler that the C\_Terminal.mqh file is located in the same directory as the C\_Mouse.mqh file. This way, if you need to move both files to another location, the compiler will always be able to find the correct file since it knows that they are located in the same directory.

Now, following the idea of always starting work with the least possible access, let's make the C\_Mouse class a private child of the C\_Terminal class. Now you can delete the **AdjustPrice** function from the C\_Mouse class, as well as the **PointPerTick** class that exists in C\_Mouse, since now you are going to use the procedure from the C\_Terminal class. Since the class is inherited privately and the **AdjustPrice** function is inside the protected part of the code, you will get in C\_Terminal the result as per table 01. Thus, it will be impossible to call the **AdjustPrice** procedure outside the C\_Mouse class as it was done earlier.

However, all these changes in the C\_Mouse class are temporary. We'll make a few more changes to add limit lines we need when using the EA manually. But we will do that later. Let's see how to make changes in the C\_Orders class which are more profound. These changes require a separate topic. So, let's move on to it.

### Changing the C\_Orders class after inheriting from C\_Terminal

We start the changes similarly to that in the C\_Mouse class. Then come differences, which you can see in the code below.

```
#include "C_Terminal.mqh"
//+------------------------------------------------------------------+
class C_Orders : private C_Terminal
{
        private :
//+------------------------------------------------------------------+
                MqlTradeRequest m_TradeRequest;
                ulong           m_MagicNumber;
                struct st00
                {
                        int     nDigits;
                        double  VolMinimal,
                                VolStep,
                                PointPerTick,
                                ValuePerPoint,
                                AdjustToTrade;
                        bool    PlotLast;
                        ulong   MagicNumber;
                }m_Infos;
//+------------------------------------------------------------------+
```

The whole principle is almost the same as that of the C\_Mouse class, but there are some differences. First, we delete the C\_Orders class structure as shown in the highlighted lines. But we need certain data from within the structure, so we'll make it private but as just a regular variable.

Since we remove the highlighted parts, you may think that it will take a lot of effort to write the new code. Actually, it is going to be quite a bit of work. Let's immediately move on to the constructor for this C\_Orders class. Changes will actually start in here. Below is the new class constructor.

```
                C_Orders(const ulong magic)
                        :C_Terminal(), m_MagicNumber(magic)
                        {
                                m_Infos.MagicNumber     = magic;
                                m_Infos.nDigits         = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
                                m_Infos.VolMinimal      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
                                m_Infos.VolStep         = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
                                m_Infos.PointPerTick    = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
                                m_Infos.ValuePerPoint   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
                                m_Infos.AdjustToTrade   = m_Infos.ValuePerPoint / m_Infos.PointPerTick;
                                m_Infos.PlotLast        = (SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE) == SYMBOL_CHART_MODE_LAST);
                        };
```

As you can see, all the internal content of the constructor has been removed, but here we are forcing the call of the C\_Terminal class constructor. This is done to ensure that it is called before anything else. Usually, the compiler does this for us, but we will implement it explicitly and at the same time we initialize a variable which indicates the magic number at another place in the code.

Normally this is done in constructors, because we want the variable value to be defined even before any code is executed — this will enable the compiler to generate an adequate code. But if the value is constant, as it normally will be, we save some time in the initialization of the C\_Orders class by doing this. However, remember the following detail: You will only have some benefit if the value is a constant, otherwise the compiler will generate a code that will not give us any practical benefit.

The next thing to do is to remove the **AdjustPrice** and **FinanceToPoints** functions from the C\_Orders class, but since this can be done directly, I won't show it here. From now on, these calls will use the code inside the C\_Terminal class.

Let's look at one of the code parts that will use the variable declared in the C\_Terminal class. This is to understand how to access the variables of the parent class. Take a look at the following code:

```
inline void CommonData(const ENUM_ORDER_TYPE type, const double Price, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
                        {
                                double Desloc;

                                ZeroMemory(m_TradeRequest);
                                m_TradeRequest.magic            = m_Infos.MagicNumber;
                                m_TradeRequest.magic            = m_MagicNumber;
                                m_TradeRequest.symbol           = _Symbol;
                                m_TradeRequest.volume           = NormalizeDouble(m_Infos.VolMinimal + (m_Infos.VolStep * (Leverage - 1)), m_Infos.nDigits);
                                m_TradeRequest.volume           = NormalizeDouble(GetTerminalInfos().VolMinimal + (GetTerminalInfos().VolStep * (Leverage - 1)), GetTerminalInfos().nDigits);
                                m_TradeRequest.price            = NormalizeDouble(Price, m_Infos.nDigits);
                                m_TradeRequest.price            = NormalizeDouble(Price, GetTerminalInfos().nDigits);
                                Desloc = FinanceToPoints(FinanceStop, Leverage);
                                m_TradeRequest.sl               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? -1 : 1)), m_Infos.nDigits);
                                m_TradeRequest.sl               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? -1 : 1)), GetTerminalInfos().nDigits);
                                Desloc = FinanceToPoints(FinanceTake, Leverage);
                                m_TradeRequest.tp               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? 1 : -1)), m_Infos.nDigits);
                                m_TradeRequest.tp               = NormalizeDouble(Desloc == 0 ? 0 : Price + (Desloc * (type == ORDER_TYPE_BUY ? 1 : -1)), GetTerminalInfos().nDigits);
                                m_TradeRequest.type_time        = (IsDayTrade ? ORDER_TIME_DAY : ORDER_TIME_GTC);
                                m_TradeRequest.stoplimit        = 0;
                                m_TradeRequest.expiration       = 0;
                                m_TradeRequest.type_filling     = ORDER_FILLING_RETURN;
                                m_TradeRequest.deviation        = 1000;
                                m_TradeRequest.comment          = "Order Generated by Experts Advisor.";
                        }
```

The highlighted parts have been removed and they have been replaced by other highlighted code. Now, please pay attention to the code highlighted in yellow. It contains something that many may have never seen. Please note that there is a function in these highlighted yellow codes, _which is considered as a structure._ But what a crazy thing is this! 😵😱

Calm down my dear reader. Do not worry. It is not crazy. It's just using programming in a slightly more exotic way than you have seen before. To understand why this is allowed and why it works, let's see the function separately:

```
GetTerminalInfos().nDigits
```

Now I want you to go back to the C\_Terminal class code and see how this function is declared. It is as follows:

```
inline const stTerminal GetTerminalInfos(void) const
                        {
                                return m_TerminalInfo;
                        }
```

Note that the GetTerminalInfos function returns a structure shown in the code below:

```
                struct stTerminal
                {
                        ENUM_SYMBOL_CHART_MODE ChartMode;
                        int     nDigits;
                        double  VolMinimal,
                                VolStep,
                                PointPerTick,
                                ValuePerPoint,
                                AdjustToTrade;
                };
```

For the compiler, what we are doing using the **GetTerminalInfos().nDigits** code would be equivalent to saying that **GetTerminalInfos()** is not a function but a variable 😲. Did you get confused? Well, things get even more interesting because for a compiler the **GetTerminalInfos().nDigits** code will be equivalent to the following code:

```
stTerminal info;
int value = info.nDigits;

value = 10;
info.nDigits = value;
```

That is, you can not only read the value, but also write it. So, if you accidentally write the following fragment:

```
GetTerminalInfos().nDigits = 10;
```

The compiler will understand that value **10** must be placed in the variable referenced by the function **GetTerminalInfos()**. This would be a problem, since the referred variable is in the C\_Terminal class, and this variable is declared in the private part of the code. It means that it cannot be modified by the previously made call. But since the **GetTerminalInfos()** function is also protected (although it can also be public and that would be the same), the variable declared as private has the same access level as the function that refers to it.

Have you realized how dangerous these things can be? That is, even if you declare a variable as private, but incorrectly write the code for a function or procedure that refers to it, then you or someone else can inadvertently change its value. And it breaks the whole concept of encapsulation.

But since, during function declaration, it was launched with the keyword **const**, this changes everything, since now the compiler will see the **GetTerminalInfo()** function differently. To understand this, simply try to use the below code anywhere in the C\_Orders class:

```
GetTerminalInfos().nDigits = 10;
```

If you try to do this, the compiler will throw an error. Because the compiler considers **GetTerminalInfos().nDigits** or any other thing inside the structure to which **GetTerminalInfos() refers** as a constant, and the constant value cannot be changed. This is considered an error.

Now do you understand how to refer to constant data using a variable? Thus, for the C\_Terminal class the structure to which the **GetTerminalInfos()** function refers is a variable, but for any other code it is a constant 😁.

Now that I have explained this part, let's move on to other modification. I think now you can understand what is happening and where the data referenced by C\_Orders comes from. The next function to change can be seen just below:

```
                ulong CreateOrder(const ENUM_ORDER_TYPE type, const double Price, const double FinanceStop, const double FinanceTake, const uint Leverage, const bool IsDayTrade)
                        {
                                double  bid, ask;

                                bid = SymbolInfoDouble(_Symbol, (m_Infos.PlotLast ? SYMBOL_LAST : SYMBOL_BID));
                                bid = SymbolInfoDouble(_Symbol, (GetTerminalInfos().ChartMode == SYMBOL_CHART_MODE_LAST ? SYMBOL_LAST : SYMBOL_BID));
                                ask = (m_Infos.PlotLast ? bid : SymbolInfoDouble(_Symbol, SYMBOL_ASK));
                                ask = (GetTerminalInfos().ChartMode == SYMBOL_CHART_MODE_LAST ? bid : SymbolInfoDouble(_Symbol, SYMBOL_ASK));
                                CommonData(type, AdjustPrice(Price), FinanceStop, FinanceTake, Leverage, IsDayTrade);
                                m_TradeRequest.action   = TRADE_ACTION_PENDING;
                                m_TradeRequest.type     = (type == ORDER_TYPE_BUY ? (ask >= Price ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_BUY_STOP) :
                                                                                    (bid < Price ? ORDER_TYPE_SELL_LIMIT : ORDER_TYPE_SELL_STOP));

                                return (((type == ORDER_TYPE_BUY) || (type == ORDER_TYPE_SELL)) ? ToServer() : 0);
                        };
```

And the last thing to be modified:

```
                bool ModifyPricePoints(const ulong ticket, const double Price, const double PriceStop, const double PriceTake)
                        {
                                ZeroMemory(m_TradeRequest);
                                m_TradeRequest.symbol   = _Symbol;
                                if (OrderSelect(ticket))
                                {
                                        m_TradeRequest.action   = (Price > 0 ? TRADE_ACTION_MODIFY : TRADE_ACTION_REMOVE);
                                        m_TradeRequest.order    = ticket;
                                        if (Price > 0)
                                        {
                                                m_TradeRequest.price      = NormalizeDouble(AdjustPrice(Price), m_Infos.nDigits);
                                                m_TradeRequest.sl         = NormalizeDouble(AdjustPrice(PriceStop), m_Infos.nDigits);
                                                m_TradeRequest.tp         = NormalizeDouble(AdjustPrice(PriceTake), m_Infos.nDigits);
                                                m_TradeRequest.price      = NormalizeDouble(AdjustPrice(Price), GetTerminalInfos().nDigits);
                                                m_TradeRequest.sl         = NormalizeDouble(AdjustPrice(PriceStop), GetTerminalInfos().nDigits);
                                                m_TradeRequest.tp         = NormalizeDouble(AdjustPrice(PriceTake), GetTerminalInfos().nDigits);
                                                m_TradeRequest.type_time  = (ENUM_ORDER_TYPE_TIME)OrderGetInteger(ORDER_TYPE_TIME) ;
                                                m_TradeRequest.expiration = 0;
                                        }
                                }else if (PositionSelectByTicket(ticket))
                                {
                                        m_TradeRequest.action   = TRADE_ACTION_SLTP;
                                        m_TradeRequest.position = ticket;
                                        m_TradeRequest.tp       = NormalizeDouble(AdjustPrice(PriceTake), m_Infos.nDigits);
                                        m_TradeRequest.sl       = NormalizeDouble(AdjustPrice(PriceStop), m_Infos.nDigits);
                                        m_TradeRequest.tp       = NormalizeDouble(AdjustPrice(PriceTake), GetTerminalInfos().nDigits);
                                        m_TradeRequest.sl       = NormalizeDouble(AdjustPrice(PriceStop), GetTerminalInfos().nDigits);
                                }else return false;
                                ToServer();

                                return (_LastError == ERR_SUCCESS);
                        };
```

Now we will finish this part. The code has the same level of stability while we have improved its robustness. Since previously we had duplicate functions and could run the risk that they will be modified in one class and remain the same in another. If some error occurred, it would not be simple to correct it, as we could fix it in one class, but it would still remain in the other, thus leaving the code less robust and reliable.

Always think about this: _A little work to improve your code in terms of stability and robustness is not a work, it's just a hobby_ 😁.

But we are not yet done with what we were going to do in this article. We wanted to add limit lines (Take Profit and Stop Loss levels) in order to know these levels when placing an order manually. This part is still missing. Without it, we cannot finish this article and move on to the next one. However, we will create a new section here to separate this topic from what we have already covered.

### Creating Take Profit and Stop Loss lines

Now let's think about the following question: what is the best place to add the code creating these lines? We have a call point. It is shown below:

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        uint            BtnStatus;
        double  Price;
        static double mem = 0;

        (*mouse).DispatchMessage(id, lparam, dparam, sparam);
        (*mouse).GetStatus(Price, BtnStatus);
        if (TerminalInfoInteger(TERMINAL_KEYSTATE_CONTROL))
        {
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_UP))  (*manager).ToMarket(ORDER_TYPE_BUY, user03, user02, user01, user04);
                if (TerminalInfoInteger(TERMINAL_KEYSTATE_DOWN))(*manager).ToMarket(ORDER_TYPE_SELL, user03, user02, user01, user04);
        }
        if (def_SHIFT_Press(BtnStatus) != def_CTRL_Press(BtnStatus))
        {
// This point ...
                if (def_BtnLeftClick(BtnStatus) && (mem == 0)) (*manager).CreateOrder(def_SHIFT_Press(BtnStatus) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, mem = Price, user03, user02, user01, user04);
        }else mem = 0;
}
```

The part marked in yellow is where we should add the call to display the Take Profit and Stop Loss lines. But there is one important detail: where should we write code for these lines?

The best alternative to this (and I think everyone will agree with it) is to add this code to the C\_Mouse class. So that when we remove the mouse, the lines will also disappear. This is what we will do. Let's open the C\_Mouse class to create the lines which will represent take profit and stop loss.

But I will do something a little different from what we have seen before. I will add the lines not to OnChartEvent, but to the event handler inside the C\_Mouse class. It is a better way, despite the fact that you will have to make some other changes to the EA code, but we will leave this for later. Let's open the C\_Mouse.mqh header file and implement everything we need.

The first thing we'll do is add some new definitions, as shown below:

```
#define def_PrefixNameObject    "MOUSE_"
#define def_MouseLineName       def_PrefixNameObject + "H"
#define def_MouseLineTake       def_PrefixNameObject + "T"
#define def_MouseLineStop       def_PrefixNameObject + "S"
#define def_MouseName           "MOUSE_H"
```

Note that the old definition has been deleted. We do things in a little different way, though programming is still pleasant. To reduce programming work, let's modify the creation procedure in a different way:

```
                void CreateLineH(void)
                void CreateLineH(const string szName, const color cor)
                        {
                                ObjectCreate(m_Infos.Id, def_MouseName, OBJ_HLINE, 0, 0, 0);
                                ObjectSetString(m_Infos.Id, def_MouseName, OBJPROP_TOOLTIP, "\n");
                                ObjectSetInteger(m_Infos.Id, def_MouseName, OBJPROP_BACK, false);
                                ObjectSetInteger(m_Infos.Id, def_MouseName, OBJPROP_COLOR, m_Infos.Cor);
                                ObjectCreate(m_Infos.Id, szName, OBJ_HLINE, 0, 0, 0);
                                ObjectSetString(m_Infos.Id, szName, OBJPROP_TOOLTIP, "\n");
                                ObjectSetInteger(m_Infos.Id, szName, OBJPROP_BACK, false);
                                ObjectSetInteger(m_Infos.Id, szName, OBJPROP_COLOR, cor);
                        }
```

Now all lines will be created in a unique way, and we only need to specify the name and color. I had to create 2 more variables to store colors, but I don't think thee need to be shown here. So, let's move on to the constructor, as now it will need to receive a lot more data than before, and you can see it below:

```
                C_Mouse(const color corPrice, const color corTake, const color corStop, const double FinanceStop, const double FinanceTake, const uint Leverage)
                        {
                                m_Infos.Id        = ChartID();
                                m_Infos.CorPrice  = corPrice;
                                m_Infos.CorTake   = corTake;
                                m_Infos.CorStop   = corStop;
                                m_Infos.PointsTake= FinanceToPoints(FinanceTake, Leverage);
                                m_Infos.PointsStop= FinanceToPoints(FinanceStop, Leverage);
                                ChartSetInteger(m_Infos.Id, CHART_EVENT_MOUSE_MOVE, true);
                                ChartSetInteger(m_Infos.Id, CHART_EVENT_OBJECT_DELETE, true);
                                CreateLineH(def_MouseLineName, m_Infos.CorPrice);
                        }
```

As I have already mentioned, I had to create a few more variables but the price of it is small compared to what we can get in terms of the features. Pay attention to the following: I will not wait for the EA call to convert the financial values to points. We are going to do it right here, right now. This will save us time later, as accessing a variable is much faster than calling a function. But what about the destructor? In fact, it is not more difficult. All we needed to do there is change the type of the function responsible for deleting object. See it below:

```
                ~C_Mouse()
                        {
                                ChartSetInteger(m_Infos.Id, CHART_EVENT_OBJECT_DELETE, false);
                                ObjectsDeleteAll(m_Infos.Id, def_PrefixNameObject);
                                ObjectDelete(m_Infos.Id, def_MouseName);
                        }
```

This function can remove all objects with the names starting in some particular way. It is very useful and versatile in many situations. Here is the last procedure that needs to be change. Let's see how I implemented the lines for the price limits in the code below:

```
                void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
                        {
                                int w;
                                datetime dt;
                                static bool bView = false;

                                switch (id)
                                        {
                                                case CHARTEVENT_OBJECT_DELETE:
                                                        if (sparam == def_MouseName) CreateLineH();
                                                        if (sparam == def_MouseLineName) CreateLineH(def_MouseLineName, m_Infos.CorPrice);
                                                        break;
                                                case CHARTEVENT_MOUSE_MOVE:
                                                        ChartXYToTimePrice(m_Infos.Id, (int)lparam, (int)dparam, w, dt, m_Infos.Price);
                                                        ObjectMove(m_Infos.Id, def_MouseName, 0, 0, m_Infos.Price = AdjustPrice(m_Infos.Price));
                                                        ObjectMove(m_Infos.Id, def_MouseLineName, 0, 0, m_Infos.Price = AdjustPrice(m_Infos.Price));
                                                        m_Infos.BtnStatus = (uint)sparam;
                                                        if (def_CTRL_Press(m_Infos.BtnStatus) != def_SHIFT_Press(m_Infos.BtnStatus))
                                                        {
								if (!bView)
								{
									if (m_Infos.PointsTake > 0) CreateLineH(def_MouseLineTake, m_Infos.CorTake);
									if (m_Infos.PointsStop > 0) CreateLineH(def_MouseLineStop, m_Infos.CorStop);
									bView = true;
								}
								if (m_Infos.PointsTake > 0) ObjectMove(m_Infos.Id, def_MouseLineTake, 0, 0, m_Infos.Price + (m_Infos.PointsTake * (def_SHIFT_Press(m_Infos.BtnStatus) ? 1 : -1)));
								if (m_Infos.PointsStop > 0) ObjectMove(m_Infos.Id, def_MouseLineStop, 0, 0, m_Infos.Price + (m_Infos.PointsStop * (def_SHIFT_Press(m_Infos.BtnStatus) ? -1 : 1)));
                                                        }else if (bView)
                                                        {
                                                                ObjectsDeleteAll(m_Infos.Id, def_PrefixNameObject);
                                                                bView = false;
                                                        }
                                                        ChartRedraw();
                                                        break;
                                        }
                        }
```

First I had to remove two lines of old code and then to add two lines of updated code. But the important detail comes when we are going to deal with the mouse move event. There we add some new lines. The first thing we do is check if the SHIFT or CTRL key is pressed, but not at the same time, and if so, then go to the next step.

Now, if the result is False, check whether the limit lines are displayed on the chart. If yes, remove all mouse lines. This is not a problem, as MetaTrader 5 immediately generates an event to notify that objects have been removed from the screen. When the screen event handler is called, you will be guided to place the price line back on the chart.

But let's get back to the moment when the limit lines will be displayed if you hold down the SHIFT or CTRL key. In this case, we check if there are already any lines on the screen. Then, if not, create them while the value is greater than zero, since we do not need a strange element on the chart. We mark this as done so as not to be trying to recreate these objects on every call. Then we will position them in their places depending on where the price line is.

### Conclusion

We have created and EA system to operate manually. We are ready for the next big step which we will consider in the next article. We will add a trigger so that the system can do something automatically. Once that's done, I'll show you what it takes to convert such a manual trading EA into a fully automated EA. The next article we will discuss how to automate the work of the EA, excluding human decisions from trading.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11237](https://www.mql5.com/pt/articles/11237)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11237.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_05.zip](https://www.mql5.com/en/articles/download/11237/ea_automatico_-_05.zip "Download EA_Automatico_-_05.zip")(5.67 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/442699)**
(1)


![Carlos Giovanni](https://c.mql5.com/avatar/2019/10/5DAB799A-F2D5.png)

**[Carlos Giovanni](https://www.mql5.com/en/users/cgnc)**
\|
8 Nov 2022 at 20:25

Dear Daniel, I would like to start by thanking you for your work.

I'm just starting to automate strategies in MetaTrader and your articles are very educational, helping me at this stage.

I'm already waiting for the next article, which should deal with automating orders.

I have a simple strategy that has only been implemented in R language to test profitability performance.

With your material, I intend to code it for MQL5.

Congratulations once again ;-)

![Revisiting Murray system](https://c.mql5.com/2/51/murrey_system_avatar.png)[Revisiting Murray system](https://www.mql5.com/en/articles/11998)

Graphical price analysis systems are deservedly popular among traders. In this article, I am going to describe the complete Murray system, including its famous levels, as well as some other useful techniques for assessing the current price position and making a trading decision.

![Creating an EA that works automatically (Part 04): Manual triggers (I)](https://c.mql5.com/2/50/aprendendo_construindo_004_avatar.png)[Creating an EA that works automatically (Part 04): Manual triggers (I)](https://www.mql5.com/en/articles/11232)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode.

![Category Theory in MQL5 (Part 3)](https://c.mql5.com/2/52/Category-Theory-part3-avatar.png)[Category Theory in MQL5 (Part 3)](https://www.mql5.com/en/articles/12085)

Category Theory is a diverse and expanding branch of Mathematics which as of yet is relatively uncovered in the MQL5 community. These series of articles look to introduce and examine some of its concepts with the overall goal of establishing an open library that provides insight while hopefully furthering the use of this remarkable field in Traders' strategy development.

![Population optimization algorithms: Invasive Weed Optimization (IWO)](https://c.mql5.com/2/51/invasive-weed-avatar.png)[Population optimization algorithms: Invasive Weed Optimization (IWO)](https://www.mql5.com/en/articles/11990)

The amazing ability of weeds to survive in a wide variety of conditions has become the idea for a powerful optimization algorithm. IWO is one of the best algorithms among the previously reviewed ones.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/11237&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069172977230086442)

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