---
title: Creating an EA that works automatically (Part 10): Automation (II)
url: https://www.mql5.com/en/articles/11286
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:08:28.153983
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/11286&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069152812358631657)

MetaTrader 5 / Trading


### Introduction

In a previous article [Creating an EA that works automatically (Part 9): Automation (I)](https://www.mql5.com/en/articles/11281), we looked at how to create a breakeven and trailing stop system that uses two different modes. One of them uses a stop line on OCO positions, while the other uses a pending order as a stop level. As I explained earlier, each of these methods has its pros and cons.

Even though the EA uses a fairly simple automation system and provides detailed instructions for making calculations, it is not truly automated. Although it already has the first level of automation, it is still operated manually, or more precisely, in a "semi-manual" way. This is because the part that is responsible for moving the line or the stop order is performed by the EA itself based on the settings specified by the user.

In this article, we will look at how to add one more level of automation to the EA so that it can stay all the time on the chart where it will be used. But don't worry, this does not mean that it will send orders or open positions continuously 24 hours a day. We will see how to control the trading volumes of the system in order to prevent overtrading by the EA and not to let it trade more than the maximum allowed volume.

The type of automation that I will show here is what many platforms offer as a way for a trader not to trade all the time. You define the time during which order sending or position opening is allowed , and during this predefined time, you can use the platform basically and theoretically.

I say that the use of the platform at that time will be theoretical, because you can disable the system, so that the whole methodology goes down the drain. Every person knows when to take a break, but, unlike a human trader, an EA does not behave this way.

I must point out that you must double-check everything that we are adding or deleting from the EA so that it can work within the predefined methodology. However, we should never let the EA run unsupervised. Always remember: NEVER ALLOW AN EA RUN UNSUPERVISED.

How to implement schedule control?

There are various ways to do this. The implementation of this type of control depends more on how the programmer wants to place it in the EA code than on which kind of programming needs to be done. Since the purpose of these articles is to show how to create an automated EA in the simplest possible way, we will create code that can be easily removed. At the same time, we can use it in a manual EA if we want to control our own trading method, but this should be decided by the user.

### Planning

The development system is interesting in that each trader comes to think of different ways and means of promoting a certain concept or even how the system itself will work. Even when the results are very similar or close to each other, the way they are implemented may differ.

However, here in MQL5, what I miss is the form of object-oriented programming present in C++, which is the so-called multiple inheritance. However, if this methodology is applied incorrectly, there can be serious problems. Therefore, when using multiple inheritance, you should be very careful when programming. But even without this C++ feature, we can generate some types of code keeping things within the inheritance system.

To understand the difference between using multiple inheritance and not using it, look at the images below. Please note that C\_ControlOfTime is the name of the class that we will use to control the time interval allowed to run the EA.

![Figure 01](https://c.mql5.com/2/48/001__7.png)

Figure 01. Modeling with multiple inheritance

![Figure 02](https://c.mql5.com/2/48/002__2.png)

Figure 02. Modeling without multiple inheritance

Note that the difference between Figure 01 and Figure 02 is that in the first figure, the C\_Manager class derives the methods implemented in the C\_Orders and C\_ControlOfTime classes through inheritance, which causes the C\_Manager class to grow rapidly. Since we cannot do that in MQL5, we will use another approach shown in Figure 02. Class C\_ControlOfTime there is inherited from C\_Orders.

But why don't we do the opposite? This is because I don't want the EA to have direct access to the C\_Orders class. However, we will need access to the implementation of the C\_ControlOfTime class. The best thing about programming is that we can often take different approaches but end up with exactly the same functionality as some other programmer could create.

What I show here is just one of many possible ways to achieve the same result. What really matters is the result. It doesn't really matter how you achieve this, as long as the integrity of your code is preserved. You can create your own techniques and ways of implementing certain things, since programming allows us to do this.

### Some more details to look at before implementation

After defining the idea of using the class modeling shown in Figure 02, we moved on to the second planning stage, in which we will create the time slot control class.

Now we need to define how the working time range will be determined, i.e. how a trader can easily set this schedule. One way would be to use a file containing time range data that the EA will stick to.

However, the use of a file is rather controversial for this sort of thing. By using a file, we can end up giving the trader more freedom to define multiple time intervals within the same day. This may be reasonable in some cases but can end up making a simple task much more difficult. The EA can perform badly in certain time intervals, giving us a lot of stress.

On the other hand, the ways the schedule is defined in a file can end up complicating such a simple task for most traders. This is due because in some cases the EA will work only within one time interval.

In the vast majority of cases, this will be a much more common fact than it may seem. So, we can do something a little better, something that allows us to stay in the middle ground. The MetaTrader 5 platform allows us to save and load the desired settings from the file. So, you would only have to create a configuration for a given period. For example, you can have a configuration to be used in the morning, and another for the afternoon. To block the EA by the time control system, for example, when the trader is going to rest a little, the trader can upload a settings file directly to the MetaTrader 5 platform, and the configuration will be maintained by the platform itself. This in my view is of great help, as it saves us the trouble of creating additional configuration files just for this purpose.

![Figure 03](https://c.mql5.com/2/48/003__2.png)

Figure 03. EA settings

Figure 03 shows the EA setup system. After configuring the EA, you can save the settings using the < SAVE > button. When you need to upload the saved configuration, use the < OPEN > button. Thus, we get a system which requires much less code while making the entire code very reliable. Some of the work will be done by the MetaTrader 5 platform itself, which will allow us to save on testing to make sure everything will work correctly.

The last detail to be determined concerns orders or positions outside the time interval. What can we do with them? The EA will not be able to send orders to open a position or place an order outside the allowed time interval. However, out of the working range, the EA will be able to manage the order or close the position that are already on the server. You can disable or change the policy I used if you want. I leave the choice to you to set this according to your own trading policy.

### The birth of the C\_ControlOfTime class

The first thing to do inside the C\_ControlOfTime.mqh header file is to create the following code:

```
#include "C_Orders.mqh"
//+------------------------------------------------------------------+
class C_ControlOfTime : protected C_Orders
{
        private :
                struct st_00
                {
                        datetime Init,
                                 End;
                }m_InfoCtrl[SATURDAY + 1];
//+------------------------------------------------------------------+
        public  :

//... Class functions ...

};
```

We are adding the C\_Orders.mqh header file to have access to the C\_Orders class. So, C\_ControlOfTime will be inherited from the C\_Orders class using the protected method. I already explained the consequences of using this type of inheritance in another article in this series: " [Creating an EA that works automatically (Part 05): Manual triggers (II)](https://www.mql5.com/en/articles/11237)".

Now, inside the private part of the control class code, we add a structure that will be used as an array with 7 elements. But why not define 7 instead of using that crazy statement? This is because the value of **SATURDAY** is defined internally in the MQL5 language as a value of the [ENUM\_DAY\_OF\_WEEK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) enumeration. So, it becomes clearer why we use the days of the week to access the array.

This is the language-level enrichment because, to those who read the code, a word is more expressive than a numeric value. This structure only has two elements: one indicates the EA operation start point, and the other on the point after which the EA can no longer work, both are of type **datetime**.

Once the array is defined, we can move on to our first code in the class, which is the class constructor shown below:

```
                C_ControlOfTime(const ulong magic)
                        :C_Orders(magic)
                        {
                                ResetLastError();
                                for (ENUM_DAY_OF_WEEK c0 = SUNDAY; c0 <= SATURDAY; c0++) ZeroMemory(m_InfoCtrl[c0]);
                        }
```

To many people, the following code may seem strange, but it's just a form of programming that's a little more advanced than what we're used to. The fact that I say that it is high-level programming has nothing to do with my experience in programming. As I said:

_"It's easier to understand code that uses a natural language than code that uses numeric values."_

The loop in this constructor is self-explanatory. Such things actually define whether the code is high level or low level.

I believe that it will be quite easy for you to understand this constructor. Previously I already explained how the code in the constructor works. If you are not sure, please read the previous articles in this series. The only difference here is actually the loop where we indicate the variable, which will start with the value **SUNDAY** and will end with **SATURDAY**. There is nothing else complicated here.

The loop works correctly only because in the enumeration Sunday is defined as the first day of the week, and Saturday is the last one. However, if **MONDAY** was set as the first day of the week, the loop would fail when executing the code in the MetaTrader 5 platform and would throw an error. Therefore, it is important to be careful when using high-level code, because if configured incorrectly, the code can generate several **runtime** errors.

Once this is done, we can move on to the next function in our class:

```
virtual void SetInfoCtrl(const ENUM_DAY_OF_WEEK index, const string szArg) final
                        {
                                string szRes[];
                                bool bLocal;

                                if (_LastError != ERR_SUCCESS) return;
                                if ((index > SATURDAY) || (index < SUNDAY)) return;
                                if (bLocal = (StringSplit(szArg, '-', szRes) == 2))
                                {
                                        m_InfoCtrl[index].Init = (StringToTime(szRes[0]) % 86400);
                                        m_InfoCtrl[index].End = (StringToTime(szRes[1]) % 86400);
                                        bLocal = (m_InfoCtrl[index].Init <= m_InfoCtrl[index].End);
                                        if (_LastError == ERR_WRONG_STRING_DATE) ResetLastError();
                                }
                                if ((_LastError != ERR_SUCCESS) || (!bLocal))
                                {
                                        Print("Error in the declaration of the time of day: ", EnumToString(index));
                                        ExpertRemove();
                                }
                        }
```

This part requires more detailed analysis, since it is not so simple for those with little experience. And I know this because I was in your shoes, my dear reader. To explain the above code in more detail, let's break it down into smaller parts.

```
if (_LastError != ERR_SUCCESS) return;
if ((index > SATURDAY) || (index < SUNDAY)) return;
```

Whenever possible, you should check the system for errors, as this code is very sensitive and subject to errors due to the nature of the thing we are working with. We start by checking if there is any failure before this specific function is called. If there is any failure, the function will be closed immediately.

The second check is also needed to avoid random failures. If for some reason we make a call depending on the day of the week indicating some number, and the processor treats is only as a number because it is not in the range between **SATURDAY** and **SUNDAY**, we won't continue executing further lines within the function.

Once this first step was taken, it was accepted and the code passed the first tests, we will make a translation of the content passed by the caller:

```
if (bLocal = (StringSplit(szArg, '-', szRes) == 2))
{
        m_InfoCtrl[index].Init = (StringToTime(szRes[0]) % 86400);
        m_InfoCtrl[index].End = (StringToTime(szRes[1]) % 86400);
        bLocal = (m_InfoCtrl[index].Init <= m_InfoCtrl[index].End);
        if (_LastError == ERR_WRONG_STRING_DATE) ResetLastError();
}
```

Here we have much more interesting code, especially for those who are starting to learn programming. Here we use the [StringSplit](https://www.mql5.com/en/docs/strings/stringsplit) function to "split" the information we receive from the caller into two parts. The character that indicates where the information will be broken, is the minus sign ( - ). What we want and expect to see are two pieces of information. If we get a different number, this will be considered an error, as can be seen in the examples below:

```
12:34 - - 18:34 <-- This is an error
12:32  -  18:34 <-- Information is correct
12:32     18:34 <-- This is an error
       -  18:34 <-- This is an error
12:34  -        <-- This is an error
```

The internal content of the information does not matter for the StringSplit function. It will "split" the data based on the specified separator. This function is very useful in certain cases, therefore it is important to properly study it. Because it helps a lot to separate information within a string of characters.

After getting these two pieces of information, we use the [StringToTime](https://www.mql5.com/en/docs/convert/stringtotime) function to translate them into a date format code.

There is another important detail: the information that we will be providing only contains hours and minutes. We are not interested in the particular date. But nothing prevents us from specifying a specific date. However, according to the implementation, the date will be ignored. You only need to enter the name. The StringToTime function will automatically add the current date. This is actually not a problem, but there is one thing that we will consider later.

To eliminate the current date value which was added by the StringToTime function, we use factorization, in which the result will be just the time value specified by the caller. In case you really don't want to remove the date added by the StringToTime function, which will always be the current date, just remove this indicated factorization.

We have the first test after the value has been converted. It is very important in order to avoid problems later. We will check if the start time is less than or the same as the specified end time. If this check is successful, we will not have to worry about this problem later. If the check fails, we will notify the user that the values are not suitable.

We also have another check, because since we are not specifying the date, the platform will generate a **runtime** error. This type of error is very irritating, but we deal with it in a very proper way. When such an error is detected, we will simply remove the indication that it happened. Since we know in advance that a date will not be informed, this will generate this error.

**Important note:** Whenever, while testing the EA, you notice that a runtime error is triggered, which does not violate the EA integrity and not making it unstable or insecure, add this type of test after the code that generates the error to minimize the error generation in the future. Some runtime errors can be ignored as they are not so critical and do not affect the EA operation in any way. However, some errors must be handled in different ways because they cause the EA to stop moving on the chart. This is called imposed robustness, because we understand that a failure can occur, but we also know that it will not compromise the system.

After implementing value conversion, we have the following code:

```
if ((_LastError != ERR_SUCCESS) || (!bLocal))
{
        Print("Error in the declaration of the time of day: ", EnumToString(index));
        ExpertRemove();
}
```

Something important can be noticed here: if the system generates any serious errors during value conversion, this constant variable will have a value different from **ERR\_SUCCESS**. This indicates that we cannot trust data that has been used or entered by the user. The same is true if the variable here has the 'false' value which indicates that some point caused it to fail. Anyway, we will inform the trader by printing a message in the terminal. Also, a request to close the EA will be generated in the MetaTrader 5 platform.

I hope the explanation of how the function works is quite clear, since it's been explained in detail. However, this is not all, we have another function to discuss:

```
virtual const bool CtrlTimeIsPassed(void) final
                        {
                                datetime dt;
                                MqlDateTime mdt;

                                TimeToStruct(TimeLocal(), mdt);
                                dt = (mdt.hour * 3600) + (mdt.min * 60);
                                return ((m_InfoCtrl[mdt.day_of_week].Init <= dt) && (m_InfoCtrl[mdt.day_of_week].End >= dt));
                        }
```

It's important not to take accept something for granted until you've tried all possible variations which can have the same results but in a different, much simpler way. It may seem discouraging, but it is a valuable approach to problem solving and finding better solutions.

Before getting to the point, let's understand the above function. It captures the local time and date and converts those values into a structure in order to split the data. Then we set up a value that will contain only hours and minutes — these are the values we used in the previous function. This is the problem I mentioned earlier. If you didn't remove the date value in the conversion function, here you would need to add the date value. Once we have these values, we will check to see if they are within the range. If they are, we will return true, if they are outside the defined range for the day, we will return false.

Now comes another question. Why do we have to do all this work of capturing, reconstituting, to check whether or not the local time is within a range of day values? Wouldn't it be simpler to use a code, where we capture the local time and check it against the defined range? Yes, it would indeed be much simpler. But, and there is always a but, we have a small problem: **the day of the week**.

If you were to set the EA operation range based only on the current day, that's fine. Just knowing the local time would be enough. But we define values for seven days of the week. And the simplest way to know what day of the week is to use exactly all the work that was done in the above function.

So really, it all depends on how you're implementing the system. If you are implementing it to be simple, the functions and procedures you need to create and define will be considerably simpler. If you are creating a system for wider use, the functions and procedures will be considerably more complicated.

That is why it is important to think and analyze before actually starting to code. Otherwise, you may end up at a dead end, where a newly built code makes older code unable to meet the requirements. So, you have to change older codes, and next thing you know, you're all wrapped up in such a mess, needing to discard everything and start from scratch.

Before changing the subject, I would like to emphasize one detail from the above function. If you want the EA to stay on 24 hours a day, with the platform operating during all this time, you may be afraid that the EA, at the turn of the day, will not know when to start operating again. This in fact will not happen, since at each call to the function it will reevaluate the whole situation and will use the current day of the week in order to verify whether the EA will be able to perform some operation or not.

For example, suppose you tell the EA that it can trade on Monday between 04:15 and 22:50 and on Tuesday between 3:15 and 20:45. You can just turn it on Monday and leave it running until Tuesday. As soon as the day turns, switching from Monday to Tuesday, it will automatically start checking which is the period allowed to operate on Tuesday. Because of this, I decided to use the week mode instead of a definition based on the current day.

Maybe I missed some details of this class, but I don't want to overcomplicate the explanation of how the functions work. Let's take a closer look at a very important detail when dealing with function inheritance. If you look carefully, you will see that both the **SetInfoCtrl** and the **CtrlTimeIsPassed** functions have very strange declarations. Why do they have such declarations? What is their purpose? These declarations are highlighted below:

```
virtual void SetInfoCtrl(const ENUM_DAY_OF_WEEK index, const string szArg) final
//+------------------------------------------------------------------+
virtual const bool CtrlTimeIsPassed(void) final
//+------------------------------------------------------------------+
```

Here, each of the words has a reason. Nothing here is being placed to decorate the code, although some people are used to doing this, but that would be another story.

What really matters in these declarations is the reserved word **"final"**. This is the big question from the existence of the word **"virtual"** in the declaration. The fact is that, when you create a class, you can work on it in several ways, overriding methods, modifying the way a function of the parent class will be executed in a child class, creating new forms based on more primitive work, and more. When you add the 'final' word to a function declaration in a class, as shown above, you tell the compiler that a child class cannot modify it in any way. It cannot even override the function written in the parent class, which is very common.

I will repeat once again, because it is important: by using the 'final' word in the declaration, you are telling the compiler that a child class cannot modify it in any way, not even override, the inherited function that is written in the parent class, which received the 'final' word in the declaration.

Thus, we guarantee that if any class, which inherits this class from here, with its methods and variables, tries to modify these methods, this attempt will be considered an error, and the program will not be compiled. The 'virtual' word serves precisely to promote the possibility of modification, but the final word prevents such a change. Who really makes the rules is the final word. So, whenever you want to ensure that a function does not undergo undue modification within a child class, lock this possibility of modification by adding a declaration, as shown above.

This will save you a lot of headaches if you are working with many classes and with a deep level of inheritance.

### Linking C\_ControlOfTime with C\_Manager and using it in the EA

Now, finally, we can link the C\_ControlOfTime class with the C\_Manager class, and the EA will have a new type of work parameter. To do this, we add the following change into the C\_Manager class code:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "C_Orders.mqh"
#include "C_ControlOfTime.mqh"
//+------------------------------------------------------------------+
#define def_MAX_LEVERAGE                10
#define def_ORDER_FINISH                false
//+------------------------------------------------------------------+
class C_Manager : private C_Orders
class C_Manager : public C_ControlOfTime
```

The crossed out lines were removed from the original code, and new lines appeared in their place. But notice the fact that the C\_Manager class will publicly inherit the C\_ControlOfTime class. We simply add everything from the C\_ControlOfTime class into the C\_Manager class. This extends the capabilities of the C\_Manager class without increasing its code. If we no longer need the capabilities which were added by inheriting the C\_ControlOfTime class, all we need to do is remove this inheritance and any possible reference points from the C\_Manager class. That simple.

Thus, we do not change the reliability level of the C\_Manager class, because it will continue to function with maximum security, stability and robustness, as if nothing had happened. If the C\_ControlOfTime class starts to cause instabilities in the C\_Manager class, we can simply remove the C\_ControlOfTime class, and the C\_Manager class will be stable again.

I think this explains why I love creating everything in the form of classes, and not as scattered functions. Things are increasing and improving very quickly, and we always have the highest possible level of stability and reliability that the language can provide us.

Now, since the constructors need to be referenced somehow, let's look at the new class constructor below:

```
//+------------------------------------------------------------------+
                C_Manager(const ulong magic, double FinanceStop, double FinanceTake, uint Leverage, bool IsDayTrade, double Trigger)
                        :C_ControlOfTime(magic),
                        :C_Orders(magic),
                        m_bAccountHedging(false),
                        m_TicketPending(0),
                        m_Trigger(Trigger)
                        {
                                string szInfo;

                                ResetLastError();
                                ZeroMemory(m_Position);
                                m_InfosManager.FinanceStop = FinanceStop;

// ... The rest of the constructor code....

                        }
//+------------------------------------------------------------------+
```

Here we have the same as with declarations. The crossed out line has been removed. In its place came a new line which passes the data to the C\_ControlOfTime class constructor. This constructor will reference the constructor of the C\_Orders class, so that it receives the magic number required to send the orders.

Now, to finish this topic, below are the points with the real use of time control:

```
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                if (!CtrlTimeIsPassed()) return;
                                if ((m_StaticLeverage >= def_MAX_LEVERAGE) || (m_TicketPending > 0) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                m_TicketPending = C_Orders::CreateOrder(type, Price, (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceStop), (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;

                                if (!CtrlTimeIsPassed()) return;
                                if ((m_StaticLeverage >= def_MAX_LEVERAGE) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                tmp = C_Orders::ToMarket(type, (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceStop), (def_ORDER_FINISH ? 0 : m_InfosManager.FinanceTake), m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                        }
//+------------------------------------------------------------------+
```

No other function within the EA or within the C\_Manager class will use the time ranges. But if you wish, you can add this control, by adding exactly the highlighted line to the functions that need to be controlled, such as the trailing stop trigger. If you don't want to use the control in some of these functions, just remove the line that was added. Did you like the way the code was planned? But all this doesn't really work without another point in the EA's code.

The only really necessary changes in the EA code can be seen below:

```
#include <Generic Auto Trader\C_Manager.mqh>
#include <Generic Auto Trader\C_Mouse.mqh>
//+------------------------------------------------------------------+
C_Manager *manager;
C_Mouse  *mouse;
//+------------------------------------------------------------------+
input int       user01   = 1;                   //Leverage Factor
input double    user02   = 100;                 //Take Profit ( FINANCE )
input double    user03   = 75;                  //Stop Loss ( FINANCE )
input bool      user04   = true;                //Day Trade ?
input color     user05  = clrBlack;             //Price Line Color
input color     user06  = clrForestGreen;       //Take Line Color
input color     user07  = clrFireBrick;         //Stop Line Color
input double    user08  = 35;                   //BreakEven ( FINANCE )
//+------------------------------------------------------------------+
input string    user90  = "00:00 - 00:00";      //Sunday
input string    user91  = "09:05 - 17:35";      //Monday
input string    user92  = "10:05 - 16:50";      //Tuesday
input string    user93  = "09:45 - 13:38";      //Wednesday
input string    user94  = "11:07 - 15:00";      //Thursday
input string    user95  = "12:55 - 16:25";      //Friday
input string    user96  = "00:00 - 00:00";      //Saturday
//+------------------------------------------------------------------+
#define def_MAGIC_NUMBER 987654321
//+------------------------------------------------------------------+
int OnInit()
{
        string szInfo;

        manager = new C_Manager(def_MAGIC_NUMBER, user03, user02, user01, user04, user08);
        mouse = new C_Mouse(user05, user06, user07, user03, user02, user01);
        for (ENUM_DAY_OF_WEEK c0 = SUNDAY; c0 <= SATURDAY; c0++)
        {
                switch (c0)
                {
                        case SUNDAY     : szInfo = user90; break;
                        case MONDAY     : szInfo = user91; break;
                        case TUESDAY    : szInfo = user92; break;
                        case WEDNESDAY  : szInfo = user93; break;
                        case THURSDAY   : szInfo = user94; break;
                        case FRIDAY     : szInfo = user95; break;
                        case SATURDAY   : szInfo = user96; break;
                }
                (*manager).SetInfoCtrl(c0, szInfo);
        }
        (*manager).CheckToleranceLevel();
        EventSetMillisecondTimer(100);

        return INIT_SUCCEEDED;
}
```

It was only necessary to add the points where the user interacts with the code.This loop will capture the data and throw it into the system, so that it can be used by the C\_Manager class, thus controlling the actual EA operation time. In this part of the code the behavior is easy to understand and does not require any additional explanation.

### Conclusion

In this article, I showed you how to add a control in order to allow the EA to operate within a certain time range. Although the system is quite simple, it needs one last explanation.

If you enter a time greater than 24 hours in the interaction system, it will be corrected for the time closest to 24 hours. That is, if you want the EA to operate until 22:59 (if you work on Forex), you should be careful to specify exactly this value. If you type 25:59, the system will change it to 23:59. Although this typing error is not common, it can happen.

I have not added any additional check to analyze this situation because it rarely happens, but I wanted to comment on this and show a possible check for such a condition. It can be seen below. The attached code already has these changes.

```
virtual void SetInfoCtrl(const ENUM_DAY_OF_WEEK index, const string szArg) final
                        {
                                string szRes[], sz1[];
                                bool bLocal;

                                if (_LastError != ERR_SUCCESS) return;
                                if ((index > SATURDAY) || (index < SUNDAY)) return;
                                if (bLocal = (StringSplit(szArg, '-', szRes) == 2))
                                {
                                        m_InfoCtrl[index].Init = (StringToTime(szRes[0]) % 86400);
                                        m_InfoCtrl[index].End = (StringToTime(szRes[1]) % 86400);
                                        bLocal = (m_InfoCtrl[index].Init <= m_InfoCtrl[index].End);
                                        for (char c0 = 0; (c0 <= 1) && (bLocal); c0++)
                                                if (bLocal = (StringSplit(szRes[0], ':', sz1) == 2))
                                                        bLocal = (StringToInteger(sz1[0]) <= 23) && (StringToInteger(sz1[1]) <= 59);
                                        if (_LastError == ERR_WRONG_STRING_DATE) ResetLastError();
                                }
                                if ((_LastError != ERR_SUCCESS) || (!bLocal))
                                {
                                        Print("Error in the declaration of the time of day: ", EnumToString(index));
                                        ExpertRemove();
                                }
                        }
```

It was necessary to add a new variable and the highlighted code that splits the time data in order to check if the entered time is below the maximum possible value within the 24-hour interval.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11286](https://www.mql5.com/pt/articles/11286)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11286.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_10.zip](https://www.mql5.com/en/articles/download/11286/ea_automatico_-_10.zip "Download EA_Automatico_-_10.zip")(10.75 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/446404)**
(12)


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
8 Jan 2023 at 11:11

**napalermo [#](https://www.mql5.com/pt/forum/438125#comment_44226953):**

Thank you very much, I'll follow those who are already here.

and seeing these TapeReading indicators made me very curious.

people used to say that you couldn't get real volume like on the b3 platforms.

i'm going to test it here.

and there's plenty of study material...lol

Let's study.

Although B3 provides two types of volume, some platforms create a third. B3 only tells us the volume traded, don't confuse this with financial volume, as they are totally different things, although they are correlated. And also the tick volume, which indicates the number of trades executed in a given period of time. This type of thing will be better understood in the next sequence of articles that I'll be posting soon. Other platforms create a third type of volume, which is financial volume, but financial volume is nothing more than taking the volume traded and multiplying it by the value of each trade, something trivial. So MetaTrader 5 doesn't include this volume in its indicators ... Although there is a discrepancy in the Times & Trade between MetaTrader and other platforms such as Profit Chart, this discrepancy is due to the type of information that other platforms use, since MetaTrader 5 ignores the information regarding the house of origin. This information is provided by B3, but the structure of MetaTrader 5 ignores it because it's of no use to traders, so some platforms merge some of the data, which creates the discrepancies in the information in Times & Trade, but in general terms the basic information remains the same, so it can be used to trade ... 😁👍

![Santiago Fallas](https://c.mql5.com/avatar/2023/5/6452b202-5a66.jpg)

**[Santiago Fallas](https://www.mql5.com/en/users/shaggo87)**
\|
4 May 2023 at 23:14

Ok thank you very much


![crt6789](https://c.mql5.com/avatar/avatar_na2.png)

**[crt6789](https://www.mql5.com/en/users/crt6789)**
\|
13 May 2023 at 13:01

Is this using the local time of the computer? Does it still have to be converted to the [server time](https://www.mql5.com/en/docs/dateandtime/timetradeserver "MQL5 documentation: TimeTradeServer function") shown on the trading charts in order to correspond to the trading charts?


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
14 May 2023 at 19:25

**crt6789 [#](https://www.mql5.com/zh/forum/447121#comment_46870777) :**

Esta é a hora local do computador usada? Ele precisa ser convertido no horário do servidor exibido no gráfico de negociação para corresponder ao gráfico de tendências de negociação?

The time to indicate is the local time... But you can change it to server time by changing one line in the code...

![crt6789](https://c.mql5.com/avatar/avatar_na2.png)

**[crt6789](https://www.mql5.com/en/users/crt6789)**
\|
20 May 2023 at 06:25

**Daniel Jose [#](https://www.mql5.com/zh/forum/447121#comment_46883895):**

The time to indicate is local time... But you can change it to server time by changing one line in the code...

May I ask: How should I rewrite the daily trading hours if I want to segment them? For example, into three segments: morning session, afternoon session, night session. Or, Asian session, European session, American session. Morning session 03:00 - 06:30, afternoon session 08:00 - 11:30, night session 15:30 - 23:00. -23:00


![Population optimization algorithms: Saplings Sowing and Growing up (SSG)](https://c.mql5.com/2/52/growing-tree-avatar.png)[Population optimization algorithms: Saplings Sowing and Growing up (SSG)](https://www.mql5.com/en/articles/12268)

Saplings Sowing and Growing up (SSG) algorithm is inspired by one of the most resilient organisms on the planet demonstrating outstanding capability for survival in a wide variety of conditions.

![Creating an EA that works automatically (Part 09): Automation (I)](https://c.mql5.com/2/50/aprendendo_construindo_009_avatar.png)[Creating an EA that works automatically (Part 09): Automation (I)](https://www.mql5.com/en/articles/11281)

Although the creation of an automated EA is not a very difficult task, however, many mistakes can be made without the necessary knowledge. In this article, we will look at how to build the first level of automation, which consists in creating a trigger to activate breakeven and a trailing stop level.

![Creating an EA that works automatically (Part 11): Automation (III)](https://c.mql5.com/2/50/aprendendo_construindo_011_avatar.png)[Creating an EA that works automatically (Part 11): Automation (III)](https://www.mql5.com/en/articles/11293)

An automated system will not be successful without proper security. However, security will not be ensured without a good understanding of certain things. In this article, we will explore why achieving maximum security in automated systems is such a challenge.

![Experiments with neural networks (Part 4): Templates](https://c.mql5.com/2/52/neural_network_experiments_004_avatar.png)[Experiments with neural networks (Part 4): Templates](https://www.mql5.com/en/articles/12202)

In this article, I will use experimentation and non-standard approaches to develop a profitable trading system and check whether neural networks can be of any help for traders. MetaTrader 5 as a self-sufficient tool for using neural networks in trading. Simple explanation.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/11286&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069152812358631657)

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