---
title: Creating an EA that works automatically (Part 07): Account types (II)
url: https://www.mql5.com/en/articles/11256
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:09:01.659051
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/11256&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069162858287137035)

MetaTrader 5 / Trading


### Introduction

In the previous article [Creating an EA that works automatically (Part 06): Account types (I)](https://www.mql5.com/en/articles/11241), we started developing a way to ensure that the automated EA works correctly and within its intended purpose. In that article, we created the C\_Manager class, which acts as an administrator, so that in case of strange or incorrect EA behavior the EA will be removed from the chart.

We started that article by explaining how to prevent pending or market orders of the EA from triggering. Although the mechanism shown there is capable of supporting the EA, we have some other problems related to the interaction between the EA and the C\_Orders class. The problem is mainly related to NETTING accounts. This is one of the topics to be covered in this article.

**However, one way or another, you should NEVER let the automated EA run unsupervised.**

Don't expect that just intelligently programming the EA is enough, because it is not. You should always be aware of what the automatic EA is doing, and if it goes beyond its designated activity, remove it from the chart as soon as possible, before it gets out of control.

### New routines for interaction between the EA and the C\_Orders class

Everything that was said in the previous articles will be worthless if the EA continues to be able to access the C\_Orders class as shown below:

```
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                C_Orders::CreateOrder(type, Price, m_InfosManager.FinanceStop, m_InfosManager.FinanceTake, m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                C_Orders::ToMarket(type, m_InfosManager.FinanceStop, m_InfosManager.FinanceTake, m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
```

Note that even when we try to limit and control things, the EA still manages to send orders to the market and send new orders to the order book. This is a security breach, because if during the EA initialization the C\_Manager class managed to impose that it will follow certain rules, then why after that we allow it to be able to place orders and perform market operations? We must set some limit here, even if it is very simple. One simple test often prevents many kinds of potential problems. So, let's add some control over this whole area in the form of an interaction between the EA and the order system present in the C\_Orders class.

```
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                if ((m_TicketPending > 0) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                m_TicketPending = C_Orders::CreateOrder(type, Price, m_InfosManager.FinanceStop, m_InfosManager.FinanceTake, m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;

                                if (m_bAccountHedging && (m_Position.Ticket > 0)) return;
                                tmp = C_Orders::ToMarket(type, m_InfosManager.FinanceStop, m_InfosManager.FinanceTake, m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                                if (PositionSelectByTicket(m_Position.Ticket)) SetInfoPositions(); else m_Position.Ticket = 0;
                        }
//+------------------------------------------------------------------+
```

We have added some rules so that the EA does not have so much freedom in using the C\_Orders class. Let's see what is happening here. Let's start with the **CreateOrder** function. If we have a value in the variable that informs the pending order ticket, a new order will be denied, but it will also be denied if we are on a HEDGING account with an open position. That simple. But if these conditions allow, we can send the a request to add a pending order to the order book, and if the request is successful, we will get the ticket of the order that will have been placed. After that we will not be able to send a new order.

Now comes a big detail, which is in the **ToMarket** function. In this case, the order is placed as a market order, i.e. at the best available price. Let's now look at the conditions which allow us to send a market order.Sending an order will only be allowed if we do not have an open position on a HEDGING account.Here is the detail:If it is possible to send a market order, this will be done, but we cannot use the value returned by the call in the first place. This is due to the fact that the value may differ from the position ticket if we are on the NETTING account. If you set the value directly in the value of the position ticket, you may lose the actual ticket value, because it is possible that in this case the return value will only be temporary. This is the reason why we have this new check here.If this is a HEDGING account, then the returned value can be safely stored in the position ticket. But in the case of a NETTING account, it will be saved only if the position ticket is zero; otherwise, the value will be ignored. Once this is done,we can update position data or reset the position ticket value, but it all depends on the check.

### Problems with the NETTING account in the automated EA

There is a potentially dangerous issue when using an automated EA on a NETTING account. In the previous article, I showed a problem that could occur in the HEDGING account. But now let's explore a different problem, which only happens on NETTING accounts. If the EA does not have a lock, it will at some point get to this loophole, which many EAs have but their users do not know about it.

The problem is the following: On a NETTING account, the trading server will automatically create an average position price as the bid price changes in a long or sell position. In case you are going SHORT, one way or another, the average price, as well as the volume of the position, will change.

For an EA which will be operated manually, there is no problem here, as the trader will be responsible for sending any requests to the server. The problem occurs with an automated EA. In this case, it can trigger when there is a certain gap or vulnerability present in its programming. Detail: The problem is not in the trading server, nor in the MetaTrader 5 platform; the problem is in the EA.

I will try to explain in a way that everyone can understand, but ideally, you should have some knowledge of how the market works, so that it is easier to understand. But I will try to make the question as clear as possible.

When the automated EA sends a buy or sell order in order to open a position, it does so based on some kind of trigger. Although we haven't talked about triggers yet, we'll get there soon. But the problem is not the trigger, as in the vast majority the trigger is not responsible.

Once the position is open, and this in a NETTING account, the EA, for one reason or another, can start using the following strategy: close a part of the volume, and open another. In this way, the average price will start to move. The problem is, it may be that the EA starts to do this in a wild, crazy and hallucinated way. If this happens, it can lose large sums of money in a matter of minutes, sometimes seconds, depending on the volume traded.

And this is the problem: the EA launches an order, and then it stays there without closing the position, sending new requests to the server in order to change the average price. This often happens without the trader, who is or should be supervising the EA, since the volume shown does not change, but the price does, and some traders do not notice this change. It can often happen that the position is already causing a big loss, before the trader actually notices that there is something wrong there.

Although some automated EAs are deliberately programmed to use this type of strategy, some people do not know or realize that there is this type of error in their automated EA. In some cases, and here comes the question of the trigger, the person may be using a type of trigger which makes the EA buy or sell and in some very specific market situation, and the EA may come to do exactly what was described above. If aware of this fact, the trader will be attentive. But if this behavior is not expected, the trader that will be supervising the EA might end up getting scared by what the EA will be doing.

Do not underestimate the system, no matter how much you trust it and believe that your automated EA is safe. **Do not underestimate it.** Because it may happen so that the EA has the loophole that you have not foreseen while programming. But fortunately, there is a relatively simple way to overcome this flaw, at least to minimize its damage a little. So, let's see how to do this in terms of programming.

### Limiting Trading Volume in the EA

The simplest way to reduce, at least a little, the problem described in the previous topic, is to limit the volume that the EA can actually work with. There is no 100% ideal method, but we can try, as much as possible, generate some form that can bring us some comfort. And the simplest way is to limit the volume that the EA can work with during the entire time it is running.

Pay attention to this: I am not saying to limit the volume of an open position, I am saying to limit the volume that the EA can trade for the entire period. That is, the EA can trade x times the minimum volume, but once the limit is reached, the EA will not be able to open a new position. It doesn't matter if it has a position equivalent to one minimum volume while the limit is 50 times this volume, if the EA has already reached this quota of 50, it will not be able to open more positions or increase the open position.

The lock is based precisely on this. But we have a detail here: There is no such thing as a 100% secure lock. Everything will depend on the trader who is supervising the EA. If the trader inadvertently turns off the lock, restarting the EA, the trader will be responsible for any failure that may occur. As it may have happened that the EA reached the quota and the trader came to restart the EA back on the chart. In this case, there is no lock.

To implement the lock, we need a variable that is static, global and private, within the C\_Manager class. It is shown in the code below:

```
static uint m_StaticLeverage;
```

However, to initialize a static variable inside a class, we cannot use the class constructor. Initialization in this case will be implemented outside the class body, but usually inside the source file, in the C\_Manager.mqh header file. Then, outside the class body, we will have the following code:

```
uint C_Manager::m_StaticLeverage = 0;
```

Don't be scared, it's just a static variable being initialized. For those who don't know, this initialization takes place even before the class constructor is actually referenced. So in some types of code, this is quite useful. But the fact that the variable is initialized outside the body of the class, does not indicate that it will be accessible outside the class. Remember: The reason is encapsulation, all variables must be declared as being private. This makes the class as robust as possible. We thus avoid security breaches or loss of reliability in the work of the class.

But here another important question arises: Why do we use a private and static global variable in the C\_Manager class? Couldn't we use a variable that wasn't at least static? The answer is NO. The reason is this: If, under any circumstances, the MetaTrader 5 platform restarts the EA, all data stored outside a static variable will be lost. You should pay attention to this. I mean that MetaTrader 5 can restart the EA, not that you delete it and then run again on the chart. These are two different situations. If the EA is removed from the chart and then re-launched then any information, even about static variables, will be lost. In these cases, the only way is to store the data in a file and then restore it by reading that file. Restart doesn't really mean that the EA has been deleted. This can happen in several situations, and all of them will be related to the triggering of the DeInit event, which will call the OnDeInit function in the code. This reset affects not only the EA, but also the indicators. Therefore, the scripts running on the chart can be deleted if the DeInit event is activated. Because scripts don't reset (they don't have this property), they just "leave" the chart, and MetaTrader 5 does not reset them automatically..

Now we need to add a definition in the code of the class to identify the maximum times, that the EA can trade the minimum volume. This definition is shown below:

```
#define def_MAX_LEVERAGE       10
```

We already have a place to store the minimum volume that the EA traded, and we have a definition of the maximum volume that can be used. So, now we can produce a way to calculate this volume, which is the most interesting part of the work. However, before that, we need to make some changes to the code from the previous article:

```
//+------------------------------------------------------------------+
                void CreateOrder(const ENUM_ORDER_TYPE type, const double Price)
                        {
                                if ((m_StaticLeverage >= def_MAX_LEVERAGE) || (m_TicketPending > 0) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                m_TicketPending = C_Orders::CreateOrder(type, Price, m_InfosManager.FinanceStop, m_InfosManager.FinanceTake, m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                        }
//+------------------------------------------------------------------+
                void ToMarket(const ENUM_ORDER_TYPE type)
                        {
                                ulong tmp;

                                if ((m_StaticLeverage >= def_MAX_LEVERAGE) || (m_bAccountHedging && (m_Position.Ticket > 0))) return;
                                tmp = C_Orders::ToMarket(type, m_InfosManager.FinanceStop, m_InfosManager.FinanceTake, m_InfosManager.Leverage, m_InfosManager.IsDayTrade);
                                m_Position.Ticket = (m_bAccountHedging ? tmp : (m_Position.Ticket > 0 ? m_Position.Ticket : tmp));
                                if (PositionSelectByTicket(m_Position.Ticket)) SetInfoPositions(); else m_Position.Ticket = 0;
                        }
//+------------------------------------------------------------------+
```

By adding these tests, I am saying that if the placed volume exceeds the specified maximum volume, then the EA will not be able to perform the operation. There are still some details to be resolved in the future, but for now it will be done like this. Also note that there is a line that was removed from the code because it disturbs the accounting.

Ok, now we need some routines to help us communicate with the EA so that the C\_Manager class could manage what the EA will be doing. To do this, we'll start by creating a very subtle but much-needed function, as shown in the following code:

```
inline void UpdatePosition(const bool bSwap = false)
                        {
                                int ret;

                                if ((m_bAccountHedging) && (m_Position.Ticket > 0) && (bSwap)) SetUserError(ERR_Unknown);
                                m_Position.Ticket = ((m_Position.Ticket == 0) && (bSwap) ? m_TicketPending : m_Position.Ticket);
                                m_TicketPending = (bSwap ? 0 : m_TicketPending);
                                if (PositionSelectByTicket(m_Position.Ticket))
                                {
                                        ret = SetInfoPositions();
                                        m_StaticLeverage += (ret > 0 ? ret : 0);
                                }
                        }
```

Do not underestimate this code above. Although it seems to be a low value code in terms of complexity, it is extremely necessary for the C\_Manager class to be able to manage what the EA is doing. To understand this code, it is necessary to understand how and when the EA will call it. But we will see this later, since the procedure inside the EA is quite different from what many people do.

Just by looking at the above code, we can notice a few things. If we are on a HEDGING account with an open position and **bSwap** is 'true', this will be an error. In this case, we only report the error, as it will be handled elsewhere. If the variable for the position ticket is empty and **bSwap** is 'true', this means that the pending order has become a position. But if we already had an open position, then the pending order (on the NETTING account) would change the position volume and, possibly, the entry price. Such situations are analyzed in this line.

If **bSwap** is set to true, this means that the pending order has ceased to exist, and this is corrected here. Now let's do the following check: We check if there is an open position. If there is, then we run a procedure that updates the data. This procedure calculates the difference between the volume before and after the update, and returns this volume.. If the returned data is positive, this indicates that the volume or lot volume factor has increased. This situation is common on NETTING accounts, but on HEDGING accounts the return value will always be the current lot volume. We will add this returned value to the value that was previously in the static variable. Thus, we can take into account the volume that the EA will use or has already used during its work.

But without thinking too much about it, you will soon realize that we need another routine. It will serve to remove or close a position. It is shown below:

```
inline void RemovePosition(void)
                        {
                                if (m_Position.Ticket == 0) return;
                                if (PositionSelectByTicket(m_Position.Ticket)) ClosePosition(m_Position.Ticket);
                                ZeroMemory(m_Position);
                        }
```

There is no error here: the EA will close the open position or will simply inform the C\_Manager class that the server has closed the position, either because the limits (take profit or stop loss) have been reached, or because something happened that caused the position to be closed, it doesn't matter. But if the EA mistakenly calls this function, while there is no open position, we simply roll back.If there is an open position, it will be closed, and at the end we will clear all the data existing in the memory area where the position data is stored.

Although these two function above work well, in many different situations they are not very suitable for what we actually need to do. So don't get too excited when you see these functions. Don't imagine that they are suitable for an automated EA with a great level of robustness, because as I just said, they work but they are not adequate. We need other, better functions for what we are building here. However, we will not see this in the current article, because I will need to show why to use another type of function. For a proper understanding, it is necessary to show the EA code — this will be done in the next article.

Based on all this work, we finally have a system which in theory is robust and quite reliable, to the point that we can finally start to automate it, so that it can run while only being supervised by a human trader. But pay attention that I used **IN THEORY**, because failures can still occur in the system. But now these problems will be checked with the **\_LastError** variable so that we can see if something went wrong. If the failure is serious, as, for example, when using the custom error enumeration discussed in the previous article, we must remove the EA from the chart.

Removing an EA from a chart is not a difficult task. In fact, it can be done easily by calling the [ExpertRemove](https://www.mql5.com/en/docs/common/expertremove) function. But this task is not our biggest problem. The real difficulty is what to do with the open position and the pending order, which probably still exist.

In the worst case, you simply close them or delete them manually. But what if you have 2 or more EAs running automatically, in a HEDGING type account, on the same asset, and one of these EAs simply decides to leave the line and this is where it is removed from the chart. Now you have to check the orders and positions in the Toolbox window, in the Trade tab (Figure 01), and try to close the trades that were opened by the bad EA that has been closed. In this case "_bad_" does not mean that the EA is badly programmed. I think the idea is clear.

![Figure 01](https://c.mql5.com/2/48/001__4.png)

Figure 01 - Where to look for orders and positions in the MetaTrader 5 platform

Here programming can help us a little, but don't expect miracles. It only helps us within its possibilities. So, let's see how we can use programming in this specific case, where a bad EA decided to go around, committing its crazy things, and ended up being expelled from the chart for its bad habits and misconduct.

### Using the Destructor

The destructor is a class function that cannot be really called by code at any point in time. In fact, the constructor, like its "colleague" destructor, is called only once during the lifetime of the class. The constructor is called when the class is born, and the destructor is called when the class dies. In both situations, you as a programmer, will hardly say when a class will be born, and when it will die. Normally this is done by the compiler, but in specific cases the programmer can say when a class will be born, and when it will die, by using some things in the code. But the call to give it life can even happen because of the parameters we use in the constructors. The same cannot be said about what happens when it dies.

However, don't worry about that for now, you'll understand it better when we get to the EA code.

Regardless of when the class dies, we can tell it what to do when that happens. In this way, we can tell the C\_Manager class, that when it is going to remove the EA from the chart, it should do so and should eliminate anything that was done and left by the EA, that is, to close the open position and remove the pending order that is in the order book. But remember to confirm this by looking at the Toolbox, figure 01, as it could be that the EA was kicked out of the chart but the destructor was unable to carry out its task.

To do this, let's add a new value to the error enumeration:

```
class C_Manager : private C_Orders
{
        enum eErrUser {ERR_Unknown, ERR_Excommunicate};
        private :
```

This value will tell the destructor that the EA has been removed from the chart and everything that the EA is responsible for must be undone. To understand how a destructor can get this value if it can't actually get any value through a parameter, look at its code below:

```
                ~C_Manager()
                {
                        if (_LastError == (ERR_USER_ERROR_FIRST + ERR_Excommunicate))
                        {
                                if (m_TicketPending > 0) RemoveOrderPendent(m_TicketPending);
                                if (m_Position.Ticket > 0) ClosePosition(m_Position.Ticket);
                                Print("EA was kicked off the chart for making a serious mistake.");
                        }
                }
```

The code is very simple. We will check the **\_LastError** value to understand what could happen. Therefore, if the error value is equal to the one that we just added to the enumeration, the EA should be removed from the chart for bad behavior. In this case if there is a pending order, a request to delete the order will be sent to the server. Also, if there is an open position, a request to close it will be sent to the server. Finally, we will inform you in the terminal about what happened.

But remember, this is not secure at all. We are just trying to get some kind of programming help. You, as a trader, must be attentive and remove any order or position that the EA may have left behind when it was removed from the chart. With this we conclude this topic that was very short, but I hope it was clear enough. But before ending this article, let's see one more thing, which we will consider in the next topic.

### Creating an Error Tolerance Level

If you looked at all the code in this article and in the previous one, and managed to understand what I'm explaining, you must be imagining that the C\_Manager class is very hard on the EA, not admitting any type of failure, even the smallest ones. Yes, this is indeed happening, but we can change this a little. There are some types of errors, and mistakes that are not so serious or that are not the EA's fault.

One such error is when the server reports **TRADE\_RETCODE\_MARKET\_CLOSED**, this happens because the market is closed. This type of error can be tolerated as it is not the EA's fault. Another type is **TRADE\_RETCODE\_CLIENT\_DISABLES\_AT** which appears because _Algo Trading_ is disabled in the platform.

There are various types of errors that can occur through no fault of the EA, i.e. they can occur for various reasons. Therefore, being too harsh on the EA, blaming it for everything that can go wrong, is not entirely fair. Therefore, we need to create some mechanism in order to control and tolerate certain types of errors. If the error does not seem to be serious, we can ignore it and leave the EA on the chart. However, if it is a really serious error, we can make some decision depending on the severity of the error.

So, in the C\_Manager class, we create a public function so that the EA can call it when it has doubts about the severity of a possible error. The function code is shown below:

```
                void CheckToleranceLevel(void)
                        {
                                switch (_LastError)
                                {
                                        case ERR_SUCCESS: return;
                                        case ERR_USER_ERROR_FIRST + ERR_Unknown:
                                                Print("A serious error has occurred in the EA system. This one cannot continue on the chart.");
                                                SetUserError(ERR_Excommunicate);
                                                ExpertRemove();
                                                break;
                                        default:
                                                Print("A low severity error has occurred in the EA system. Your code is:", _LastError);
                                                ResetLastError();
                                }
                        }
```

I'm not providing a final solution here, and it's not 100% correct either. I'm just demonstrating how to proceed in order to create a way to allow mistakes in an EA. The code above shows 2 examples where the tolerances will be different.

There is a possibility of a more serious error, in which case the EA will be removed from the chart. The correct way to do this is to use two functions, one of which configures the EA exception error, and the other one sets a command to delete it. Thus, the C\_Manager class destructor will try to remove or undo what the EA was doing. There is a second way, in which the error is lighter. In this case, we simply show a warning to the trader and remove the error indication to leave the value clear, so if the call occurs without an error, the function simply returns.

Ideally, we should facilitate error handling here by defining one by one which errors are valid and which are not. Another thing is that we don't have this call inside the C\_Manager class, and the error tolerance would be very high. You really shouldn't do this. You can add a call to the above function at certain times, so you don't forget to add calls to the EA. But for now, this call will be made within the EA's code, at very specific points.

### Conclusion

This article complements the previous one, so its content is not too difficult to read and understand. I think there was enough material to complete the task.

I know that many people would already like to look at the EA code in order to use what has been shown here, but things are not so simple when it comes to an automated EA. So, in the next article, I will try to share some of my experience in this matter, and we will consider the precautions, problems, and risks associated with the EA coding. We will focus on the EA code only, because based on these last two articles, I will have something to show in the next part of this series. Therefore, study calmly and try to understand how this class system works, because there will be a lot more material in the next article.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11256](https://www.mql5.com/pt/articles/11256)

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

**[Go to discussion](https://www.mql5.com/en/forum/443910)**

![Data Science and Machine Learning (Part 13): Improve your financial market analysis with Principal Component Analysis (PCA)](https://c.mql5.com/2/52/pca_avatar.png)[Data Science and Machine Learning (Part 13): Improve your financial market analysis with Principal Component Analysis (PCA)](https://www.mql5.com/en/articles/12229)

Revolutionize your financial market analysis with Principal Component Analysis (PCA)! Discover how this powerful technique can unlock hidden patterns in your data, uncover latent market trends, and optimize your investment strategies. In this article, we explore how PCA can provide a new lens for analyzing complex financial data, revealing insights that would be missed by traditional approaches. Find out how applying PCA to financial market data can give you a competitive edge and help you stay ahead of the curve

![Neural networks made easy (Part 33): Quantile regression in distributed Q-learning](https://c.mql5.com/2/50/Neural_Networks_Made_Easy_q-learning_avatar.png)[Neural networks made easy (Part 33): Quantile regression in distributed Q-learning](https://www.mql5.com/en/articles/11752)

We continue studying distributed Q-learning. Today we will look at this approach from the other side. We will consider the possibility of using quantile regression to solve price prediction tasks.

![Learn how to design a trading system by Fibonacci](https://c.mql5.com/2/52/learnhow_trading_system_fibonacci_avatar.png)[Learn how to design a trading system by Fibonacci](https://www.mql5.com/en/articles/12301)

In this article, we will continue our series of creating a trading system based on the most popular technical indicator. Here is a new technical tool which is the Fibonacci and we will learn how to design a trading system based on this technical indicator.

![MQL5 Cookbook — Macroeconomic events database](https://c.mql5.com/2/51/mql5-recipes-database.png)[MQL5 Cookbook — Macroeconomic events database](https://www.mql5.com/en/articles/11977)

The article discusses the possibilities of handling databases based on the SQLite engine. The CDatabase class has been formed for convenience and efficient use of OOP principles. It is subsequently involved in the creation and management of the database of macroeconomic events. The article provides the examples of using multiple methods of the CDatabase class.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/11256&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069162858287137035)

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