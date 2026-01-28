---
title: Creating an EA that works automatically (Part 08): OnTradeTransaction
url: https://www.mql5.com/en/articles/11248
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:08:50.804752
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/11248&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069159065831014654)

MetaTrader 5 / Trading


### Introduction

In previous articles: [Creating an EA that works automatically (Part 06): Account types (I)](https://www.mql5.com/en/articles/11241) and [Creating an EA that works automatically (Part 07): Account types (II)](https://www.mql5.com/en/articles/11256), we were focusing on the importance of being careful when designing an EA that trades automatically.

Before we can really understand how the EA code for automation should work, we need to understand how it interacts with the trade server. Take a look at the Figure 01:

![Figure 01](https://c.mql5.com/2/48/001__5.png)

Figure 01. Message Flow

Figure 01 shows the message flow that allows the EA to send orders or requests to the trade server. Pay attention to the direction of the arrows.

The only moment when the arrows are bidirectional is when the C\_Orders class sends an order to the server using the OrderSend function because at this time it receives a response form the server via the structure. Other than this point, all other points are directional. But here I am only showing the process for submitting market orders or for placing orders on the order book. Thus, we have a very simple system.

In the case of a 100% automated EA, there are still some things that we need. And for an EA with minimal automation, we still need to add some details. Everything will be happening between the EA and the C\_Manager class. We will not be adding code in any other part of the EA. Well, there is another thing. In the EA that is 100% automated we will have to remove the C\_Mouse class (as for a 100% automated EA it is of no use). It is very important to understand what a message flow is, as without this we won't be able to proceed with further topics.

### Adding control and accessibility features

The biggest problem is that many MQL5 language users do not use some of the features to create EAs, although this language provides such features. Perhaps it comes from ignorance or for some other reason, but it does not matter. If you are going to use all the features provided by MQL5, then in order to increase the code robustness and reliability, you really need to think about using some of the resources that this language puts at your disposal.

The first thing we will do is add three new functions into the C\_Manager class. They will serve for the class to release the EA or to know what the EA is planning to do. The first of these functions is shown below:

```
inline void EraseTicketPendig(const ulong ticket)
                        {
                                m_TicketPending = (ticket == m_TicketPending ? 0 : m_TicketPending);
                        }
```

This function removes the value of the pending ticket, when the informed ticket is equal to the pending ticket. Normally this will not actually happen. A pending order will not be removed by the EA. The removal normally happens due to the interference of the trader or the EA user, which is not advisable. However, if the EA notices that a pending order it has placed in the order book has been deleted by the user, then it must inform the C\_Manager class so that it allows the EA to place a new pending order in the order book if necessary.

The next new function is shown below:

```
                void PedingToPosition(void)
                        {
                                ResetLastError();
                                if ((m_bAccountHedging) && (m_Position.Ticket > 0)) SetUserError(ERR_Unknown);
                                        else m_Position.Ticket = (m_Position.Ticket == 0 ? m_TicketPending : m_Position.Ticket);
                                m_TicketPending = 0;
                                if (_LastError != ERR_SUCCESS) UpdatePosition(m_Position.Ticket);
                                CheckToleranceLevel();
                        }
```

The EA uses this code to notify the C\_Manager class that a pending order has just been turned into a position or some changes have been made to the position. Please note that when executing this function, the C\_Manager class will delete the pending order ticket, allowing the EA to place a new pending order. However, the case will continue only if there is no critical error which will be analyzed by the function we discussed in the previous article. But this function actually does not work lone, it needs another one which is shown below:

```
                void UpdatePosition(const ulong ticket)
                        {
                                int ret;

                                if ((ticket == 0) || (ticket != m_Position.Ticket)) return;
                                if (PositionSelectByTicket(m_Position.Ticket))
                                {
                                        ret = SetInfoPositions();
                                        m_StaticLeverage += (ret > 0 ? ret : 0);
                                }else ZeroMemory(m_Position);
                                ResetLastError();
                        }
```

There are two other functions missing from the C\_Manager class. But since these are automation functions, we won't cover them in detail now.

Now, in a much more complete way, we finally have the C\_Manager class and the EA which are friendly with each other. Both can work and make sure they don't become aggressive or unfriendly. Thus, the message flow between the EA and the C\_Manager class becomes as the one in Figure 02:

![Figure 02](https://c.mql5.com/2/48/002__1.png)

Figure 02. Message flow with new functions

This flow may seem too complicated or completely non-functional, but this is exactly what has been implemented so far.

Looking at Figure 02, you might think that the EA code is very complex. But it is much simpler than what many people consider to be a necessary code for an EA. Especially when it comes to an automated EA. Remember the following: The EA does not actually generate any trades. It is just a means or tool for communicating with the trade server. So it actually just reacts to triggers which are applied to it.

Based on this understanding, let's go through the EA code in its current state, before it becomes automated. But for those who haven't seen it, the EA code hasn't undergone major changes since the last article in which it had appeared, which was [Creating an EA that works automatically (Part 05): Manual triggers (II)](https://www.mql5.com/en/articles/11237). The only changes that have actually been made are shown below:

```
int OnInit()
{
        manager = new C_Manager(def_MAGIC_NUMBER, user03, user02, user01, user04, user08);
        mouse = new C_Mouse(user05, user06, user07, user03, user02, user01);
        (*manager).CheckToleranceLevel();

        return INIT_SUCCEEDED;
}
```

Actually, we only had to add a new line. This enabled the check if a more serious or critical error has happened. But a question arises: how to make the EA inform the C\_Manager class about what is happening in relation to the order system? Many people don't know what to do in this situation, trying to find out how to learn about what is being done on the trade server. But herein lies the danger.

The first thing you should really understand is that the MetaTrader 5 platform and the MQL5 language are not just ordinary tools. You will not actually create a program that must constantly search for information. This is because the system is based on events and not on processes. In event-based programming, you don't have to think in steps, you have to think differently.

To understand this, think about the following: If you are driving, basically your intention will be to arrive at a certain destination. But along the way, you will have to solve some things which will happen and which are apparently not interconnected. But all these things will influence your direction, for example, braking, accelerating, having to change the path due to something unforeseen that happened. You know that these events can happen, but you have no idea when they will happen.

That's what event-based programming is all about: You have access to some events, which are provided by the specific language for a specific job. All you have to do is create some logic that can resolve the issues that a given event raises in order to have some kind of useful result.

MQL5 provides some events that we can (and others that we must) handle for each type of situation. A lot of people get lost when trying to understand the logic behind this, but it is not complicated. As soon as you understand this, programming will become much simpler. Because the language itself provides you with the necessary means to handle any problems.

This is the first point: Primarily use MQL5 language to solve problems. If that somehow isn't enough, add specific functionality. You can use another language like C/C++ or even Python, but first try using MQL5.

The second point: You must not try to capture information, no matter where it comes from. You should simply, whenever possible, use and respond to the events that the MetaTrader 5 platform will be generating.

The third point: Don't use functions or try to use events that aren't really useful to your code. Use exactly what you need and try to always use the right event for the right job.

Based on these 3 points, we have 3 options to choose from, for the EA to interact with the C\_Manager class or any other class that needs to receive data provided by the MetaTrader 5 platform. The first option is to use an event trigger for each new ticket received. This event will call the **OnTick** function. However, sincerely, I don't recommend using this function. We will see the reason another time.

The second option is to use a time event which fires the **OnTime** function. However, this option is not suitable for what we are doing now. This is because we would have to keep checking the list of orders or positions with each triggering of the time event. This is not effective at all, making the EA a dead weight for the MetaTrader 5 platform.

The last option is to use the Trade event which fires the **OnTrade** function. It is activated every time there is a change in the order system, i.e. there is a new order or a change in the position. But the OnTrade function is not very suitable in some cases, while in others it can save us from doing certain tasks and thus will make things much simpler. Instead of using **OnTrade**, we will use **OnTradeTransaction**.

What is OnTradeTransaction and what is it for?

It is perhaps the most complex event handling function that MQL5 has, so please use this article as a good source to learn about it. I will try to explain and provide as much information as possible of what I have understood and learned about the use of this function.

To make things easier to explain, at least at this initial stage, let's look at the function code in the EA:

```
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
{
        switch (trans.type)
        {
                case TRADE_TRANSACTION_POSITION:
                        manager.UpdatePosition(trans.position);
                        break;
                case TRADE_TRANSACTION_ORDER_DELETE:
                        if (trans.order == trans.position) (*manager).PendingToPosition();
                        else (*manager).UpdatePosition(trans.position);
                        break;
                case TRADE_TRANSACTION_REQUEST: if ((request.symbol == _Symbol) && (result.retcode == TRADE_RETCODE_DONE) && (request.magic == def_MAGIC_NUMBER)) switch (request.action)
                        {
                                case TRADE_ACTION_DEAL:
                                        (*manager).UpdatePosition(request.order);
                                        break;
                                case TRADE_ACTION_SLTP:
                                        (*manager).UpdatePosition(trans.position);
                                        break;
                                case TRADE_ACTION_REMOVE:
                                        (*manager).EraseTicketPending(request.order);
                                        break;
                        }
                        break;
        }
}
```

I know that this code seems rather strange to many, especially to those who are used to utilizing other methods to find out what is happening with their orders and positions. But I guarantee that if you really understand how the OnTradeTransaction function works, then you will start using it in all your EAs because it really helps a lot.

However, to explain how this works, we will avoid talking in terms of log file data whenever possible, because if you try to see the logic by following files or patterns found in log files, you can go crazy. This is because sometimes the data will not have any pattern. The reason is that this function is an event handler. These events come from the trade server, so forget about log files. Let's focus on handling the events which are sent from the trade server, regardless of the order in which they appear.

Basically, we will be looking here into three structures and their contents. These structures are filled by the trade server. You must understand that anything done here will be a processing of what the server has provided. The trading constants which we are checking here are what we actually need in the EA. You may need more constants depending on what you are creating. To find out what they will be, look in the documentation [Trade Transaction Types](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_transaction_type). You will see there 11 different enumerations, each of them for something specific.

Please note that at some points I use a variable that will refer to the [MqlTradeTransaction](https://www.mql5.com/en/docs/constants/structures/mqltradetransaction) structure. This structure is quite complex, but complex in terms of what the server sees and understands. But for us it depends on which kind of things we actually want to check, analyze and know. What interests us is the 'type' field of this structure, because it enables further system.. In this code, we deal with three transaction types performed by the server: **TRADE\_TRANSACTION\_REQUEST,** **TRADE\_TRANSACTION\_ORDER\_DELETE** and **TRADE\_TRANSACTION\_POSITION**. This is why they are used here.

Since it is hard to explain a transaction type without an example, let's first look at **TRADE\_TRANSACTION\_POSITION** which has only one line:

```
                case TRADE_TRANSACTION_POSITION:
                        manager.UpdatePosition(trans.position);
                        break;
```

This event will trigger whenever something happens in an open position — not in any position, but only in the one that has undergone some kind of modification. The server informs about it, and we pass it to the C\_Manager class to have it updated if this is the position observed by the EA. Otherwise, it will be ignored. This saves us a lot of time when trying to figure out which position was actually changed.

Next on the list is **TRADE\_TRANSACTION\_ORDER\_DELETE.** Its code may seem confusing to many:

```
                case TRADE_TRANSACTION_ORDER_DELETE:
                        if (trans.order == trans.position) (*manager).PendingToPosition();
                        else (*manager).UpdatePosition(trans.position);
                        break;
```

When an order is converted to a position, that order triggers an event in which the server reports that the order has been deleted. The same thing happens and the same event is triggered when a position is closed. The difference between the event reporting the order conversion into a position and the position closing lies in the value provided by the server.

When an order is converted to a position, we get the value specified in the position ticket, so we inform C\_Manager that an order turned into a position. When a position is closed, these values will be different. But it can also happen that you have an open position and an order was added, due to which the position volume changed. In such cases the value in **trans.order** and **trans.position** will be different. In this case, we make an update request to C\_Manager..

In some cases this event may be accompanies by **TRADE\_TRANSACTION\_POSITION**. But this is not always the case. To make the explanation easier, let's separate the information since it is very important to understand this code.

Let us first deal with the case when **trans.order** equals **trans.position** — they can be the same. So don't expect that they will always be different. When they are equal, the server starts the **TRADE\_TRANSACTION\_ORDER\_DELETE** enumeration, which however does not happen on its own but is accompanied by other enumerations. We don't have to deal with all of them, but only this specific one. The server will inform us that the order has just turned into a position. At this time, the order will be closed and the position will be opened with the same ticket value as the order that closed.

But it may happen that the server does not send us the **TRADE\_TRANSACTION\_POSITION enumeration**. Although at first you might be waiting for this enumeration, but the server simply won't fire it. But for sure it will trigger the removal. The indicated values will equal. In this case, we know that this is an order that was in the order book and that it became a position, but in the case of market orders, everything works a little differently. We will see this case later.

Now, if **trans.order** differs from **trans.position**, the server will also trigger other enumerations. But again, don't count on the specific one to come. It may happen that the server will not trigger it, but it will fire the one I'm using. In this case, it indicates that the position has just been closed for some reason, which we are not analyzing here. In any case, we will receive information about the structures received by the TradeTransaction event. That is why this event handler is so interesting: you don't have to go out looking for the information. The events are there, you only need to go to the correct structure and read the information. Is it clear the checks are performed in this way?

Usually in programs that do not use this event handler, the programmer creates a loop to go through all open positions or pending orders, trying to find out which was executed or closed. This is pure waste of time as this makes the EA busy with completely useless things, which can be captured more easily. This is because the trading server has already done all the heavy lifting for us, informing us which pending order was closed, which position was opened, or which position was closed. And we here are creating loops to find out this information.

Now we come to the part that will take the longest to explain, and yet I will not cover all cases. The reason is the same as in the previous case: It is not simple to explain all cases without having the examples. However, what is explained here will help many people. For greater convenience, let's see the fragment that we will now study. It is as follows:

```
                case TRADE_TRANSACTION_REQUEST: if ((request.symbol == _Symbol) && (result.retcode == TRADE_RETCODE_DONE) && (request.magic == def_MAGIC_NUMBER)) switch (request.action)
                        {
                                case TRADE_ACTION_DEAL:
                                        (*manager).UpdatePosition(request.order);
                                        break;
                                case TRADE_ACTION_SLTP:
                                        (*manager).UpdatePosition(trans.position);
                                        break;
                                case TRADE_ACTION_REMOVE:
                                        (*manager).EraseTicketPending(request.order);
                                        break;
                        }
                        break;
```

This **TRADE\_TRANSACTION\_REQUEST** enumeration triggers in almost all cases. It would be rather strange if it didn't. So a lot of the things we can do in terms of testing can be done within it. But since this is an enumeration which the server fires a lot, we need to filter things in it.

Normally the server triggers this enumeration after some request made by the EA or by the platform. This is when the user does something related to the order system. But don't count on this every time, as sometimes the server simply triggers this enumeration. Sometimes without an apparent reason, because of this we have to filter what is being informed.

First we filter the asset. You can use any structure for this but I prefer this one. Next we check if the request is accepted by the server, for which we use this test. And finally, we check the magic number to further filter things. Now comes the most confusing part. Because you don't know how to fill in the rest of the code.

When we use a **switch** to check the action type, we do not (and will not) analyze the action taken on the server. That's not what we're actually going to do. In fact, we will be doing exactly a counter check of what was sent to the server either by the EA or by the platform. There are 6 types of actions, which are classic. They are in the [ENUM\_TRADE\_REQUEST\_ACTIONS](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions) enumeration. To simplify the task, let's see the following table. It is the same as in the documentation. I used it to make explanation easier, but my description is slightly different from the documentation.

| Action type | Action description |
| --- | --- |
| TRADE\_ACTION\_DEAL | Place a trade order to be executed at the market price |
| TRADE\_ACTION\_PENDING | Place an order in the order book to be executed according to the specified parameters |
| TRADE\_ACTION\_SLTP | Change the stop loss and take profit values of a position |
| TRADE\_ACTION\_MODIFY | Change the parameters of a pending order which is in the order bool |
| TRADE\_ACTION\_REMOVE | Delete a pending order that is still in the order book |
| TRADE\_ACTION\_CLOSE\_BY | Close position |

Table 01

If you really follow what we have been programming since the beginning of this series of articles, but did not pay due attention, then when programming, you will have to check where the **Action type** field was used in our code. This is not to mention the **OnTradeTransaction** event handler because it doesn't count. These enumerators have already been used. But where? In the C\_Orders class.

Open the source code and pay attention to the following procedures in the C\_Orders class: **CreateOrder, ToMarket, ModifyPricePoints** and **ClosePosition**. We will consider each of them except ClosePosition, where I do not use the **TRADE\_ACTION\_CLOSE\_BY** enumeration.

Why is it so important here, in the **OnTradeTransaction** event handler? The reason is that these are the same enumeration that we will see when analyzing to which action type the **TRADE\_TRANSACTION\_REQUEST** enumeration refers to. That is why in the **OnTradeTransaction** event handler code we see **TRADE\_ACTION\_DEAL** and **TRADE\_ACTION\_SLTP**, as well as **TRADE\_ACTION\_REMOVE** — the EA should pay attention exactly to them.

But what about the rest? For or purpose of creating an automated EA, other types are not important, as they are used for other things. If you want to see how other types are applied, check out the article [Developing a trading EA from scratch (Part 25): Providing system robustness (II)](https://www.mql5.com/en/articles/10606), in which I show how other enumerations can be used.

Since I have explained where these cases come from, let's break down what each one does, starting with the one we see below:

```
        case TRADE_ACTION_DEAL:
                (*manager).UpdatePosition(request.order);
                break;
```

The enumerator is called when a market order is executed. But not only in this case. There is a second one where it is also called. When I spoke about **TRADE\_TRANSACTION\_ORDER\_DELETE**, I mentioned that there was a case when **trans.order** and **trans.postion** could be equal. This is the second case when the server triggers **TRADE\_ACTION\_DEAL**. So, we can add the value of the ticket as an open position now. But note that if something happens, for example, another position is still open, then an error will occur that will cause the EA to terminate. It's not shown here but can be seen in the **UpdatePosition** code.

See the next code below:

```
        case TRADE_ACTION_SLTP:
                (*manager).UpdatePosition(trans.position);
                break;
```

This enumerator will be triggered when changing the limit values, known as take profit and stop loss. It will simply update the stop and take values. This version is very simple, and there are ways to make it a little more interesting, but that's enough for now. And the last enumeration that we will have in the code is right below:

```
        case TRADE_ACTION_REMOVE:
                (*manager).EraseTicketPending(request.order);
                break;
```

This code is triggered when an order is deleted, and we need to notify the C\_Manager class so that the EA can send a pending order. Normally, a pending order is not removed from the order book, but it may be that the user has done so. If an order is accidentally or intentionally removed from the order book, and the EA does not notify the C\_Manager class, this will prevent the EA from sending another pending order.

### Conclusion

In this article, we looked at how to use the event handling system to process issues related to the order system faster and better. With this system the EA will work faster, so that it will not have to constantly search for the required data. True, we are still dealing with the EA that does not have any level of automation, but soon we will add automation, even if it is basic, to let the EA manage breakeven and trailing stop.

The attached file provides the full version of the code that we have covered in the last three article. I recommend that you carefully study this code to understand how everything actually works, as it will help you a lot in the next steps of our work.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11248](https://www.mql5.com/pt/articles/11248)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11248.zip "Download all attachments in the single ZIP archive")

[EA\_Automatico\_-\_08.zip](https://www.mql5.com/en/articles/download/11248/ea_automatico_-_08.zip "Download EA_Automatico_-_08.zip")(8.59 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/444464)**
(12)


![Gad Benisty](https://c.mql5.com/avatar/2021/12/61C79323-DF36.jpeg)

**[Gad Benisty](https://www.mql5.com/en/users/gadben)**
\|
20 Jun 2023 at 21:28

[OnTradeTransaction](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction "MQL5 Documentation: the OnTradeTransaction function") is great but I have a real issue.

I started to use it to catch events (opening a position, closing, modifying) as I did before in MT4 (checking constantly the positions to guess what happened). In that aspect, MT5 approach is very clean.

BUT WHAT HAPPENS IF THE EA IS DOWN or OFF DURING ONE HOUR : it will not receive any event of course but when it restart it will not get the events it messed during 1 hour. So to guess what happened it will need to do the old MT4 way by analysing the positions to guess what happened. To solve this issue I must keep the 2 ways to detect events : the MT5 way and the MT4 way as backup.


![Gad Benisty](https://c.mql5.com/avatar/2021/12/61C79323-DF36.jpeg)

**[Gad Benisty](https://www.mql5.com/en/users/gadben)**
\|
20 Jun 2023 at 21:29

[OnTradeTransaction](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction "MQL5 Documentation: the OnTradeTransaction function") is taking up to 5 seconds to detect events on pending orders (modification and deletion). Is this normal ?

For market positions it's immediate.


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
21 Jun 2023 at 12:56

**Gad Benisty [#](https://www.mql5.com/en/forum/444464#comment_47652988):** [OnTradeTransaction](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction "MQL5 Documentation: the OnTradeTransaction function") is great but I have a real issue. I started to use it to catch events (opening a position, closing, modifying) as I did before in MT4 (checking constantly the positions to guess what happened). In that aspect, MT5 approach is very clean. BUT WHAT HAPPENS IF THE EA IS DOWN or OFF DURING ONE HOUR : it will not receive any event of course but when it restart it will not get the events it messed during 1 hour. So to guess what happened it will need to do the old MT4 way by analysing the positions to guess what happened. To solve this issue I must keep the 2 ways to detect events : the MT5 way and the MT4 way as backup.

I agree... that's why during Expert Advisor startup, a check of positions or pending orders is done. But this is seen in an article a little further on in this same sequence.

Automated translation applied by moderator

![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
21 Jun 2023 at 16:03

**Daniel Jose [#](https://www.mql5.com/en/forum/444464#comment_47664679):**

Concordo ... por isto que durante a inicialização do Expert Advisor, é feita uma checagem das posições ou ordens pendentes. Porém isto é visto em um artigo um pouco mais a frente nesta mesma sequencia.

Please post in English on this forum.


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
22 Jun 2023 at 10:33

**Alain Verleyen [#](https://www.mql5.com/en/forum/444464#comment_47669656) :**

Please post in English on this forum.

Sorry... I forgot to tell the system to translate. 👍

![Category Theory in MQL5 (Part 4): Spans, Experiments, and Compositions](https://c.mql5.com/2/52/Category-Theory-p4-avatar.png)[Category Theory in MQL5 (Part 4): Spans, Experiments, and Compositions](https://www.mql5.com/en/articles/12394)

Category Theory is a diverse and expanding branch of Mathematics which as of yet is relatively uncovered in the MQL5 community. These series of articles look to introduce and examine some of its concepts with the overall goal of establishing an open library that provides insight while hopefully furthering the use of this remarkable field in Traders' strategy development.

![Data Science and Machine Learning(Part 14): Finding Your Way in the Markets with Kohonen Maps](https://c.mql5.com/2/52/data_science_ml_kohonen_maps_avatar.png)[Data Science and Machine Learning(Part 14): Finding Your Way in the Markets with Kohonen Maps](https://www.mql5.com/en/articles/12261)

Are you looking for a cutting-edge approach to trading that can help you navigate complex and ever-changing markets? Look no further than Kohonen maps, an innovative form of artificial neural networks that can help you uncover hidden patterns and trends in market data. In this article, we'll explore how Kohonen maps work, and how they can be used to develop smarter, more effective trading strategies. Whether you're a seasoned trader or just starting out, you won't want to miss this exciting new approach to trading.

![Creating a comprehensive Owl trading strategy](https://c.mql5.com/2/0/Example_of_creating_Avatar.png)[Creating a comprehensive Owl trading strategy](https://www.mql5.com/en/articles/12026)

My strategy is based on the classic trading fundamentals and the refinement of indicators that are widely used in all types of markets. This is a ready-made tool allowing you to follow the proposed new profitable trading strategy.

![Neural networks made easy (Part 34): Fully Parameterized Quantile Function](https://c.mql5.com/2/50/Neural_Networks_Made_Easy_quantile-parameterized_avatar.png)[Neural networks made easy (Part 34): Fully Parameterized Quantile Function](https://www.mql5.com/en/articles/11804)

We continue studying distributed Q-learning algorithms. In previous articles, we have considered distributed and quantile Q-learning algorithms. In the first algorithm, we trained the probabilities of given ranges of values. In the second algorithm, we trained ranges with a given probability. In both of them, we used a priori knowledge of one distribution and trained another one. In this article, we will consider an algorithm which allows the model to train for both distributions.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/11248&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069159065831014654)

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