---
title: Modelling Requotes in Tester and Expert Advisor Stability Analysis
url: https://www.mql5.com/en/articles/1442
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:03:32.074407
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/1442&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071540762635610610)

MetaTrader 4 / Trading systems


### Introduction

The Strategy Tester embedded in the Meta Trader 4 Client Terminal is a very good verification/quality evaluation tool for a trading strategy realized in the Expert Advisor. But at the current moment, it has two quite critical (to my mind) features. First, it does not use the real tick history for a symbol and models ticks on the basis of one-minute bars. Second, it does not consider the phenomenon that brokers requote rather often, especially on small trade volumes or on very large ones, and also on the "thin", low-liquid market. Under the conditions of possible requotes, no opportunity to check up the EA leads to appearance of "grails", trading on “spikes”, leaps and non-market quotes. It was explained and proved many times that such a strategy could not be effective on a real account but, unfortunately, quite often it is difficult to realize whether your Expert Advisor plays on “spikes” or not. On a history chart with trades imposed on it, it is sometimes possible to see that the Tester opens trades on leaps, but not every time. It is also difficult to predict whether it will whittle the requote strategy during these moments or just lower the profitability. In this article, I am going to state my own method to solve this problem.

### Assumptions and Definitions

Requote is the broker’s response to the order you sent to the trade server. This broker’s response informs about that the price (at which you try to open a trade) is not actual any more. It often happens on low-liquid or volatile markets where the price can change strongly during your request processing by the terminal. Also requotes are very frequent when trying to open at blowouts and non-market quotes.

Let us consider that the Expert Advisor opens and closes only orders like OP\_SELL and OP\_BUY. It does not change an essence, but makes some functions easier.

### Requotes’ Modeling in the Strategy Tester

Let us aim to simulate requotes with beforehand set probability at the opening and closing of orders. First of all let us enter a special external variable which will set the occurrence probability of our artificial requote. According to my investigation/monitoring, the probability of the requotation on low-liquid pairs blowouts comes nearer to 90 % - therefore let us take this value at the Expert Advisor analysis.

```
extern int RQF_TEST = 90;
```

As we will use the function MathRand returning an arbitrary value, it is better to initialize the sequence of pseudo-random numbers. For this purpose the function MathSrand will be carried out in the beginning of the Expert Advisor's work. For more detailed information concerning random value generation and the purpose of this function you can learn in the reference book on MQL language. It should be said that if we will not use MathSrand, despite of the numbers " randomness" from MathRand function, at any run of the strategy we will receive the same result. And, generally speaking, it does not suit us:

```
int start()
  {
    //----
    MathSrand(TimeCurrent());
```

After that we should write the own functions OrderSend and OrderClose. Let us name them MyOrderSend and MyOrderClose:

```
int MyOrderSend(int req_prob, string symbol, int cmd, double volume, double price,
                int slippage, double stoploss, double takeprofit,

                string comment="", int magic=0, datetime expiration=0,
                color arrow_color=CLR_NONE)
  {
    if(IsTesting() && (MathRand() % 100) < req_prob && (cmd == OP_SELL || cmd == OP_BUY))
        return (-1); //modelling requote
    return(OrderSend(symbol, cmd, volume, price, slippage, stoploss, takeprofit,
           comment, magic, expiration, arrow_color));
  }

bool MyOrderClose(int req_prob, int ticket, double lots, double price, int slippage,
                  color Color=CLR_NONE)
  {
    if(IsTesting() && (MathRand() % 100) < req_prob)
        return (false); //modelling requote
    return (OrderClose(ticket, lots, price, slippage, Color));
  }
```

Now we should replace all OrderSend and OrderClose functions in the EA by MyOrderSend and MyOrderClose with the indication of the entered before external variable RQF\_TEST as the first argument. Below there is an example from my own EA.

OrderClose -> MyOrderClose:

```
if(MyOrderClose(RQF_TEST, ticket, amount, Bid, 0, Blue))
// PREV: if(OrderClose(ticket, amount, Bid, 0, Blue))
  {
    Print("Skalpel: Order #" + ticket + " closed! (BUY, " + Bid + ")");
            //... Something
  }
else
  {
    // Requote or something like this
    Print("Skalpel: Error while closing order #" + ticket + " (BUY)!");
    // ... Something
  }
```

OrderSend -> MyOrderSend:

```
ticket = MyOrderSend(RQF_TEST, Symbol(), OP_BUY, amount, Ask, 0,
                     Bid-StopLoss*Point, Bid+TakeProfit*Point,
                     NULL, EXPERT_MAGIC, 0, Blue);
// PREV: OrderSend(Symbol(), OP_BUY, amount, Ask, 0, Bid - StopLoss*Point,
// Bid+TakeProfit*Point, NULL, EXPERT_MAGIC, 0, Blue);

if(ticket > 0)
    Print("Skalpel: Order #" + ticket + " opened! (BUY)");
else
    Print("Skalpel: Error while placing order.");
    // ... Requote or something like this
```

### Conclusions And the Analysis of Results

First of all the value of variable RQF\_TEST should be explained. It sets the quantity of requotes for 100 trades and, accordingly, can take on a value from 0, when there is no requotes at all, up to 100, when it is absolutely impossible to open or to close the order. If RQF\_TEST is equal 90, as in an example, it means that approximately 90 attempts to open or close the transaction of 100 will end in failure, i.e. the requote will be simulated.

Actually, the results, which were received at various values of RQF\_TEST, show the stability of your strategy to requotes and to differences in tick flow from the tester and the broker.

If results deteriorate when RQF\_TEST growing, it is necessary to check expediency of such a strategy, since the sensitiveness to requotes means that you play on sharp, temporary blowouts or you pip. It was said a lot about the consequences of such advisers’ usage.

As an example let us consider a chart of the classical EA's balance working on blowouts, with various values of RQF\_TEST. The Expert Advisor is taken from the article "My first Grail" ().And it was slightly transformed for visualization. All orders of limit type are realized by usual market ones, and also parameters are gleaned so that charts show parameters’ deterioration when requoting most evidently.

No reqoutes (RQF\_TEST = 0):

![](https://c.mql5.com/2/14/testergraph_2.gif)

In 90 cases of 100 there is a requote (RQF\_TEST = 90):

![](https://c.mql5.com/2/14/testergraph-2_2.gif)

Obviously, the situation is extremely deplorable. Especially on the stipulation that testing EURCHF is the extremely illiquid pair and requotes are very frequent even on the quiet markets, much less on blowouts ones. Therefore, the expediency of our EA usage is vanishing despite of the very beautiful chart when requotation is absenting.

### Remarks And Additions

Actually, it is rather difficult to find the adviser which could become from profitable to unprofitable because of requotes. I was searching such an Expert Advisor, that its quality cutback could be clearly demonstrated at the balance chart, for a long time. Usually requotation (even at the level of 50 %) axes the quantity of transactions and the profit, when the chart outwardly is the same (constant). There is a simple explanation to it: if the broker does not allow you to open the order on the “stake” in 90 cases from 100, then in the residual 10 % of the situations you’ll receive some pips of profits, and until you aren’t forbidden to use EAs. Such situations were described in the article "My first Grail ". But even if we assume, that the broker will not encumber (it is scarcely), 10 times reduced profit (and it is exactly such figure when requoting at the level of 90 % - the adviser simply "misses" 90 “stakes” from 100) will rob of all "Grail" advantages – it will be less profitable, rather than the bank deposit.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1442](https://www.mql5.com/ru/articles/1442)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1442.zip "Download all attachments in the single ZIP archive")

[Graal2.mq4](https://www.mql5.com/en/articles/download/1442/Graal2.mq4 "Download Graal2.mq4")(11.1 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**[Go to discussion](https://www.mql5.com/en/forum/39313)**

![How to Develop a Reliable and Safe Trade Robot in MQL 4](https://c.mql5.com/2/14/327_2.png)[How to Develop a Reliable and Safe Trade Robot in MQL 4](https://www.mql5.com/en/articles/1462)

The article deals with the most common errors that occur in developing and using of an Expert Advisor. An exemplary safe automated trading system is described, as well.

![Ten "Errors" of a Newcomer in Trading?](https://c.mql5.com/2/13/193_3.png)[Ten "Errors" of a Newcomer in Trading?](https://www.mql5.com/en/articles/1424)

The article substantiates approach to building a trading system as a sequence of opening and closing the interrelated orders regarding the existing conditions - prices and the current values of each order's profit/loss, not only and not so much the conventional "alerts". We are giving an exemplary realization of such an elementary trading system.

![MQL4 Language for Newbies. Introduction](https://c.mql5.com/2/14/404_19.gif)[MQL4 Language for Newbies. Introduction](https://www.mql5.com/en/articles/1475)

This sequence of articles is intended for traders, who know nothing about programming, but have a desire to learn MQL4 language as quick as possible with minimal time and effort inputs. If you are afraid of such phrases as "object orientation" or "three dimensional arrays", this article is what you need. The lessons are designed for the maximally quick result. Moreover, the information is delivered in a comprehensible manner. We shall not go too deep into the theory, but you will gain the practical benefit already from the first lesson.

![Alternative Log File with the Use of HTML and CSS](https://c.mql5.com/2/14/385_10.gif)[Alternative Log File with the Use of HTML and CSS](https://www.mql5.com/en/articles/1432)

In this article we will describe the process of writing a simple but a very powerful library for making the html files, will learn to adjust their displaying and will see how they can be easily implemented and used in your expert or the script.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=foqzvaakcljngfvsmxxxjonyulqjcrro&ssn=1769191411907827285&ssn_dr=0&ssn_sr=0&fv_date=1769191411&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1442&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Modelling%20Requotes%20in%20Tester%20and%20Expert%20Advisor%20Stability%20Analysis%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176919141111517491&fz_uniq=5071540762635610610&sv=2552)

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