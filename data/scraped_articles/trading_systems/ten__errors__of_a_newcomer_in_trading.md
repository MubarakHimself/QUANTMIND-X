---
title: Ten "Errors" of a Newcomer in Trading?
url: https://www.mql5.com/en/articles/1424
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:58:14.866685
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/1424&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083247761152808901)

MetaTrader 4 / Trading systems


### Introduction

A newcomer in trading is told over and over again: "trend is your friend, don't you move against the trend"
or "place your stop orders as short as possible, but allow your profit to grow" (see, for instance,
\[1\]). There seems not to be any room for doubt of validity of these statements, especially where this validity
is demonstratively proved by many researches (see, for example, \[2, pp. 35-40\]). Who would mind to place a position
"by trend" and gain profits?! But what if we have made a mistake and it has turned out to be "non-trend"?
Then the initial statements begin to be developed: to reduce risks, it is necessary to utilize hedging since
we can be mistaken in defining the trend or in forecasting random change in prices at a market with great volatility,
etc. It means we should take some measures – a danger foreseen is half avoided – in case prices goes in a
direction we didn't foreseen.

Thus, we have to consider price movements to be of random nature. So all attempts
to act "as taught" do not guarantee positive results. Otherwise, why
do those "errors of newcomers in trading" occur? They are newcomers,
so they did not have time to forget the copy-book maxims, which have been presented
as practically the "Ten Commandments".

The author by no means wants to upbraid my esteemed colleague Collector. He conscientiously
gave in his article \[1\] a brief description of well-known, recognized by many people
and often repeated statements.The ideas are so widely spread because many press
towards getting positive results being at the market with only one open position.
It means they think it is necessary to close a position before opening a new one.
I would compare this to fishing with one single rod - a nice recreational activity
for amateurs, but professionals normally use quite different hooks and lines. When
the matter concerns a **trading system**, it is better not to limit the analysis by one single order. The possibility to
use a number of orders opened and closed both consecutively and simultaneously
according to the situation for positions already opened together with usual alerts
should, in the author's opinion, be immediately considered as the basic opportunity
to adapt oneself to the price behavior, as the basic "range of discretion"
of the trading system developer, not only as an instrument of saving.

### Ten "Errors" of a Newcomer in Trading?

Suppose we are "at minus" with a long position. According to the accepted rules, we should close it
as soon as possible – "make your losses smallest". But, I say, we have just decided to buy, it means
we were sure that the price would go up – for example, a "checked" oscillator had drawn divergence
in the oversold zone. Why now, when the price is lower and the buying is more than preferable, should we exit
the market? It is more logical to enter in the same direction in order to reinforce the long position. This,
at least, looks more consequent. Now the total positive result will be the total plus on two positions, not one
by one. The author sometimes opens the third position in such cases, but a short one. Then the total result depends
on three orders.

It is control over a finite collection of orders (including rules of opening, closing,
choosing a volume, modifying the StopLoss and TakeProfit levels in time regarding
price changes and other conditions) that, in general, must form the basis of a
trading system. Limitation of the amount of orders by one, two or three is a particular
case for this general approach.

Thus, nobody minds following the trend. The matter is: How can we recognize it properly
and on time? We may say that this is "to be or not to be?" of the most
trading systems known.

If Bollinger bands are compared at different averaging periods, it is easy to see
that they are broader for longer periods (se Figure below: the smaller period is
70 minutes, the larger one is 370 minutes). Dispersion on a larger period is contributed
by that what is mean value on a smaller period.

![](https://c.mql5.com/2/14/cuplhfxckyufrtflub_1.gif)

The mean line (moving average) when shifted back by a half of the selected period can be considered the trendline
(a posteriori). Assuming that we know a priori how that next value of the short-period average will change, the
random price changing can be considered to have dispersion as on the smaller time frame, but determined changing
mean value (drift, trend). Looking at the obviously nonstationary process of price changing (short period is
equal to the bar length), everybody 'sees' (just because everybody wants to see!) the sum of the random process
(characterized by dispersion on a short interval, visually within the range – by the difference between high
and low, i.e., by a rather small value) and an unknown, but nonrandom, a predefined process (trend) with a significantly
larger range. This creates the illusion of possibility and the mass wish to guess the trend direction in the
nearest future. We 'see' objects of technical analysis in the price chart on time in the same way as we do see
animals or things looking at clouds. We see only the things we want and ready to see - this is how our perception
works.

Statistics knows all. It states that, in average, the probability to gues properly
approaches to 0.5, also for very successful traders. But the latter ones don't
earn their profits due to guessing. they do it due to their experiences, ability
to control an open position, hadging, portfolio, etc., as well as due to their
luck - just read their interviews. Would not it be easier for a beginner to refuse
those attempts and just recognize that the process is fully random and practically
stationary? What will it yield for us?

First, it will be clear that any deviation from the average will most probably result
in returning to the initial state and, therefore, one should not be afraid of placing
orders against the trend. There is no need, either, to be afraid of losing with
one open position or to be in a hurry to close it.

Second, Forex trading will become a stabilizing factor for the world economy – trading against the trend will
create a negative feedback and considerably reduce the exchange rate fluctuations. By the way, the economy itself
is known to have such a self-regulation – the growing exchange rate will raise the prices of exporting and
stimulate importing which brings the rate to lower values, and so on. Sorry for possible oversimplifying (and
for tautology). There is no need to overestimate the situation and be afraid that the rate will be fixed if everybody
starts trading against the trend - this is a very distant prospect and, which is the most important, there are
other (fundamental) factors that influence prices (this is when the fundamental analysis starts working!).

Third, ten "errors" of a newcomer in trading described in \[1\] turn not
to be errors at all, but proper steps. Let us look through them one by one \[1\].

01. **Trading when market has just opened**




    Since we have given up all hope to guess the trend direction, there is no need to
    wait for a proper moment - we should enter the market as soon as it is feasible.
    It would also make sense to open two positions of the same volume, but differently
    directed - a short one and a long one. One of them will gain profits earlier, another
    one can do it later, when the price returns and goes a profitable direction. At
    that, at the moment of opening and until either of the two positions is closed,
    the trade can be 100% hedged, the risk approaches to zero (we can only lose on
    commission, if any, on spread, and on the difference between the swaps of long
    and short positions provided it takes more than a day until we close them).

02. **Undue hurry in taking profit**




    It is never too early to take the profit! We will not make our situation worse by this. If we have fixed the profit
    in, for example, a long position and the price has decreased by a value exceeding the spread+commission, we can
    buy again – we will be able to double the profit taken on the same segment, but we surely won't lose the profit
    fixed before! For example, we bought at 1.2300, closed at 1.2340; then the price fell to 1.2320 – buy. If the
    price goes upagain, we will earn again in the range between 1. 2320 and 1.2340. If we had not fixed the profit
    at 1.2340, we would have at 1.2320 just twenty unclear pips instead of forty appreciable ones.

03. **Adding lots in a losing position**




    … is sometimes just necessary if a losing position is a result of deviation from mean, i.e., the probability
    of return to the mean increased. Lots should be added to a posing position, and the further the price goes in
    a "wrong way", the more lots should be added.

04. **Closing positions starting with the best one**




    This issue has much in common with issue 2. It is better to close profitable positions, not losing ones – the
    latter ones can become profitable if we don't close them now!

05. **Revenge**




    This feeling does not occur if one does not close losing positions or closes them
    together with the profitable ones, obtaining a total positive result as it was
    done in a trading system \[4\], the test results of which are given at the end of
    this present article. Besides, only humans can feel revenge. Having created an
    automated trading system, we will protect ourselves against emotional steps.

06. **The most preferable positions**




    When adding lots to a losing position, the latest "addition" will, of course, be the most preferable.
    If the price goes on falling (we are now speaking about a long position), we add again. But it is the last "adding"
    that must give us the total plus – it will be in the very bottom, at the very beginning of a turn.

07. **Trading by the principle of 'bought for ever'**




    Trading by such principle is possible for two reasons. First, as I have already noticed, one should not be in
    a hurry to close a losing position if even it is very "old" - one should just wait until better time
    comes (see Clauses 1 and 4). Second, one can earn using swaps – 350% per annum - which is not bad, as well.
    \[3:356\]

08. **Closing of a profitable strategic position on the first day**




    Here we repeat Clause 2 – it is never too early to close a profitable position.

09. **Closing a position when alerted to open an opposite position**




    Highly respected Collector in his article \[1\] does not exclude such a possibility.
    The author of this present article does not consider this to be an error - it's
    just an element of a trading system.

10. **Doubts**




    In my opinion, there are no traders without doubts. George Soros said once (rephrasing
    the Napoleon's well-knwon saying): "One jumps into the market, then figures
    out what to do next". The idea is ok but the first part - "close all
    positions". I would rephrase it as follows: Let your PC to manage them and
    go for a walk.


So, the "Ten Commandments" postulated in \[1\] or anywhere else by anybody should not be considered as
the ultimate truth or a cure-all solution against losses. At present, there is only one way to make fewer mistakes
for a beginning or an advanced trader – model his or her own trading systems on his or her PC and check them
on historical data – this does not guarantee faultless operations, but arms with accurate computation, not
with implicit faith.

### Explanation of Trading Strategy

However, it would be reasonable to check the trading strategy based on the proposed approach – "no nightingales
live on fairytales!", we are lucky to have all those wonderful tools in MT4. To check it excluding influences
of any additional factors (selection of entering time and leaving by alerts), we will not use alerts at all in
the Expert Advisor \[4\] – we will do without "to be or not to be". We will open two opposite orders
at the same time to be executed instantly, i.e., we make mistakes #1 and #7.

```
   kk=0;
   tic=-1;
   if (sob)
         {
         if(max_lot_b==0.0)lotsi=0.1;else lotsi=2.0*max_lot_b;
         while(tic==-1 && kk<3)
            {
            tic=OrderSend(Symbol(),OP_BUY,lotsi,Ask,slip,0,Ask+(tp+25)*Point," ",m,0,Yellow);
            Print("tic_buy=",tic);
            if (tic==-1)
               {
               gle=GetLastError();
               kk++;
               Print("Error #",gle," at buy ",kk);
               Sleep(6000);
               RefreshRates();
               }
            }
         lastt=CurTime();
         return;
         }
      tic=-1;
      kk=0;
      if (sos)
         {
         if(max_lot_s==0.0)lotsi=0.1;else lotsi=2.0*max_lot_s;
         while(tic==-1 && kk<3)
            {
            tic=OrderSend(Symbol(),OP_SELL,lotsi,Bid,slip,0,Bid-(tp+25)*Point," ",m,0,Red);
            Print("tic_sell=",tic);
            if (tic==-1)
               {
               gle=GetLastError();
               kk++;
               Print("Error #",gle," at sell ",kk);
               Sleep(6000);
               RefreshRates();
               }
            }
         lastt=CurTime();
         return;
         }
```

If one of them touches the TakeProfit level, open it again after the profit has
been fixed, i.e., we make "mistakes" ## 2, 4 and 8 one by on.

```
   sob=(kol_buy()<1 || buy_lev_min-sh*Point>Ask) &&
      AccountFreeMargin()>AccountBalance()*0.5;
   sos=(kol_sell()<1 || sell_lev_max+sh*Point<Bid) &&
      AccountFreeMargin()>AccountBalance()*0.5;
```

The second, losing order will be strengthened with doubled volume after the price has changed at a certain interval,
then - after the same interval - strengthen it again, and so on until it reaches the preset profit level and
we close all order in the same direction – make "mistakes" ##3 and 6 in succession.

```
   if(M_ob[kb_max][2]>0.0)scb=M_ob[kb_max][7]/(M_ob[kb_max][2]*10)>tp;
   if(M_os[ks_max][2]>0.0)scs=M_os[ks_max][7]/(M_os[ks_max][2]*10)>tp;

   kk=0;
   ii=0;
   if (scb)
      {
      while(kol_buy()>0 && kk<3)
         {
         kk++;
         for(i=1;i<=kb;i++)
            {
            if(M_ob[i][0]==0)break;else ii=M_ob[i][0];
            if (!OrderClose(ii,M_ob[i][2],Bid,slip,White))
               {
               gle=GetLastError();
               Print("Error #",gle," at close buy ",ii," (",kk,")");
               Sleep(6000);
               RefreshRates();
               }
            }
         }
      }
   kk=0;
   ii=0;
   if (scs)
      {
      while(kol_sell()>0 && kk<3)
         {
         kk++;
         for(i=1;i<=ks;i++)
            {
            if(M_os[i][0]==0)break;else ii=M_os[i][0];
            if (!OrderClose(ii,M_os[i][2],Ask,slip,White))
               {
               gle=GetLastError();
               Print("Error #",gle," at close sell ",ii," (",kk,")");
               Sleep(6000);
               RefreshRates();
               }
            }
         }
      }
```

We do all this permanently making "mistake" #10. The only "mistake"
of those listed above that we have not made yet is "mistake" #9, but
it was not a real "mistake" from the very beginning. We protected ourselves
from "mistake" #5 having given control to our PC. In arrays M\_ob and
M\_os, the current information about open positions is stored:

```
int kb,kb_max=0;
   kb=kol_buy()+1;
   double M_ob[11][8];
   ArrayResize(M_ob,kb);
   int ks=0,ks_max=0;
   ks=kol_sell()+1;
   double M_os[11][8];
   ArrayResize(M_os,ks);

   ArrayInitialize(M_ob,0.0);

   int kbi=0;
   for(i=0;i<OrdersTotal();i++)
     {
     if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
     if(OrderSymbol()==Symbol() && OrderType()==OP_BUY)
        {
        kbi++;
        M_ob[kbi][0]=OrderTicket();
        M_ob[kbi][1]=OrderOpenPrice();
        M_ob[kbi][2]=OrderLots();
        M_ob[kbi][3]=OrderType();
        M_ob[kbi][4]=OrderMagicNumber();
        M_ob[kbi][5]=OrderStopLoss();
        M_ob[kbi][6]=OrderTakeProfit();
        M_ob[kbi][7]=OrderProfit();
        }
      }
   M_ob[0][0]=kb;

   double max_lot_b=0.0;
   for(i=1;i<kb;i++)
      {
      if(M_ob[i][2]>max_lot_b)
         {
         max_lot_b=M_ob[i][2];
         kb_max=i;
         }
      }
   double buy_lev_min=M_ob[kb_max][1];

   ArrayInitialize(M_os,0.0);
   int ksi=0;
   for(i=0;i<OrdersTotal();i++)
     {
     if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
     if(OrderSymbol()==Symbol() && OrderType()==OP_SELL)
        {
        ksi++;
        M_os[ksi][0]=OrderTicket();
        M_os[ksi][1]=OrderOpenPrice();
        M_os[ksi][2]=OrderLots();
        M_os[ksi][3]=OrderType();
        M_os[ksi][4]=OrderMagicNumber();
        M_os[ksi][5]=OrderStopLoss();
        M_os[ksi][6]=OrderTakeProfit();
        M_os[ksi][7]=OrderProfit();
        }
      }
   M_os[0][0]=ks;

   double max_lot_s=0.0;
   for(i=1;i<ks;i++)
      {
      if(M_os[i][2]>max_lot_s)
         {
         max_lot_s=M_os[i][2];
         ks_max=i;
         }
      }
   double sell_lev_max=M_os[ks_max][1];
```

There can be a certain amount of intervals moving against the trend, so one has to have a sufficient deposit (in
the example above – not less than $50000). However, the system works at smaller initial deposits, only the
interval of sl should be greater. If the deposit is $1000, the interval should be 300, the profit will be a bit
smaller, as well, in this case.

### Test Results

We conducted testing on USDCHF. One-minute chart was chosen to exclude the modelling
quality influence. We got similar results on other symbols, but sl and tp inputs
should be matched for better results.

**Strategy Tester Report**

**Expert Advisor Frank\_ud**

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Symbol | USDCHF (Swiss Franc vs US Dollar) |
| Timeframe | 1 Minute (M1) 2006.02.16 18:06 - 2006.09.27 18:02 |
| Model | Open prices only (fastest method to analyze the bar just completed) |
| Inputs | tp=65; sl=41; |
|  |
| Bars in history | 200061 | Ticks modelled | 400022 | Modelling quality | n/a |
|  |
| Initial deposit | 50000.00 |  |  |  |  |
| Net profit | 168959.39 | Gross profit | 204777.37 | Gross loss | -35817.98 |
| Profit factor | 5.72 | Expected payoff | 413.10 |  |  |
| Absolute drawdown | 0.00 | Maximal drawdown | 5602.61 (3.41%) | Relative drawdown | 4.56% (2611.85) |
|  |
| Total trades | 409 | Short positions (won %) | 205 (59.51%) | Long positions (won %) | 204 (61.27%) |
|  | Profit trades (% of total) | 247 (60.39%) | Loss trades (% of total) | 162 (39.61%) |
| Largest | profit trade | 18874.04 | loss trade | -1461.09 |
| Average | profit trade | 829.06 | loss trade | -221.10 |
| Maximum | consecutive wins (profit in money) | 14 (6950.34) | consecutive losses (loss in money) | 7 (-5602.61) |
| Maximal | consecutive profit (count of wins) | 25476.79 (8) | consecutive loss (count of losses) | -5602.61 (7) |
| Average | consecutive wins | 4 | consecutive losses | 3 |

![](https://c.mql5.com/2/14/testergraph_1.gif)

| # | Time | Type | Order | Lots | Price | S / L | T / P | Profit | Balance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2006.02.16 19:51 | buy | 1 | 0.10 | 1.3120 | 0.0000 | 1.3210 |  |  |
| 2 | 2006.02.16 19:52 | sell | 2 | 0.10 | 1.3115 | 0.0000 | 1.3025 |  |  |
| 3 | 2006.02.17 14:06 | sell | 3 | 0.20 | 1.3158 | 0.0000 | 1.3068 |  |
| 4 | 2006.02.20 04:21 | t/p | 3 | 0.20 | 1.3068 | 0.0000 | 1.3068 | 135.19 | 50135.19 |
| 5 | 2006.02.20 04:21 | buy | 4 | 0.20 | 1.3073 | 0.0000 | 1.3163 |  |
| 6 | 2006.02.23 16:15 | close | 2 | 0.10 | 1.3028 | 0.0000 | 1.3025 | 58.02 | 50193.21 |
| 7 | 2006.02.23 16:15 | buy | 5 | 0.40 | 1.3028 | 0.0000 | 1.3118 |  |
| 8 | 2006.02.23 16:16 | sell | 6 | 0.10 | 1.3038 | 0.0000 | 1.2948 |  |
| 9 | 2006.02.23 16:55 | sell | 7 | 0.20 | 1.3081 | 0.0000 | 1.2991 |  |
| 10 | 2006.02.24 10:14 | close | 1 | 0.10 | 1.3114 | 0.0000 | 1.3210 | 3.84 | 50197.05 |
| 11 | 2006.02.24 10:14 | close | 4 | 0.20 | 1.3114 | 0.0000 | 1.3163 | 75.17 | 50272.22 |
| 12 | 2006.02.24 10:14 | close | 5 | 0.40 | 1.3114 | 0.0000 | 1.3118 | 266.53 | 50538.75 |
| 13 | 2006.02.24 10:15 | buy | 8 | 0.10 | 1.3110 | 0.0000 | 1.3200 |  |
| 14 | 2006.02.24 10:36 | sell | 9 | 0.40 | 1.3123 | 0.0000 | 1.3033 |  |
| 15 | 2006.02.24 17:58 | sell | 10 | 0.80 | 1.3167 | 0.0000 | 1.3077 |  |
| 16 | 2006.02.27 01:20 | t/p | 8 | 0.10 | 1.3200 | 0.0000 | 1.3200 | 69.22 | 50607.97 |
| 17 | 2006.02.27 01:20 | buy | 11 | 0.10 | 1.3205 | 0.0000 | 1.3295 |  |
| 18 | 2006.02.27 01:22 | sell | 12 | 1.60 | 1.3211 | 0.0000 | 1.3121 |  |
| 19 | 2006.02.28 12:47 | buy | 13 | 0.20 | 1.3163 | 0.0000 | 1.3253 |  |
| 20 | 2006.02.28 17:24 | close | 6 | 0.10 | 1.3123 | 0.0000 | 1.2948 | -68.52 | 50539.45 |
| 21 | 2006.02.28 17:24 | close | 7 | 0.20 | 1.3123 | 0.0000 | 1.2991 | -71.52 | 50467.93 |
| 22 | 2006.02.28 17:24 | close | 9 | 0.40 | 1.3123 | 0.0000 | 1.3033 | -10.01 | 50457.92 |
| 23 | 2006.02.28 17:24 | close | 10 | 0.80 | 1.3123 | 0.0000 | 1.3077 | 248.21 | 50706.13 |
| 24 | 2006.02.28 17:24 | close | 12 | 1.60 | 1.3123 | 0.0000 | 1.3121 | 1052.91 | 51759.04 |
| 25 | 2006.02.28 17:25 | buy | 14 | 0.40 | 1.3113 | 0.0000 | 1.3203 |  |
| 26 | 2006.02.28 17:26 | sell | 15 | 0.10 | 1.3111 | 0.0000 | 1.3021 |  |
| 27 | 2006.03.01 15:07 | buy | 16 | 0.80 | 1.3064 | 0.0000 | 1.3154 |  |
| 28 | 2006.03.01 18:52 | close | 11 | 0.10 | 1.3150 | 0.0000 | 1.3295 | -39.72 | 51719.31 |
| 29 | 2006.03.01 18:52 | close | 13 | 0.20 | 1.3150 | 0.0000 | 1.3253 | -17.66 | 51701.65 |
| 30 | 2006.03.01 18:52 | close | 14 | 0.40 | 1.3150 | 0.0000 | 1.3203 | 116.76 | 51818.41 |
| 31 | 2006.03.01 18:52 | close | 16 | 0.80 | 1.3150 | 0.0000 | 1.3154 | 523.19 | 52341.60 |
| 32 | 2006.03.01 18:53 | buy | 17 | 0.10 | 1.3162 | 0.0000 | 1.3252 |

etc.

As you can see, the test result confirms the feasability of "refutation of
postulates".

The author of this present article having started trading practices in 2002 and
still considering himself to be a beginner in trading is sure of only two postulations
(axioms, truths, as you prefer) on FOREX:

- the market is not bound to do anything for anybody;
- the price is not bound to move as predicted whoever made this prediction.


Better safe than sorry!

### List of References:

1. Ten Basic Errors of a Newcomer in Trading: An article by Collector at [/en/articles/1418](https://www.mql5.com/en/articles/1418)
2. Torgovaya sistema treidera: faktor uspekha / Pod red.V.I. Safina. – SPb.: Piter, 2004. – 240 p.: In Russian
(Trading System of a Trader: A Success Factor)
3. Yakimkin V.N., Cand.Sc. (Physics/Mathematics). Forex: kak zarabotat' bol'shie den'gi"
\- M.: IKF Omega-L, 2005. - 413 p.: In Russian (Forex: How to Earn Much Money)
4. [https://www.mql5.com/en/code/7097](https://www.mql5.com/en/code/7097)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1424](https://www.mql5.com/ru/articles/1424)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39303)**
(3)


![Vlad Vahnovanu](https://c.mql5.com/avatar/avatar_na2.png)

**[Vlad Vahnovanu](https://www.mql5.com/en/users/vladv)**
\|
17 Aug 2007 at 11:12

It a very good article!Thanks.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
28 Apr 2008 at 18:51

Great articel rsi, thank you very much for it! One thing is just wondered; you say that "...however, the system works at smaller initial deposits, only the
interval of sl should be greater. If the deposit is $1000, the interval should be 300...". I just couldn't find any "interval" for stoploss from your code, so could you please explain a bit more what you mean with this?

Thanks in advance!

![Zerit0](https://c.mql5.com/avatar/avatar_na2.png)

**[Zerit0](https://www.mql5.com/en/users/zerit0)**
\|
23 May 2012 at 23:51

Thanks for sharing!


![Modelling Requotes in Tester and Expert Advisor Stability Analysis](https://c.mql5.com/2/14/246_2.png)[Modelling Requotes in Tester and Expert Advisor Stability Analysis](https://www.mql5.com/en/articles/1442)

Requote is a scourge for many Expert Advisors, especially for those that have rather sensitive conditions of entering/exiting a trade. In the article, a way to check up the EA for the requotes stability is offered.

![Alternative Log File with the Use of HTML and CSS](https://c.mql5.com/2/14/385_10.gif)[Alternative Log File with the Use of HTML and CSS](https://www.mql5.com/en/articles/1432)

In this article we will describe the process of writing a simple but a very powerful library for making the html files, will learn to adjust their displaying and will see how they can be easily implemented and used in your expert or the script.

![How to Develop a Reliable and Safe Trade Robot in MQL 4](https://c.mql5.com/2/14/327_2.png)[How to Develop a Reliable and Safe Trade Robot in MQL 4](https://www.mql5.com/en/articles/1462)

The article deals with the most common errors that occur in developing and using of an Expert Advisor. An exemplary safe automated trading system is described, as well.

![Simultaneous Displaying of the Signals of Several Indicators from the Four Timeframes](https://c.mql5.com/2/14/325_3.png)[Simultaneous Displaying of the Signals of Several Indicators from the Four Timeframes](https://www.mql5.com/en/articles/1461)

While manual trading you have to keep an eye on the values of several indicators. It is a little bit different from mechanical trading. If you have two or three indicators and you have chosen a one timeframe for trading, it is not a complicated task. But what will you do if you have five or six indicators and your trading strategy requires considering the signals on the several timeframes?

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/1424&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083247761152808901)

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