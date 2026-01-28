---
title: How Reliable is Night Trading?
url: https://www.mql5.com/en/articles/1373
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:54:48.937705
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/1373&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083207959690876663)

MetaTrader 4 / Trading systems


### Introduction

Over the last six months, we have seen a lot of Expert Advisors that earned good money without virtually any risk involved, by simply trading in the night on EURCAD, GBPCAD, EURCHF and other currency pairs. Trading on Alpari's PAMM accounts that employ this tactics has demonstrated indeed amazing results. Negative feedback on using this trading system is however not uncommon. The purpose of this article is to see if night flat trading is really profitable and to identify all hidden dangers of such trading on a real account.

### Theory

The figure below shows typical night flats. The image is not the best for the given strategy but this choice is intentional. The night flat boundaries are quite vague. We can say that for EURCHF the flat starts at 18:00-20:00 and ends at 10:00-14:00 server time. We will take it as the period from 18:00 to 10:00.

The night flat usually lasts during the Pacific and Asian trading session and is characterized by low volatility of the majority of currency pairs.

**![](https://c.mql5.com/2/13/1_1_eng.gif)**

### Implementation

First off, let's roughly define how the Expert Advisor for night flat trading should operate.

1) Deal opening time – not earlier than a specified hour

2) Deal closing time – not later than a specified hour

3) TP – quite close, from 10 to 50 points

4) SL – greater than TP, from 30 to 70 points

5) Market entry – price is near the flat boundaries

6) Flat boundaries are determined in the last hours before the beginning of the flat

7) Number of night deals may be limited which is due to the fact that entering into Sell or Buy more than once and exiting with profit is rare; the second market entry often results in closing with a loss by the end of the night flat.

The above points can be implemented like this:

```
// External variables
extern double    Lots=1;
extern int       h_beg=20;
extern int       h_end=10;
extern int       TakeProfit=20;
extern int       StopLoss=90;

// Auxiliary variables
double max;
double min;
int slippage=5;
int magik=5;
int pos=0;

//Counter of deals over the night session
int Buy_count=0;
int Sell_count=0;

//Function for closing an order
bool CloseOrder()
{
   int ticket,i;
   double Price_close;
   int err;
   int count=0;
   int time;
       for(i=OrdersTotal()-1;i>=0;i--)
       {
          if (OrderSelect(i,SELECT_BY_POS,MODE_TRADES))
            if (OrderSymbol()==Symbol())
                if(OrderMagicNumber()==magik)
                   {
                     if(OrderType()==OP_BUY)
                        Price_close=NormalizeDouble(Bid,Digits);
                     if(OrderType()==OP_SELL)
                        Price_close=NormalizeDouble(Ask,Digits);

                     if(OrderClose(OrderTicket(),OrderLots(),Price_close,slippage))
                           return (true);
                   }
      }
return(false);
}

//Taking a decision
int GetAction()
{
   int TotalDeals;
   int type=0;
   int N=OrdersTotal();

// Counting current positions
   for(int i = N - 1; i >= 0 ; i--)
      {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES))
         if( OrderSymbol()==Symbol())
            TotalDeals++;
            type=OrderType();
       }

   if (TotalDeals==0)
   {
      pos=0;
   }
   else
   {
      if (type==OP_BUY)
         pos=2;
      if (type==OP_SELL)
         pos=-2;
   }

// Beginning of the night session, determination of the flat boundaries by two preceding bars
   if (Hour()==h_beg)
   {
      max=MathMax(High[1], High[2]);
      min=MathMin(Low[1],Low[2]);
      Buy_count=0;
      Sell_count=0;
   }

// End of session, closing of all positions and disabling trading operations by the Expert Advisor
   if ((Hour()>=h_end)&&(Hour()<h_beg))
   {
      Buy_count=1;
      Sell_count=1;
      max=0;
      min=0;
      if (TotalDeals!=0)
         CloseOrder();
   }

// Checking for position opening
      if ((Bid>max)&&(max!=0)&&(pos==0)&&(Sell_count==0))
            pos=-1;

      if ((Ask<min)&&(min!=0)&&(pos==0)&&(Buy_count==0))
            pos=1;

   return (0);
}

//Processed at every tick
int start()
  {
   int action;
   double profit;
   double stop=0;
   double price=0;
   int ticket=-1;

   GetAction();
   if (pos==1)
         {
               stop=NormalizeDouble(Ask-StopLoss*Point,Digits);
               profit=(min+TakeProfit*Point);
               pos=2;

               while (ticket<0)
               {
                  if (!IsTradeContextBusy( ))
                     ticket=OrderSend(Symbol(),OP_BUY,Lots,Ask,slippage,0,0,NULL,magik,0,Green);
                  if (ticket>0)
                     OrderModify(ticket,0,stop,profit,0,Blue);

                  if(ticket<0)
                     Print("OrderSend failed with error #",GetLastError());

                  Sleep(5000);
                  RefreshRates();
               }
               Buy_count=1;
         }

   if (pos==-1)
         {
               stop=NormalizeDouble(Bid+StopLoss*Point,Digits);
               profit=(max-TakeProfit*Point);
               pos=-2;

               while (ticket<0)
               {
                  if (!IsTradeContextBusy( ))
                     ticket=OrderSend(Symbol(),OP_SELL,Lots,Bid,slippage,0,0,NULL,magik,0,Green);

                  if (ticket>0)
                     OrderModify(ticket,0,stop,profit,0,Blue);

                  if(ticket<0)
                     Print("OrderSend failed with error #",GetLastError());

                  Sleep(5000);
                  RefreshRates();
               }
               Sell_count=1;
        }
   return(0);
  }
```

### Testing

As can be seen on the below chart, some deals are truly amazing: buying virtually at LOW and selling virtually at HIGH of the night price range. There are some exceptions that are usually closed after the set time.

[![](https://c.mql5.com/2/13/2_small.gif)](https://c.mql5.com/2/13/2.gif)

The testing shows that the given Expert Advisor would be able to earn more than 30% over a couple of months trading with one lot with the deposit of $10,000. If we further increase the risks, we will be able to see how trading on Alpari's PAMM accounts could yield results of 1000% over just a couple of months.

[![](https://c.mql5.com/2/13/1_2_small.gif)](https://c.mql5.com/2/13/1_2.gif)

### Practical Application

Of course, it's not all roses, otherwise we would have seen a dramatic increase in dollar millionaires over the last 6 months to a year. What stands in the way of earning superprofits when doing night trading?

1) Night spreads are 1.5 to 2 times wider for the currencies that are especially good for night trading – EURCHF, GBPCAD and EURCAD.

2) Upon large dramatic profits trading conditions in some Dealing Centers may become considerably more unfavorable

3) One may come across a series of large losing trades, e.g. upon intervention on EURCHF/USDCHF

4) TP values need to be constantly adjusted, mostly downwards. And while in winter we could take profit at 40 points for a deal on EURCAD, now this value is around 20.

The Expert Advisor for night trading provided in the article is very basic and can be made considerably more complex. This can be done in many ways, including:

1) scaling in;

2) volatility analysis;

3) news analysis (to avoid intervention risks);

4) analysis of major currency pairs underlying the cross currency pairs (e.g., for EURCHF those will be EURUSD and USDCHF);

5) more complex entry algorithms.

The template provided in the article can be used in real work. It earned around 50% per annum over 6 months during which it was used. And even though this Expert Advisor itself is currently of little usefulness, this article can give impulses to new ideas that will potentially be used by the forum users and the author of this article.

### Conclusion

Night trading offers good profit opportunities in winter, while getting trickier in spring and even more so in summer. Given the trend, it is difficult to make any statements regarding autumn. However this strategy has the right to exist as the deposit on a real account was increased through its application.

The future will tell whether the strategy has outlived its usefulness or not. Being fundamentally very basic, the Expert Advisor provided in the article can nevertheless be shelved and used again when stable night flats are seen again.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1373](https://www.mql5.com/ru/articles/1373)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Easy Stock Market Trading with MetaTrader](https://www.mql5.com/en/articles/1566)
- [Price Forecasting Using Neural Networks](https://www.mql5.com/en/articles/1482)
- [Automated Choice of Brokerage Company for an Efficient Operation of Expert Advisors](https://www.mql5.com/en/articles/1476)
- [How to Develop a Reliable and Safe Trade Robot in MQL 4](https://www.mql5.com/en/articles/1462)
- [How Not to Fall into Optimization Traps?](https://www.mql5.com/en/articles/1434)
- [Construction of Fractal Lines](https://www.mql5.com/en/articles/1429)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39135)**
(4)


![sixty8doors](https://c.mql5.com/avatar/avatar_na2.png)

**[sixty8doors](https://www.mql5.com/en/users/sixty8doors)**
\|
13 Sep 2013 at 17:46

Very interesting! I'm having very good returns trading at night while sleeping. Im in Pacific time, which is when Europe is awake. I set up my trades and go to bed. In the morning Im usually pleasantly surprised.

I would love to give this a try. How do I ago about testing this? Do I copy/paste the code and save it as an MQ4 file and put into my experts folder?

Many Thanks! Dave

![savitara](https://c.mql5.com/avatar/2015/11/563B7F94-D60E.jpg)

**[savitara](https://www.mql5.com/en/users/savitara)**
\|
7 Nov 2015 at 18:05

**sixty8doors:**

Very interesting! I'm having very good returns trading at night while sleeping. Im in Pacific time, which is when Europe is awake. I set up my trades and go to bed. In the morning Im usually pleasantly surprised.

I would love to give this a try. How do I ago about testing this? Do I copy/paste the code and save it as an MQ4 file and put into my experts folder?

Many Thanks! Dave

Its easy to create MQ4 file. Open [Meta Trader](https://www.mql5.com/en/blogs/tags/metals "All about traded groups of metals") 4 and Click Navigator (Ctrl+N).Right click Expert Advisor and choose Create in Meta Editor. When the Meta Editor opens choose expert Advisor template. Then click Next to get a window where you can give a name to your file.Click next until you get an edit environment for your file. Delete the contents in the file (Ctrl+A) Then copy the program code from the source article and paste in this new file. Click Compile and run. You will be back in MetaTrader. Check in Navigator if your new file name is visible.  Now you can run it in tester. Happy testing


![Dmitri Mikhalev](https://c.mql5.com/avatar/2019/1/5C421EEC-51A3.jpg)

**[Dmitri Mikhalev](https://www.mql5.com/en/users/nasatech)**
\|
21 Nov 2019 at 13:50

Hi!

I am new to MQL and did the EA and it seems to work when testing, and that's great and all. I am really thankful!

... But as I said, I am kinda new and would love some more explanation on everything.

I guess I could start by asking my own questions, and maybe, if you have time I'd love to read the answers! :)

So...

1\. the integers ticket, err and

time that are defined within the function for closing orders. They are not being used, am I
correct on that? I have commented them out and nothing bad seems to happen from that.

2\. The "Auxillary" double variables, max and min
are for calculating the boundries by two proceeding bars... could you explain that part step by step? what it means and what one would do if one
wanted to use 3 bars or 6 bars, etc.?

3\. Even though the extern int TakeProfit is saying 20 (or whatever it
might be) the take profits during testing can be something completely different. I am sure it depends on something?

Also, when setting up expert properties in MT4, the settings look something like this (image below). There are 4 different columns -
"Value", "Start", "Step" and "Stop"... What are they, decided upon in the code? Even if they aren't thre because of anything in the code, if
somebody has answers, I would still like to know!

[![exp](https://c.mql5.com/3/299/exp__1.PNG)](https://c.mql5.com/3/299/exp.PNG "https://c.mql5.com/3/299/exp.PNG")

Thank you in advance!

Best regards!

![Dmitri Mikhalev](https://c.mql5.com/avatar/2019/1/5C421EEC-51A3.jpg)

**[Dmitri Mikhalev](https://www.mql5.com/en/users/nasatech)**
\|
21 Nov 2019 at 13:57

Oh!

and also... why is there a "int slippage = 5" in the auxiliary variables? what is it for?


![Building an Automatic News Trader](https://c.mql5.com/2/0/cover.png)[Building an Automatic News Trader](https://www.mql5.com/en/articles/719)

This is the continuation of Another MQL5 OOP class article which showed you how to build a simple OO EA from scratch and gave you some tips on object-oriented programming. Today I am showing you the technical basics needed to develop an EA able to trade the news. My goal is to keep on giving you ideas about OOP and also cover a new topic in this series of articles, working with the file system.

![MQL5 Cookbook: Writing the History of Deals to a File and Creating Balance Charts for Each Symbol in Excel](https://c.mql5.com/2/0/avatar11.png)[MQL5 Cookbook: Writing the History of Deals to a File and Creating Balance Charts for Each Symbol in Excel](https://www.mql5.com/en/articles/651)

When communicating in various forums, I often used examples of my test results displayed as screenshots of Microsoft Excel charts. I have many times been asked to explain how such charts can be created. Finally, I now have some time to explain it all in this article.

![Simple Methods of Forecasting Directions of the Japanese Candlesticks](https://c.mql5.com/2/17/836_34.png)[Simple Methods of Forecasting Directions of the Japanese Candlesticks](https://www.mql5.com/en/articles/1374)

Knowing the direction of the price movement is sufficient for getting positive results from trading operations. Some information on the possible direction of the price can be obtained from the Japanese candlesticks. This article deals with a few simple approaches to forecasting the direction of the Japanese candlesticks.

![Money Management Revisited](https://c.mql5.com/2/17/801_12.gif)[Money Management Revisited](https://www.mql5.com/en/articles/1367)

The article deals with some issues arising when traders apply various money management systems to Forex trading. Experimental data obtained from performing trading deals using different money management (MM) methods is also described.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/1373&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083207959690876663)

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