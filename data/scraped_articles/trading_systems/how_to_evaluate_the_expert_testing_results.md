---
title: How to Evaluate the Expert Testing Results
url: https://www.mql5.com/en/articles/1403
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T14:00:01.265581
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/1403&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083270644738562095)

MetaTrader 4 / Tester


    First, a few words about
testing procedure. Before starting to test, the testing subsystem loads
the expert, sets its parameters previously defined by the user, and
calls the init() function. Then the Tester plays through the generated
sequence and calls the start() function every time. When the test
sequence is exhausted, the Tester calls the deinit() function. At this,
the entire trading history generated during testing is available. The
expert efficiency can be analyzed at this moment.


The CalculateSummary function below provides calculation of test
results, i.e., the data given in the standard report of the Strategy
Tester.

```
void CalculateSummary(double initial_deposit)
  {
   int    sequence=0, profitseqs=0, lossseqs=0;
   double sequential=0.0, prevprofit=EMPTY_VALUE, drawdownpercent, drawdown;
   double maxpeak=initial_deposit, minpeak=initial_deposit, balance=initial_deposit;
   int    trades_total=HistoryTotal();
//---- initialize summaries
   InitializeSummaries(initial_deposit);
//----
   for(int i=0; i<trades_total; i++)
     {
      if(!OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)) continue;
      int type=OrderType();
      //---- initial balance not considered
      if(i==0 && type==OP_BALANCE) continue;
      //---- calculate profit
      double profit=OrderProfit()+OrderCommission()+OrderSwap();
      balance+=profit;
      //---- drawdown check
      if(maxpeak<balance)
        {
         drawdown=maxpeak-minpeak;
         if(maxpeak!=0.0)
           {
            drawdownpercent=drawdown/maxpeak*100.0;
            if(RelDrawdownPercent<drawdownpercent)
              {
               RelDrawdownPercent=drawdownpercent;
               RelDrawdown=drawdown;
              }
           }
         if(MaxDrawdown<drawdown)
           {
            MaxDrawdown=drawdown;
            if(maxpeak!=0.0) MaxDrawdownPercent=MaxDrawdown/maxpeak*100.0;
            else MaxDrawdownPercent=100.0;
           }
         maxpeak=balance;
         minpeak=balance;
        }
      if(minpeak>balance) minpeak=balance;
      if(MaxLoss>balance) MaxLoss=balance;
      //---- market orders only
      if(type!=OP_BUY && type!=OP_SELL) continue;
      SummaryProfit+=profit;
      SummaryTrades++;
      if(type==OP_BUY) LongTrades++;
      else             ShortTrades++;
      //---- loss trades
      if(profit<0)
        {
         LossTrades++;
         GrossLoss+=profit;
         if(MinProfit>profit) MinProfit=profit;
         //---- fortune changed
         if(prevprofit!=EMPTY_VALUE && prevprofit>=0)
           {
            if(ConProfitTrades1<sequence ||
               (ConProfitTrades1==sequence && ConProfit2<sequential))
              {
               ConProfitTrades1=sequence;
               ConProfit1=sequential;
              }
            if(ConProfit2<sequential ||
               (ConProfit2==sequential && ConProfitTrades1<sequence))
              {
               ConProfit2=sequential;
               ConProfitTrades2=sequence;
              }
            profitseqs++;
            AvgConWinners+=sequence;
            sequence=0;
            sequential=0.0;
           }
        }
      //---- profit trades (profit>=0)
      else
        {
         ProfitTrades++;
         if(type==OP_BUY)  WinLongTrades++;
         if(type==OP_SELL) WinShortTrades++;
         GrossProfit+=profit;
         if(MaxProfit<profit) MaxProfit=profit;
         //---- fortune changed
         if(prevprofit!=EMPTY_VALUE && prevprofit<0)
           {
            if(ConLossTrades1<sequence ||
               (ConLossTrades1==sequence && ConLoss2>sequential))
              {
               ConLossTrades1=sequence;
               ConLoss1=sequential;
              }
            if(ConLoss2>sequential ||
               (ConLoss2==sequential && ConLossTrades1<sequence))
              {
               ConLoss2=sequential;
               ConLossTrades2=sequence;
              }
            lossseqs++;
            AvgConLosers+=sequence;
            sequence=0;
            sequential=0.0;
           }
        }
      sequence++;
      sequential+=profit;
      //----
      prevprofit=profit;
     }
//---- final drawdown check
   drawdown=maxpeak-minpeak;
   if(maxpeak!=0.0)
     {
      drawdownpercent=drawdown/maxpeak*100.0;
      if(RelDrawdownPercent<drawdownpercent)
        {
         RelDrawdownPercent=drawdownpercent;
         RelDrawdown=drawdown;
        }
     }
   if(MaxDrawdown<drawdown)
     {
      MaxDrawdown=drawdown;
      if(maxpeak!=0) MaxDrawdownPercent=MaxDrawdown/maxpeak*100.0;
      else MaxDrawdownPercent=100.0;
     }
//---- consider last trade
   if(prevprofit!=EMPTY_VALUE)
     {
      profit=prevprofit;
      if(profit<0)
        {
         if(ConLossTrades1<sequence ||
            (ConLossTrades1==sequence && ConLoss2>sequential))
           {
            ConLossTrades1=sequence;
            ConLoss1=sequential;
           }
         if(ConLoss2>sequential ||
            (ConLoss2==sequential && ConLossTrades1<sequence))
           {
            ConLoss2=sequential;
            ConLossTrades2=sequence;
           }
         lossseqs++;
         AvgConLosers+=sequence;
        }
      else
        {
         if(ConProfitTrades1<sequence ||
            (ConProfitTrades1==sequence && ConProfit2<sequential))
           {
            ConProfitTrades1=sequence;
            ConProfit1=sequential;
           }
         if(ConProfit2<sequential ||
            (ConProfit2==sequential && ConProfitTrades1<sequence))
           {
            ConProfit2=sequential;
            ConProfitTrades2=sequence;
           }
         profitseqs++;
         AvgConWinners+=sequence;
        }
     }
//---- collecting done
   double dnum, profitkoef=0.0, losskoef=0.0, avgprofit=0.0, avgloss=0.0;
//---- average consecutive wins and losses
   dnum=AvgConWinners;
   if(profitseqs>0) AvgConWinners=dnum/profitseqs+0.5;
   dnum=AvgConLosers;
   if(lossseqs>0)   AvgConLosers=dnum/lossseqs+0.5;
//---- absolute values
   if(GrossLoss<0.0) GrossLoss*=-1.0;
   if(MinProfit<0.0) MinProfit*=-1.0;
   if(ConLoss1<0.0)  ConLoss1*=-1.0;
   if(ConLoss2<0.0)  ConLoss2*=-1.0;
//---- profit factor
   if(GrossLoss>0.0) ProfitFactor=GrossProfit/GrossLoss;
//---- expected payoff
   if(ProfitTrades>0) avgprofit=GrossProfit/ProfitTrades;
   if(LossTrades>0)   avgloss  =GrossLoss/LossTrades;
   if(SummaryTrades>0)
     {
      profitkoef=1.0*ProfitTrades/SummaryTrades;
      losskoef=1.0*LossTrades/SummaryTrades;
      ExpectedPayoff=profitkoef*avgprofit-losskoef*avgloss;
     }
//---- absolute drawdown
   AbsoluteDrawdown=initial_deposit-MaxLoss;
  }
```


For calculations to be correct, the value of the initial deposit must
be known. For this, in the init() function, the AccountBalance()
function must be called that will give the balance value at the testing
start.

```
void init()
  {
   ExtInitialDeposit=AccountBalance();
  }
```


In the above CalculateSummary function, as well as in a standard
report, the profit is calculated in the deposit currency. Other trade
results, such as the "Largest profit trade" or the "Maximal consecutive
loss" that are calculated on profit basis, are also measured in money
terms. It is easy to recalculate the profit in points then.

```
...
      //---- market orders only
      if(type!=OP_BUY && type!=OP_SELL) continue;
      //---- calculate profit in points
      profit=(OrderClosePrice()-OrderOpenPrice())/MarketInfo(OrderSymbol(),MODE_POINT);
      SummaryProfit+=profit;
...
```

    The obtained results can be written into the report file using the WriteReport function.

```
void WriteReport(string report_name)
  {
   int handle=FileOpen(report_name,FILE_CSV|FILE_WRITE,'\t');
   if(handle<1) return;
//----
   FileWrite(handle,"Initial deposit           ",InitialDeposit);
   FileWrite(handle,"Total net profit          ",SummaryProfit);
   FileWrite(handle,"Gross profit              ",GrossProfit);
   FileWrite(handle,"Gross loss                ",GrossLoss);
   if(GrossLoss>0.0)
      FileWrite(handle,"Profit factor             ",ProfitFactor);
   FileWrite(handle,"Expected payoff           ",ExpectedPayoff);
   FileWrite(handle,"Absolute drawdown         ",AbsoluteDrawdown);
   FileWrite(handle,"Maximal drawdown          ",
                     MaxDrawdown,
                     StringConcatenate("(",MaxDrawdownPercent,"%)"));
   FileWrite(handle,"Relative drawdown         ",
                     StringConcatenate(RelDrawdownPercent,"%"),
                     StringConcatenate("(",RelDrawdown,")"));
   FileWrite(handle,"Trades total                 ",SummaryTrades);
   if(ShortTrades>0)
      FileWrite(handle,"Short positions(won %)    ",
                        ShortTrades,
                        StringConcatenate("(",100.0*WinShortTrades/ShortTrades,"%)"));
   if(LongTrades>0)
      FileWrite(handle,"Long positions(won %)     ",
                        LongTrades,
                        StringConcatenate("(",100.0*WinLongTrades/LongTrades,"%)"));
   if(ProfitTrades>0)
      FileWrite(handle,"Profit trades (% of total)",
                        ProfitTrades,
                        StringConcatenate("(",100.0*ProfitTrades/SummaryTrades,"%)"));
   if(LossTrades>0)
      FileWrite(handle,"Loss trades (% of total)  ",
                        LossTrades,
                        StringConcatenate("(",100.0*LossTrades/SummaryTrades,"%)"));
   FileWrite(handle,"Largest profit trade      ",MaxProfit);
   FileWrite(handle,"Largest loss trade        ",-MinProfit);
   if(ProfitTrades>0)
      FileWrite(handle,"Average profit trade      ",GrossProfit/ProfitTrades);
   if(LossTrades>0)
      FileWrite(handle,"Average loss trade        ",-GrossLoss/LossTrades);
   FileWrite(handle,"Average consecutive wins  ",AvgConWinners);
   FileWrite(handle,"Average consecutive losses",AvgConLosers);
   FileWrite(handle,"Maximum consecutive wins (profit in money)",
                     ConProfitTrades1,
                     StringConcatenate("(",ConProfit1,")"));
   FileWrite(handle,"Maximum consecutive losses (loss in money)",
                     ConLossTrades1,
                     StringConcatenate("(",-ConLoss1,")"));
   FileWrite(handle,"Maximal consecutive profit (count of wins)",
                     ConProfit2,
                     StringConcatenate("(",ConProfitTrades2,")"));
   FileWrite(handle,"Maximal consecutive loss (count of losses)",
                     -ConLoss2,
                     StringConcatenate("(",ConLossTrades2,")"));
//----
   FileClose(handle);
  }
```

    An example of how these functions are used to generate a report is given below.

```
void deinit()
  {
   if(!IsOptimization())
     {
      if(!IsTesting()) ExtInitialDeposit=CalculateInitialDeposit();
      CalculateSummary(ExtInitialDeposit);
      WriteReport("MACD_Sample_Report.txt");
     }
  }
```


You can see that the reports can be generated not only after testing,
but at deinitialization of live expert advisor. You may ask how to get
to know the initial deposit size if the account history was downloaded
in the terminal only partially (for example, only one-month history was
requested in the Account History tab). The CalculateInitialDeposit
function helps to solve this problem.

```
double CalculateInitialDeposit()
  {
   double initial_deposit=AccountBalance();
//----
   for(int i=HistoryTotal()-1; i>=0; i--)
     {
      if(!OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)) continue;
      int type=OrderType();
      //---- initial balance not considered
      if(i==0 && type==OP_BALANCE) break;
      if(type==OP_BUY || type==OP_SELL)
        {
         //---- calculate profit
         double profit=OrderProfit()+OrderCommission()+OrderSwap();
         //---- and decrease balance
         initial_deposit-=profit;
        }
      if(type==OP_BALANCE || type==OP_CREDIT)
         initial_deposit-=OrderProfit();
     }
//----
   return(initial_deposit);
  }
```

    This is the way in which the reports are generated in MetaTrader 4 Client Terminal.

![](https://c.mql5.com/2/13/testerreportr281s29.png)

    It can be compared to data calculated using the exposed program.

```
Initial deposit             10000
Total net profit            -13.16
Gross profit                20363.32
Gross loss                  20376.48
Profit factor               0.99935416
Expected payoff             -0.01602923
Absolute drawdown           404.28
Maximal drawdown            1306.36 (11.5677%)
Relative drawdown           11.5966%    (1289.78)
Trades total                    821
Short positions(won %)      419 (24.821%)
Long positions(won %)       402 (31.592%)
Profit trades (% of total)  231 (28.1364%)
Loss trades (% of total)    590 (71.8636%)
Largest profit trade        678.08
Largest loss trade          -250
Average profit trade        88.15290043
Average loss trade          -34.53640678
Average consecutive wins    1
Average consecutive losses  4
Maximum consecutive wins (profit in money)  4   (355.58)
Maximum consecutive losses (loss in money)  15  (-314.74)
Maximal consecutive profit (count of wins)  679.4   (2)
Maximal consecutive loss (count of losses)  -617.16 (8)
```


The SummaryReport.mq4 file attached to this article is recommended to
be placed in the experts\\include directory and inserted using the
#include directive.

```
#include <SummaryReport.mq4>

double ExtInitialDeposit;
```

Translated from Russian by MetaQuotes Software Corp.

Original article: [/ru/articles/1403](https://www.mql5.com/ru/articles/1403)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1403](https://www.mql5.com/ru/articles/1403)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1403.zip "Download all attachments in the single ZIP archive")

[SummaryReport.mq4](https://www.mql5.com/en/articles/download/1403/SummaryReport.mq4 "Download SummaryReport.mq4")(12 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Testing of Expert Advisors in the MetaTrader 4 Client Terminal: An Outward Glance](https://www.mql5.com/en/articles/1417)
- [How to Use Crashlogs to Debug Your Own DLLs](https://www.mql5.com/en/articles/1414)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39192)**
(10)


![Jinsong Zhang](https://c.mql5.com/avatar/2010/6/4C2450DB-041A.jpg)

**[Jinsong Zhang](https://www.mql5.com/en/users/song_song)**
\|
15 Sep 2010 at 16:52

mark

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
30 Jan 2011 at 02:35

Good. Then I can use Excel or MATLAB to draw a picture.


![blackmore](https://c.mql5.com/avatar/avatar_na2.png)

**[blackmore](https://www.mql5.com/en/users/blackmore)**
\|
15 Nov 2011 at 12:40

does this work for optimisation also?


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
5 Sep 2012 at 04:42

Hi Slawa et al,

Very much a NuB here.

I need a Back Test and Optimization Report Summary that doesn't include all of the trades. I'm hoping to add in the picture of the graph as well. But this looks like just what I need thanks (< 8)

Before I start, what and where are the two programs that generate the Mt4 Optimization and Back [Test reports](https://www.mql5.com/en/articles/403 "Article \"Visualize a Strategy in the MetaTrader 5 Tester\"")? Are they compiled (C++) MQL4 and are they directly accessible by us? It would be great if they were as I just need to eliminate the large lists of all of the trades and could possibly temporarily disable this aspect of them. Or just have to separate report Summary versions: One with the trades included and one without.

But \*.mqh files go into the include folder and not \*.mq4 files done't they? I tried it both as an mq4 and an mqh file and in both instances the compiler checks them out as being just fine themselves. However as soon as the EA is compiled, it goes crazy as it encounters 2 _init_, 2 _deinit_ sections and flags these as errors and a whole LOT of: ')' - unbalanced right parenthesis errors. I've looked it over and I can't see how it can work in the setup with putting basically what are 2 complete EAs together and combining the code into one _init_ section and one _deinit_ section.

So I tried 'Copying and pasting' everything from your ReportSummary program that is in the _init_ and _deinit_ sections into an EA and 'remarked these out in your report EA and I can get these two functioning alright this way, but then it does the same thing with the 'start' function which being the body of the program is huge. Thus it would basically melding the two EAs together. But I know that include files can and do work very well for such things. But aren't they usually just a bunch of indivdiual function definitions etc so that one can just call them from any EA at any time without having to remake them every time?

Their is little doubt that you and the other folks here know a LOT more about this than I do and that I am making a fundamental error right from the very start? I'm fairly certain that what I have been trying: combining them both into one physical EA, is not the problem and approach to use. What fundamental mistake am I making right at the very start of this please that thwarts the whole thing for me?

![SidhuKid](https://c.mql5.com/avatar/avatar_na2.png)

**[SidhuKid](https://www.mql5.com/en/users/sidhukid)**
\|
15 Apr 2022 at 10:34

Hi Slawa, Does this function also exist for MQL5?

![Genetic Algorithms: Mathematics](https://c.mql5.com/2/13/133_1.png)[Genetic Algorithms: Mathematics](https://www.mql5.com/en/articles/1408)

Genetic (evolutionary) algorithms are used for optimization purposes. An example of such purpose can be neuronet learning, i.e., selection of such weight values that allow reaching the minimum error. At this, the genetic algorithm is based on the random search method.

![Requirements Applicable to Articles Offered for Publishing at MQL4.com](https://c.mql5.com/2/17/99_6.gif)[Requirements Applicable to Articles Offered for Publishing at MQL4.com](https://www.mql5.com/en/articles/1402)

Requirements Applicable to Articles Offered for Publishing at MQL4.com

![One-Minute Data Modelling Quality Rating](https://c.mql5.com/2/17/89_1.gif)[One-Minute Data Modelling Quality Rating](https://www.mql5.com/en/articles/1513)

One-Minute Data Modelling Quality Rating

![Expert Advisor Sample](https://c.mql5.com/2/17/82_1.gif)[Expert Advisor Sample](https://www.mql5.com/en/articles/1510)

The principles of MQL4-programs development are shown on sample of creating a simple Expert Advisor system based on the standard MACD indicator.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lvwfcakjnglvgprscfgbrlrngjrdkvhg&ssn=1769252399399770159&ssn_dr=0&ssn_sr=0&fv_date=1769252399&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1403&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Evaluate%20the%20Expert%20Testing%20Results%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925239993974430&fz_uniq=5083270644738562095&sv=2552)

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