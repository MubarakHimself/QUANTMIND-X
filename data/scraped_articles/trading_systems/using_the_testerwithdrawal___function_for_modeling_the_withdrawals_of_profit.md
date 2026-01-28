---
title: Using the TesterWithdrawal() Function for Modeling the Withdrawals of Profit
url: https://www.mql5.com/en/articles/131
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:56:13.423172
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/131&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083224125947778891)

MetaTrader 5 / Tester


### Introduction

The aim of speculative trading is getting a profit. Usually, when a trade system is tested, all the aspects are analyzed except for the following one - withdrawing a part of earned money for living. Even if the trading is not the only source of money for the trader, the questions about the planned profit for a trade period (month, quarter, year) arise sooner or later. The [TesterWithdrawal()](https://www.mql5.com/en/docs/common/testerwithdrawal) function in MQL5 is the one intended for emulating the withdrawals of money from the account.

### 1\. What Can We Check?

There are opinions about trading as well as trading systems that imply withdrawing excess of money in a risky situation. Only a small part of the total capital is at the trade account; this is a protection from losing all the money as a reason of some unforeseen circumstances.

Practically, it is an example of the Money Management strategies. Many funds that accept investments for management give a certain quota (a part of the total capital) to each trader, but the entire capital is never given to traders. In other words, in addition to the limits on the volume of positions opened, there are limitations on the capital that a trader can command. In the end, each trader has the access only to a small part of assets for making risky speculations.

Another problem is aggressive reinvestment of earned assets during testing. Traders aim to make the balance and equity lines go up vertically during the optimization. Though everybody understands that there are certain rules at each level of money; there is no point in believing that a system profitable at trading with 0.1 lots is good when trading 100 lots.

In addition there is a psychological problem - every practicing trader can confess to intervene in working of the trading system at a certain level of money on the account, though they have promised not to do it. Thus, from this point of view, withdrawing the excesses is a stabilizing factor in trading.

There is a function in MQL5 intended to implement the withdrawal of money during testing of an Expert Advisor. The next section is devoted to it.

### 2\. How to Use the Function?

The [TesterWithdrawal()](https://www.mql5.com/en/docs/common/testerwithdrawal) function is intended only for modeling of withdrawing money from the account in the strategy tester and it doesn't affect the working of Expert Advisor in the normal mode.

```
bool TesterWithdrawal(double money);
```

The input parameter **money** is intended for specifying the amount of money to be withdrawn in the deposit currency. By the returned value you can detect if the operation is successful. Each withdrawal operation performed during testing is shown with the corresponding entry at the "Journal" tab as well as at the "Results" tab as is shown in the figure 1.

![Entries about the successful withdrawal of assets during testing](https://c.mql5.com/2/1/pict-1_eng__2.png)

Figure 1. Entries about the successful withdrawal of assets during testing

In case, the size of withdrawal exceeds the size of [free margin](https://www.mql5.com/en/docs/constants/environment_state/accountinformation) and the withdrawal is not performed, the following entry appears in the "Journal":

![Entry about the unsuccessful withdrawal operation during testing](https://c.mql5.com/2/1/pict-2_eng.png)

Figure 2. Entry about the unsuccessful withdrawal operation during testing

When calling this function, the size of the current [balance](https://www.mql5.com/en/docs/constants/environment_state/accountinformation) and [equity](https://www.mql5.com/en/docs/constants/environment_state/accountinformation) decrease by the amount of withdrawn money. However, the strategy tester doesn't consider the withdrawal when calculating the profit and loss, but it calculates the total sum of withdrawals in the "Withdrawal" rate, as is shown in the figure 3.

![Rate of the total sum of withdrawals in the report](https://c.mql5.com/2/1/pict-3_eng.png)

Figure 3. Rate of the total sum of withdrawals in the report

It should be noted that a reverse function ofdepositing assets to a trade account doesn't exist in MQL5 yet, and unlikely it will be implemented. Who needs systems that require constant increase of deposit?

Next section is devoted to the example of an Expert Advisor not only with a trade strategy implemented, but with a toolkit for modelling money withdrawals using the function we observe.

### 3\. A Guinea Pig for Testing

We are going to conduct the testing on a rather simple Expert Advisor. Those who are interested in the principles of Expert Advisor's operation can find the description of its trade system below.

The maximum and minimum values of the price are calculated at a specified period of 5M data, which is set using the PERIOD input variable. Calculated levels form a horizontal channel with the support (CalcLow variable) and resistance (CalcHigh variable) lines. Pending orders are placed at the borders of the channel. The game is played on the piercing of the channel.

A pending order is placed when the price goes inside the channel for not less than the value of the INSIDE\_LEVEL input variable. In addition, only one position or pending order can be opened in one direction.

If the channel becomes narrower when the levels are recalculated, and the price still stays within it, then the pending order is moved closer to the market price. The step of moving pending orders is set using the ORDER\_STEP input variable.

Size of profit from a deal is specified in the TAKE\_PROFIT input variable, and the maximal loss is specified using STOP\_LOSS.

The Expert Advisor implies moving of the Stop Loss with a step specified in the TRAILING\_STOP input variable.

You can trade both with a fixed lot or with one calculated as a percentage of the deposit. The lot value is set using the LOT input variable, and the method of lot calculation is set using the LOT\_TYPE variable. There is a possibility of correcting the lot (LOT\_CORRECTION = true). In case you trade with a fixed lot and its size exceeds the one allowed for opening a position, the lot size is corrected to the closest allowed value.

For tuning of the algorithm of the Expert Advisor, you can turn on the function of writing the log (WRITE\_LOG\_FILE = true). So that the entries about all trade operations will be written to the text file log.txt.

Let's add several [input variables](https://www.mql5.com/en/docs/basis/variables/inputvariables) to the Expert Advisor for managing the money withdrawals.

The first parameters will be used as a flag of possibility of withdrawing assets.

```
input bool WDR_ENABLE = true; // Allow withdrawal
```

The second parameter will determine the periodicity of withdrawals.

```
enum   wdr_period
  {
   days      = -2, // Day
   weeks     = -1, // Week
   months    =  1, // Month
   quarters  =  3, // Quarter
   halfyears =  6, // Half a year
   years     = 12  // Year
  };
```

```
input wdr_period WDR_PERIOD = weeks; // Periodicity of withdrawals
```

The third parameter will set the amount of money to be withdrawn.

```
input double WDR_VALUE = 1; // Amount of money to be withdrawn
```

The fourth parameter will determine the method of calculation of the amount to be withdrawn.

```
enum lot_type
   {
      fixed,  // Fixed
      percent // Percentage of deposit
   };
```

```
input lot_type WDR_TYPE = percent; // Method of calculation of the withdrawal amount
```

To calculate the amount of one-time withdrawal of money from the account, the **wdr\_val** **ue** is used. The **wdr\_summa** variable is used for calculating the total amount of money to be withdrawn.

Also we'll calculate the total number of successful withdrawal operations using the **wdr\_count** variable. Values of those variables are necessary to form our own report about the result of testing. The entire functionality of withdrawing money is implemented in the function below.

```
//--- Function of withdrawing assets from the account
//+------------------------------------------------------------------+
bool TimeOfWithDrawal()
//+------------------------------------------------------------------+
  {
   if(!WDR_ENABLE) return(false); // exit if withdrawal is prohibited

   if( tick.time > dt_debit + days_delay * DAY) // periodic withdrawals with specified period
     {
      dt_debit = dt_debit + days_delay * DAY;
      days_delay = Calc_Delay();// Updating the value of period-number of days between withdrawal operations

      if(WDR_TYPE == fixed) wdr_value = WDR_VALUE;
      else wdr_value = AccountInfoDouble(ACCOUNT_BALANCE) * 0.01 * WDR_VALUE;

      if(TesterWithdrawal(wdr_value))
        {
         wdr_count++;
         wdr_summa = wdr_summa + wdr_value;
         return(true);
        }
     }
   return(false);
  }
```

Now we need to prepare our Expert Advisor for testing in the optimization mode. There will be several optimization parameters, so let's declare an input variable using which we will choose the necessary parameter.

```
enum opt_value
{
   opt_total_wdr,      // Total sum of withdrawal
   opt_edd_with_wdr,   // Drawdown with consideration of withdrawal
   opt_edd_without_wdr // Drawdown without consideration of withdrawal
};
```

```
input opt_value OPT_PARAM = opt_total_wdr; // Optimization by the parameter
```

Also we need to add another significant function to the Expert Advisor - [OnTester()](https://www.mql5.com/en/docs/basis/function/events#ontester); the returned value of this function determines the optimization criterion.

```
//--- Displaying information about testing
//+------------------------------------------------------------------+
double OnTester(void)
//+------------------------------------------------------------------+
  {
   //--- Calculation of parameters for the report
   CalculateSummary(initial_deposit);
   CalcEquityDrawdown(initial_deposit, true, false);
   //--- Creation of the report
   GenerateReportFile("report.txt");

   //--- Returned value is the optimization criterion
   if (OPT_PARAM == opt_total_wdr) return(wdr_summa);
   else return(RelEquityDrawdownPercent);
  }
```

The next section is devoted to the detailed consideration of testing of our Expert Advisor.

### 4\. Testing the Expert Advisor

For the integrity of test, first of all we need to obtain the results of testing of our Expert Advisor with the function of withdrawing disabled. To do it, before testing, set the "Allow withdrawal" parameter to **false** as is shown in the figure 5.

For those who are interested in repeating the results of testing, the settings and values of the input parameters of the Expert Advisor are given below in the figures 4 and 5.

![Settings for testing the Expert Advisor](https://c.mql5.com/2/1/pict-4_eng.png)

Figure 4. Settings for testing the Expert Advisor

![Parameters of the Expert Advisor with the function of withdrawing disabled](https://c.mql5.com/2/1/pict-5_eng.png)

Figure 5. Parameters of the Expert Advisor with the function of withdrawing disabled

The results obtained at the end of testing are shown in the figures 6 and 7.

![Change of balance for 6 months of test working of the Expert Advisor ](https://c.mql5.com/2/1/pict-6_eng.png)

Figure 6. Change of balance for 6 months of test working of the Expert Advisor

![The table of results of working of the Expert Advisor](https://c.mql5.com/2/1/pict-7_eng.png)

Figure 7. The table of results of working of the Expert Advisor

The parameter that is interesting to us is the relative drawdown of equity. In this case, it is equal to 4.75%.

Now let's check how this parameter changes if we enable the money withdrawals. Let's conduct testing with different amounts and periodicity of withdrawals.

The figure 8 demonstrates the parameters of the Expert Advisor with optimization and withdrawals enabled; and the figure 9 demonstrates the results of such testing.

![The parameter of the Expert Advisor with optimization and withdrawals enabled](https://c.mql5.com/2/1/pict-8_eng.png)

Figure 8. The parameter of the Expert Advisor with optimization and withdrawals enabled

![The results of calculation of the relative drawdown of equity](https://c.mql5.com/2/1/pict-9_eng.png)

Figure 9. The results of calculation of the relative drawdown of equity

The results of testing are a bit surprising, since the drawdown is lower with the withdrawal periodicity of one day or one week than with the withdrawals disabled at all.

In addition, the line of daily drawdown increases to 100% and then goes back to 8% and continues descending. To interpret the results correctly, you need to know how the relative drawdown of equity is calculated in the strategy tester. This is the subject of the next section.

### 5\. Calculation of the Drawdown of Equity

Inquisitive minds will be curious to know how the calculation of equity is performed in the strategy tester, and how to calculate this parameter on their own.

If the [TesterWithdrawal()](https://www.mql5.com/en/docs/common/testerwithdrawal) function is not used in your Expert Advisor, then the calculation of the parameter we analyze has no difference from its calculation in MetaTrader 4; it is described in the ["What the Numbers in the Expert Testing Report Mean"](https://www.mql5.com/en/articles/1486) article; and the source code of it and many other parameters of the resultant tester report is given at the ["How to Evaluate the Expert Testing Results"](https://www.mql5.com/en/articles/1403) article.

A visual description of calculation of the relative and maximal drawdown in the strategy tester is the  figure 10,  below.

![Calculation of the drawdowns of equity without consideration of withdrawals](https://c.mql5.com/2/1/pict-10_eng.png)

Figure 10. Calculation of the drawdowns of equity without consideration of withdrawals

During its working, the strategy tester determines the current maximum and minimum values of equity. With appearing of a new equity maximum, which is marked on the chart with the blue checkmark, the maximal and minimal drawdowns are recalculated and the biggest value is saved to be displayed in the resultant report.

The important thing is the last recalculation of the parameters at the end of testing, because there can arise a situation when the last unregistered extremum of equity gives the maximum value of drawdown. Changes of maximal values of drawdowns are shown with the blue and red colors respectively. Grey color represents the drawdown registered at every coming of new maximum.

It should be noted that the call of the [TesterWithDrawal()](https://www.mql5.com/en/docs/common/testerwithdrawal) function changes the algorithm of calculation of drawdowns in the strategy tester. The difference from the previous variant is the recalculation of drawdown values not only at the coming of new equity maximums, but at the withdrawals of assets as well. The visual demonstration of it is at the figure 11.

![Calculation of drawdowns considering withdrawals](https://c.mql5.com/2/1/pict-11_eng.png)

Figure 11. Calculation of drawdowns considering withdrawals

The moment of withdrawing assets are shown with the green checkmarks at the picture above. Money withdrawals are rather frequent, thus it doesn't allow the maximum drawdown values, determined by the extremums on the chart, to be fixed. As a result, the resultant drawdown with the consideration of withdrawals can be lower than the one without the consideration, what has been noted at the figure 9.

If the money withdrawals are much bigger than the growth of equity as the result of getting profit, this can lead to low rates of drawdowns. The reason why is this situation doesn't allow the extremums to form on the chart; and in the limit it will look like a straight descending line, where the relative drawdown tents to zero. This effect has started arising with the daily withdrawals of more than 1000$ at the chart shown in the figure 9.

Let's reproduce the calculation of maximal and relative drawdown of equity in MQL5 by combining the entire algorithm into the single procedure CalcEquityDrawdown as described below.

```
double RelEquityDrawdownPercent; // relative drawdown of equity in percentage terms
double MaxEquityDrawdown;        // maximal drawdown of equity
```

```
//--- Calculation of the drawdown of equity
//+------------------------------------------------------------------+
void CalcEquityDrawdown(double initial_deposit, // initial deposit
                        bool finally)          // flag of calculation that registers extremums
//+------------------------------------------------------------------+
  {
   double drawdownpercent;
   double drawdown;
   double equity;
   static double maxpeak = 0.0, minpeak = 0.0;

   //--- exclusion of consideration of profit withdrawals for the calculation of drawdowns
   if(wdr_ignore) equity = AccountInfoDouble(ACCOUNT_EQUITY) + wdr_summa;
   else equity = AccountInfoDouble(ACCOUNT_EQUITY);

   if(maxpeak == 0.0) maxpeak = equity;
   if(minpeak == 0.0) minpeak = equity;

   //--- check of conditions of extremum
   if((maxpeak < equity)||(finally))
    {
      //--- calculation of drawdowns
      drawdown = maxpeak - minpeak;
      drawdownpercent = drawdown / maxpeak * 100.0;

      //--- Saving maximal values of drawdowns
      if(MaxEquityDrawdown < drawdown) MaxEquityDrawdown = drawdown;
      if(RelEquityDrawdownPercent < drawdownpercent) RelEquityDrawdownPercent = drawdownpercent;

      //--- nulling the values of extremums
      maxpeak = equity;
      minpeak = equity;
    }

   if(minpeak > equity) minpeak = equity;
 }
```

To keep the calculations precise, we need to call the procedure written above from the body of our Expert Advisor every time a new tick comes and with the parameters given below.

```
CalcEquityDrawdown(initial_deposit, false);
```

In addition, to get the reliable values of the drawdowns, it should also be called at the end of the Expert Advisor in the [OnTester()](https://www.mql5.com/en/docs/basis/function/events#ontester) function with the **finally = true** parameters that indicate that the calculation is over. And if the current unfixed drawdown is greater than the maximal registered, it will replace the maximum one and will be shown in the resultant report.

```
CalcEquityDrawdown(initial_deposit, true);
```

If the algorithm of withdrawals is implemented in an Expert Advisor, then for the correct calculation of drawdowns you need to call the CalcEquityDrawdown function with the **finally** **=true** parameter every time a withdrawal is performed. The order of calling can be as following:

```
//--- Withdrawing assets and calculating drawdowns of equity
if(TimeOfWithDrawal())
    CalcEquityDrawdown(initial_deposit, true);
else
    CalcEquityDrawdown(initial_deposit, false);
```

To be sure in the correctness of methodology described above, let's compare the calculated data with the data of the strategy tester. To do it, we need to make the [OnTester()](https://www.mql5.com/en/docs/basis/function/events#ontester) function return the value of the parameter we want to check - the relative drawdown of equity that is stored in the RelEquityDrawdownPercent variable.

It is implemented in the code of the Expert Advisor by setting the "Drawdown with consideration of withdrawal" value for the "Optimization by the parameter" input parameter. The rest of parameters should be left without changes as is shown in the figure 8. The results of such testing are shown in the figure 12.

[![Comparison of the results of our calculations with the data of the strategy tester](https://c.mql5.com/2/1/pict-12_eng__1.png)](https://c.mql5.com/2/1/pict-12_eng.png)

Figure 12. Comparison of the results of our calculations with the data of the strategy tester

The comparison of the obtained results proves that the algorithm of calculation of the relative drawdown is correct.

Let's add another variant to the calculation of drawdown. In it, we are going to exclude the influence of money withdrawals on equity; and observe the change of the relative drawdown of equity.

For this purpose, let's add a variable to the Expert Advisor, we will use it to determine if it is necessary to consider the withdrawals when calculating the parameters of the Expert Advisor.

```
bool wdr_ignore; // calculation of rate without the consideration of withdrawals
```

The value of **wdr\_ignore** is equal to **true** if the "Optimization by the parameter" variable is set to "Drawdown without consideration of withdrawal".

Besides it, you need to correct the procedure of calculation of drawdown CalcEquityDrawdown by adding the processing of that parameter to it as is shown below.

```
if (wdr_ignore)
  equity = AccountInfoDouble(ACCOUNT_EQUITY) + wdr_summa;
else
  equity = AccountInfoDouble(ACCOUNT_EQUITY);
```

Now everything is ready for obtaining new values of the drawdowns. Let's perform testing with the new algorithm of calculation enabled. The result of testing is shown in the figure 13.

![The results of the calculation of drawdown without the consideration of withdrawals](https://c.mql5.com/2/1/pict-13_eng.png)

Figure 13. The results of the calculation of drawdown without the consideration of withdrawals

The results show that neither the fact of withdrawing assets, nor the amount of money withdrawn depend on the drawdown. Thus, we have obtained a pure factor that is not affected by the [TesterWithDrawal()](https://www.mql5.com/en/docs/common/testerwithdrawal) function. However, for using this kind of calculation, we need to correct the values of profit and loss, because their real values have changed, and these factors are not correct in the resultant report of the tester.

Let's calculate the profit, loss and their ratio (profitability) and save the obtained values as a report in a text file. The procedure of calculation of the listed parameters is given below.

```
double SummaryProfit;     // Total net profit
double GrossProfit;       // Gross profit
double GrossLoss;         // Gross loss
double ProfitFactor;      // Profitability
```

```
//--- Calculation of parameters for the report
//+------------------------------------------------------------------+
void CalculateSummary(double initial_deposit)
//+------------------------------------------------------------------+
  {
   double drawdownpercent, drawdown;
   double maxpeak = initial_deposit,
          minpeak = initial_deposit,
          balance = initial_deposit;

   double profit = 0.0;

   //--- Select entire history
   HistorySelect(0, TimeCurrent());
   int trades_total = HistoryDealsTotal();

   //--- Searching the deals in the history
   for(int i=0; i < trades_total; i++)
     {
      long ticket = HistoryDealGetTicket(i);
      long type   = HistoryDealGetInteger(ticket, DEAL_TYPE);

      //--- Initial deposit is not considered
      if((i == 0)&&(type == DEAL_TYPE_BALANCE)) continue;

      //--- Calculation of profit
      profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) +
                 HistoryDealGetDouble(ticket, DEAL_COMMISSION) +
               HistoryDealGetDouble(ticket, DEAL_SWAP);

      balance += profit;

      if(minpeak > balance) minpeak = balance;

      //---
      if((!wdr_ignore)&&(type != DEAL_TYPE_BUY)&&(type != DEAL_TYPE_SELL)) continue;

      //---
      if(profit < 0) GrossLoss   += profit;
      else           GrossProfit += profit;
      SummaryProfit += profit;
     }

   if(GrossLoss < 0.0) GrossLoss *= -1.0;
   //--- Profitability
   if(GrossLoss > 0.0) ProfitFactor = GrossProfit / GrossLoss;
  }
```

The function for generating the report in a text file is given below.

```
//--- Forming the report
//+------------------------------------------------------------------+
void GenerateReportFile(string filename)
//+------------------------------------------------------------------+
  {
   string str, msg;

   ResetLastError();
   hReportFile = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(hReportFile != INVALID_HANDLE)
     {

      StringInit(str,65,'-'); // separator

      WriteToReportFile(str);
      WriteToReportFile("| Period of testing: " + TimeToString(first_tick.time, TIME_DATE) + " - " +
                        TimeToString(tick.time,TIME_DATE) + "\t\t\t|");
      WriteToReportFile(str);

      //----
      WriteToReportFile("| Initial deposit \t\t\t"+DoubleToString(initial_deposit, 2));
      WriteToReportFile("| Total net profit    \t\t\t"+DoubleToString(SummaryProfit, 2));
      WriteToReportFile("| Gross profit     \t\t\t"+DoubleToString(GrossProfit, 2));
      WriteToReportFile("| Gross loss      \t\t\t"+DoubleToString(-GrossLoss, 2));
      if(GrossLoss > 0.0)
         WriteToReportFile("| Profitability       \t\t\t"+DoubleToString(ProfitFactor,2));
      WriteToReportFile("| Relative drawdown of equity \t"+
                        StringFormat("%1.2f%% (%1.2f)", RelEquityDrawdownPercent, MaxEquityDrawdown));

      if(WDR_ENABLE)
        {
         StringInit(msg, 10, 0);
         switch(WDR_PERIOD)
           {
            case day:     msg = "day";    break;
            case week:    msg = "week";  break;
            case month:   msg = "month";   break;
            case quarter: msg = "quarter"; break;
            case year:    msg = "year";     break;
           }

         WriteToReportFile(str);
         WriteToReportFile("| Periodicity of withdrawing       \t\t" + msg);

         if(WDR_TYPE == fixed) msg = DoubleToString(WDR_VALUE, 2);
         else msg = DoubleToString(WDR_VALUE, 1) + " % from deposit " + DoubleToString(initial_deposit, 2);

         WriteToReportFile("| Amount of money withdrawn     \t\t" + msg);
         WriteToReportFile("| Number of withdrawal operations \t\t" + IntegerToString(wdr_count));
         WriteToReportFile("| Withdrawn from account          \t\t" + DoubleToString(wdr_summa, 2));
        }

      WriteToReportFile(str);
      WriteToReportFile(" ");

      FileClose(hReportFile);
     }
  }
```

The result of execution of this function is the creation of a text file filled with information as is shown in the figure 14.

![File of the report generated by the GenerateReportFile procedure](https://c.mql5.com/2/1/pict-14_eng.png)

Figure 14. File of the report generated by the GenerateReportFile procedure

The function is written in a manner to add each new report to the existing contents of the file. Owing to this fact, we can compare the values of the results of testing with different input parameters of the Expert Advisor.

As you can see from the report, when calculating without the consideration of withdrawals (lower table) the total net profit is smaller and the gross loss is greater by the amount of money withdrawn, comparing to the calculations of the tester (upper table). It should be noted that to get the real value of profit and loss when withdrawing assets in the strategy tester, you should subtract the value of the "Withdrawal" parameter from the total net profit and add it to the gross loss.

### 6\. Analysis of Result

On the basis of results obtained from all the conducted experiments, we can make several conclusions:

1. Using the [TesterWithDrawal()](https://www.mql5.com/en/docs/common/testerwithdrawal) function leads to the changes in the algorithm of calculating the drawdowns in the strategy tester. Comparing several different Expert Advisors by the value of the relative drawdown can be incorrect if one of them contains a mechanism of withdrawing money. Using this function, you can make a pragmatic calculation of how much money you can periodically take from the account depending on the specified, acceptable percentage of the drawdown of equity.
2. Usage of this function can be implemented as a synthetic destabilizing factor of trading used for checking the stability of working of your Expert Advisor and adjusting the logic of the code responsible for the money management. If your Expert Advisor has logic of making decisions on the basis of balance or equity levels, then the usage of this function gives additional opportunities for testing and tuning.
3. When recalculating the relative drawdown without the consideration of withdrawals, using the algorithm described in the article, you can obtain a pure value of the relative drawdown which is not affected by the usage of this function.

### Conclusion

This article covers the usage of the [TesterWithdrawal()](https://www.mql5.com/en/docs/common/testerwithdrawal) function for modeling the process of withdrawing assets from an account, and its influence on the algorithm of calculation of the drawdown of equity in the strategy tester.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/131](https://www.mql5.com/ru/articles/131)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/131.zip "Download all attachments in the single ZIP archive")

[testexpert.mq5](https://www.mql5.com/en/articles/download/131/testexpert.mq5 "Download testexpert.mq5")(33.95 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to create bots for Telegram in MQL5](https://www.mql5.com/en/articles/2355)
- [Guide to writing a DLL for MQL5 in Delphi](https://www.mql5.com/en/articles/96)
- [Connection of Expert Advisor with ICQ in MQL5](https://www.mql5.com/en/articles/64)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1978)**
(14)


![Henrique Vilela](https://c.mql5.com/avatar/2020/8/5F3D6A6A-2AAD.jpeg)

**[Henrique Vilela](https://www.mql5.com/en/users/hvilela)**
\|
13 Jul 2017 at 07:33

**Andrey Voytenko:**

It should be noted that a reverse function ofdepositingassets to a trade account doesn't exist in MQL5 yet, and unlikely it will be implemented. Who needs systems that require constant increase of deposit?

I do! :)

Also tried to pass a negative value to the [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") with no success. It's ignored. :/

![Sumit Dutta](https://c.mql5.com/avatar/2018/4/5AE0D924-07B8.jpg)

**[Sumit Dutta](https://www.mql5.com/en/users/loveyourockg)**
\|
22 Apr 2022 at 22:08

\\*\\*\\*

![Sumit Dutta](https://c.mql5.com/avatar/2018/4/5AE0D924-07B8.jpg)

**[Sumit Dutta](https://www.mql5.com/en/users/loveyourockg)**
\|
22 Apr 2022 at 22:09

**Sumit Dutta [#](https://www.mql5.com/en/forum/1978#comment_32703849):**

\\*\\*\\*

its not work whats is wrong told me or solve it

![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
23 Apr 2022 at 05:51

**Sumit Dutta [#](https://www.mql5.com/en/forum/1978/page2#comment_32704152):**

its not work whats is wrong told me or solve it

Please paste your code correctly: click the ![Code](https://c.mql5.com/3/385/Code.png)button, then paste your code into the popup window.

![Eblawsch](https://c.mql5.com/avatar/avatar_na2.png)

**[Eblawsch](https://www.mql5.com/en/users/eblawsch)**
\|
25 Jul 2022 at 23:55

Very interesting article. Thanks a lot for this very promising idea!

Is it possible to program these function as a kind of addition for an already existing compiled EA (bought here in the mq5 market)?

Or does these code needs to be implemented always directly inside of the EA's code?

Thanks and best regards,

Eblawsch

![Interview with Alexander Topchylo (ATC 2010)](https://c.mql5.com/2/0/35.png)[Interview with Alexander Topchylo (ATC 2010)](https://www.mql5.com/en/articles/527)

Alexander Topchylo (Better) is the winner of the Automated Trading Championship 2007. Alexander is an expert in neural networks - his Expert Advisor based on a neural network was on top of best EAs of year 2007. In this interview Alexander tells us about his life after the Championships, his own business and new algorithms for trading systems.

![How to Quickly Create an Expert Advisor for Automated Trading Championship 2010](https://c.mql5.com/2/0/Fast_Expert_Advisor_Writing_MQL5.png)[How to Quickly Create an Expert Advisor for Automated Trading Championship 2010](https://www.mql5.com/en/articles/148)

In order to develop an expert to participate in Automated Trading Championship 2010, let's use a template of ready expert advisor. Even novice MQL5 programmer will be capable of this task, because for your strategies the basic classes, functions, templates are already developed. It's enough to write a minimal amount of code to implement your trading idea.

![Several Ways of Finding a Trend in MQL5](https://c.mql5.com/2/0/Determine_Trend_MQL5.png)[Several Ways of Finding a Trend in MQL5](https://www.mql5.com/en/articles/136)

Any trader would give a lot for opportunity to accurately detect a trend at any given time. Perhaps, this is the Holy Grail that everyone is looking for. In this article we will consider several ways to detect a trend. To be more precise - how to program several classical ways to detect a trend by means of MQL5.

![20 Trade Signals in MQL5](https://c.mql5.com/2/0/20_Trading_Signals_MQL5__1.png)[20 Trade Signals in MQL5](https://www.mql5.com/en/articles/130)

This article will teach you how to receive trade signals that are necessary for a trade system to work. The examples of forming 20 trade signals are given here as separate custom functions that can be used while developing Expert Advisors. For your convenience, all the functions used in the article are combined in a single mqh include file that can be easily connected to a future Expert Advisor.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/131&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083224125947778891)

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