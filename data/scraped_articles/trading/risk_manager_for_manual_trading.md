---
title: Risk manager for manual trading
url: https://www.mql5.com/en/articles/14340
categories: Trading, Trading Systems
relevance_score: -2
scraped_at: 2026-01-24T14:16:09.265242
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/14340&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083451368372444150)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/14340#n1)
- [Defining the functionality](https://www.mql5.com/en/articles/14340#n2)
- [Input parameters and class constructor](https://www.mql5.com/en/articles/14340#n3)
- [Working with risk limit periods](https://www.mql5.com/en/articles/14340#n4)
- [Controlling the use of limits](https://www.mql5.com/en/articles/14340#n5)
- [Class event handler](https://www.mql5.com/en/articles/14340#n6)
- [Mechanism for controlling daily target profit](https://www.mql5.com/en/articles/14340#n7)
- [Defining a method for launching monitoring in the EA structure](https://www.mql5.com/en/articles/14340#n8)
- [The final implementation and possibilities for extending the class](https://www.mql5.com/en/articles/14340#n9)
- [Usage example](https://www.mql5.com/en/articles/14340#n10)
- [Conclusion](https://www.mql5.com/en/articles/14340#n11)

### Introduction

Hello everyone! In this article we will continue to talk about risk management methodology. In the previous article [Balancing risk when trading multiple instruments simultaneously](https://www.mql5.com/en/articles/14163/), we talked about the basic concepts of risk. Now we will implement from scratch the basic Risk Manager Class for safe trading. We will also see how limiting risks in trading systems affects the effectiveness of trading strategies.

Risk Manager was my first class, which I wrote in 2019 shortly after I learned the basics of programming. At that time, I understood from my own experience that the psychological state of a trader greatly influences the effectiveness of trading, especially when it comes to the "consistency" and "impartiality" of trading decision making. Gambling, emotional transactions and inflating risks in an attempt to cover the losses as quickly as possible can drain any account, even if you use an effective trading strategy that has shown very good results in tests.

The purpose of this article is to show that risk control using a risk manager increases its effectiveness and reliability. To confirm this thesis, we will create a simple base risk manager class for manual trading from scratch and test it using a very simple fractal breakout strategy.

### Defining the functionality

When implementing our algorithm specifically for manual trading, we will only implement control over time risk limits for the day, week and month. Once the actual loss amount reaches or exceeds the limits set by the user, the EA must automatically close all open positions and inform the user about the impossibility of further trading. It should be noted here that the information will be purely "advisory in nature", it will be displayed in the comment line in the lower left corner of the chart with the running EA. This is because we are creating a risk manager specifically for manual trading, so, "if absolutely necessary", the user can remove this EA from the chart at any time and continue trading. However, I would really not recommend doing this, because if the market goes against you, it is better to return to trading the next day and avoid large losses rather, trying to figure out what exactly went wrong in your manual trading. If you integrate this class into your algorithmic trading, you will need to implement the restriction on sending orders when the limit is reached and, preferably, integrate this class directly into the EA structure. We'll talk about this in more detail a little further.

### Input parameters and class constructor

We decided that we will only implement risk control by period and the criterion of achieving the daily profit rate. To do this, we introduce several variables of the type [double](https://www.mql5.com/en/docs/basis/types/double) with memory class modifier [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) for the user to manually enter risk values as a percentage of the deposit for each period of time, as well as the target daily profit percentage to lock in profits. To indicate the control of target daily profit, we introduce an additional variable of the type [bool](https://www.mql5.com/en/docs/basis/types/integer) for the ability to enable/disable this functionality if the trader wants to consider each entry separately and is confident that there is no correlation between the selected instruments. This type of switch variable is also referred to as "flag". Let's declare the following code at the global level. For convenience, we previously "wrapped" it in a named block using the [group](https://www.mql5.com/en/docs/basis/variables/inputvariables#group) keyword.

```
input group "RiskManagerBaseClass"
input double inp_riskperday    = 1;          // risk per day as a percentage of deposit
input double inp_riskperweek   = 3;          // risk per week
input double inp_riskpermonth  = 9;          // risk per month
input double inp_plandayprofit = 3;          // target daily profit

input bool dayProfitControl = true;          // whether to close positions after reaching daily profit
```

The declared variables are initialized with default values according to the following logic. We will start from daily risk, since this class works best for intraday trading, but can also be used for medium-term trading and investing. Obviously, if you trade medium-term or as an investor, then it makes no sense for you to control intraday risk and you can set the same values for the daily and weekly risks. Furthermore, if you only do long-term investments, you can set all limit values equal to a monthly drawdown. Here we will look at the logic of the default parameters for intraday trading.

We decided that we would be comfortable trading with a daily risk of 1% of the deposit. If the daily limit is exceeded, we close the terminal until tomorrow. Next, we define the weekly limit as follows. There are usually 5 trading days in a week, which means if we get 3 losing days in a row, then we stop trading until the beginning of the next week. Simply because it is more likely that you did not understand the market this week, or something has changed and if you continue trading, you will accumulate such a large loss during this period that you will not be able to cover it even at the expense of the next week. A similar logic applies to setting a monthly limit when trading intraday. We accept the condition that if we had 3 unprofitable weeks in a month, it is better not to trade the fourth, since it will take a lot of time to "improve" the yield curve at the expense of future periods. We also do not want to "scare" investors with a large loss for a separate month.

We set the size of the target daily profit based on the daily risk, taking into account the characteristics of your trading system. What to consider here. First, whether you trade correlated instruments, how often your trading system gives entry signals, whether you trade with fixed proportions between stop loss and take profit for each individual transaction, or the size of the deposit. I would like to note that I HIGHLY STRONGLY DO NOT RECOMMEND trading without a stop loss and without a risk manager at the same time. Losing you deposit is just a matter of time in this case. Therefore, we either set stops for each trade separately, or use a risk manager to limit the risk by period. In our current example of default parameters, I set the conditions for daily profit as 1 to 3 relative to daily risk. It is also better to use these parameters alongside the mandatory setting of risk-profitability for EACH trade through the ratio of stop loss and take profit, also 1 to 3 (take profit is greater than stop loss).

The structure of our limits can be depicted as follows.

![Figure 1. Limit structure](https://c.mql5.com/2/72/limits.png)

Figure 1. Limit structure

Next, we declare our custom data type RiskManagerBase using the [class](https://www.mql5.com/en/docs/basis/types/classes) keyword. The input parameters will need to be stored within our custom RiskManagerBase class. Since our input parameters are measured in percentages, while limits are tracked in the deposit currency, we need to enter several corresponding fields of type [double](https://www.mql5.com/en/docs/basis/types/double) with the [protected](https://www.mql5.com/en/docs/basis/types/classes) access modifier to our custom class.

```
protected:

   double    riskperday,                     // risk per day as a percentage of deposit
             riskperweek,                    // risk per week as a percentage of deposit
             riskpermonth,                   // risk per month as a percentage of deposit
             plandayprofit                   // target daily profit as a percentage of deposit
             ;

   double    RiskPerDay,                     // risk per day in currency
             RiskPerWeek,                    // risk per week in currency
             RiskPerMonth,                   // risk per month in currency
             StartBalance,                   // account balance at the EA start time, in currency
             StartEquity,                    // account equity at the limit update time, in currency
             PlanDayEquity,                  // target account equity value per day, in currency
             PlanDayProfit                   // target daily profit, in currency
             ;

   double    CurrentEquity,                  // current equity value
             CurrentBallance;                // current balance
```

For the convenience of calculating risk limits by period in the deposit currency, based on the input parameters, we will declare the RefreshLimits() method inside our class, also with the access modifier [protected](https://www.mql5.com/en/docs/basis/types/classes). Let's describe this method outside the class as follows. We will provide for the future the type of return value of the type [bool](https://www.mql5.com/en/docs/basis/types/integer) in case we need to expand our method with the ability to check the correctness of the obtained data. For now, we describe the method in the following form.

```
//+------------------------------------------------------------------+
//|                        RefreshLimits                             |
//+------------------------------------------------------------------+
bool RiskManagerBase::RefreshLimits(void)
  {
   CurrentEquity    = NormalizeDouble(AccountInfoDouble(ACCOUNT_EQUITY),2);   // request current equity value
   CurrentBallance  = NormalizeDouble(AccountInfoDouble(ACCOUNT_BALANCE),2);  // request current balance

   StartBalance     = NormalizeDouble(AccountInfoDouble(ACCOUNT_BALANCE),2);  // set start balance
   StartEquity      = NormalizeDouble(AccountInfoDouble(ACCOUNT_EQUITY),2);   // request current equity value

   PlanDayProfit    = NormalizeDouble(StartEquity * plandayprofit/100,2);     // target daily profit, in currency
   PlanDayEquity    = NormalizeDouble(StartEquity + PlanDayProfit/100,2);     // target equity, in currency

   RiskPerDay       = NormalizeDouble(StartEquity * riskperday/100,2);        // risk per day in currency
   RiskPerWeek      = NormalizeDouble(StartEquity * riskperweek/100,2);       // risk per week in currency
   RiskPerMonth     = NormalizeDouble(StartEquity * riskpermonth/100,2);      // risk per month in currency

   return(true);
  }
```

A convenient way is to call this method in code every time we need to recalculate limit values when changing time periods, as well as when initially changing field values when calling the class constructor. We write the following code in the class constructor to initialize the starting values of the fields.

```
//+------------------------------------------------------------------+
//|                        RiskManagerBase                           |
//+------------------------------------------------------------------+
RiskManagerBase::RiskManagerBase()
  {
   riskperday         = inp_riskperday;                                 // set the value for the internal variable
   riskperweek        = inp_riskperweek;                                // set the value for the internal variable
   riskpermonth       = inp_riskpermonth;                               // set the value for the internal variable
   plandayprofit      = inp_plandayprofit;                              // set the value for the internal variable

   RefreshLimits();                                                     // update limits
  }
```

After deciding on the logic of the input parameters and the starting data state for our class, we move on to implementing the accounting of limits.

### Working with risk limit periods

To work with risk limit periods, we will need additional variable with the [protected](https://www.mql5.com/en/docs/basis/types/classes) access type. First, let's declare our own flag for each period in the form of [bool](https://www.mql5.com/en/docs/basis/types/integer) type variables, which will store data on reaching the set risk limits, as well as the main flag, which will inform about the possibility of continuing trading only if all limits are available at the same time. This is necessary to avoid the situation when the monthly limit has already been exceeded, but there is still a daily limit and thus trading is allowed. This will limit trading when any time limit is reached before the next time period. We will also need variables of the same type to control the daily profit and the onset of a new trading day. Plus we will add fields of type [double](https://www.mql5.com/en/docs/basis/types/double) to store information on actual profit and loss for each period: day, week and month. Additionally, we will provide separate values for swap and commission in trading operations.

```
   bool              RiskTradePermission;    // general variable - whether opening of new trades is allowed
   bool              RiskDayPermission;      // flag prohibiting trading if daily limit is reached
   bool              RiskWeekPermission;     // flag to prohibit trading if daily limit is reached
   bool              RiskMonthPermission;    // flag to prohibit trading if monthly limit is reached

   bool              DayProfitArrive;        // variable to control if daily target profit is achieved
   bool              NewTradeDay;            // variable for a new trading day

   //--- actual limits
   double            DayorderLoss;           // accumulated daily loss
   double            DayorderProfit;         // accumulated daily profit
   double            WeekorderLoss;          // accumulated weekly loss
   double            WeekorderProfit;        // accumulated weekly profit
   double            MonthorderLoss;         // accumulated monthly loss
   double            MonthorderProfit;       // accumulated monthly profit
   double            MonthOrderSwap;         // monthly swap
   double            MonthOrderCommis;       // monthly commission
```

We specifically do not include expenses from commissions and swaps in the losses of the corresponding periods, so that in the future we can separate losses incurred from the decisions making tool from losses related to the commission and swap requirements of different brokers. Now that we have declared the corresponding fields of our class, let's move on to controlling the use of limits.

### Controlling the use of limits

To control the actual use of limits, we will need to handle events associated with the onset of each new period, as well as events associated with the appearance of completed trading operations. To correctly keep track of actually used limits, we will announce the internal method ForOnTrade() in the [protected](https://www.mql5.com/en/docs/basis/types/classes) access are of our class.

First, we will need to provide variables in the method to account for the current time, as well as the start time of the day, week and month. For these purposes, we will use a special predefined data type of the [struct](https://www.mql5.com/en/docs/basis/types/classes) structure type in the [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) format. We will immediately initialize them with the current terminal time in the following form.

```
   MqlDateTime local, start_day, start_week, start_month;               // create structure to filter dates
   TimeLocal(local);                                                    // fill in initially
   TimeLocal(start_day);                                                // fill in initially
   TimeLocal(start_week);                                               // fill in initially
   TimeLocal(start_month);                                              // fill in initially
```

Note that to initially initialize the current time, we use the predefined function [TimeLocal()](https://www.mql5.com/en/docs/dateandtime/timelocal) instead of [TimeCurrent()](https://www.mql5.com/en/docs/dateandtime/timecurrent) because the first one uses local time, and the second one takes time from the last tick received from the broker, which may cause incorrect accounting of limits due to the difference in time zones between different brokers. Next, we need to reset the start time of each period to get the start date values for each of them. We will do this by accessing the public fields of our structures as follows.

```
//--- reset to have the report from the beginning of the period
   start_day.sec     = 0;                                               // from the day beginning
   start_day.min     = 0;                                               // from the day beginning
   start_day.hour    = 0;                                               // from the day beginning

   start_week.sec    = 0;                                               // from the week beginning
   start_week.min    = 0;                                               // from the week beginning
   start_week.hour   = 0;                                               // from the week beginning

   start_month.sec   = 0;                                               // from the month beginning
   start_month.min   = 0;                                               // from the month beginning
   start_month.hour  = 0;                                               // from the month beginning
```

To correctly obtain data for the week and month, we need to define the logic for finding the beginning of the week and month. In the case of a month, everything is quite simple, we know that every month starts on the first day. Dealing with a week is a little more complicated because there is no specific reporting point and the date will change every time. Here we can use the special day\_of\_week field of the [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) structure. It allows you to get the number of the week day from the current date starting from zero. Knowing this value, we can easily find out the start date of the current week as follows.

```
//--- determining the beginning of the week
   int dif;                                                             // day of week difference variable
   if(start_week.day_of_week==0)                                        // if this is the first day of the week
     {
      dif = 0;                                                          // then reset
     }
   else
     {
      dif = start_week.day_of_week-1;                                   // if not the first, then calculate the difference
      start_week.day -= dif;                                            // subtract the difference at the beginning of the week from the number of the day
     }

//---month
   start_month.day         = 1;                                         // everything is simple with the month
```

Now that we have the exact start dates of each period relative to the current moment, we can move on to requesting historical data on transactions carried out on the account. Initially, we will need to declare the necessary variables to account for closed orders and reset the values of the variables in which the financial results of transactions will be collected for each selected period.

```
//---
   uint     total  = 0;                                                 // number of selected trades
   ulong    ticket = 0;                                                 // order number
   long     type;                                                       // order type
   double   profit = 0,                                                 // order profit
            commis = 0,                                                 // order commission
            swap   = 0;                                                 // order swap

   DayorderLoss      = 0;                                               // daily loss without commission
   DayorderProfit    = 0;                                               // daily profit
   WeekorderLoss     = 0;                                               // weekly loss without commission
   WeekorderProfit   = 0;                                               // weekly profit
   MonthorderLoss    = 0;                                               // monthly loss without commission
   MonthorderProfit  = 0;                                               // monthly profit
   MonthOrderCommis  = 0;                                               // monthly commission
   MonthOrderSwap    = 0;                                               // monthly swap
```

We will request historical data on closed orders through the predefined terminal function [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect). The parameters of this function will use the dates we received earlier for each period. To do this, we will need to bring our [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) variable type to the type required by the parameters [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) function, which is [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime). For this, we will use a predefined terminal function [StructToTime()](https://www.mql5.com/en/docs/dateandtime/structtotime). We will request data on transactions in the same way, substituting the necessary values for the beginning and end of the required period.

After each call of the [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) function, we need to get the number of selected orders using the predefined terminal function [HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal) and put this value into our local variable total. After getting the number of closed deals, we can organize a loop with the [for](https://www.mql5.com/en/docs/basis/operators/for) operator, requesting the number of each order through the predefined terminal function [HistoryDealGetTicket()](https://www.mql5.com/en/docs/trading/historydealgetticket). This will allow us to access the data of each order. We will get access to the data of each order using predefined terminal functions [HistoryDealGetDouble()](https://www.mql5.com/en/docs/trading/historydealgetdouble) and [HistoryDealGetInteger()](https://www.mql5.com/en/docs/trading/historydealgetinteger), passing the previously received order number to them. We will need to specify the corresponding deal property identifier from the [ENUM\_DEAL\_PROPERTY\_INTEGER](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_property_integer) and [ENUM\_DEAL\_PROPERTY\_DOUBLE](https://www.mql5.com/en/docs/trading/historydealgetdouble) enumerations. We will also need to add a filter via a Boolean selection operator [if](https://www.mql5.com/en/docs/basis/operators/if) to consider only trades from trading operations by checking for the [DEAL\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) and [DEAL\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) values from the [ENUM\_DEAL\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) enumeration to filter out other account operations, such as balance transactions and bonus accruals. So, we will end up with the following code for selecting the data.

```
//--- now select data by --==DAY==--
   HistorySelect(StructToTime(start_day),StructToTime(local));          // select required history
//--- check
   total  = HistoryDealsTotal();                                        // number number of selected deals
   ticket = 0;                                                          // order number
   profit = 0;                                                          // order profit
   commis = 0;                                                          // order commission
   swap   = 0;                                                          // order swap

//--- for all deals
   for(uint i=0; i<total; i++)                                          // loop through all selected orders
     {
      //--- try to get deals ticket
      if((ticket=HistoryDealGetTicket(i))>0)                            // get the number of each in order
        {
         //--- get deals properties
         profit    = HistoryDealGetDouble(ticket,DEAL_PROFIT);          // get data on financial results
         commis    = HistoryDealGetDouble(ticket,DEAL_COMMISSION);      // get data on commission
         swap      = HistoryDealGetDouble(ticket,DEAL_SWAP);            // get swap data
         type      = HistoryDealGetInteger(ticket,DEAL_TYPE);           // get data on operation type

         if(type == DEAL_TYPE_BUY || type == DEAL_TYPE_SELL)            // if the deal is form a trading operatoin
           {
            if(profit>0)                                                // if financial result of current order is greater than 0,
              {
               DayorderProfit += profit;                                // add to profit
              }
            else
              {
               DayorderLoss += MathAbs(profit);                         // if loss, add up
              }
           }
        }
     }

//--- now select data by --==WEEK==--
   HistorySelect(StructToTime(start_week),StructToTime(local));         // select the required history
//--- check
   total  = HistoryDealsTotal();                                        // number number of selected deals
   ticket = 0;                                                          // order number
   profit = 0;                                                          // order profit
   commis = 0;                                                          // order commission
   swap   = 0;                                                          // order swap

//--- for all deals
   for(uint i=0; i<total; i++)                                          // loop through all selected orders
     {
      //--- try to get deals ticket
      if((ticket=HistoryDealGetTicket(i))>0)                            // get the number of each in order
        {
         //--- get deals properties
         profit    = HistoryDealGetDouble(ticket,DEAL_PROFIT);          // get data on financial results
         commis    = HistoryDealGetDouble(ticket,DEAL_COMMISSION);      // get data on commission
         swap      = HistoryDealGetDouble(ticket,DEAL_SWAP);            // get swap data
         type      = HistoryDealGetInteger(ticket,DEAL_TYPE);           // get data on operation type

         if(type == DEAL_TYPE_BUY || type == DEAL_TYPE_SELL)            // if the deal is form a trading operatoin
           {
            if(profit>0)                                                // if financial result of current order is greater than 0,
              {
               WeekorderProfit += profit;                               // add to profit
              }
            else
              {
               WeekorderLoss += MathAbs(profit);                        // if loss, add up
              }
           }
        }
     }

//--- now select data by --==MONTH==--
   HistorySelect(StructToTime(start_month),StructToTime(local));        // select the required history
//--- check
   total  = HistoryDealsTotal();                                        // number number of selected deals
   ticket = 0;                                                          // order number
   profit = 0;                                                          // order profit
   commis = 0;                                                          // order commission
   swap   = 0;                                                          // order swap

//--- for all deals
   for(uint i=0; i<total; i++)                                          // loop through all selected orders
     {
      //--- try to get deals ticket
      if((ticket=HistoryDealGetTicket(i))>0)                            // get the number of each in order
        {
         //--- get deals properties
         profit    = HistoryDealGetDouble(ticket,DEAL_PROFIT);          // get data on financial results
         commis    = HistoryDealGetDouble(ticket,DEAL_COMMISSION);      // get data on commission
         swap      = HistoryDealGetDouble(ticket,DEAL_SWAP);            // get swap data
         type      = HistoryDealGetInteger(ticket,DEAL_TYPE);           // get data on operation type

         MonthOrderSwap    += swap;                                     // sum up swaps
         MonthOrderCommis  += commis;                                   // sum up commissions

         if(type == DEAL_TYPE_BUY || type == DEAL_TYPE_SELL)            // if the deal is form a trading operatoin
           {
            if(profit>0)                                                // if financial result of current order is greater than 0,
              {
               MonthorderProfit += profit;                              // add to profit
              }
            else
              {
               MonthorderLoss += MathAbs(profit);                       // if loss, sum up
              }
           }
        }
     }
```

The above method can be called every time we need to update the current limit usage values. We can update the values of actual limits, as well as call this function, when generating various terminal events. Since the point of this method is to update limits, this can be done when events related to changes in current orders occur, such as [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) and [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction), and whenever a new tick emerges with the [NewTick](https://www.mql5.com/en/docs/runtime/event_fire#newtick) event. Since our method is quite resource-efficient, we will update actual limits at every tick. Now let's implement the event handler necessary to handle events related to dynamic cancellation and trade resolution.

### Class event handler

To handle events, we define an internal method of our ContoEvents() class with the [protected](https://www.mql5.com/en/docs/basis/types/classes) access level. To do this, we declare additional auxiliary fields with the same access level. To be able to instantly track the start time of a new trading period, which we need for changing the trading permission flags, we need to store the values of the last recorded period and the current period. For these purposes, we can use simple arrays declared with the [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) data type to store the values of the corresponding periods.

```
   //--- additional auxiliary arrays
   datetime          Periods_old[3];         // 0-day,1-week,2-mn
   datetime          Periods_new[3];         // 0-day,1-week,2-mn
```

In the first dimension, we will store the values of the day, in the second the week, and in the third the month. If it is necessary to further expand the controlled periods, you can declare these arrays not statically, but dynamically. But here we only work with three time periods. Now let's add to our class constructor the primary initialization of these array variables as follows.

```
   Periods_new[0] = iTime(_Symbol, PERIOD_D1, 1);                       // initialize the current day with the previous period
   Periods_new[1] = iTime(_Symbol, PERIOD_W1, 1);                       // initialize the current week with the previous period
   Periods_new[2] = iTime(_Symbol, PERIOD_MN1, 1);                      // initialize the current month with the previous period
```

We will initialize each corresponding period using a predefined terminal function [iTime()](https://www.mql5.com/en/docs/series/itime) passing in the parameters the corresponding period of [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) from the period preceding the current one. We deliberately do not initialize the Periods\_old\[\] array. In this case, after calling the constructor and our ContoEvents() method, we ensure that the event of the new trading period beginning is triggered and all the flags for starting trading are opened, and only then closed by code if there are no limits left. Otherwise, the class may not work correctly when reinitialized. The described method will contain simple logic: if the current period is not equal to the previous one, it means a new corresponding period has started and you can reset the limits and allow trading by changing the values in the flags. Also, for each period, we will call the previously described RefreshLimits() method to recalculate the input limits.

```
//+------------------------------------------------------------------+
//|                     ContoEvents                                  |
//+------------------------------------------------------------------+
void RiskManagerBase::ContoEvents()
  {
// check the start of a new trading day
   NewTradeDay    = false;                                              // variable for new trading day set to false
   Periods_old[0] = Periods_new[0];                                     // copy to old, new
   Periods_new[0] = iTime(_Symbol, PERIOD_D1, 0);                       // update new for day
   if(Periods_new[0]!=Periods_old[0])                                   // if do not match, it's a new day
     {
      Print(__FUNCTION__+" line"+IntegerToString(__LINE__)+", New trade day!");  // inform
      NewTradeDay = true;                                               // variable to true

      DayProfitArrive     = false;                                      // reset flag of reaching target profit after a new day started
      RiskDayPermission = true;                                         // allow opening new positions

      RefreshLimits();                                                  // update limits

      DayorderLoss = 0;                                                 // reset daily financial result
      DayorderProfit = 0;                                               // reset daily financial result
     }

// check the start of a new trading week
   Periods_old[1]    = Periods_new[1];                                  // copy data to old period
   Periods_new[1]    = iTime(_Symbol, PERIOD_W1, 0);                    // fill new period for week
   if(Periods_new[1]!= Periods_old[1])                                  // if periods do not match, it's a new week
     {
      Print(__FUNCTION__+" line"+IntegerToString(__LINE__)+", New trade week!"); // inform

      RiskWeekPermission = true;                                        // allow opening new positions

      RefreshLimits();                                                  // update limits

      WeekorderLoss = 0;                                                // reset weekly losses
      WeekorderProfit = 0;                                              // reset weekly profits
     }

// check the start of a new trading month
   Periods_old[2]    = Periods_new[2];                                  // copy the period to the old one
   Periods_new[2]    = iTime(_Symbol, PERIOD_MN1, 0);                   // update new period for month
   if(Periods_new[2]!= Periods_old[2])                                  // if do not match, it's a new month
     {
      Print(__FUNCTION__+" line"+IntegerToString(__LINE__)+", New trade Month!");   // inform

      RiskMonthPermission = true;                                       // allow opening new positions

      RefreshLimits();                                                  // update limits

      MonthorderLoss = 0;                                               // reset the month's loss
      MonthorderProfit = 0;                                             // reset the month's profit
     }

// set the permission to open new positions true only if everything is true
// set to true
   if(RiskDayPermission    == true &&                                   // if there is a daily limit available
      RiskWeekPermission   == true &&                                   // if there is a weekly limit available
      RiskMonthPermission  == true                                      // if there is a monthly limit available
     )                                                                  //
     {
      RiskTradePermission=true;                                         // if all are allowed, trading is allowed
     }

// set to false if at least one of them is false
   if(RiskDayPermission    == false ||                                  // no daily limit available
      RiskWeekPermission   == false ||                                  // or no weekly limit available
      RiskMonthPermission  == false ||                                  // or no monthly limit available
      DayProfitArrive      == true                                      // or target profit is reached
     )                                                                  // then
     {
      RiskTradePermission=false;                                        // prohibit trading
     }
   }
```

Also in this method, we have added control over the state of the data in the main variable of the flag for the possibility of opening new positions, RiskTradePermission. Through logical selection operators, we implement enabling permission through this variable only if all permissions are true, and disabling it if at least one of the flags does not allow trading. This variable will be very useful if you integrate this class into an already created algorithmic EA; you can simply receive it via a getter and insert it into the code with conditions for placing your orders. In our case, it will simply serve as a flag to start informing the user about the absence of free trading limits. Now that our class has "learned" how to control risks when the specified losses are achieved, let's move on to implementing the functionality to control the achievement of the target profit.

### Mechanism for controlling daily target profit

In the previous part of our [articles](https://www.mql5.com/en/articles/14340#n3), we have declared a flag for launching the control over target profit and an input variable for determining its value relative to the size of the account deposit. According to the logic of our class that controls the achievement of target profit, all open positions will be closed if the total profit for all positions has reached the target value. To close all positions on an account, we will declare in our class the internal method AllOrdersClose() with the [public](https://www.mql5.com/en/docs/basis/types/classes) access level. For this method to work, we will need to receive data on open positions and automatically send orders to close them.

In order not to waste time writing our own implementations of this functionality, we will use ready-made internal classes of the terminal. We will use the internal standard terminal class [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) to work with open positions and the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class to close open positions. Let's declare the variables of these two classes also with the [protected](https://www.mql5.com/en/docs/basis/types/classes) access level without using pointer with default constructor as follows.

```
   CTrade            r_trade;                // instance
   CPositionInfo     r_position;             // instance
```

When working with these objects, within the framework of the functionality we need now, we will not need to configure them additionally, so we will not write them in the constructor of our class. Here is the implementation of this method using declared classes:

```
//+------------------------------------------------------------------+
//|                       AllOrdersClose                             |
//+------------------------------------------------------------------+
bool RiskManagerBase::AllOrdersClose()                                  // closing market positions
  {
   ulong ticket = 0;                                                    // order ticket
   string symb;

   for(int i = PositionsTotal(); i>=0; i--)                             // loop through open positoins
     {
      if(r_position.SelectByIndex(i))                                   // if a position selected
        {
         ticket = r_position.Ticket();                                  // remember position ticket

         if(!r_trade.PositionClose(ticket))                             // close by ticket
           {
            Print(__FUNCTION__+". Error close order. "+IntegerToString(ticket)); // if not, inform
            return(false);                                              // return false
           }
         else
           {
            Print(__FUNCTION__+". Order close success. "+IntegerToString(ticket)); // if not, inform
            continue;                                                   // if everything is ok, continue
           }
        }
     }
   return(true);                                                        // return true
  }
```

We will call the described method both when the target profit is achieved and when limits are reached. It also returns a [bool](https://www.mql5.com/en/docs/basis/types/integer) value in case it is necessary to handle errors in sending closing orders. To provide functionality for controlling whether target profit is achieved, we will supplement our event handling method ContoEvents() with the following code immediately after the code already described above.

```
//--- daily
   if(dayProfitControl)							// check if functionality is enabled by the user
     {
      if(CurrentEquity >= (StartEquity+PlanDayProfit))                  // if equity exceeds or equals start + target profit,
        {
         DayProfitArrive = true;                                        // set flag that target profit is reached
         Print(__FUNCTION__+", PlanDayProfit has been arrived.");       // inform about the event
         Print(__FUNCTION__+", CurrentEquity = "+DoubleToString(CurrentEquity)+
               ", StartEquity = "+DoubleToString(StartEquity)+
               ", PlanDayProfit = "+DoubleToString(PlanDayProfit));
         AllOrdersClose();                                              // close all open orders

         StartEquity = CurrentEquity;                                   // rewrite starting equity value

         //--- send a push notification
         ResetLastError();                                              // reset the last error
         if(!SendNotification("The planned profitability for the day has been achieved. Equity: "+DoubleToString(CurrentEquity)))// notification
           {
            Print(__FUNCTION__+IntegerToString(__LINE__)+", Error of sending notification: "+IntegerToString(GetLastError()));// if not, print
           }
        }
     }
```

The method includes sending of a push notification to the user to notify that this event has occurred. For this, we use the predefined terminal function [SendNotification](https://www.mql5.com/en/docs/network/sendnotification). To complete the minimum required functionality of our class, we just need to assemble one more class method with [public](https://www.mql5.com/en/docs/basis/types/classes) access, which will be called when a risk manager is connected to the structure of our EA.

### Defining a method for launching monitoring in the EA structure

To add the monitoring functionality from an instance of our risk manager class to the EA structure, we will declare the public method ContoMonitor(). In this method, we will collect all previously declared event handling methods and will also supplement it with functionality for comparing the actually used limits with the values approved by the user in the input parameters. Let's declare this method with the [public](https://www.mql5.com/en/docs/basis/types/classes) access level and describe it outside the class as follows.

```
//+------------------------------------------------------------------+
//|                       ContoMonitor                               |
//+------------------------------------------------------------------+
void RiskManagerBase::ContoMonitor()                                    // monitoring
  {
   ForOnTrade();                                                        // update at each tick

   ContoEvents();                                                       // event block

//---
   double currentProfit = AccountInfoDouble(ACCOUNT_PROFIT);

   if((MathAbs(DayorderLoss)+MathAbs(currentProfit) >= RiskPerDay &&    // if equity is less than or equal to the start balance minus the daily risk
       currentProfit<0                                            &&    // profit below zero
       RiskDayPermission==true)                                         // day trading is allowed
      ||                                                                // OR
      (RiskDayPermission==true &&                                       // day trading is allowed
       MathAbs(DayorderLoss) >= RiskPerDay)                             // loss exceed daily risk
   )

     {
      Print(__FUNCTION__+", EquityControl, "+"ACCOUNT_PROFIT = "  +DoubleToString(currentProfit));// notify
      Print(__FUNCTION__+", EquityControl, "+"RiskPerDay = "      +DoubleToString(RiskPerDay));   // notify
      Print(__FUNCTION__+", EquityControl, "+"DayorderLoss = "    +DoubleToString(DayorderLoss)); // notify
      RiskDayPermission=false;                                          // prohibit opening new orders during the day
      AllOrdersClose();                                                 // close all open positions
     }

// check if there is a WEEK limit available for opening a new position if there are no open ones
   if(
      MathAbs(WeekorderLoss)>=RiskPerWeek &&                            // if weekly loss is greater than or equal to the weekly risk
      RiskWeekPermission==true)                                         // and we traded
     {
      RiskWeekPermission=false;                                         // prohibit opening of new orders during the day
      AllOrdersClose();                                                 // close all open positions

      Print(__FUNCTION__+", EquityControl, "+"WeekorderLoss = "+DoubleToString(WeekorderLoss));  // notify
      Print(__FUNCTION__+", EquityControl, "+"RiskPerWeek = "+DoubleToString(RiskPerWeek));      // notify
     }

// check if there is a MONTH limit available for opening a new position if there are no open ones
   if(
      MathAbs(MonthorderLoss)>=RiskPerMonth &&                          // if monthly loss is greater than or equal to the monthly risk
      RiskMonthPermission==true)                                        // we traded
     {
      RiskMonthPermission=false;                                        // prohibit opening of new orders during the day
      AllOrdersClose();                                                 // close all open positions

      Print(__FUNCTION__+", EquityControl, "+"MonthorderLoss = "+DoubleToString(MonthorderLoss));  // notify
      Print(__FUNCTION__+", EquityControl, "+"RiskPerMonth = "+DoubleToString(RiskPerMonth));      // notify
     }
  }
```

The operating logic of our method is very simple: if the actual loss limit for a month or week exceeds the one set by the user, the trading flag for a given period is set to prohibited and, accordingly, trading is prohibited. The only difference is in the daily limits, where we also need to control the presence of open positions; for this, we will also add control of the current profit from open positions through the logical operator OR. When the risk limits are reached, we call our method for closing positions and print the log about this event.

At this stage, to fully complete the class, we only need to add a method for the user to control the current limits. The simplest and most convenient way would be to display the necessary information through the standard predefined terminal function, [Comment()](https://www.mql5.com/en/docs/common/comment). To work with this function, we will need to pass to it a [string](https://www.mql5.com/en/docs/basis/types/stringconst) type parameter containing information to display on the chart. To get these values from our class, we declare the Message() method with the [public](https://www.mql5.com/en/docs/basis/types/classes) access level, which will return [string](https://www.mql5.com/en/docs/basis/types/stringconst) data with collected data on all the variables the user needs.

```
//+------------------------------------------------------------------+
//|                        Message                                   |
//+------------------------------------------------------------------+
string RiskManagerBase::Message(void)
  {
   string msg;                                                          // message

   msg += "\n"+" ----------Risk-Manager---------- ";                    // common
//---
   msg += "\n"+"RiskTradePer = "+(string)RiskTradePermission;           // final trade permission
   msg += "\n"+"RiskDayPer   = "+(string)RiskDayPermission;             // daily risk available
   msg += "\n"+"RiskWeekPer  = "+(string)RiskWeekPermission;            // weekly risk available
   msg += "\n"+"RiskMonthPer = "+(string)RiskMonthPermission;           // monthly risk available

//---limits and inputs
   msg += "\n"+" -------------------------------- ";                    //
   msg += "\n"+"RiskPerDay   = "+DoubleToString(RiskPerDay,2);          // daily risk in usd
   msg += "\n"+"RiskPerWeek  = "+DoubleToString(RiskPerWeek,2);         // weekly risk in usd
   msg += "\n"+"RiskPerMonth = "+DoubleToString(RiskPerMonth,2);        // monthly risk usd
//--- current profits and losses for periods
   msg += "\n"+" -------------------------------- ";                    //
   msg += "\n"+"DayLoss     = "+DoubleToString(DayorderLoss,2);         // daily loss
   msg += "\n"+"DayProfit   = "+DoubleToString(DayorderProfit,2);       // daily profit
   msg += "\n"+"WeekLoss    = "+DoubleToString(WeekorderLoss,2);        // weekly loss
   msg += "\n"+"WeekProfit  = "+DoubleToString(WeekorderProfit,2);      // weekly profit
   msg += "\n"+"MonthLoss   = "+DoubleToString(MonthorderLoss,2);       // monthly loss
   msg += "\n"+"MonthProfit = "+DoubleToString(MonthorderProfit,2);     // monthly profit
   msg += "\n"+"MonthCommis = "+DoubleToString(MonthOrderCommis,2);     // monthly commissions
   msg += "\n"+"MonthSwap   = "+DoubleToString(MonthOrderSwap,2);       // monthly swaps
//--- for current monitoring

   if(dayProfitControl)                                                 // if control daily profit
     {
      msg += "\n"+" ---------dayProfitControl-------- ";                //
      msg += "\n"+"DayProfitArrive = "+(string)DayProfitArrive;         // daily profit achieved
      msg += "\n"+"StartBallance   = "+DoubleToString(StartBalance,2);  // starting balance
      msg += "\n"+"PlanDayProfit   = "+DoubleToString(PlanDayProfit,2); // target profit
      msg += "\n"+"PlanDayEquity   = "+DoubleToString(PlanDayEquity,2); // target equity
     }
   return(msg);                                                         // return value
  }
```

The message for the user created by the method will look like this.

![Figure 2. Data output format.](https://c.mql5.com/2/72/CHFJPYzWeekly__1.png)

Figure 2. Data output format.

This method can be modified or supplemented by adding elements for working with graphics in the terminal. But we will use it like this since it provides to the user sufficient data from our class to make a decision. If desired, you can refine this format in the future and make it more beautiful in terms of graphics. Let's now discuss the possibilities of expanding this class when using individual trading strategies.

### The final implementation and possibilities for extending the class

As we mentioned earlier, the functionality we described here is the minimum necessary and the most universal for almost all trading strategies. It allows controlling risks and preventing loss of deposit in one day. In this part of the article, we will look at several more possibilities for expanding this class.

- Control spread size when trading with a short stop loss
- Control slippage for open positions
- Control target monthly profit

For the first point, we can implement additional functionality for trading systems that use trading with short stop loss. You can declare the SpreadMonitor(int intSL) method that takes as a parameter the technical or calculated stop loss for an instrument in points to compare it with the current spread level. This method will prohibit placing an order if the spread widens greatly relative to the stop loss in a proportion determined by the user, to avoid the high risk of closing the position at stop loss due to the spread.

To control slippage at the time of opening, in accordance with the second point, you can declare the SlippageCheck() method. This method will close each individual transaction if the broker opened it at a price very different from the stated one, due to which the deal risk exceeded the expected value. This will allow, in case stop loss is triggered, not to spoil the statistics by high risk trading per one separate entry. Also, when trading with a fixed stop loss to take profit ratio, this ratio worsens due to slippage and it is better to close the position with a small loss than to incur larger losses later.

Similar to the logic of controlling daily profit, it is possible to implement a corresponding method to control target monthly profit. This method can be used when trading longer-term strategies. The class we described already has all the necessary functionality for use in manual intraday trading, and it can be integrated into the final implementation of a trading EA, which should be launched on the instrument chart simultaneously with the start of manual trading.

The final assembly of the project includes connecting our class using the [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) preprocessor directive.

```
#include <RiskManagerBase.mqh>
```

Next, we declare the pointer of our risk manager object at the global level.

```
RiskManagerBase *RMB;
```

When initializing our EA, we manually allocate memory for our object to prepare it before launch.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {

   RMB = new RiskManagerBase();

//---
   return(INIT_SUCCEEDED);
  }
```

When we remove our EA from the chart, we need to clear the memory from our object to avoid a memory leak. For this, write the following in the EA's [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

   delete RMB;

  }
```

Also, if necessary, in the same event you can call the [Comment(" ")](https://www.mql5.com/en/docs/common/comment) method, passing an empty string into it, so that the chart is cleared of comments when the EA is removed from the symbol chart.

We call the main monitoring method of our class upon the event of receiving a new tick for the symbol.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   RMB.ContoMonitor();

   Comment(RMB.Message());
  }
//+------------------------------------------------------------------+
```

This completes the assembly of our EA with the built-in risk manager and it is completely ready for use (file ManualRiskManager.mq5). In order to test several cases of its use, we will make a small addition to the current code to simulate the process of manual trading.

### Usage example

To visualize the process of manual trading with and without using a risk manager, we will need additional code that models manual trading. Since in this article we will not touch upon the topic of choosing trading strategies, we will not implement the full trading functionality in code. Instead, we will visually take entries from the daily chart and add ready-made data into our EA. We will use a very simple strategy for making trading decisions and will see the final financial result for this strategy with the only difference: with and without risk control.

As examples of entries, we will use a simple strategy with breakouts of a fractal level, for the USDJPY instrument, over a period of two months. Let's see how this strategy performs with and without risk control. Schematically, the strategy signals for manual entries will be as follows.

![Figure 3. Entries using a test strategy](https://c.mql5.com/2/72/model.png)

Figure 3. Entries using a test strategy

To model this strategy, let's write a small addition as a universal unit test for any manual strategy, so that each user can test their entries with minor modifications. During this test, the strategy will execute pre-loaded ready-made signals, without implementing its own logic for enetering the market. For this, we first need to declare an additional structure, [struct](https://www.mql5.com/en/docs/basis/types/classes), which will store our fractal-based entries.

```
//+------------------------------------------------------------------+
//|                         TradeInputs                              |
//+------------------------------------------------------------------+
struct TradeInputs
  {
   string             symbol;                                           // symbol
   ENUM_POSITION_TYPE direction;                                        // direction
   double             price;                                            // price
   datetime           tradedate;                                        // date
   bool               done;                                             // trigger flag
  };
```

The main class that will be responsible for modeling trading signals is TradeModel. The class constructor will accept a container with signal input parameters, and its main Processing() method will monitor every tick whether the time of the entry point has arrived based on the input values. Since we are simulating intraday trading, at the end of the day we will be eliminating all positions using the previously declared AllOrdersClose() method in our risk manager class. Here is our auxiliary class.

```
//+------------------------------------------------------------------+
//|                        TradeModel                                |
//+------------------------------------------------------------------+
class TradeModel
  {
protected:

   CTrade               *cTrade;                                        // to trade
   TradeInputs       container[];                                       // container of entries

   int               size;                                              // container size

public:
                     TradeModel(const TradeInputs &inputs[]);
                    ~TradeModel(void);

   void              Processing();                                      // main modeling method
  };
```

To enable convenient order placing, we will use the standard terminal class [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade), which contains all the functionality we need. This will save time on developing our auxiliary class. To pass input parameters when creating a class instance, we define our constructor with one input parameter of the entries container.

```
//+------------------------------------------------------------------+
//|                          TradeModel                              |
//+------------------------------------------------------------------+
TradeModel::TradeModel(const TradeInputs &inputs[])
  {
   size = ArraySize(inputs);                                            // get container size
   ArrayResize(container, size);                                        // resize

   for(int i=0; i<size; i++)                                            // loop through inputs
     {
      container[i] = inputs[i];                                         // copy to internal
     }

//--- trade class
   cTrade=new CTrade();                                                 // create trade instance
   if(CheckPointer(cTrade)==POINTER_INVALID)                            // if instance not created,
     {
      Print(__FUNCTION__+IntegerToString(__LINE__)+" Error creating object!");   // notify
     }
   cTrade.SetTypeFillingBySymbol(Symbol());                             // fill type for the symbol
   cTrade.SetDeviationInPoints(1000);                                   // deviation
   cTrade.SetExpertMagicNumber(123);                                    // magic number
   cTrade.SetAsyncMode(false);                                          // asynchronous method
  }
```

In the constructor, we initialize the container of input parameters with the desired value, remember its size and create an object of our [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class with the necessary settings. Most of the parameters here is not configured by the user, since they will not affect the purpose of creating our unit test, so we leave them hardcoded.

The destructor of our TradeModel class will only require the removal of a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) object.

```
//+------------------------------------------------------------------+
//|                         ~TradeModel                              |
//+------------------------------------------------------------------+
TradeModel::~TradeModel(void)
  {
   if(CheckPointer(cTrade)!=POINTER_INVALID)                            // if there is an instance,
     {
      delete cTrade;                                                    // delete
     }
  }
```

Now we will implement our main processing method for the operation of our class in the structure of our entire project. Let's implement the logic for placing orders according to Figure 3:

```
//+------------------------------------------------------------------+
//|                         Processing                               |
//+------------------------------------------------------------------+
void TradeModel::Processing(void)
  {
   datetime timeCurr = TimeCurrent();                                   // request current time

   double bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);                  // take bid
   double ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);                  // take ask

   for(int i=0; i<size; i++)                                            // loop through inputs
     {
      if(container[i].done==false &&                                    // if we haven't traded yet AND
         container[i].tradedate <= timeCurr)                            // date is correct
        {
         switch(container[i].direction)                                 // check trade direction
           {
            //---
            case  POSITION_TYPE_BUY:                                    // if Buy,
               if(container[i].price >= ask)                            // check if price has reached and
                 {
                  if(cTrade.Buy(0.1))                                   // by the same lot
                    {
                     container[i].done = true;                          // if time has passed, put a flag
                     Print("Buy has been done");                        // notify
                    }
                  else                                                  // if hasn't passed,
                    {
                     Print("Error: buy");                               // notify
                    }
                 }
               break;                                                   // complete the case
            //---
            case  POSITION_TYPE_SELL:                                   // if Sell
               if(container[i].price <= bid)                            // check if price has reached and
                 {
                  if(cTrade.Sell(0.1))                                  // sell the same lot
                    {
                     container[i].done = true;                          // if time has passed, put a flag
                     Print("Sell has been done");                       // notify
                    }
                  else                                                  // if hasn't passed,
                    {
                     Print("Error: sell");                              // notify
                    }
                 }
               break;                                                   // complete the case

            //---
            default:
               Print("Wrong inputs");                                   // notify
               return;
               break;
           }
        }
     }
  }
```

The logic of this method is quite simple. If there are unprocessed entries in the container for which the modeling time has come, we place these orders in accordance with the direction and price of the fractal marked in Figure 3. This functionality is enough for testing the risk manager, so we can integrate it into our main project.

First, let's connect our test class to the EA code as follows.

```
#include <TradeModel.mqh>
```

Now, in the OnInit() function, we create an instance of our TradeInputs data array structure and pass this array to the constructor of the TradeModel class to initialize it.

```
//---
   TradeInputs modelInputs[] =
     {
        {"USDJPYz", POSITION_TYPE_SELL, 146.636, D'2024-01-31',false},
        {"USDJPYz", POSITION_TYPE_BUY,  148.794, D'2024-02-05',false},
        {"USDJPYz", POSITION_TYPE_BUY,  148.882, D'2024-02-08',false},
        {"USDJPYz", POSITION_TYPE_SELL, 149.672, D'2024-02-08',false}
     };

//---
   tModel = new TradeModel(modelInputs);
```

Do not forget to clear the memory of our tModel object in the DeInit() function. The main functionality will be performed in the OnTick() function, supplemented with the following code.

```
   tModel.Processing();                                                 // place orders

   MqlDateTime time_curr;                                               // current time structure
   TimeCurrent(time_curr);                                              // request current time

   if(time_curr.hour >= 23)                                             // if end of day
     {
      RMB.AllOrdersClose();                                             // close all positions
     }
```

Now let's compare the results of the same strategy with and without the risk control class. Let's run the unit test file ManualRiskManager(UniTest1) without the risk control method. For the period January to March 2024, we get the following result of our strategy.

![Figure 4. Test data without using a risk manager](https://c.mql5.com/2/73/TesterGraphReport2024.03.18.png)

Figure 4. Test data without using a risk manager

As a result, we obtain a positive mathematical expectation for this strategy with the following parameters.

| # | Parameter name | Parameter value |
| --- | --- | --- |
| 1 | EA | ManualRiskManager(UniTest1) |
| 2 | Symbol | USDJPY |
| 3 | Chart Timeframes | М15 |
| 4 | Time range | 2024.01.01 - 2024.03.18 |
| 5 | Forward testing | NO |
| 6 | Delays | No delays, perfect performance |
| 7 | Simulation | Every Tick |
| 8 | Initial deposit | USD 10,000 |
| 9 | Leverage | 1:100 |

Table 1. Input parameters for the strategy tester

Now let's run the unit test file ManualRiskManager(UniTest2), where we use our risk manager class with the following input parameters.

| Input parameter name | Variable value |
| --- | --- |
| inp\_riskperday | 0.25 |
| inp\_riskperweek | 0.75 |
| inp\_riskpermonth | 2.25 |
| inp\_plandayprofit | 0.78 |
| dayProfitControl | true |

Table 2. Input parameters for the risk manager

The logic for generating input parameters is similar to the logic described above when designing the structure of input parameters in [Part 3](https://www.mql5.com/en/articles/14340#n3). The profit curve will look like this.

![Figure 5. Test data using a risk manager](https://c.mql5.com/2/73/TesterGraphReport2024.03.18__1.png)

Figure 5. Test data using a risk manager

A summary of the testing results of the two cases is presented in the following table.

| # | Value | No Risk Manager | Risk Manager | Change |
| --- | --- | --- | --- | --- |
| 1 | Total Net Profit: | 41.1 | 144.48 | +103.38 |
| 2 | Balance Drawdown Maximal: | 0.74% | 0.25% | Reduced by 3 times |
| 3 | Equity Drawdown Maximal: | 1.13% | 0.58% | Reduced by 2 times |
| 4 | Expected Payoff: | 10.28 | 36.12 | More than 3 times growth |
| 5 | Sharpe Ratio: | 0.12 | 0.67 | 5 times growth |
| 6 | Profit Trades (% of total): | 75% | 75% | - |
| 7 | Average Profit Trade: | 38.52 | 56.65 | Growth by 50% |
| 8 | Average loss trade: | -74.47 | -25.46 | Reduced by 3 times |
| 9 | Average risk return | 0.52 | 2.23 | 4 times growth |

Table 3. Comparison of the financial results of trading with and without the risk manager

Based on the results of our unit tests, we can conclude that the use of risk control through our risk manager class has significantly increased the efficiency of trading using the same simple strategy, by limiting risks and fixing profits for each transaction relative to the fixed risk. This made it possible to reduce the balance drawdown by 3 times and equity balance by 2 times. The Expected Payoff for the strategy increased by more than 3 times, and the Sharpe ratio increased by more than 5 times. The average profitable trade increased by 50%, and the average unprofitable trade decreased by three times, which made it possible to bring the average risk return on the account to almost the target value of 1 to 3. The table below provide a detailed comparison of financial results for each individual trade from our pool.

| Date | Symbol | Direction | Lot | No Risk Manager | Risk Manager | Change |
| --- | --- | --- | --- | --- | --- | --- |
| 2024.01.31 | USDJPY | buy | 0.1 | 25.75 | 78 | \+ 52.25 |
| 2024.02.05 | USDJPY | sell | 0.1 | 13.19 | 13.19 | - |
| 2024.02.08 | USDJPY | sell | 0.1 | 76.63 | 78.75 | \+ 2.12 |
| 2024.02.08 | USDJPY | buy | 0.1 | -74.47 | -25.46 | \+ 49.01 |
| Total | - | - | - | 41.10 | 144.48 | \+ 103.38 |

Table 4. Comparison of executed trades with and without the risk manager

### Conclusion

Based on the theses presented in the article, the following conclusions can be drawn. Using the risk manager even in manual trading can significantly increase the effectiveness of strategies, including profitable ones. In the case of a losing strategy, the use of the risk manager can assist in securing deposits, limiting losses. As previously mentioned in the introduction, we try to mitigate the psychological factor. You should not turn off the risk manager trying to immediately recover losses. It can be better to wait out the period when the limits are completed and , without emotions, start trading again. Use the time when trading is prohibited by the risk manager to analyzing your trading strategy to understand what caused losses and how to avoid them in the future.

Thanks to everyone who read this article to the end. I really hope that this article will save at least one deposit from being completely lost. In this case I will consider that my efforts were not wasted. I will be happy to see your comments or private messages, especially whether I should start a new article where we can adapt this class to a purely algorithmic EA. Your feedback is welcome. Thank you!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14340](https://www.mql5.com/ru/articles/14340)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14340.zip "Download all attachments in the single ZIP archive")

[ManualRiskManagereUniTest1m.mq5](https://www.mql5.com/en/articles/download/14340/manualriskmanagereunitest1m.mq5 "Download ManualRiskManagereUniTest1m.mq5")(4.87 KB)

[ManualRiskManager9UniTest2p.mq5](https://www.mql5.com/en/articles/download/14340/manualriskmanager9unitest2p.mq5 "Download ManualRiskManager9UniTest2p.mq5")(4.85 KB)

[ManualRiskManager.mq5](https://www.mql5.com/en/articles/download/14340/manualriskmanager.mq5 "Download ManualRiskManager.mq5")(3.04 KB)

[RiskManagerBase.mqh](https://www.mql5.com/en/articles/download/14340/riskmanagerbase.mqh "Download RiskManagerBase.mqh")(61.38 KB)

[TradeModel.mqh](https://www.mql5.com/en/articles/download/14340/trademodel.mqh "Download TradeModel.mqh")(12.99 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Visualizing deals on a chart (Part 2): Data graphical display](https://www.mql5.com/en/articles/14961)
- [Visualizing deals on a chart (Part 1): Selecting a period for analysis](https://www.mql5.com/en/articles/14903)
- [Risk manager for algorithmic trading](https://www.mql5.com/en/articles/14634)
- [Balancing risk when trading multiple instruments simultaneously](https://www.mql5.com/en/articles/14163)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/470847)**
(8)


![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
23 Mar 2024 at 06:06

**ZlobotTrader [#](https://www.mql5.com/ru/forum/464456#comment_52815240):**

Useful article. Thank you!

Thank you very much) Very much appreciated. Do you think to write the next one about algorithmic successor of risk manager?

![Aleksei Iakunin](https://c.mql5.com/avatar/avatar_na2.png)

**[Aleksei Iakunin](https://www.mql5.com/en/users/winyak)**
\|
26 Mar 2024 at 20:40

Of course you write - it is very useful for beginners, but since your articles are oriented at beginners (subjective opinion),

then pay attention a little more to "chewing up" the code.

Good luck)

![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
27 Mar 2024 at 06:32

**Алексей [#](https://www.mql5.com/ru/forum/464456#comment_52849271):**

Of course write-beginners are very useful, but since your articles are oriented to beginners( subjective opinion),

pay more attention to "chewing up" the code.

Good luck)

Accepted, thanks)

![HasiTrader](https://c.mql5.com/avatar/2024/12/67516503-24ad.jpg)

**[HasiTrader](https://www.mql5.com/en/users/hasitrader)**
\|
12 Aug 2024 at 21:40

Hello [@Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)

You have done a very good job.

The important thing is that the EA should can prevent the opening of a new deal after exceeding the defined limitations.

As you mentioned, there is no way to prevent users' manual trading. But your EA can immediately close any new trade that is opened after exceeding the limits (this will only cause a loss to the trader equal to the spread).  This leads to the trader avoiding to open further trades. Maybe this method is better than informing the user using comment.

Good luck

![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
13 Aug 2024 at 17:01

**HasiTrader [#](https://www.mql5.com/en/forum/470847#comment_54281302):**

Hello [@Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)

You have done a very good job.

The important thing is that the EA should can prevent the opening of a new deal after exceeding the defined limitations.

As you mentioned, there is no way to prevent users' manual trading. But your EA can immediately close any new trade that is opened after exceeding the limits (this will only cause a loss to the trader equal to the spread).  This leads to the trader avoiding to open further trades. Maybe this method is better than informing the user using comment.

Good luck

Servus! Thanks for the feedback! I agree with you. In many situations, discipline begins to play a much greater role in trading than applied knowledge, for example, in technical analysis.

![Build Self Optimizing Expert Advisors With MQL5 And Python (Part II): Tuning Deep Neural Networks](https://c.mql5.com/2/87/Build_Self_Optimizing_Expert_Advisors_With_MQL5_And_Python_Part_II___LOGO__2.png)[Build Self Optimizing Expert Advisors With MQL5 And Python (Part II): Tuning Deep Neural Networks](https://www.mql5.com/en/articles/15413)

Machine learning models come with various adjustable parameters. In this series of articles, we will explore how to customize your AI models to fit your specific market using the SciPy library.

![Data Science and ML (Part 28): Predicting Multiple Futures for EURUSD, Using AI](https://c.mql5.com/2/87/Data_Science_and_ML_Part_28___LOGO.png)[Data Science and ML (Part 28): Predicting Multiple Futures for EURUSD, Using AI](https://www.mql5.com/en/articles/15465)

It is a common practice for many Artificial Intelligence models to predict a single future value. However, in this article, we will delve into the powerful technique of using machine learning models to predict multiple future values. This approach, known as multistep forecasting, allows us to predict not only tomorrow's closing price but also the day after tomorrow's and beyond. By mastering multistep forecasting, traders and data scientists can gain deeper insights and make more informed decisions, significantly enhancing their predictive capabilities and strategic planning.

![Integrating MQL5 with data processing packages (Part 1): Advanced Data analysis and Statistical Processing](https://c.mql5.com/2/87/Integrating_MQL5_with_data_processing_packages_Part_1___LOGO.png)[Integrating MQL5 with data processing packages (Part 1): Advanced Data analysis and Statistical Processing](https://www.mql5.com/en/articles/15155)

Integration enables seamless workflow where raw financial data from MQL5 can be imported into data processing packages like Jupyter Lab for advanced analysis including statistical testing.

![Role of random number generator quality in the efficiency of optimization algorithms](https://c.mql5.com/2/73/The_role_of_the_quality_of_the_random_number_generator___LOGO.png)[Role of random number generator quality in the efficiency of optimization algorithms](https://www.mql5.com/en/articles/14413)

In this article, we will look at the Mersenne Twister random number generator and compare it with the standard one in MQL5. We will also find out the influence of the random number generator quality on the results of optimization algorithms.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ppnhxspytrkymopafejquvlikwyqkjrk&ssn=1769253367965701365&ssn_dr=0&ssn_sr=0&fv_date=1769253367&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14340&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Risk%20manager%20for%20manual%20trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925336772421914&fz_uniq=5083451368372444150&sv=2552)

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