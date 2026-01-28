---
title: Developing a multi-currency Expert Advisor (Part 14): Adaptive volume change in risk manager
url: https://www.mql5.com/en/articles/15085
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 12
scraped_at: 2026-01-22T17:11:55.958023
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/15085&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048838733739958186)

MetaTrader 5 / Trading


### Introduction

In one of the [previous](https://www.mql5.com/en/articles/14764) articles of the series, I touched on the topic of risk control and developed a risk manager class that implements basic functionality. It allows setting a maximum daily loss level and a maximum overall loss level, upon reaching which trading stops and all open positions are closed. If a daily loss was reached, trading was resumed the next day, and if an overall loss was reached, it was not resumed at all.

As you might remember, the possible areas for the risk manager development were a smoother change in the size of positions (for example, a two-fold reduction when half the limit is exceeded), and a more "intelligent" restoration of volumes (for example, only when the loss exceeds a position reduction level). We can also add a maximum target profit parameter, upon reaching which trading also stops. This parameter is unlikely to be useful when trading on a personal account. However, it will be in great demand when trading on the accounts of prop trading companies. Once the planned level of profit is reached, trading can usually be continued on another account only.

I also mentioned the introduction of time-based trading restrictions as one of the possible directions for the development of a risk manager but I will not touch on this topic here leaving it for the future. For now, let's try to implement adaptive volume changes in the risk manager and see if there can be any benefit from this.

### Base case

Let's use the EA from the previous article. We will add the ability to handle the risk manager parameters to it. We will also make some other minor additions to it, which will be discussed below. For now, let's agree on the EA parameters that we will use to evaluate the results of the changes made.

First, let's fix the specific composition of single instances of trading strategies that will be used in the test EA. Set IDs of the best passes after the second optimization stage in the _passes\__ parameter for each of the three optimized symbols and each of the three timeframes (9 IDs in total). Each ID will hide a normalized group of 16 single instances of trading strategies. Thus, the final group will contain a total of 144 instances of trading strategies, divided into 9 groups of 16 strategies. The final group will not be normalized, since we did not select a normalizing factor for it.

Second, we will use the fixed trading balance of USD 10,000 and our standard expected maximum drawdown of 10% for the scaling factor of 1. We will try to change the latter in the range from 1 to 10. At the same time, the maximum allowable drawdown will also increase, but we will now additionally control it with our risk manager preventing it from exceeding 10%.

To do this, we turn on the risk manager and set the maximum overall loss value equal to 10% of the base balance of USD 10,000, that is, USD 1000. For the maximum daily loss, set the value to be twice as small, that is, USD 500. As the balance grows, these two parameters will remain unchanged.

Let's set all the input values as described above:

![](https://c.mql5.com/2/110/1621057408566__2.png)

Fig. 1. Test EA inputs for the test EA with the original risk manager

Let's run the optimization on the interval of 2021 and 2022 to see how the EA works with different values of the scaling multiplier for position sizes ( _scale\__). We get the following results:

![](https://c.mql5.com/2/110/4893684179335__2.png)

Fig. 2. Results of optimizing the scale\_ parameter in the test EA with the original risk manager

The results are sorted in ascending order of the _scale\__ parameter value, that is, the lower the line, the larger the size of opened positions used by the EA. It is clearly visible that starting from a certain critical value, the final result is a loss of slightly more than 1000.

During the test interval we encountered several drawdowns of varying depths. In the passes where no drawdown dropped the equity level below USD 9000, trading continued until the end of the interval. In the passes, in which the level of funds dropped below USD 9000 during the drawdown, trading stopped at that moment and did not resume. It was in these passages that we suffered a loss of around USD 1000. Its slight excess over the calculated value is most likely explained by the fact that we used the EA's operating mode only on new minute bars. Therefore, prices had time to change a little more within a minute from the moment the specified loss was precisely reached until the moment the risk manager checked and made a decision to close all positions.

These differences are minor in most runs, and we can ignore them by agreeing to set a slightly lower limit in the parameters, or by changing, as planned, the way the risk manager works. The only passage that is of concern is the one for _scale\__ = 5.5, when after closing all positions the loss exceeded the calculated one by more than 20% and amounted to approximately USD 1234.

For further analysis, let's look at the balance and equity curve graph for the very first pass ( _scale\__ = 1.0). Since the remaining passes differ only in the size of the positions opened, their balance and equity graphs will have the same appearance as for the first pass. The only difference is that they will be more stretched vertically.

![](https://c.mql5.com/2/110/3906368866592__2.png)

![](https://c.mql5.com/2/110/565417745248__2.png)

Fig. 3. Pass results with scale\_ = 1.0 in the test EA with the original risk manager

Let's compare them with the results without the risk manager:

![](https://c.mql5.com/2/110/2068214654080__2.png)

![](https://c.mql5.com/2/110/4758581172939__2.png)

Fig. 4. Pass results with scale\_ = 1.0 in the test EA without the risk manager

Without the risk manager, the overall profit was several percent higher, and the maximum drawdown on equity remained the same. This suggests that normalizing strategies when grouping them together, as well as their subsequent joint work produces good results: the risk manager had to close positions when the daily loss was exceeded only about three times in two years.

Let's now look at the results of the pass with the risk manager and _scale\__ = 3\. The profit was about 50% higher, but the drawdown also increased three times.

![](https://c.mql5.com/2/110/5206442769346__2.png)

![](https://c.mql5.com/2/110/6100368686493__2.png)

Fig. 5. Pass results with scale\_ = 3.0 in the test EA with the original risk manager

However, despite the drawdown increasing in absolute terms to almost USD 3000, the risk manager had to prevent the drawdown from exceeding USD 500 per day. In other words, there were several consecutive days when the risk manager closed all positions and reopened them at the beginning of the next day. From the previous maximum value of equity, the current amount eventually went into the red down to USD 3000, but the drawdown did not exceed USD 500 for each individual day relative to the maximum equity or funds at the beginning of the day. However, using such an increased position size is dangerous, as there is also a limit on the overall loss. In this case, we were lucky that the big drawdown happened some time after the start of the test period. The balance size had time to grow and thus increased the value of the maximum overall loss, which was USD 1000 at the start and was counted from the value of the initial balance. If the test period had started right before the drawdown reached USD 3000, the overall limit would have been exceeded and trading would have been stopped.

In order to take into account the impact of a possible bad start time, we will change the risk manager parameters so that the level of overall loss is calculated not from the initial balance, but from the last maximum of the balance or equity. But to do this, we will first need to make additions to the risk manager code, since the ability to set the desired parameter value is not yet implemented.

### CVirtualRiskManager upgrade

We have quite a few changes planned for the risk manager class. Many of them require changes to the same methods. This makes it somewhat difficult to describe, as it is difficult to separate edits that relate to different added features. Therefore, we will describe the already completed version of the code, starting from the simplest passage.

### Updating enumerations

Before the class description, we declared several enumerations to be used later. We will slightly expand the composition of these enumerations. For example, the _ENUM\_RM\_STATE_ enumeration, containing possible states of the risk manager, receives two new states:

- RM\_STATE\_RESTORE — state that occurs after the start of a new daily period, until the moment when the sizes of open positions are fully restored. Previously, this condition did not arise, since we immediately restored the sizes of positions after the start of a new day. Now we have the option to do this not immediately, but only after prices return to more favorable values for opening. Let's look at this in more detail later.

- RM\_STATE\_OVERALL\_PROFIT — state that occurs after the specified profit has been reached. After this event, trading stops.

We have turned the previous enumeration ENUM\_RM\_CALC\_LIMIT into three separate enumerations: for daily loss, overall loss and overall profit. The values of these enumerations will be used to determine two things:

- how to use the number passed in the parameters: as an absolute or as a relative value specified as a percentage of (from) the daily level or the highest values of the balance or equity;
- the value to count the threshold level from (to) — the daily level or the highest balance or funds values.

These options are indicated in the comments for the enumeration values.

```
// Possible risk manager states
enum ENUM_RM_STATE {
   RM_STATE_OK,            // Limits are not exceeded
   RM_STATE_DAILY_LOSS,    // Daily limit is exceeded
   RM_STATE_RESTORE,       // Recovery after daily limit
   RM_STATE_OVERALL_LOSS,  // Overall limit exceeded
   RM_STATE_OVERALL_PROFIT // Overall profit reached
};

// Possible methods for calculating limits
enum ENUM_RM_CALC_DAILY_LOSS {
   RM_CALC_DAILY_LOSS_MONEY_BB,    // [$] to Daily Level
   RM_CALC_DAILY_LOSS_PERCENT_BB,  // [%] from Base Balance to Daily Level
   RM_CALC_DAILY_LOSS_PERCENT_DL   // [%] from/to Daily Level
};

// Possible methods for calculating general limits
enum ENUM_RM_CALC_OVERALL_LOSS {
   RM_CALC_OVERALL_LOSS_MONEY_BB,           // [$] to Base Balance
   RM_CALC_OVERALL_LOSS_MONEY_HW_BAL,       // [$] to HW Balance
   RM_CALC_OVERALL_LOSS_MONEY_HW_EQ_BAL,    // [$] to HW Equity or Balance
   RM_CALC_OVERALL_LOSS_PERCENT_BB,         // [%] from/to Base Balance
   RM_CALC_OVERALL_LOSS_PERCENT_HW_BAL,     // [%] from/to HW Balance
   RM_CALC_OVERALL_LOSS_PERCENT_HW_EQ_BAL   // [%] from/to HW Equity or Balance
};

// Possible methods for calculating overall profit
enum ENUM_RM_CALC_OVERALL_PROFIT {
   RM_CALC_OVERALL_PROFIT_MONEY_BB,           // [$] to Base Balance
   RM_CALC_OVERALL_PROFIT_PERCENT_BB,         // [%] from/to Base Balance
};
```

### Class Description

We have added new properties and methods in the protected section in the _CVirtualRiskManager_ class description. In the public section, everything remains unchanged. Added strings are highlighted in green:

```
//+------------------------------------------------------------------+
//| Risk management class (risk manager)                             |
//+------------------------------------------------------------------+
class CVirtualRiskManager : public CFactorable {
protected:
// Main constructor parameters
   bool              m_isActive;             // Is the risk manager active?

   double            m_baseBalance;          // Base balance

   ENUM_RM_CALC_DAILY_LOSS   m_calcDailyLossLimit; // Method of calculating the maximum daily loss
   double            m_maxDailyLossLimit;          // Parameter of calculating the maximum daily loss
   double            m_closeDailyPart;             // Threshold part of the daily loss

   ENUM_RM_CALC_OVERALL_LOSS m_calcOverallLossLimit;  // Method of calculating the maximum overall loss
   double            m_maxOverallLossLimit;           // Parameter of calculating the maximum overall loss
   double            m_closeOverallPart;              // Threshold part of the overall loss

   ENUM_RM_CALC_OVERALL_PROFIT m_calcOverallProfitLimit; // Method for calculating maximum overall profit
   double            m_maxOverallProfitLimit;            // Parameter for calculating the maximum overall profit

   double            m_maxRestoreTime;             // Waiting time for the best entry on a drawdown
   double            m_lastVirtualProfitFactor;    // Initial best drawdown multiplier

// Current state
   ENUM_RM_STATE     m_state;                // State
   double            m_lastVirtualProfit;    // Profit of open virtual positions at the moment of loss limit
   datetime          m_startRestoreTime;     // Start time of restoring the size of open positions

// Updated values
   double            m_balance;              // Current balance
   double            m_equity;               // Current equity
   double            m_profit;               // Current profit
   double            m_dailyProfit;          // Daily profit
   double            m_overallProfit;        // Overall profit
   double            m_baseDailyBalance;     // Daily basic balance
   double            m_baseDailyEquity;      // Daily base balance
   double            m_baseDailyLevel;       // Daily base level
   double            m_baseHWBalance;        // balance High Watermark
   double            m_baseHWEquityBalance;  // equity or balance High Watermark
   double            m_virtualProfit;        // Profit of open virtual positions

// Managing the size of open positions
   double            m_baseDepoPart;         // Used (original) part of the overall balance
   double            m_dailyDepoPart;        // Multiplier of the used part of the overall balance by daily loss
   double            m_overallDepoPart;      // Multiplier of the used part of the overall balance by overall loss

// Protected methods
   double            DailyLoss();            // Maximum daily loss
   double            OverallLoss();          // Maximum overall loss

   void              UpdateProfit();         // Update current profit values
   void              UpdateBaseLevels();     // Updating daily base levels

   void              CheckLimits();          // Check for excess of permissible losses
   bool              CheckDailyLossLimit();     // Check for excess of the permissible daily loss
   bool              CheckOverallLossLimit();   // Check for excess of the permissible overall loss
   bool              CheckOverallProfitLimit(); // Check if the specified profit has been achieved

   void              CheckRestore();         // Check the need for restoring the size of open positions
   bool              CheckDailyRestore();       // Check if the daily multiplier needs to be restored
   bool              CheckOverallRestore();     // Check if the overall multiplier needs to be restored

   double            VirtualProfit();        // Determine the profit of open virtual positions
   double            RestoreVirtualProfit(); // Determine the profit of open virtual positions to restore

   void              SetDepoPart();          // Set the values of the used part of the overall balance

public:
   ...
};
```

The _m\_closeDailyPart_ and _m\_closeOverallPart_ properties allow us to make smoother changes to the size of positions. Their usage is similar to each other, and the only difference is which limit (daily or general) each property refers to. For example, if we set _m\_closeDailyPart_ = 0.5, then when the loss reaches half of the daily limit, the position sizes will be halved. If the loss continues to grow and reaches half of the remaining half of the daily limit, then the position sizes (already halved earlier) will be halved again.

The reduction in the size of positions will be carried out by changing the _m\_dailyDepoPart_ and _m\_overallDepoPart_ properties. Their values are used in the method of setting the used portion of the overall balance for trading. They are included in the equation as multipliers, so halving any of them results in halving the overall volume:

```
//+------------------------------------------------------------------+
//| Set the value of the used part of the overall balance            |
//+------------------------------------------------------------------+
void CVirtualRiskManager::SetDepoPart() {
   CMoney::DepoPart(m_baseDepoPart * m_dailyDepoPart * m_overallDepoPart);
}
```

The _m\_baseDepoPart_ property used in the function contains the value of the original size of the used part of the balance for trading.

The _m\_maxRestoreTime_ and _m\_lastVirtualProfitFactor_ properties are used to determine the possibility of restoring the size of open positions.

The first property specifies the time in minutes, after which the sizes will be restored even in case of a non-negative value of virtual profit. That is, after this time, the EA will again open real market positions corresponding to virtual positions, even if during the time when the real positions were closed, prices went in the right direction and the virtual profit became greater than when reaching the limits. Until that time, volume recovery only occurs if the estimated profit of virtual positions is less than some estimated value that changes over time.

The second property specifies a multiplier that determines how many times the loss of virtual positions at the start of a new daily period should be greater than the loss of virtual positions at the time the limits are reached, so that the position sizes are restored immediately. For example, a value of 1 for this parameter would mean that the recovery of sizes at the beginning of the day would only occur if the virtual positions remained in the same drawdown or went into an even deeper drawdown compared to the last moment the limits were reached.

It is also worth noting that now reaching the limit is considered to have been triggered not only when the drawdown exceeds the established limit, but also, for example, when it reaches half of the daily limit, if _m\_closeDailyPart_ is 0.5.

### Calculating the profit of virtual positions

When applying partial closing and subsequent restoration of the sizes of open positions, we will need to correctly determine what the floating profit or loss would be at the moment if all positions opened according to the strategy still remained open. Therefore, the method for calculating the current profit of open virtual positions has been added to the risk manager class. They are not closed when drawdown limits are reached, so we can determine at any given time what the approximate profit would have been on open market positions corresponding to virtual positions. This calculated value does not exactly correspond to the real state of affairs due to the lack of accounting for commissions and swaps, but this accuracy will be quite sufficient for our purposes.

Last time we did not need this method for any calculations, the results of which would determine further actions. Now we need it. Besides, we should make some small additions to it for the correct calculation of the profit of virtual positions. The point is that the method used to determine the profit of one virtual position _CMoney::Profit()_ applies the balance usage multiplier _CMoney::DepoPart()_ for calculation. If we have reduced this multiplier, then the calculation of virtual profit will be made for the reduced position sizes, rather than for the sizes present before the reduction.

We are interested in the profit for the initial position sizes, so before calculating it, we temporarily return the initial balance utilization multiplier. After that, we calculate the profit of virtual positions and then set the current balance usage multiplier again by calling the _SetDepoPart() method:_

```
//+------------------------------------------------------------------+
//| Determine the profit of open virtual positions                   |
//+------------------------------------------------------------------+
double CVirtualRiskManager::VirtualProfit() {
// Access the receiver object
   CVirtualReceiver *m_receiver = CVirtualReceiver::Instance();

// Set the initial balance usage multiplier
   CMoney::DepoPart(m_baseDepoPart);

   double profit = 0;

// Find the profit sum for all virtual positions
   FORI(m_receiver.OrdersTotal(), profit += CMoney::Profit(m_receiver.Order(i)));

// Restore the current balance usage multiplier
   SetDepoPart();

   return profit;
}
```

### High Watermark

For better analysis, we would like to add the ability to calculate the maximum loss level not only from the initial account balance, but also, for example, from the last achieved balance maximum. Therefore, we have added _m\_baseHWBalance_ and _m\_baseHWEquityBalance_ to the list of properties that should be constantly updated. In the _UpdateProfit()_ method, we have added their calculation along with the check that the overall profit is calculated relative to the highest values of the balance or equity, rather than the base balance:

```
//+------------------------------------------------------------------+
//| Updating current profit values                                   |
//+------------------------------------------------------------------+
void CVirtualRiskManager::UpdateProfit() {
// Current equity
   m_equity = AccountInfoDouble(ACCOUNT_EQUITY);

// Current balance
   m_balance = AccountInfoDouble(ACCOUNT_BALANCE);

// Maximum balance (High Watermark)
   m_baseHWBalance = MathMax(m_balance, m_baseHWBalance);

// Maximum balance or equity (High Watermark)
   m_baseHWEquityBalance = MathMax(m_equity, MathMax(m_balance, m_baseHWEquityBalance));

// Current profit
   m_profit = m_equity - m_balance;

// Current daily profit relative to the daily level
   m_dailyProfit = m_equity - m_baseDailyLevel;

// Current overall profit relative to base balance
   m_overallProfit = m_equity - m_baseBalance;

// If we take the overall profit relative to the highest balance,
   if(m_calcOverallLossLimit       == RM_CALC_OVERALL_LOSS_MONEY_HW_BAL
         || m_calcOverallLossLimit == RM_CALC_OVERALL_LOSS_PERCENT_HW_BAL) {
      // Recalculate it
      m_overallProfit = m_equity - m_baseHWBalance;
   }

// If we take the overall profit relative to the highest balance or equity,
   if(m_calcOverallLossLimit       == RM_CALC_OVERALL_LOSS_MONEY_HW_EQ_BAL
         || m_calcOverallLossLimit == RM_CALC_OVERALL_LOSS_PERCENT_HW_EQ_BAL) {
      // Recalculate it
      m_overallProfit = m_equity - m_baseHWEquityBalance;
   }

// Current profit of virtual open positions
   m_virtualProfit = VirtualProfit();

   ...
}
```

### Limit check method

This method has also undergone changes: we have split its code into several auxiliary methods, so now it looks like this:

```
//+------------------------------------------------------------------+
//| Check loss limits                                                |
//+------------------------------------------------------------------+
void CVirtualRiskManager::CheckLimits() {
   if(false
         || CheckDailyLossLimit()     // Check daily limit
         || CheckOverallLossLimit()   // Check overall limit
         || CheckOverallProfitLimit() // Check overall profit
     ) {
      // Remember the current level of virtual profit
      m_lastVirtualProfit = m_virtualProfit;

      // Notify the recipient about changes
      CVirtualReceiver::Instance().Changed();
   }
}
```

In the method checking the daily loss limit, we first check whether the daily limit or a specified portion of it, defined through the _m\_closeDailyPart_ parameter, has been reached. If yes, then we reduce the multiplier of the used part of the overall balance by the daily loss. If it has already become too small, then we reset it completely. After this, we set the value of the used part of the overall balance and switch the risk manager to the state of achieved daily loss.

```
//+------------------------------------------------------------------+
//| Check daily loss limit                                           |
//+------------------------------------------------------------------+
bool CVirtualRiskManager::CheckDailyLossLimit() {
// If daily loss is reached and positions are still open
   if(m_dailyProfit < -DailyLoss() * (1 - m_dailyDepoPart * (1 - m_closeDailyPart))
      && CMoney::DepoPart() > 0) {

      // Reduce the multiplier of the used part of the overall balance by the daily loss
      m_dailyDepoPart *= (1 - m_closeDailyPart);

      // If the multiplier is already too small,
      if(m_dailyDepoPart < 0.05) {
         // Set it to 0
         m_dailyDepoPart = 0;
      }

      // Set the value of the used part of the overall balance
      SetDepoPart();

      ...

      // Set the risk manager to the achieved daily loss state
      m_state = RM_STATE_DAILY_LOSS;

      return true;
   }

   return false;
}
```

The method for checking the overall loss limit works in a similar way. The only difference is the risk manager is switched to the state of achieved overall loss only if the multiplier of the used part of the overall balance for the overall loss has become equal to zero:

```
//+------------------------------------------------------------------+
//| Check the overall loss limit                                     |
//+------------------------------------------------------------------+
bool CVirtualRiskManager::CheckOverallLossLimit() {
// If overall loss is reached and positions are still open
   if(m_overallProfit < -OverallLoss() * (1 - m_overallDepoPart * (1 - m_closeOverallPart))
         && CMoney::DepoPart() > 0) {
      // Reduce the multiplier of the used part of the overall balance by the overall loss
      m_overallDepoPart *= (1 - m_closeOverallPart);

      // If the multiplier is already too small,
      if(m_overallDepoPart < 0.05) {
         // Set it to 0
         m_overallDepoPart = 0;

         // Set the risk manager to the achieved overall loss state
         m_state = RM_STATE_OVERALL_LOSS;
      }

      // Set the value of the used part of the overall balance
      SetDepoPart();

      ...

      return true;
   }

   return false;
}
```

Checking whether the specified profit has been achieved looks even simpler: when it is achieved, we reset the corresponding multiplier and set the risk manager to the state of the achieved overall profit:

```
//+------------------------------------------------------------------+
//| Check if the specified profit has been achieved                  |
//+------------------------------------------------------------------+
bool CVirtualRiskManager::CheckOverallProfitLimit() {
// If overall loss is reached and positions are still open
   if(m_overallProfit > m_maxOverallProfitLimit && CMoney::DepoPart() > 0) {
      // Reduce the multiplier of the used part of the overall balance by the overall loss
      m_overallDepoPart = 0;

      // Set the risk manager to the achieved overall profit state
      m_state = RM_STATE_OVERALL_PROFIT;

      // Set the value of the used part of the overall balance
      SetDepoPart();

      ...

      return true;
   }

   return false;
}
```

All three methods return a boolean value that answers the question: has the corresponding limit been reached? If yes, we save the current level of virtual profit in the _CheckLimits()_ method and notify the recipient of market position volumes about changes in the composition of positions.

In the daily baseline update method, we have added recalculation of daily profit and switching to recovery state if the daily loss limit was previously reached:

```
//+------------------------------------------------------------------+
//| Update daily base levels                                         |
//+------------------------------------------------------------------+
void CVirtualRiskManager::UpdateBaseLevels() {
// Update balance, funds and base daily level
   m_baseDailyBalance = m_balance;
   m_baseDailyEquity = m_equity;
   m_baseDailyLevel = MathMax(m_baseDailyBalance, m_baseDailyEquity);

   m_dailyProfit = m_equity - m_baseDailyLevel;

   ...

// If the daily loss level was reached earlier, then
   if(m_state == RM_STATE_DAILY_LOSS) {
      // Switch to the state of restoring the sizes of open positions
      m_state = RM_STATE_RESTORE;

      // Remember restoration start time
      m_startRestoreTime = TimeCurrent();
   }
}
```

### Restoring position sizes

We have also divided the method that previously performed this task into several methods. The method for checking whether recovery is necessary is called at the top level:

```
//+------------------------------------------------------------------+
//| Check the need for restoring the size of open positions          |
//+------------------------------------------------------------------+
void CVirtualRiskManager::CheckRestore() {
// If we need to restore the state to normal, then
   if(m_state == RM_STATE_RESTORE) {
      // Check the possibility of restoring the daily loss multiplier to normal
      bool dailyRes = CheckDailyRestore();

      // Check the possibility of restoring the overall loss multiplier to normal
      bool overallRes = CheckOverallRestore();

      // If at least one of them has recovered,
      if(dailyRes || overallRes) {

...

         // Set the value of the used part of the overall balance
         SetDepoPart();

         // Notify the recipient about changes
         CVirtualReceiver::Instance().Changed();

         // If both multipliers are restored to normal,
         if(dailyRes && overallRes) {
            // Set normal state
            m_state = RM_STATE_OK;
         }
      }
   }
}
```

The auxiliary methods for checking whether the daily and overall multiplier should be restored currently work the same way, but in the future we might alter their behavior. For now, they are checking whether the current virtual profit value is less than the desired level. If so, then it is beneficial for us to reopen real positions at current prices, so we need to restore their sizes:

```
//+------------------------------------------------------------------+
//| Check if the daily multiplier needs to be restored               |
//+------------------------------------------------------------------+
bool CVirtualRiskManager::CheckDailyRestore() {
// If the current virtual profit is less than the one desired for recovery,
   if(m_virtualProfit <= RestoreVirtualProfit()) {
      // Restore the daily loss multiplier
      m_dailyDepoPart = 1.0;
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Check if the overall multiplier needs to be restored             |
//+------------------------------------------------------------------+
bool CVirtualRiskManager::CheckOverallRestore() {
// If the current virtual profit is less than the one desired for recovery,
   if(m_virtualProfit <= RestoreVirtualProfit()) {
      // Restore the overall loss multiplier
      m_overallDepoPart = 1.0;
      return true;
   }

   return false;
}
```

The _RestoreVirtualProfit()_ method calculates the desired level of virtual profit. We used a simple linear interpolation with two parameters: the more time passes, the less profitable the level, at which we agree to reopen real positions.

```
//+------------------------------------------------------------------+
//| Determine the profit of virtual positions for recovery           |
//+------------------------------------------------------------------+
double CVirtualRiskManager::RestoreVirtualProfit() {
// If the maximum recovery time is not specified,
   if(m_maxRestoreTime == 0) {
      // Return the current value of the virtual profit
      return m_virtualProfit;
   }

// Find the elapsed time since the start of recovery in minutes
   double t = (TimeCurrent() - m_startRestoreTime) / 60.0;

// Return the calculated value of the desired virtual profit
// depending on the time elapsed since the start of recovery
   return m_lastVirtualProfit * m_lastVirtualProfitFactor * (1 - t / m_maxRestoreTime);
}
```

Save the changes in the _VirtualRiskManager.mqh_ file of the current folder.

### Test

First, let's launch the EA with parameters that specify the complete closure of positions upon reaching the daily and total limits. The results should be the same as in Fig. 3.

![](https://c.mql5.com/2/110/4165020401391__2.png)

![](https://c.mql5.com/2/110/754584888399__2.png)

Fig. 6. EA results with scale\_ = 1.0 with the same settings as the original risk manager

The results for drawdown coincide completely, while in case of profit and normalized average annual profit ( _OnTester result_) the result is slightly more than expected. Well, that's encouraging: we have not broken previously implemented things.

Now let's see what happens if we set adaptive closing of positions upon reaching a part of the daily and overall limits with thevalue of 0.5:

![](https://c.mql5.com/2/110/1951599289903__2.png)

![](https://c.mql5.com/2/110/3378855416296__2.png)

Fig. 7. EA results with scale\_ = 1.0, rmCloseDailyPart\_ = 0.5 and rmCloseOverallPart\_ = 0.5

The drawdown narrowed slightly, which led to a slight increase in the normalized average annual profit. Let's now try to connect the restoration of position sizes at the best price. Let's set the waiting time for the best entry during the drawdown to 1440 minutes (1 day) in the parameters, as well as set the initial best drawdown multiplier to 1.5. These are just values taken intuitively.

![](https://c.mql5.com/2/110/4485478653336__2.png)

![](https://c.mql5.com/2/110/26997780469__2.png)

Fig. 8. EA results with scale\_ = 1.0, rmCloseDailyPart\_ = 0.5, rmCloseOverallPart\_ = 0.5, rmMaxRestoreTime\_ = 1440 and rmLastVirtualProfitFactor\_ = 1.5

The result of normalized average annual profit improved again by a few percentage points. It appears that the efforts to implement this mechanism are justified.

Now let's try to increase the size of the opened positions three times, without changing the risk manager parameters since the last launch. This should result in the risk manager being triggered more frequently. Let's see how it copes with the task.

![](https://c.mql5.com/2/110/416839647602__2.png)

![](https://c.mql5.com/2/110/2986239204457__2.png)

Fig. 9. EA results with scale\_ = 3.0, rmCloseDailyPart\_ = 0.5, rmCloseOverallPart\_ = 0.5, rmMaxRestoreTime\_ = 1440 and rmLastVirtualProfitFactor\_ = 1.5

Compared to a similar launch with the original risk manager, here we again see an improvement in results across all indicators. Compared to the previous launch, profit increased by about two times, but the drawdown increased three times, which reduced the value of the normalized average annual profit. However, for each trading day, the drawdown still did not exceed the value set in the risk manager parameters.

It is also interesting to see what happens if the size of the opened positions is increased even more, say, 10 times?

![](https://c.mql5.com/2/110/2623063232899__2.png)

![](https://c.mql5.com/2/110/5872563860575__2.png)

Fig. 10. EA results with scale\_ = 3.0, rmCloseDailyPart\_ = 0.5, rmCloseOverallPart\_ = 0.5, rmMaxRestoreTime\_ = 1440 and rmLastVirtualProfitFactor\_ = 1.5

As we can see, with such position sizes, the risk manager was no longer able to maintain compliance with the specified maximum total loss, and trading was stopped. This means the risk manager is not a universal tool allowing you to prevent a specified maximum drawdown regardless of any quirks of a trading strategy. It only complements the mandatory money management rules that should be present in a trading strategy.

Finally, let's try to select the best parameters for the risk manager using genetic optimization. Unfortunately, each pass takes about three minutes for a two-year interval, so we will have to wait a significant amount of time for the optimization to complete. As an optimization criterion, we will select a user criterion that calculates the normalized average annual profit. After some time, the top of the table with the optimization results looks like this:

![](https://c.mql5.com/2/110/3442453267045__2.png)

Fig. 11. EA optimization results with the risk manager

As we can see, selecting good risk manager parameters can not only protect trading from increased drawdowns, but also improve trading results: the difference for different passes amounted to 20% of additional profit. Although even the best of the already found sets of parameters is slightly inferior to the results obtained in Fig. 8 with intuitively selected values. Further optimization could probably improve this result a little more, but there is no particular need for it.

### Conclusion

Let's sum it all up briefly. We have improved an important component of any successful trading system - the risk manager - making it more flexible and customizable to specific needs. We now have a mechanism that allows us to more accurately comply with the established limits on acceptable drawdown.

Although we have tried to make it as reliable as possible, its capabilities should be used wisely. It is worth remembering that this is an extreme means of protecting capital, so you need to try to select trading parameters so that the risk manager either does not have to interfere with trading at all, or it does so on a very rare occasions. As we can see from the tests, when the recommended position sizes are greatly exceeded, the rate of equity change can be so sharp that even the risk manager may not have time to save the funds from exceeding the specified drawdown level.

While working on this risk manager update, new ideas for improving it emerged. However, making it too complicated is not a good thing either. Therefore, I will put aside my work on the risk manager for a while and return to more pressing EA development issues in the following parts.

Thank you for your attention! See you soon!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15085](https://www.mql5.com/ru/articles/15085)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15085.zip "Download all attachments in the single ZIP archive")

[SimpleVolumesExpert.mq5](https://www.mql5.com/en/articles/download/15085/simplevolumesexpert.mq5 "Download SimpleVolumesExpert.mq5")(15.83 KB)

[VirtualRiskManager.mqh](https://www.mql5.com/en/articles/download/15085/virtualriskmanager.mqh "Download VirtualRiskManager.mqh")(45.58 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/479365)**
(1)


![Cristian-bogdan Buzatu](https://c.mql5.com/avatar/avatar_na2.png)

**[Cristian-bogdan Buzatu](https://www.mql5.com/en/users/buza20)**
\|
14 Jan 2025 at 23:09

Hello, it's a very exciting work, thank you for your time and generosity! Could you please put together in a zip file everything we need in order to successfully compile your latest version on this script?


![News Trading Made Easy (Part 6): Performing Trades (III)](https://c.mql5.com/2/108/News_Trading_Made_Easy_oPart_6h_Performing_Trades_zIIIs___LOGO.png)[News Trading Made Easy (Part 6): Performing Trades (III)](https://www.mql5.com/en/articles/16170)

In this article news filtration for individual news events based on their IDs will be implemented. In addition, previous SQL queries will be improved to provide additional information or reduce the query's runtime. Furthermore, the code built in the previous articles will be made functional.

![Forex spread trading using seasonality](https://c.mql5.com/2/83/Trading_spreads_in_the_forex_market_using_seasonality__LOGO__1.png)[Forex spread trading using seasonality](https://www.mql5.com/en/articles/14035)

The article examines the possibilities of generating and providing reporting data on the use of the seasonality factor when trading spreads on Forex.

![Artificial Electric Field Algorithm (AEFA)](https://c.mql5.com/2/83/Artificial_Electric_Field_Algorithm___LOGO.png)[Artificial Electric Field Algorithm (AEFA)](https://www.mql5.com/en/articles/15162)

The article presents an artificial electric field algorithm (AEFA) inspired by Coulomb's law of electrostatic force. The algorithm simulates electrical phenomena to solve complex optimization problems using charged particles and their interactions. AEFA exhibits unique properties in the context of other algorithms related to laws of nature.

![Econometric tools for forecasting volatility: GARCH model](https://c.mql5.com/2/82/Econometric_Tools_for_Volatility_Forecasting__GARCH_Model____LOGO2.png)[Econometric tools for forecasting volatility: GARCH model](https://www.mql5.com/en/articles/15223)

The article describes the properties of the non-linear model of conditional heteroscedasticity (GARCH). The iGARCH indicator has been built on its basis for predicting volatility one step ahead. The ALGLIB numerical analysis library is used to estimate the model parameters.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/15085&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048838733739958186)

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