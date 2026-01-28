---
title: Library for easy and quick development of MetaTrader programs (part XIII): Account object events
url: https://www.mql5.com/en/articles/6995
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:40:27.924300
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/6995&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070473128255100535)

MetaTrader 5 / Examples


### Contents

- [The concept of an account event](https://www.mql5.com/en/articles/6995#node01)
- [Methods of working with account events](https://www.mql5.com/en/articles/6995#node02)
- [Testing the account events](https://www.mql5.com/en/articles/6995#node03)
- [What's next?](https://www.mql5.com/en/articles/6995#node04)


### The concept of an account event

Previously, we have created a separate class that tracks order and position events ( [parts \\
IV](https://www.mql5.com/en/articles/5724)- [XI](https://www.mql5.com/en/articles/6921)) and sends data on detected changes to the CEngine main library object.

In order to track account events, we will use another method: since we are able to track events on a single account only (the one the
terminal is currently connected to), a separate class for that would be redundant. Instead, we will create the methods of working with
events directly in the account collection class.

To detect any changes in the account properties, we are going to compare the current account properties with their previous status. If
changes are detected, an event will be sent to the control program chart.

We have already developed some functionality for tracking event accounts [in \\
the previous article](https://www.mql5.com/en/articles/6952) when creating the account object collection. In this article, we will improve the existing functionality to make
it fully usable.

### Methods of working with account events

Let's start by creating the necessary enumerations and macro substitutions in the **Defines.mqh** file. Since a
response was received from the developers about the actual size allocated for the account string properties in bytes, we will set them
tightly in the

**CAccount** class code, thus eliminating the need for the macro substitution setting the size of uchar arrays for storing the account
object string properties.

Remove the macro substitution from the Defines.mqh listing

```
#define UCHAR_ARRAY_SIZE            (64)                       // Size of uchar arrays for storing string properties
```

I have also decided to add all the necessary enumerations and macro substitutions for working with account objects to the very end of the
file — after the data for working with trading events since we are going to send not only account event codes but also implemented trading
event ones to the program. This means the numerical values of the account event codes are to start from the numerical value of the very last
trading event code+1. If the number of trading events increases, we will declare the macro substitution containing the total number of all
trading events in order not to re-write numerical values of the account event codes. Also, we will set the macro substitution specifying the
total number of account events in case we implement some other events (like symbol events). In this case, numerical codes of these new events
start from the total number of account events+1.

At the end of the list of possible trading events on the account, insert
the macro substitution specifying the value corresponding to the numerical value of the last trading event+1:

```
//+------------------------------------------------------------------+
//| List of possible trading events on the account                   |
//+------------------------------------------------------------------+
enum ENUM_TRADE_EVENT
  {
   TRADE_EVENT_NO_EVENT = 0,                                // No trading event
   TRADE_EVENT_PENDING_ORDER_PLASED,                        // Pending order placed
   TRADE_EVENT_PENDING_ORDER_REMOVED,                       // Pending order removed
//--- enumeration members matching the ENUM_DEAL_TYPE enumeration members
//--- (constant order below should not be changed, no constants should be added/deleted)
   TRADE_EVENT_ACCOUNT_CREDIT = DEAL_TYPE_CREDIT,           // Accruing credit (3)
   TRADE_EVENT_ACCOUNT_CHARGE,                              // Additional charges
   TRADE_EVENT_ACCOUNT_CORRECTION,                          // Correcting entry
   TRADE_EVENT_ACCOUNT_BONUS,                               // Accruing bonuses
   TRADE_EVENT_ACCOUNT_COMISSION,                           // Additional commissions
   TRADE_EVENT_ACCOUNT_COMISSION_DAILY,                     // Commission charged at the end of a trading day
   TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY,                   // Commission charged at the end of a trading month
   TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY,               // Agent commission charged at the end of a trading day
   TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY,             // Agent commission charged at the end of a month
   TRADE_EVENT_ACCOUNT_INTEREST,                            // Accrued interest on free funds
   TRADE_EVENT_BUY_CANCELLED,                               // Canceled buy deal
   TRADE_EVENT_SELL_CANCELLED,                              // Canceled sell deal
   TRADE_EVENT_DIVIDENT,                                    // Accruing dividends
   TRADE_EVENT_DIVIDENT_FRANKED,                            // Accruing franked dividends
   TRADE_EVENT_TAX                        = DEAL_TAX,       // Tax
//--- constants related to the DEAL_TYPE_BALANCE deal type from the DEAL_TYPE_BALANCE enumeration
   TRADE_EVENT_ACCOUNT_BALANCE_REFILL     = DEAL_TAX+1,     // Replenishing account balance
   TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL = DEAL_TAX+2,     // Withdrawing funds from an account
//--- Remaining possible trading events//--- (constant order below can be changed, constants can be added/deleted)
   TRADE_EVENT_PENDING_ORDER_ACTIVATED    = DEAL_TAX+3,     // Pending order activated by price
   TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL,             // Pending order partially activated by price
   TRADE_EVENT_POSITION_OPENED,                             // Position opened
   TRADE_EVENT_POSITION_OPENED_PARTIAL,                     // Position opened partially
   TRADE_EVENT_POSITION_CLOSED,                             // Position closed
   TRADE_EVENT_POSITION_CLOSED_BY_POS,                      // Position closed by an opposite one
   TRADE_EVENT_POSITION_CLOSED_BY_SL,                       // Position closed by StopLoss
   TRADE_EVENT_POSITION_CLOSED_BY_TP,                       // Position closed by TakeProfit
   TRADE_EVENT_POSITION_REVERSED_BY_MARKET,                 // Position reversal by a new deal (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_PENDING,                // Position reversal by activating a pending order (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL,         // Position reversal by partial market order execution (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL,        // Position reversal by partial pending order activation (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET,               // Added volume to a position by a new deal (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL,       // Added volume to a position by partial activation of an order (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING,              // Added volume to a position by activating a pending order (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL,      // Added volume to a position by partial activation of a pending order (netting)
   TRADE_EVENT_POSITION_CLOSED_PARTIAL,                     // Position closed partially
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS,              // Position closed partially by an opposite one
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL,               // Position closed partially by StopLoss
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP,               // Position closed partially by TakeProfit
   TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER,                  // StopLimit order activation
   TRADE_EVENT_MODIFY_ORDER_PRICE,                          // Changing order price
   TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS,                // Changing order and StopLoss price
   TRADE_EVENT_MODIFY_ORDER_PRICE_TAKE_PROFIT,              // Changing order and TakeProfit price
   TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT,    // Changing order, StopLoss and TakeProfit price
   TRADE_EVENT_MODIFY_ORDER_STOP_LOSS_TAKE_PROFIT,          // Changing order's StopLoss and TakeProfit price
   TRADE_EVENT_MODIFY_ORDER_STOP_LOSS,                      // Changing order Stop Loss
   TRADE_EVENT_MODIFY_ORDER_TAKE_PROFIT,                    // Changing order Take Profit
   TRADE_EVENT_MODIFY_POSITION_STOP_LOSS_TAKE_PROFIT,       // Changing position StopLoss and TakeProfit
   TRADE_EVENT_MODIFY_POSITION_STOP_LOSS,                   // Changing position StopLoss
   TRADE_EVENT_MODIFY_POSITION_TAKE_PROFIT,                 // Changing position TakeProfit
  };
#define TRADE_EVENTS_NEXT_CODE      (TRADE_EVENT_MODIFY_POSITION_TAKE_PROFIT+1)// The code of the next event after the last trading event code
//+------------------------------------------------------------------+
```

This is the code the account event codes are to start from.

In the previous article, we have set the account integer, real and string
properties, as well as possible account sorting criteria and placed them before the data for working with event accounts. Now we
move them to the end and provide additional data for working with account events —

the list of account event flags and possible account events:

```
//+------------------------------------------------------------------+
//| Data for working with accounts                                   |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of account event flags                                      |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_EVENT_FLAGS
  {
   ACCOUNT_EVENT_FLAG_NO_EVENT            =  0,             // No event
   ACCOUNT_EVENT_FLAG_LEVERAGE            =  1,             // Changing the leverage
   ACCOUNT_EVENT_FLAG_LIMIT_ORDERS        =  2,             // Changing the maximum allowed number of active pending orders
   ACCOUNT_EVENT_FLAG_TRADE_ALLOWED       =  4,             // Changing permission to trade for the account
   ACCOUNT_EVENT_FLAG_TRADE_EXPERT        =  8,             // Changing permission for auto trading for the account
   ACCOUNT_EVENT_FLAG_BALANCE             =  16,            // The balance exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_EQUITY              =  32,            // The equity exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_PROFIT              =  64,            // The profit exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_CREDIT              =  128,           // Changing the credit in a deposit currency
   ACCOUNT_EVENT_FLAG_MARGIN              =  256,           // The reserved margin on an account in the deposit currency exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_FREE         =  512,           // The free funds available for opening a position in a deposit currency exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_LEVEL        =  1024,          // The margin level on an account in % exceeds the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_INITIAL      =  2048,          // The funds reserved on an account to ensure a guarantee amount for all pending orders exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_MAINTENANCE  =  4096,          // The funds reserved on an account to ensure a minimum amount for all open positions exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_MARGIN_SO_CALL      =  8192,          // Changing the Margin Call level
   ACCOUNT_EVENT_FLAG_MARGIN_SO_SO        =  16384,         // Changing the Stop Out level
   ACCOUNT_EVENT_FLAG_ASSETS              =  32768,         // The current assets on an account exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_LIABILITIES         =  65536,         // The current liabilities on an account exceed the specified change value +/-
   ACCOUNT_EVENT_FLAG_COMISSION_BLOCKED   =  131072,        // The current sum of blocked commissions on an account exceeds the specified change value +/-
  };
//+------------------------------------------------------------------+
//| List of possible account events                                  |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_EVENT
  {
   ACCOUNT_EVENT_NO_EVENT = TRADE_EVENTS_NEXT_CODE,         // No event
   ACCOUNT_EVENT_LEVERAGE_INC,                              // Increasing the leverage
   ACCOUNT_EVENT_LEVERAGE_DEC,                              // Decreasing the leverage
   ACCOUNT_EVENT_LIMIT_ORDERS_INC,                          // Increasing the maximum allowed number of active pending orders
   ACCOUNT_EVENT_LIMIT_ORDERS_DEC,                          // Decreasing the maximum allowed number of active pending orders
   ACCOUNT_EVENT_TRADE_ALLOWED_ON,                          // Enabling trading for the account
   ACCOUNT_EVENT_TRADE_ALLOWED_OFF,                         // Disabling trading for the account
   ACCOUNT_EVENT_TRADE_EXPERT_ON,                           // Enabling auto trading for the account
   ACCOUNT_EVENT_TRADE_EXPERT_OFF,                          // Disabling auto trading for the account
   ACCOUNT_EVENT_BALANCE_INC,                               // The balance exceeds the specified value
   ACCOUNT_EVENT_BALANCE_DEC,                               // The balance falls below the specified value
   ACCOUNT_EVENT_EQUITY_INC,                                // The equity exceeds the specified value
   ACCOUNT_EVENT_EQUITY_DEC,                                // The equity falls below the specified value
   ACCOUNT_EVENT_PROFIT_INC,                                // The profit exceeds the specified value
   ACCOUNT_EVENT_PROFIT_DEC,                                // The profit falls below the specified value
   ACCOUNT_EVENT_CREDIT_INC,                                // The credit exceeds the specified value
   ACCOUNT_EVENT_CREDIT_DEC,                                // The credit falls below the specified value
   ACCOUNT_EVENT_MARGIN_INC,                                // Increasing the reserved margin on an account in the deposit currency
   ACCOUNT_EVENT_MARGIN_DEC,                                // Decreasing the reserved margin on an account in the deposit currency
   ACCOUNT_EVENT_MARGIN_FREE_INC,                           // Increasing the free funds available for opening a position in a deposit currency
   ACCOUNT_EVENT_MARGIN_FREE_DEC,                           // Decreasing the free funds available for opening a position in a deposit currency
   ACCOUNT_EVENT_MARGIN_LEVEL_INC,                          // Increasing the margin level on an account in %
   ACCOUNT_EVENT_MARGIN_LEVEL_DEC,                          // Decreasing the margin level on an account in %
   ACCOUNT_EVENT_MARGIN_INITIAL_INC,                        // Increasing the funds reserved on an account to ensure a guarantee amount for all pending orders
   ACCOUNT_EVENT_MARGIN_INITIAL_DEC,                        // Decreasing the funds reserved on an account to ensure a guarantee amount for all pending orders
   ACCOUNT_EVENT_MARGIN_MAINTENANCE_INC,                    // Increasing the funds reserved on an account to ensure a minimum amount for all open positions
   ACCOUNT_EVENT_MARGIN_MAINTENANCE_DEC,                    // Decreasing the funds reserved on an account to ensure a minimum amount for all open positions
   ACCOUNT_EVENT_MARGIN_SO_CALL_INC,                        // Increasing the Margin Call level
   ACCOUNT_EVENT_MARGIN_SO_CALL_DEC,                        // Decreasing the Margin Call level
   ACCOUNT_EVENT_MARGIN_SO_SO_INC,                          // Increasing the Stop Out level
   ACCOUNT_EVENT_MARGIN_SO_SO_DEC,                          // Decreasing the Stop Out level
   ACCOUNT_EVENT_ASSETS_INC,                                // Increasing the current asset size on the account
   ACCOUNT_EVENT_ASSETS_DEC,                                // Decreasing the current asset size on the account
   ACCOUNT_EVENT_LIABILITIES_INC,                           // Increasing the current liabilities on the account
   ACCOUNT_EVENT_LIABILITIES_DEC,                           // Decreasing the current liabilities on the account
   ACCOUNT_EVENT_COMISSION_BLOCKED_INC,                     // Increasing the current sum of blocked commissions on an account
   ACCOUNT_EVENT_COMISSION_BLOCKED_DEC,                     // Decreasing the current sum of blocked commissions on an account<
  };
#define ACCOUNT_EVENTS_NEXT_CODE       (ACCOUNT_EVENT_COMISSION_BLOCKED_DEC+1)   // The code of the next event after the last account event code
//+------------------------------------------------------------------+
//| Account integer properties                                       |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_PROP_INTEGER
  {
   ACCOUNT_PROP_LOGIN,                                      // Account number
   ACCOUNT_PROP_TRADE_MODE,                                 // Trading account type
   ACCOUNT_PROP_LEVERAGE,                                   // Provided leverage
   ACCOUNT_PROP_LIMIT_ORDERS,                               // Maximum allowed number of active pending orders
   ACCOUNT_PROP_MARGIN_SO_MODE,                             // Mode of setting the minimum available margin level
   ACCOUNT_PROP_TRADE_ALLOWED,                              // Permission to trade for the current account from the server side
   ACCOUNT_PROP_TRADE_EXPERT,                               // Permission to trade for an EA from the server side
   ACCOUNT_PROP_MARGIN_MODE,                                // Margin calculation mode
   ACCOUNT_PROP_CURRENCY_DIGITS,                            // Number of digits for an account currency necessary for accurate display of trading results
   ACCOUNT_PROP_SERVER_TYPE                                 // Trade server type (MetaTrader 5, MetaTrader 4)
  };
#define ACCOUNT_PROP_INTEGER_TOTAL    (10)                  // Total number of account's integer properties
#define ACCOUNT_PROP_INTEGER_SKIP     (0)                   // Number of account's integer properties not used in sorting
//+------------------------------------------------------------------+
//| Account real properties                                          |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_PROP_DOUBLE
  {
   ACCOUNT_PROP_BALANCE = ACCOUNT_PROP_INTEGER_TOTAL,       // Account balance in a deposit currency
   ACCOUNT_PROP_CREDIT,                                     // Credit in a deposit currency
   ACCOUNT_PROP_PROFIT,                                     // Current profit on an account in the account currency
   ACCOUNT_PROP_EQUITY,                                     // Equity on an account in the deposit currency
   ACCOUNT_PROP_MARGIN,                                     // Reserved margin on an account in a deposit currency
   ACCOUNT_PROP_MARGIN_FREE,                                // Free funds available for opening a position in a deposit currency
   ACCOUNT_PROP_MARGIN_LEVEL,                               // Margin level on an account in %
   ACCOUNT_PROP_MARGIN_SO_CALL,                             // Margin Call level
   ACCOUNT_PROP_MARGIN_SO_SO,                               // Stop Out level
   ACCOUNT_PROP_MARGIN_INITIAL,                             // Funds reserved on an account to ensure a guarantee amount for all pending orders
   ACCOUNT_PROP_MARGIN_MAINTENANCE,                         // Funds reserved on an account to ensure a minimum amount for all open positions
   ACCOUNT_PROP_ASSETS,                                     // Current assets on an account
   ACCOUNT_PROP_LIABILITIES,                                // Current liabilities on an account
   ACCOUNT_PROP_COMMISSION_BLOCKED                          // Current sum of blocked commissions on an account
  };
#define ACCOUNT_PROP_DOUBLE_TOTAL     (14)                  // Total number of account's real properties
#define ACCOUNT_PROP_DOUBLE_SKIP      (0)                   // Number of account's real properties not used in sorting
//+------------------------------------------------------------------+
//| Account string properties                                        |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_PROP_STRING
  {
   ACCOUNT_PROP_NAME = (ACCOUNT_PROP_INTEGER_TOTAL+ACCOUNT_PROP_DOUBLE_TOTAL), // Client name
   ACCOUNT_PROP_SERVER,                                     // Trade server name
   ACCOUNT_PROP_CURRENCY,                                   // Deposit currency
   ACCOUNT_PROP_COMPANY                                     // Name of a company serving an account
  };
#define ACCOUNT_PROP_STRING_TOTAL     (4)                   // Total number of account's string properties
#define ACCOUNT_PROP_STRING_SKIP      (0)                   // Number of account string properties not used in sorting
//+------------------------------------------------------------------+
//| Possible account sorting criteria                                |
//+------------------------------------------------------------------+
#define FIRST_ACC_DBL_PROP            (ACCOUNT_PROP_INTEGER_TOTAL-ACCOUNT_PROP_INTEGER_SKIP)
#define FIRST_ACC_STR_PROP            (ACCOUNT_PROP_INTEGER_TOTAL-ACCOUNT_PROP_INTEGER_SKIP+ACCOUNT_PROP_DOUBLE_TOTAL-ACCOUNT_PROP_DOUBLE_SKIP)
enum ENUM_SORT_ACCOUNT_MODE
  {
   SORT_BY_ACCOUNT_LOGIN               =  0,                      // Sort by account number
   SORT_BY_ACCOUNT_TRADE_MODE          =  1,                      // Sort by trading account type
   SORT_BY_ACCOUNT_LEVERAGE            =  2,                      // Sort by leverage
   SORT_BY_ACCOUNT_LIMIT_ORDERS        =  3,                      // Sort by maximum acceptable number of existing pending orders
   SORT_BY_ACCOUNT_MARGIN_SO_MODE      =  4,                      // Sort by mode for setting the minimum acceptable margin level
   SORT_BY_ACCOUNT_TRADE_ALLOWED       =  5,                      // Sort by permission to trade for the current account
   SORT_BY_ACCOUNT_TRADE_EXPERT        =  6,                      // Sort by permission to trade for an EA
   SORT_BY_ACCOUNT_MARGIN_MODE         =  7,                      // Sort by margin calculation mode
   SORT_BY_ACCOUNT_CURRENCY_DIGITS     =  8,                      // Sort by number of digits for an account currency
   SORT_BY_ACCOUNT_SERVER_TYPE         =  9,                      // Sort by trade server type (MetaTrader 5, MetaTrader 4)

   SORT_BY_ACCOUNT_BALANCE             =  FIRST_ACC_DBL_PROP,     // Sort by an account balance in the deposit currency
   SORT_BY_ACCOUNT_CREDIT              =  FIRST_ACC_DBL_PROP+1,   // Sort by credit in a deposit currency
   SORT_BY_ACCOUNT_PROFIT              =  FIRST_ACC_DBL_PROP+2,   // Sort by the current profit on an account in the deposit currency
   SORT_BY_ACCOUNT_EQUITY              =  FIRST_ACC_DBL_PROP+3,   // Sort by an account equity in the deposit currency
   SORT_BY_ACCOUNT_MARGIN              =  FIRST_ACC_DBL_PROP+4,   // Sort by an account reserved margin in the deposit currency
   SORT_BY_ACCOUNT_MARGIN_FREE         =  FIRST_ACC_DBL_PROP+5,   // Sort by account free funds available for opening a position in the deposit currency
   SORT_BY_ACCOUNT_MARGIN_LEVEL        =  FIRST_ACC_DBL_PROP+6,   // Sort by account margin level in %
   SORT_BY_ACCOUNT_MARGIN_SO_CALL      =  FIRST_ACC_DBL_PROP+7,   // Sort by margin level requiring depositing funds to an account (Margin Call)
   SORT_BY_ACCOUNT_MARGIN_SO_SO        =  FIRST_ACC_DBL_PROP+8,   // Sort by margin level, at which the most loss-making position is closed (Stop Out)
   SORT_BY_ACCOUNT_MARGIN_INITIAL      =  FIRST_ACC_DBL_PROP+9,   // Sort by funds reserved on an account to ensure a guarantee amount for all pending orders
   SORT_BY_ACCOUNT_MARGIN_MAINTENANCE  =  FIRST_ACC_DBL_PROP+10,  // Sort by funds reserved on an account to ensure a minimum amount for all open positions
   SORT_BY_ACCOUNT_ASSETS              =  FIRST_ACC_DBL_PROP+11,  // Sort by the amount of the current assets on an account
   SORT_BY_ACCOUNT_LIABILITIES         =  FIRST_ACC_DBL_PROP+12,  // Sort by the current liabilities on an account
   SORT_BY_ACCOUNT_COMMISSION_BLOCKED  =  FIRST_ACC_DBL_PROP+13,  // Sort by the current amount of blocked commissions on an account

   SORT_BY_ACCOUNT_NAME                =  FIRST_ACC_STR_PROP,     // Sort by a client name
   SORT_BY_ACCOUNT_SERVER              =  FIRST_ACC_STR_PROP+1,   // Sort by a trade server name
   SORT_BY_ACCOUNT_CURRENCY            =  FIRST_ACC_STR_PROP+2,   // Sort by a deposit currency
   SORT_BY_ACCOUNT_COMPANY             =  FIRST_ACC_STR_PROP+3    // Sort by a name of a company serving an account
  };
//+------------------------------------------------------------------+
```

Since multiple account properties can change at once, we are going to work with a set of event flags in order not to miss a single one of the occurred
changes. The flags will be added to an event code variable. Then the presence of a certain flag within the variable will be checked. Based on
the flags present in the event code, we will define what exactly happened in the account properties. All detected events are to be stored in
the array. The ability to access it is to be provided in the CEngine class and, later, in the program.

As you can see, the list of account integer properties features yet another
property — trade server type. Accordingly, the total
number of integer properties has been changed from **9** to **10**.

This account property is to define whether an account belongs to MetaTrader 5 or MetaTrader 4. I have decided to add the property because
the list of all accounts the program ever connected to receives all accounts — both MetaTrader 5 and MetaTrader 4 ones if the library-based
program is launched in both terminals. The property allows us to distinguish the trade server types. It is to be present in the account
description displayed in the journal by the PrintShort() method of the CAccount class. Besides, we are able to sort account objects by the
platform they belong to.

**Let's move on to the Acount.mqh file. Add the new property**
**and set the exact size of the arrays for storing the account string properties**
**in the private section of the account data structure:**

```
//+------------------------------------------------------------------+
//| Account class                                                    |
//+------------------------------------------------------------------+
class CAccount : public CObject
  {
private:
   struct SData
     {
      //--- Account integer properties
      long           login;                        // ACCOUNT_LOGIN (Account number)
      int            trade_mode;                   // ACCOUNT_TRADE_MODE (Trading account type)
      long           leverage;                     // ACCOUNT_LEVERAGE (Leverage)
      int            limit_orders;                 // ACCOUNT_LIMIT_ORDERS (Maximum allowed number of active pending orders)
      int            margin_so_mode;               // ACCOUNT_MARGIN_SO_MODE (Mode of setting the minimum available margin level)
      bool           trade_allowed;                // ACCOUNT_TRADE_ALLOWED (Permission to trade for the current account from the server side)
      bool           trade_expert;                 // ACCOUNT_TRADE_EXPERT (Permission to trade for an EA from the server side)
      int            margin_mode;                  // ACCOUNT_MARGIN_MODE (Margin calculation mode)
      int            currency_digits;              // ACCOUNT_CURRENCY_DIGITS (Number of digits for an account currency)
      int            server_type;                  // Trade server type (MetaTrader 5, MetaTrader 4)
      //--- Account real properties
      double         balance;                      // ACCOUNT_BALANCE (Account balance in the deposit currency)
      double         credit;                       // ACCOUNT_CREDIT (Credit in the deposit currency)
      double         profit;                       // ACCOUNT_PROFIT (Current profit on an account in the deposit currency)
      double         equity;                       // ACCOUNT_EQUITY (Equity on an account in the deposit currency)
      double         margin;                       // ACCOUNT_MARGIN (Reserved margin on an account in the deposit currency)
      double         margin_free;                  // ACCOUNT_MARGIN_FREE (Free funds available for opening a position in the deposit currency)
      double         margin_level;                 // ACCOUNT_MARGIN_LEVEL (Margin level on an account in %)
      double         margin_so_call;               // ACCOUNT_MARGIN_SO_CALL (Margin Call level)
      double         margin_so_so;                 // ACCOUNT_MARGIN_SO_SO (Stop Out level)
      double         margin_initial;               // ACCOUNT_MARGIN_INITIAL (Funds reserved on an account to ensure a guarantee amount for all pending orders)
      double         margin_maintenance;           // ACCOUNT_MARGIN_MAINTENANCE (Funds reserved on an account to ensure a minimum amount for all open positions)
      double         assets;                       // ACCOUNT_ASSETS (Current assets on an account)
      double         liabilities;                  // ACCOUNT_LIABILITIES (Current liabilities on an account)
      double         comission_blocked;            // ACCOUNT_COMMISSION_BLOCKED (Current sum of blocked commissions on an account)
      //--- Account string properties
      uchar          name[128];                    // ACCOUNT_NAME (Client name)
      uchar          server[64];                   // ACCOUNT_SERVER (Trade server name)
      uchar          currency[32];                 // ACCOUNT_CURRENCY (Deposit currency)
      uchar          company[128];                 // ACCOUNT_COMPANY (Name of a company serving an account)
     };
   SData             m_struct_obj;                                      // Account object structure
   uchar             m_uchar_array[];                                   // uchar array of the account object structure
```

**In the class public section, add yet another method for a simplified**
**access to the account object properties returning the trade server type:**

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the account object properties  |
//+------------------------------------------------------------------+
//--- Return the account's integer propertie
   ENUM_ACCOUNT_TRADE_MODE    TradeMode(void)                     const { return (ENUM_ACCOUNT_TRADE_MODE)this.GetProperty(ACCOUNT_PROP_TRADE_MODE);       }
   ENUM_ACCOUNT_STOPOUT_MODE  MarginSOMode(void)                  const { return (ENUM_ACCOUNT_STOPOUT_MODE)this.GetProperty(ACCOUNT_PROP_MARGIN_SO_MODE); }
   ENUM_ACCOUNT_MARGIN_MODE   MarginMode(void)                    const { return (ENUM_ACCOUNT_MARGIN_MODE)this.GetProperty(ACCOUNT_PROP_MARGIN_MODE);     }
   long              Login(void)                                  const { returnthis.GetProperty(ACCOUNT_PROP_LOGIN);                                      }
   long              Leverage(void)                               const { returnthis.GetProperty(ACCOUNT_PROP_LEVERAGE);                                   }
   long              LimitOrders(void)                            const { returnthis.GetProperty(ACCOUNT_PROP_LIMIT_ORDERS);                               }
   long              TradeAllowed(void)                           const { returnthis.GetProperty(ACCOUNT_PROP_TRADE_ALLOWED);                              }
   long              TradeExpert(void)                            const { returnthis.GetProperty(ACCOUNT_PROP_TRADE_EXPERT);                               }
   long              CurrencyDigits(void)                         const { returnthis.GetProperty(ACCOUNT_PROP_CURRENCY_DIGITS);                            }
   long              ServerType(void)                             const { returnthis.GetProperty(ACCOUNT_PROP_SERVER_TYPE);                                }
```

**In the methods of the account properties description, add yet another method**
**returning the trade server type description:**

```
//+------------------------------------------------------------------+
//| Descriptions of the account object properties                    |
//+------------------------------------------------------------------+
//--- Return the description of the account's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_ACCOUNT_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_ACCOUNT_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_ACCOUNT_PROP_STRING property);
//--- Return the trading account type name (demo, contest, real)
   string            TradeModeDescription(void)    const;
//--- Return the trade server type name (MetaTrader 5, MetaTrader 4)
   string            ServerTypeDescription(void)    const;
//--- Return the description of the mode for setting the minimum available margin level
   string            MarginSOModeDescription(void) const;
//--- Return the description of the margin calculation mode
   string            MarginModeDescription(void)   const;
//--- Display the description of the account properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(constbool full_prop=false);
//--- Display a short account description in the journal
   void              PrintShort(void);
```

**In the class constructor, fill in the property storing the trade server**
**type:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccount::CAccount(void)
  {
//--- Save integer properties
   this.m_long_prop[ACCOUNT_PROP_LOGIN]                              = ::AccountInfoInteger(ACCOUNT_LOGIN);
   this.m_long_prop[ACCOUNT_PROP_TRADE_MODE]                         = ::AccountInfoInteger(ACCOUNT_TRADE_MODE);
   this.m_long_prop[ACCOUNT_PROP_LEVERAGE]                           = ::AccountInfoInteger(ACCOUNT_LEVERAGE);
   this.m_long_prop[ACCOUNT_PROP_LIMIT_ORDERS]                       = ::AccountInfoInteger(ACCOUNT_LIMIT_ORDERS);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_SO_MODE]                     = ::AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
   this.m_long_prop[ACCOUNT_PROP_TRADE_ALLOWED]                      = ::AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   this.m_long_prop[ACCOUNT_PROP_TRADE_EXPERT]                       = ::AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_MODE]                        = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_MARGIN_MODE) #else ACCOUNT_MARGIN_MODE_RETAIL_HEDGING#endif ;
   this.m_long_prop[ACCOUNT_PROP_CURRENCY_DIGITS]                    = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2#endif ;
   this.m_long_prop[ACCOUNT_PROP_SERVER_TYPE]                        = (::TerminalInfoString(TERMINAL_NAME)=="MetaTrader 5" ? 5 : 4);

//--- Save real properties
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_BALANCE)]          = ::AccountInfoDouble(ACCOUNT_BALANCE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_CREDIT)]           = ::AccountInfoDouble(ACCOUNT_CREDIT);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_PROFIT)]           = ::AccountInfoDouble(ACCOUNT_PROFIT);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_EQUITY)]           = ::AccountInfoDouble(ACCOUNT_EQUITY);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN)]           = ::AccountInfoDouble(ACCOUNT_MARGIN);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_FREE)]      = ::AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_LEVEL)]     = ::AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_SO_CALL)]   = ::AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_SO_SO)]     = ::AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_INITIAL)]   = ::AccountInfoDouble(ACCOUNT_MARGIN_INITIAL);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_MAINTENANCE)]=::AccountInfoDouble(ACCOUNT_MARGIN_MAINTENANCE);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_ASSETS)]           = ::AccountInfoDouble(ACCOUNT_ASSETS);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_LIABILITIES)]      = ::AccountInfoDouble(ACCOUNT_LIABILITIES);
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_COMMISSION_BLOCKED)]=::AccountInfoDouble(ACCOUNT_COMMISSION_BLOCKED);

//--- Save string properties
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_NAME)]             = ::AccountInfoString(ACCOUNT_NAME);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_SERVER)]           = ::AccountInfoString(ACCOUNT_SERVER);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_CURRENCY)]         = ::AccountInfoString(ACCOUNT_CURRENCY);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_COMPANY)]          = ::AccountInfoString(ACCOUNT_COMPANY);
  }
//+-------------------------------------------------------------------+
```

Here, simply read the [terminal \\
name](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus#enum_terminal_info_string). If this is MetaTrader 5, set 5, otherwise, set 4.

**Add filling in the new property to the ObjectToStruct()**
**method for creating an object structure:**

```
//+------------------------------------------------------------------+
//| Create the account object structure                              |
//+------------------------------------------------------------------+
bool CAccount::ObjectToStruct(void)
  {
//--- Save the integer properties
   this.m_struct_obj.login=this.Login();
   this.m_struct_obj.trade_mode=this.TradeMode();
   this.m_struct_obj.leverage=this.Leverage();
   this.m_struct_obj.limit_orders=(int)this.LimitOrders();
   this.m_struct_obj.margin_so_mode=this.MarginSOMode();
   this.m_struct_obj.trade_allowed=this.TradeAllowed();
   this.m_struct_obj.trade_expert=this.TradeExpert();
   this.m_struct_obj.margin_mode=this.MarginMode();
   this.m_struct_obj.currency_digits=(int)this.CurrencyDigits();
   this.m_struct_obj.server_type=(int)this.ServerType();
//--- Save the real properties
   this.m_struct_obj.balance=this.Balance();
   this.m_struct_obj.credit=this.Credit();
   this.m_struct_obj.profit=this.Profit();
   this.m_struct_obj.equity=this.Equity();
   this.m_struct_obj.margin=this.Margin();
   this.m_struct_obj.margin_free=this.MarginFree();
   this.m_struct_obj.margin_level=this.MarginLevel();
   this.m_struct_obj.margin_so_call=this.MarginSOCall();
   this.m_struct_obj.margin_so_so=this.MarginSOSO();
   this.m_struct_obj.margin_initial=this.MarginInitial();
   this.m_struct_obj.margin_maintenance=this.MarginMaintenance();
   this.m_struct_obj.assets=this.Assets();
   this.m_struct_obj.liabilities=this.Liabilities();
   this.m_struct_obj.comission_blocked=this.ComissionBlocked();
//--- Save the string properties
   ::StringToCharArray(this.Name(),this.m_struct_obj.name);
   ::StringToCharArray(this.Server(),this.m_struct_obj.server);
   ::StringToCharArray(this.Currency(),this.m_struct_obj.currency);
   ::StringToCharArray(this.Company(),this.m_struct_obj.company);
   //--- Save the structure to the uchar array
   ::ResetLastError();
   if(!::StructToCharArray(this.m_struct_obj,this.m_uchar_array))
     {
      ::Print(DFUN,TextByLanguage("Не удалось сохранить структуру объекта в uchar-массив, ошибка ","Failed to save object structure to uchar array, error "),(string)::GetLastError());
      returnfalse;
     }
   returntrue;
  }
//+------------------------------------------------------------------+
```

while the **method for creating an object from the StructToObject() structure**
**receives obtaining a property from the structure:**

```
//+------------------------------------------------------------------+
//| Create the account object from the structure                     |
//+------------------------------------------------------------------+
void CAccount::StructToObject(void)
  {
//--- Save integer properties
   this.m_long_prop[ACCOUNT_PROP_LOGIN]                              = this.m_struct_obj.login;
   this.m_long_prop[ACCOUNT_PROP_TRADE_MODE]                         = this.m_struct_obj.trade_mode;
   this.m_long_prop[ACCOUNT_PROP_LEVERAGE]                           = this.m_struct_obj.leverage;
   this.m_long_prop[ACCOUNT_PROP_LIMIT_ORDERS]                       = this.m_struct_obj.limit_orders;
   this.m_long_prop[ACCOUNT_PROP_MARGIN_SO_MODE]                     = this.m_struct_obj.margin_so_mode;
   this.m_long_prop[ACCOUNT_PROP_TRADE_ALLOWED]                      = this.m_struct_obj.trade_allowed;
   this.m_long_prop[ACCOUNT_PROP_TRADE_EXPERT]                       = this.m_struct_obj.trade_expert;
   this.m_long_prop[ACCOUNT_PROP_MARGIN_MODE]                        = this.m_struct_obj.margin_mode;
   this.m_long_prop[ACCOUNT_PROP_CURRENCY_DIGITS]                    = this.m_struct_obj.currency_digits;
   this.m_long_prop[ACCOUNT_PROP_SERVER_TYPE]                        = this.m_struct_obj.server_type;
//--- Save real properties
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_BALANCE)]          = this.m_struct_obj.balance;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_CREDIT)]           = this.m_struct_obj.credit;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_PROFIT)]           = this.m_struct_obj.profit;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_EQUITY)]           = this.m_struct_obj.equity;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN)]           = this.m_struct_obj.margin;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_FREE)]      = this.m_struct_obj.margin_free;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_LEVEL)]     = this.m_struct_obj.margin_level;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_SO_CALL)]   = this.m_struct_obj.margin_so_call;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_SO_SO)]     = this.m_struct_obj.margin_so_so;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_INITIAL)]   = this.m_struct_obj.margin_initial;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_MARGIN_MAINTENANCE)]=this.m_struct_obj.margin_maintenance;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_ASSETS)]           = this.m_struct_obj.assets;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_LIABILITIES)]      = this.m_struct_obj.liabilities;
   this.m_double_prop[this.IndexProp(ACCOUNT_PROP_COMMISSION_BLOCKED)]=this.m_struct_obj.comission_blocked;
//--- Save string properties
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_NAME)]             = ::CharArrayToString(this.m_struct_obj.name);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_SERVER)]           = ::CharArrayToString(this.m_struct_obj.server);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_CURRENCY)]         = ::CharArrayToString(this.m_struct_obj.currency);
   this.m_string_prop[this.IndexProp(ACCOUNT_PROP_COMPANY)]          = ::CharArrayToString(this.m_struct_obj.company);
  }
//+------------------------------------------------------------------+
```

Now add displaying the trade server type **in the method for displaying short descriptions of the account properties:**

```
//+------------------------------------------------------------------+
//| Display a short account description in the journal               |
//+------------------------------------------------------------------+
void CAccount::PrintShort(void)
  {
   string mode=(this.MarginMode()==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING ? ", Hedge" : this.MarginMode()==ACCOUNT_MARGIN_MODE_EXCHANGE ? ", Exhange" : "");
   string names=TextByLanguage("Счёт ","Account ")+(string)this.Login()+": "+this.Name()+" ("+this.Company()+" ";
   string values=::DoubleToString(this.Balance(),(int)this.CurrencyDigits())+" "+this.Currency()+", 1:"+(string)+this.Leverage()+mode+", "+this.TradeModeDescription()+" "+this.ServerTypeDescription()+")";
   ::Print(names,values);
  }
//+------------------------------------------------------------------+
```

**Add the new property description to the method returning the**
**description of the account integer properties:**

```
//+------------------------------------------------------------------+
//| Display a description of an account integer property             |
//+------------------------------------------------------------------+
string CAccount::GetPropertyDescription(ENUM_ACCOUNT_PROP_INTEGER property)
  {
   return
     (
      property==ACCOUNT_PROP_LOGIN           ?  TextByLanguage("Номер счёта","Account number")+": "+(string)this.GetProperty(property)                                                    :
      property==ACCOUNT_PROP_TRADE_MODE      ?  TextByLanguage("Тип торгового счета","Account trade mode")+": "+this.TradeModeDescription()                                               :
      property==ACCOUNT_PROP_LEVERAGE        ?  TextByLanguage("Размер предоставленного плеча","Account leverage")+": "+(string)this.GetProperty(property) :
      property==ACCOUNT_PROP_LIMIT_ORDERS    ?  TextByLanguage("Максимально допустимое количество действующих отложенных ордеров","Maximum allowed number of active pending orders")+": "+
                                                   (string)this.GetProperty(property)                                                                                                     :
      property==ACCOUNT_PROP_MARGIN_SO_MODE  ?  TextByLanguage("Режим задания минимально допустимого уровня залоговых средств","Mode for setting the minimal allowed margin")+": "+
                                                   this.MarginSOModeDescription()                                                          :
      property==ACCOUNT_PROP_TRADE_ALLOWED   ?  TextByLanguage("Разрешенность торговли для текущего счета","Allowed trade for the current account")+": "+
                                                   (this.GetProperty(property) ? TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))                                                 :
      property==ACCOUNT_PROP_TRADE_EXPERT    ?  TextByLanguage("Разрешенность торговли для эксперта","Allowed trade for an Expert Advisor")+": "+
                                                   (this.GetProperty(property) ? TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))                                                 :
      property==ACCOUNT_PROP_MARGIN_MODE     ?  TextByLanguage("Режим расчета маржи","Margin calculation mode")+": "+
                                                   this.MarginModeDescription()                                                         :
      property==ACCOUNT_PROP_CURRENCY_DIGITS ?  TextByLanguage("Количество знаков после запятой для валюты счета","Number of decimal places in account currency")+": "+
                                                   (string)this.GetProperty(property)                                                                                                     :
      property==ACCOUNT_PROP_SERVER_TYPE     ?  TextByLanguage("Тип торгового сервера","Type of trading server")+": "+
                                                   (string)this.GetProperty(property)                                                                                                     :
      ""
     );
  }
//+------------------------------------------------------------------+
```

**Implement the method returning the name of the trade server type:**

```
//+------------------------------------------------------------------+
//| Return the trading server type name                              |
//+------------------------------------------------------------------+
string CAccount::ServerTypeDescription(void) const
  {
   return(this.ServerType()==5 ? "MetaTrader5" : "MetaTrader4");
  }
//+------------------------------------------------------------------+
```

**This concludes the improvement of the CAccount class.**

Now let's introduce the necessary changes to the account collection class since we decided to track events from the **CAccountCollection**
class. All found changes that occurred simultaneously will be written to the int array. To achieve this, use the ready-made class of the int or
uint type variables' dynamic array of the [**CArrayInt**](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayint)
standard library.

**To use it, include the class file to the AccountsCollection.mqh library file:**

```
//+------------------------------------------------------------------+
//|                                           AccountsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright"Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayInt.mqh>
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Accounts\Account.mqh"
//+------------------------------------------------------------------+
//| Account collection                                               |
//+------------------------------------------------------------------+
class CAccountsCollection : public CListObj
  {
```

Let's determine what we want to get. We need to know the moments when some account properties changed. A property may be enabled or disabled —
for example, permission to trade on an account in general or permission to use EAs for trading. We need to know if this property has
changed. In the same way, changing MarginCall or StopOut level may affect the EA efficiency in sustaining a drawdown, etc. Besides, we
also need to track increase and decrease of the funds and the balance to make certain EA-related decisions.

For example, there is a certain number of open positions, and we want to close some of them when a positive or negative amount of funds has
been reached. First, we need to set a certain threshold, upon exceeding which an event is generated. Next, we need to make a certain
decision depending on whether we are facing growth or decline of funds above/below the specified value. The same logic is applicable to
the current profit on an account, balance, free margin and deposit load.

Thus, for some properties, we need to have threshold growth/decline values and the current property value, while for others, we only need
the value that may become enabled/disabled or changed/unchanged.

Add all the necessary class member variables **to the private section**
**of the class** in order to store the growth/decline values of tracked properties and their current values, remove
the

**SavePrevValues() method which turned out to be redundant** and add the methods for initializing
tracked and controlled account data, the
method for checking account changes and returning a change code, the
method setting the event type and filling the event list and finally the
method returning the flag presence in the account event:

```
//--- Save the current data status values of the current account as previous ones
   void              SavePrevValues(void)
```

```
//+------------------------------------------------------------------+
//| Account collection                                               |
//+------------------------------------------------------------------+
class CAccountsCollection : public CListObj
  {
private:
   struct MqlDataAccount
     {
      double         hash_sum;               // Account data hash sum
      //--- Account integer properties
      long           login;                  // ACCOUNT_LOGIN (Account number)
      long           leverage;               // ACCOUNT_LEVERAGE (Leverage)
      int            limit_orders;           // ACCOUNT_LIMIT_ORDERS (Maximum allowed number of active pending orders)
      bool           trade_allowed;          // ACCOUNT_TRADE_ALLOWED (Permission to trade for the current account from the server side)
      bool           trade_expert;           // ACCOUNT_TRADE_EXPERT (Permission to trade for an EA from the server side)
      //--- Account real properties
      double         balance;                // ACCOUNT_BALANCE (Account balance in a deposit currency)
      double         credit;                 // ACCOUNT_CREDIT (Credit in a deposit currency)
      double         profit;                 // ACCOUNT_PROFIT (Current profit on an account in the account currency)
      double         equity;                 // ACCOUNT_EQUITY (Equity on an account in the deposit currency)
      double         margin;                 // ACCOUNT_MARGIN (Reserved margin on an account in a deposit currency)
      double         margin_free;            // ACCOUNT_MARGIN_FREE (Free funds available for opening a position in a deposit currency)
      double         margin_level;           // ACCOUNT_MARGIN_LEVEL (Margin level on an account in %)
      double         margin_so_call;         // ACCOUNT_MARGIN_SO_CALL (Margin Call)
      double         margin_so_so;           // ACCOUNT_MARGIN_SO_SO (Stop Out)
      double         margin_initial;         // ACCOUNT_MARGIN_INITIAL (Funds reserved on an account to ensure a guarantee amount for all pending orders)
      double         margin_maintenance;     // ACCOUNT_MARGIN_MAINTENANCE (Funds reserved on an account to ensure a minimum amount for all open positions)
      double         assets;                 // ACCOUNT_ASSETS (Current assets on an account)
      double         liabilities;            // ACCOUNT_LIABILITIES (Current liabilities on an account)
      double         comission_blocked;      // ACCOUNT_COMMISSION_BLOCKED (Current sum of blocked commissions on an account)
     };
   MqlDataAccount    m_struct_curr_account;              // Account current data
   MqlDataAccount    m_struct_prev_account;              // Account previous data

   MqlTick           m_tick;                             // Tick structure
   string            m_symbol;                           // Current symbol
   long              m_chart_id;                         // Control program chart ID
   CListObj          m_list_accounts;                    // Account object list
   CArrayInt         m_list_changes;                     // Account change list
   string            m_folder_name;                      // Name of the folder storing the account objects
   int               m_index_current;                    // Index of an account object featuring the current account data
//--- Tracking account changes
   bool              m_is_account_event;                 // Event flag in the account data
   int               m_change_code;                      // Account change code
   //--- Leverage
   long              m_changed_leverage_value;           // Leverage change value
   bool              m_is_change_leverage_inc;           // Leverage increase flag
   bool              m_is_change_leverage_dec;           // Leverage decrease flag
   //--- Number of active pending orders
   int               m_changed_limit_orders_value;       // Change value of the maximum allowed number of active pending orders
   bool              m_is_change_limit_orders_inc;       // Increase flag of the maximum allowed number of active pending orders
   bool              m_is_change_limit_orders_dec;       // Decrease flag of the maximum allowed number of active pending orders
   //--- Trading on an account
   bool              m_is_change_trade_allowed_on;       // The flag allowing to trade for the current account from the server side
   bool              m_is_change_trade_allowed_off;      // The flag prohibiting trading for the current account from the server side
   //--- Auto trading on an account
   bool              m_is_change_trade_expert_on;        // The flag allowing to trade for an EA from the server side
   bool              m_is_change_trade_expert_off;       //The flag prohibiting trading for an EA from the server side
   //--- Balance
   double            m_control_balance_inc;              // Tracked balance growth value
   double            m_control_balance_dec;              // Tracked balance decrease value
   double            m_changed_balance_value;            // Balance change value
   bool              m_is_change_balance_inc;            // The flag of the balance change exceeding the growth value
   bool              m_is_change_balance_dec;            // The flag of the balance change exceeding the decrease value
   //--- Credit
   double            m_changed_credit_value;             // Credit change value
   bool              m_is_change_credit_inc;             // Credit increase flag
   bool              m_is_change_credit_dec;             // Credit decrease flag
   //--- Profit
   double            m_control_profit_inc;               // Tracked profit growth value
   double            m_control_profit_dec;               // Tracked profit decrease value
   double            m_changed_profit_value;             // Profit change value
   bool              m_is_change_profit_inc;             // The flag of the profit change exceeding the growth value
   bool              m_is_change_profit_dec;             // The flag of the profit change exceeding the decrease value
   //--- Funds (equity)
   double            m_control_equity_inc;               // Tracked funds growth value
   double            m_control_equity_dec;               // Tracked funds decrease value
   double            m_changed_equity_value;             // Funds change value
   bool              m_is_change_equity_inc;             // The flag of the funds change exceeding the growth value
   bool              m_is_change_equity_dec;             // The flag of the funds change exceeding the decrease value
   //--- Margin
   double            m_control_margin_inc;               // Tracked margin growth value
   double            m_control_margin_dec;               // Tracked margin decrease value
   double            m_changed_margin_value;             // Margin change value
   bool              m_is_change_margin_inc;             // The flag of the margin change exceeding the growth value
   bool              m_is_change_margin_dec;             // The flag of the margin change exceeding the decrease value
   //--- Free margin
   double            m_control_margin_free_inc;          // Tracked free margin growth value
   double            m_control_margin_free_dec;          // Tracked free margin decrease value<
   double            m_changed_margin_free_value;        // Free margin change value
   bool              m_is_change_margin_free_inc;        // The flag of the free margin change exceeding the growth value
   bool              m_is_change_margin_free_dec;        // The flag of the free margin change exceeding the decrease value
   //--- Margin level
   double            m_control_margin_level_inc;         // Tracked margin level growth value
   double            m_control_margin_level_dec;         // Tracked margin level decrease value
   double            m_changed_margin_level_value;       // Margin level change value
   bool              m_is_change_margin_level_inc;       // The flag of the margin level change exceeding the growth value
   bool              m_is_change_margin_level_dec;       // The flag of the margin level change exceeding the decrease value
   //--- Margin Call
   double            m_changed_margin_so_call_value;     // Margin Call level change value
   bool              m_is_change_margin_so_call_inc;     // Margin Call level increase value
   bool              m_is_change_margin_so_call_dec;     // Margin Call level decrease value
   //--- MarginStopOut
   double            m_changed_margin_so_so_value;       // Margin StopOut level change value
   bool              m_is_change_margin_so_so_inc;       // Margin StopOut level increase flag
   bool              m_is_change_margin_so_so_dec;       // Margin StopOut level decrease flag
   //--- Guarantee sum for pending orders
   double            m_control_margin_initial_inc;       // Tracked growth value of the reserved funds for providing the guarantee sum for pending orders
   double            m_control_margin_initial_dec;       // Tracked decrease value of the reserved funds for providing the guarantee sum for pending orders
   double            m_changed_margin_initial_value;     // The change value of the reserved funds for providing the guarantee sum for pending orders
   bool              m_is_change_margin_initial_inc;     // The flag of the reserved funds for providing the guarantee sum for pending orders exceeding the growth value
   bool              m_is_change_margin_initial_dec;     // The flag of the reserved funds for providing the guarantee sum for pending orders exceeding the decrease value
   //--- Guarantee sum for open positions
   double            m_control_margin_maintenance_inc;   // The tracked increase value of the funds reserved on an account to ensure a minimum amount for all open positions
   double            m_control_margin_maintenance_dec;   // The tracked decrease value of the funds reserved on an account to ensure a minimum amount for all open positions
   double            m_changed_margin_maintenance_value; // The change value of the funds reserved on an account to ensure a minimum amount for all open positions
   bool              m_is_change_margin_maintenance_inc; // The flag of the funds reserved on an account to ensure a minimum amount for all open positions exceeding the growth value
   bool              m_is_change_margin_maintenance_dec; // The flag of the funds reserved on an account to ensure a minimum amount for all open positions exceeding the decrease value
   //--- Assets
   double            m_control_assets_inc;               // Tracked assets growth value
   double            m_control_assets_dec;               // Tracked assets decrease value
   double            m_changed_assets_value;             // Assets change value
   bool              m_is_change_assets_inc;             // The flag of the assets change exceeding the growth value
   bool              m_is_change_assets_dec;             // The flag of the assets change exceeding the decrease value
   //--- Liabilities
   double            m_control_liabilities_inc;          // Tracked liabilities growth value
   double            m_control_liabilities_dec;          // Tracked liabilities decrease value
   double            m_changed_liabilities_value;        // Liabilities change values
   bool              m_is_change_liabilities_inc;        // The flag of the liabilities change exceeding the growth value
   bool              m_is_change_liabilities_dec;        // The flag of the liabilities change exceeding the decrease value
   //--- Blocked commissions
   double            m_control_comission_blocked_inc;    // Tracked blocked commissions growth value
   double            m_control_comission_blocked_dec;    // Tracked blocked commissions decrease value
   double            m_changed_comission_blocked_value;  // Blocked commissions changed value
   bool              m_is_change_comission_blocked_inc;  // The flag of the tracked commissions change exceeding the growth value
   bool              m_is_change_comission_blocked_dec;  // The flag of the tracked commissions change exceeding the decrease value

//--- Initialize the variables of (1) tracked, (2) controlled account data
   void              InitChangesParams(void);
   void              InitControlsParams(void);
//--- Check account changes, return a change code
   int               SetChangeCode(void);
//--- Set an event type and fill in the event list
   void              SetTypeEvent(void);
//--- return the flag presence in the account event
   bool              IsPresentEventFlag(constint change_code)   const { return (this.m_change_code & change_code)==change_code;               }
//--- Write the current account data to the account object properties
   void              SetAccountsParams(CAccount* account);

//--- Check the account object presence in the collection list
   bool              IsPresent(CAccount* account);
//--- Find and return the account object index with the current account data
   int               Index(void);
public:
```

The SavePrevValues() method has turned out to be redundant. When defining the account property change, we need to write the new property
value to the structure storing the data on the changed property's previous status right away. The remaining properties should be left
the same they were during the previous check before their actual change. The SavePrevValues() method saves all account properties as
previous ones thus violating tracking a certain change value set by default for each of the tracked properties that can be set
separately using specific methods of setting a tracked value.

**Let's write the method of initializing tracked data outside the class body:**

```
//+------------------------------------------------------------------+
//| Initialize the variables of tracked account data                 |
//+------------------------------------------------------------------+
void CAccountsCollection::InitChangesParams(void)
  {
//--- List and code of changes
   this.m_list_changes.Clear();                    // Clear the change list
   this.m_list_changes.Sort();                     // Sort the change list
//--- Leverage
   this.m_changed_leverage_value=0;                // Leverage change value
   this.m_is_change_leverage_inc=false;            // Leverage increase flag
   this.m_is_change_leverage_dec=false;            // Leverage decrease flag
//--- Number of active pending orders
   this.m_changed_limit_orders_value=0;            // Change value of the maximum allowed number of active pending orders
   this.m_is_change_limit_orders_inc=false;        // Increase flag of the maximum allowed number of active pending orders
   this.m_is_change_limit_orders_dec=false;        // Decrease flag of the maximum allowed number of active pending orders
//--- Trading on an account
   this.m_is_change_trade_allowed_on=false;        // The flag allowing to trade for the current account from the server side
   this.m_is_change_trade_allowed_off=false;       // The flag prohibiting trading for the current account from the server side
//--- Auto trading on an account
   this.m_is_change_trade_expert_on=false;         // The flag allowing to trade for an EA from the server side
   this.m_is_change_trade_expert_off=false;        // The flag prohibiting trading for an EA from the server side
//--- Balance
   this.m_changed_balance_value=0;                 // Balance change value
   this.m_is_change_balance_inc=false;             // The flag of the balance change exceeding the growth value
   this.m_is_change_balance_dec=false;             // The flag of the balance change exceeding the decrease value
//--- Credit
   this.m_changed_credit_value=0;                  // Credit change value
   this.m_is_change_credit_inc=false;              // Credit increase flag
   this.m_is_change_credit_dec=false;              // Credit decrease flag
//--- Profit
   this.m_changed_profit_value=0;                  // Profit change value
   this.m_is_change_profit_inc=false;              // The flag of the profit change exceeding the growth value
   this.m_is_change_profit_dec=false;              // The flag of the profit change exceeding the decrease value
//--- Funds
   this.m_changed_equity_value=0;                  // Funds change value
   this.m_is_change_equity_inc=false;              // The flag of the funds change exceeding the growth value
   this.m_is_change_equity_dec=false;              // The flag of the funds change exceeding the decrease value
//--- Margin
   this.m_changed_margin_value=0;                  // Margin change value
   this.m_is_change_margin_inc=false;              // The flag of the margin change exceeding the growth value
   this.m_is_change_margin_dec=false;              // The flag of the margin change exceeding the decrease value
//--- Free margin
   this.m_changed_margin_free_value=0;             // Free margin change value
   this.m_is_change_margin_free_inc=false;         // The flag of the free margin change exceeding the growth value
   this.m_is_change_margin_free_dec=false;         // The flag of the free margin change exceeding the decrease value
//--- Margin level
   this.m_changed_margin_level_value=0;            // Margin level change value
   this.m_is_change_margin_level_inc=false;        // The flag of the margin level change exceeding the growth value
   this.m_is_change_margin_level_dec=false;        // The flag of the margin level change exceeding the decrease value
//--- Margin Call level
   this.m_changed_margin_so_call_value=0;          // Margin Call level change value
   this.m_is_change_margin_so_call_inc=false;      // Margin Call level increase flag
   this.m_is_change_margin_so_call_dec=false;      // Margin Call level decrease flag
//--- Margin StopOut level
   this.m_changed_margin_so_so_value=0;            // Margin StopOut level change value
   this.m_is_change_margin_so_so_inc=false;        // Margin StopOut level increase flag
   this.m_is_change_margin_so_so_dec=false;        // Margin StopOut level decrease flag
//--- Guarantee sum for pending orders
   this.m_changed_margin_initial_value=0;          // The change value of the reserved funds for providing the guarantee sum for pending orders
   this.m_is_change_margin_initial_inc=false;      // The flag of the reserved funds for providing the guarantee sum for pending orders exceeding the growth value
   this.m_is_change_margin_initial_dec=false;      // The flag of the reserved funds for providing the guarantee sum for pending orders exceeding the decrease value
//--- Guarantee sum for open positions
   this.m_changed_margin_maintenance_value=0;      // The change value of the funds reserved on an account to ensure a minimum amount for all open positions
   this.m_is_change_margin_maintenance_inc=false;  // The flag of the funds reserved on an account to ensure a minimum amount for all open positions exceeding the growth value
   this.m_is_change_margin_maintenance_dec=false;  // The flag of the funds reserved on an account to ensure a minimum amount for all open positions exceeding the decrease value
//--- Assets
   this.m_changed_assets_value=0;                  // Assets change value
   this.m_is_change_assets_inc=false;              // The flag of the assets change exceeding the growth value
   this.m_is_change_assets_dec=false;              // The flag of the assets change exceeding the decrease value
//--- Liabilities
   this.m_changed_liabilities_value=0;             // Liabilities change value
   this.m_is_change_liabilities_inc=false;         // The flag of the liabilities change exceeding the growth value
   this.m_is_change_liabilities_dec=false;         // The flag of the liabilities change exceeding the decrease value
//--- Blocked commissions
   this.m_changed_comission_blocked_value=0;       // Blocked commissions change value
   this.m_is_change_comission_blocked_inc=false;   // The flag of the tracked commissions change exceeding the growth value
   this.m_is_change_comission_blocked_dec=false;   // The flag of the tracked commissions change exceeding the decrease value
  }
//+------------------------------------------------------------------+
```

**The method of initializing controlled data:**

```
//+------------------------------------------------------------------+
//| Initialize the variables of controlled account data              |
//+------------------------------------------------------------------+
void CAccountsCollection::InitControlsParams(void)
  {
//--- Balance
   this.m_control_balance_inc=50;                  // Tracked balance growth value
   this.m_control_balance_dec=50;                  // Tracked balance decrease value
//--- Profit
   this.m_control_profit_inc=20;                   // Tracked profit growth value
   this.m_control_profit_dec=20;                   // Tracked profit decrease value
//--- Funds (equity)
   this.m_control_equity_inc=15;                   // Tracked funds growth value
   this.m_control_equity_dec=15;                   // Tracked funds decrease value
//--- Margin
   this.m_control_margin_inc=1000;                 // Tracked margin growth value
   this.m_control_margin_dec=1000;                 // Tracked margin decrease value
//--- Free margin
   this.m_control_margin_free_inc=1000;            // Tracked free margin growth value
   this.m_control_margin_free_dec=1000;            // Tracked free margin decrease value
//--- Margin level
   this.m_control_margin_level_inc=10000;          // Tracked margin level growth value
   this.m_control_margin_level_dec=500;            // Tracked margin level decrease value
//--- Guarantee sum for pending orders
   this.m_control_margin_initial_inc=1000;         // Tracked growth value of the reserved funds for providing the guarantee sum for pending orders
   this.m_control_margin_initial_dec=1000;         // Tracked decrease value of the reserved funds for providing the guarantee sum for pending orders
//--- Guarantee sum for open positions
   this.m_control_margin_maintenance_inc=1000;     // The tracked increase value of the funds reserved on an account to ensure a minimum amount for all open positions
   this.m_control_margin_maintenance_dec=1000;     // The tracked decrease value of the funds reserved on an account to ensure a minimum amount for all open positions
//--- Assets
   this.m_control_assets_inc=1000;                 // Tracked assets growth value
   this.m_control_assets_dec=1000;                 // Tracked assets decrease value
//--- Liabilities
   this.m_control_liabilities_inc=1000;            // Tracked liabilities growth value
   this.m_control_liabilities_dec=1000;            // Tracked liabilities decrease value
//--- Blocked commissions
   this.m_control_comission_blocked_inc=1000;      // Tracked blocked commissions growth value
   this.m_control_comission_blocked_dec=1000;      // Tracked blocked commissions decrease value
  }
//+------------------------------------------------------------------+
```

In this method, we simply initialize the variables by default values. If the properties exceed the values set in the variables, an
appropriate account event is generated. These properties can also be set by directly calling the appropriate methods of setting
controlled account properties.

**The method that checks the account properties change and fills in the event code with the appropriate flags:**

```
//+------------------------------------------------------------------+
//| Check account changes, return the change code                    |
//+------------------------------------------------------------------+
int CAccountsCollection::SetChangeCode(void)
  {
   this.m_change_code=ACCOUNT_EVENT_FLAG_NO_EVENT;

   if(this.m_struct_curr_account.trade_allowed!=this.m_struct_prev_account.trade_allowed)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_TRADE_ALLOWED;
   if(this.m_struct_curr_account.trade_expert!=this.m_struct_prev_account.trade_expert)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_TRADE_EXPERT;
   if(this.m_struct_curr_account.leverage-this.m_struct_prev_account.leverage!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_LEVERAGE;
   if(this.m_struct_curr_account.limit_orders-this.m_struct_prev_account.limit_orders!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_LIMIT_ORDERS;
   if(this.m_struct_curr_account.balance-this.m_struct_prev_account.balance!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_BALANCE;
   if(this.m_struct_curr_account.credit-this.m_struct_prev_account.credit!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_CREDIT;
   if(this.m_struct_curr_account.profit-this.m_struct_prev_account.profit!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_PROFIT;
   if(this.m_struct_curr_account.equity-this.m_struct_prev_account.equity!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_EQUITY;
   if(this.m_struct_curr_account.margin-this.m_struct_prev_account.margin!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_MARGIN;
   if(this.m_struct_curr_account.margin_free-this.m_struct_prev_account.margin_free!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_MARGIN_FREE;
   if(this.m_struct_curr_account.margin_level-this.m_struct_prev_account.margin_level!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_MARGIN_LEVEL;
   if(this.m_struct_curr_account.margin_so_call-this.m_struct_prev_account.margin_so_call!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_MARGIN_SO_CALL;
   if(this.m_struct_curr_account.margin_so_so-this.m_struct_prev_account.margin_so_so!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_MARGIN_SO_SO;
   if(this.m_struct_curr_account.margin_initial-this.m_struct_prev_account.margin_initial!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_MARGIN_INITIAL;
   if(this.m_struct_curr_account.margin_maintenance-this.m_struct_prev_account.margin_maintenance!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_MARGIN_MAINTENANCE;
   if(this.m_struct_curr_account.assets-this.m_struct_prev_account.assets!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_ASSETS;
   if(this.m_struct_curr_account.liabilities-this.m_struct_prev_account.liabilities!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_LIABILITIES;
   if(this.m_struct_curr_account.comission_blocked-this.m_struct_prev_account.comission_blocked!=0)
      this.m_change_code+=ACCOUNT_EVENT_FLAG_COMISSION_BLOCKED;
//---
   returnthis.m_change_code;
  }
//+------------------------------------------------------------------+
```

First, an event code is reset in the method. Next, the values of
the controlled account parameters are compared in the

current data structure and the previous
data structure. If the data is not similar, the
appropriate flag is added to the event code.

**The method setting an event type and adding the event to the list of account changes:**

```
//+------------------------------------------------------------------+
//| Set the account object event type                                |
//+------------------------------------------------------------------+
void CAccountsCollection::SetTypeEvent(void)
  {
   this.InitChangesParams();
   ENUM_ACCOUNT_EVENT event=ACCOUNT_EVENT_NO_EVENT;
//--- Changing permission to trade for the account
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_TRADE_ALLOWED))
     {
      if(!this.m_struct_curr_account.trade_allowed)
        {
         this.m_is_change_trade_allowed_off=true;
         event=ACCOUNT_EVENT_TRADE_ALLOWED_OFF;
         this.m_struct_prev_account.trade_allowed=this.m_struct_curr_account.trade_allowed;
        }
      else
        {
         this.m_is_change_trade_allowed_on=true;
         event=ACCOUNT_EVENT_TRADE_ALLOWED_ON;
         this.m_struct_prev_account.trade_allowed=this.m_struct_curr_account.trade_allowed;
        }
      if(this.m_list_changes.Search(event)==WRONG_VALUE)
         this.m_list_changes.Add(event);
     }
//--- Changing permission for auto trading for the account
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_TRADE_EXPERT))
     {
      if(!this.m_struct_curr_account.trade_expert)
        {
         this.m_is_change_trade_expert_off=true;
         event=ACCOUNT_EVENT_TRADE_EXPERT_OFF;
         this.m_struct_prev_account.trade_expert=false;
        }
      else
        {
         this.m_is_change_trade_expert_on=true;
         event=ACCOUNT_EVENT_TRADE_EXPERT_ON;
         this.m_struct_prev_account.trade_expert=true;
        }
      if(this.m_list_changes.Search(event)==WRONG_VALUE)
         this.m_list_changes.Add(event);
     }
//--- Change the leverage
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_LEVERAGE))
     {
      this.m_changed_leverage_value=this.m_struct_curr_account.leverage-this.m_struct_prev_account.leverage;
      if(this.m_changed_leverage_value>0)
        {
         this.m_is_change_leverage_inc=true;
         event=ACCOUNT_EVENT_LEVERAGE_INC;
        }
      else
        {
         this.m_is_change_leverage_dec=true;
         event=ACCOUNT_EVENT_LEVERAGE_DEC;
        }
      if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
         this.m_struct_prev_account.leverage=this.m_struct_curr_account.leverage;
     }
//--- Changing the maximum allowed number of active pending orders
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_LIMIT_ORDERS))
     {
      this.m_changed_limit_orders_value=this.m_struct_curr_account.limit_orders-this.m_struct_prev_account.limit_orders;
      if(this.m_changed_limit_orders_value>0)
        {
         this.m_is_change_limit_orders_inc=true;
         event=ACCOUNT_EVENT_LIMIT_ORDERS_INC;
        }
      else
        {
         this.m_is_change_limit_orders_dec=true;
         event=ACCOUNT_EVENT_LIMIT_ORDERS_DEC;
        }
      if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
         this.m_struct_prev_account.limit_orders=this.m_struct_curr_account.limit_orders;
     }
//--- Changing the credit in a deposit currency
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_CREDIT))
     {
      this.m_changed_credit_value=this.m_struct_curr_account.credit-this.m_struct_prev_account.credit;
      if(this.m_changed_credit_value>0)
        {
         this.m_is_change_credit_inc=true;
         event=ACCOUNT_EVENT_CREDIT_INC;
        }
      else
        {
         this.m_is_change_credit_dec=true;
         event=ACCOUNT_EVENT_CREDIT_DEC;
        }
      if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
         this.m_struct_prev_account.credit=this.m_struct_curr_account.credit;
     }
//--- Changing the Margin Call level
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_MARGIN_SO_CALL))
     {
      this.m_changed_margin_so_call_value=this.m_struct_curr_account.margin_so_call-this.m_struct_prev_account.margin_so_call;
      if(this.m_changed_margin_so_call_value>0)
        {
         this.m_is_change_margin_so_call_inc=true;
         event=ACCOUNT_EVENT_MARGIN_SO_CALL_INC;
        }
      else
        {
         this.m_is_change_margin_so_call_dec=true;
         event=ACCOUNT_EVENT_MARGIN_SO_CALL_DEC;
        }
      if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
         this.m_struct_prev_account.margin_so_call=this.m_struct_curr_account.margin_so_call;
     }
//--- Changing the Stop Out level
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_MARGIN_SO_SO))
     {
      this.m_changed_margin_so_so_value=this.m_struct_curr_account.margin_so_so-this.m_struct_prev_account.margin_so_so;
      if(this.m_changed_margin_so_so_value>0)
        {
         this.m_is_change_margin_so_so_inc=true;
         event=ACCOUNT_EVENT_MARGIN_SO_SO_INC;
        }
      else
        {
         this.m_is_change_margin_so_so_dec=true;
         event=ACCOUNT_EVENT_MARGIN_SO_SO_DEC;
        }
      if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
         this.m_struct_prev_account.margin_so_so=this.m_struct_curr_account.margin_so_so;
     }
//--- The balance exceeds the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_BALANCE))
     {
      this.m_changed_balance_value=this.m_struct_curr_account.balance-this.m_struct_prev_account.balance;
      if(this.m_changed_balance_value>this.m_control_balance_inc)
        {
         this.m_is_change_balance_inc=true;
         event=ACCOUNT_EVENT_BALANCE_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.balance=this.m_struct_curr_account.balance;
        }
      elseif(this.m_changed_balance_value<-this.m_control_balance_dec)
        {
         this.m_is_change_balance_dec=true;
         event=ACCOUNT_EVENT_BALANCE_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.balance=this.m_struct_curr_account.balance;
        }
     }
//--- The profit exceeds the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_PROFIT))
     {
      this.m_changed_profit_value=this.m_struct_curr_account.profit-this.m_struct_prev_account.profit;
      if(this.m_changed_profit_value>this.m_control_profit_inc)
        {
         this.m_is_change_profit_inc=true;
         event=ACCOUNT_EVENT_PROFIT_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.profit=this.m_struct_curr_account.profit;
        }
      elseif(this.m_changed_profit_value<-this.m_control_profit_dec)
        {
         this.m_is_change_profit_dec=true;
         event=ACCOUNT_EVENT_PROFIT_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.profit=this.m_struct_curr_account.profit;
        }
     }
//--- The equity exceeds the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_EQUITY))
     {
      this.m_changed_equity_value=this.m_struct_curr_account.equity-this.m_struct_prev_account.equity;
      if(this.m_changed_equity_value>this.m_control_equity_inc)
        {
         this.m_is_change_equity_inc=true;
         event=ACCOUNT_EVENT_EQUITY_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.equity=this.m_struct_curr_account.equity;
        }
      elseif(this.m_changed_equity_value<-this.m_control_equity_dec)
        {
         this.m_is_change_equity_dec=true;
         event=ACCOUNT_EVENT_EQUITY_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.equity=this.m_struct_curr_account.equity;
        }
     }
//--- The reserved margin on an account in the deposit currency change exceeds the specified value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_MARGIN))
     {
      this.m_changed_margin_value=this.m_struct_curr_account.margin-this.m_struct_prev_account.margin;
      if(this.m_changed_margin_value>this.m_control_margin_inc)
        {
         this.m_is_change_margin_inc=true;
         event=ACCOUNT_EVENT_MARGIN_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin=this.m_struct_curr_account.margin;
        }
      elseif(this.m_changed_margin_value<-this.m_control_margin_dec)
        {
         this.m_is_change_margin_dec=true;
         event=ACCOUNT_EVENT_MARGIN_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin=this.m_struct_curr_account.margin;
        }
     }
//--- The free funds available for opening a position in a deposit currency exceed the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_MARGIN_FREE))
     {
      this.m_changed_margin_free_value=this.m_struct_curr_account.margin_free-this.m_struct_prev_account.margin_free;
      if(this.m_changed_margin_free_value>this.m_control_margin_free_inc)
        {
         this.m_is_change_margin_free_inc=true;
         event=ACCOUNT_EVENT_MARGIN_FREE_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin_free=this.m_struct_curr_account.margin_free;
        }
      elseif(this.m_changed_margin_free_value<-this.m_control_margin_free_dec)
        {
         this.m_is_change_margin_free_dec=true;
         event=ACCOUNT_EVENT_MARGIN_FREE_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin_free=this.m_struct_curr_account.margin_free;
        }
     }
//--- The margin level on an account in % exceeds the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_MARGIN_LEVEL))
     {
      this.m_changed_margin_level_value=this.m_struct_curr_account.margin_level-this.m_struct_prev_account.margin_level;
      if(this.m_changed_margin_level_value>this.m_control_margin_level_inc)
        {
         this.m_is_change_margin_level_inc=true;
         event=ACCOUNT_EVENT_MARGIN_LEVEL_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin_level=this.m_struct_curr_account.margin_level;
        }
      elseif(this.m_changed_margin_level_value<-this.m_control_margin_level_dec)
        {
         this.m_is_change_margin_level_dec=true;
         event=ACCOUNT_EVENT_MARGIN_LEVEL_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin_level=this.m_struct_curr_account.margin_level;
        }
     }
//--- The funds reserved on an account to ensure a guarantee amount for all pending orders exceed the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_MARGIN_INITIAL))
     {
      this.m_changed_margin_initial_value=this.m_struct_curr_account.margin_initial-this.m_struct_prev_account.margin_initial;
      if(this.m_changed_margin_initial_value>this.m_control_margin_initial_inc)
        {
         this.m_is_change_margin_initial_inc=true;
         event=ACCOUNT_EVENT_MARGIN_INITIAL_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin_initial=this.m_struct_curr_account.margin_initial;
        }
      elseif(this.m_changed_margin_initial_value<-this.m_control_margin_initial_dec)
        {
         this.m_is_change_margin_initial_dec=true;
         event=ACCOUNT_EVENT_MARGIN_INITIAL_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin_initial=this.m_struct_curr_account.margin_initial;
        }
     }
//--- The funds reserved on an account to ensure a minimum amount for all open positions exceed the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_MARGIN_MAINTENANCE))
     {
      this.m_changed_margin_maintenance_value=this.m_struct_curr_account.margin_maintenance-this.m_struct_prev_account.margin_maintenance;
      if(this.m_changed_margin_maintenance_value>this.m_control_margin_maintenance_inc)
        {
         this.m_is_change_margin_maintenance_inc=true;
         event=ACCOUNT_EVENT_MARGIN_MAINTENANCE_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin_maintenance=this.m_struct_curr_account.margin_maintenance;
        }
      elseif(this.m_changed_margin_maintenance_value<-this.m_control_margin_maintenance_dec)
        {
         this.m_is_change_margin_maintenance_dec=true;
         event=ACCOUNT_EVENT_MARGIN_MAINTENANCE_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.margin_maintenance=this.m_struct_curr_account.margin_maintenance;
        }
     }
//--- The current assets on an account exceed the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_ASSETS))
     {
      this.m_changed_assets_value=this.m_struct_curr_account.assets-this.m_struct_prev_account.assets;
      if(this.m_changed_assets_value>this.m_control_assets_inc)
        {
         this.m_is_change_assets_inc=true;
         event=ACCOUNT_EVENT_ASSETS_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.assets=this.m_struct_curr_account.assets;
        }
      elseif(this.m_changed_assets_value<-this.m_control_assets_dec)
        {
         this.m_is_change_assets_dec=true;
         event=ACCOUNT_EVENT_ASSETS_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.assets=this.m_struct_curr_account.assets;
        }
     }
//--- The current liabilities on an account exceed the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_LIABILITIES))
     {
      this.m_changed_liabilities_value=this.m_struct_curr_account.liabilities-this.m_struct_prev_account.liabilities;
      if(this.m_changed_liabilities_value>this.m_control_liabilities_inc)
        {
         this.m_is_change_liabilities_inc=true;
         event=ACCOUNT_EVENT_LIABILITIES_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.liabilities=this.m_struct_curr_account.liabilities;
        }
      elseif(this.m_changed_liabilities_value<-this.m_control_liabilities_dec)
        {
         this.m_is_change_liabilities_dec=true;
         event=ACCOUNT_EVENT_LIABILITIES_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.liabilities=this.m_struct_curr_account.liabilities;
        }
     }
//--- The current sum of blocked commissions on an account exceeds the specified change value +/-
   if(this.IsPresentEventFlag(ACCOUNT_EVENT_FLAG_COMISSION_BLOCKED))
     {
      this.m_changed_comission_blocked_value=this.m_struct_curr_account.comission_blocked-this.m_struct_prev_account.comission_blocked;
      if(this.m_changed_comission_blocked_value>this.m_control_comission_blocked_inc)
        {
         this.m_is_change_comission_blocked_inc=true;
         event=ACCOUNT_EVENT_COMISSION_BLOCKED_INC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.comission_blocked=this.m_struct_curr_account.comission_blocked;
        }
      elseif(this.m_changed_comission_blocked_value<-this.m_control_comission_blocked_dec)
        {
         this.m_is_change_comission_blocked_dec=true;
         event=ACCOUNT_EVENT_COMISSION_BLOCKED_DEC;
         if(this.m_list_changes.Search(event)==WRONG_VALUE && this.m_list_changes.Add(event))
            this.m_struct_prev_account.comission_blocked=this.m_struct_curr_account.comission_blocked;
        }
     }
  }
//+------------------------------------------------------------------+
```

The method features two types of event definition logic:

1. simple tracking of a property permission/change,

2. tracking a change exceeding a specified value in the direction of its increase/decrease.

Since the method is quite bulky, we will use the two types of the account event definition as an example:

First, all change flags and data are reset and the event type is set to zero.

**Next, for the first logic type (using permission to trade on an account):**

- check the flag of the permission to trade on an account in the event code
- if trading is
currently prohibited ,
permission has just been disabled

  - set the flag prohibiting trading on the account
  - set
     the "trading on the account disabled" event
  - save
     the current status of the account property in the previous data structure for subsequent check

- otherwise, if trading is currently allowed

  - set the flag enabling trading on the account
  - set the "trading on the account enabled" event
  - save the current status of the account property in the previous data structure for subsequent check

- if there is no such an event in the change list
- add an event to the list

**For the second logic type (using changing the sum of blocked commissions as an example):**

- check the flag of changing the sum of blocked commissions
- calculate the change of the sum of blocked commissions
- if the change value exceeds the controlled growth value

  - set the flag of the sum of blocked commissions growth
  - set the "sum of blocked commissions increase exceeds the specified
     value" event
  - if there is no such an event in the change list, and the event is
     successfully added to the list
    - save the current status of the account property in the previous
       data structure for subsequent check

- otherwise, if the change value exceeds the controlled decrease value
  - set the flag of the sum of blocked commissions decrease
  - set the "sum of blocked commissions decrease exceeds the specified value" event
  - if there is no such an event in the change list, and the event is successfully added to the list
    - save the current status of the account property in the previous data structure for subsequent check

**In the public section of the class, add the methods** returning the account
event code, the account event list, the
account event by its index in the list; the methods setting
and

returning a symbol, the
method returning the controlling program chart ID and the method
returning the account event description. Also, add the methods for
receiving and setting the parameters of tracked changes:

```
public:
//--- Return the full account collection list "as is"
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_accounts;                                         }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
//--- Return the (1) current account object index, (2) the flag of the event occurred in the account data, (3) event type
   int               IndexCurrentAccount(void)                                                        const { return this.m_index_current;                                           }
   bool              IsAccountEvent(void)                                                             const { return this.m_is_account_event;                                        }
//--- Return the (1) object account event code, (2) event list, (3) account event by its number in the list
   int               GetEventCode(void)                                                               const { return this.m_change_code;                                             }
   CArrayInt        *GetListChanges(void)                                                                   { return &this.m_list_changes;                          }
   ENUM_ACCOUNT_EVENT GetEvent(constint shift=WRONG_VALUE);
//--- (1) Set and (2) return the current symbol
   void              SetSymbol(conststring symbol)                                                          { this.m_symbol=symbol;                                 }
   string            GetSymbol(void)                                                                  const { return this.m_symbol;                                 }
//--- Set the control program chart ID
   void              SetChartID(constlong id)                                                               { this.m_chart_id=id;                                   }

//--- Constructor, destructor
                     CAccountsCollection();
                    ~CAccountsCollection();
//--- Add the account object to the list
   bool              AddToList(CAccount* account);
//--- (1) Save account objects from the list to the files
//--- (2) Save account objects from the files to the list
   bool              SaveObjects(void);
   bool              LoadObjects(void);
//--- Return the account event description
   string            EventDescription(const ENUM_ACCOUNT_EVENT event);
//--- Update the current account data
   void              Refresh(void);

//--- Get and set the parameters of tracked changes
   //--- Leverage:
   //--- (1) Leverage change value, (2) Leverage increase flag, (3) Leverage decrease flag
   long              GetValueChangedLeverage(void)                                                    const { return this.m_changed_leverage_value;                  }
   bool              IsIncreaseLeverage(void)                                                         const { return this.m_is_change_leverage_inc;                  }
   bool              IsDecreaseLeverage(void)                                                         const { return this.m_is_change_leverage_dec;                  }
   //--- Number of active pending orders:
   //--- (1) Change value, (2) Increase flag, (3) Decrease flag
   int               GetValueChangedLimitOrders(void)                                                 const { return this.m_changed_limit_orders_value;              }
   bool              IsIncreaseLimitOrders(void)                                                      const { return this.m_is_change_limit_orders_inc;              }
   bool              IsDecreaseLimitOrders(void)                                                      const { return this.m_is_change_limit_orders_dec;              }
   //--- Trading on an account:
   //--- (1) The flag allowing to trade for the current account, (2) The flag prohibiting trading for the current account from the server side
   bool              IsOnTradeAllowed(void)                                                           const { return this.m_is_change_trade_allowed_on;              }
   bool              IsOffTradeAllowed(void)                                                          const { return this.m_is_change_trade_allowed_off;             }
   //--- Auto trading on an account:
   //--- (1) The flag allowing to trade for an EA, (2) The flag prohibiting trading for an EA from the server side
   bool              IsOnTradeExpert(void)                                                            const { return this.m_is_change_trade_expert_on;               }
   bool              IsOffTradeExpert(void)                                                           const { return this.m_is_change_trade_expert_off;              }
   //--- Balance:
   //--- setting the tracked value of the balance (1) growth, (2) decrease
   //--- getting (3) the balance change value,
   //--- getting the flag of the balance change exceeding the (4) growth value, (5) decrease value
   void              SetControlBalanceInc(const double value)                                               { this.m_control_balance_inc=::fabs(value);              }
   void              SetControlBalanceDec(const double value)                                               { this.m_control_balance_dec=::fabs(value);              }
   double            GetValueChangedBalance(void)                                                     const { return this.m_changed_balance_value;                   }
   bool              IsIncreaseBalance(void)                                                          const { return this.m_is_change_balance_inc;                   }
   bool              IsDecreaseBalance(void)                                                          const { return this.m_is_change_balance_dec;                   }
   //--- Credit:
   //--- getting (1) the credit change value, (2) credit increase flag, (3) decrease flag
   double            GetValueChangedCredit(void)                                                      const { return this.m_changed_credit_value;                    }
   bool              IsIncreaseCredit(void)                                                           const { return this.m_is_change_credit_inc;                    }
   bool              IsDecreaseCredit(void)                                                           const { return this.m_is_change_credit_dec;                    }
   //--- Profit:
   //--- setting the tracked profit (1) growth, (2) decrease value
   //--- getting the (3) profit change value,
   //--- getting the flag of the profit change exceeding the (4) growth, (5) decrease value
   void              SetControlProfitInc(const double value)                                                { this.m_control_profit_inc=::fabs(value);               }
   void              SetControlProfitDec(const double value)                                                { this.m_control_profit_dec=::fabs(value);               }
   double            GetValueChangedProfit(void)                                                      const { return this.m_changed_profit_value;                    }
   bool              IsIncreaseProfit(void)                                                           const { return this.m_is_change_profit_inc;                    }
   bool              IsDecreaseProfit(void)                                                           const { return this.m_is_change_profit_dec;                    }
   //--- Equity:
   //--- setting the tracked equity (1) growth, (2) decrease value
   //--- getting the (3) equity change value,
   //--- getting the flag of the equity change exceeding the (4) growth, (5) decrease value
   void              SetControlEquityInc(const double value)                                                { this.m_control_equity_inc=::fabs(value);               }
   void              SetControlEquityDec(const double value)                                                { this.m_control_equity_dec=::fabs(value);               }
   double            GetValueChangedEquity(void)                                                      const { return this.m_changed_equity_value;                    }
   bool              IsIncreaseEquity(void)                                                           const { return this.m_is_change_equity_inc;                    }
   bool              IsDecreaseEquity(void)                                                           const { return this.m_is_change_equity_dec;                    }
   //--- Margin:
   //--- setting the tracked margin (1) growth, (2) decrease value
   //--- getting the (3) margin change value,
   //--- getting the flag of the margin change exceeding the (4) growth, (5) decrease value
   void              SetControlMarginInc(const double value)                                                { this.m_control_margin_inc=::fabs(value);               }
   void              SetControlMarginDec(const double value)                                                { this.m_control_margin_dec=::fabs(value);               }
   double            GetValueChangedMargin(void)                                                      const { return this.m_changed_margin_value;                    }
   bool              IsIncreaseMargin(void)                                                           const { return this.m_is_change_margin_inc;                    }
   bool              IsDecreaseMargin(void)                                                           const { return this.m_is_change_margin_dec;                    }
   //--- Free margin:
   //--- setting the tracked free margin (1) growth, (2) decrease value
   //--- getting the (3) free margin change value,
   //--- getting the flag of the free margin change exceeding the (4) growth, (5) decrease value
   void              SetControlMarginFreeInc(const double value)                                            { this.m_control_margin_free_inc=::fabs(value);          }
   void              SetControlMarginFreeDec(const double value)                                            { this.m_control_margin_free_dec=::fabs(value);          }
   double            GetValueChangedMarginFree(void)                                                  const { return this.m_changed_margin_free_value;               }
   bool              IsIncreaseMarginFree(void)                                                       const { return this.m_is_change_margin_free_inc;               }
   bool              IsDecreaseMarginFree(void)                                                       const { return this.m_is_change_margin_free_dec;               }
   //--- Margin level:
   //--- setting the tracked margin level (1) growth, (2) decrease value
   //--- getting the (3) margin level change value,
   //--- getting the flag of the margin level change exceeding the (4) growth, (5) decrease value
   void              SetControlMarginLevelInc(const double value)                                           { this.m_control_margin_level_inc=::fabs(value);         }
   void              SetControlMarginLevelDec(const double value)                                           { this.m_control_margin_level_dec=::fabs(value);         }
   double            GetValueChangedMarginLevel(void)                                                 const { return this.m_changed_margin_level_value;              }
   bool              IsIncreaseMarginLevel(void)                                                      const { return this.m_is_change_margin_level_inc;              }
   bool              IsDecreaseMarginLevel(void)                                                      const { return this.m_is_change_margin_level_dec;              }
   //--- Margin Call:
   //--- getting (1) Margin Call change value, (2) increase flag, (3) decrease flag
   double            GetValueChangedMarginCall(void)                                                  const { return this.m_changed_margin_so_call_value;            }
   bool              IsIncreaseMarginCall(void)                                                       const { return this.m_is_change_margin_so_call_inc;            }
   bool              IsDecreaseMarginCall(void)                                                       const { return this.m_is_change_margin_so_call_dec;            }
   //--- Margin StopOut:
   //--- getting (1) Margin StopOut change value, (2) increase flag, (3) decrease flag
   double            GetValueChangedMarginStopOut(void)                                               const { return this.m_changed_margin_so_so_value;              }
   bool              IsIncreaseMarginStopOut(void)                                                    const { return this.m_is_change_margin_so_so_inc;              }
   bool              IsDecreasMarginStopOute(void)                                                    const { return this.m_is_change_margin_so_so_dec;              }
   //--- Guarantee sum for pending orders:
   //--- setting the tracked value of the reserved funds for providing the guarantee sum for pending orders (1) growth, (2) decrease
   //--- getting the change value of the (3) reserved funds for providing the guarantee sum,
   //--- getting the flag of the reserved funds for providing the guarantee sum for pending orders exceeding the (4) growth, (5) decrease value
   void              SetControlMarginInitialInc(const double value)                                         { this.m_control_margin_initial_inc=::fabs(value);       }
   void              SetControlMarginInitialDec(const double value)                                         { this.m_control_margin_initial_dec=::fabs(value);       }
   double            GetValueChangedMarginInitial(void)                                               const { return this.m_changed_margin_initial_value;            }
   bool              IsIncreaseMarginInitial(void)                                                    const { return this.m_is_change_margin_initial_inc;            }
   bool              IsDecreaseMarginInitial(void)                                                    const { return this.m_is_change_margin_initial_dec;            }
   //--- Guarantee sum for open positions:
   //--- setting the tracked value of the funds reserved on an account to ensure a minimum amount for all open positions (1) growth, (2) decrease
   //--- getting the change value of the (3) funds reserved on an account to ensure a minimum amount for all open positions,
   //--- getting the flag of the funds reserved on an account to ensure a minimum amount for all open positions exceeding the (4) growth, (5) decrease value
   void              SetControlMarginMaintenanceInc(const double value)                                     { this.m_control_margin_maintenance_inc=::fabs(value);   }
   void              SetControlMarginMaintenanceDec(const double value)                                     { this.m_control_margin_maintenance_dec=::fabs(value);   }
   double            GetValueChangedMarginMaintenance(void)                                           const { return this.m_changed_margin_maintenance_value;        }
   bool              IsIncreaseMarginMaintenance(void)                                                const { return this.m_is_change_margin_maintenance_inc;        }
   bool              IsDecreaseMarginMaintenance(void)                                                const { return this.m_is_change_margin_maintenance_dec;        }
   //--- Assets:
   //--- setting the tracked value of the assets (1) growth, (2) decrease
   //--- getting (3) the assets change value,
   //--- getting the flag of the change exceeding the (4) growth, (5) decrease value
   void              SetControlAssetsInc(const double value)                                                { this.m_control_assets_inc=::fabs(value);               }
   void              SetControlAssetsDec(const double value)                                                { this.m_control_assets_dec=::fabs(value);               }
   double            GetValueChangedAssets(void)                                                      const { return this.m_changed_assets_value;                    }
   bool              IsIncreaseAssets(void)                                                           const { return this.m_is_change_assets_inc;                    }
   bool              IsDecreaseAssets(void)                                                           const { return this.m_is_change_assets_dec;                    }
   //--- Liabilities:
   //--- setting the tracked value of the liabilities (1) growth, (2) decrease
   //--- getting (3) the liabilities change value,
   //--- getting the flag of the liabilities change exceeding the (4) growth, (5) decrease value
   void              SetControlLiabilitiesInc(const double value)                                           { this.m_control_liabilities_inc=::fabs(value);          }
   void              SetControlLiabilitiesDec(const double value)                                           { this.m_control_liabilities_dec=::fabs(value);          }
   double            GetValueChangedLiabilities(void)                                                 const { return this.m_changed_liabilities_value;               }
   bool              IsIncreaseLiabilities(void)                                                      const { return this.m_is_change_liabilities_inc;               }
   bool              IsDecreaseLiabilities(void)                                                      const { return this.m_is_change_liabilities_dec;               }
   //--- Blocked commissions:
   //--- setting the tracked blocked commissions (1) growth, (2) decrease value
   //--- getting (3) the blocked commissions change value,
   //--- getting the flag of the tracked commissions change exceeding the (4) growth, (5) decrease value
   void              SetControlComissionBlockedInc(const double value)                                      { this.m_control_comission_blocked_inc=::fabs(value);    }
   void              SetControlComissionBlockedDec(const double value)                                      { this.m_control_comission_blocked_dec=::fabs(value);    }
   double            GetValueChangedComissionBlocked(void)                                            const { return this.m_changed_comission_blocked_value;         }
   bool              IsIncreaseComissionBlocked(void)                                                 const { return this.m_is_change_comission_blocked_inc;         }
   bool              IsDecreaseComissionBlocked(void)                                                 const { return this.m_is_change_comission_blocked_dec;         }
  };
//+------------------------------------------------------------------+
```

**Beyond the class body, implement the method returning the account event by its number in the list:**

```
//+------------------------------------------------------------------+
//| Return the account event by its number in the list               |
//+------------------------------------------------------------------+
ENUM_ACCOUNT_EVENT CAccountsCollection::GetEvent(constint shift=WRONG_VALUE)
  {
   int total=this.m_list_changes.Total();
   if(total==0)
      return ACCOUNT_EVENT_NO_EVENT;
   int index=(shift<0 || shift>total-1 ? total-1 : total-shift-1);
   int event=this.m_list_changes.At(index);
   return ENUM_ACCOUNT_EVENT(event!=NULL ? event : ACCOUNT_EVENT_NO_EVENT);
  }
//+------------------------------------------------------------------+
```

The events in the list of account property changes are located in the order they were added — the very first one is located at index 0, while the
very last one is located at (list\_size-1) index. However, we want to let users to obtain a desired event as in a time series — the zero index
should contain the very last event. To achieve this, the method features the index calculation: index = (list\_size -
desired\_event\_number-1). In this case, if we pass 0, the last event in the list is returned; if 1, the last but one; if a number exceeds the list
size, the last event is returned.

So, the index of a desired event is passed to the method.


First,

check the number of events in the list. If there are no events, return 'no event'.


Next,

check a desired event index. If the passed value is less than zero or goes beyond the
array size, the index specifies the last event in the list,
otherwise,

calculate the event index in the list according to the rule: if 0 is
passed to the method, we want to get the last event (as in a time series), if 1 — the last but one, etc. Alternatively, if you need to get the last
event, pass -1 as an index input.

Next, get the event from the list by the calculated index and return it.


If no event is received, return

NULL. This means that the method operation result should be checked for validity
before  we use it.

**Implement the method returning an account event description:**

```
//+------------------------------------------------------------------+
//| Return an account event description                              |
//+------------------------------------------------------------------+
string CAccountsCollection::EventDescription(const ENUM_ACCOUNT_EVENT event)
  {
   int total=this.m_list_accounts.Total();
   if(total==0)
      return(DFUN+TextByLanguage("Ошибка. Список изменений пуст","Error. List of changes is empty"));
   CAccount* account=this.m_list_accounts.At(this.m_index_current);
   if(account==NULL)
      return(DFUN+TextByLanguage("Ошибка. Не удалось получить данные аккаунта","Error. Failed to get account data"));
   constint dg=(account.MarginSOMode()==ACCOUNT_STOPOUT_MODE_MONEY ? (int)account.CurrencyDigits() : 2);
   conststring curency=" "+account.Currency();
   conststring mode_lev=(account.IsPercentsForSOLevels() ? "%" : " "+curency);   return
     (
      event==ACCOUNT_EVENT_NO_EVENT                ? TextByLanguage("Нет события","No event")                                                                                                                                                                     :
      event==ACCOUNT_EVENT_TRADE_ALLOWED_ON        ? TextByLanguage("Торговля на счёте разрешена","Trading on account allowed now")                                                                                                                        :
      event==ACCOUNT_EVENT_TRADE_ALLOWED_OFF       ? TextByLanguage("Торговля на счёте запрещена","Trading on account prohibited now")                                                                                                                     :
      event==ACCOUNT_EVENT_TRADE_EXPERT_ON         ? TextByLanguage("Автоторговля на счёте разрешена","Autotrading on account allowed now")                                                                                                                :
      event==ACCOUNT_EVENT_TRADE_EXPERT_OFF        ? TextByLanguage("Автоторговля на счёте запрещена","Autotrade on account prohibited now")                                                                                                               :
      event==ACCOUNT_EVENT_LEVERAGE_INC            ?
         TextByLanguage("Плечо увеличено на ","Leverage increased by ")+(string)this.GetValueChangedLeverage()+" (1:"+(string)account.Leverage()+")"                                                                                                              :
      event==ACCOUNT_EVENT_LEVERAGE_DEC            ?
         TextByLanguage("Плечо уменьшено на ","Leverage decreased by ")+(string)this.GetValueChangedLeverage()+" (1:"+(string)account.Leverage()+")"                                                                                                              :
      event==ACCOUNT_EVENT_LIMIT_ORDERS_INC        ?
         TextByLanguage("Максимально допустимое количество действующих отложенных ордеров увеличено на","Maximum allowable number of active pending orders increased by ")+(string)this.GetValueChangedLimitOrders()+" ("+(string)account.LimitOrders()+")"       :
      event==ACCOUNT_EVENT_LIMIT_ORDERS_DEC        ?
         TextByLanguage("Максимально допустимое количество действующих отложенных ордеров уменьшено на","Maximum allowable number of active pending orders decreased by ")+(string)this.GetValueChangedLimitOrders()+" ("+(string)account.LimitOrders()+")"       :
      event==ACCOUNT_EVENT_BALANCE_INC             ?
         TextByLanguage("Баланс счёта увеличен на ","Account balance increased by ")+::DoubleToString(this.GetValueChangedBalance(),dg)+curency+" ("+::DoubleToString(account.Balance(),dg)+curency+")"                                                           :
      event==ACCOUNT_EVENT_BALANCE_DEC             ?
         TextByLanguage("Баланс счёта уменьшен на ","Account balance decreased by ")+::DoubleToString(this.GetValueChangedBalance(),dg)+curency+" ("+::DoubleToString(account.Balance(),dg)+curency+")"                                                           :
      event==ACCOUNT_EVENT_EQUITY_INC              ?
         TextByLanguage("Средства увеличены на ","Equity increased by ")+::DoubleToString(this.GetValueChangedEquity(),dg)+curency+" ("+::DoubleToString(account.Equity(),dg)+curency+")"                                                                         :
      event==ACCOUNT_EVENT_EQUITY_DEC              ?
         TextByLanguage("Средства уменьшены на ","Equity decreased by ")+::DoubleToString(this.GetValueChangedEquity(),dg)+curency+" ("+::DoubleToString(account.Equity(),dg)+curency+")"                                                                         :
      event==ACCOUNT_EVENT_PROFIT_INC              ?
         TextByLanguage("Текущая прибыль счёта увеличена на ","Account current profit increased by ")+::DoubleToString(this.GetValueChangedProfit(),dg)+curency+" ("+::DoubleToString(account.Profit(),dg)+curency+")"                                            :
      event==ACCOUNT_EVENT_PROFIT_DEC              ?
         TextByLanguage("Текущая прибыль счёта уменьшена на ","Account current profit decreased by ")+::DoubleToString(this.GetValueChangedProfit(),dg)+curency+" ("+::DoubleToString(account.Profit(),dg)+curency+")"                                            :
      event==ACCOUNT_EVENT_CREDIT_INC              ?
         TextByLanguage("Предоставленный кредит увеличен на ","Credit increased by ")+::DoubleToString(this.GetValueChangedCredit(),dg)+curency+" ("+::DoubleToString(account.Credit(),dg)+curency+")"                                                            :
      event==ACCOUNT_EVENT_CREDIT_DEC              ?
         TextByLanguage("Предоставленный кредит уменьшен на ","Credit decreased by ")+::DoubleToString(this.GetValueChangedCredit(),dg)+curency+" ("+::DoubleToString(account.Credit(),dg)+curency+")"                                                            :
      event==ACCOUNT_EVENT_MARGIN_INC              ?
         TextByLanguage("Залоговые средства увеличены на ","Margin increased by ")+::DoubleToString(this.GetValueChangedMargin(),dg)+curency+" ("+::DoubleToString(account.Margin(),dg)+curency+")"                                                               :
      event==ACCOUNT_EVENT_MARGIN_DEC              ?
         TextByLanguage("Залоговые средства уменьшены на ","Margin decreased by ")+::DoubleToString(this.GetValueChangedMargin(),dg)+curency+" ("+::DoubleToString(account.Margin(),dg)+curency+")"                                                               :
      event==ACCOUNT_EVENT_MARGIN_FREE_INC         ?
         TextByLanguage("Свободные средства увеличены на ","Free margin increased by ")+::DoubleToString(this.GetValueChangedMarginFree(),dg)+curency+" ("+::DoubleToString(account.MarginFree(),dg)+curency+")"                                                  :
      event==ACCOUNT_EVENT_MARGIN_FREE_DEC         ?
         TextByLanguage("Свободные средства уменьшены на ","Free margin decreased by ")+::DoubleToString(this.GetValueChangedMarginFree(),dg)+curency+" ("+::DoubleToString(account.MarginFree(),dg)+curency+")"                                                  :
      event==ACCOUNT_EVENT_MARGIN_LEVEL_INC        ?
         TextByLanguage("Уровень залоговых средств увеличен на ","Margin level increased by ")+::DoubleToString(this.GetValueChangedMarginLevel(),dg)+"%"+" ("+::DoubleToString(account.MarginLevel(),dg)+"%)"                                                    :
      event==ACCOUNT_EVENT_MARGIN_LEVEL_DEC        ?
         TextByLanguage("Уровень залоговых средств уменьшен на ","Margin level decreased by ")+::DoubleToString(this.GetValueChangedMarginLevel(),dg)+"%"+" ("+::DoubleToString(account.MarginLevel(),dg)+"%)"                                                    :
      event==ACCOUNT_EVENT_MARGIN_INITIAL_INC      ?
         TextByLanguage("Гарантийная сумма по отложенным ордерам увеличена на ","Guarantee sum for pending orders increased by ")+::DoubleToString(this.GetValueChangedMarginInitial(),dg)+curency+" ("+::DoubleToString(account.MarginInitial(),dg)+curency+")" :
      event==ACCOUNT_EVENT_MARGIN_INITIAL_DEC      ?
         TextByLanguage("Гарантийная сумма по отложенным ордерам уменьшена на ","Guarantee sum for pending orders decreased by ")+::DoubleToString(this.GetValueChangedMarginInitial(),dg)+curency+" ("+::DoubleToString(account.MarginInitial(),dg)+curency+")" :
      event==ACCOUNT_EVENT_MARGIN_MAINTENANCE_INC  ?
         TextByLanguage("Гарантийная сумма по позициям увеличена на ","Guarantee sum for positions increased by ")+::DoubleToString(this.GetValueChangedMarginMaintenance(),dg)+curency+" ("+::DoubleToString(account.MarginMaintenance(),dg)+curency+")"        :
      event==ACCOUNT_EVENT_MARGIN_MAINTENANCE_DEC  ?
         TextByLanguage("Гарантийная сумма по позициям уменьшена на ","Guarantee sum for positions decreased by ")+::DoubleToString(this.GetValueChangedMarginMaintenance(),dg)+curency+" ("+::DoubleToString(account.MarginMaintenance(),dg)+curency+")"        :
      event==ACCOUNT_EVENT_MARGIN_SO_CALL_INC      ?
         TextByLanguage("Увеличен уровень Margin Call на ","Increased Margin Call level by ")+::DoubleToString(this.GetValueChangedMarginCall(),dg)+mode_lev+" ("+::DoubleToString(account.MarginSOCall(),dg)+mode_lev+")"                                        :
      event==ACCOUNT_EVENT_MARGIN_SO_CALL_DEC      ?
         TextByLanguage("Уменьшен уровень Margin Call на ","Decreased Margin Call level by ")+::DoubleToString(this.GetValueChangedMarginCall(),dg)+mode_lev+" ("+::DoubleToString(account.MarginSOCall(),dg)+mode_lev+")"                                        :
      event==ACCOUNT_EVENT_MARGIN_SO_SO_INC        ?
         TextByLanguage("Увеличен уровень Stop Out на ","Increased Margin Stop Out level by ")+::DoubleToString(this.GetValueChangedMarginStopOut(),dg)+mode_lev+" ("+::DoubleToString(account.MarginSOSO(),dg)+mode_lev+")"                                      :
      event==ACCOUNT_EVENT_MARGIN_SO_SO_DEC        ?
         TextByLanguage("Уменьшен уровень Stop Out на ","Decreased Margin Stop Out level by ")+::DoubleToString(this.GetValueChangedMarginStopOut(),dg)+mode_lev+" ("+::DoubleToString(account.MarginSOSO(),dg)+mode_lev+")"                                      :
      event==ACCOUNT_EVENT_ASSETS_INC              ?
         TextByLanguage("Размер активов увеличен на ","Assets increased by ")+::DoubleToString(this.GetValueChangedAssets(),dg)+" ("+::DoubleToString(account.Assets(),dg)+")"                                                                                    :
      event==ACCOUNT_EVENT_ASSETS_DEC              ?
         TextByLanguage("Размер активов уменьшен на ","Assets decreased by ")+::DoubleToString(this.GetValueChangedAssets(),dg)+" ("+::DoubleToString(account.Assets(),dg)+")"                                                                                    :
      event==ACCOUNT_EVENT_LIABILITIES_INC         ?
         TextByLanguage("Размер обязательств увеличен на ","Liabilities increased by ")+::DoubleToString(this.GetValueChangedLiabilities(),dg)+" ("+::DoubleToString(account.Liabilities(),dg)+")"                                                                :
      event==ACCOUNT_EVENT_LIABILITIES_DEC         ?
         TextByLanguage("Размер обязательств уменьшен на ","Liabilities decreased by ")+::DoubleToString(this.GetValueChangedLiabilities(),dg)+" ("+::DoubleToString(account.Liabilities(),dg)+")"                                                                :
      event==ACCOUNT_EVENT_COMISSION_BLOCKED_INC   ?
         TextByLanguage("Размер заблокированных комиссий увеличен на ","Blocked commissions increased by ")+::DoubleToString(this.GetValueChangedComissionBlocked(),dg)+" ("+::DoubleToString(account.ComissionBlocked(),dg)+")"                                   :
      event==ACCOUNT_EVENT_COMISSION_BLOCKED_DEC   ?
         TextByLanguage("Размер заблокированных комиссий уменьшен на ","Blocked commissions decreased by ")+::DoubleToString(this.GetValueChangedComissionBlocked(),dg)+" ("+::DoubleToString(account.ComissionBlocked(),dg)+")"                                   :
      ::EnumToString(event)
     );
  }
//+------------------------------------------------------------------+
```

Here the account event is passed to the method whose
description should be obtained.

Check the size of the account object list. If it is empty, return the error description.
Since we are able to track only the current account events,

we get the current account object from the account list by the current account
object index. If no object is received, also return the error
description.

Next, get the necessary account properties for the
correct event description display, check the event and return its
description.

In the initialization list of the class constructor, initialize symbol variables and control program chart ID with the default values — the
current symbol and the current chart,

clear the tick structure we will need to define the event time and initialize
editable and controlled account parameters:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccountsCollection::CAccountsCollection(void) : m_folder_name(DIRECTORY+"Accounts"),
                                                 m_is_account_event(false),
                                                 m_symbol(::Symbol()),
                                                 m_chart_id(::ChartID())
  {
   this.m_list_accounts.Clear();
   this.m_list_accounts.Sort(SORT_BY_ACCOUNT_LOGIN);
   this.m_list_accounts.Type(COLLECTION_ACCOUNT_ID);
   ::ZeroMemory(this.m_struct_prev_account);
   ::ZeroMemory(this.m_tick);
   this.InitChangesParams();
   this.InitControlsParams();
//--- Create the folder for storing account files
   ::ResetLastError();
   if(!::FolderCreate(this.m_folder_name,FILE_COMMON))
      Print(DFUN,TextByLanguage("Не удалось создать папку хранения файлов. Ошибка ","Could not create file storage folder. Error "),::GetLastError());
//--- Create the current account object and add it to the list
   CAccount* account=new CAccount();
   if(account!=NULL)
     {
      if(!this.AddToList(account))
        {
         Print(DFUN_ERR_LINE,TextByLanguage("Ошибка. Не удалось добавить текущий объект-аккаунт в список-коллекцию.","Error. Failed to add current account object to collection list."));
         delete account;
        }
      else
         account.PrintShort();
     }
   else
      Print(DFUN,TextByLanguage("Ошибка. Не удалось создать объект-аккаунт с данными текущего счёта.","Error. Failed to create an account object with current account data."));

//--- Download account objects from the files to the collection
   this.LoadObjects();
//--- Save the current account index
   this.m_index_current=this.Index();
  }
//+------------------------------------------------------------------+
```

**Finally, add tracking account changes to the method for updating the account data:**

```
//+------------------------------------------------------------------+
//| Update the current account data                                  |
//+------------------------------------------------------------------+
void CAccountsCollection::Refresh(void)
  {
   this.m_is_account_event=false;
   if(this.m_index_current==WRONG_VALUE)
      return;
   CAccount* account=this.m_list_accounts.At(this.m_index_current);
   if(account==NULL)
      return;
   ::ZeroMemory(this.m_struct_curr_account);
   this.SetAccountsParams(account);

//--- First launch
   if(!this.m_struct_prev_account.login)
     {
      this.m_struct_prev_account=this.m_struct_curr_account;
      return;
     }
//--- f the account hash sum changed
   if(this.m_struct_curr_account.hash_sum!=this.m_struct_prev_account.hash_sum)
     {
      this.m_change_code=this.SetChangeCode();
      this.SetTypeEvent();
      int total=this.m_list_changes.Total();
      if(total>0)
        {
         this.m_is_account_event=true;
         for(int i=0;i<total;i++)
           {
            ENUM_ACCOUNT_EVENT event=this.GetEvent(i);
            if(event==NULL || !::SymbolInfoTick(this.m_symbol,this.m_tick))
               continue;
            string sparam=TimeMSCtoString(this.m_tick.time_msc)+": "+this.EventDescription(event);
            Print(sparam);
            ::EventChartCustom(this.m_chart_id,(ushort)event,this.m_tick.time_msc,(double)i,sparam);
           }
        }
      this.m_struct_prev_account.hash_sum=this.m_struct_curr_account.hash_sum;
     }
  }
//+------------------------------------------------------------------+
```

First, reset the account event flag. Since we have removed the
SavePrevValues() method,

add the string from it instead
— saving the current data structure in the previous data one during the first launch. When changing the hash, check
and set the event code and the occurred event type.

In the method for setting the event type, all events that occurred simultaneously in the account properties are passed to the change list.
Therefore,

check the change list size first. If it is not empty, set
the occurred change flag, receive the event in a loop by the
list data,

set its string description consisting of time in milliseconds and
the event description,

temporarily display the event description in the journal (this feature is to be
removed in the future, all the library system messages are to be displayed in the journal only if logging is enabled) and, finally, send
the event to the control program using

[EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom).

**In the EventChartCustom() function parameters, pass:**

- an event — to **event\_id**,

- an event time in milliseconds — to **lparam**

- event index in the list of simultaneously occurred account changes — to **dparam**

- event string description — to **sparam**

At the very end of the method, make sure to save the current hash as
the previous one for a subsequent verification.

**This concludes the improvement of the CAccountsCollection class.**

Let's move on to the **CEngine** class which is the library's alpha and omega. We need to add all the necessary functionality for
working with account events to it.

In the private section of the class, add the variables for
storing the flag of the account properties change event and the last
event occurred on the account. In the public section, add the methods returning the
list of account events occurred simultaneously and the last
account event:

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CArrayObj            m_list_counters;                 // List of timer counters
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_tester;                     // Flag of working in the tester
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   bool                 m_is_account_event;              // Account change event flag
   ENUM_TRADE_EVENT     m_last_trade_event;              // Account last trading event
   ENUM_ACCOUNT_EVENT   m_last_account_event;            // Last event in account properties
//--- Return the counter index by id
   int                  CounterIndex(constint id) const;
//--- Return (1) the first launch flag, (2) the flag presence in a trading event
   bool                 IsFirstStart(void);
//--- Working with (1) order, deal and position, as well as (2) account events
   void                 TradeEventsControl(void);
   void                 AccountEventsControl(void);
//--- Return the last (1) market pending order, (2) market order, (3) last position, (4) position by ticket
   COrder*              GetLastMarketPending(void);
   COrder*              GetLastMarketOrder(void);
   COrder*              GetLastPosition(void);
   COrder*              GetPosition(constulong ticket);
//--- Return the last (1) removed pending order, (2) historical market order, (3) historical order (market or pending) by its ticket
   COrder*              GetLastHistoryPending(void);
   COrder*              GetLastHistoryOrder(void);
   COrder*              GetHistoryOrder(constulong ticket);
//--- Return the (1) first and the (2) last historical market orders from the list of all position orders, (3) the last deal
   COrder*              GetFirstOrderPosition(constulong position_id);
   COrder*              GetLastOrderPosition(constulong position_id);
   COrder*              GetLastDeal(void);
public:
   //--- Return the list of market (1) positions, (2) pending orders and (3) market orders
   CArrayObj*           GetListMarketPosition(void);
   CArrayObj*           GetListMarketPendings(void);
   CArrayObj*           GetListMarketOrders(void);
   //--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals, (4) all position market orders by its id
   CArrayObj*           GetListHistoryOrders(void);
   CArrayObj*           GetListHistoryPendings(void);
   CArrayObj*           GetListDeals(void);
   CArrayObj*           GetListAllOrdersByPosID(constulong position_id);
//--- Return the list of (1) accounts, (2) account events
   CArrayObj*           GetListAllAccounts(void)                        { returnthis.m_accounts.GetList();          }
   CArrayInt*           GetListAccountEvents(void)                      { returnthis.m_accounts.GetListChanges();   }
//--- Return the list of order, deal and position events
   CArrayObj*           GetListAllOrdersEvents(void)                    { returnthis.m_events.GetList();            }
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_events.ResetLastTradeEvent(); }
//--- Return the (1) last trading event, (2) the last event in the account properties, (3) hedging account flag, (4) flag of working in the tester
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { returnthis.m_last_trade_event;            }
   ENUM_ACCOUNT_EVENT   LastAccountEvent(void)                    const { returnthis.m_last_account_event;          }
   bool                 IsHedge(void)                             const { returnthis.m_is_hedge;                    }
   bool                 IsTester(void)                            const { returnthis.m_is_tester;                   }
//--- Create the timer counter
   void                 CreateCounter(constint id,constulong frequency,constulong pause);
//--- Timer
   void                 OnTimer(void);
//--- Constructor/Destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

**Add initialization of the last account event in the**
**class constructor's initialization list:**

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),m_last_trade_event(TRADE_EVENT_NO_EVENT),m_last_account_event(ACCOUNT_EVENT_NO_EVENT)
  {
```

**Add the AccountEventsControl() method**, which previously was used only for calling the Refresh() method of the account
collection class:

```
//+------------------------------------------------------------------+
//| Check account events                                             |
//+------------------------------------------------------------------+
void CEngine::AccountEventsControl(void)
  {
//--- Check account property changes and set the flag of the account change events
   this.m_accounts.Refresh();
   this.m_is_account_event=this.m_accounts.IsAccountEvent();
//--- If there are account property changes
   if(this.m_is_account_event)
     {
      //--- Get the last event of the account property change
      this.m_last_account_event=this.m_accounts.GetEvent();
     }
  }
//+------------------------------------------------------------------+
```

Here all is simple. First, update the account data. If
any of the account properties has changed, simply write the last event
to the variable. All the remaining data on the event is retrieved in the library-based control program.

### Testing the account events

To test the account event, we can use the EA from the [previous article](https://www.mql5.com/en/articles/6952),
since the library finds all account properties on its own, sends an appropriate message to the chart event and displays the description of
the occurred account event in the journal.

But let's try to move beyond the library "sandbox" and handle some account events in the program, for example increasing the equity.

Currently, the access to the program features from the outside is very limited. However, this is the case only till we collect and handle the necessary
data — the library displays various events in the journal to test the correctness of created classes, collected data and tracked events.
Further on, we will introduce a simple and convenient access to any library data greatly simplifying their retrieval from the program.

To obtain data on account property changes right now, let's make slight improvements in the CEngine class. We will need to gain access to the
current account object and events.

To do this, **add the necessary methods in the public section of the CEngine class (the Engine.mqh file):**

```
public:
   //--- Return the list of market (1) positions, (2) pending orders and (3) market orders
   CArrayObj*           GetListMarketPosition(void);
   CArrayObj*           GetListMarketPendings(void);
   CArrayObj*           GetListMarketOrders(void);
   //--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals, (4) all position market orders by its id
   CArrayObj*           GetListHistoryOrders(void);
   CArrayObj*           GetListHistoryPendings(void);
   CArrayObj*           GetListDeals(void);
   CArrayObj*           GetListAllOrdersByPosID(constulong position_id);
//--- Return the list of (1) accounts, (2) account events, (3) account change event by its index in the list
//--- (4) the current account, (5) event description
   CArrayObj*           GetListAllAccounts(void)                        { return this.m_accounts.GetList();          }
   CArrayInt*           GetListAccountEvents(void)                      { return this.m_accounts.GetListChanges();   }
   ENUM_ACCOUNT_EVENT   GetAccountEventByIndex(const int index)         { return this.m_accounts.GetEvent(index);    }
   CAccount*            GetAccountCurrent(void);
   string               GetAccountEventDescription(ENUM_ACCOUNT_EVENT event);

```

Here we added the method for receiving an account event by its index in the change
list, the method for receiving the current account object
and the method for receiving the account event description.

The method for receiving an account event by index simply
returns the operation result of the previously described GetEvent() method in the account collection class.

At the end of the listing, **implement the method for receiving the current account object:**

```
//+------------------------------------------------------------------+
//| Return the current account                                       |
//+------------------------------------------------------------------+
CAccount* CEngine::GetAccountCurrent(void)
  {
   int index=this.m_accounts.IndexCurrentAccount();
   if(index==WRONG_VALUE)
      return NULL;
   CArrayObj* list=this.m_accounts.GetList();
   return(list!=NULL ? (list.At(index)!=NULL ? list.At(index) : NULL) : NULL);
  }
//+------------------------------------------------------------------+
```

First, get the current account index from the account collection class. If
no index is received, return

NULL. Next, get the
full list of accounts and return the necessary current account by
its index in the account list. In case of a list or account
error, return NULL.

Also, **let's implement the method returning an account event description:**

```
//+------------------------------------------------------------------+
//| Return the account event description                             |
//+------------------------------------------------------------------+
string CEngine::GetAccountEventDescription(ENUM_ACCOUNT_EVENT event)
  {
   return this.m_accounts.EventDescription(event);
  }
//+------------------------------------------------------------------+
```

The method returns the operation result of the account collection class method returning the account event description.

To test the account events, use the EA from the previous article **TestDoEasyPart12\_2.mq5** located in
\\MQL5\\Experts\\TestDoEasy\\Part12\ and save it as

**TestDoEasyPart13.mq5** in \\MQL5\\Experts\\TestDoEasy\\Part13.

**Remove the following input right away**

```
inputbool     InpFullProperties    =  false;// Show full accounts properties
```

together with the fast check of the account collection **in the OnInit() handler:**

```
//--- Fast check of the account object collection
   CArrayObj* list=engine.GetListAllAccounts();
   if(list!=NULL)
     {
      int total=list.Total();
      if(total>0)
         Print("\n",TextByLanguage("=========== Список сохранённых аккаунтов ===========","=========== List of saved accounts ==========="));
      for(int i=0;i<total;i++)
        {
         CAccount* account=list.At(i);
         if(account==NULL)
            continue;
         Sleep(100);
         if(InpFullProperties)
            account.Print();
         else
            account.PrintShort();
        }
     }
//---
```

_Since we changed the account object structure (changed the size of uchar arrays for storing account string properties and added_
_another integer property), all previously saved account object files will no longer be downloaded correctly. If they are present in_
_\\Files\\DoEasy\\Accounts\ of the common terminal folder, delete them before launching the test EA. They will be created anew when_
_switching from one account to another with an already new object structure size._

The OnInit() handler now looks as follows:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();
//--- Set EA global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      butt_data[i].name=prefix+EnumToString((ENUM_BUTTONS)i);
      butt_data[i].text=EnumToButtText((ENUM_BUTTONS)i);
     }
   lot=NormalizeLot(Symbol(),fmax(InpLots,MinimumLots(Symbol())*2.0));
   magic_number=InpMagic;
   stoploss=InpStopLoss;
   takeprofit=InpTakeProfit;
   distance_pending=InpDistance;
   distance_stoplimit=InpDistanceSL;
   slippage=InpSlippage;
   trailing_stop=InpTrailingStop*Point();
   trailing_step=InpTrailingStep*Point();
   trailing_start=InpTrailingStart;
   stoploss_to_modify=InpStopLossModify;
   takeprofit_to_modify=InpTakeProfitModify;

//--- Check and remove remaining EA graphical objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      returnINIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);

//--- Set CTrade trading class parameters
#ifdef __MQL5__
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
#endif
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

In order to handle all events arriving to the program from the library and avoid cluttering the built-in OnChartEvent(), implement a
separate

**OnDoEasyEvent()** handler to process all obtained events in it. This makes the code more readable and, more importantly, allows us to
handle events in the tester, which is exactly what we need right now, since we are going to handle the "equity increase exceeding the
specified growth value" account event, and it is much faster and easier to check everything in the tester to achieve that.

**Let's add the necessary functionality for handling the account events in the OnTick() handler:**

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Initialize the last events
   static ENUM_TRADE_EVENT last_trade_event=WRONG_VALUE;
   static ENUM_ACCOUNT_EVENT last_account_event=WRONG_VALUE;
//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();
      PressButtonsControl();
     }
//--- If the last trading event changed
   if(engine.LastTradeEvent()!=last_trade_event)
     {
      last_trade_event=engine.LastTradeEvent();
      Comment("\nLast trade event: ",EnumToString(last_trade_event));
     }
//--- If the last account event changed
   if(engine.LastAccountEvent()!=last_account_event)
     {
      last_account_event=engine.LastAccountEvent();
      //--- If this is a tester
      if(MQLInfoInteger(MQL_TESTER))
        {
         //--- Get the list of all account events occurred simultaneously
         CArrayInt* list=engine.GetListAccountEvents();
         if(list!=NULL)
           {
            //--- Get the next event in a loop
            int total=list.Total();
            for(int i=0;i<total;i++)
              {
               ENUM_ACCOUNT_EVENT event=(ENUM_ACCOUNT_EVENT)list.At(i);
               if(event==NULL)
                  continue;
               string sparam=engine.GetAccountEventDescription(event);
               long lparam=TimeCurrent()*1000;
               double dparam=(double)i;
               //--- Send an event to the event handler
               OnDoEasyEvent(CHARTEVENT_CUSTOM+event,lparam,dparam,sparam);
              }
           }
        }
      Comment("\nLast account event: ",EnumToString(last_account_event));
     }
//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();
      TrailingOrders();
     }
  }
//+------------------------------------------------------------------+
```

Here we introduced the new variable storing the type of the last account
event. Check its current status in relation to the type of the last event
returned by the CAccountsCollection class. If the status has changed, there was an account event. Next, for
the tester only, get a new event in a loop according to the account list of events occurred simultaneously and send
it to the library event handler. The listing comments contain all actions for receiving events and sending them to the handler. When
working with the tester, we cannot access data on an event time in milliseconds. Therefore, simply send the

current time \* 1000.

Now let's **improve the OnChartEvent() event handler:**

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   if(MQLInfoInteger(MQL_TESTER))
      return;
   if(id==CHARTEVENT_OBJECT_CLICK && StringFind(sparam,"BUTT_")>0)
     {
      PressButtonEvents(sparam);
     }
//--- DoEasy library event
   if(id>=CHARTEVENT_CUSTOM)
     {
      OnDoEasyEvent(id,lparam,dparam,sparam);
     }
  }
//+------------------------------------------------------------------+
```

Here we added calling the library event handler in case
the event ID indicates a library event.

Finally, **implement the library event handler:**

```
//+------------------------------------------------------------------+
//| Handling DoEasy library events                                   |
//+------------------------------------------------------------------+
void OnDoEasyEvent(const int id,
                   const long &lparam,
                   const double &dparam,
                   const string &sparam)
  {
   int idx=id-CHARTEVENT_CUSTOM;
   string event="::"+string(idx);
   int digits=Digits();
//--- Handling trading events
   if(idx<TRADE_EVENTS_NEXT_CODE)
     {
      event=EnumToString((ENUM_TRADE_EVENT)ushort(idx));
      digits=(int)SymbolInfoInteger(sparam,SYMBOL_DIGITS);
     }
//--- Handling account events
   else if(idx<ACCOUNT_EVENTS_NEXT_CODE)
     {
      event=EnumToString((ENUM_ACCOUNT_EVENT)ushort(idx));
      digits=0;

      //--- if this is an equity increase
      if((ENUM_ACCOUNT_EVENT)idx==ACCOUNT_EVENT_EQUITY_INC)
        {
         Print(DFUN,sparam);
         //--- Close a position with the highest profit when the equity exceeds the value
         //--- specified in the CAccountsCollection::InitControlsParams() method for
         //--- the m_control_equity_inc variable tracking the equity growth by 15 units (by default)
         //--- AccountCollection file, InitControlsParams() method, string 1199

         //--- Get the list of all open positions
         CArrayObj* list_positions=engine.GetListMarketPosition();
         //--- Sort the list by profit considering commission and swap
         list_positions.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the position index with the highest profit
         int index=CSelect::FindOrderMax(list_positions,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list_positions.At(index);
            if(position!=NULL)
              {
               //--- Get a ticket of a position with the highest profit and close the position by a ticket
               #ifdef __MQL5__
                  trade.PositionClose(position.Ticket());
               #else
                  PositionClose(position.Ticket(),position.Volume());
               #endif
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

When sending an event to the control program chart, the value of CHARTEVENT\_CUSTOM equal to 1000 is added to the code. Therefore, in order to
receive a true code, we need to

subtract the CHARTEVENT\_CUSTOM value from the obtained code. If
the event's numerical value is located within the range from TRADE\_EVENTS\_NEXT\_CODE to ACCOUNT\_EVENTS\_NEXT\_CODE-1, this means an
account property change event has arrived. Since we want to close the most profitable position when the equity increases exceeding the
value specified in the settings (the default growth value is 15 and higher), we need to

check the "equity exceeding the specified value" event, display
the event description in the journal and close the most profitable position. The
code contains all the necessary comments.

If we now simply launch the EA on the chart, we will get the journal entries concerning the trading being enabled/disabled after a while:

```
2019.06.1010:56:33.8772019.06.1006:55:29.279: Trading on the account is prohibited now
2019.06.1011:08:56.5492019.06.1007:08:51.900: Trading on the account is allowed now
```

On MetaQuotes-Demo, such enabling/disabling can be observed several times a day allowing us to check how the library defines such
events on a demo account.

Now launch the EA in the tester and open more positions to quickly detect the equity increase event followed by closing the most profitable
position:

![](https://c.mql5.com/2/36/2019-06-10_14-40-19.gif)

As we can see, the most profitable position is automatically closed when the equity exceeds the specified value. The journal displays the
messages about the tracked account event.

### What's next?

In the next part, I am going to start working with symbols. I want to implement symbol objects, symbol object collection and symbol events.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6995#node00)

**Previous articles within the series:**

[Part 1. Concept, data management.](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals.](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search.](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept.](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program.](https://www.mql5.com/en/articles/6211)

[Part \\
6\. Netting account events.](https://www.mql5.com/en/articles/6383)

[Part 7. StopLimit order activation events, preparing \\
the functionality for order and position modification events.](https://www.mql5.com/en/articles/6482)

[Part 8. Order and \\
position modification events.](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 - Preparing data.](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and activating pending \\
orders.](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure events.](https://www.mql5.com/en/articles/6921)

[Part 12. Account object class and account object collection.](https://www.mql5.com/en/articles/6952)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6995](https://www.mql5.com/ru/articles/6995)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6995.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6995/mql5.zip "Download MQL5.zip")(129.92 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/6995/mql4.zip "Download MQL4.zip")(127.28 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/321250)**
(12)


![Mohammad Bazrkar](https://c.mql5.com/avatar/2017/12/5A36777B-2BC1.jpg)

**[Mohammad Bazrkar](https://www.mql5.com/en/users/mhdbzr)**
\|
14 Oct 2019 at 14:47

Thank you Artyom, I found it before,

I think there is a bug: when we Open or Close more than one trades in the same second, it is missing one of them or only detect one of them; also it is detecting twice some times!

let me show you my logs

2019.10.14 15:58:14.489MT4EventsEA EURUSD,M1: uninit reason 4

2019.10.14 15:57:28.424MT4EventsEA EURUSD,M1: 2019.10.14 15:27:22.000: Account balance decreased by -0.75 USD (5445.52 USD)

2019.10.14 15:57:28.423MT4EventsEA EURUSD,M1: 2019.10.14 15:27:22.000: Margin level decreased by -16456.49% (0.00%)

2019.10.14 15:57:27.895MT4EventsEA EURUSD,M1: CEventsCollection::CreateNewEvent, Line 767: This event already in the list.

2019.10.14 15:57:27.474MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:27:13.000 - EURUSD Close Buy #542264335 at price 1.10304, profit -0.14 USD

2019.10.14 15:57:27.472MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:27:12.000 - EURUSD Close Buy #542264333 at price 1.10304, profit -0.14 USD

2019.10.14 15:57:19.095MT4EventsEA EURUSD,M1: 2019.10.14 15:27:13.000: Margin level decreased by -32915.87% (16456.49%)

2019.10.14 15:57:18.844MT4EventsEA EURUSD,M1: - Position opened: 2019.10.14 15:27:13.000 - EURUSD Open 0.01 Buy #542264335 \[0.01 Market order Buy #542264335\] at price 1.10304

2019.10.14 15:57:18.002MT4EventsEA EURUSD,M1: - Position opened: 2019.10.14 15:27:12.000 - EURUSD Open 0.01 Buy #542264333 \[0.01 Market order Buy #542264333\] at price 1.10304

2019.10.14 15:57:17.547MT4EventsEA EURUSD,M1: 2019.10.14 15:27:12.000: Margin level increased by 49372.35% (49372.35%)

2019.10.14 15:57:17.175MT4EventsEA EURUSD,M1: - Position opened: 2019.10.14 15:27:11.000 - EURUSD Open 0.01 Buy #542264331 \[0.01 Market order Buy #542264331\] at price 1.10301

2019.10.14 15:56:17.144MT4EventsEA EURUSD,M1: 2019.10.14 15:26:11.000: Margin level decreased by -12343.32% (0.00%)

2019.10.14 15:56:16.800MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:22:40.000 - EURUSD Close Buy #542263747 at price 1.10301, profit -0.07 USD

2019.10.14 15:56:16.798MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:22:41.000 - EURUSD Close Buy #542263749 at price 1.10301, profit -0.07 USD

2019.10.14 15:56:16.347MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:22:42.000 - EURUSD Close Buy #542263753 at price 1.10304, profit -0.14 USD

2019.10.14 15:56:15.926MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:22:42.000 - EURUSD Close Buy #542263753 at price 1.10304, profit -0.14 USD

2019.10.14 15:56:09.140MT4EventsEA EURUSD,M1: initialized

here is MT4 Account's Trade History:

[![](https://c.mql5.com/3/294/image__31.png)](https://c.mql5.com/3/294/image__30.png "https://c.mql5.com/3/294/image__30.png")

and it is my code:

```
void OnTick() { DoChecks(); }
void OnTimer() { DoChecks(); }
//+------------------------------------------------------------------+
//| My Checking function                                             |
//+------------------------------------------------------------------+
void DoChecks() {
   engine.OnTimer();

   //\\//\\ Trade Events
   if(engine.LastTradeEvent()!=TRADE_EVENT_NO_EVENT) { //engine.LastTradeEvent()!=last_trade_event
      CArrayObj* list=engine.GetListAllOrdersEvents();
      if(list!=NULL) {
         int total=list.Total();
         for(int i=0; i<total ; i++){
            CheckTradeEvent(list.At(i));
         }
      }
      //--
      engine.ResetLastTradeEvent();
   }

   //\\//\\ Account events
   if(engine.LastAccountEvent()!=ACCOUNT_EVENT_NO_EVENT){
      CArrayInt* list=engine.GetListAccountEvents();
      if(list!=NULL){
         int total=list.Total();
         for(int i=0;i<total;i++){
            ENUM_ACCOUNT_EVENT event=(ENUM_ACCOUNT_EVENT)list.At(i);
            if(event==NULL){ continue; }
            CheckAccountEvent(event);
         }
      }
      //--
      engine.ResetLastAccountEvent();
   }

}
```

What do you think about it, is there any problem in my implementation?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
14 Oct 2019 at 15:04

**Mohammad Bazrkar:**

Thank you Artyom, I found it before,

I think there is a bug: when we Open or Close more than one trades in the same second, it is missing one of them or only detect one of them; also it is detecting twice some times!

let me show you my logs

2019.10.14 15:58:14.489MT4EventsEA EURUSD,M1: uninit reason 4

2019.10.14 15:57:28.424MT4EventsEA EURUSD,M1: 2019.10.14 15:27:22.000: Account balance decreased by -0.75 USD (5445.52 USD)

2019.10.14 15:57:28.423MT4EventsEA EURUSD,M1: 2019.10.14 15:27:22.000: Margin level decreased by -16456.49% (0.00%)

2019.10.14 15:57:27.895MT4EventsEA EURUSD,M1: CEventsCollection::CreateNewEvent, Line 767: This event already in the list.

2019.10.14 15:57:27.474MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:27:13.000 - EURUSD Close Buy #542264335 at price 1.10304, profit -0.14 USD

2019.10.14 15:57:27.472MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:27:12.000 - EURUSD Close Buy #542264333 at price 1.10304, profit -0.14 USD

2019.10.14 15:57:19.095MT4EventsEA EURUSD,M1: 2019.10.14 15:27:13.000: Margin level decreased by -32915.87% (16456.49%)

2019.10.14 15:57:18.844MT4EventsEA EURUSD,M1: - Position opened: 2019.10.14 15:27:13.000 - EURUSD Open 0.01 Buy #542264335 \[0.01 Market order Buy #542264335\] at price 1.10304

2019.10.14 15:57:18.002MT4EventsEA EURUSD,M1: - Position opened: 2019.10.14 15:27:12.000 - EURUSD Open 0.01 Buy #542264333 \[0.01 Market order Buy #542264333\] at price 1.10304

2019.10.14 15:57:17.547MT4EventsEA EURUSD,M1: 2019.10.14 15:27:12.000: Margin level increased by 49372.35% (49372.35%)

2019.10.14 15:57:17.175MT4EventsEA EURUSD,M1: - Position opened: 2019.10.14 15:27:11.000 - EURUSD Open 0.01 Buy #542264331 \[0.01 Market order Buy #542264331\] at price 1.10301

2019.10.14 15:56:17.144MT4EventsEA EURUSD,M1: 2019.10.14 15:26:11.000: Margin level decreased by -12343.32% (0.00%)

2019.10.14 15:56:16.800MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:22:40.000 - EURUSD Close Buy #542263747 at price 1.10301, profit -0.07 USD

2019.10.14 15:56:16.798MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:22:41.000 - EURUSD Close Buy #542263749 at price 1.10301, profit -0.07 USD

2019.10.14 15:56:16.347MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:22:42.000 - EURUSD Close Buy #542263753 at price 1.10304, profit -0.14 USD

2019.10.14 15:56:15.926MT4EventsEA EURUSD,M1: - Position closed: 2019.10.14 15:22:42.000 - EURUSD Close Buy #542263753 at price 1.10304, profit -0.14 USD

2019.10.14 15:56:09.140MT4EventsEA EURUSD,M1: initialized

here is MT4 Account's Trade History:

and it is my code:

What do you think about it, is there any problem in my implementation?

Wait for the next article in the Russian segment of the resource.

In the next article, tracking of [trading events](https://www.mql5.com/en/articles/232 "Article: Trade Events in MetaTrader 5 ") that occurred simultaneously in one cycle is already ready and working.

And it also provides an example of tracking trading events in an adviser.

![Mohammad Bazrkar](https://c.mql5.com/avatar/2017/12/5A36777B-2BC1.jpg)

**[Mohammad Bazrkar](https://www.mql5.com/en/users/mhdbzr)**
\|
14 Oct 2019 at 18:32

**Artyom Trishkin:**

Wait for the next article in the Russian segment of the resource.

In the next article, tracking of trading events that occurred simultaneously in one cycle is already ready and working.

And it also provides an example of tracking trading events in an adviser.

Thank you for your helpful replies and articles.

![MQL_User](https://c.mql5.com/avatar/avatar_na2.png)

**[MQL\_User](https://www.mql5.com/en/users/mql_user)**
\|
10 Jul 2022 at 07:20

A small clarification regarding the definition of the terminal type in the constructor:

```
this.m_long_prop[ACCOUNT_PROP_SERVER_TYPE]                        = (::TerminalInfoString(TERMINAL_NAME)=="MetaTrader 5" ? 5 : 4);
```

It would probably be more correct to write it like this:

this.m\_long\_prop\[ACCOUNT\_PROP\_SERVER\_TYPE\] = (::StringFind [(TerminalInfoString(](https://www.mql5.com/en/docs/check/terminalinfostring "MQL5 documentation: TerminalInfoString function") TERMINAL\_NAME), "MetaTrader 5") != -1 ? 5 : 4);

Since the broker's name may also appear there....

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
10 Jul 2022 at 07:27

**MQL\_User (TerminalInfoString( TERMINAL\_NAME), "MetaTrader 5") != -1 ? 5 : 4);**
**Since the broker's name may also appear there....**

Can you please send me a link to a broker where this happens?

![Library for easy and quick development of MetaTrader programs (part XIV): Symbol object](https://c.mql5.com/2/36/MQL5-avatar-doeasy__9.png)[Library for easy and quick development of MetaTrader programs (part XIV): Symbol object](https://www.mql5.com/en/articles/7014)

In this article, we will create the class of a symbol object that is to be the basic object for creating the symbol collection. The class will allow us to obtain data on the necessary symbols for their further analysis and comparison.

![Library for easy and quick development of MetaTrader programs (part XII): Account object class and collection of account objects](https://c.mql5.com/2/36/MQL5-avatar-doeasy__7.png)[Library for easy and quick development of MetaTrader programs (part XII): Account object class and collection of account objects](https://www.mql5.com/en/articles/6952)

In the previous article, we defined position closure events for MQL4 in the library and got rid of the unused order properties. Here we will consider the creation of the Account object, develop the collection of account objects and prepare the functionality for tracking account events.

![Optimization management (Part II): Creating key objects and add-on logic](https://c.mql5.com/2/36/mql5-avatar-opt_control__1.png)[Optimization management (Part II): Creating key objects and add-on logic](https://www.mql5.com/en/articles/7059)

This article is a continuation of the previous publication related to the creation of a graphical interface for optimization management. The article considers the logic of the add-on. A wrapper for the MetaTrader 5 terminal will be created: it will enable the running of the add-on as a managed process via C#. In addition, operation with configuration files and setup files is considered in this article. The application logic is divided into two parts: the first one describes the methods called after pressing a particular key, while the second part covers optimization launch and management.

![Library for easy and quick development of MetaTrader programs (part XI). Compatibility with MQL4 - Position closure events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__6.png)[Library for easy and quick development of MetaTrader programs (part XI). Compatibility with MQL4 - Position closure events](https://www.mql5.com/en/articles/6921)

We continue the development of a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the tenth part, we resumed our work on the library compatibility with MQL4 and defined the events of opening positions and activating pending orders. In this article, we will define the events of closing positions and get rid of the unused order properties.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ivcddegjnjyjpvxppxjjnefwvbrqjwrd&ssn=1769186424897731654&ssn_dr=0&ssn_sr=0&fv_date=1769186424&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6995&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XIII)%3A%20Account%20object%20events%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918642478885983&fz_uniq=5070473128255100535&sv=2552)

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