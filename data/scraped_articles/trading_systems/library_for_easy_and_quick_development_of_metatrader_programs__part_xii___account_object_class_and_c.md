---
title: Library for easy and quick development of MetaTrader programs (part XII): Account object class and collection of account objects
url: https://www.mql5.com/en/articles/6952
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:40:38.676876
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ogdpjkeyirnpoovlbwpvahdjwewvcmqk&ssn=1769186436435021721&ssn_dr=0&ssn_sr=0&fv_date=1769186436&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6952&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XII)%3A%20Account%20object%20class%20and%20collection%20of%20account%20objects%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918643625577299&fz_uniq=5070475739595216515&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Account object](https://www.mql5.com/en/articles/6952#node01)
- [Testing the account object](https://www.mql5.com/en/articles/6952#node02)
- [Collection of account objects](https://www.mql5.com/en/articles/6952#node03)
- [Testing the account collection](https://www.mql5.com/en/articles/6952#node04)
- [What's next?](https://www.mql5.com/en/articles/6952#node05)


Before developing library trading classes, we need to create some additional trading-related classes. More specifically, we need data on a
trading account and traded symbols. This article will be devoted to the account object.

Since account data may change right during trading, we are going to prepare an account object followed by account object collection. Next, we
will implement tracking events on the account. This will allow us to detect changes in leverage, balance, profit/loss, equity and account
limitations.

### Account object

A trading account volume is identical to previously created objects I have described in the past articles. The only difference between
this object and the previously considered ones is that it is not a kind of abstract object with successors specifying a certain object
status. The account object is an independent object featuring all account properties. Such objects are to be added to the account object
collection allowing users to compare data of different accounts by various parameters.

As usual, we start with the development of all enumerations of the account object properties that are necessary for working with the class.


Open the

[account properties](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_integer)
in the editor help:

For the [AccountInfoInteger()](https://www.mql5.com/en/docs/account/accountinfointeger) function

ENUM\_ACCOUNT\_INFO\_INTEGER

|     |     |     |
| --- | --- | --- |
| ID | Description | Property type |
| ACCOUNT\_LOGIN | Account number | long |
| ACCOUNT\_TRADE\_MODE | Trading account type | [ENUM\_ACCOUNT\_TRADE\_MODE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode) |
| ACCOUNT\_LEVERAGE | Leverage | long |
| ACCOUNT\_LIMIT\_ORDERS | Maximum allowed number of active pending orders | int |
| ACCOUNT\_MARGIN\_SO\_MODE | Mode of setting the minimum available margin level | [ENUM\_ACCOUNT\_STOPOUT\_MODE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_stopout_mode) |
| ACCOUNT\_TRADE\_ALLOWED | [Trading permission](https://www.mql5.com/en/docs/runtime/tradepermission) of the current account | bool |
| ACCOUNT\_TRADE\_EXPERT | [Trading permission](https://www.mql5.com/en/docs/runtime/tradepermission) of an EA | bool |
| ACCOUNT\_MARGIN\_MODE | Margin calculation mode | [ENUM\_ACCOUNT\_MARGIN\_MODE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_margin_mode) |
| ACCOUNT\_CURRENCY\_DIGITS | Number of digits for an account currency necessary for accurate display of trading results | int |

For the [AccountInfoDouble()](https://www.mql5.com/en/docs/account/accountinfodouble) function

ENUM\_ACCOUNT\_INFO\_DOUBLE

|     |     |     |
| --- | --- | --- |
| ID | Description | Property type |
| ACCOUNT\_BALANCE | Account balance in a deposit currency | double |
| ACCOUNT\_CREDIT | Credit in a deposit currency | double |
| ACCOUNT\_PROFIT | Current profit on an account in the account currency | double |
| ACCOUNT\_EQUITY | Equity on an account in the deposit currency | double |
| ACCOUNT\_MARGIN | Reserved margin on an account in the deposit currency | double |
| ACCOUNT\_MARGIN\_FREE | Free funds available for opening a position on an account in the deposit currency | double |
| ACCOUNT\_MARGIN\_LEVEL | Margin level on an account in % | double |
| ACCOUNT\_MARGIN\_SO\_CALL | Margin level, at which a deposit to an account is required (Margin Call). Depending on ACCOUNT\_MARGIN\_SO\_MODE, <br>the property is set either in % or deposit currency | double |
| ACCOUNT\_MARGIN\_SO\_SO | Margin level, at which the most loss-making position is closed (Stop Out). Depending on ACCOUNT\_MARGIN\_SO\_MODE, <br>the property is set either in % or deposit currency | double |
| ACCOUNT\_MARGIN\_INITIAL | Funds reserved on an account to ensure a guarantee amount for all pending orders | double |
| ACCOUNT\_MARGIN\_MAINTENANCE | Funds reserved on an account to ensure a minimum amount for all open positions | double |
| ACCOUNT\_ASSETS | Current assets on an account | double |
| ACCOUNT\_LIABILITIES | Current liabilities on an account | double |
| ACCOUNT\_COMMISSION\_BLOCKED | Current sum of blocked commissions on an account | double |

For the [AccountInfoString()](https://www.mql5.com/en/docs/account/accountinfostring) function

ENUM\_ACCOUNT\_INFO\_STRING

|     |     |     |
| --- | --- | --- |
| ID | Description | Property type |
| ACCOUNT\_NAME | Client name | string |
| ACCOUNT\_SERVER | Trade server name | string |
| ACCOUNT\_CURRENCY | Deposit currency | string |
| ACCOUNT\_COMPANY | Name of a company serving an account | string |

The account object will feature all these properties, which are to be set in the class constructor.

In the **Defines.mqh** file of the library, add integer,


real and string
properties of the account object corresponding to the account property tables displayed above.

Since we have
already created the enumerations for working with account events, it would be reasonable to place account property data before the
previously created

data for working with account events:

```
//+------------------------------------------------------------------+
//| Data for working with accounts                                   |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Account integer properties                                       |
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
   ACCOUNT_PROP_CURRENCY_DIGITS                             // Number of digits for an account currency necessary for accurate display of trading results
  };
#define ACCOUNT_PROP_INTEGER_TOTAL    (9)                   // Total number of account's integer properties
#define ACCOUNT_PROP_INTEGER_SKIP     (0)                   // Number of account's integer properties not used in sorting
//+------------------------------------------------------------------+
//| Account real properties                                          |
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
   ACCOUNT_PROP_MARGIN_SO_CALL,                             // Margin level, at which a deposit to an account is required (Margin Call)
   ACCOUNT_PROP_MARGIN_SO_SO,                               // Margin level, at which the most loss-making position is closed (Stop Out)
   ACCOUNT_PROP_MARGIN_INITIAL,                             // Funds reserved on an account to ensure a guarantee amount for all pending orders
   ACCOUNT_PROP_MARGIN_MAINTENANCE,                         // Funds reserved on an account to ensure a minimum amount for all open positions
   ACCOUNT_PROP_ASSETS,                                     // Current assets on an account
   ACCOUNT_PROP_LIABILITIES,                                // Current liabilities on an account
   ACCOUNT_PROP_COMMISSION_BLOCKED                          // Current sum of blocked commissions on an account
  };
#define ACCOUNT_PROP_DOUBLE_TOTAL     (14)                  // Total number of account's real properties
#define ACCOUNT_PROP_DOUBLE_SKIP      (0)                   // Number of account's real properties not used in sorting
//+------------------------------------------------------------------+
//| Account string properties                                        |
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
//| Possible account sorting criteria                                |
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

   SORT_BY_ACCOUNT_BALANCE             =  FIRST_ACC_DBL_PROP,     // Sort by an account balance in the deposit currency
   SORT_BY_ACCOUNT_CREDIT              =  FIRST_ACC_DBL_PROP+1,   // Sort by credit in a deposit currency
   SORT_BY_ACCOUNT_PROFIT              =  FIRST_ACC_DBL_PROP+2,   // Sort by the current profit on an account in the deposit currency
   SORT_BY_ACCOUNT_EQUITY              =  FIRST_ACC_DBL_PROP+3,   // Sort by an account equity in the deposit currency
   SORT_BY_ACCOUNT_MARGIN              =  FIRST_ACC_DBL_PROP+4,   // Served by an account reserved margin in the deposit currency
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
//| Data for working with account events                             |
//+------------------------------------------------------------------+
```

Here all is familiar for us from the previous articles, so there is no point in dwelling on arranging enumerations and setting
unused object properties and macro substitutions for specifying the number of properties passed for an accurate calculation of an
address of the initial enumeration constant for the next type of object properties.
All that was described in the previous articles, namely in the " [Implementing \\
event handling on a netting account](https://www.mql5.com/en/articles/6383#node02)" section of the sixth part of the library description.

The only thing we can dwell on here is " [Margin \\
calculation mode](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo/caccountinfomarginmode)":

Since there is no ENUM\_ACCOUNT\_MARGIN\_MODE enumeration in MQL4, we need to specify it for
compilation in MQL4. Let's add it to the end of the

**ToMQL4.mqh** file:

```
//+------------------------------------------------------------------+
//| Margin calculation mode                                          |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_MARGIN_MODE
  {
   ACCOUNT_MARGIN_MODE_RETAIL_NETTING,
   ACCOUNT_MARGIN_MODE_EXCHANGE,
   ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
  };
//+------------------------------------------------------------------+
```

**Now when all the data has been prepared, an account object can be created.**

In the **\\MQL5\\Include\\DoEasy\\Objects\** library folder, create the **Accounts** subfolder and introduce the new **CAccount**
class in the **Account.mqh** file.

**In the newly created class file, add the declarations of all necessary methods**.

Most of these methods are
already "standard" for the library objects. However, there is a small caveat here: since this class does not imply the existence of
descendants, there is no protected class constructor in it that accepts and sets the status of an object. Therefore, the account object
features no "status" property, and its

constructor accepts no arguments. At the same time, we will leave the virtual
methods returning the flags of the object supporting a certain property for possible future class descendants:

```
//+------------------------------------------------------------------+
//|                                                      Account.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "..\..\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| Account class                                                    |
//+------------------------------------------------------------------+
class CAccount : public CObject
  {
private:
   long              m_long_prop[ACCOUNT_PROP_INTEGER_TOTAL];           // Integer properties
   double            m_double_prop[ACCOUNT_PROP_DOUBLE_TOTAL];          // Real properties
   string            m_string_prop[ACCOUNT_PROP_STRING_TOTAL];          // String properties

//--- Return the index of the array the account's (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_ACCOUNT_PROP_DOUBLE property) const { return(int)property-ACCOUNT_PROP_INTEGER_TOTAL;                          }
   int               IndexProp(ENUM_ACCOUNT_PROP_STRING property) const { return(int)property-ACCOUNT_PROP_INTEGER_TOTAL-ACCOUNT_PROP_DOUBLE_TOTAL;}
public:
//--- Constructor
                     CAccount(void);
protected:

public:
//--- Set (1) integer, (2) real and (3) string properties of an account
   void              SetProperty(ENUM_ACCOUNT_PROP_INTEGER property,long value)        { this.m_long_prop[property]=value;                                  }
   void              SetProperty(ENUM_ACCOUNT_PROP_DOUBLE property,double value)       { this.m_double_prop[this.IndexProp(property)]=value;                }
   void              SetProperty(ENUM_ACCOUNT_PROP_STRING property,string value)       { this.m_string_prop[this.IndexProp(property)]=value;                }
//--- Return (1) integer, (2) real and (3) string properties of an account from the property array
   long              GetProperty(ENUM_ACCOUNT_PROP_INTEGER property)             const { return this.m_long_prop[property];                                 }
   double            GetProperty(ENUM_ACCOUNT_PROP_DOUBLE property)              const { return this.m_double_prop[this.IndexProp(property)];               }
   string            GetProperty(ENUM_ACCOUNT_PROP_STRING property)              const { return this.m_string_prop[this.IndexProp(property)];               }

//--- Return the flag of calculating MarginCall and StopOut levels in %
   bool              IsPercentsForSOLevels(void)                                 const { return this.MarginSOMode()==ACCOUNT_STOPOUT_MODE_PERCENT;          }
//--- Return the flag of supporting the property by the account object
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_INTEGER property)               { return true; }
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_DOUBLE property)                { return true; }
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_STRING property)                { return true; }

//--- Compare CAccount objects by all possible properties (for sorting the lists by a specified account object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CAccount objects by account properties (to search for equal account objects)
   bool              IsEqual(CAccount* compared_account) const;
//+------------------------------------------------------------------+
//| Methods of a simplified access to the account object properties  |
//+------------------------------------------------------------------+
//--- Return the account's integer properties
   ENUM_ACCOUNT_TRADE_MODE    TradeMode(void)                     const { return (ENUM_ACCOUNT_TRADE_MODE)this.GetProperty(ACCOUNT_PROP_TRADE_MODE);        }
   ENUM_ACCOUNT_STOPOUT_MODE  MarginSOMode(void)                  const { return (ENUM_ACCOUNT_STOPOUT_MODE)this.GetProperty(ACCOUNT_PROP_MARGIN_SO_MODE);  }
   ENUM_ACCOUNT_MARGIN_MODE   MarginMode(void)                    const { return (ENUM_ACCOUNT_MARGIN_MODE)this.GetProperty(ACCOUNT_PROP_MARGIN_MODE);      }
   long              Login(void)                                  const { return this.GetProperty(ACCOUNT_PROP_LOGIN);                                      }
   long              Leverage(void)                               const { return this.GetProperty(ACCOUNT_PROP_LEVERAGE);                                   }
   long              LimitOrders(void)                            const { return this.GetProperty(ACCOUNT_PROP_LIMIT_ORDERS);                               }
   long              TradeAllowed(void)                           const { return this.GetProperty(ACCOUNT_PROP_TRADE_ALLOWED);                              }
   long              TradeExpert(void)                            const { return this.GetProperty(ACCOUNT_PROP_TRADE_EXPERT);                               }
   long              CurrencyDigits(void)                         const { return this.GetProperty(ACCOUNT_PROP_CURRENCY_DIGITS);                            }

//--- Return the account's real properties
   double            Balance(void)                                const { return this.GetProperty(ACCOUNT_PROP_BALANCE);                                    }
   double            Credit(void)                                 const { return this.GetProperty(ACCOUNT_PROP_CREDIT);                                     }
   double            Profit(void)                                 const { return this.GetProperty(ACCOUNT_PROP_PROFIT);                                     }
   double            Equity(void)                                 const { return this.GetProperty(ACCOUNT_PROP_EQUITY);                                     }
   double            Margin(void)                                 const { return this.GetProperty(ACCOUNT_PROP_MARGIN);                                     }
   double            MarginFree(void)                             const { return this.GetProperty(ACCOUNT_PROP_MARGIN_FREE);                                }
   double            MarginLevel(void)                            const { return this.GetProperty(ACCOUNT_PROP_MARGIN_LEVEL);                               }
   double            MarginSOCall(void)                           const { return this.GetProperty(ACCOUNT_PROP_MARGIN_SO_CALL);                             }
   double            MarginSOSO(void)                             const { return this.GetProperty(ACCOUNT_PROP_MARGIN_SO_SO);                               }
   double            MarginInitial(void)                          const { return this.GetProperty(ACCOUNT_PROP_MARGIN_INITIAL);                             }
   double            MarginMaintenance(void)                      const { return this.GetProperty(ACCOUNT_PROP_MARGIN_MAINTENANCE);                         }
   double            Assets(void)                                 const { return this.GetProperty(ACCOUNT_PROP_ASSETS);                                     }
   double            Liabilities(void)                            const { return this.GetProperty(ACCOUNT_PROP_LIABILITIES);                                }
   double            ComissionBlocked(void)                       const { return this.GetProperty(ACCOUNT_PROP_COMMISSION_BLOCKED);                         }

//--- Return the account's string properties
   string            Name(void)                                   const { return this.GetProperty(ACCOUNT_PROP_NAME);                                       }
   string            Server(void)                                 const { return this.GetProperty(ACCOUNT_PROP_SERVER);                                     }
   string            Currency(void)                               const { return this.GetProperty(ACCOUNT_PROP_CURRENCY);                                   }
   string            Company(void)                                const { return this.GetProperty(ACCOUNT_PROP_COMPANY);                                    }

//+------------------------------------------------------------------+
//| Descriptions of the account object properties                    |
//+------------------------------------------------------------------+
//--- Return the description of the account's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_ACCOUNT_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_ACCOUNT_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_ACCOUNT_PROP_STRING property);
//--- Return a name of a trading account type
   string            TradeModeDescription(void)    const;
//--- Return the description of the mode for setting the minimum available margin level
   string            MarginSOModeDescription(void) const;
//--- Return the description of the margin calculation mode
   string            MarginModeDescription(void)   const;
//--- Display the description of the account properties in the journal (full_prop=true - all properties, false - supported ones only)
   void              Print(const bool full_prop=false);
//--- Display a short account description in the journal
   void              PrintShort(void);
//---
  };
//+------------------------------------------------------------------+
```

**Write the class constructor outside the class body:**

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
   this.m_long_prop[ACCOUNT_PROP_MARGIN_MODE]                        = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_MARGIN_MODE) #else ACCOUNT_MARGIN_MODE_RETAIL_HEDGING #endif ;
   this.m_long_prop[ACCOUNT_PROP_CURRENCY_DIGITS]                    = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif ;

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

All is clear here: the appropriate account property is assigned to each object property using the [AccountInfo \\
functions](https://www.mql5.com/en/docs/account).

For the two properties not present in MQL4,
the selection is made during the conditional compilation directives: we get the appropriate properties for the

_"margin calculation mode"_ and _"number of decimal places for an account currency" properties_ for MQL5, while
for

MQL4, simply return ACCOUNT\_MARGIN\_MODE\_RETAIL\_HEDGING
(hedge account) from the [ENUM\_ACCOUNT\_MARGIN\_MODE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_margin_mode)
enumeration for the first property and two decimal places for
the second one.

**Let's implement the method for searching and sorting account objects in their collection list**.

The method
is identical to the

[previously described ones](https://www.mql5.com/en/articles/5654#node04) in the library objects. Therefore, I will
provide only its listing here:

```
//+-------------------------------------------------------------------+
//|Compare CAccount objects by all possible properties                |
//+-------------------------------------------------------------------+
int CAccount::Compare(const CObject *node,const int mode=0) const
  {
   const CAccount *account_compared=node;
//--- compare integer properties of two accounts
   if(mode<ACCOUNT_PROP_INTEGER_TOTAL)
     {
      long value_compared=account_compared.GetProperty((ENUM_ACCOUNT_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_ACCOUNT_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- comparing real properties of two accounts
   else if(mode<ACCOUNT_PROP_DOUBLE_TOTAL+ACCOUNT_PROP_INTEGER_TOTAL)
     {
      double value_compared=account_compared.GetProperty((ENUM_ACCOUNT_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_ACCOUNT_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- comparing string properties of two accounts
   else if(mode<ACCOUNT_PROP_DOUBLE_TOTAL+ACCOUNT_PROP_INTEGER_TOTAL+ACCOUNT_PROP_STRING_TOTAL)
     {
      string value_compared=account_compared.GetProperty((ENUM_ACCOUNT_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_ACCOUNT_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);

     }
   return 0;
  }
//+------------------------------------------------------------------+
```

To compare two account objects, we need to compare their **unchangeable** properties to determine whether these objects belong to
different accounts. Account number (login), user name and company name are provided for accurate identification. These are the
properties of the two compared accounts we are going to check in the account objects comparison method of the account objects:

```
//+------------------------------------------------------------------+
//| Compare CAccount objects by account properties                   |
//+------------------------------------------------------------------+
bool CAccount::IsEqual(CAccount *compared_account) const
  {
   if(this.GetProperty(ACCOUNT_PROP_COMPANY)!=compared_account.GetProperty(ACCOUNT_PROP_COMPANY) ||
      this.GetProperty(ACCOUNT_PROP_LOGIN)!=compared_account.GetProperty(ACCOUNT_PROP_LOGIN)     ||
      this.GetProperty(ACCOUNT_PROP_NAME)!=compared_account.GetProperty(ACCOUNT_PROP_NAME)
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

The pointer to a compared object is passed to the method and the
three properties of the two objects (

company name, account
number and client name) are checked. If any of the object
properties are not equal, they belong to different accounts —

return false. After
all three comparisons are successfully passed,

return true
— the objects are equal.

**Other class methods** are "service" ones and do not need to be dwelled on as their listings contain all necessary info. Besides, we
analyzed similar methods in the previous parts of the library description:

```
//+------------------------------------------------------------------+
//| Display account properties in the journal                        |
//+------------------------------------------------------------------+
void CAccount::Print(const bool full_prop=false)
  {
   ::Print("============= ",TextByLanguage("Начало списка параметров аккаунта","Beginning of Account parameter list")," ==================");
   int beg=0, end=ACCOUNT_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_ACCOUNT_PROP_INTEGER prop=(ENUM_ACCOUNT_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=ACCOUNT_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_ACCOUNT_PROP_DOUBLE prop=(ENUM_ACCOUNT_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=ACCOUNT_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_ACCOUNT_PROP_STRING prop=(ENUM_ACCOUNT_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("================== ",TextByLanguage("Конец списка параметров аккаунта","End of Account parameter list")," ==================\n");
  }
//+------------------------------------------------------------------+
//| Display the brief account description in the journal             |
//+------------------------------------------------------------------+
void CAccount::PrintShort(void)
  {
   string mode=(this.MarginMode()==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING ? ", Hedge" : this.MarginMode()==ACCOUNT_MARGIN_MODE_EXCHANGE ? ", Exhange" : "");
   string names=TextByLanguage("Счёт ","Account ")+(string)this.Login()+": "+this.Name()+" ("+this.Company()+" ";
   string values=DoubleToString(this.Balance(),(int)this.CurrencyDigits())+" "+this.Currency()+", 1:"+(string)+this.Leverage()+mode+", "+this.TradeModeDescription()+")";
   ::Print(names,values);
  }
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
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return a description of an account real property                 |
//+------------------------------------------------------------------+
string CAccount::GetPropertyDescription(ENUM_ACCOUNT_PROP_DOUBLE property)
  {
   return
     (
      property==ACCOUNT_PROP_BALANCE         ?  TextByLanguage("Баланс счета","Account balance")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())+" "+this.Currency()                                            :
      property==ACCOUNT_PROP_CREDIT          ?  TextByLanguage("Предоставленный кредит","Account credit")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())+" "+this.Currency()                                            :
      property==ACCOUNT_PROP_PROFIT          ?  TextByLanguage("Текущая прибыль на счете","Current profit of an account")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())+" "+this.Currency()                                            :
      property==ACCOUNT_PROP_EQUITY          ?  TextByLanguage("Собственные средства на счете","Account equity")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())+" "+this.Currency()                                            :
      property==ACCOUNT_PROP_MARGIN          ?
         TextByLanguage("Зарезервированные залоговые средства на счете","Account margin used in deposit currency")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())+" "+this.Currency()                                            :
      property==ACCOUNT_PROP_MARGIN_FREE     ?
         TextByLanguage("Свободные средства на счете, доступные для открытия позиции","Account free margin")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())                                                                :
      property==ACCOUNT_PROP_MARGIN_LEVEL    ?  TextByLanguage("Уровень залоговых средств на счете в процентах","Account margin level in percentage")+": "+
                                                   ::DoubleToString(this.GetProperty(property),1)+"%"                                                                                     :
      property==ACCOUNT_PROP_MARGIN_SO_CALL  ?  TextByLanguage("Уровень залоговых средств для наступления Margin Call","Margin call level")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(this.IsPercentsForSOLevels() ? 1 : (int)this.CurrencyDigits()))+
                                                   (this.IsPercentsForSOLevels() ? "%" : this.Currency())                                                                                 :
      property==ACCOUNT_PROP_MARGIN_SO_SO    ?  TextByLanguage("Уровень залоговых средств для наступления Stop Out","Margin stop out level")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(this.IsPercentsForSOLevels() ? 1 : (int)this.CurrencyDigits()))+
                                                   (this.IsPercentsForSOLevels() ? "%" : this.Currency())                                                                                 :
      property==ACCOUNT_PROP_MARGIN_INITIAL  ?
         TextByLanguage("Зарезервированные средства для обеспечения гарантийной суммы по всем отложенным ордерам","Amount reserved on account to cover margin of all pending orders ")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())                                                                :
      property==ACCOUNT_PROP_MARGIN_MAINTENANCE  ?
         TextByLanguage("Зарезервированные средства для обеспечения минимальной суммы по всем открытым позициям","Min equity reserved on account to cover min amount of all open positions")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())                                                                :
      property==ACCOUNT_PROP_ASSETS          ?  TextByLanguage("Текущий размер активов на счёте","Current account assets")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())                                                                :
      property==ACCOUNT_PROP_LIABILITIES     ?  TextByLanguage("Текущий размер обязательств на счёте","Current liabilities on account")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())                                                                :
      property==ACCOUNT_PROP_COMMISSION_BLOCKED ?
         TextByLanguage("Сумма заблокированных комиссий по счёту","Currently blocked commission amount on account")+": "+
                                                   ::DoubleToString(this.GetProperty(property),(int)this.CurrencyDigits())                                                                :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return a description of an account string property               |
//+------------------------------------------------------------------+
string CAccount::GetPropertyDescription(ENUM_ACCOUNT_PROP_STRING property)
  {
   return
     (
      property==ACCOUNT_PROP_NAME      ?  TextByLanguage("Имя клиента","Client name")+": \""+this.GetProperty(property)+"\""                                                     :
      property==ACCOUNT_PROP_SERVER    ?  TextByLanguage("Имя торгового сервера","Trade server name")+": \""+this.GetProperty(property)+"\""                                     :
      property==ACCOUNT_PROP_CURRENCY  ?  TextByLanguage("Валюта депозита","Account currency")+": \""+this.GetProperty(property)+"\""                                            :
      property==ACCOUNT_PROP_COMPANY   ?  TextByLanguage("Имя компании, обслуживающей счет","Name of company that serves account")+": \""+this.GetProperty(property)+"\""  :
      ""
     );
  }
//+------------------------------------------------------------------+
//| Return a trading account type name                               |
//+------------------------------------------------------------------+
string CAccount::TradeModeDescription(void) const
  {
   return
     (
      this.TradeMode()==ACCOUNT_TRADE_MODE_DEMO    ? TextByLanguage("Демонстрационный счёт","Demo account") :
      this.TradeMode()==ACCOUNT_TRADE_MODE_CONTEST ? TextByLanguage("Конкурсный счёт","Contest account")    :
      this.TradeMode()==ACCOUNT_TRADE_MODE_REAL    ? TextByLanguage("Реальный счёт","Real account")         :
      TextByLanguage("Неизвестный тип счёта","Unknown account type")
     );
  }
//+------------------------------------------------------------------+
//| Return a description of a mode for setting                       |
//| minimum available margin level                                   |
//+------------------------------------------------------------------+
string CAccount::MarginSOModeDescription(void) const
  {
   return
     (
      this.MarginSOMode()==ACCOUNT_STOPOUT_MODE_PERCENT  ? TextByLanguage("Уровень задается в процентах","Account stop out mode in percentage") :
      TextByLanguage("Уровень задается в деньгах","Account stop out mode in money")
     );
  }
//+------------------------------------------------------------------+
//| Return a description of a margin calculation mode                |
//+------------------------------------------------------------------+
string CAccount::MarginModeDescription(void) const
  {
   return
     (
      this.MarginMode()==ACCOUNT_MARGIN_MODE_RETAIL_NETTING ? TextByLanguage("Внебиржевой рынок в режиме \"Неттинг\"","Netting mode") :
      this.MarginMode()==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING ? TextByLanguage("Внебиржевой рынок в режиме \"Хеджинг\"","Hedging mode") :
      TextByLanguage("Биржевой рынок","Exchange markets mode")
     );
  }
//+------------------------------------------------------------------+
```

Find the full listing of the account class in the files attached to the article. Let's test the class.

### Testing the account object

To check if the class receives account data correctly, **temporarily** includethe class file to the library's main object — CEngine
class:

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Services\TimerCounter.mqh"
#include "Objects\Accounts\Account.mqh"
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
```

After including the account object class file, the program is able to see the object.

For class testing purposes, take the EA from the [previous article](https://www.mql5.com/en/articles/6921#node03)
— \\MQL5\\Experts\\TestDoEasy\\Part11\\TestDoEasyPart11.mq5 and save it under the name **TestDoEasyPart12\_1.mq5** in the
folder

**\\MQL5\\Experts\\TestDoEasy\\Part12**.

To include and test the account object, add strings to the OnInit()
handler of the EA (the check is to be performed during the initialization):

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal,
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
      return INIT_FAILED;
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
//--- Fast check of the account object
   CAccount* acc=new CAccount();
   if(acc!=NULL)
     {
      acc.PrintShort();
      acc.Print();
      delete acc;
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Here create the account object and, if successful, display a
brief account data and the full list of account parameters
afterwards in the journal. Remove the account object upon
completion.

Launch the EA on the chart of any symbol and see the Experts logs:

![](https://c.mql5.com/2/36/2019-05-23_15-50-18.gif)

All account data is displayed correctly.

### Collection of account objects

Since all EAs are re-initialized when changing an account, destructors are called first followed by class constructors. Thus, the EA
loses the previous account object present before changing the account. To maintain the account collection, we need to remember data of
the previous accounts the terminal connected to. To do this, save data of the current account object to the file in the account
collection class destructor and download data from the files in the class constructor. Thus, the account collection is filled with
data on all accounts the terminal connected to during the program operation based on the library. Since the files are stored in the
common folder of all client terminals, each launched terminal sees all accounts the terminal on the PC connected to, provided that the
library-based program works in that terminal.



The new account collection class will have the ability to compare data on all existing accounts by their different parameters.

To save the account object to the file, we need to create the method for saving data to a file in the CAccount class.

All created object are
inherited from

[CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) — the base object of the [standard \\
library](https://www.mql5.com/en/docs/standardlibrary). The class also features the virtual methods of saving and loading
an object to a file:

```
//+------------------------------------------------------------------+
//| Class CObject.                                                   |
//| Purpose: Base class for storing elements.                        |
//+------------------------------------------------------------------+
class CObject
  {
private:
   CObject          *m_prev;               // previous item of list
   CObject          *m_next;               // next item of list

public:
                     CObject(void): m_prev(NULL),m_next(NULL)            {                 }
                    ~CObject(void)                                       {                 }
   //--- methods to access protected data
   CObject          *Prev(void)                                    const { return(m_prev); }
   void              Prev(CObject *node)                                 { m_prev=node;    }
   CObject          *Next(void)                                    const { return(m_next); }
   void              Next(CObject *node)                                 { m_next=node;    }
   //--- methods for working with files
   virtual bool      Save(const int file_handle)                         { return(true);   }
   virtual bool      Load(const int file_handle)                         { return(true);   }
   //--- method of identifying the object
   virtual int       Type(void)                                    const { return(0);      }
   //--- method of comparing the objects
   virtual int       Compare(const CObject *node,const int mode=0) const { return(0);      }
  };
//+------------------------------------------------------------------+
```

The methods do nothing, and we need to redefine them in our descendant classes where they are needed (CAccount class).

To save all account object properties to the file, we will use a simple structure and save it to the file. However, the object fields contain
lines, which means this is not a POD structure. Thus, we need to convert all string properties of the object when saving them to the uchar
arrays of the structure fields with the constant size. In this case, we will be able to save all data on the account object properties to the
file as a structure using the

[FileWriteArray()](https://www.mql5.com/en/docs/files/filewritearray) function.

**To create the directory for storing the library files and the constant size of uchar arrays, create macro substitutions in the**
**Defines.mqh files:**

```
#define DIRECTORY                   ("DoEasy\\")               // Library directory for placing class object folders
#define UCHAR_ARRAY_SIZE            (64)                       // Size of uchar arrays for storing string properties
```

Since the length of the comment line is limited to 64 characters, the array size is created to fit this value. Further on, we may need to save order
objects to files, and the length less than 64 symbols may turn out to be unsuitable. It may well turn out that a larger string size is allocated
for account string properties. If the test shows the size is insufficient for storing names of companies serving the account, it can always
be increased.

**Create the necessary structure for saving account object properties**
**and**

**class member variables for working with files in the private**
**section of the CAccount class:**

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
      //--- Account real properties
      double         balance;                      // ACCOUNT_BALANCE (Account balance in the deposit currency)
      double         credit;                       // ACCOUNT_CREDIT (Credit in the deposit currency)
      double         profit;                       // ACCOUNT_PROFIT (Current profit on an account in the deposit currency)
      double         equity;                       // ACCOUNT_EQUITY (Equity on an account in the deposit currency)
      double         margin;                       // ACCOUNT_MARGIN (Reserved margin on an account in the deposit currency)
      double         margin_free;                  // ACCOUNT_MARGIN_FREE (Free funds available for opening a position in the deposit currency)
      double         margin_level;                 // ACCOUNT_MARGIN_LEVEL (Margin level on an account in %)
      double         margin_so_call;               // ACCOUNT_MARGIN_SO_CALL (Margin Call level)
      double         margin_so_so;                 // ACCOUNT_MARGIN_SO_SO (StopOut level)
      double         margin_initial;               // ACCOUNT_MARGIN_INITIAL (Funds reserved on an account to ensure a guarantee amount for all pending orders)
      double         margin_maintenance;           // ACCOUNT_MARGIN_MAINTENANCE (Funds reserved on an account to ensure a minimum amount for all open positions)
      double         assets;                       // ACCOUNT_ASSETS (Current assets on an account)
      double         liabilities;                  // ACCOUNT_LIABILITIES (Current liabilities on an account)
      double         comission_blocked;            // ACCOUNT_COMMISSION_BLOCKED (Current sum of blocked commissions on an account)
      //--- Account string properties
      uchar          name[UCHAR_ARRAY_SIZE];       // ACCOUNT_NAME (Client name)
      uchar          server[UCHAR_ARRAY_SIZE];     // ACCOUNT_SERVER (Trade server name)
      uchar          currency[UCHAR_ARRAY_SIZE];   // ACCOUNT_CURRENCY (Deposit currency)
      uchar          company[UCHAR_ARRAY_SIZE];    // ACCOUNT_COMPANY (Name of a company serving an account)
     };
   SData             m_struct_obj;                                      // Account object structure
   uchar             m_uchar_array[];                                   // uchar array of the account object structure


//--- Object properties
   long              m_long_prop[ACCOUNT_PROP_INTEGER_TOTAL];           // Integer properties
   double            m_double_prop[ACCOUNT_PROP_DOUBLE_TOTAL];          // Real properties
   string            m_string_prop[ACCOUNT_PROP_STRING_TOTAL];          // String properties

//--- Return the array of the index the account (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_ACCOUNT_PROP_DOUBLE property) const { return(int)property-ACCOUNT_PROP_INTEGER_TOTAL;                                   }
   int               IndexProp(ENUM_ACCOUNT_PROP_STRING property) const { return(int)property-ACCOUNT_PROP_INTEGER_TOTAL-ACCOUNT_PROP_DOUBLE_TOTAL;         }
protected:
//--- Create (1) the account object structure and (2) the account object from the structure
   bool              ObjectToStruct(void);
   void              StructToObject(void);
public:
```

According to the listing, two methods are additionally declared in the protected section of the CAccount class — the
one for creating the structure from the object property fields and a reverse method for
creating the account object from the structure.

The first method is to be used for writing the account object to the file, while the second one is to be used for reading from the file.

**Let's write the methods outside the class body:**

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
   //--- Saving the structure to the uchar array
   ::ResetLastError();
   if(!::StructToCharArray(this.m_struct_obj,this.m_uchar_array))
     {
      ::Print(DFUN,TextByLanguage("Не удалось сохранить структуру объекта в uchar-массив, ошибка ","Failed to save object structure to uchar array, error "),(string)::GetLastError());
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

As we can see in the listing, all integer and real object properties are saved in the structure fields of the same name. To
save the string properties,

[convert the string to the uchar array](https://www.mql5.com/en/docs/convert/stringtochararray) and save it
in the appropriate structure fields.

After saving the object properties, the entire structure [is \\
saved to the uchar array](https://www.mql5.com/en/docs/convert/structtochararray), which is then saved to the file.

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

The method of the reverse transformation of the structure fields into the account object properties is almost identical to the first one
discussed above.

Here, the object account string properties are obtained by [converting \\
the uchar arrays of the structure to strings](https://www.mql5.com/en/docs/convert/chararraytostring).

**Now all is ready to create the virtual methods of saving and**
**uploading/downloading the object to/from the file.**

In the public section of the CAccount class, declare the Save()
and Load() virtual methods:

```
public:
//--- Constructor
                     CAccount(void);
//--- Set (1) integer, (2) real and (3) string account properties
   void              SetProperty(ENUM_ACCOUNT_PROP_INTEGER property,long value)        { this.m_long_prop[property]=value;                                  }
   void              SetProperty(ENUM_ACCOUNT_PROP_DOUBLE property,double value)       { this.m_double_prop[this.IndexProp(property)]=value;                }
   void              SetProperty(ENUM_ACCOUNT_PROP_STRING property,string value)       { this.m_string_prop[this.IndexProp(property)]=value;                }
//--- Return (1) integer, (2) real and (3) string account properties from the properties array
   long              GetProperty(ENUM_ACCOUNT_PROP_INTEGER property)             const { return this.m_long_prop[property];                                 }
   double            GetProperty(ENUM_ACCOUNT_PROP_DOUBLE property)              const { return this.m_double_prop[this.IndexProp(property)];               }
   string            GetProperty(ENUM_ACCOUNT_PROP_STRING property)              const { return this.m_string_prop[this.IndexProp(property)];               }

//--- Return the flag of the MarginCall and StopOut levels calculation in %
   bool              IsPercentsForSOLevels(void)                                 const { return this.MarginSOMode()==ACCOUNT_STOPOUT_MODE_PERCENT;          }
//--- Return the flag of the order supporting the property
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_INTEGER property)               { return true; }
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_DOUBLE property)                { return true; }
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_STRING property)                { return true; }

//--- Compare CAccount objects with one another by all possible properties (for sorting the lists by a specified account object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CAccount objects by account properties (to search for equal account objects)
   bool              IsEqual(CAccount* compared_account) const;
//--- (1) Save the account object to the file, (2), download the account object from the file
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
//+------------------------------------------------------------------+
//| Methods of simplified access to the account object properties    |
//+------------------------------------------------------------------+
```

**Let's write the methods of saving the account object to the file and downloading it from the file:**

```
//+------------------------------------------------------------------+
//| Save the account object to the file                              |
//+------------------------------------------------------------------+
bool CAccount::Save(const int file_handle)
  {
   if(!this.ObjectToStruct())
     {
      Print(DFUN,TextByLanguage("Не удалось создать структуру объекта.","Could not create object structure"));
      return false;
     }
   if(::FileWriteArray(file_handle,this.m_uchar_array)==0)
     {
      Print(DFUN,TextByLanguage("Не удалось записать uchar-массив в файл.","Could not write uchar array to file"));
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

where:

- the handle of the file already opened for writing is passed to the method,




- save all the object fields to the POD structure,
- write the POD structure to the file whose handle
is received by the method


**The method of downloading the object data from the file:**

```
//+------------------------------------------------------------------+
//| Download the account object from the file                        |
//+------------------------------------------------------------------+
bool CAccount::Load(const int file_handle)
  {
   if(::FileReadArray(file_handle,this.m_uchar_array)==0)
     {
      Print(DFUN,TextByLanguage("Не удалось загрузить uchar-массив из файла.","Could not load uchar array from file"));
      return false;
     }
   if(!::CharArrayToStruct(this.m_struct_obj,this.m_uchar_array))
     {
      Print(DFUN,TextByLanguage("Не удалось создать структуру объекта из uchar-массива.","Could not create object structure from uchar array"));
      return false;
     }
   this.StructToObject();
   return true;
  }
//+------------------------------------------------------------------+
```

where:

- the handle of the file previously opened for reading is passed to the
method
- upload file
data to the uchar array
- save array data to the POD structure
- write POD structure data to the object fields

**We have improved the account object for uploading/downloading data to/from the file.**

The operation of the account collection is to be arranged the following way: when launching the execution program, the current
account is checked, the account object featuring the current account data is created and placed to the account collection list.
Next, we are going to view the folder with the files of the previously saved accounts. If it contains files, we will read them one by
one checking if consistency with the current account is maintained and placing them to the account collection list. After
creating the list, check the status of the current account in the timer and record the occurred changes if any.



For some changes, we are going to create events and send them to the program to control changes in the account parameters. For
example, a sudden change in a leverage is a very tangible and unpleasant event a user and his/her program should be notified of in
time.



Since we need to work in the timer, as well as create a new collection list, we will create macro substitutions with the timer
parameters and list ID for them in the **Defines.mqh** file. Also, make sure to change the names of the previously created macro substitutions for the orders, deals and
positions collection timer (add "

ORD" to their name to distinguish between macro substitutions
belonging to different collection timers). Set the pause for updating account data to

one second. I think, this will be enough for tracking changes and
decreasing a load on the system:

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- Describe the function with the error line number
#define DFUN_ERR_LINE               (__FUNCTION__+(TerminalInfoString(TERMINAL_LANGUAGE)=="Russian" ? ", Page " : ", Line ")+(string)__LINE__+": ")
#define DFUN                        (__FUNCTION__+": ")        // "Function description"
#define COUNTRY_LANG                ("Russian")                // Country language
#define END_TIME                    (D'31.12.3000 23:59:59')   // End date for requesting account history data
#define TIMER_FREQUENCY             (16)                       // Minimal frequency of the library timer in milliseconds
//--- Parameters of orders and deals collection timer
#define COLLECTION_ORD_PAUSE        (250)                      // Orders and deals collection timer pause in milliseconds
#define COLLECTION_ORD_COUNTER_STEP (16)                       // Increment of the orders and deals collection timer counter
#define COLLECTION_ORD_COUNTER_ID   (1)                        // Orders and deals collection timer counter ID
//--- Parameters of the account collection timer
#define COLLECTION_ACC_PAUSE        (1000)                     // Account collection timer pause in milliseconds
#define COLLECTION_ACC_COUNTER_STEP (16)                       // Account timer counter increment
#define COLLECTION_ACC_COUNTER_ID   (2)                        // Account timer counter ID
//--- Collection list IDs
#define COLLECTION_HISTORY_ID       (0x7778+1)                 // Historical collection list ID
#define COLLECTION_MARKET_ID        (0x7778+2)                 // Market collection list ID
#define COLLECTION_EVENTS_ID        (0x7778+3)                 // Event collection list ID
#define COLLECTION_ACCOUNT_ID       (0x7778+4)                 // Account collection list ID
//--- Data parameters for file operations
#define DIRECTORY                   ("DoEasy\\")               // Library directory for storing object folders
#define UCHAR_ARRAY_SIZE            (64)                       // Size of uchar arrays for storing string properties
```

In the CEngine class text, replace COLLECTION\_PAUSE, COLLECTION\_COUNTER\_STEP and COLLECTION\_COUNTER\_ID with the appropriate new
macro substitution names: COLLECTION\_ORD\_PAUSE, COLLECTION\_ORD\_COUNTER\_STEP and COLLECTION\_ORD\_COUNTER\_ID.

Since we create the account collection, this implies the ability to compare the properties of several account objects. To do this, add
selection and sorting methods to the account collection in the CSelect class for selecting objects fitting the criterion. The class
was

[described in the third part of the library description](https://www.mql5.com/en/articles/5687#node01).

**Open the Select.mqh file selected in the service class folder of the \\MQL5\\Include\\DoEasy\\Services library, connect**
**the file with the account class to it and add the new methods for**
**working with account objects:**

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\Event.mqh"
#include "..\Objects\Accounts\Account.mqh"
//+------------------------------------------------------------------+
//| Storage list                                                     |
//+------------------------------------------------------------------+
CArrayObj   ListStorage; // Storage object for storing sorted collection lists
//+------------------------------------------------------------------+
//| Class for sorting objects meeting the criterion                  |
//+------------------------------------------------------------------+
class CSelect
  {
private:
   //--- Two values comparison method
   template<typename T>
   static bool       CompareValues(T value1,T value2,ENUM_COMPARER_TYPE mode);
public:
//+------------------------------------------------------------------+
//| Methods of working with orders                                   |
//+------------------------------------------------------------------+
   //--- Return the list of orders with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the order index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property);
   //--- Return the order index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with events                                   |
//+------------------------------------------------------------------+
   //--- Return the list of events with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the event index with the maximum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property);
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property);
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property);
   //--- Return the event index with the minimum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property);
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property);
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with accounts                                 |
//+------------------------------------------------------------------+
   //--- Return the list of accounts with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the event index with the maximum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property);
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property);
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property);
   //--- Return the event index with the minimum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property);
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property);
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property);
  };
//+------------------------------------------------------------------+
```

**Implement declared methods outside the class body:**

```
//+------------------------------------------------------------------+
//| Methods of working with account lists                            |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of accounts with one integer                     |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CAccount *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of accounts with one real                        |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CAccount *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of accounts with one string                      |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CAccount *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the account index in the list                             |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CAccount *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CAccount *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the account index in the list                             |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CAccount *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CAccount *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the account index in the list                             |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CAccount *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CAccount *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the account index in the list                             |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindAccountMin(CArrayObj* list_source,ENUM_ACCOUNT_PROP_INTEGER property)
  {
   int index=0;
   CAccount* min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      CAccount* obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the account index in the list                             |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindAccountMin(CArrayObj* list_source,ENUM_ACCOUNT_PROP_DOUBLE property)
  {
   int index=0;
   CAccount* min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      CAccount* obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the account index in the list                             |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindAccountMin(CArrayObj* list_source,ENUM_ACCOUNT_PROP_STRING property)
  {
   int index=0;
   CAccount* min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++){
      CAccount* obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
      }
   return index;
  }
//+------------------------------------------------------------------+
```

The work of the methods was considered in the third part of the library description, so we will not dwell on their description here. You can
always

[get back to the necessary article if necessary](https://www.mql5.com/en/articles/5687#node01).

**Let's create a workpiece of the account collection class.**

In the **MQL5\\Include\\DoEasy\\Collections\** library file, create the new **AccountsCollection.mqh** class
file,

include the necessary class files and fill it with the standard
methods right away:

```
//+------------------------------------------------------------------+
//|                                           AccountsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Accounts\Account.mqh"
//+------------------------------------------------------------------+
//| Account class                                                    |
//+------------------------------------------------------------------+
class CAccountsCollection : public CListObj
  {
private:
   CListObj          m_list_accounts;                 // List of account objects
public:
//--- Return the full event collection list "as is"
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_accounts;                                            }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);   }
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);   }
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);   }
//--- Constructor
                     CAccountsCollection();

  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccountsCollection::CAccountsCollection(void)
  {
   this.m_list_accounts.Clear();
   this.m_list_accounts.Sort(SORT_BY_ACCOUNT_LOGIN);
   this.m_list_accounts.Type(COLLECTION_ACCOUNT_ID);
  }
//+------------------------------------------------------------------+
```

This short code in the class constructor is used to prepare the list
where account objects are to be stored:

- the list is cleared,
- the list is set to be sorted by an account number
and



- the account collection list ID is assigned to the list.


The work of the account collection class is arranged as follows: when the program is attached to a symbol chart, we have access to the
current data of a single account. We are able to track the changes of its properties and respond to their changes. The remaining accounts
can only be "tracked" in the program — their last state at the time of connection to a new account. Therefore, the account connection list
will contain the objects of all accounts we ever connected to, although we will be able to track changes of the current account only.
Besides, we will be able to compare data of all the accounts we have by any property.

To track important account properties, use the hash control — comparing the sum of all account properties as of the current time with the
sum obtained during the previous check. As soon as the sum changes, we check what exactly has changed and set the appropriate change
flag. Next, when tracking account events (changes of important account properties), the flag signals that we need to check all tracked
properties and send events about the changed properties to the program.

Let's **add all the necessary variables and class methods and analyze them afterwards:**

```
//+------------------------------------------------------------------+
//|                                           AccountsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Accounts\Account.mqh"
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
   MqlDataAccount    m_struct_curr_account;  // Account current data
   MqlDataAccount    m_struct_prev_account;  // Account previous data

   CListObj          m_list_accounts;                                   // Account object list
   string            m_folder_name;                                     // Name of a folder account objects are stored in
   int               m_index_current;                                   // Index of an account object featuring the current account data
   bool              m_is_account_event;                                // Event flag in the account data

//--- Write the current account data to the account object properties
  void               SetAccountsParams(CAccount* account);
//--- Save the current data status values of the current account as previous ones
   void              SavePrevValues(void)                                                                   { this.m_struct_prev_account=this.m_struct_curr_account;                }
//--- Check the account object presence in the collection list
   bool              IsPresent(CAccount* account);
//--- Find and return the account object index with the current account data
   int               Index(void);
public:
//--- Return the full account collection list "as is"
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_accounts;                                         }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
//--- Return the (1) current account object index, (2) occurred event flag in the account data
   int               IndexCurrentAccount(void)                                                        const { return this.m_index_current;                                          }
   bool              IsAccountEvent(void)                                                             const { return this.m_is_account_event;                                       }

//--- Constructor, destructor
                     CAccountsCollection();
                    ~CAccountsCollection();
//--- Add the account object to the list
   bool              AddToList(CAccount* account);
//--- (1) Save account objects from the list to the files
//--- (2) Save account objects from the files to the list
   bool              SaveObjects(void);
   bool              LoadObjects(void);
//--- Update the current account data
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

The class private section features the MqlDataAccount structure for
storing important account properties. It is to store all tracked account object properties. We have two variables having the
structure type: the first one

stores the current account data, while the other stores
previous data. The only property remaining unchanged in the structure is a **login** storing the account
number. The field value is to be used to define the first launch. If the structure in the 'login' field contains zero, this is the first
launch, and the current account status should be saved as the previous one for their subsequent comparison. In the structure's

hash field, set the sum of the values of all structure fields and
compare it with the value set in the structure of the "previous" account status. If a mismatch between the values of these two fields is
detected, a change in the account object properties is considered to be detected.

Since the account collection list is to store the data of different accounts (all accounts we connected to during the library-based
program operation and the current one), while we cannot track account data saved in the list by reading from the file, we need to know the
exact

index of the account object in the list, which is the current account object
we need to track. This index is to be used to obtain the account object and check the status of its properties in the timer. We also have the

class member variable, which is to be used as the flag of changing the account
object properties, and the variable to feature the address of the
folder in the library directory where we are going to store the class objects.

The same private section features the four methods. Let's have a look at their implementation.

**The method for writing data of the current account to the account object properties:**

```
//+------------------------------------------------------------------+
//| Write the current account data to the account object properties  |
//+------------------------------------------------------------------+
void CAccountsCollection::SetAccountsParams(CAccount *account)
  {
   if(account==NULL)
      return;
//--- Account number
   this.m_struct_curr_account.login=account.Login();
//--- Leverage
   account.SetProperty(ACCOUNT_PROP_LEVERAGE,::AccountInfoInteger(ACCOUNT_LEVERAGE));
   this.m_struct_curr_account.leverage=account.Leverage();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.leverage;
//--- Maximum allowed number of active pending orders
   account.SetProperty(ACCOUNT_PROP_LIMIT_ORDERS,::AccountInfoInteger(ACCOUNT_LIMIT_ORDERS));
   this.m_struct_curr_account.limit_orders=(int)account.LimitOrders();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.limit_orders;
//--- Permission to trade for the current account from the server side
   account.SetProperty(ACCOUNT_PROP_TRADE_ALLOWED,::AccountInfoInteger(ACCOUNT_TRADE_ALLOWED));
   this.m_struct_curr_account.trade_allowed=account.TradeAllowed();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.trade_allowed;
//--- Permission to trade for an EA from the server side
   account.SetProperty(ACCOUNT_PROP_TRADE_EXPERT,::AccountInfoInteger(ACCOUNT_TRADE_EXPERT));
   this.m_struct_curr_account.trade_expert=account.TradeExpert();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.trade_expert;
//--- Account balance in a deposit currency
   account.SetProperty(ACCOUNT_PROP_BALANCE,::AccountInfoDouble(ACCOUNT_BALANCE));
   this.m_struct_curr_account.balance=account.Balance();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.balance;
//--- Credit in a deposit currency
   account.SetProperty(ACCOUNT_PROP_CREDIT,::AccountInfoDouble(ACCOUNT_CREDIT));
   this.m_struct_curr_account.credit=account.Credit();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.credit;
//--- Current profit on an account in the account currency
   account.SetProperty(ACCOUNT_PROP_PROFIT,::AccountInfoDouble(ACCOUNT_PROFIT));
   this.m_struct_curr_account.profit=account.Profit();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.profit;
//--- Equity on an account in the deposit currency
   account.SetProperty(ACCOUNT_PROP_EQUITY,::AccountInfoDouble(ACCOUNT_EQUITY));
   this.m_struct_curr_account.equity=account.Equity();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.equity;
//--- Reserved margin on an account in a deposit currency
   account.SetProperty(ACCOUNT_PROP_MARGIN,::AccountInfoDouble(ACCOUNT_MARGIN));
   this.m_struct_curr_account.margin=account.Margin();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.margin;
//--- Free funds available for opening a position in a deposit currency
   account.SetProperty(ACCOUNT_PROP_MARGIN_FREE,::AccountInfoDouble(ACCOUNT_MARGIN_FREE));
   this.m_struct_curr_account.margin_free=account.MarginFree();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.margin_free;
//--- Margin level on an account in %
   account.SetProperty(ACCOUNT_PROP_MARGIN_LEVEL,::AccountInfoDouble(ACCOUNT_MARGIN_LEVEL));
   this.m_struct_curr_account.margin_level=account.MarginLevel();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.margin_level;
//--- Margin Call level
   account.SetProperty(ACCOUNT_PROP_MARGIN_SO_CALL,::AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL));
   this.m_struct_curr_account.margin_so_call=account.MarginSOCall();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.margin_so_call;
//--- StopOut level
   account.SetProperty(ACCOUNT_PROP_MARGIN_SO_SO,::AccountInfoDouble(ACCOUNT_MARGIN_SO_SO));
   this.m_struct_curr_account.margin_so_so=account.MarginSOSO();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.margin_so_so;
//--- Funds reserved on an account to ensure a guarantee amount for all pending orders
   account.SetProperty(ACCOUNT_PROP_MARGIN_INITIAL,::AccountInfoDouble(ACCOUNT_MARGIN_INITIAL));
   this.m_struct_curr_account.margin_initial=account.MarginInitial();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.margin_initial;
//--- Funds reserved on an account to ensure a minimum amount for all open positions
   account.SetProperty(ACCOUNT_PROP_MARGIN_MAINTENANCE,::AccountInfoDouble(ACCOUNT_MARGIN_MAINTENANCE));
   this.m_struct_curr_account.margin_maintenance=account.MarginMaintenance();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.margin_maintenance;
//--- Current assets on an account
   account.SetProperty(ACCOUNT_PROP_ASSETS,::AccountInfoDouble(ACCOUNT_ASSETS));
   this.m_struct_curr_account.assets=account.Assets();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.assets;
//--- Current liabilities on an account
   account.SetProperty(ACCOUNT_PROP_LIABILITIES,::AccountInfoDouble(ACCOUNT_LIABILITIES));
   this.m_struct_curr_account.liabilities=account.Liabilities();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.liabilities;
//--- Current sum of blocked commissions on an account
   account.SetProperty(ACCOUNT_PROP_COMMISSION_BLOCKED,::AccountInfoDouble(ACCOUNT_COMMISSION_BLOCKED));
   this.m_struct_curr_account.comission_blocked=account.ComissionBlocked();
   this.m_struct_curr_account.hash_sum+=(double)this.m_struct_curr_account.comission_blocked;
  }
//+------------------------------------------------------------------+
```

Let's have a look at the leverage using the update as an example:

the
method receives the pointer to the account object, the current
account data is added to the account object fields and the fields of
the structure featuring the account current status. Next, the value
of each obtained property is added to the hash.

The **SavePrevValues()** method for saving the structure of the current account status to the previous status structure
simply copies the current status structure to the previous status one.



**The method for checking the presence of the account object in the collection list:**

```
//+------------------------------------------------------------------+
//| Check the presence of the account object in the collection list  |
//+------------------------------------------------------------------+
bool CAccountsCollection::IsPresent(CAccount *account)
  {
   int total=this.m_list_accounts.Total();
   if(total==0)
      return false;
   for(int i=0;i<total;i++)
     {
      CAccount* check=this.m_list_accounts.At(i);
      if(check==NULL)
         continue;
      if(check.IsEqual(account))
         return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to the account object
whose data should be found in the collection list. The search is performed by an account number, as well as client and company names by the

**IsEqual()** method I have previously described when creating the account object class.

Use
the account object list (in a loop)to obtain the object from the list
and compare its data with the data of the object passed to the method.


If the data matches, return

true.

Otherwise, return


false if no equal objects are found upon the loop completion.

**The method returning the account object index in the list featuring the current account data:**

```
//+------------------------------------------------------------------+
//| Return the account object index with the current account data    |
//+------------------------------------------------------------------+
int CAccountsCollection::Index(void)
  {
   int total=this.m_list_accounts.Total();
   if(total==0)
      return WRONG_VALUE;
   for(int i=0;i<total;i++)
     {
      CAccount* account=this.m_list_accounts.At(i);
      if(account==NULL)
         continue;
      if(account.Login()==::AccountInfoInteger(ACCOUNT_LOGIN)    &&
         account.Company()==::AccountInfoString(ACCOUNT_COMPANY) &&
         account.Name()==::AccountInfoString(ACCOUNT_NAME)
        ) return i;
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

Use the account object list (in a loop)to
obtain the object and compare its account data (login, client and company names) with the account data the program is launched
at.

The loop index is returned in case of a match. Upon the loop
completion,

 -1 is returned if the object with the current account data is not found.

**The following methods are added in the public class section:**

The method returning the variable value storing the account object index
with the current account data, the method returning the flag of an
occurred account property change. Also, there is a class
destructor (to save all accounts from the list to the files), the
method adding the account object to the collection list, the methods
of saving and uploading/downloading the objects to/from the file and the
method for updating the current account data in the current account object:

```
public:
//--- Return the full account collection list "as is"
   CArrayObj        *GetList(void)                                                                          { return &this.m_list_accounts;                                         }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)   { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
   CArrayObj        *GetList(ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByAccountProperty(this.GetList(),property,value,mode);}
//--- Return the (1) current object account index, (2) flag of an occurred event in the account data
   int               IndexCurrentAccount(void)                                                        const { return this.m_index_current;                                          }
   bool              IsAccountEvent(void)                                                             const { return this.m_is_account_event;                                       }

//--- Constructor, destructor
                     CAccountsCollection();
                    ~CAccountsCollection();
//--- Add the account object to the list
   bool              AddToList(CAccount* account);
//--- (1) Save account objects from the list to the files
//--- (2) Save account objects from the files to the list
   bool              SaveObjects(void);
   bool              LoadObjects(void);
//--- Update the current account data
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

**Let's consider these methods.**

The library files are to be saved in the **Files\\DoEasy\** terminal folder featuring folders for each class (if the
class needs to save the files). There is also the

**m\_folder\_name** class member variable for setting the name of the folder storing account objects. Initialize it in the initialization
list of the class constructor right away together with the flag
variable of an occurred account properties change:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccountsCollection::CAccountsCollection(void) : m_folder_name(DIRECTORY+"Accounts"),m_is_account_event(false)
  {
   this.m_list_accounts.Clear();
   this.m_list_accounts.Sort(SORT_BY_ACCOUNT_LOGIN);
   this.m_list_accounts.Type(COLLECTION_ACCOUNT_ID);
   ::ZeroMemory(this.m_struct_prev_account);

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

Next, in the class constructor, reset the structure with the previous
data on the current account and create the folder for storing the class
files, which is to be located at "Common\_data\_folder"\\Files\\DoEasy\\Accounts for the class.

An account
object with the current account data is then created and added to the account collection list using the **AddToList()**
method. If no object is added to the list, the appropriate message is sent to the journal, otherwise the message with brief account
properties (login, client name, company name, account balance, leverage and account type if not netting) is displayed.



The next step is uploading account objects to the collection list. These are
the account objects with their save files present in the folder storing the class objects.

The last step is looking
for the object index with the current account data and assigning

**m\_index\_current** to its variable whose value is returned by the **IndexCurrentAccount()** method for
using in programs.



**The method for saving all objects from the collection list to the appropriate files is called in the class destructor:**

```
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CAccountsCollection::~CAccountsCollection(void)
  {
//--- Save account objects from the list to the files
   this.SaveObjects();
  }
//+------------------------------------------------------------------+
```

**The method for adding the account object to the collection list:**

```
//+------------------------------------------------------------------+
//| Add the account object to the list                               |
//+------------------------------------------------------------------+
bool CAccountsCollection::AddToList(CAccount *account)
  {
   if(account==NULL)
      return false;
   if(!this.IsPresent(account))
      return this.m_list_accounts.Add(account);
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to the account object,
then

the IsPresent() method is used to check the presence of such an object in the
collection list. If there is no such object yet, it is added
to the collection list and the result of its adding is returned.

**The method saving account objects from the collection list to the files:**

```
//+------------------------------------------------------------------+
//| Save account objects from the list to the files                  |
//+------------------------------------------------------------------+
bool CAccountsCollection::SaveObjects(void)
  {
   bool res=true;
   int total=this.m_list_accounts.Total();
   if(total==0)
      return false;
   for(int i=0;i<total;i++)
     {
      CAccount* account=this.m_list_accounts.At(i);
      if(account==NULL)
         continue;
      string file_name=this.m_folder_name+"\\"+account.Server()+" "+(string)account.Login()+".bin";
      if(::FileIsExist(file_name,FILE_COMMON))
         ::FileDelete(file_name,FILE_COMMON);
      ::ResetLastError();
      int handle=::FileOpen(file_name,FILE_WRITE|FILE_BIN|FILE_COMMON);
      if(handle==INVALID_HANDLE)
        {
         ::Print(DFUN,TextByLanguage("Не удалось открыть для записи файл ","Could not open file for writing: "),file_name,TextByLanguage(". Ошибка ",". Error "),(string)::GetLastError());
         return false;
        }
      res &=account.Save(handle);
      ::FileClose(handle);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

Use the collection list (in a loop)to
obtain the account object from the list and create a file name
consisting of the path to the account object folder, server name and login (account number) with the ".bin" extension. If
such a file exists in the account object folder, the file is deleted and a
new one is opened for writing. The opened file handle is passed to the
Save() virtual method of the CAccount class I have described earlier, and the file
saving result is added to the

**res** variable returning the result of writing to the file of all account objects from the collection list. The file
opened for writing is closed after saving the object.

**The method downloading account objects from the files to the collection list:**

```
//+------------------------------------------------------------------+
//| Download account objects from the files to the list              |
//+------------------------------------------------------------------+
bool CAccountsCollection::LoadObjects(void)
  {
   bool res=true;
   string name="";
   long handle_search=::FileFindFirst(this.m_folder_name+"\\*",name,FILE_COMMON);
   if(handle_search!=INVALID_HANDLE)
     {
      do
        {
         string file_name=this.m_folder_name+"\\"+name;
         ::ResetLastError();
         int handle_file=::FileOpen(m_folder_name+"\\"+name,FILE_BIN|FILE_READ|FILE_COMMON);
         if(handle_file!=INVALID_HANDLE)
           {
            CAccount* account=new CAccount();
            if(account!=NULL)
              {
               if(!account.Load(handle_file))
                 {
                  delete account;
                  ::FileClose(handle_file);
                  res &=false;
                  continue;
                 }
               if(this.IsPresent(account))
                 {
                  delete account;
                  ::FileClose(handle_file);
                  res &=false;
                  continue;
                 }
               if(!this.AddToList(account))
                 {
                  delete account;
                  res &=false;
                 }
              }
           }
         ::FileClose(handle_file);
        }
      while(::FileFindNext(handle_search,name));
      ::FileFindClose(handle_search);
     }
   return res;
  }
//+------------------------------------------------------------------+
```

First, find the very first file in the folder storing the
library account object files. Next,

open another detected file for reading in the do-while loop,


create a new account object, upload
data from the file using the

**Load()** virtual method of the **CAccount** class to it. If there is no such object (having the same account
data), the

object is added to the list. In
case of any error when uploading data to the object from the file or when adding the object to the list, make sure to remove this new object
(to avoid memory leaks) and close the file.

Upon the loop completion, the result of uploading data to the account objects
from the files and placing them to the collection list is returned.



**The method for updating the current account object data:**

```
//+------------------------------------------------------------------+
//| Update the current account data                                  |
//+------------------------------------------------------------------+
void CAccountsCollection::Refresh(void)
  {
   if(this.m_index_current==WRONG_VALUE)
      return;
   CAccount* account=this.m_list_accounts.At(this.m_index_current);
   if(account==NULL)
      return;
   ::ZeroMemory(this.m_struct_curr_account);
   this.m_is_account_event=false;
   this.SetAccountsParams(account);

//--- First launch
   if(!this.m_struct_prev_account.login)
     {
      this.SavePrevValues();
     }
//--- If the account hash changed
   if(this.m_struct_curr_account.hash_sum!=this.m_struct_prev_account.hash_sum)
     {
      this.m_is_account_event=true;
      this.SavePrevValues();
     }
  }
//+------------------------------------------------------------------+
```

Here, the first thing we do is check the validity of the index of the
account object containing the current account data. If it has not been obtained for some reason, exit
the method. Next, obtain the account object with the current account
data by its index in the list, reset the data structure of the current
account, reset the flag of changing the account object properties and call
the method for setting the account object properties. The same method copies the most recent (newly read) properties to the
current account data structure for their subsequent comparison with the account previous status and detecting the changes.



Next, define what data is set in the data structure of the account previous status. If
the login field features zero, this means the structure has never been filled, and this is the first launch. Thus, we simply fill
the structure having the previous status data with the data from the current status structure.

Next, check
the hash change by comparing the current status hash with the previous status one. If there are changes, set
the flag of the occurred account property change event and save the
current status as the previous one for their subsequent comparison.

Later, implement tracking important account status
changes and sending event messages concerning this important changes to the program.



Since the entire work with the class is performed from the library base object (CEngine class), move to the **Engine.mqh**
file and add the necessary functionality.

**First, include the account collection file:**

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Services\TimerCounter.mqh"
//+------------------------------------------------------------------+
```

**In the class private section, create the account collection object**
**and add the method for working with the account collection:**

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
   ENUM_TRADE_EVENT     m_last_trade_event;              // Account last trading event
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return (1) the first launch flag, (2) the flag presence in a trading event
   bool                 IsFirstStart(void);
//--- Working with (1) order, deal and position, as well as (2) account events
   void                 TradeEventsControl(void);
   void                 AccountEventsControl(void);
//--- Return the last (1) market pending order, (2) market order, (3) last position, (4) position by ticket
   COrder*              GetLastMarketPending(void);
   COrder*              GetLastMarketOrder(void);
   COrder*              GetLastPosition(void);
   COrder*              GetPosition(const ulong ticket);
//--- Return the last (1) removed pending order, (2) historical market order, (3) historical order (market or pending) by its ticket
   COrder*              GetLastHistoryPending(void);
   COrder*              GetLastHistoryOrder(void);
   COrder*              GetHistoryOrder(const ulong ticket);
//--- Return the (1) first and the (2) last historical market orders from the list of all position orders, (3) the last deal
   COrder*              GetFirstOrderPosition(const ulong position_id);
   COrder*              GetLastOrderPosition(const ulong position_id);
   COrder*              GetLastDeal(void);
public:
```

**In the class constructor, create a new timer counter for working**
**with the account collection:**

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),m_last_trade_event(TRADE_EVENT_NO_EVENT)
  {
   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_ORD_COUNTER_ID,COLLECTION_ORD_COUNTER_STEP,COLLECTION_ORD_PAUSE);
   this.CreateCounter(COLLECTION_ACC_COUNTER_ID,COLLECTION_ACC_COUNTER_STEP,COLLECTION_ACC_PAUSE);

   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_is_tester=::MQLInfoInteger(MQL_TESTER);
   ::ResetLastError();
   #ifdef __MQL5__
      if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
        }
   //---__MQL4__
   #else
      if(!this.IsTester() && !::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: ",(string)::GetLastError());
        }
   #endif
  }
//+------------------------------------------------------------------+
```

We have discussed the timers and their counters [in the third part of the \\
library description](https://www.mql5.com/en/articles/5687).

**In the OnTimer() class handler, add the account collection timer:**

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of historical orders, deals, market orders and positions collections
   int index=this.CounterIndex(COLLECTION_ORD_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the order, deal and position collections events
            if(counter.IsTimeDone())
               this.TradeEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
            this.TradeEventsControl();
        }
     }
//--- Account collection timer
   index=this.CounterIndex(COLLECTION_ACC_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the account collections
            if(counter.IsTimeDone())
               this.AccountEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
            this.AccountEventsControl();
        }
     }
  }
//+------------------------------------------------------------------+
```

The account collection timer works similarly to the order, deal and position collection timer discussed in the third part of the library
description (in the section devoted to developing the library base object — the

[CEngine](https://www.mql5.com/en/articles/5687#node02) class). The only difference from the order, deal and position
collection timer is calling another collection events handling method — the AccountEventsControl() one.



**Let's add the method for checking the changes in the current account properties:**

```
//+------------------------------------------------------------------+
//| Check the account events                                         |
//+------------------------------------------------------------------+
void CEngine::AccountEventsControl(void)
  {
   this.m_accounts.Refresh();
  }
//+------------------------------------------------------------------+
```

The method simply calls the Refresh() method of the CAccountsCollection class.

**In the public section of the CEngine class, write the two methods returning the event**
**and account collection lists to the program. This allows us to**
**directly access the collection lists from our programs:**

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
   CArrayObj*           GetListAllOrdersByPosID(const ulong position_id);
//--- Return the account list
   CArrayObj*           GetListAllAccounts(void)                        { return this.m_accounts.GetList();    }
//--- Return the event list
   CArrayObj*           GetListAllEvents(void)                          { return this.m_events.GetList();      }
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_events.ResetLastTradeEvent(); }
//--- Return the (1) last trading event, (2) hedge account flag, (3) flag of working in the tester
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_last_trade_event;      }
   bool                 IsHedge(void)                             const { return this.m_is_hedge;              }
   bool                 IsTester(void)                            const { return this.m_is_tester;             }
//--- Create the timer counter
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

Everything is ready for testing the account collection class. But before we start testing, let's change the **CEventsCollection::Refresh**
event collection class. Add the check to the string 233 of the
listing to eliminate occasional activations when defining events causing an old event to be sent to the program together with the new
one:

```
//--- If the event is in the account history
   if(is_history_event)
     {
      //--- If the number of historical orders increased (MQL5, MQL4)
      if(new_history_orders>0 && new_market_pendings<0)
        {
         //--- Receive the list of removed pending orders only
```

I have also fixed rather dumb errors made when writing trading functions for MQL4 for working in the MetaTrader 4 (the DELib.mqh file).
The matter is that in MQL4 the OrderSend() functions return an order ticket rather than a boolean value. Apparently, I am starting to
forget MQL4 :)

Let's consider the following example:

Checking the MQL4 function
operation result looked as follows (this is correct for MQL5):

```
if( ! OrderSend(sym,ORDER_TYPE_BUY,volume,price,deviation,sl,tp,comment,(int)magic,0,clrBlue))
```

I have fixed the error by implementing correct checks for MQL4:

```
if(OrderSend(sym,ORDER_TYPE_BUY,volume,price,deviation,sl,tp,comment,(int)magic,0,clrBlue)==WRONG_VALUE)
```

This is not a big deal for the tester but it is still an error.

I will introduce the full-fledged trading classes
soon, so these functions will be removed from the library.

### Testing the account collection

Let's use the TestDoEasyPart12\_1.mq5 EA we have already developed and save it under the name of **TestDoEasyPart12\_2.mq5** in the same
folder

**\\MQL5\\Experts\\TestDoEasy**.

In the EA inputs, introduce the variable for switching the look of the existing
account data displayed in the journal — brief (default is false) or full (true):

```
//--- input variables
input ulong    InpMagic             =  123;  // Magic number
input double   InpLots              =  0.1;  // Lots
input uint     InpStopLoss          =  50;   // StopLoss in points
input uint     InpTakeProfit        =  50;   // TakeProfit in points
input uint     InpDistance          =  50;   // Pending orders distance (points)
input uint     InpDistanceSL        =  50;   // StopLimit orders distance (points)
input uint     InpSlippage          =  0;    // Slippage in points
input double   InpWithdrawal        =  10;   // Withdrawal funds (in tester)
input uint     InpButtShiftX        =  40;   // Buttons X shift
input uint     InpButtShiftY        =  10;   // Buttons Y shift
input uint     InpTrailingStop      =  50;   // Trailing Stop (points)
input uint     InpTrailingStep      =  20;   // Trailing Step (points)
input uint     InpTrailingStart     =  0;    // Trailing Start (points)
input uint     InpStopLossModify    =  20;   // StopLoss for modification (points)
input uint     InpTakeProfitModify  =  60;   // TakeProfit for modification (points)
input bool     InpFullProperties    =  false;// Show full accounts properties
//--- global variables
```

Add the following code to the OnInit() handler:

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
      return INIT_FAILED;
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
//--- Fast check of the account object
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
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Here we get the account collection list using the GetListAllAccounts() method of the CEngine class. We obtain each subsequent object from it
in a loop displaying its properties in the journal depending on the input value — either a brief entry, or the full list of the account object
properties.

**Launch the EA and see what it displays in the journal when the brief entry is selected (Show full accounts properties = false):**

![](https://c.mql5.com/2/36/Short.png)

**Now select the full list** **— press F7 and set "Show full accounts properties" to 'true' in the parameters window:**

![](https://c.mql5.com/2/36/2019-06-01_03-44-30.gif)

Now the full list of properties for each of the existing accounts is displayed in the journal.

Note that in order to write accounts into the file, you need to connect to the first account, re-connect to the second one, then move on to the
third one and so on. In other words, previous account data is written to the file at each connection to a new account.



### What's next?

In the next article, we will track some important events of changing the account properties and start working on symbol objects and their
collection.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6952#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6952](https://www.mql5.com/ru/articles/6952)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6952.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6952/mql5.zip "Download MQL5.zip")(124.15 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/6952/mql4.zip "Download MQL4.zip")(124.15 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/320988)**
(1)


![Fabien Amazo](https://c.mql5.com/avatar/2020/7/5F07B4AD-F407.png)

**[Fabien Amazo](https://www.mql5.com/en/users/fabienamazo)**
\|
10 Jul 2020 at 00:24

Doing good so far, that's an amazing job you've done, thanks very much


![Library for easy and quick development of MetaTrader programs (part XIII): Account object events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__8.png)[Library for easy and quick development of MetaTrader programs (part XIII): Account object events](https://www.mql5.com/en/articles/6995)

The article considers working with account events for tracking important changes in account properties affecting the automated trading. We have already implemented some functionality for tracking account events in the previous article when developing the account object collection.

![Library for easy and quick development of MetaTrader programs (part XI). Compatibility with MQL4 - Position closure events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__6.png)[Library for easy and quick development of MetaTrader programs (part XI). Compatibility with MQL4 - Position closure events](https://www.mql5.com/en/articles/6921)

We continue the development of a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the tenth part, we resumed our work on the library compatibility with MQL4 and defined the events of opening positions and activating pending orders. In this article, we will define the events of closing positions and get rid of the unused order properties.

![Library for easy and quick development of MetaTrader programs (part XIV): Symbol object](https://c.mql5.com/2/36/MQL5-avatar-doeasy__9.png)[Library for easy and quick development of MetaTrader programs (part XIV): Symbol object](https://www.mql5.com/en/articles/7014)

In this article, we will create the class of a symbol object that is to be the basic object for creating the symbol collection. The class will allow us to obtain data on the necessary symbols for their further analysis and comparison.

![Optimization management (Part I): Creating a GUI](https://c.mql5.com/2/36/mql5-avatar-opt_control.png)[Optimization management (Part I): Creating a GUI](https://www.mql5.com/en/articles/7029)

This article describes the process of creating an extension for the MetaTrader terminal. The solution discussed helps to automate the optimization process by running optimizations in other terminals. A few more articles will be written concerning this topic. The extension has been developed using the C# language and design patterns, which additionally demonstrates the ability to expand the terminal capabilities by developing custom modules, as well as the ability to create custom graphical user interfaces using the functionality of a preferred programming language.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ohrzwtxicmixezpxmnhykkwyisjcxezd&ssn=1769186436435021721&ssn_dr=0&ssn_sr=0&fv_date=1769186436&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6952&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XII)%3A%20Account%20object%20class%20and%20collection%20of%20account%20objects%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691864362544199&fz_uniq=5070475739595216515&sv=2552)

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