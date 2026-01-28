---
title: Cross-Platform Expert Advisor: Reuse of Components from the MQL5 Standard Library
url: https://www.mql5.com/en/articles/2574
categories: Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:26:20.038846
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=avmjvvobhnwaaweanduvyoeheudbnjxi&ssn=1769178378088057878&ssn_dr=0&ssn_sr=0&fv_date=1769178378&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2574&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Cross-Platform%20Expert%20Advisor%3A%20Reuse%20of%20Components%20from%20the%20MQL5%20Standard%20Library%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917837850518277&fz_uniq=5068231030772528906&sv=2552)

MetaTrader 5 / Integration


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/2574#Introduction)
- [Folder Structure](https://www.mql5.com/en/articles/2574#FolderStructure)
- [CSymbolInfo](https://www.mql5.com/en/articles/2574#CSymbolInfo)
- [CSymbolManager](https://www.mql5.com/en/articles/2574#CSymbolManager)

  - [Adding Symbols](https://www.mql5.com/en/articles/2574#AddSymbols)
  - [Refreshing Symbols](https://www.mql5.com/en/articles/2574#RefreshSymbols)


- [CAccountInfo](https://www.mql5.com/en/articles/2574#CAccountInfo)
- [CExpertTrade](https://www.mql5.com/en/articles/2574#CExpertTrade)
- [CTradeManager](https://www.mql5.com/en/articles/2574#CTradeManager)


### Introduction

There exists some components in the
MQL5 Standard Library that may prove to be useful in the development
of cross-platform expert advisors. However, their incompatibilities
with the MQL4 compiler make them unusable for the MQL4 version
cross-platform expert advisors. There are at least two courses of
action in this case:

- Rewrite from scratch, keeping only
the components supported for both versions

- Copy the classes and modify them
so header files would compile in MQL4


In this article, we will explore the
second option. Using this method would make a rather loose
implementation on the MQL4 side. However, its main disadvantage is
that we don't have to rewrite a lot of code. It would also make
updating to the latest version of the MQL5 classes to be performed
easier than rewriting everything from scratch.

### Folder Structure

Unlike in most classes that will be
used, class objects reused from the MQL5 standard library will need
to be copied somewhere within the include folder of the MQL4 data
folder. Therefore, a little organization will be needed in order for
the expert advisor, regardless of version, would be able to use them.
Methods to achieve this may vary, but one of the easiest way is to
link a particular header file, and make that header file deal with
the appropriate header file that would need to be used. This is
somewhat similar to what is normally done on custom-made class
objects:

1. For the MQL5 version, the main
    header file should link to the original file location of the header
    file to be used.

2. For the MQL4 version, the main
    header file should link to a copy (modified) of the MQL5 header file
    to be used.


One way to do this is to create a “Lib”
folder within the base directory. This will be where the base header
file are to be placed. A similar folder will be created under the
MQL4 folder of the cross-platform directory (let the cross-platform
be “MQLx-Reuse”). This is where the copies (modified) of some
MQL5 classes are to be placed.

**\|-Include**

> **\|-MQLx-Reuse**

> > **\|-Base**

> > > **\|-Lib**

> > > > **<Main Header Files>**

> > **\|-MQL4**

> > > **\|-Lib**

> > > > **<Copy of MQL5 Header Files>**

> > **\|-MQL5**

The MQL5 folder will not have a “Lib”
folder since the main header files would link directly to the header
file as they are originally found within the MQL5 data folder. For
example, suppose we need to reuse [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) for a cross-platform
expert advisor. This particular class is part of the MQL5 standard
library, under trade classes. It can be access using the following
[#include](https://www.mql5.com/en/docs/basis/preprosessor/include) directive:

```
#include <Trade\SymbolInfo.mqh>
```

In order to use this on our current
setup, we need to create the main header file under the lib folder of
the Base folder, with the following code:

(/Base/Lib/SymbolInfo.mqh)

```
#ifdef __MQL5__
   #include <Trade\SymbolInfo.mqh>
#else
   #include "..\..\MQL4\Lib\SymbolInfo.mqh"
#endif
```

The MQL4 version calls an include directive
to a header file within the “Lib” folder. But this time, it would
be the “Lib” folder of the “MQL4” folder within our custom
directory (“MQLx-Reuse”).

As stated earlier, the MQL5 version
does not need a “Lib” folder since on the main header file
located on the Base folder, it would already point to
<Trade\\SymbolInfo.mqh>, which is where the class is normally
found within the MQL5 Include folder.

This is just a recommended approach,
and is by no means compulsory. However, it is helpful if one has a
separate folder used exclusively for header files that are reused or
borrowed from MQL5.

### CSymbolInfo

Using an MQL4 compiler, compiling the
original CSymbolInfo class header file would generate compilation
errors. The errors are caused mostly by incompatibilities with MQL4,
particularly with the use of functions [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) and
[SymbolInfoInteger](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger). Most of the calls to these functions can be seen
within the Refresh() method of the class. The unmodified code of the CSymbolInfo class
is shown below:

```
bool CSymbolInfo::Refresh(void)
  {
   long tmp=0;
//---
   if(!SymbolInfoDouble(m_name,SYMBOL_POINT,m_point))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_TICK_VALUE,m_tick_value))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_TICK_VALUE_PROFIT,m_tick_value_profit))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_TICK_VALUE_LOSS,m_tick_value_loss))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_TICK_SIZE,m_tick_size))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_CONTRACT_SIZE,m_contract_size))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_VOLUME_MIN,m_lots_min))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_VOLUME_MAX,m_lots_max))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_VOLUME_STEP,m_lots_step))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_VOLUME_LIMIT,m_lots_limit))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_SWAP_LONG,m_swap_long))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_SWAP_SHORT,m_swap_short))
      return(false);
   if(!SymbolInfoInteger(m_name,SYMBOL_DIGITS,tmp))
      return(false);
   m_digits=(int)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_ORDER_MODE,tmp))
      return(false);
   m_order_mode=(int)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_TRADE_EXEMODE,tmp))
      return(false);
   m_trade_execution=(ENUM_SYMBOL_TRADE_EXECUTION)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_TRADE_CALC_MODE,tmp))
      return(false);
   m_trade_calcmode=(ENUM_SYMBOL_CALC_MODE)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_TRADE_MODE,tmp))
      return(false);
   m_trade_mode=(ENUM_SYMBOL_TRADE_MODE)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_SWAP_MODE,tmp))
      return(false);
   m_swap_mode=(ENUM_SYMBOL_SWAP_MODE)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_SWAP_ROLLOVER3DAYS,tmp))
      return(false);
   m_swap3=(ENUM_DAY_OF_WEEK)tmp;
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_INITIAL,m_margin_initial))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_MAINTENANCE,m_margin_maintenance))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_LONG,m_margin_long))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_SHORT,m_margin_short))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_LIMIT,m_margin_limit))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_STOP,m_margin_stop))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_STOPLIMIT,m_margin_stoplimit))
      return(false);
   if(!SymbolInfoInteger(m_name,SYMBOL_EXPIRATION_MODE,tmp))
      return(false);
   m_trade_time_flags=(int)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_FILLING_MODE,tmp))
      return(false);
   m_trade_fill_flags=(int)tmp;
//--- succeed
   return(true);
  }
```

Similar to what was discussed in the [first article](https://www.mql5.com/en/articles/2569), we use a common header file which would, ideally, consolidate the similarities in code between the MQL4 and MQL5 versions. It is actually possible to rewrite the CSymbolInfo class into three separate files so that the similarities can be shared in a single file, and the differences be coded in the other two class files. However, in this article, we will deal with the easier (and faster) approach: copy the MQL5 CSymbolInfo class file, and then comment the lines that incompatible with MQL4. For both versions, however, the resulting file structure would look like the following:

![SymbolInfo file structure](https://c.mql5.com/2/24/csymbolinfo.png)

Some property identifiers such as
SYMBOL\_TRADE\_TICK\_VALUE\_PROFIT and SYMBOL\_TRADE\_TICK\_VALUE\_LOSS are
not supported in MQL4. There is no certainty that these would be
supported in the future. However, it is still better to put them in comments
until they are actually supported, or new developments arise which
may make a workaround on these features possible.

The following code shows the same
function, but with those commented that may cause a call to the
Refresh method to return false in MQL4:

```
bool CSymbolInfo::Refresh(void)
  {
   long tmp=0;
//---
   if(!SymbolInfoDouble(m_name,SYMBOL_POINT,m_point))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_TICK_VALUE,m_tick_value))
      return(false);
//if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_TICK_VALUE_PROFIT,m_tick_value_profit))
//return(false);
//if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_TICK_VALUE_LOSS,m_tick_value_loss))
//return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_TICK_SIZE,m_tick_size))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_TRADE_CONTRACT_SIZE,m_contract_size))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_VOLUME_MIN,m_lots_min))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_VOLUME_MAX,m_lots_max))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_VOLUME_STEP,m_lots_step))
      return(false);
//if(!SymbolInfoDouble(m_name,SYMBOL_VOLUME_LIMIT,m_lots_limit))
//return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_SWAP_LONG,m_swap_long))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_SWAP_SHORT,m_swap_short))
      return(false);
   if(!SymbolInfoInteger(m_name,SYMBOL_DIGITS,tmp))
      return(false);
   m_digits=(int)tmp;
//if(!SymbolInfoInteger(m_name,SYMBOL_ORDER_MODE,tmp))
//return(false);
//m_order_mode=(int)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_TRADE_EXEMODE,tmp))
      return(false);
   m_trade_execution=(ENUM_SYMBOL_TRADE_EXECUTION)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_TRADE_CALC_MODE,tmp))
      return(false);
   m_trade_calcmode=(ENUM_SYMBOL_CALC_MODE)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_TRADE_MODE,tmp))
      return(false);
   m_trade_mode=(ENUM_SYMBOL_TRADE_MODE)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_SWAP_MODE,tmp))
      return(false);
   m_swap_mode=(ENUM_SYMBOL_SWAP_MODE)tmp;
   if(!SymbolInfoInteger(m_name,SYMBOL_SWAP_ROLLOVER3DAYS,tmp))
      return(false);
   m_swap3=(ENUM_DAY_OF_WEEK)tmp;
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_INITIAL,m_margin_initial))
      return(false);
   if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_MAINTENANCE,m_margin_maintenance))
      return(false);
//if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_LONG,m_margin_long))
//return(false);
//if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_SHORT,m_margin_short))
//return(false);
//if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_LIMIT,m_margin_limit))
//return(false);
//if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_STOP,m_margin_stop))
//return(false);
//if(!SymbolInfoDouble(m_name,SYMBOL_MARGIN_STOPLIMIT,m_margin_stoplimit))
//return(false);
//if(!SymbolInfoInteger(m_name,SYMBOL_EXPIRATION_MODE,tmp))
//return(false);
//m_trade_time_flags=(int)tmp;
//if(!SymbolInfoInteger(m_name,SYMBOL_FILLING_MODE,tmp))
//return(false);
//m_trade_fill_flags=(int)tmp;
//--- succeed
   return(true);
  }
```

There are built-in enumerations that
are available in MQL5 but not in MQL4. For this class, the
ENUM\_SYMBOL\_CALC\_MODE and ENUM\_SYMBOL\_SWAP\_MODE would be needed for
the header file to be compiled using an MQL4 compiler. To include
these enumerations, we just need to declare them the header file of
the class:

```
enum ENUM_SYMBOL_CALC_MODE
  {
   SYMBOL_CALC_MODE_FOREX,
   SYMBOL_CALC_MODE_FUTURES,
   SYMBOL_CALC_MODE_CFD,
   SYMBOL_CALC_MODE_CFDINDEX,
   SYMBOL_CALC_MODE_CFDLEVERAGE,
   SYMBOL_CALC_MODE_EXCH_STOCKS,
   SYMBOL_CALC_MODE_EXCH_FUTURES,
   SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS
  };
enum ENUM_SYMBOL_SWAP_MODE
  {
   SYMBOL_SWAP_MODE_DISABLED,
   SYMBOL_SWAP_MODE_POINTS,
   SYMBOL_SWAP_MODE_CURRENCY_SYMBOL,
   SYMBOL_SWAP_MODE_CURRENCY_MARGIN,
   SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT,
   SYMBOL_SWAP_MODE_INTEREST_CURRENT,
   SYMBOL_SWAP_MODE_INTEREST_OPEN,
   SYMBOL_SWAP_MODE_REOPEN_CURRENT,
   SYMBOL_SWAP_MODE_REOPEN_BID
  };
```

Note that not all of the unsupported
features in MQL4 were commented out. The changes made so far are the
minimum steps necessary to make the header file compile using an MQL4
compiler. It is recommended to consult the documentation of both languages when unsure whether or not a particular method or feature is supported in both versions.

Since each version has its own separate
copy of CSymbolInfo, the base header file will contain no class
definitions. Rather, it would only point to the location of the
header file to include, based on the compiler version being used:

```
#ifdef __MQL5__
   #include <Trade\SymbolInfo.mqh>
#else
   #include "..\..\MQL4\Lib\SymbolInfo.mqh"
#endif
```

The MQL5 header file will point to
the original location of the CSymbolInfo within the MQL5 standard
library. On the other hand, the MQL4 header file will point to the CSymbolInfo header file, which is the
modified copy of the MQL5 CSymbolInfo header file, located under Lib.

### CSymbolManager

To accommodate multiple currency expert
advisors, a collection of instances of CSymbolInfo will be needed.
This would be the purpose of the CSymbolManager. The class extends
CArrayObj to allow it to store various instances of symbol
information objects, along with some additional methods. Its relationship with CSymbolInfo instances is roughly illustrated in the following figure:

![Symbol Manager File Structure](https://c.mql5.com/2/24/symbolmanager.png)

That is, it behaves like CArrayObj, although the class is best used to store a specific type of object (CSymbolInfo instances). Since CSymbolInfo is based on CArrayObj and CArrayObj is available in both MQL4 and MQL5, we can limit most of the code to the base header file, and then reference the empty class files for the two versions (depending on what compiler is being used). In the end, it would have the same file structure as the CSymbolInfo class files discussed earlier:

![Symbol Manager File Structure](https://c.mql5.com/2/24/symbolmanagerclass.png)

#### Adding Symbols

The instances of CSymbolnfo are added
similar to the [Add](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjadd) method of CArrayObj . However, unlike CArrayObj,
the instance to be added should be unique. That is, there should be
no two elements that share the same symbol name.

```
bool CSymbolManagerBase::Add(CSymbolInfo *node)
  {
   if(Search(node.Name())==-1)
      return CArrayObj::Add(node);
   return false;
  }
```

The derived class will use a custom
Search method. It will not use the Search method of CArrayObj, since
that would make use of the Compare method of CSymbolInfo, which is
actually the method of [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) (CSymbolInfo has no explicit [Compare](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectcompare)
method). Using the Compare method would require us to extend
CSymbolInfo, which may be a more complicated solution since we
already have separate copies of the class. The new Search method will
compare the objects with the name of the symbols they track:

```
int CSymbolManagerBase::Search(string symbol=NULL)
  {
   if(symbol==NULL)
      symbol= Symbol();
   for(int i=0;i<Total();i++)
     {
      CSymbolInfo *item=At(i);
      if(StringCompare(item.Name(),symbol)==0)
         return i;
     }
   return -1;
  }
```

For multi-currency expert advisors,
there may exist a primary symbol. It can be the current symbol, but
it could also be a different one. This is the symbol which is
accessed more often than the rest. In this class, we would allow an
expert to be able to set a primary symbol during initialization. If
none is selected, the first CSymbolInfo instance will be treated as
the primary symbol:

```
void CSymbolManagerBase::SetPrimary(string symbol=NULL)
  {
   if(symbol==NULL)
      symbol= Symbol();
   m_symbol_primary=Get(symbol);
  }

CSymbolInfo *CSymbolManagerBase::GetPrimary(void)
  {
   if (!CheckPointer(m_symbol_primary) && Total()>0)
      SetPrimary(0);
   return m_symbol_primary;
  }
```

The are many possible approaches to this. The code above is one of the simplest.

#### Refreshing Symbols

To refresh the rates of all the symbols in the
symbol manager, the class simply iterates on all the objects
currently stored in it, and would call their [RefreshRates](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinforefreshrates) method one by
one. Note that for a huge collection of symbols, this may not be the
most practical method to call. Rather, in this case, it may be better to call the
RefreshRates function of a particular CSymbolInfo instance. The following
code shows the RefreshRates method of the class:

```
bool CSymbolManagerBase::RefreshRates(void)
  {
   for(int i=0;i<Total();i++)
     {
      CSymbolInfo *symbol=At(i);
      if(!CheckPointer(symbol))
         continue;
      if(!symbol.RefreshRates())
         return false;
     }
   return true;
  }
```

Note that the [Refresh](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinforefresh) and [RefreshRates](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinforefreshrates) methods in CSymbolInfo are different. The former is used during initialization, while the latter is used for updating the symbol information as of the last processed tick.

The following is the entire code of the
base implementation of the CSymbolManager class:

(SymbolManagerBase.mqh)

```
#include <Arrays\ArrayObj.mqh>
#include "..\Lib\SymbolInfo.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CSymbolManagerBase : public CArrayObj
  {
protected:
   CSymbolInfo      *m_symbol_primary;
   CObject          *m_container;
public:
                     CSymbolManagerBase(void);
                    ~CSymbolManagerBase(void);
   virtual bool      Add(CSymbolInfo*);
   virtual void      Deinit(void);
   CSymbolInfo      *Get(string);
   virtual bool      RefreshRates(void);
   virtual int       Search(string);
   virtual CObject *GetContainer(void);
   virtual void      SetContainer(CObject*);
   virtual void      SetPrimary(string);
   virtual void      SetPrimary(const int);
   virtual CSymbolInfo *GetPrimary(void);
   virtual string    GetPrimaryName(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolManagerBase::CSymbolManagerBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolManagerBase::~CSymbolManagerBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CObject *CSymbolManagerBase::GetContainer(void)
  {
   return m_container;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolManagerBase::SetContainer(CObject *container)
  {
   m_container=container;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolManagerBase::Deinit(void)
  {
   Shutdown();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSymbolManagerBase::Add(CSymbolInfo *node)
  {
   if(Search(node.Name())==-1)
      return CArrayObj::Add(node);
   return false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolInfo *CSymbolManagerBase::Get(string symbol=NULL)
  {
   if(symbol==NULL)
      symbol= Symbol();
   for(int i=0;i<Total();i++)
     {
      CSymbolInfo *item=At(i);
      if(!CheckPointer(item))
         continue;
      if(StringCompare(item.Name(),symbol)==0)
         return item;
     }
   return NULL;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSymbolManagerBase::RefreshRates(void)
  {
   for(int i=0;i<Total();i++)
     {
      CSymbolInfo *symbol=At(i);
      if(!CheckPointer(symbol))
         continue;
      if(!symbol.RefreshRates())
         return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSymbolManagerBase::Search(string symbol=NULL)
  {
   if(symbol==NULL)
      symbol= Symbol();
   for(int i=0;i<Total();i++)
     {
      CSymbolInfo *item=At(i);
      if(StringCompare(item.Name(),symbol)==0)
         return i;
     }
   return -1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CSymbolManagerBase::SetPrimary(string symbol=NULL)
  {
   if(symbol==NULL)
      symbol= Symbol();
   m_symbol_primary=Get(symbol);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CSymbolManagerBase::SetPrimary(const int idx)
  {
   m_symbol_primary=At(idx);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolInfo *CSymbolManagerBase::GetPrimary(void)
  {
   if (!CheckPointer(m_symbol_primary) && Total()>0)
      SetPrimary(0);
   return m_symbol_primary;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CSymbolManagerBase::GetPrimaryName(void)
  {
   if(CheckPointer(m_symbol_primary))
      return m_symbol_primary.Name();
   return NULL;
  }
//+------------------------------------------------------------------+
#ifdef __MQL5__
#include "..\..\MQL5\Symbol\SymbolManager.mqh"
#else
#include "..\..\MQL4\Symbol\SymbolManager.mqh"
#endif
//+------------------------------------------------------------------+
```

All of the methods are cross-platform
compatible. Thus, the platform-specific classes will be identical and
would contain no additional methods:

(SymbolManager.mqh, MQL4 and MQL5)

```
class CSymbolManager : public CSymbolManagerBase
  {
public:
                     CSymbolManager(void);
                    ~CSymbolManager(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolManager::CSymbolManager(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolManager::~CSymbolManager(void)
  {
  }
//+------------------------------------------------------------------+
```

### CAccountInfo

Similar to what was done with
CSymbolInfo, most of the unsupported features will need to be put in
comments. The unsupported features are have something to do with the
margin mode (only in Metatrader 5), and the functions [OrderCalcMargin](https://www.mql5.com/en/docs/trading/ordercalcmargin)
and [OrderCalcProfit](https://www.mql5.com/en/docs/trading/ordercalcprofit), which have no direct counterparts in MQL4. Thus, like what we did with the CSymbolInfo class, we create separate copies of the same class for the two platforms, and then modify the MQL4 version of the class so that it would compile using the MQL4 compiler. The file structure would be similar to the other classes discussed so far:

![CAccountInfo File Structure](https://c.mql5.com/2/24/accountinfo_class_structure.png)

The modified
class is shown below.

```
//+------------------------------------------------------------------+
//|                                                  AccountInfo.mqh |
//|                   Copyright 2009-2016, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#include <Object.mqh>
//+------------------------------------------------------------------+
//| Class CAccountInfo.                                              |
//| Appointment: Class for access to account info.                   |
//|              Derives from class CObject.                         |
//+------------------------------------------------------------------+
class CAccountInfo : public CObject
  {
public:
                     CAccountInfo(void);
                    ~CAccountInfo(void);
   //--- fast access methods to the integer account propertyes
   long              Login(void) const;
   ENUM_ACCOUNT_TRADE_MODE TradeMode(void) const;
   string            TradeModeDescription(void) const;
   long              Leverage(void) const;
   ENUM_ACCOUNT_STOPOUT_MODE StopoutMode(void) const;
   string            StopoutModeDescription(void) const;
   //ENUM_ACCOUNT_MARGIN_MODE MarginMode(void) const;
   //string            MarginModeDescription(void) const;
   bool              TradeAllowed(void) const;
   bool              TradeExpert(void) const;
   int               LimitOrders(void) const;
   //--- fast access methods to the double account propertyes
   double            Balance(void) const;
   double            Credit(void) const;
   double            Profit(void) const;
   double            Equity(void) const;
   double            Margin(void) const;
   double            FreeMargin(void) const;
   double            MarginLevel(void) const;
   double            MarginCall(void) const;
   double            MarginStopOut(void) const;
   //--- fast access methods to the string account propertyes
   string            Name(void) const;
   string            Server(void) const;
   string            Currency(void) const;
   string            Company(void) const;
   //--- access methods to the API MQL5 functions
   long              InfoInteger(const ENUM_ACCOUNT_INFO_INTEGER prop_id) const;
   double            InfoDouble(const ENUM_ACCOUNT_INFO_DOUBLE prop_id) const;
   string            InfoString(const ENUM_ACCOUNT_INFO_STRING prop_id) const;
   //--- checks
   //double            OrderProfitCheck(const string symbol,const ENUM_ORDER_TYPE trade_operation,
                                      //const double volume,const double price_open,const double price_close) const;
   //double            MarginCheck(const string symbol,const ENUM_ORDER_TYPE trade_operation,
                                 //const double volume,const double price) const;
   //double            FreeMarginCheck(const string symbol,const ENUM_ORDER_TYPE trade_operation,
                                     //const double volume,const double price) const;
   //double            MaxLotCheck(const string symbol,const ENUM_ORDER_TYPE trade_operation,
                                 //const double price,const double percent=100) const;
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccountInfo::CAccountInfo(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CAccountInfo::~CAccountInfo(void)
  {
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_LOGIN"                           |
//+------------------------------------------------------------------+
long CAccountInfo::Login(void) const
  {
   return(AccountInfoInteger(ACCOUNT_LOGIN));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_TRADE_MODE"                      |
//+------------------------------------------------------------------+
ENUM_ACCOUNT_TRADE_MODE CAccountInfo::TradeMode(void) const
  {
   return((ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_TRADE_MODE" as string            |
//+------------------------------------------------------------------+
string CAccountInfo::TradeModeDescription(void) const
  {
   string str;
//---
   switch(TradeMode())
     {
      case ACCOUNT_TRADE_MODE_DEMO   : str="Demo trading account";    break;
      case ACCOUNT_TRADE_MODE_CONTEST: str="Contest trading account"; break;
      case ACCOUNT_TRADE_MODE_REAL   : str="Real trading account";    break;
      default                        : str="Unknown trade account";
     }
//---
   return(str);
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_LEVERAGE"                        |
//+------------------------------------------------------------------+
long CAccountInfo::Leverage(void) const
  {
   return(AccountInfoInteger(ACCOUNT_LEVERAGE));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_MARGIN_SO_MODE"                  |
//+------------------------------------------------------------------+
ENUM_ACCOUNT_STOPOUT_MODE CAccountInfo::StopoutMode(void) const
  {
   return((ENUM_ACCOUNT_STOPOUT_MODE)AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_MARGIN_SO_MODE" as string        |
//+------------------------------------------------------------------+
string CAccountInfo::StopoutModeDescription(void) const
  {
   string str;
//---
   switch(StopoutMode())
     {
      case ACCOUNT_STOPOUT_MODE_PERCENT: str="Level is specified in percentage"; break;
      case ACCOUNT_STOPOUT_MODE_MONEY  : str="Level is specified in money";      break;
      default                          : str="Unknown stopout mode";
     }
//---
   return(str);
  }
/*
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_MARGIN_MODE"                     |
//+------------------------------------------------------------------+
ENUM_ACCOUNT_MARGIN_MODE CAccountInfo::MarginMode(void) const
  {
   return((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE));
  }
*/
/*
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_MARGIN_MODE" as string           |
//+------------------------------------------------------------------+
string CAccountInfo::MarginModeDescription(void) const
  {
   string str;
//---
   switch(MarginMode())
     {
      case ACCOUNT_MARGIN_MODE_RETAIL_NETTING: str="Netting";  break;
      case ACCOUNT_MARGIN_MODE_EXCHANGE      : str="Exchange"; break;
      case ACCOUNT_MARGIN_MODE_RETAIL_HEDGING: str="Hedging";  break;
      default                                : str="Unknown margin mode";
     }
//---
   return(str);
  }
*/
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_TRADE_ALLOWED"                   |
//+------------------------------------------------------------------+
bool CAccountInfo::TradeAllowed(void) const
  {
   return((bool)AccountInfoInteger(ACCOUNT_TRADE_ALLOWED));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_TRADE_EXPERT"                    |
//+------------------------------------------------------------------+
bool CAccountInfo::TradeExpert(void) const
  {
   return((bool)AccountInfoInteger(ACCOUNT_TRADE_EXPERT));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_LIMIT_ORDERS"                    |
//+------------------------------------------------------------------+
int CAccountInfo::LimitOrders(void) const
  {
   return((int)AccountInfoInteger(ACCOUNT_LIMIT_ORDERS));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_BALANCE"                         |
//+------------------------------------------------------------------+
double CAccountInfo::Balance(void) const
  {
   return(AccountInfoDouble(ACCOUNT_BALANCE));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_CREDIT"                          |
//+------------------------------------------------------------------+
double CAccountInfo::Credit(void) const
  {
   return(AccountInfoDouble(ACCOUNT_CREDIT));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_PROFIT"                          |
//+------------------------------------------------------------------+
double CAccountInfo::Profit(void) const
  {
   return(AccountInfoDouble(ACCOUNT_PROFIT));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_EQUITY"                          |
//+------------------------------------------------------------------+
double CAccountInfo::Equity(void) const
  {
   return(AccountInfoDouble(ACCOUNT_EQUITY));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_MARGIN"                          |
//+------------------------------------------------------------------+
double CAccountInfo::Margin(void) const
  {
   return(AccountInfoDouble(ACCOUNT_MARGIN));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_FREEMARGIN"                      |
//+------------------------------------------------------------------+
double CAccountInfo::FreeMargin(void) const
  {
   return(AccountInfoDouble(ACCOUNT_FREEMARGIN));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_MARGIN_LEVEL"                    |
//+------------------------------------------------------------------+
double CAccountInfo::MarginLevel(void) const
  {
   return(AccountInfoDouble(ACCOUNT_MARGIN_LEVEL));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_MARGIN_SO_CALL"                  |
//+------------------------------------------------------------------+
double CAccountInfo::MarginCall(void) const
  {
   return(AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_MARGIN_SO_SO"                    |
//+------------------------------------------------------------------+
double CAccountInfo::MarginStopOut(void) const
  {
   return(AccountInfoDouble(ACCOUNT_MARGIN_SO_SO));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_NAME"                            |
//+------------------------------------------------------------------+
string CAccountInfo::Name(void) const
  {
   return(AccountInfoString(ACCOUNT_NAME));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_SERVER"                          |
//+------------------------------------------------------------------+
string CAccountInfo::Server(void) const
  {
   return(AccountInfoString(ACCOUNT_SERVER));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_CURRENCY"                        |
//+------------------------------------------------------------------+
string CAccountInfo::Currency(void) const
  {
   return(AccountInfoString(ACCOUNT_CURRENCY));
  }
//+------------------------------------------------------------------+
//| Get the property value "ACCOUNT_COMPANY"                         |
//+------------------------------------------------------------------+
string CAccountInfo::Company(void) const
  {
   return(AccountInfoString(ACCOUNT_COMPANY));
  }
//+------------------------------------------------------------------+
//| Access functions AccountInfoInteger(...)                         |
//+------------------------------------------------------------------+
long CAccountInfo::InfoInteger(const ENUM_ACCOUNT_INFO_INTEGER prop_id) const
  {
   return(AccountInfoInteger(prop_id));
  }
//+------------------------------------------------------------------+
//| Access functions AccountInfoDouble(...)                          |
//+------------------------------------------------------------------+
double CAccountInfo::InfoDouble(const ENUM_ACCOUNT_INFO_DOUBLE prop_id) const
  {
   return(AccountInfoDouble(prop_id));
  }
//+------------------------------------------------------------------+
//| Access functions AccountInfoString(...)                          |
//+------------------------------------------------------------------+
string CAccountInfo::InfoString(const ENUM_ACCOUNT_INFO_STRING prop_id) const
  {
   return(AccountInfoString(prop_id));
  }
/*
//+------------------------------------------------------------------+
//| Access functions OrderCalcProfit(...).                            |
//| INPUT:  name            - symbol name,                           |
//|         trade_operation - trade operation,                       |
//|         volume          - volume of the opening position,        |
//|         price_open      - price of the opening position,         |
//|         price_close     - price of the closing position.         |
//+------------------------------------------------------------------+
double CAccountInfo::OrderProfitCheck(const string symbol,const ENUM_ORDER_TYPE trade_operation,
                                      const double volume,const double price_open,const double price_close) const
  {
   double profit=EMPTY_VALUE;
//---
   if(!OrderCalcProfit(trade_operation,symbol,volume,price_open,price_close,profit))
      return(EMPTY_VALUE);
//---
   return(profit);
  }
*/
/*
//+------------------------------------------------------------------+
//| Access functions OrderCalcMargin(...).                           |
//| INPUT:  name            - symbol name,                           |
//|         trade_operation - trade operation,                       |
//|         volume          - volume of the opening position,        |
//|         price           - price of the opening position.         |
//+------------------------------------------------------------------+
double CAccountInfo::MarginCheck(const string symbol,const ENUM_ORDER_TYPE trade_operation,
                                 const double volume,const double price) const
  {
   double margin=EMPTY_VALUE;
//---
   if(!OrderCalcMargin(trade_operation,symbol,volume,price,margin))
      return(EMPTY_VALUE);
//---
   return(margin);
  }
*/
/*
//+------------------------------------------------------------------+
//| Access functions OrderCalcMargin(...).                           |
//| INPUT:  name            - symbol name,                           |
//|         trade_operation - trade operation,                       |
//|         volume          - volume of the opening position,        |
//|         price           - price of the opening position.         |
//+------------------------------------------------------------------+
double CAccountInfo::FreeMarginCheck(const string symbol,const ENUM_ORDER_TYPE trade_operation,
                                     const double volume,const double price) const
  {
   return(FreeMargin()-MarginCheck(symbol,trade_operation,volume,price));
  }
*/
/*
//+------------------------------------------------------------------+
//| Access functions OrderCalcMargin(...).                           |
//| INPUT:  name            - symbol name,                           |
//|         trade_operation - trade operation,                       |
//|         price           - price of the opening position,         |
//|         percent         - percent of available margin [1-100%].  |
//+------------------------------------------------------------------+
double CAccountInfo::MaxLotCheck(const string symbol,const ENUM_ORDER_TYPE trade_operation,
                                 const double price,const double percent) const
  {
   double margin=0.0;
//--- checks
   if(symbol=="" || price<=0.0 || percent<1 || percent>100)
     {
      Print("CAccountInfo::MaxLotCheck invalid parameters");
      return(0.0);
     }
//--- calculate margin requirements for 1 lot
   if(!OrderCalcMargin(trade_operation,symbol,1.0,price,margin) || margin<0.0)
     {
      Print("CAccountInfo::MaxLotCheck margin calculation failed");
      return(0.0);
     }
//---
   if(margin==0.0) // for pending orders
      return(SymbolInfoDouble(symbol,SYMBOL_VOLUME_MAX));
//--- calculate maximum volume
   double volume=NormalizeDouble(FreeMargin()*percent/100.0/margin,2);
//--- normalize and check limits
   double stepvol=SymbolInfoDouble(symbol,SYMBOL_VOLUME_STEP);
   if(stepvol>0.0)
      volume=stepvol*MathFloor(volume/stepvol);
//---
   double minvol=SymbolInfoDouble(symbol,SYMBOL_VOLUME_MIN);
   if(volume<minvol)
      volume=0.0;
//---
   double maxvol=SymbolInfoDouble(symbol,SYMBOL_VOLUME_MAX);
   if(volume>maxvol)
      volume=maxvol;
//--- return volume
   return(volume);
  }
*/
//+------------------------------------------------------------------+
```

The platform can only access one account at a time. So, creating a collection of instances of CAccountInfo may no longer be necessary.

### CExpertTrade

Despite having the same end-objectives,
MQL4 and MQL5 differ in how trade operations are executed. This means
that we can possibly use the CExpertTrade class on the MQL5 version
of a cross-platform expert advisor, but not entirely on MQL4. In this
case, for the MQL4 version, we will create a new header file
containing a class of the same name. We still retain the base header class file so that it uses the correct header file, depending on the type of compiler being used, according to the following structure:

![CExpertTrade File Structure](https://c.mql5.com/2/24/cexperttradestructure.png)

This class would emulate most of
the members and methods found in MQL5 CexpertTrade, such that
when we execute code such as the following:

```
CExpertTrade trade;
trade.Buy(/*params*/);
```

both versions would be able to
understand that we intend to enter a long position.

The minimum requirement is to emulate
the methods use to enter positions on the platform. That is, the Buy
and Sell method of CExpertTrade. For market exit, we will diverge on our
implementation: OrderClose for the MQL4 version, while using the
native methods PositionClose (hedging) or Buy and Sell methods
(netting) for the MQL5 version. The following code shows the MQL4
version for CExpertTrade. This would be one of the parts where the code may vary largely from one programmer to another:

```
#include <Arrays\ArrayInt.mqh>
enum ENUM_ORDER_TYPE_TIME
  {
   ORDER_TIME_GTC,
   ORDER_TIME_DAY,
   ORDER_TIME_SPECIFIED,
   ORDER_TIME_SPECIFIED_DAY
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CExpertTrade : public CObject
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
  {
protected:
   int               m_magic;
   ulong             m_deviation;
   ENUM_ORDER_TYPE_TIME m_order_type_time;
   datetime          m_order_expiration;
   bool              m_async_mode;
   uint              m_retry;
   int               m_sleep;
   color             m_color_long;
   color             m_color_short;
   color             m_color_buystop;
   color             m_color_buylimit;
   color             m_color_sellstop;
   color             m_color_selllimit;
   color             m_color_modify;
   color             m_color_exit;
   CSymbolInfo      *m_symbol;
public:
                     CExpertTrade(void);
                    ~CExpertTrade(void);
   //--- setters and getters
   color             ArrowColor(const ENUM_ORDER_TYPE type);
   uint              Retry() {return m_retry;}
   void              Retry(uint retry){m_retry=retry;}
   int               Sleep() {return m_sleep;}
   void              Sleep(int sleep){m_sleep=sleep;}
   void              SetAsyncMode(const bool mode) {m_async_mode=mode;}
   void              SetExpertMagicNumber(const int magic) {m_magic=magic;}
   void              SetDeviationInPoints(const ulong deviation) {m_deviation=deviation;}
   void              SetOrderExpiration(const datetime expire) {m_order_expiration=expire;}
   bool              SetSymbol(CSymbolInfo *);
   //-- trade methods
   virtual ulong     Buy(const double,const double,const double,const double,const string);
   virtual ulong     Sell(const double,const double,const double,const double,const string);
   virtual bool      OrderDelete(const ulong);
   virtual bool      OrderClose(const ulong,const double,const double);
   virtual bool      OrderCloseAll(CArrayInt *,const bool);
   virtual bool      OrderModify(const ulong,const double,const double,const double,const ENUM_ORDER_TYPE_TIME,const datetime,const double);
   virtual ulong     OrderOpen(const string,const ENUM_ORDER_TYPE,const double,const double,const double,const double,const double,const ENUM_ORDER_TYPE_TIME,const datetime,const string);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpertTrade::CExpertTrade(void) : m_magic(0),
                                   m_deviation(10),
                                   m_order_type_time(0),
                                   m_symbol(NULL),
                                   m_async_mode(0),
                                   m_retry(3),
                                   m_sleep(100),
                                   m_color_long(clrGreen),
                                   m_color_buystop(clrGreen),
                                   m_color_buylimit(clrGreen),
                                   m_color_sellstop(clrRed),
                                   m_color_selllimit(clrRed),
                                   m_color_short(clrRed),
                                   m_color_modify(clrNONE),
                                   m_color_exit(clrNONE)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpertTrade::~CExpertTrade(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpertTrade::SetSymbol(CSymbolInfo *symbol)
  {
   if(symbol!=NULL)
     {
      m_symbol=symbol;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpertTrade::OrderModify(const ulong ticket,const double price,const double sl,const double tp,
                               const ENUM_ORDER_TYPE_TIME type_time,const datetime expiration,const double stoplimit=0.0)
  {
   return ::OrderModify((int)ticket,price,sl,tp,expiration,m_color_modify);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpertTrade::OrderDelete(const ulong ticket)
  {
   return ::OrderDelete((int)ticket);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong CExpertTrade::Buy(const double volume,const double price,const double sl,const double tp,const string comment="")
  {
   if(m_symbol==NULL)
      return false;
   m_symbol.RefreshRates();
   string symbol=m_symbol.Name();
   double stops_level=m_symbol.StopsLevel()*m_symbol.Point();
   double ask=m_symbol.Ask();
   if(symbol=="")
      return 0;
   if(price!=0)
     {
      if(price>ask+stops_level)
         return OrderOpen(symbol,ORDER_TYPE_BUY_STOP,volume,0.0,price,sl,tp,m_order_type_time,m_order_expiration,comment);
      if(price<ask-stops_level)
         return OrderOpen(symbol,ORDER_TYPE_BUY_LIMIT,volume,0.0,price,sl,tp,m_order_type_time,m_order_expiration,comment);
     }
   return OrderOpen(symbol,ORDER_TYPE_BUY,volume,0.0,ask,sl,tp,m_order_type_time,m_order_expiration,comment);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong CExpertTrade::Sell(const double volume,const double price,const double sl,const double tp,const string comment="")
  {
   if(m_symbol==NULL)
      return false;
   m_symbol.RefreshRates();
   string symbol=m_symbol.Name();
   double stops_level=m_symbol.StopsLevel()*m_symbol.Point();
   double bid=m_symbol.Bid();
   if(symbol=="")
      return 0;
   if(price!=0)
     {
      if(price>bid+stops_level)
         return OrderOpen(symbol,ORDER_TYPE_SELL_LIMIT,volume,0.0,price,sl,tp,m_order_type_time,m_order_expiration,comment);
      if(price<bid-stops_level)
         return OrderOpen(symbol,ORDER_TYPE_SELL_STOP,volume,0.0,price,sl,tp,m_order_type_time,m_order_expiration,comment);
     }
   return OrderOpen(symbol,ORDER_TYPE_SELL,volume,0.0,bid,sl,tp,m_order_type_time,m_order_expiration,comment);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong CExpertTrade::OrderOpen(const string symbol,const ENUM_ORDER_TYPE order_type,const double volume,
                              const double limit_price,const double price,const double sl,const double tp,
                              const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC,const datetime expiration=0,
                              const string comment="")
  {
   bool res;
   ulong ticket=0;
   color arrowcolor=ArrowColor(order_type);
   datetime expire=0;
   if(order_type>1 && expiration>0) expire=expiration*1000+TimeCurrent();
   double stops_level=m_symbol.StopsLevel();
   for(uint i=0;i<m_retry;i++)
     {
      if(ticket>0)
         break;
      if(IsStopped())
         return 0;
      if(IsTradeContextBusy() || !IsConnected())
        {
         ::Sleep(m_sleep);
         continue;
        }
      if(stops_level==0 && order_type<=1)
        {
         ticket=::OrderSend(symbol,order_type,volume,price,(int)(m_deviation*m_symbol.Point()),0,0,comment,m_magic,expire,arrowcolor);
         ::Sleep(m_sleep);
         for(uint j=0;j<m_retry;j++)
           {
            if(res) break;
            if(ticket>0 && (sl>0 || tp>0))
              {
               if(OrderSelect((int)ticket,SELECT_BY_TICKET))
                 {
                  res=OrderModify((int)ticket,OrderOpenPrice(),sl,tp,OrderExpiration());
                  ::Sleep(m_sleep);
                 }
              }
           }
        }
      else
        {
         ticket=::OrderSend(symbol,order_type,volume,price,(int)m_deviation,sl,tp,comment,m_magic,expire,arrowcolor);
         ::Sleep(m_sleep);
        }
     }
   return ticket>0?ticket:0;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpertTrade::OrderClose(const ulong ticket,const double lotsize=0.0,const double price=0.0)
  {
   if(!OrderSelect((int)ticket,SELECT_BY_TICKET))
      return false;
   if(OrderCloseTime()>0)
      return true;
   double close_price=0.0;
   int deviation=0;
   if(OrderSymbol()==m_symbol.Name() && price>0.0)
     {
      close_price=NormalizeDouble(price,m_symbol.Digits());
      deviation=(int)(m_deviation*m_symbol.Point());
     }
   else
     {
      close_price=NormalizeDouble(OrderClosePrice(),(int)MarketInfo(OrderSymbol(),MODE_DIGITS));
      deviation=(int)(m_deviation*MarketInfo(OrderSymbol(),MODE_POINT));
     }
   double lots=(lotsize>0.0 || lotsize>OrderLots())?lotsize:OrderLots();
   return ::OrderClose((int)ticket,lots,close_price,deviation,m_color_exit);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpertTrade::OrderCloseAll(CArrayInt *other_magic,const bool restrict_symbol=true)
  {
   bool res=true;
   int total= OrdersTotal();
   for(int i=total-1;i>=0;i--)
     {
      double bid=0.0,ask=0.0;
      if(!OrderSelect(i,SELECT_BY_POS)) continue;
      if(OrderSymbol()!=m_symbol.Name() && restrict_symbol) continue;
      if(OrderMagicNumber()!=m_magic && other_magic.Search(OrderMagicNumber())<0) continue;
      m_symbol.RefreshRates();
      if(OrderSymbol()==m_symbol.Name())
        {
         bid = m_symbol.Bid();
         ask = m_symbol.Ask();
        }
      else
        {
         bid = MarketInfo(OrderSymbol(),MODE_BID);
         ask = MarketInfo(OrderSymbol(),MODE_ASK);
        }
      if(res) res=OrderClose(OrderTicket(),OrderLots(),OrderType()==ORDER_TYPE_BUY?bid:ask);
     }
   return res;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
color CExpertTrade::ArrowColor(const ENUM_ORDER_TYPE type)
  {
   switch(type)
     {
      case ORDER_TYPE_BUY:        return m_color_long;
      case ORDER_TYPE_SELL:       return m_color_short;
      case ORDER_TYPE_BUY_STOP:   return m_color_buystop;
      case ORDER_TYPE_BUY_LIMIT:  return m_color_buylimit;
      case ORDER_TYPE_SELL_STOP:  return m_color_sellstop;
      case ORDER_TYPE_SELL_LIMIT: return m_color_selllimit;
     }
   return clrNONE;
  }
//+------------------------------------------------------------------+
```

Note that for the MQL5 version, the
CExpertTrade class inherits from the CTrade class (which then extends CObject). On the other hand, for the MQL4 version, the CExpertTrade class directly inherits from CObject. This means that for the MQL4 version, CTrade and CExpertTrade are consolidated into a single
object.

Some features such as ORDER\_TYPE\_TIME have no direct counterparts in MQL4. However, this may be useful in case one would need to extend this trade class and emulate the MQL5 order lifetime in MQL4.

### CTradeManager

The native MQL5 class CExpertTrade has
a method named SetSymbol. This method allows it to switch its symbol
information pointer to point to another symbol information instance
(CSymbolInfo). With this setup, most of the methods found in the
class would no longer need to have a string symbol parameter, which
denotes the name of the instrument to process.

In most cases, simply using
CExpertTrade alone and simply switching symbols is sufficient.
However, there are a few considerations when using this approach. In
some cases, the deviation or maximum slippage would need to be
updated when the symbol pointer is changed. This is true especially
when the expert advisor will deal with instruments that have
different digits. Another factor is the magic number, which would
need to be updated if an expert advisor would prefer to use a
different magic number for positions entered on each symbol it trades
on.

One way to approach this problem is to
use a trade manager. Similar to the symbol manager covered earlier,
this object would extend CArrayObj, and would virtually have the same
set of methods as CSymbolManager. We would have a base header file, which would reference the correct descendant depending on the compiler being used to compile the file. And like the symbol manager, the trade manager deals with storage and retrieval of data. Thus, most of its code can be found on the base header file. The file structure is shown in the following figure.

![Trade Manager File Structure](https://c.mql5.com/2/24/trademanager__1.png)

The code for the base class for our trade manager is shown below.

(TradeManagerBase.mqh)

```
#include <Arrays\ArrayObj.mqh>
#include "ExpertTradeXBase.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CTradeManagerBase : public CArrayObj
  {
public:
                     CTradeManagerBase(void);
                    ~CTradeManagerBase(void);
   virtual int       Search(string);
   virtual bool      Add(CExpertTradeX*);
   virtual void      Deinit(void);
   CExpertTrade     *Get(const string);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTradeManagerBase::CTradeManagerBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTradeManagerBase::~CTradeManagerBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTradeManagerBase::Deinit(void)
  {
   Shutdown();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CTradeManagerBase::Add(CExpertTradeX *node)
  {
   if(Search(node.Name())==-1)
      return CArrayObj::Add(node);
   return false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CTradeManagerBase::Search(string symbol=NULL)
  {
   if(symbol==NULL)
      symbol= Symbol();
   for(int i=0;i<Total();i++)
     {
      CExpertTradeX *item=At(i);
      if(StringCompare(item.Name(),symbol)==0)
         return i;
     }
   return -1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpertTrade *CTradeManagerBase::Get(const string name=NULL)
  {
   if(name==NULL && Total()>0)
      return At(0);
   for(int i=0;i<Total();i++)
     {
      CExpertTradeX *item=At(i);
      if(StringCompare(item.Name(),name)==0)
         return item;
     }
   return NULL;
  }
//+------------------------------------------------------------------+
#ifdef __MQL5__
#include "..\..\MQL5\Trade\TradeManager.mqh"
#else
#include "..\..\MQL4\Trade\TradeManager.mqh"
#endif
//+------------------------------------------------------------------+
```

One may also simply use a single
instance of CExpertTrade, and simply extend it so it would have its
own collection of symbols to switch on. But this would also use additional memory space, since one would also need to store the magic numbers
and deviations, and link them to a specific symbol.

Like the symbol manager, the class also
uses a custom Search method to compare two trade objects (trade
objects should be unique). The comparison is performed using the
names of the trade objects, which is then based on the name of the instance of CSymbolInfo they link to. However, for the MQL5 version, the
CExpertTrade object does not return either the symbol pointer or the
name of the symbol. In this case, we need to extend CExpertTrade to
allow instances of this object to be able to return the name of the
symbol it contains. We would give the object a name of CExpertTradeX. Like all the other classes defined in this article, there should be a base header file, which decides on what header to choose as its descendant depending on the compiler being used:

![CExpertTradeX File Structure](https://c.mql5.com/2/24/cexperttradexstructure.png)

The following shows the base implementation of the said class:

(CExpertTradeXBase.mqh)

```
#include "ExpertTradeBase.mqh"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CExpertTradeXBase : public CExpertTrade
  {
public:
                     CExpertTradeXBase(void);
                    ~CExpertTradeXBase(void);
   string            Name(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpertTradeXBase::CExpertTradeXBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpertTradeXBase::~CExpertTradeXBase(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CExpertTradeXBase::Name(void)
  {
   if (CheckPointer(m_symbol))
      return m_symbol.Name();
   return NULL;
  }
//+------------------------------------------------------------------+
#ifdef __MQL5__
#include "..\..\MQL5\Trade\ExpertTradeX.mqh"
#else
#include "..\..\MQL4\Trade\ExpertTradeX.mqh"
#endif
//+------------------------------------------------------------------+
```

Note that this CExpertTradeX class extends CExpertTrade. It would appear that this object inherits from a single
object. However, in reality, the MQL4 and MQL5 versions have different versions of CExpertTrade. This is a fairly simple object, and
its methods are cross-platform compatible, so like some classes, we
declare its platform-specific classes with no additional methods:

(CExpertTradeX.mqh, MQL4 and MQL5 version)

```
class CExpertTradeX : public CExpertTradeXBase
  {
public:
                     CExpertTradeX(void);
                    ~CExpertTradeX(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpertTradeX::CExpertTradeX(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpertTradeX::~CExpertTradeX(void)
  {
  }
//+------------------------------------------------------------------+
```

### Conclusion

In this article, we have demonstrated
how some native components of the MQL5 Standard Library can possibly
restructured in order to make them useable in MQL4 expert advisors,
rather than coding these classes from scratch for the MQL4 version.
The classes CSymbolInfo, CAccount, and CExpertTrade were modified,
and managers such as CSymbolManager and CTradeManager were developed
to allow expert advisors and scripts to handle multiple instances of one of the said
objects (CSymbolInfo, CAccount, and CExpertTrade).

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2574.zip "Download all attachments in the single ZIP archive")

[MQLx-Reuse.zip](https://www.mql5.com/en/articles/download/2574/mqlx-reuse.zip "Download MQLx-Reuse.zip")(92.47 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Cross-Platform Expert Advisor: The CExpertAdvisor and CExpertAdvisors Classes](https://www.mql5.com/en/articles/3622)
- [Cross-Platform Expert Advisor: Custom Stops, Breakeven and Trailing](https://www.mql5.com/en/articles/3621)
- [Cross-Platform Expert Advisor: Stops](https://www.mql5.com/en/articles/3620)
- [Cross-Platform Expert Advisor: Time Filters](https://www.mql5.com/en/articles/3395)
- [Cross-Platform Expert Advisor: Money Management](https://www.mql5.com/en/articles/3280)
- [Cross-Platform Expert Advisor: Signals](https://www.mql5.com/en/articles/3261)
- [Cross-Platform Expert Advisor: Order Manager](https://www.mql5.com/en/articles/2961)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/94363)**
(3)


![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
3 Sep 2016 at 17:12

Hi!

You write at the beginning: "However, its main **disadvantage** is that we don't have to rewrite a lot of code."

You mean "..  **advantage**.." \- correct?

![Heavybabar](https://c.mql5.com/avatar/avatar_na2.png)

**[Heavybabar](https://www.mql5.com/en/users/heavybabar)**
\|
26 Feb 2017 at 03:21

> Hi,
>
> Very nice job and clear explanations. I'll stay in touch, as at that time nothing to reward you with, just words. THANKS !

![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
16 Dec 2017 at 22:15

I never understood why someone want to use a wrapper class like SymbolInfo ?

There is only 1 useful function :

```
double CSymbolInfo::NormalizePrice(const double price) const
```

All the rest is just overload to slow down your code. A well code coded Symbol class could be useful, this "standard" one is just useless. Don't even want to talk about "CAccountInfo".

[@Enrico Lambino](https://www.mql5.com/en/users/iceron) why didn't you create your own classes instead  ?

![Graphical Interfaces IX: The Color Picker Control (Chapter 1)](https://c.mql5.com/2/23/IX.png)[Graphical Interfaces IX: The Color Picker Control (Chapter 1)](https://www.mql5.com/en/articles/2579)

With this article we begin chapter nine of series of articles dedicated to creating graphical interfaces in MetaTrader trading terminals. It consists of two chapters where new elements of controls and interface, such as color picker, color button, progress bar and line chart are presented.

![LifeHack for trader: four backtests are better than one](https://c.mql5.com/2/23/ava__3.png)[LifeHack for trader: four backtests are better than one](https://www.mql5.com/en/articles/2552)

Before the first single test, every trader faces the same question — "Which of the four modes to use?" Each of the provided modes has its advantages and features, so we will do it the easy way - run all four modes at once with a single button! The article shows how to use the Win API and a little magic to see all four testing chart at the same time.

![Graphical Interfaces IX: The Progress Bar and Line Chart Controls (Chapter 2)](https://c.mql5.com/2/23/IX__1.png)[Graphical Interfaces IX: The Progress Bar and Line Chart Controls (Chapter 2)](https://www.mql5.com/en/articles/2580)

The second chapter of the part nine is dedicated to the progress bar and line chart controls. As always, there will be detailed examples provided to reveal how these controls can be used in custom MQL applications.

![Graphical Interfaces VIII: the File Navigator Control (Chapter 3)](https://c.mql5.com/2/23/av8__2.png)[Graphical Interfaces VIII: the File Navigator Control (Chapter 3)](https://www.mql5.com/en/articles/2541)

In the previous chapters of the eighth part of the series, our library has been reinforced by several classes for developing mouse pointers, calendars and tree views. The current article deals with the file navigator control that can also be used as part of an MQL application graphical interface.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ydfofenhhbguclnbtulswjkjyqrkpwbl&ssn=1769178378088057878&ssn_dr=0&ssn_sr=0&fv_date=1769178378&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2574&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Cross-Platform%20Expert%20Advisor%3A%20Reuse%20of%20Components%20from%20the%20MQL5%20Standard%20Library%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917837850524438&fz_uniq=5068231030772528906&sv=2552)

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