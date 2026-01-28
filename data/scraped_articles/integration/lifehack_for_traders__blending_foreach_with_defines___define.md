---
title: LifeHack for traders: Blending ForEach with defines (#define)
url: https://www.mql5.com/en/articles/4332
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:06:17.077740
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/4332&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083338178804324775)

MetaTrader 5 / Examples


— What makes you cool, bro?

— The defines make you cool, bro

                                  (с)  fxsaber

You still write in MQL4 and want to switch to MQL5? We will show you where to start from! Now you can work comfortably in MQL5 MetaEditor and at the same time use the MQL4 notation (actually, such an opportunity appeared a bit earlier, although in this article, I want to provide a more complete and detailed description of how to migrate MQL4 functions to MQL5).

### A good programmer is a lazy programmer

Creating Expert Advisors almost always entails a lot of work with loops. Loops surround us everywhere: searching orders, trades in history, chart objects, Market Watch symbols, bars in an indicator buffer. To make a programmer's life a bit easier, MetaEditor features [snippets](https://www.metatrader5.com/en/metaeditor/help/development/intelligent_management#snippet "https://www.metatrader5.com/en/metaeditor/help/development/intelligent_management#snippet") meaning that when you enter the first characters, they automatically turn into a small piece of code after pressing Tab. This is how the 'for' loop snippet works:

![](https://c.mql5.com/2/31/for_snippet.gif)

Not bad, but it does not cover all our needs. Consider the simplest example: suppose that we need to search through all Market Watch symbols.

```
   int total=SymbolsTotal(true);
   for(int i=0;i<total;i++)
     {
      string symbol=SymbolName(i,true);
      PrintFormat("%d. %s",i+1,symbol);
     }
```

It would be great to develop a MetaEditor snippet starting with **fes** (for\_each\_symbol) and unfolding into the following block:

![](https://c.mql5.com/2/31/fes.gif)

There are no custom snippets in MetaEditor, so we will apply the 'defines'. The [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) macro substitution was invented by lazy smart programmers who pursued several goals. Among them are the ease of reading and the convenience of writing repetitive code.

Many programming languages, in addition to the standard [for](https://www.mql5.com/en/docs/basis/operators/for) loop, features its variations, for example: [for(<typename> element:Collection)](https://www.mql5.com/go?link=https://www.geeksforgeeks.org/for-each-loop-in-java/ "/go?link=https://www.geeksforgeeks.org/for-each-loop-in-java/") or [for each (type identifier in expression)](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/en-en/library/ms177202.aspx "/go?link=https://msdn.microsoft.com/en-us/en-en/library/ms177202.aspx"). If we could write the following code

```
for(ulong order_id in History)
  {
   working with order_id
  }
```

, the life of the programmer would be a little easier. You can find opponents and supporters of this approach on the Internet. Here I will show you how to do a similar thing with the #define macros.

Let's start with a simple task - get the names of all Market Watch symbols. Let's move straight forward and write the following macro:

```
#define ForEachSymbol(s,i)  string s; int total=SymbolsTotal(true); for(int i=0;i<total;i++,s=SymbolName(i,true))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachSymbol(symbol,index)
     {
      PrintFormat("%d. %s",index,symbol);
     }
  }
```

The compiler perfectly understands this entry returning no errors. We start debugging by pressing **F5** and see that something has gone wrong:

```
1. (null)
2. GBPUSD
3. USDCHF
4. USDJPY
...
```

The problem is that the s = SymbolName (i, true) expression in the 'for' loop is calculated after iteration, and our 's' variable is not initialized at the first iteration, when i=0. ['for' statement](https://www.mql5.com/en/docs/basis/operators/for "'for' loop operator"):

The 'for' statement consists of three expressions and an executable statement:

|     |
| --- |
| for(expression1; expression2; expression3)<br>statement; |

Expression1 describes the loop initialization. Expression2 — check for the loop completion condition. If it is 'true', then the for loop body operator is executed. Everything repeats until expression2 becomes 'false'. If it is 'false', the loop terminates and control is passed to the next statement. ExpressionЗ is calculated aftereach iteration.

The process is simple. Let's make a couple of edits:

```
#define ForEachSymbol(s,i)  string s=SymbolName(0,true); int total=SymbolsTotal(true); for(int i=1;i<total;i++,s=SymbolName(i,true))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachSymbol(symbol,index)
     {
      PrintFormat("%d. %s",index,symbol); // index+1 is replaced with index
     }
  }
```

and get a necessary result. We have developed the ForEachSymbol macro with _symbol_ and _index_ parameters as if this has been a normal ' _for_' loop, and worked with these variables in the pseudo-loop body as if they have already been declared and initialized using necessary values. Thus, we can get the desired properties of the Market Watch symbols using the [SymbolInfoXXX](https://www.mql5.com/en/docs/marketinformation)() functions. For example:

```
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachSymbol(symbol,index)
     {
      //--- prepare data
      double spread=SymbolInfoDouble(symbol,SYMBOL_ASK)-SymbolInfoDouble(symbol,SYMBOL_BID);
      double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
      long digits=SymbolInfoInteger(symbol,SYMBOL_DIGITS);
      string str_spread=DoubleToString(spread/point,0);
      string str_point=DoubleToString(point,digits);
      //--- display data
      Print(index,". ",symbol," spread=",str_spread," points (",
            digits," digits",", point=",str_point,")");
     }
/* Sample output
        1. EURUSD spread=3 points (5 digits, point=0.00001)
        2. USDCHF spread=8 points (5 digits, point=0.00001)
        3. USDJPY spread=5 points (3 digits, point=0.001)
        4. USDCAD spread=9 points (5 digits, point=0.00001)
        5. AUDUSD spread=5 points (5 digits, point=0.00001)
        6. NZDUSD spread=10 points (5 digits, point=0.00001)
        7. USDSEK spread=150 points (5 digits, point=0.00001)
*/
  }
```

Now we can write a similar macro for searching graphical objects on the chart:

```
#define ForEachObject(name,i)   string name=ObjectName(0,0); int total=ObjectsTotal(0); for(int i=1;i<=total;i++,name=ObjectName(0,i-1))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachObject(objectname,index)
     {
      Print(index,": objectname=\"",objectname,"\", objecttype=",
            EnumToString((ENUM_OBJECT)ObjectGetInteger(0,objectname,OBJPROP_TYPE)));
     }
/* Sample output
        1: objectname="H1 Arrow 61067", objecttype=OBJ_ARROW_UP
        2: objectname="H1 Rectangle 31152", objecttype=OBJ_RECTANGLE
        3: objectname="H1 StdDev Channel 56931", objecttype=OBJ_STDDEVCHANNEL
        4: objectname="H1 Trendline 6605", objecttype=OBJ_TREND
*/
  }
```

The ForEachObject macro definition string has become a bit longer. Besides, it is more difficult to comprehend a single-string replacement code. But it turns out that this issue has already been solved as well: a macro definition can now be divided into strings using a backslash '\'. The result is as follows:

```
#define ForEachObject(name,i)   string name=ObjectName(0,0);   \
   int ob_total=ObjectsTotal(0);                               \
   for(int i=1;i<=ob_total;i++,name=ObjectName(0,i-1))
```

For the compiler, all these three strings look like a single long string, while becoming more comprehensible. Now, we need to create similar macros for working with trading entities — orders, positions and trades.

Let's start with searching for orders. Only the [HistorySelect](https://www.mql5.com/en/docs/trading/historyselect)() history selection function is added here:

```
#define ForEachOrder(ticket,i)    HistorySelect(0,TimeCurrent());    \
  ulong ticket=OrderGetTicket(0);                                    \
  int or_total=OrdersTotal();                                        \
  for(int i=1;i<or_total;i++,ticket=OrderGetTicket(i))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachOrder(orderticket,index)
     {
      Print(index,": #",orderticket," ",OrderGetString(ORDER_SYMBOL)," ",
            EnumToString((ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE)));
     }
/* Sample output
   1: #13965457 CADJPY ORDER_TYPE_SELL_LIMIT
   2: #14246567 AUDNZD ORDER_TYPE_SELL_LIMIT
*/
  }
```

You can notice that there is no error handling in this macro (and the two previous ones). For example, what if HistorySelect() returns false? In this case, we will not be able to pass through all orders in the loop. Besides, who actually analyzes HistorySelect() execution result? Thus, this macro contains nothing that might be forbidden for the usual way of developing a program.

The most strongest criticism of using #define is the fact that macro substitutions do not allow for [code debugging](https://www.metatrader5.com/en/metaeditor/help/development/debug "https://www.metatrader5.com/en/metaeditor/help/development/debug"). I agree with this, although, as fxsaber says, "A reliably fixed patient requires no anesthesia debugged macro requires no debugging".

Next, let's develop the macro for searching positions in a similar way:

```
#define ForEachPosition(ticket,i) HistorySelect(0,TimeCurrent());    \
   ulong ticket=PositionGetTicket(0);                                \
   int po_total=PositionsTotal();                                    \
   for(int i=1;i<=po_total;i++,ticket=PositionGetTicket(i-1))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachPosition(positionid,index)
     {
      Print(index,": ",PositionGetString(POSITION_SYMBOL)," postionID #",positionid);
     }
/* Sample output
   1: AUDCAD postionID #13234934
   2: EURNZD postionID #13443909
   3: AUDUSD postionID #14956799
   4: EURUSD postionID #14878673
*/
```

Search trades in history:

```
#define ForEachDeal(ticket,i) HistorySelect(0,TimeCurrent());        \
   ulong ticket=HistoryDealGetTicket(0);                             \
   int total=HistoryDealsTotal();                                    \
   for(int i=1;i<=total;i++,ticket=HistoryDealGetTicket(i-1))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   //---
   ForEachDeal(dealticket,index)
     {
      Print(index,": deal #",dealticket,",  order ticket=",
            HistoryDealGetInteger(dealticket,DEAL_ORDER));
     }
  }
```

Search orders in history:

```
#define ForEachHistoryOrder(ticket,i) HistorySelect(0,TimeCurrent());\
   ulong ticket=HistoryOrderGetTicket(0);                            \
   int total=HistoryOrdersTotal();                                   \
   for(int i=1;i<=total;i++,ticket=HistoryOrderGetTicket(i-1))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachHistoryOrder(historyorderticket,index)
     {
      Print(index,": #",historyorderticket);
     }
  }
```

Collect all macro substitutions in a single ForEach.mqh4 file:

```
//+------------------------------------------------------------------+
//|                                                      ForEach.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Market Watch symbols search loop                                 |
//+------------------------------------------------------------------+
#define ForEachSymbol(symbol,i)  string symbol=SymbolName(0,true);   \
   int os_total=SymbolsTotal(true);                                  \
   for(int i=1;i<os_total;i++,symbol=SymbolName(i,true))
//+------------------------------------------------------------------+
//| Chart's main window objects search loop                          |
//+------------------------------------------------------------------+
#define ForEachObject(name,i)   string name=ObjectName(0,0);         \
   int ob_total=ObjectsTotal(0);                                     \
   for(int i=1;i<=ob_total;i++,name=ObjectName(0,i-1))
//+------------------------------------------------------------------+
//| Active orders search loop                                        |
//+------------------------------------------------------------------+
#define ForEachOrder(ticket,i)    HistorySelect(0,TimeCurrent());    \
   ulong ticket=OrderGetTicket(0);                                   \
   int or_total=OrdersTotal();                                       \
   for(int i=1;i<or_total;i++,ticket=OrderGetTicket(i))
//+------------------------------------------------------------------+
//| Open positions search loop                                       |
//+------------------------------------------------------------------+
#define ForEachPosition(ticket,i) HistorySelect(0,TimeCurrent());    \
   ulong ticket=PositionGetTicket(0);                                \
   int po_total=PositionsTotal();                                    \
   for(int i=1;i<=po_total;i++,ticket=PositionGetTicket(i-1))
//+------------------------------------------------------------------+
//| History trades search loop                                       |
//+------------------------------------------------------------------+
#define ForEachDeal(ticket,i) HistorySelect(0,TimeCurrent());        \
   ulong ticket=HistoryDealGetTicket(0);                             \
   int dh_total=HistoryDealsTotal();                                 \
   for(int i=1;i<=dh_total;i++,ticket=HistoryDealGetTicket(i-1))
//+------------------------------------------------------------------+
//| History orders search loop                                       |
//+------------------------------------------------------------------+
#define ForEachHistoryOrder(ticket,i) HistorySelect(0,TimeCurrent());\
   ulong ticket=HistoryOrderGetTicket(0);                            \
   int oh_total=HistoryOrdersTotal();                                \
   for(int i=1;i<=oh_total;i++,ticket=HistoryOrderGetTicket(i-1))
//+------------------------------------------------------------------+
```

Note: we had to add a prefix for the 'total' variable of each macro, so that there are no conflicts if we decide to use more than one macro in our code. This is the biggest disadvantage of this macro: we hide the variable declaration inside it, while that variable is visible from the outside. This can lead to hard-to-detect errors when compiling.

In addition, when using a parametric macro, the compiler does not give any hints as it does for functions. You will have to learn these 6 macros by heart if you want to use them. Although this is not so difficult, since the first parameter is always an entity that is looped, and the second parameter is always the index of the loop, which starts with 1 (one).

Finally, let's add more macros to search in a reverse order. In this case, we need to move from the end of the list to its beginning. Add the Back suffix to the macro name and make some minor changes. Here is how the macro for searching through Market Watch symbols looks like.

```
#define ForEachSymbolBack(symbol,i) int s_start=SymbolsTotal(true)-1;\
   string symbol=SymbolName(s_start,true);                           \
   for(int i=s_start;i>=0;i--,symbol=SymbolName(i,true))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachSymbolBack(symbol,index)
     {
      //--- prepare data
      double spread=SymbolInfoDouble(symbol,SYMBOL_ASK)-SymbolInfoDouble(symbol,SYMBOL_BID);
      double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
      long digits=SymbolInfoInteger(symbol,SYMBOL_DIGITS);
      string str_spread=DoubleToString(spread/point,0);
      string str_point=DoubleToString(point,digits);
      //--- output data
      Print(index,". ",symbol," spread=",str_spread," points (",
            digits," digits",", point=",str_point,")");
     }
/* Sample output
   3. USDJPY spread=5 points (3 digits, point=0.001)
   2. USDCHF spread=8 points (5 digits, point=0.00001)
   1. GBPUSD spread=9 points (5 digits, point=0.00001)
   0. EURUSD spread=2 points (5 digits, point=0.00001)
*/
  }
```

As you can see, the value of the _index_ variable changes from _size-1_ to 0 here. We have completed our acquaintance with [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant). Now, it is time to develop functions for easy MQL4-style access.

### 1\. What groups of MQL4 functions are described in the article

Note: Replace such variables as Point, Digits and Bar with Point(), Digits() and Bar(Symbol(),Period())

The MQL4 AccountXXXX, MQL4 MarketInfo, MQL4 Status check and MQL4 Predefined variables groups are to be converted into MQL5. Thus, the four files: **AccountInfo.mqh**, **MarketInfo.mqh**, **Check.mqh** and **Predefined.mqh** _are to be added to \[date folder\]\\MQL5\\Include\\SimpleCall\\._ There are seven files in the folder to convert MQL4 functions into MQL5: **AccountInfo.mqh, Check.mqh, IndicatorsMQL4.mqh, IndicatorsMQL5.mqh, MarketInfo.mqh, Predefined.mqh** and **Series.mqh**.

You should include all these files. Also, please note the limitation: **IndicatorsMQL4.mqh** and **IndicatorsMQL5.mqh** files cannot be included together — you only can choose one of them. Therefore, the folder features two files: SimpleCallMQL4.mqh includes all files plus **IndicatorsMQL4.mqh** and SimpleCallMQL5.mqh includes all files plus **IndicatorsMQL5.mqh**.

**Example of including based on MACD Sample.mq4**

Copy MACD Sample.mq4 to MQL5 folder with EAs — for example, to \[data folder\]\\MQL5\\Experts\\, and change the file extension to mq5. Thus, we obtain **MACD Sample.mq5** file. Compile and get 59 errors and one warning.

Now, connect SimpleCallMQL4.mqh:

```
//+------------------------------------------------------------------+
//|                                                  MACD Sample.mq4 |
//|                   Copyright 2005-2014, MetaQuotes Software Corp. |
//|                                              http://www.mql4.com |
//+------------------------------------------------------------------+
#property copyright   "2005-2014, MetaQuotes Software Corp."
#property link        "http://www.mql4.com"
//---
#include <SimpleCall\SimpleCallMQL4.mqh>
//---
input double TakeProfit    =50;
```

Compile again and get 39 errors and one warning. Now, manually replace ( **Ctrl+H**) Bars with Bars(Symbol(),Period()) and Point with Point(). Compile. 35 errors remain. They are all related to trading functions. But we will not talk about trading functions in this article.

**1.1. MQL4 AccountXXXX**

MQL4 AccountXXXX functions are converted in the file \[date folder\]\\MQL5\\Include\\SimpleCall\ **AccountInfo.mqh**

The table of matching MQL4 functions with MQL5 ones looks as follows:

| MQL4 | MQL5 AccountInfoXXXX | MQL5 [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) | Notes |
| --- | --- | --- | --- |
| AccountInfoDouble | AccountInfoDouble | CAccountInfo::InfoDouble or CAccountInfo::double methods |  |
| AccountInfoInteger | AccountInfoInteger | CAccountInfo::InfoInteger or CAccountInfo::integer methods |  |
| AccountInfoString | AccountInfoString | CAccountInfo::InfoString or CAccountInfo::text methods |  |
| AccountBalance | AccountInfoDouble(ACCOUNT\_BALANCE) or | CAccountInfo::Balance |  |
| AccountCredit | AccountInfoDouble(ACCOUNT\_CREDIT) or | CAccountInfo::Credit |  |
| AccountCompany | AccountInfoString(ACCOUNT\_COMPANY) or | CAccountInfo::Company |  |
| AccountCurrency | AccountInfoString(ACCOUNT\_CURRENCY) or | CAccountInfo::Currency |  |
| AccountEquity | AccountInfoDouble(ACCOUNT\_EQUITY) or | CAccountInfo::Equity |  |
| AccountFreeMargin | AccountInfoDouble(ACCOUNT\_FREEMARGIN) or | CAccountInfo::FreeMargin |  |
| AccountFreeMarginCheck | \-\-\- /--- | CAccountInfo::FreeMarginCheck |  |
| AccountFreeMarginMode | No equivalent | No equivalent |  |
| AccountLeverage | AccountInfoInteger(ACCOUNT\_LEVERAGE) | CAccountInfo::Leverage | In MQL4, it has the int type, in MQL5, it is long |
| AccountMargin | AccountInfoDouble(ACCOUNT\_MARGIN) | CAccountInfo::Margin |  |
| AccountName | AccountInfoString(ACCOUNT\_NAME) | CAccountInfo::Name |  |
| AccountNumber | AccountInfoInteger(ACCOUNT\_LOGIN) | CAccountInfo::Login | In MQL4, it has the int type, in MQL5, it is long |
| AccountProfit | AccountInfoDouble(ACCOUNT\_PROFIT) | CAccountInfo::Profit |  |
| AccountServer | AccountInfoString(ACCOUNT\_SERVER) | CAccountInfo::Server |  |
| AccountStopoutLevel | AccountInfoDouble(ACCOUNT\_MARGIN\_SO\_SO) | CAccountInfo::MarginStopOut | In MQL4, it has the int type, in MQL5, it is double |
| AccountStopoutMode | AccountInfoInteger(ACCOUNT\_MARGIN\_SO\_MODE) | CAccountInfo::StopoutMode |  |

Please note that MQL5 has no equivalent for MQL4 AccountFreeMarginMode. You use MQL4 AccountFreeMarginMode further at your own risk. When detecting MQL4 AccountFreeMarginMode, a warning is sent to the log and NaN ("not a number") is returned.

For other MQL4 AccountXXXX functions, there are equivalents in two versions: via AccountInfoXXXX or via CAccountInfo trading class. For AccountInfoDouble, AccountInfoInteger and AccountInfoString, there are no differences between MQL4 and MQL5.

**The file header has the following code. How it works?**

```
//---
#define  OP_BUY                     ORDER_TYPE_BUY
#define  OP_SELL                    ORDER_TYPE_SELL
//--- returns balance value of the current account
```

First, let's have a look at [#define help:](https://www.mql5.com/en/docs/basis/preprosessor/constant)

The #define directive can be used to assign mnemonic names to constants. There are two forms:

```
#define identifier expression                   // parameter-free form
#define identifier(par1,... par8) expression    // parametric form
```

The #define directive substitutes 'expression' for all further found entries of 'identifier' in the source text.

In case of our code, this description looks as follows (we used parameter-free form): #define directive substitutes _**ORDER\_TYPE\_BUY**_ or all further found entries of _**OP\_BUY**_ in the source text. The #define directive substitutes _**ORDER\_TYPE\_SELL**_ for all further found entries of _**OP\_SELL**_ in the source text. In other words, after including the file \[date folder\]\\MQL5\\Include\\SimpleCall\ **AccountInfo.mqh** to your MQL5 EA, you can use _**OP\_BUY**_ and _**OP\_SELL**_ MQL4 types in a familiar way. The compiler will not return an error.

So, the first group of functions that we bring to the MQL5 type: AccountBalance(), AccountCredit(), AccountCompany(), AccountCurrency(), AccountEquity() and AccountFreeMargin():

```
//--- returns balance value of the current account
#define  AccountBalance(void)       AccountInfoDouble(ACCOUNT_BALANCE)
//--- returns credit value of the current account
#define  AccountCredit(void)        AccountInfoDouble(ACCOUNT_CREDIT)
//--- returns the brokerage company name where the current account was registered
#define  AccountCompany(void)       AccountInfoString(ACCOUNT_COMPANY)
//--- returns currency name of the current account
#define  AccountCurrency(void)      AccountInfoString(ACCOUNT_CURRENCY)
//--- returns equity value of the current account
#define  AccountEquity(void)        AccountInfoDouble(ACCOUNT_EQUITY)
//--- returns free margin value of the current account
#define  AccountFreeMargin(void)    AccountInfoDouble(ACCOUNT_MARGIN_FREE)
```

Here, in the MQL4 XXXX(void) functions, the parametric #define form is used where "void" acts as a parameter. This can be easily checked: if we remove "void", we get "unexpected in macro format parameter list" error during the compilation:

![unexpected in macro format parameter list](https://c.mql5.com/2/30/2018-01-31_07h40_54__1.png)

In case of MQL4 AccountFreeMarginCheck, we will act differently — set AccountFreeMarginCheck as a normal function with only an MQL5 code used inside it:

```
//--- returns free margin that remains after the specified order has been opened
//---    at the current price on the current account
double   AccountFreeMarginCheck(string symbol,int cmd,double volume)
  {
   double price=0.0;
   ENUM_ORDER_TYPE trade_operation=(ENUM_ORDER_TYPE)cmd;
   if(trade_operation==ORDER_TYPE_BUY)
      price=SymbolInfoDouble(symbol,SYMBOL_ASK);
   if(trade_operation==ORDER_TYPE_SELL)
      price=SymbolInfoDouble(symbol,SYMBOL_BID);
//---
   double margin_check=EMPTY_VALUE;
   double margin=EMPTY_VALUE;
   margin_check=(!OrderCalcMargin(trade_operation,symbol,volume,price,margin))?EMPTY_VALUE:margin;
//---
   return(AccountInfoDouble(ACCOUNT_FREEMARGIN)-margin_check);
  }
```

As we said above, AccountFreeMarginMode has no equivalent in MQL5, therefore, "not a number" and a warning are issued when AccountFreeMarginMode is detected in the code:

```
//--- returns the calculation mode of free margin allowed to open orders on the current account
double AccountFreeMarginMode(void)
  {
   string text="MQL4 functions \"AccountFreeMarginMode()\" has no analogs in MQL5. Returned \"NAN - not a number\"";
   Alert(text);
   Print(text);
   return(double("nan"));
  }
```

For other MQL4 functions, we proceed in a similar way:

```
//--- returns leverage of the current account
#define  AccountLeverage(void)      (int)AccountInfoInteger(ACCOUNT_LEVERAGE)
//--- returns margin value of the current account
#define  AccountMargin(void)        AccountInfoDouble(ACCOUNT_MARGIN)
//--- returns the current account name
#define  AccountName(void)          AccountInfoString(ACCOUNT_NAME)
//--- returns the current account number
#define  AccountNumber(void)        (int)AccountInfoInteger(ACCOUNT_LOGIN)
//--- returns profit value of the current account
#define  AccountProfit(void)        AccountInfoDouble(ACCOUNT_PROFIT)
//--- returns the connected server name
#define  AccountServer(void)        AccountInfoString(ACCOUNT_SERVER)
//--- returns the value of the Stop Out level
#define  AccountStopoutLevel(void)  (int)AccountInfoDouble(ACCOUNT_MARGIN_SO_SO)
//--- returns the calculation mode for the Stop Out level
int      AccountStopoutMode(void)
  {
   ENUM_ACCOUNT_STOPOUT_MODE stopout_mode=(ENUM_ACCOUNT_STOPOUT_MODE)AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
   if(stopout_mode==ACCOUNT_STOPOUT_MODE_PERCENT)
      return(0);
   return(1);
  }
```

**1.2. MQL4 MarketInfo**

MQL4 MarketInfotXXXX functions are converted in the file \[date folder\]\\MQL5\\Include\\SimpleCall\ **MarketInfo.mqh**

MQL4 MarketInfo is of double type, but in MQL5, the equivalents of MarketInfo are present in SymbolInfoInteger() (get the 'long' type) and in SymbolInfoDouble() (get the 'double' type). Here, we can clearly notice the awkwardness of using the MarketInfo (type conversion) obsolete function. Using MQL5 SYMBOL\_DIGITS as an example:

MarketInfo(Symbol(),MODE\_DIGITS) <= (double)SymbolInfoInteger(symbol,SYMBOL\_DIGITS)

**1.2.1 Ambiguity with MQL4 MODE\_TRADEALLOWED**

In MQL4, this is a simple 'bool' flag, while in MQL5, a symbol may have several kinds of permissions/prohibitions:

**ENUM\_SYMBOL\_TRADE\_MODE**

| ID | Description |
| --- | --- |
| SYMBOL\_TRADE\_MODE\_DISABLED | Disable trading on a symbol |
| SYMBOL\_TRADE\_MODE\_LONGONLY | Enable buys only |
| SYMBOL\_TRADE\_MODE\_SHORTONLY | Enable sells only |
| SYMBOL\_TRADE\_MODE\_CLOSEONLY | Enable close only |
| YMBOL\_TRADE\_MODE\_FULL | No trading limitations |

I propose 'false' only in case of SYMBOL\_TRADE\_MODE\_DISABLED, and 'true' for partial limitations or full access (accompanied by Alert and Print warning of partial limitations).

**1.2.2. Ambiguity with MQL4 MODE\_SWAPTYPE**

In MQL4, MODE\_SWAPTYPE returns only four values (swap calculation method. 0 — in points; 1 — in symbol's base currency; 2 — in %; 3 — in margin funds currency), whereas in MQL5, the ENUM\_SYMBOL\_SWAP\_MODE enumeration contains nine values having values similar to MQL4 ones:

| MQL4 MODE\_SWAPTYPE | MQL5 ENUM\_SYMBOL\_SWAP\_MODE |
| --- | --- |
| No equivalent | SYMBOL\_SWAP\_MODE\_DISABLED |
| 0 - in points | SYMBOL\_SWAP\_MODE\_POINTS |
| 1 - in symbol's base currency | SYMBOL\_SWAP\_MODE\_CURRENCY\_SYMBOL |
| 3 - in margin funds currency | SYMBOL\_SWAP\_MODE\_CURRENCY\_MARGIN |
| No equivalent | SYMBOL\_SWAP\_MODE\_CURRENCY\_DEPOSIT |
| 2 - in % | SYMBOL\_SWAP\_MODE\_INTEREST\_CURRENT |
| 2 - in % | SYMBOL\_SWAP\_MODE\_INTEREST\_OPEN |
| No equivalent | SYMBOL\_SWAP\_MODE\_REOPEN\_CURRENT |
| No equivalent | SYMBOL\_SWAP\_MODE\_REOPEN\_BID |

In MQL5, there may be two options for calculating swaps in %:

```
/*
SYMBOL_SWAP_MODE_INTEREST_CURRENT
Swaps are charged in annual % of the instrument price at the moment of swap calculation (banking mode is 360 days a year)
SYMBOL_SWAP_MODE_INTEREST_OPEN
Swaps are charged in annual % of a position opening price by symbol (banking mode is 360 days a year)
*/
```

Let's consider them as one type. For other MQL5 options having no equivalents in MQL4, a warning is issued and "not a number" is returned.

**1.2.3. Ambiguity with MQL4 MODE\_PROFITCALCMODE and MODE\_MARGINCALCMODE**

In MQL4, MODE\_PROFITCALCMODE (profit calculation method) returns only three values: 0 — Forex; 1 — CFD; 2 — Futures, while MODE\_MARGINCALCMODE (margin calculation method) returns four: 0 — Forex; 1 — CFD; 2 — Futures; 3 — CFDs for indices. In MQL5, it is possible to define the margin funds calculation method for an instrument (ENUM\_SYMBOL\_CALC\_MODE enumeration), but there is no way to calculate the profit. I assume that in MQL5, the method of calculating the margin funds is equal to the profit calculation method, therefore, a similar value with MQL5 ENUM\_SYMBOL\_CALC\_MODE is returned for MQL4 MODE\_PROFITCALCMODE and MODE\_MARGINCALCMODE.

The MQL5 ENUM\_SYMBOL\_CALC\_MODE enumeration contains ten methods. Let's compare MQL4 MODE\_PROFITCALCMODE with MQL5 ENUM\_SYMBOL\_CALC\_MODE:

MQL4 MODE\_PROFITCALCMODE "Forex" <==> MQL5 SYMBOL\_CALC\_MODE\_FOREX

MQL4 MODE\_PROFITCALCMODE "CFD" <==> MQL5 SYMBOL\_CALC\_MODE\_CFD

MQL4 MODE\_PROFITCALCMODE "Futures" <==> MQL5 SYMBOL\_CALC\_MODE\_FUTURES

For other MQL5 options having no equivalents in MQL4, a warning is issued and "not a number" is returned:

```
...
#define MODE_PROFITCALCMODE   1000//SYMBOL_TRADE_CALC_MODE
#define MODE_MARGINCALCMODE   1001//SYMBOL_TRADE_CALC_MODE
...
      case MODE_PROFITCALCMODE:
        {
         ENUM_SYMBOL_CALC_MODE profit_calc_mode=(ENUM_SYMBOL_CALC_MODE)SymbolInfoInteger(symbol,SYMBOL_TRADE_CALC_MODE);
         switch(profit_calc_mode)
           {
            case  SYMBOL_CALC_MODE_FOREX:
               return((double)0);
            case  SYMBOL_CALC_MODE_FUTURES:
               return((double)2);
            case  SYMBOL_CALC_MODE_CFD:
               return((double)1);
            default :
              {
               string text="MQL4 MODE_PROFITCALCMODE returned MQL5 "+EnumToString(profit_calc_mode);
               Alert(text);
               Print(text);
               return(double("nan"));
              }
           }
        }
      case MODE_MARGINCALCMODE:
        {
         ENUM_SYMBOL_CALC_MODE profit_calc_mode=(ENUM_SYMBOL_CALC_MODE)SymbolInfoInteger(symbol,SYMBOL_TRADE_CALC_MODE);
         switch(profit_calc_mode)
           {
            case  SYMBOL_CALC_MODE_FOREX:
               return((double)0);
            case  SYMBOL_CALC_MODE_FUTURES:
               return((double)2);
            case  SYMBOL_CALC_MODE_CFD:
               return((double)1);
            default :
              {
               string text="MQL4 MODE_MARGINCALCMODE returned MQL5 "+EnumToString(profit_calc_mode);
               Alert(text);
               Print(text);
               return(double("nan"));
              }
           }
        }
```

**1.3. MQL4 Status check**

Status check MQL4 function is converted in the file \[date folder\]\\MQL5\\Include\\SimpleCall\ **Check.mqh**

It includes the following elements:

- Digits
- Point
- IsConnected
- IsDemo
- IsDllsAllowed
- IsExpertEnabled
- IsLibrariesAllowed
- IsOptimization
- IsTesting
- IsTradeAllowed
- IsTradeContextBusy
- IsVisualMode
- TerminalCompany
- TerminalName
- TerminalPath

| MQL4 | MQL5 | MQL5 classes | Note |
| --- | --- | --- | --- |
| Digits |  |  | MQL4 allows using Digits and Digits() simultaneously |
| Point |  |  | MQL4 allows using Point and Point() simultaneously |
| IsConnected | TerminalInfoInteger(TERMINAL\_CONNECTED) | CTerminalInfo::IsConnected |  |
| IsDemo | AccountInfoInteger(ACCOUNT\_TRADE\_MODE) | CAccountInfo::TradeMode | Get one of the values from the ENUM\_ACCOUNT\_TRADE\_MODE enumeration |
| IsDllsAllowed | TerminalInfoInteger(TERMINAL\_DLLS\_ALLOWED) and MQLInfoInteger(MQL\_DLLS\_ALLOWED) | CTerminalInfo::IsDLLsAllowed | TERMINAL\_DLLS\_ALLOWED has the highest status, while MQL\_DLLS\_ALLOWED can be ignored |
| IsExpertEnabled | TerminalInfoInteger(TERMINAL\_TRADE\_ALLOWED) | CTerminalInfo::IsTradeAllowed | The terminal's AutoTrading button's status |
| IsLibrariesAllowed | MQLInfoInteger(MQL\_DLLS\_ALLOWED) | -/- | The check is pointless: if the program applies DLLs and you do not allow using them (dependencies tab of the program), you will simply not be able to run the program. |
| IsOptimization | MQLInfoInteger(MQL\_OPTIMIZATION) | -/- | The MQL5 program has four modes: debugging, code profiling, tester and optimization |
| IsTesting | MQLInfoInteger(MQL\_TESTER) | -/- | The MQL5 program has four modes: debugging, code profiling, tester and optimization |
| IsTradeAllowed | MQLInfoInteger(MQL\_TRADE\_ALLOWED) | -/- | "Allow automated trading" checkbox status in the program properties |
| IsTradeContextBusy | -/- | -/- | "false" is returned |
| IsVisualMode | MQLInfoInteger(MQL\_VISUAL\_MODE) |  | The MQL5 program has four modes: debugging, code profiling, tester and optimization |
| TerminalCompany | TerminalInfoString(TERMINAL\_COMPANY) | CTerminalInfo::Company |  |
| TerminalName | TerminalInfoString(TERMINAL\_NAME) | CTerminalInfo::Name |  |
| TerminalPath | TerminalInfoString(TERMINAL\_PATH) | CTerminalInfo::Path |  |

The first two points (Digits and Point) cannot be implemented, since MQL4 and MQL5 are completely confused here. In particular, MQL4 may face Digits and Digits(), as well as Point and Point() simultaneously. For example, I do not know how to covert Digits and Digits() to Digits() via 'define'. The remaining points are found as XXXX() and can be replaced with MQL5 equivalents.

The implementation is as follows:

```
//+------------------------------------------------------------------+
//|                                                        Check.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "http://wmua.ru/slesar/"
#property version   "1.003"
//--- checks connection between client terminal and server
#define IsConnected        (bool)TerminalInfoInteger(TERMINAL_CONNECTED)
//--- checks if the Expert Advisor runs on a demo account
#define IsDemo             (bool)(AccountInfoInteger(ACCOUNT_TRADE_MODE)==(ENUM_ACCOUNT_TRADE_MODE)ACCOUNT_TRADE_MODE_DEMO)
//--- checks if the DLL function call is allowed for the Expert Advisor
#define IsDllsAllowed      (bool)TerminalInfoInteger(TERMINAL_DLLS_ALLOWED)
//--- checks if Expert Advisors are enabled for running
#define IsExpertEnabled    (bool)TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)
//--- checks if the Expert Advisor can call library function
#define IsLibrariesAllowed (bool)MQLInfoInteger(MQL_DLLS_ALLOWED)
//--- checks if Expert Advisor runs in the Strategy Tester optimization mode
#define IsOptimization     (bool)MQLInfoInteger(MQL_OPTIMIZATION)
//--- checks if the Expert Advisor runs in the testing mode
#define IsTesting                (bool)MQLInfoInteger(MQL_TESTER)
//--- checks if the Expert Advisor is allowed to trade and trading context is not busy
#define IsTradeAllowed     (bool)MQLInfoInteger(MQL_TRADE_ALLOWED)
//--- returns the information about trade context
#define IsTradeContextBusy  false
//--- checks if the Expert Advisor is tested in visual mode
#define IsVisualMode          (bool)MQLInfoInteger(MQL_VISUAL_MODE)
//--- returns the name of company owning the client terminal
#define TerminalCompany    TerminalInfoString(TERMINAL_COMPANY)
//--- returns client terminal name
#define TerminalName          TerminalInfoString(TERMINAL_NAME)
//--- returns the directory, from which the client terminal was launched
#define TerminalPath          TerminalInfoString(TERMINAL_PATH)
//+------------------------------------------------------------------+
```

**1.4. MQL4 Predefined variables**

Implementation in the file \[data folder\]\\MQL5\\Include\\SimpleCall\ **Predefined.mqh**

MQL4 predefined variables:

- \_Digits
- \_Point
- \_LastError
- \_Period
- \_RandomSeed
- \_StopFlag
- \_Symbol
- \_UninitReason
- Ask
- Bars
- Bid
- Close
- Digits
- High
- Low
- Open
- Point
- Time
- Volume

\_XXXX predefined variables are converted into MQL5 functions using #define non-parametric form:

```
//--- the _Digits variable stores number of digits after a decimal point,
#define _Digits         Digits()
//--- the _Point variable contains the point size of the current symbol in the quote currency
#define _Point          Point()
//--- the _LastError variable contains code of the last error
#define _LastError      GetLastError()
//--- the _Period variable contains the value of the timeframe of the current chart
#define _Period         Period()
//#define _RandomSeed
//--- the _StopFlag variable contains the flag of the program stop
#define _StopFlag       IsStopped()
//--- the _Symbol variable contains the symbol name of the current chart
#define _Symbol         Symbol()
//--- the _UninitReason variable contains the code of the program uninitialization reason
#define _UninitReason   UninitializeReason()
//#define Bars            Bars(Symbol(),Period());
//#define Digits
//#define Point
```

The only exception is made for "\_RandomSeed" — this variable stores the current status of the pseudo-random integer generator. It cannot be converted into MQL5 for Bars (more precisely, Bars is left for manual replacement). There is no solution for Digits and Point. As mentioned above, Digits and Digits(), as well as Point and Point() can be found in the text simultaneously.

MQL4 Ask and Bid are replaced with the custom (self-written) GetAsk() and GetBid() functions:

```
//--- the latest known seller's price (ask price) for the current symbol
#define Ask             GetAsk()
//--- the latest known buyer's price (offer price, bid price) of the current symbol
#define Bid             GetBid()
...
//--- the latest known seller's price (ask price) for the current symbol
double GetAsk()
  {
   MqlTick tick;
   SymbolInfoTick(Symbol(),tick);
   return(tick.ask);
  }
//--- the latest known buyer's price (offer price, bid price) of the current symbol
double GetBid()
  {
   MqlTick tick;
   SymbolInfoTick(Symbol(),tick);
   return(tick.bid);
  }
```

We could, of course, write a macro for the Ask in a simpler way:

```
#define Ask SymbolInfoDouble(__Symbol,SYMBOL_ASK)
```

But the note for [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble "SymbolInfoDouble") says:

If the function is used to obtain data on the last tick, it would be better to apply [SymbolInfoTick()](https://www.mql5.com/en/docs/marketinformation/symbolinfotick). It is possible that there are no quotes on a symbol since the terminal is connected to a trading account. In this case, the requested value will be undefined.

In most cases, it is sufficient to use the [SymbolInfoTick()](https://www.mql5.com/en/docs/marketinformation/symbolinfotick) function, which allows you to get the Ask, Bid, Last and Volume values per one call, as well as the last tick arrival time.

**Now, let's introduce something new — use " [operator](https://www.mql5.com/en/docs/basis/function/operationoverload)"**

The 'operator' keyword will be used to overload (re-assign) the \[\] indexation operator. This is necessary for translating MQL4 timeseries arrays (Open\[\], High\[\], Low\[\], Close\[\], Time\[\], Volume\[\]) in MQL5 form.

Here is what the help tells us about " [operator](https://www.mql5.com/en/docs/basis/function/operationoverload)":

Operation overloading allows the use of the operating notation (written in the form of simple expressions) for complex objects - structures and classes.

Thus, we can assume that we will need to create a class to overload the \[\] indexation operator.

Remembering [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant):

The #define directive can be used to assign mnemonic names to constants. There are two forms:

```
#define identifier expression                   // parameter-free form
#define identifier(par1,... par8) expression    // parametric form
```

The #define directive substitutes 'expression' for all further found entries of 'identifier' in the source text.

the following code can be read as: the #define directive substitutes 159for all further found entries in the source text.

```
#define SeriesVolume(Volume,T) 159
//+-----------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   long a=SeriesVolume(Volume,long);
  }
```

In other words, the code in OnStart is converted into:

```
   long a=159;
```

**Step 2**

The entire class has been placed into #define here,

```
//+------------------------------------------------------------------+
//|                                                      Test_en.mq5 |
//|                                      Copyright 2012, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#define SeriesVolume(Volume,T) class CVolume       \
  {                                                \
  public:                                          \
    T operator[](const int i) const                \
    {                                              \
    long val[1];                                   \
    if(CopyTickVolume(Symbol(),Period(),i,1,val)==1)\
      return(val[0]);                              \
    else                                           \
      return(-1);                                  \
    }                                              \
  };                                               \
CVolume Volume;
//---
SeriesVolume(Volume,long)
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   Print(Volume[4]);
  }
//+------------------------------------------------------------------+
```

This substitution can be represented as follows:

```
//+------------------------------------------------------------------+
//|                                                      Test_en.mq5 |
//|                                      Copyright 2012, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
class CVolume
  {
  public:
    long operator[](const int i) const
    {
    long val[1];
    if(CopyTickVolume(Symbol(),Period(),i,1,val)==1)
      return(val[0]);
    else
      return(-1);
    }
  };
CVolume Volume;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   Print(Volume[4]);
  }
```

In other words, in OnStart, we actually refer to the Volume object of the CVolume class, to the \[\] method, where we pass the i index. The same principle is applied for Open, High, Low, Close and Time MQL4 series.

Finally, we have MQL4 iXXX functions: iOpen, iHigh, iLow, iClose, iTime and iVolume. We should apply the custom function declaration method for them.

Example for iClose:

```
//--- returns Close price value for the bar of specified symbol with timeframe and shift
double   iClose(
                string                    symbol,              // symbol
                ENUM_TIMEFRAMES           timeframe,           // timeframe
                int                       shift                // shift
                )
  {
   double result=0.0;
//---
   double val[1];
   ResetLastError();
   int copied=CopyClose(symbol,timeframe,shift,1,val);
   if(copied>0)
      result=val[0];
   else
      Print(__FUNCTION__,": CopyClose error=",GetLastError());
//---
   return(result);
  }
```

**2\. Changes in other files**

In \[data folder\]\\MQL5\\Include\\SimpleCall\ **IndicatorsMQL4**.mqh, all MQL4 line names are now set in the header:

```
double NaN=double("nan");
#define MODE_MAIN          0
#define MODE_SIGNAL        1
#define MODE_PLUSDI        1
#define MODE_MINUSDI       2
#define MODE_GATORJAW      1
#define MODE_GATORTEETH    2
#define MODE_GATORLIPS     3
#define MODE_UPPER         1
#define MODE_LOWER         2
#define MODE_TENKANSEN     1
#define MODE_KIJUNSEN      2
#define MODE_SENKOUSPANA   3
#define MODE_SENKOUSPANB   4
#define MODE_CHIKOUSPAN    5
```

In \[data folder\]\\MQL5\\Include\\SimpleCall\ **Series.mqh**, we have removed

```
#define MODE_OPEN    0
//#define MODE_LOW     1
//#define MODE_HIGH    2
#define MODE_CLOSE   3
#define MODE_VOLUME  4
//#define MODE_TIME    5
```

Now, they are written in \[data folder\]\\MQL5\\Include\\SimpleCall\ **Header.mqh**:

```
//+------------------------------------------------------------------+
//|                                                       Header.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                           http://wmua.ru/slesar/ |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "http://wmua.ru/slesar/"
//---
#define MODE_LOW     10001
#define MODE_HIGH    10002
#define MODE_TIME    10005
```

as well as iClose, iHigh, iLow, iOpen, iTime and iVolume — now they are set in \[data folder\]\\MQL5\\Include\\SimpleCall\\Predefined.mqh

### Conclusion

In the [previous article](https://www.mql5.com/en/articles/4318), we discussed how to write indicator calls in MQL4 style and described the consequences. It turned out that the simplicity of writing entails a slowdown in the operation of EAs having no built-in control over created indicators. In this article, we continued to look for ways to simplify the code and had a look at the #define macro substitution.

As a result, you can make almost any MQL4 EA work in MetaTrader 5 using the codes attached to the article. You only need to include the necessary files that overload or add the necessary functions and predefined variables.

Only the simplified trading functions of MQL4 are missing for complete compatibility. But this issue can be solved as well. Let's repeat again the pros and cons of this approach as a conclusion:

**Cons:**

- limitation in processing the returned error when accessing indicators;
- drop in test speed when accessing more than one indicator simultaneously;

- the need to correctly highlight the indicator lines depending on whether IndicatorsMQL5.mqh or IndicatorsMQL4.mqh is connected;
- inability to debug the #define macro substitution;
- no tooltip on the parametric #define arguments;

- potential collisions of variables hidden behind macros.


**Pros**

- simplicity of code writing — one string instead of multiple ones;

- visibility and conciseness — the less the code amount, the easier it is for understanding;
- macro substitutions are highlighted in red making it easier to see user IDs and functions;
- ability to develop custom snippet equivalents.


I myself remain an adherent of the classical MQL5 approach and consider the methods described in the article as a lifehack. Perhaps, the articles will enable those accustomed to writing a code in MQL4 style to overcome the psychological barrier in their transition to MetaTrader 5 platform, which is much more convenient on all counts.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/4332](https://www.mql5.com/ru/articles/4332)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4332.zip "Download all attachments in the single ZIP archive")

[ForEachSymbol.mq5](https://www.mql5.com/en/articles/download/4332/foreachsymbol.mq5 "Download ForEachSymbol.mq5")(3.73 KB)

[ForEachObject.mq5](https://www.mql5.com/en/articles/download/4332/foreachobject.mq5 "Download ForEachObject.mq5")(2.82 KB)

[ForEachOrder.mq5](https://www.mql5.com/en/articles/download/4332/foreachorder.mq5 "Download ForEachOrder.mq5")(2.59 KB)

[ForEachPosition.mq5](https://www.mql5.com/en/articles/download/4332/foreachposition.mq5 "Download ForEachPosition.mq5")(2.91 KB)

[ForEachDeal.mq5](https://www.mql5.com/en/articles/download/4332/foreachdeal.mq5 "Download ForEachDeal.mq5")(1.24 KB)

[ForEachHistoryOrder.mq5](https://www.mql5.com/en/articles/download/4332/foreachhistoryorder.mq5 "Download ForEachHistoryOrder.mq5")(1.18 KB)

[SimpleCall.zip](https://www.mql5.com/en/articles/download/4332/simplecall.zip "Download SimpleCall.zip")(24.17 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717)
- [Gap - a profitable strategy or 50/50?](https://www.mql5.com/en/articles/5220)
- [Elder-Ray (Bulls Power and Bears Power)](https://www.mql5.com/en/articles/5014)
- [Improving Panels: Adding transparency, changing background color and inheriting from CAppDialog/CWndClient](https://www.mql5.com/en/articles/4575)
- [How to create a graphical panel of any complexity level](https://www.mql5.com/en/articles/4503)
- [Comparing speeds of self-caching indicators](https://www.mql5.com/en/articles/4388)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/233243)**
(62)


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
25 Apr 2020 at 02:45

```
#define ForEachSymbol(s,i)  string s=SymbolName(0,true); int os_total=SymbolsTotal(true); for(int i=1;i<os_total;i++,s=SymbolName(i,true))
```

There is a bug, first symbol process is with position\_index 0, next one is with position\_index 2. You are missing position\_index 1 and your loop is executed only os\_total-1 times.

```
#define ForEachOrder(ticket,i)    HistorySelect(0,TimeCurrent());  ulong ticket=OrderGetTicket(0); int or_total=OrdersTotal();   for(int i=1;i<or_total;i++,ticket=OrderGetTicket(i))
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   ForEachOrder(orderticket,index)
     {
      Print(index,": #",orderticket," ",OrderGetString(ORDER_SYMBOL)," ",
            EnumToString((ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE)));
     }
/* Sample output
   1: 13965457  CADJPY ORDER_TYPE_SELL_LIMIT
   2: 14246567  AUDNZD ORDER_TYPE_SELL_LIMIT
*/
  }
```

In this one there is the same bug as previously.

And additionally you are mixing functions to work on open orders and history selection. If your intend was to work with open orders, there is no need to use HistorySelect().

* * *

By the way these bugs demonstrate well the problem with without macro. It's hard or impossible to debug.

The most strongest criticism of using #define is the fact that macro substitutions do not allow for [code debugging](https://www.metatrader5.com/en/metaeditor/help/development/debug "https://www.metatrader5.com/en/metaeditor/help/development/debug"). I agree with this, although, as fxsaber says, "A reliably fixed patient requires no anesthesiadebugged macro requires no debugging".

![](https://c.mql5.com/3/317/hysterical_40x40.gif)

I stopped reading after that, sorry.

![daengrani](https://c.mql5.com/avatar/avatar_na2.png)

**[daengrani](https://www.mql5.com/en/users/daengrani)**
\|
26 Feb 2022 at 10:39

**MetaQuotes:**

New article [LifeHack for traders: Blending ForEach with defines (#define)](https://www.mql5.com/en/articles/4332) has been published:

Author: [Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn "barabashkakvn")

maybe you considered to add 2 more defines for mql4, OrderCalcProfit() and OrderCalcMargin()


![Vladimir Karputov](https://c.mql5.com/avatar/2024/2/65d8b5a2-f9d9.jpg)

**[Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn)**
\|
26 Feb 2022 at 10:47

**daengrani [#](https://www.mql5.com/en/forum/233243#comment_27974188):**

maybe you considered to add 2 more defines for mql4, OrderCalcProfit() and OrderCalcMargin()

No.  The old terminal has not been supported for a long time.


![William Oswaldo Mayorga Urduy](https://c.mql5.com/avatar/2021/1/5FF7F514-1B37.jpg)

**[William Oswaldo Mayorga Urduy](https://www.mql5.com/en/users/oswaldomayorgau)**
\|
17 Aug 2022 at 04:04

**MetaQuotes:**

Published article [LifeHack for trailerers: cooking ForEach using #define](https://www.mql5.com/en/articles/4332):

Author: [Vladimir Karputov](https://www.mql5.com/en/users/barabashkakvn "barabashkakvn")

very good article, it helps a lot.

![StrikerHoot](https://c.mql5.com/avatar/avatar_na2.png)

**[StrikerHoot](https://www.mql5.com/en/users/strikerhoot)**
\|
12 Dec 2024 at 19:32

To Vladimir Karputov,

Thanks for your delicated illustaration of macros!

Learned a lot from this article!

![Money Management by Vince. Implementation as a module for MQL5 Wizard](https://c.mql5.com/2/30/MQL5-avatar-capital-001.png)[Money Management by Vince. Implementation as a module for MQL5 Wizard](https://www.mql5.com/en/articles/4162)

The article is based on 'The Mathematics of Money Management' by Ralph Vince. It provides the description of empirical and parametric methods used for finding the optimal size of a trading lot. Also the article features implementation of trading modules for the MQL5 Wizard based on these methods.

![Controlled optimization: Simulated annealing](https://c.mql5.com/2/31/icon__1.png)[Controlled optimization: Simulated annealing](https://www.mql5.com/en/articles/4150)

The Strategy Tester in the MetaTrader 5 trading platform provides only two optimization options: complete search of parameters and genetic algorithm. This article proposes a new method for optimizing trading strategies — Simulated annealing. The method's algorithm, its implementation and integration into any Expert Advisor are considered. The developed algorithm is tested on the Moving Average EA.

![How to create Requirements Specification for ordering an indicator](https://c.mql5.com/2/31/Spec_Indicator.png)[How to create Requirements Specification for ordering an indicator](https://www.mql5.com/en/articles/4304)

Most often the first step in the development of a trading system is the creation of a technical indicator, which can identify favorable market behavior patterns. A professionally developed indicator can be ordered from the Freelance service. From this article you will learn how to create a proper Requirements Specification, which will help you to obtain the desired indicator faster.

![LifeHack for traders: Fast food made of indicators](https://c.mql5.com/2/30/LifeHack_MQL4.png)[LifeHack for traders: Fast food made of indicators](https://www.mql5.com/en/articles/4318)

If you have newly switched to MQL5, then this article will be useful. First, the access to the indicator data and series is done in the usual MQL4 style. Second, this entire simplicity is implemented in MQL5. All functions are as clear as possible and perfectly suited for step-by-step debugging.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/4332&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083338178804324775)

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