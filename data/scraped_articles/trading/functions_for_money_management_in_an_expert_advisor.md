---
title: Functions for Money Management in an Expert Advisor
url: https://www.mql5.com/en/articles/113
categories: Trading
relevance_score: 6
scraped_at: 2026-01-23T11:30:30.406240
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/113&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062517109361124260)

MetaTrader 5 / Trading


### Introduction

The MQL5 language provides an opportunity to obtain a vast amounts of information about the current [terminal conditions](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus), of the [mql5-program](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info), as well as the [financial instrument](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) and the [trading account](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants). In order to organize the functions of capital management, we will need to study the properties from the last two listed sections, as well as acquaint ourselves with the following functions:

- [SymbolInfoInteger()](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger)
- [SymbolInfoDouble()](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble)
- [SymbolInfoString()](https://www.mql5.com/en/docs/marketinformation/symbolinfostring)
- [AccountInfoInteger()](https://www.mql5.com/en/docs/account/accountinfointeger)
- [AccountInfoDouble()](https://www.mql5.com/en/docs/account/accountinfodouble)
- [AccountInfoString()](https://www.mql5.com/en/docs/account/accountinfostring)

Although in this article, the main focus is kept on the use of functions in the Expert Advisors, all of these descriptions can be applied to indicators and scripts.

### Obtaining information about your account balance

The first two important characteristics of a trading account - the balance and equity. To obtain these values, use the AccountInfoDouble() function:

```
   double balance=AccountInfoDouble(ACCOUNT_BALANCE);
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
```

The next thing that interests us, is the size of the deposit funds for open positions, and the total floating profit or loss on the account, for all open positions.


```
   double margin=AccountInfoDouble(ACCOUNT_MARGIN);
   double float_profit=AccountInfoDouble(ACCOUNT_PROFIT);
```

In order to be able to open new positions or strengthen the existing ones, we need free resources, not participating in the deposit.

```
   double free_margin=AccountInfoDouble(ACCOUNT_FREEMARGIN);
```

Here it should be noted, that the above values are expressed in monetary terms.

Monetary values, returned by the AccountInfoDouble() function, are expressed in **deposit currency**. To find out the deposit currency, use the [AccountInfoString()](https://www.mql5.com/en/docs/account/accountinfostring) function.

```
string account_currency=AccountInfoString(ACCOUNT_CURRENCY);
```

**The level of personal funds**

The account has another important characteristic - the level at which the event Stop Out occurs (a mandatory closing of a positions due to a shortage of personal funds necessary for maintaining open positions). To obtain this value, re-use the AccountInfoDouble() function:


```
double stopout_level=AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
```

The function only returns the value itself, but it doesn't explain what type of units this value is expressed in. There are two modes of level specification for Stop Out: in percents and in currency. In order to find this out, use the AccountInfoInteger() function:


```
//--- Get account currency
string account_currency=AccountInfoString(ACCOUNT_CURRENCY);

//--- Stop Out level
   double stopout_level=AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);

//--- Stop Out mode
   ENUM_ACCOUNT_STOPOUT_MODE so_mode=(ENUM_ACCOUNT_STOPOUT_MODE)AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
   if(so_mode==ACCOUNT_STOPOUT_MODE_PERCENT)

      PrintFormat("Stop Out level in percents %.2f%%",stopout_level);
   else
      PrintFormat("Stop Out level in currency %.2f %s",stopout_level,account_currency);
```

### Additional information about the account

Often in calculations it required to know the size of the provided on the trading account leverage. You can obtain this information by using the AccountInfoInteger() function:


```
   int leverage=(int)AccountInfoInteger(ACCOUNT_LEVERAGE);
```

In order to avoid accidentally running the unregulated Expert Advisor on a real account, you need to know the type of the account.


```
   ENUM_ACCOUNT_TRADE_MODE mode=(ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE);
   switch(mode)
     {
      case ACCOUNT_TRADE_MODE_DEMO:    Comment("Account demo");               break;
      case ACCOUNT_TRADE_MODE_CONTEST: Comment(com,"Account Contest");        break;
      case ACCOUNT_TRADE_MODE_REAL:    Comment(com,"Account Real");           break;
      default:                         Comment(com,"Account unknown type");
     }
```

Trading is not possible on every account, for example, on competitive accounts, trading operations can only be done after the beginning of the competition. This information can also be obtained by the AccountInfoInteger() function:


```
   bool trade_allowed=(bool)AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   if(trade_allowed)
      Print("Trade is allowed");
   else
      Print(com,"Trade is not allowed");
```

Even if trading on this account is permitted, it doesn't mean that the Expert Advisor has the right to trade. To check whether the Expert Advisor is permitted to trade, write:


```
   if(trade_allowed)
     {
      bool trade_expert=(bool)AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
      if(trade_expert)
         Print("Experts are allowed to trade");

      else
         Print("Experts are not allowed to trade");
```

These examples can be found in the attached Expert Advisor _Account\_Info.mq5_ . They can be used in MQL5 programs of any complexity.


![](https://c.mql5.com/2/1/fig1_en__1.png)

### Information about the instrument

Each financial instrument has its own descriptions and is placed on a path, which this instrument characterizes. If we open the EURUSD properties window in the terminal, we will see something like this:


![](https://c.mql5.com/2/1/symbol_properties_MQL5.png)

In this case, the description for EURUSD is - "EURUSD, Euro vs US Dollar". To obtain this information, we use the [SymbolInfoString()](https://www.mql5.com/en/docs/marketinformation/symbolinfostring) function:


```
   string symbol=SymbolInfoString(_Symbol,SYMBOL_DESCRIPTION);
   Print("Symbol: "+symbol);

   string symbol_path=SymbolInfoString(_Symbol,SYMBOL_PATH);
   Print("Path: "+symbol_path);
```

To find out the size of a standard contract, use the [SymbolInfoDouble()](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble):


```
   double lot_size=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_CONTRACT_SIZE);
   Print("Standard contract: "+DoubleToString(lot_size,2));
```

It is a characteristic of FOREX instruments to sell one currency while buying another. The contract is indicated in the currency, which is necessary to perform the purchase. This is a base currency, and it can be obtained using the SymbolInfoString() function:


```
   string base_currency=SymbolInfoString(_Symbol,SYMBOL_CURRENCY_BASE);
   Print("Base currency: "+base_currency);
```

Price changes in the instrument lead to a change in the price of a purchased assets, and therefore, to a profit variation for an open position (the profit can be negative if the position is losing). Thus, the price change leads to changes in income, expressed in a particular currency. This currency is called the quote currency. For a currency pair EURUSD the base currency is usually the Euro, and the quote currency is the U.S. dollar. To obtain the quote currency you can also use the SymbolInfoString() function:


```
   string profit_currency=SymbolInfoString(_Symbol,SYMBOL_CURRENCY_PROFIT);

   Print("Currency quotes: "+profit_currency);
```

To open a position on the instrument you need funds, and these funds are also expressed in a particular currency. This currency is called the currency margin or deposit. For the FOREX instruments the margin and the base currencies are usually the same. To obtain the value of the deposit currency, use the SymbolInfoString() function:


```
   string margin_currency=SymbolInfoString(_Symbol,SYMBOL_CURRENCY_MARGIN);
   Print("Currency deposit: "+margin_currency);
```

All of the described functions are given in the code of the _Symbol\_Info.mq5_ Expert Advisor. The figure below demonstrates the output of information on the symbol EURUSD, using the [Comment()](https://www.mql5.com/en/docs/common/comment) function.

![](https://c.mql5.com/2/1/fig3_en__1.png)

### Calculating the size of the deposit

The information about financial instruments, most needed for traders, is the size of the funds, required for opening a position on it. Without knowing how much money is needed to buy or sell a specified number of lots, we can not implement the Expert Advisor's system for capital managing. In addition, controlling the account balance also becomes difficult.

If you have difficulties with understanding the further discussion, I recommend you to read the article **[Forex Trading ABC](https://www.mql5.com/en/articles/1453)**. The explanations described in it are also applicable to this article.


We need to calculate the size of the margin in the currency deposit, ie recalculate the deposit from the mortgage currency to the deposit currency, by dividing the obtained value by the amount of the given account leverage. To do this we write the GetMarginForOpening() function:

```
//+------------------------------------------------------------------+
//|  Return amount of equity needed to open position                 |
//+------------------------------------------------------------------+
double GetMarginForOpening(double lot,string symbol,ENUM_POSITION_TYPE direction)
  {
   double answer=0;
//---
    ...
//--- Return result - amount of equity in account currency, required to open position in specified volume
   return(answer);
  }
```

where:


- lot - the volume of the open position;

- symbol - the name of the financial instrument;

- the alleged [position direction](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type).


So, we have the following information for calculating the size of the margin (monetary funds for mortgage of the open position):


- deposit currency

- mortgage currency

- Currency quotes (may be needed for Cross Currency pairs)

- the contract size


Write this in the MQL5 language:


```
//--- Get contract size
   double lot_size=SymbolInfoDouble(symbol,SYMBOL_TRADE_CONTRACT_SIZE);

//--- Get account currency
   string account_currency=AccountInfoString(ACCOUNT_CURRENCY);

//--- Margin currency
   string margin_currency=SymbolInfoString(_Symbol,SYMBOL_CURRENCY_MARGIN);

//--- Profit currency
   string profit_currency=SymbolInfoString(_Symbol,SYMBOL_CURRENCY_PROFIT);

//--- Calculation currency
   string calc_currency="";
//--- Reverse quote - true, Direct quote - false
   bool mode;
```

The mode variable affects how we will be calculating the size of the contract in the deposit currency. Consider this based on examples, in all further cases let's assume that the deposit currency is the US dollar.


The currency pairs are usually divided into three categories:

- Direct Currency Pairs - the U.S. dollar exchange rate to a particular currency. Examples: USDCHF, USDCAD, USDJPY, USDSEK;

- Reverse Currency Pair - the exchange rate of a particular currency to the U.S. dollar. Examples: EURUSD, GBPUSD, AUDUSD, NZDUSD;

- Cross Currency Pairs - a currency pair, which does not involve the U.S. dollar. Examples: AUDCAD, EURJPY, EURCAD.


**1\. EURUSD - the reverse currency pair**

We will call the currency pairs, in which the quote currency is the account currency, reverse currency pairs. In our examples, the account currency is represented with the U.S. dollar, so our classification of the currency pairs will coincide with the generally accepted classification. But if your trading account uses a different currency (not USD), it will not coincide. In this case, take into consideration the account currency, in order to understand any further explanations.

The contract size for EURUSD - 100 000 euros. We need to express 100 000 euros in the currency of the deposit - U.S. dollars. To do this you need to know the exchange rate, according to which the euro can be counted into dollars. We introduce the concept _computation currency_ , that is, the currency needed for converting the mortgage currency to the deposit currency.


```
//--- Calculation currency
   string calc_currency="";
```

Fortunately, the EURUSD currency pair displays the exchange rate of the euro against the dollar, and hence, for this case, the symbol of EURUSD, for which you need to calculate the mortgage size, is precisely the exchange rate:


```
//--- If profit currency and account currency are equal
   if(profit_currency==account_currency)
     {
      calc_currency=symbol;
      mode=true;
     }
```

We have established the value of the _mode_ as true, which means that for transferring Euros to dollars (mortgaged currency is convertible into the deposit currency), we will multiply the current exchange rate of EURUSD by the size of the contract. If _mode_  = false, then we divide the contract size by the exchange rate of the computational currency. For obtaining the current prices on the instrument, use the [SymbolInfoTick()](https://www.mql5.com/en/docs/marketinformation/symbolinfotick) function.

```
//--- We know calculation currency, let's get its last prices
   MqlTick tick;
   SymbolInfoTick(calc_currency,tick);
```

This function puts the current price and time of the last update of prices into the variableof the[MqlTick](https://www.mql5.com/en/docs/constants/structures/mqltick)type - thisstructurewasspecially designed for this purpose.

Therefore it is sufficient enough to obtain the latest price on this symbol, multiply it by the size of the contract and then by the number of lots. But which calculation price should we take, considering that there is a purchase price and a sale price for this instrument? so logically: if we are buying, the price for calculations is equal to the Ask price, and if we are selling, will need to take the Bid price.


```
//--- Now we have everything for calculation
   double calc_price;
//--- Calculate for Buy
   if(direction==POSITION_TYPE_BUY)
     {
      //--- Reverse quote
      if(mode)
        {
         //--- Calculate using Buy price for reverse quote
         calc_price=tick.ask;
         answer=lot*lot_size*calc_price;
        }
     }

//--- calculate for Sell
   if(direction==POSITION_TYPE_SELL)
     {
      //--- Reverse quote
      if(mode)
        {
         //--- Calculate using Sell price for reverse quote

         calc_price=tick.bid;
         answer=lot*lot_size*calc_price;
        }
     }
```

Thus, in our example, for the EURUSD symbol the deposit currency is Euro, the contract size is 100 000, and the last Ask price  = 1.2500. Account currency - U.S. dollar, and the calculation currency is the same EURUSD currency pair. Multiply 100 000 by 1.2500 and get 125 000 U.S. dollars - this is exactly how much a standard contract for purchasing 1 EURUSD lot is, if the Ask price =1.2500.

We can conclude that if the quote currency is equal to the account currency, then to obtain the value of one lot of the account currency, we simply multiply the size of the contract by the appropriate price, Bid or Ask, depending on the intended direction of the position.


```
margin=lots*lot_size*rate/leverage;
```

**2\. USDCHF -** **direct currency pair**

The mortgage currency and the account currency for USDCHF match - the U.S. dollar. The currency pairs, in which the mortgage currency and the account currency are the same, we will call direct currency pairs. Contract size - 100 000. This is the simplest situation, simply return the product.

```
//--- if the base currency symbol and the deposit currency are the same
   if(margin_currency==account_currency)
     {
      calc_currency=symbol;
      //--- Just return the contract value, multiplied by the number of lots
      return(lot*lot_size);
     }
```

If the deposit currency coincides with the account currency, then the value of the deposit in the account currency is equal to the product of the standard contract multiplied by the number of lots (contracts) divided by the size of the leverage.


**margin=lots\*lot\_size/leverage;**

**3\. CADCHF - cross- currency pair**

The CADCHF currency pair is taken for illustrative purposes, and any other pair, in which the deposit currency and the quote currency coincide with the account currency, can be used. These currency pairs are called cross, because in order to calculate the margin and profit on them, we need to know the exchange rate of some other currency pair, which intersects with that one on one of the currencies.

Usually, a cross-currency pairs are the pairs, the quotes of which do not use the U.S. dollar. But we will call all pairs, which do not include the account currency in its quotes, cross-currency pairs. Thus, if the account currency is in Euro, then the pair GBPUSD will be a cross-currency pair, since the deposit currency is in British pounds, and currency quotes are in U.S. dollars. In this case, to calculate the margin, we will have to express the pound (GBP) in Euro (EUR).

But we will continue to consider an example in which the symbol is the currency pair CADCHF. The deposit currency is in Canadian Dollars (CAD) and does not coincide with the U.S. dollar (USD). The quote currency is in Swiss francs and also does not coincide with the American dollar.


We only can say that the deposit for opening a position in 1 lot equals to 100,000 Canadian dollars. Our task is to recalculate the deposit into the account currency, in U.S. dollars. To do this we need to find the currency pair, the exchange rate of which contains the U.S. dollar and the deposit currency - CAD. There are a total of two potential options:


- CADUSD

- USDCAD


We have the output data for the CADCHF:


```
margin_currency=CAD (Canadian dollar)
profit_currency=CHF (Swiss frank)
```

We do not know in advance which of the currency pairs exists in the terminal, and in terms of the MQL5 language, neither option is preferable. Therefore, we write the GetSymbolByCurrencies() function, which for the given set of currencies will give us the **first matching** currency pair for calculations.

```
//+------------------------------------------------------------------+
//| Return symbol with specified margin currency and profit currency |
//+------------------------------------------------------------------+
string GetSymbolByCurrencies(string margin_currency,string profit_currency)
  {
//--- In loop process all symbols, that are shown in Market Watch window
   for(int s=0;s<SymbolsTotal(true);s++)
     {
      //--- Get symbol name by number in Market Watch window
      string symbolname=SymbolName(s,true);

      //--- Get margin currency
      string m_cur=SymbolInfoString(symbolname,SYMBOL_CURRENCY_MARGIN);

      //--- Get profit currency (profit on price change)
      string p_cur=SymbolInfoString(symbolname,SYMBOL_CURRENCY_PROFIT);

      //--- if If symbol matches both currencies, return symbol name
      if(m_cur==margin_currency && p_cur==profit_currency) return(symbolname);
     }
   return(NULL);
  }
```

As can be seen from the code, we begin the enumeration of all symbols, available in the "Market View" window ( [SymbolsTotal()](https://www.mql5.com/en/docs/marketinformation/symbolstotal) function with "true" parameter will give us this amount). In order to get the name of each symbol **by the number in the list of the** "Market View", we use the [SymbolName()](https://www.mql5.com/en/docs/marketinformation/symbolname) function with **true** parameter! If we set the parameter to "false", then we will enumerate all of the symbols presented on the trading server, and this is usually much more than what is selected in the terminal.



Next, we use the name of the symbol to obtain the currency deposit and the quotes, and to compare them with the ones that were passed to the GetSymbolByCurrencies() function. In case of success, we returns the name of the symbol, and the work of the function is completed successfully and ahead of schedule. If the loop is completed, and we reach the last line of the function, then nothing fit and the symbol was not found, - return [NULL](https://www.mql5.com/en/docs/basis/types/void).

Now that we can obtain the calculation currency for the cross-currency pair, by using the GetSymbolByCurrencies() function, we will make two attempts: in the first attempt we'll search for the symbol, the deposit currency of which is the margin\_currency (deposit currency CADCHF - CAD), and the quote currency is the currency of the account (USD). In other words, we are looking for something similar to the pair of CADUSD.

```
//--- If calculation currency is still not determined
//--- then we have cross currency
   if(calc_currency="")
     {
      calc_currency=GetSymbolByCurrencies(margin_currency,account_currency);
      mode=true;
      //--- If obtained value is equal to NULL, then this symbol is not found
      if(calc_currency==NULL)
        {
         //--- Lets try to do it reverse
         calc_currency=GetSymbolByCurrencies(account_currency,margin_currency);
         mode=false;
        }
     }
```

If the attempt fails, try to find another option: look for a symbol, the deposit currency of which is account\_currency (USD), and the quote currency is margin\_currency (deposit currency for CADCHF - CAD). We are looking for something similar to the USDCAD.


Now that we found the calculations currency pair, it can be one of two options - direct or reverse. The mode variable assumes the value "true" for the inverse currency pair. If we have a direct currency pair, then the value is equal to "false". For the "true" value, we multiply it by the exchange rate of the currency pair, for the false value- we divide it by the deposit value of a standard contract in the account currency.

Here is the final calculation of the deposit size in the account currency for the found calculation currency. It is fit for both options - the direct and the reverse currency pairs.

```
//--- We know calculation currency, let's get its last prices
   MqlTick tick;
   SymbolInfoTick(calc_currency,tick);

//--- Now we have everything for calculation
   double calc_price;
//--- Calculate for Buy
   if(direction==POSITION_TYPE_BUY)
     {
      //--- Reverse quote
      if(mode)
        {
         //--- Calculate using Buy price for reverse quote
         calc_price=tick.ask;
         answer=lot*lot_size*calc_price;
        }
      //--- Direct quote
      else
        {
         //--- Calculate using Sell price for direct quote
         calc_price=tick.bid;
         answer=lot*lot_size/calc_price;
        }
     }

//--- Calculate for Sell
   if(direction==POSITION_TYPE_SELL)
     {
      //--- Reverse quote
      if(mode)
        {
         //--- Calculate using Sell price for reverse quote
         calc_price=tick.bid;
         answer=lot*lot_size*calc_price;
        }
      //--- Direct quote
      else
        {
         //--- Calculate using Buy price for direct quote
         calc_price=tick.ask;
         answer=lot*lot_size/calc_price;
        }
     }
```

Return the obtained result


```
 //--- Return result - amount of equity in account currency, required to open position in specified volume
return  (Answer);
```

The GetMarginForOpening() function completes its work at this point. The last thing that needs to be done is to divide the obtained value by the size of the provided leverage - and then we will obtain the value of the margin for open positions with the specified volume in the assumed direction. Keep in mind, that for the symbols, representing the reverse or the cross-currency pair, the value of the margin will vary with each tick.

Here is a part of the _SymbolInfo\_Advanced.mq5_ Expert Advisor code. The complete code is attached as a file.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- String variable for comment
   string com="\r\n";
   StringAdd(com,Symbol());
   StringAdd(com,"\r\n");

//--- Size of standard contract
   double lot_size=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_CONTRACT_SIZE);

//--- Margin currency
   string margin_currency=SymbolInfoString(_Symbol,SYMBOL_CURRENCY_MARGIN);
   StringAdd(com,StringFormat("Standard contract: %.2f %s",lot_size,margin_currency));
   StringAdd(com,"\r\n");

//--- Leverage
   int leverage=(int)AccountInfoInteger(ACCOUNT_LEVERAGE);
   StringAdd(com,StringFormat("Leverage: 1/%d",leverage));
   StringAdd(com,"\r\n");

//--- Calculate value of contract in account currency
   StringAdd(com,"Deposit for opening positions in 1 lot consists ");

//--- Calculate margin using leverage
   double margin=GetMarginForOpening(1,Symbol(),POSITION_TYPE_BUY)/leverage;
   StringAdd(com,DoubleToString(margin,2));
   StringAdd(com," "+AccountInfoString(ACCOUNT_CURRENCY));

   Comment(com);
  }
```

and the result of its work on the chart.

![](https://c.mql5.com/2/1/fig4_en__1.png)

### Conclusion

The provided examples demonstrate how easy and simple it is to obtain information about the most important characteristics of the trading account and about the properties of financial instruments.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/113](https://www.mql5.com/ru/articles/113)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/113.zip "Download all attachments in the single ZIP archive")

[account\_info\_en.mq5](https://www.mql5.com/en/articles/download/113/account_info_en.mq5 "Download account_info_en.mq5")(4.18 KB)

[symbol\_info\_en.mq5](https://www.mql5.com/en/articles/download/113/symbol_info_en.mq5 "Download symbol_info_en.mq5")(2.2 KB)

[symbolinfo\_advanced\_en.mq5](https://www.mql5.com/en/articles/download/113/symbolinfo_advanced_en.mq5 "Download symbolinfo_advanced_en.mq5")(5.86 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1336)**
(58)


![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
8 Aug 2012 at 18:16

**Urain:**

I know, I've been reported :)

I'd like to understand where things come from and where they go. So to say a detailed analysis.

There is also [ENUM\_SYMBOL\_CALC\_MODE](https://www.mql5.com/ru/docs/constants/environment_state/marketinfoconstants#enum_symbol_calc_mode)

![Mykola Demko](https://c.mql5.com/avatar/2014/7/53C7D9B0-F88C.jpg)

**[Mykola Demko](https://www.mql5.com/en/users/urain)**
\|
8 Aug 2012 at 18:31

**Rosh:**

There is also [ENUM\_SYMBOL\_CALC\_MODE](https://www.mql5.com/ru/docs/constants/environment_state/marketinfoconstants#enum_symbol_calc_mode).

Thank you Rashid, you are always on top of your game, you see the root, so to speak, you see what users do not understand.

This is what you need, in the description of [ENUM\_SYMBOL\_CALC\_MODE](https://www.mql5.com/ru/docs/constants/environment_state/marketinfoconstants#enum_symbol_calc_mode) all formulas are described.

![Olegs Kucerenko](https://c.mql5.com/avatar/2014/9/541D8919-E599.jpg)

**[Olegs Kucerenko](https://www.mql5.com/en/users/karlson)**
\|
8 Aug 2012 at 19:44

Tell me what Percentage is and where to look it up.

| |     |     |     |
| --- | --- | --- |
| SYMBOL\_CALC\_MODE\_CFD | [CFD mode](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_calc_mode "MQL5 documentation: Information about the tool") \- calculation of margin and profit for CFDs | Margin: Lots \*ContractSize\*MarketPrice\*Percentage/100<br>Profit: (close\_price-open\_price)\*Contract\_Size\*Lots | |  |
| --- | --- |
|  |  |

IBM gives the result 2 (what is above from the table) - trading without leverage.

![Olegs Kucerenko](https://c.mql5.com/avatar/2014/9/541D8919-E599.jpg)

**[Olegs Kucerenko](https://www.mql5.com/en/users/karlson)**
\|
8 Aug 2012 at 20:07

Further it is not clear and therefore interesting:

MQ #IBM.The price of the share is naturally unambiguous, the calculation is performed without leverage ( type 2 ) .

![](https://c.mql5.com/3/10/Mq.png)

Liteforex #IBM.Calculation is also without leverage (type 2).

![](https://c.mql5.com/3/10/Lite.png)

[Contract sizes](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double "MQL5 documentation: Symbol properties") are the same, calculations are the same, margins are different. It remains that this **Percentage** influences.

Further...Terminal gives out type 2 - trading without leverage. Nevertheless, here is what is written in the specification:

**\\*\\*\\*\\* The leverage for all CFDs is fixed and equal to 1:20.**

Based on this we have **19902 / 20 =995 $**. So in this formula **Percentage** is the leverage.

![apirakkamjan](https://c.mql5.com/avatar/avatar_na2.png)

**[apirakkamjan](https://www.mql5.com/en/users/apirakkamjan)**
\|
3 Jul 2019 at 06:32

**Beginners beware!!!**

This articles was translated with **many confusing word**. They sometimes use  " _deposit_" instead " _margin_" which is totally different in MQL5 language.

![Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://c.mql5.com/2/0/Expert_Advisor_classes_MQL5.png)[Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://www.mql5.com/en/articles/116)

This article focuses on the object oriented approach to doing what we did in the article "Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners" - creating a simple Expert Advisor. Most people think this is difficult, but I want to assure you that by the time you finish reading this article, you will be able to write your own Expert Advisor which is object oriented based.

![Creating an Expert Advisor, which Trades on a Number of Instruments](https://c.mql5.com/2/0/multi_assets_EA_MQL5__1.png)[Creating an Expert Advisor, which Trades on a Number of Instruments](https://www.mql5.com/en/articles/105)

The concept of diversification of assets on financial markets is quiet old, and has always attracted beginner traders. In this article, the author proposes a maximally simple approach to a construction of a multi-currency Expert Advisor, for an initial introduction to this direction of trading strategies.

![How to Order a Trading Robot in MQL5 and MQL4](https://c.mql5.com/2/0/order_EA_MQL5.png)[How to Order a Trading Robot in MQL5 and MQL4](https://www.mql5.com/en/articles/117)

"Freelance" is the largest freelance service for ordering MQL4/MQL5 trading robots and technical indicators. Hundreds of professional developers are ready to develop a custom trading application for the MetaTrader 4/5 terminal.

![Creating Information Boards Using Standard Library Classes and Google Chart API](https://c.mql5.com/2/0/info_panel_MQL5.png)[Creating Information Boards Using Standard Library Classes and Google Chart API](https://www.mql5.com/en/articles/102)

The MQL5 programming language primarily targets the creation of automated trading systems and complex instruments of technical analyses. But aside from this, it allows us to create interesting information systems for tracking market situations, and provides a return connection with the trader. The article describes the MQL5 Standard Library components, and shows examples of their use in practice for reaching these objectives. It also demonstrates an example of using Google Chart API for the creation of charts.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/113&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062517109361124260)

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