---
title: Formulating Dynamic Multi-Pair EA (Part 1): Currency Correlation and Inverse Correlation
url: https://www.mql5.com/en/articles/15378
categories: Trading, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:00:47.987149
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/15378&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068917216222576414)

MetaTrader 5 / Examples


### Introduction

When you have a solid trading strategy or system with a favorable win rate or profit factor, diversifying trades across correlated and inversely correlated currency pairs can enhance overall performance. I will demonstrate how to develop a system that identifies correlations and inverse correlations between multiple currency pairs, enabling traders to capitalize on these relationships for improved trading opportunities

During major trading events, such as Non-Farm Payroll (NFP) announcements, the market often moves rapidly in a predetermined direction. In such scenarios, execution across multiple currency pairs can be streamlined by designating one primary currency pair. The trades initiated on this primary pair would determine the corresponding trades on other pairs, leveraging the correlation and inverse correlation relationships between them. This approach can significantly enhance efficiency and consistency during high-impact market events.

**What will be covered:**

- The ability to change and modify currency pairs.
- The indexing of the currency pair that will serve as a signal provider to other currency pairs.
- Determining the base and quote currencies to trade.

In trading, correlation refers to the relationship between the price movements of different currency pairs. When two currency pairs are positively correlated, they tend to move in the same direction. For example, GBPUSD and EURUSD are often positively correlated, meaning that when GBPUSD rallies, EURUSD also tends to rally. This is because both pairs share the USD as the quote currency, and any broad weakness or strength in the USD will likely impact both pairs in the same way.

On the other hand, inverse correlation exit when two currency pairs move in opposite directions. A classic example is the relationship between GBPUSD and USDCAD. When GBPUSD moves up (Bullish), USDCAD often moves down (bearish). This happens because in the first pair (GBPUSD), the USD is the quote currency, while in the second pair (USDCAD), the USD is the base currency. As the USD weakens, GBPUSD rises, while USDCAD tends to fall.

![](https://c.mql5.com/2/90/CorPairs.png)

We will formulate dynamic multi pair EA to handle multiple currency pairs simultaneously. The system will provide flexibility by enabling you to input, change and modify currency pairs according to your trading strategy. A key feature of this system is its ability to define a primary or "main" currency pair, which acts as a signal provider for other currency pairs.

At the core of the system is the ability to dynamically adjust the list of currency pairs being traded. Traders can easily customize which pairs are included in the trading strategy, making it adaptable to various market conditions or trading plans. The EA accepts inputs for different currency pairs, allowing users to add, remove or switch between pairs as needed.

One of the most innovative aspects is the designation of a primary or main currency pair. This pair is not only actively traded, but also serves as the reference point for generating signals for other pairs. By monitoring this main pair, the EA identifies trading signals—whether buy or sell—and applies them to the selected correlated or inversely correlated pairs.

The system also supports dynamic adjustments based on the strength of correlations between currency pairs. For example, if a strong bullish signal is detected in the main currency pair, the EA can automatically open corresponding trades in pairs that historically move in the same direction. Conversely, for pairs that typically move inversely to the main pair, the EA can open opposite positions, effectively hedging against potential market fluctuations.

[![Forex Matrix](https://c.mql5.com/2/90/forexMatrix.png)](https://www.mql5.com/en/quotes/currencies/forex-matrix)

### Formulation

```
#include <Trade\Trade.mqh>
CTrade trade;
```

This imports the MetaTrader 5 trade library and creates an instance of the \`CTrade\` class, allowing you to manage trading operations such as opening, closing, and modifying orders.

```
int handles[];
```

This array is used to store the handles for various indicators or objects, which are necessary for tracking technical analysis indicators across multiple currency pairs.is array is used to store the handles for various indicators or objects, which are necessary for tracking technical analysis indicators across multiple currency pairs.

```
MqlTick previousTick, currentTick;
```

These variables store tick data for symbol prices. The \`previousTick\` holds the last tick data, while \`currentTick\` stores the current tick data.

```
inputstring Symbols = "XAUUSD, GBPUSD, USDCAD, USDJPY";
inputstring Base_Quote = "USD";
inputint Currecy_Main = 0;
```

These inputs allow the user to customize the EA:

- \`Symbols\`: A comma-separated list of currency pairs that the EA will monitor and trade.
- \`Base-Quote\`: The currency for determining correlations.
- \`Currency-Main\`: An index that specifies the main currency pair to use for generating signals.

```
string symb_List[];
string Formatted_Symbs[];
int Num_symbs = 0;
```

- \`symb-List\`: An array that holds the raw list of symbols to be processed.
- \`Formatted-Symbs\`: An array that stores the processed symbols.
- \`Num-symb\`: Holds the total number of symbols to be used after parsing the \`Symbols\` input.

```
intOnInit(){
    string sprtr = ",";
    ushort usprtr;
    usprtr = StringGetCharacter(sprtr, 0);
    StringSplit(Symbols, usprtr, symb_List);
    Num_symbs = ArraySize(symb_List);
    ArrayResize(Formatted_Symbs, Num_symbs);
```

The \`OnInit\` function is called once when the EA is loaded, setting up initial values and configurations. We then define the comma as the separator (\`sprtr\`), which will be used to split the input string. The function \`StringGetCharacter()\` converts the separator into a \`ushort\` (unsigned short) that is needed for the \`StringSplit()\` function. The \`StringSplit()\` function breaks the \`Symbols\` input ( a comma separated string) into an array of individual symbols. The \`Symb-List\[\]\` array holds the parsed symbols. The \`Formatted-Symbs\[\]\` array is resized to match the number of parsed symbols. We will use this array for further processing, such as adding any formatting or adjustments needed for trading logic.

```
for(int i = 0; i < Num_symbs; i++){
       Formatted_Symbs[i] = symb_List[i];
    }
```

We loop through the number of symbols and transfer symbols from \`symb-List\[\]\` array to \`Formatted-Symbs\[\]\` array. At this stage, no additional formatting is done.

```
ArrayResize(handles, ArraySize(Formatted_Symbs));
```

Here, we resize the \`handles\[\]\` array to match the size of \`Formatted-Symbs\[\]\` array. Each element in \`handles\[\]\` will hold the RSI handle for the corresponding symbol.

```
for(int i = 0; i < ArraySize(Formatted_Symbs); i++){
      handles[i] = iRSI(Formatted_Symbs[i], PERIOD_CURRENT, 14, PRICE_CLOSE);
    }
```

This loop initializes the RSI indicator handle for each symbol.

```

void OnTick(){

   if(isNewBar()){
      for(int i = 0; i < ArraySize(Formatted_Symbs); i++){
         Sig_trade(Formatted_Symbs[Currecy_Main], handles[Currecy_Main]);
      }
   }
}
```

Here we first check if we have a new bar, and then we have a for loop in order to detect the index of main currency that will generate signals . We then simply call the \`Sig-trade()\` function which carries the trade logic, the function takes string parameter for the symbol and integer parameter for the RSI handle.

```
void Sig_trade(string symb, int handler){
   double rsi[];
   CopyBuffer(handler, MAIN_LINE, 1, 1, rsi);

   bool RSIBuy = rsi[0] < 30;
   bool RSISell = rsi[0] > 70;

   // Check if the current symbol is a base USD pair
   bool isBaseUSD = StringSubstr(symb, 0, 3) == Base_Quote;
   bool isQuoteUSD = StringSubstr(symb, 3, 3) == Base_Quote;
   string Bcurr = SymbolInfoString(symb, SYMBOL_CURRENCY_BASE);
   string Qcurr = SymbolInfoString(symb, SYMBOL_CURRENCY_PROFIT);
```

We will use a fairly simple strategy, which is we use the RSI strategy to buy when the RSI dips below the 30 level and sell when it breaks above the 70 level.

- \`StringSubstr(symb, 0, 3) == Base-Quote\`: This Extracts the first three characters of the currency pair symbol ( symb ) and checks if they equal "USD". This determines if the symbol is a base USD pair.
- \`StringSubstr(symb, 3, 3) == Base-Quote\`: Extracts the three characters starting from the fourth position of the currency pair symbol ( symb ) and checks if they equal "USD". This determines if the symbol is a quote USD pair.
- \`Bcurr = SymbolInfoString(symb, SYMBOL-CURRENCY-BASE)\`: Retrieves the base currency of the symbol (the first currency in the pair) and stores it in the \`Bcurr\` variable.
- \`Qcurr = SymbolInfoString(symb, SYMBOL-CURRENCY-PROFIT)\`: Retrieves the quote currency of the symbol (the second currency in the pair) and stores it in \`Qcurr\` variable.

```
for(int i = PositionsTotal() - 1; i >= 0; i--){
      ulong posTicket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == symb){
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY){
            if(RSISell){
               trade.PositionClose(posTicket);
            }
            RSIBuy = false;
         }elseif(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL){
            if(RSIBuy){
               trade.PositionClose(posTicket);
            }
            RSISell = false;
         }
      }
   }
```

To manage all positions we loop through all the open positions. \`PositionsTotal()\` function returns the total number of currently open positions. The loop starts from the last position (\`PositionsTotal() - 1\`) and iterates backward. We use the backward iteration to avoid issues when modifying the list of open positions while looping. We use the function \`PositionGetTicket()\` to retrieve the ticket number of the position at index \`i\`. The ticket is a unique identifier for closing or modifying a specific position.

We then use \`PositionGetString()\` function to retrieve the symbol of the current position. From there we compare this symbol with \`symb\` (the symbol being analyzed). If they match, the position is relevant. We then check if the current position is a buy order, with \`PositionGetInteger()\` function. If the position is a buy and the RSI suggests a sell (\`RSISell\` is true), the position is closed using \`trade.PositionClose(posTicket)\`. After that we assign \`RSIBuy = false\` to ensure that no further buy trades will be opened since the current signal i indicating a sell.

The same applies to the sell positions, we check if the current position is a sell order. If the position is a sell and the RSI suggests a buy signal (\`RSIBuy\` is true), the position is closed using \`trade.PositionClose(posTicket)\`. We also assign \`RSISell = false\` to prevent new sell trades from being opened since the signal is for a buy position. We use the code above to manage all the open positions since we don't use stop loss and take profit. So the code will open and close positions depending solely on the RSI.

```
for(int i = 0; i < ArraySize(Formatted_Symbs); i++){
      string currSymb = Formatted_Symbs[i];

      // Get base and quote currencies for the looped symbol
      string currBaseCurr = SymbolInfoString(currSymb, SYMBOL_CURRENCY_BASE);
      string currQuoteCurr = SymbolInfoString(currSymb, SYMBOL_CURRENCY_PROFIT);
```

To actually open trades across all the currency pairs we loop through the size (number of elements) in the \`Formatted-Symbs\[\]\` array. The loop iterates over each element (currency pair) in the \`Formatted-Symbs\` array. We store the current symbol on the chart in the variable \`currSymb\` from \`Formatted-Symbs\` array. This symbol will be used to fetch relevant information about the currency pair.

```
if(RSIBuy){

         if(currQuoteCurr == Base_Quote){
            trade.PositionOpen(currSymb, ORDER_TYPE_BUY, volume, currentTick.ask, NULL, NULL, "Correlation");
         }

         if(currBaseCurr == Base_Quote){
            trade.PositionOpen(currSymb, ORDER_TYPE_SELL, volume, currentTick.bid, NULL, NULL, "Correlation");
         }
      }
```

Here we check and execute only if the RSI indicator generates a buy signal (typically when the RSI value is below a certain threshold, indicating an oversold condition). We then check if the quote currency of the current symbol matches the specified \`Base-Quote\` currency. If it matches, a buy order is opened for that currency pair. The logic behind this is that when you are buying a currency pair, you are buying the base currency and selling the quote currency. Therefore, if your trading strategy is bullish on pairs where the quote is your \`Base-Quote\`, you buy that pair.

We proceed to check if the base currency of the current symbol matches the specified \`Base-Qoute\` currency. If it matches, a sell order is opened for that currency pair. The reasoning here is that if your strategy is bullish on pairs where the base currency is the specified \`Base-Quote\`, you would sell that pair. Selling the pair would effectively mean selling the base currency and buying the quote currency.

```
      if(RSISell){

         if(currBaseCurr == Base_Quote){
            trade.PositionOpen(currSymb, ORDER_TYPE_SELL, volume, currentTick.bid, NULL, NULL, "Correlation");
         }
         if(currQuoteCurr == Base_Quote){
            trade.PositionOpen(currSymb, ORDER_TYPE_BUY, volume, currentTick.ask, NULL, NULL, "Correlation");
         }
      }
```

Here we handle the logic when an RSI sell signal is detected. This block is executed if the RSI indicator generates a sell signal (typically when the RSI is above a certain threshold, indicating an overbought condition). We check if the base currency of the current symbol matches the specified \`Base-Quote\` with \`currBaseCurr == Base-Quote\`. If it matches, a sell order is executed for that currency pair. The logic is still the same, if you are bearish on pairs where the base currency is the specified \`Base-Quote\`. Then you want to sell that pair. Selling that pair means you are selling the base currency and buying the quote currency.

We then proceed to check the quote currency of the current symbol if it matches the specified \`Base-Quote\` currency. If it matches a buy order is opened for that currency pair. The reasoning behind is that if your strategy is bearish on pairs where the quote currency is the specified \`Base-Quote\`, you would open a buy trade on that pair. This is because buying that pair would involve buying the base currency and selling the quote currency.

```
bool isNewBar()
  {
//--- memorize the time of opening of the last bar in the static variable
   staticdatetime last_time=0;
//--- current time
   datetime lastbar_time= (datetime) SeriesInfoInteger(Symbol(),Period(),SERIES_LASTBAR_DATE);

//--- if it is the first call of the function
   if(last_time==0)
     {
      //--- set the time and exit
      last_time=lastbar_time;
      return(false);
     }

//--- if the time differs
   if(last_time!=lastbar_time)
     {
      //--- memorize the time and return true
      last_time=lastbar_time;
      return(true);
     }
//--- if we passed to this line, then the bar is not new; return false
   return(false);
  }
```

Above is the \`isNewBar\` function so that the EA does not run or execute multiple orders.

**Results on the strategy tester**

![Strategy Tester](https://c.mql5.com/2/90/Tester3.png)

Based on the test results above, we can confirm that the system successfully opens trades according to the specified currency correlations and inverse correlations. As discussed earlier, when a buy signal is generated and the main currency is a quote USD pair, all quote USD pairs in the \`Formatted-Symbs\` array execute buy orders, while base USD pairs execute sell orders. This behavior is consistent with the expected functionality, demonstrating that the system effectively implements the correlation logic we aimed to achieve.

To effectively and correctly use the system, it is crucial to consider using same-pair based grouping. For example, when trading EUR-based pairs like EURUSD, EURGBP, and EURJPY, setting "EUR" as the \`Base-Quote\` input would allow the base currency to guide the trading logic when a signal is detected. This approach can be applied to other currency groups, such as USDEUR, or any other custom pairs that your broker allows, ensuring the system's logic is correctly aligned with your trading strategy.

The system is dynamic, allowing you to easily choose the major currency pair that will provide or generate signals by simply indexing the desired pair. This flexibility is particularly useful since the RSI indicator is used to generate signals, and the RSI buffer and handle are applied across all currency pairs. This ensures that the selected major pair effectively drives the trading decisions for the other correlated pairs in the system.

### Conclusion

In summary, we have explored and developed a dynamic multi-pair EA that processed trading signals based on RSI indicator and applies them across various correlated and inversely correlated currency pairs. The  EA allows input for customizable currency pairs and designates one main currency pair to drive trading decisions for the other currency pairs. Through analysis of base and quote currencies, the EA opens trades that align with overall market trend, utilizing correlation logic.

In conclusion, dynamic multi pair Expert Advisor can achieve more consistent trading results by systematically applying correlation and inverse correlation strategies. This approach not only optimizes trading efficiency by automating signal propagation. By integrating correlation and inverse correlation into trading system, traders can leverage the relationship between correlated currency pairs.

![Equity Curve](https://c.mql5.com/2/92/strrr.png)

![BackTest](https://c.mql5.com/2/92/strrr2.png)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15378.zip "Download all attachments in the single ZIP archive")

[Dynamic\_Multi-Pair.mq5](https://www.mql5.com/en/articles/download/15378/dynamic_multi-pair.mq5 "Download Dynamic_Multi-Pair.mq5")(6.1 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing Market Memory Zones Indicator: Where Price Is Likely To Return](https://www.mql5.com/en/articles/20973)
- [Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)
- [Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/472670)**
(1)


![Daniel Opoku](https://c.mql5.com/avatar/avatar_na2.png)

**[Daniel Opoku](https://www.mql5.com/en/users/wamek)**
\|
25 Aug 2025 at 14:50

**MetaQuotes:**

Check out the new article: [Formulating Dynamic Multi-Pair EA (Part 1): Currency Correlation and Inverse Correlation](https://www.mql5.com/en/articles/15378).

Author: [Hlomohang John Borotho](https://www.mql5.com/en/users/JohnHlomohang "JohnHlomohang")

Thank you for this article [@Hlomohang John Borotho](https://www.mql5.com/en/users/johnhlomohang)

This portion of the code. It seems looping through generate same output.

```
      for(int i = 0; i < ArraySize(Formatted_Symbs); i++){
         Sig_trade(Formatted_Symbs[Currecy_Main], handles[Currecy_Main]);
      }
```

You have set currecy\_Main as user input. currecy\_Main=0.

So we get something like this throughout the loop.

```
      for(int i = 0; i < ArraySize(Formatted_Symbs); i++){
         Sig_trade(Formatted_Symbs[0], handles[0]);
      }
```

Can we take away the loop from the code and still achieve same results?

![MQL5 Wizard Techniques you should know (Part 37): Gaussian Process Regression with Linear and Matérn Kernels](https://c.mql5.com/2/92/MQL5_Wizard_Techniques_you_should_know_Part_37___LOGO.png)[MQL5 Wizard Techniques you should know (Part 37): Gaussian Process Regression with Linear and Matérn Kernels](https://www.mql5.com/en/articles/15767)

Linear Kernels are the simplest matrix of its kind used in machine learning for linear regression and support vector machines. The Matérn kernel on the other hand is a more versatile version of the Radial Basis Function we looked at in an earlier article, and it is adept at mapping functions that are not as smooth as the RBF would assume. We build a custom signal class that utilizes both kernels in forecasting long and short conditions.

![Creating an MQL5-Telegram Integrated Expert Advisor (Part 5): Sending Commands from Telegram to MQL5 and Receiving Real-Time Responses](https://c.mql5.com/2/92/MQL5-Telegram_Integrated_Expert_Advisor_lPart_5.png)[Creating an MQL5-Telegram Integrated Expert Advisor (Part 5): Sending Commands from Telegram to MQL5 and Receiving Real-Time Responses](https://www.mql5.com/en/articles/15750)

In this article, we create several classes to facilitate real-time communication between MQL5 and Telegram. We focus on retrieving commands from Telegram, decoding and interpreting them, and sending appropriate responses back. By the end, we ensure that these interactions are effectively tested and operational within the trading environment

![Reimagining Classic Strategies in MQL5 (Part II): FTSE100 and UK Gilts](https://c.mql5.com/2/92/Reimagining_Classic_Strategies_in_MQL5_Part_II____LOGO2.png)[Reimagining Classic Strategies in MQL5 (Part II): FTSE100 and UK Gilts](https://www.mql5.com/en/articles/15771)

In this series of articles, we explore popular trading strategies and try to improve them using AI. In today's article, we revisit the classical trading strategy built on the relationship between the stock market and the bond market.

![Developing a multi-currency Expert Advisor (Part 9): Collecting optimization results for single trading strategy instances](https://c.mql5.com/2/76/Developing_a_multi-currency_advisor_gPart_9e_SQL____LOGO.png)[Developing a multi-currency Expert Advisor (Part 9): Collecting optimization results for single trading strategy instances](https://www.mql5.com/en/articles/14680)

Let's outline the main stages of the EA development. One of the first things to be done will be to optimize a single instance of the developed trading strategy. Let's try to collect all the necessary information about the tester passes during the optimization in one place.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/15378&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068917216222576414)

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