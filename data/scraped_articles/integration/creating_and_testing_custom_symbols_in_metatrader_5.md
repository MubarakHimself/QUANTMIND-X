---
title: Creating and testing custom symbols in MetaTrader 5
url: https://www.mql5.com/en/articles/3540
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:25:51.705592
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/3540&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068220100080760558)

MetaTrader 5 / Examples


Custom symbols in MetaTrader 5 offer new opportunities for developing trading systems and analyzing any financial markets. Now traders are able to plot charts and test trading strategies on an unlimited number of financial instruments. To do this, they only need to create their custom symbol based on a tick or minute history. Custom symbols can be used to test any trading robot [from the Market](https://www.mql5.com/en/market/mt5/expert) or the free source code library.

### Creating a custom symbol

Let's create a custom symbol based on the one already present in the [Market Watch](https://www.metatrader5.com/en/terminal/help/trading/market_watch "https://www.metatrader5.com/en/terminal/help/trading/market_watch"). Open the Symbols window by the right mouse button and select the one you would like to use to create a custom symbol.

![](https://c.mql5.com/2/29/terminal_symbols__1.png)

After clicking "Create Custom Symbol", set its name and change the required parameters in the [contract specification](https://www.metatrader5.com/en/terminal/help/trading/market_watch#specification "https://www.metatrader5.com/en/terminal/help/trading/market_watch#specification") if necessary.

![](https://c.mql5.com/2/29/custom_symbol_create__1.png)

All custom symbols are placed to the separate <Custom> directory of the Symbols tree and are always located there regardless of a broker you are currently connected to. Price data of custom symbols are saved in a separate Custom directory outside of the directories where data of trade servers are stored:

C:\\Users\\\[windows account\]\\AppData\\Roaming\\MetaQuotes\\Terminal\\\[instance id\]\\bases\\Custom

This is another advantage of creating a custom symbol — you can simply copy the necessary symbols from each broker to your [custom group](https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments#manage "https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments#manage"). Just like conventional symbols, you can delete a custom symbol only if there are no open charts with it and it is not present in the Market Watch.

### Configuring a custom symbol

You are able to set the quotes accuracy, contract size, symbol currency, settlement method and all other parameters affecting [test results](https://www.metatrader5.com/en/terminal/help/algotrading/testing_report "https://www.metatrader5.com/en/terminal/help/algotrading/testing_report") of a trading strategy involving the custom symbol.

![](https://c.mql5.com/2/29/specification__1.gif)

### Importing history

After creating a custom symbol, we need to add a quote history for it. First, let's see how to create a history based on an already existing symbol. On the Symbols window, open the Bars or Ticks tab depending on how you want to prepare the history. Make a request for a desired period and perform [export](https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments#export "https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments#export").  To receive bars, select the M1 timeframe, since the entire history in MetaTrader 5 is based on minute data.

![](https://c.mql5.com/2/29/export_bars__1.gif)

The export is done in the form of a text CSV file with the name having the following look: EURUSD\_M1\_201701020000\_201707251825.csv, which contains the symbol name, timeframe and the time boundaries of the exported history up to a minute. Below is how the format looks when exporting bars:

```
<DATE>	        <TIME>	        <OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL><VOL>	     <SPREAD>
2017.01.02	00:03:00	1.05141	1.05141	1.05141	1.05141	6	15000000	118
2017.01.02	00:04:00	1.05141	1.05141	1.05141	1.05141	2	5000000	        112
2017.01.02	00:05:00	1.05158	1.05158	1.05148	1.05158	10	17000000	101
2017.01.02	00:06:00	1.05148	1.05158	1.05148	1.05158	7	13000000	101
```

When exporting a tick history, the CSV file becomes much larger, and its format receives data on each tick up to milliseconds. Using these data, the terminal forms a one-minute history all other timeframes are to be based on.

```
<DATE>          <TIME>          <BID>   <ASK>   <LAST>  <VOLUME>
2017.07.03      00:03:47.212    1.14175 1.14210 0.00000 0
2017.07.03      00:03:47.212    1.14168 1.14206 0.00000 0
2017.07.03      00:03:47.717    1.14175 1.14206 0.00000 0
2017.07.03      00:03:54.241    1.14175 1.14205 0.00000 0
2017.07.03      00:03:57.982    1.14165 1.14201 0.00000 0
2017.07.03      00:04:07.795    1.14175 1.14201 0.00000 0
2017.07.03      00:04:55.432    1.14164 1.14200 0.00000 0
2017.07.03      00:14:33.743    1.14173 1.14203 0.00000 0
2017.07.03      00:14:33.743    1.14173 1.14201 0.00000 0
2017.07.03      00:16:44.901    1.14174 1.14195 0.00000 0
```

Therefore, if you form a history for your custom symbol using any third-party sources, you need to prepare the data in accordance with the formats displayed above.

To [import the history](https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments#import_history "https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments#import_history"), perform the similar steps. Find your EURUSD\_my custom symbol in the **Custom\\<Custom group>** folder, go to the Ticks tab, select the necessary CSV file and click "Import Ticks" (do the same to import bars).

![](https://c.mql5.com/2/29/import_ticks__1.gif)

After importing the history, you can edit it by adding, deleting or changing any bars and ticks.

Created custom symbols become available in the Market Watch, and you can open charts for them. Thus, custom symbols allow you to apply the entire [rich arsenal](https://www.metatrader5.com/en/trading-platform/technical-analysis "https://www.metatrader5.com/en/trading-platform/technical-analysis") of the MetaTrader 5 technical analysis, including the launch of any [custom indicators](https://www.mql5.com/en/code/mt5/indicators) and [analytical tools](https://www.mql5.com/en/market/mt5/indicator) from the Market.

### Testing trading strategies on a custom symbol

The multi-threaded MetaTrader 5 strategy tester allows you to test strategies trading on multiple financial instruments [on real ticks](https://www.mql5.com/en/articles/2612). Use all its advantages to test strategies on custom symbols. To do this, import a high-quality minute (preferably tick) history and set the properties for each instrument necessary for a detailed reconstruction of the trading environment. After that, select the necessary EA and set the test settings. The entire process is similar to working with conventional trading symbols provided by your broker.

![](https://c.mql5.com/2/29/tester__1.png)

Provide the tester with all the necessary symbols you may need to calculate margin requirements and profit in your trading account currency. When calculating the margin and profit, the strategy tester automatically uses available cross rates. Suppose that we have created AUDCAD.custom symbol with the **[Forex](https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex "https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex")** type of margin calculation, and our account currency is USD. In this case, the tester searches for the necessary symbols in the following order **based on the Forex symbol name**:

1. first, the search is performed for the symbols of AUDUSD.custom (for calculating the margin) and USDCAD.custom (for calculating the trade profit) forms
2. if any of these symbols is not present, the search is performed for the first symbol corresponding to the necessary currency pairs by name (AUDUSD and USDCAD respectively). For example, AUDUSD.b and USDCAD.b symbols have been found. This means their rates are to be used to calculate the margin and profit.

Instruments with other types of margin calculation (CFD, Futures and **[Stock Exchange](https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_exchange "https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_exchange")**) require a currency pair to convert the instrument currency into a deposit one. Suppose that we have created a custom symbol with profit and margin currency expressed in GBP, while the deposit currency is CHF. In this case, the search for testing symbols is performed in the following order:

1. The presence of a trading symbol corresponding to GBPCHF (GBP vs CHF) is checked.
2. If no such symbol exists, the search is performed for the first trading symbol that corresponds to GBPCHF by its name, for example GBPCHF.b or GBPCHF.def.

When testing using custom symbols, make sure that the trading account has all the necessary currency pairs. Otherwise, the calculation of financial results and margin requirements during testing will not be possible.

### Optimizing strategies on a custom symbol in a local network

Apart from your own agents, you are able to use agents from a local network and [remote agents](https://www.metatrader5.com/en/terminal/help/algotrading/metatester "https://www.metatrader5.com/en/terminal/help/algotrading/metatester") to optimize trading strategies on custom symbols. This is yet another advantage of the MetaTrader 5 strategy tester allowing you to shorten the time spent searching for optimal parameters of your trading system.

The use of [MQL5 Cloud Network](https://www.metatrader5.com/en/terminal/help/mql5cloud "https://www.metatrader5.com/en/terminal/help/mql5cloud") for optimization using custom symbols is not allowed. This is due to the fact that custom symbols with the same names, but different price histories may exist on computers of different traders. In addition to the discrepancy of test results between network agents, this may cause mass reloading and synchronization of history data, which leads to excessive internet usage.

### Functions for working with custom symbols

You can also work with custom symbols using MQL5 language. The functions from the ["Custom symbols"](https://www.mql5.com/en/docs/customsymbols "Custom symbols") section are designed for that. An MQL5 application allows you to quickly create necessary financial instruments with specified [properties](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) based on data from third-party sources. Thus, you can automate collecting and preparing history data for any symbols, as well as create your custom indices and other derivatives and test them in the [MetaTrader 5 strategy tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester").

|     |     |
| --- | --- |
| Function | Action |
| [CustomSymbolCreate](https://www.mql5.com/en/docs/customsymbols/customsymbolcreate) | Create a custom symbol with the specified name in the specified group |
| [CustomSymbolDelete](https://www.mql5.com/en/docs/customsymbols/customsymboldelete) | Delete a custom symbol with the specified name |
| [CustomSymbolSetInteger](https://www.mql5.com/en/docs/customsymbols/customsymbolsetinteger) | Set the integer type property value for a custom symbol |
| [CustomSymbolSetDouble](https://www.mql5.com/en/docs/customsymbols/customsymbolsetdouble) | Set the real type property value for a custom symbol |
| [CustomSymbolSetString](https://www.mql5.com/en/docs/customsymbols/customsymbolsetstring) | Set the string type property value for a custom symbol |
| [CustomSymbolSetMarginRate](https://www.mql5.com/en/docs/customsymbols/customsymbolsetmarginrate) | Set the margin rates depending on the order type and direction for a custom symbol |
| [CustomSymbolSetSessionQuote](https://www.mql5.com/en/docs/customsymbols/customsymbolsetsessionquote) | Set the start and end time of the specified quotation session for the specified symbol and week day |
| [CustomSymbolSetSessionTrade](https://www.mql5.com/en/docs/customsymbols/customsymbolsetsessiontrade) | Set the start and end time of the specified trading session for the specified symbol and week day |
| [CustomRatesDelete](https://www.mql5.com/en/docs/customsymbols/customratesdelete) | Delete all bars from the price history of the custom symbol in the specified time interval |
| [CustomRatesReplace](https://www.mql5.com/en/docs/customsymbols/customratesreplace) | Fully replace the price history of the custom symbol within the specified time interval with the data from the MqlRates type array |
| [CustomRatesUpdate](https://www.mql5.com/en/docs/customsymbols/customratesupdate) | Add missing bars to the custom symbol history and replace existing data with the ones from the MqlRates type array |
| [CustomTicksDelete](https://www.mql5.com/en/docs/customsymbols/customticksdelete) | Delete all ticks from the price history of the custom symbol in the specified time interval |
| [CustomTicksReplace](https://www.mql5.com/en/docs/customsymbols/customticksreplace) | Fully replace the price history of the custom symbol within the specified time interval with the data from the MqlTick type array |

These functions complement the capabilities of the ["Getting Market Information"](https://www.mql5.com/en/docs/marketinformation) section. Now, you can not only obtain properties of any symbols but also set them for custom instruments.

### Test your trading ideas on any symbols in MetaTrader 5!

The MetaTrader 5 platform offers the widest [opportunities](https://www.mql5.com/en/articles/384) for algorithmic traders. Here are just a few of them:

- [84 built-in](https://www.mql5.com/en/articles/384#1_3) technical analysis tools,
- multi-threaded strategy tester with [20 000 agents](https://cloud.mql5.com/ "https://cloud.mql5.com/") from MQL5 Cloud Network,
- [asynchronous trading operations](https://www.mql5.com/en/articles/2635) within one **(!)** millisecond for execution,
- 4000 [free examples](https://www.mql5.com/en/code) in source codes,

- MetaEditor with [debugging and profiling](https://www.mql5.com/en/articles/2661).

Traders have access to all the advantages of the MetaTrader 5 platform even on the symbols their brokers do not have yet. Create your own symbols and test trading strategies in any financial market!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/3540](https://www.mql5.com/ru/articles/3540)

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
**[Go to discussion](https://www.mql5.com/en/forum/216050)**
(66)


![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
3 Aug 2018 at 21:02

**fxsaber:**

1881 build seems to be without this bug.

Just on 1881 are deleted

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
4 Aug 2018 at 10:17

**Ivan Titov:**

Just on 1881 are deleted

I have not tested this solution:

Properties

Security tab
Advanced
Edit authorisations
Select the proper user or user group
Edit the authorisations as needed. I removed the permission to Delete the folder or the files inside it

Problem solved...

![Quantum Capital International Group Ltd](https://c.mql5.com/avatar/2020/4/5E96871E-4BA3.png)

**[Yang Chih Chou](https://www.mql5.com/en/users/fxchess)**
\|
6 Aug 2023 at 03:16

Can it possible to test multi-currency in MT5 with [custom symbols](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5 ").


![ogflagg](https://c.mql5.com/avatar/avatar_na2.png)

**[ogflagg](https://www.mql5.com/en/users/ogflagg)**
\|
29 Apr 2024 at 10:13

I created a Renko bar [custom symbol](https://www.mql5.com/en/articles/3540 "Article: Creating and Testing Custom Symbols in MetaTrader 5 ") but my ea isn’t trading on it


![Wadim Skwortsov](https://c.mql5.com/avatar/2024/7/66A11CF5-90BA.png)

**[Wadim Skwortsov](https://www.mql5.com/en/users/wadimskwortsov)**
\|
24 Jul 2024 at 15:28

For a similar tool, there are solutions to add via ["Create Custom Symbol](https://www.mql5.com/en/docs/customsymbols/customsymbolcreate "MQL5 documentation: CustomSymbolCreate function")":

https://www.google.com/finance/quote/TSLA:NASDAQ

и

https://www.google.com/finance/quote/BTC-USD

Thank you!

Vadim S

![Graphical Interfaces XI: Text edit boxes and Combo boxes in table cells (build 15)](https://c.mql5.com/2/28/MQL5-avatar-XI-build_15.png)[Graphical Interfaces XI: Text edit boxes and Combo boxes in table cells (build 15)](https://www.mql5.com/en/articles/3394)

In this update of the library, the Table control (the CTable class) will be supplemented with new options. The lineup of controls in the table cells is expanded, this time adding text edit boxes and combo boxes. As an addition, this update also introduces the ability to resize the window of an MQL application during its runtime.

![Deep Neural Networks (Part II). Working out and selecting predictors](https://c.mql5.com/2/48/Deep_Neural_Networks_02.png)[Deep Neural Networks (Part II). Working out and selecting predictors](https://www.mql5.com/en/articles/3507)

The second article of the series about deep neural networks will consider the transformation and choice of predictors during the process of preparing data for training a model.

![Using cloud storage services for data exchange between terminals](https://c.mql5.com/2/28/7l8-fbt8.png)[Using cloud storage services for data exchange between terminals](https://www.mql5.com/en/articles/3331)

Cloud technologies are becoming more popular. Nowadays, we can choose between paid and free storage services. Is it possible to use them in trading? This article proposes a technology for exchanging data between terminals using cloud storage services.

![The Flag Pattern](https://c.mql5.com/2/28/MQL5-avatar-flag-001__1.png)[The Flag Pattern](https://www.mql5.com/en/articles/3229)

The article provides the analysis of the following patterns: Flag, Pennant, Wedge, Rectangle, Contracting Triangle, Expanding Triangle. In addition to analyzing their similarities and differences, we will create indicators for detecting these patterns on the chart, as well as a tester indicator for the fast evaluation of their effectiveness.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=rwxwxlskblkdpcmguwxkoynvskaeszan&ssn=1769178350506942120&ssn_dr=0&ssn_sr=0&fv_date=1769178350&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F3540&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20and%20testing%20custom%20symbols%20in%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917835069241681&fz_uniq=5068220100080760558&sv=2552)

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