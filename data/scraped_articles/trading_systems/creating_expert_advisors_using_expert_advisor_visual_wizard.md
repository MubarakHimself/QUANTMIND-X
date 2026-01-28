---
title: Creating Expert Advisors Using Expert Advisor Visual Wizard
url: https://www.mql5.com/en/articles/347
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:51:41.632035
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/347&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062772561130989724)

MetaTrader 5 / Examples


### Introduction

[Expert Advisor Visual Wizard](https://www.mql5.com/go?link=http://www.molanis.com/products/expert-advisor-visual-wizard "http://www.molanis.com/products/expert-advisor-visual-wizard") for MetaTrader 5 provides a highly intuitive graphical environment with a comprehensive set of predefined trading blocks that let you design Expert Advisors in minutes. No coding, programming or MQL5 knowledge is required.

The click, drag and drop approach of [Expert Advisor Visual Wizard](https://www.mql5.com/go?link=http://www.molanis.com/products/expert-advisor-visual-wizard "http://www.molanis.com/products/expert-advisor-visual-wizard") allows you to create visual representations of forex trading strategies and signals as you would with pencil and paper. These trading diagrams are analyzed automatically by [Molanis](https://www.mql5.com/go?link=http://www.molanis.com/ "http://www.molanis.com/")’ MQL5 code generator that transforms them into ready to use Expert Advisors. The interactive graphical environment simplifies the design process and eliminates the need to write [MQL5](https://www.mql5.com/en/docs) code.

With Expert Advisor Visual Wizard you only need to follow a 3 step process:

![Fig. 1. Using  Expert Advisor Visual Wizard](https://c.mql5.com/2/3/Figure_1.png)

Fig. 1. Using Expert Advisor Visual Wizard

### 1\. Developing a Trading Diagram

A trading diagram is a graphical representation of an Expert Advisor. It shows the ‘flow’ through a trading decision system. Trading diagrams are made of trading blocks that are connected to create complex Expert Advisors.

To create a trading diagram, you only need to add the trading blocks, set their configuration parameters and make the necessary connections.

**Moving Average Strategy**

Typically, two moving averages can be used to create an Expert Advisor with these trading conditions:

- Buy when the short period moving average is above the long period moving average (red line is above the green line)
- Sell when the short period moving average is below the long period moving average (red line is below the green line)

![Fig. 2. Buy and Sell signals](https://c.mql5.com/2/3/Fig2_Buy_Sell_Signals.png)

Fig. 2. Buy and Sell signals

Instead of spending a long time coding this EA, with Expert Advisor Visual Wizard you can create a trading diagram that represents the moving average strategy in seconds.

Launch Expert Advisor Visual Wizard:

![Expert Advisor Visual Wizard](https://c.mql5.com/2/3/Fig3_Molanis_Expert_Advisor_Visual_Wizard.png)

Fig. 3. Expert Advisor Visual Wizard

A. Just drag and drop two Technical Analysis blocks into the trading diagram:

![Fig. 4. Adding TA boxes](https://c.mql5.com/2/3/Fig4_Adding_TA_boxes.png)

Fig. 4. Adding TA blocks

To define the moving average trading conditions, click on the TA icon and select the options as shown in the pictures:

Options to go long (or buy):

![Fig. 5. Options to go long (buy)](https://c.mql5.com/2/3/Fig5_TA_buy.png)

Fig. 5. Options to go long (buy)

Options to go short (or sell):

![Fig. 6. Options to go short (or sell)](https://c.mql5.com/2/3/Fig6_TA_sell.png)

Fig. 6. Options to go short (or sell)

B. Drag and drop one BUY block and one SELL block:

![Fig. 7. Adding Buy and Sell blocks](https://c.mql5.com/2/3/Fig7_Adding_Buy_Sell_blocks.png)

Fig. 7. Adding Buy and Sell blocks

Click on the BUY icon to define the lot size, take profit, stop loss, and trailing stops for your EA as shown in the picture:

![Fig8_Buy_options](https://c.mql5.com/2/3/Fig8_Buy_options___1.png)

Fig. 8. Options of Buy trading block

Repeat the same procedure for the SELL icon:

![Fig. 9. Options of Sell trading block](https://c.mql5.com/2/3/Fig9_Sell_options_.png)

Fig. 9. Options of Sell trading block

C. Connect all the blocks to get a trading diagram like the following:

![Fig. 10. Connected Blocks](https://c.mql5.com/2/3/Fig10_Connected_Blocks__1.png)

Fig. 10. Connected blocks

### 2\. Generating the Expert Advisor

After the trading diagram is complete you need to generate the EA clicking on Generate MQL5 Code in Trading Diagram of the main menu:

![Fig. 11. Generate MQL5 code](https://c.mql5.com/2/3/Fig11_Generate_MQL5_code.png)

Fig. 11. Generate MQL5 code

The EA Visual Wizard transforms your trading diagram into a fully working EA. It also gives you access to the EA’s MQL5 code:

![Fig. 12. MQL5 code generated](https://c.mql5.com/2/3/Fig12_Code_Generated__1.png)

Fig. 12. MQL5 code generated

### 3\. Trading with MetaTrader 5

After you generate the EA it will be available in MetaTrader 5 for you to trade. Just attach it to a chart to start trading.

![Fig. 13 Expert Advisor input parameters](https://c.mql5.com/2/3/Fig13_EA_input_parameters.png)

Fig. 13 Expert Advisor input parameters

All expert advisors generated with Molanis’ software have MetaTrader variables to manage:

- Trading bars or ticks;
- Alert mode (Does not trade but gives signals);
- 4 or 5 decimals;
- Time filter;
- ECN orders;
- Maximum volume size;
- Maximum percentage at risk;
- Lot size management.

### 4\. Creating Expert Advisors that use Custom Indicators

**Breakthrough of the Price Channel Range Strategy**

EA logic: Positions are opened when the price pierces the borders of the Price Channel. To create this Expert Advisor we need to use the custom indicator Price Channel by [Sergey Gritsay](https://www.mql5.com/en/users/sergey1294).

You can add any well-written custom indicator into the EA Visual Wizard with the Import Custom Indicator button.

Under a TA block, select Custom Indicator – [iCustom](https://www.mql5.com/en/docs/indicators/icustom), and then click on Import Custom Indicator.

![Fig. 14. Editing a trading condition](https://c.mql5.com/2/3/Fig14_TA_options.png)

Fig. 14. Editing a trading condition

Select the custom indicator you want to import.

Custom indicators must be located in the indicators directory (terminal\_data\_folder\_\\MQL5\\Indicators).

![Fig. 15. Importing custom indicator](https://c.mql5.com/2/3/Fig15_Select_Custom_indicator.png)

Fig. 15. Importing custom indicator

The import custom indicator feature reads the indicator code and based on standard rules of coding gets the number of modes (signals) and the indicator parameters. This feature cannot select the mode or shift for you. It's your job to know the right signal and parameters for your EA.

![Fig. 16. Indicator was imported](https://c.mql5.com/2/3/Fig16_molanis-import-result.PNG)

Fig. 16. Indicator was imported

After you finish importing the custom indicator, you can use it to define trading conditions in your Expert Advisor.

For sell:

![Fig. 17. Trading conditions for sell](https://c.mql5.com/2/3/Fig17_TA_Sell_Rules__1.png)

Fig. 17. Trading conditions for sell

For buy:

![Fig. 18. Trading conditions for buy](https://c.mql5.com/2/3/Fig18_TA_Buy_Rules__1.png)

Fig. 18. Trading conditions for buy

Now, just create a setup like the one and you are done.

![Fig. 19. The Expert Advisor diagram](https://c.mql5.com/2/3/Fig19_EA_Diagram__1.png)

Fig. 19. The Expert Advisor diagram

### Conclusion

The EA Visual Wizard is a great tool to create Expert Advisors in minutes. We have developed 15 examples based on the 20 Trade Signals in article ["20 Trade Signals in MQL5"](https://www.mql5.com/en/articles/130). I encourage readers to review them at our [examples page](https://www.mql5.com/go?link=http://www.molanis.com/products/expert-advisor-visual-wizard/expert-advisors-mt5 "http://www.molanis.com/products/expert-advisor-visual-wizard/expert-advisors-mt5"). [http://www.molanis.com/products/expert-advisor-visual-wizard/expert-advisors-mt5](https://www.mql5.com/go?link=http://www.molanis.com/products/expert-advisor-visual-wizard/expert-advisors-mt5 "http://www.molanis.com/products/expert-advisor-visual-wizard/expert-advisors-mt5")

I am attaching the code for the Example 1 - simple moving average strategy explained in part 1 of this article. Example7, Breakthrough of the Price Channel Range was used to explain the import process for part 4 of this article.

[Read about the 15 examples](https://www.mql5.com/go?link=http://www.molanis.com/products/expert-advisor-visual-wizard/expert-advisors-mt5 "http://www.molanis.com/products/expert-advisor-visual-wizard/expert-advisors-mt5")

01. Simple moving average;
02. Multi-Currency Simple moving average;
03. Multi-timeframe Simple moving average;
04. Multi-timeframe Advanced Simple moving average;
05. Moving Average Crossover;
06. Intersection of the Main and Signal Line of MACD;
07. Breakthrough of the Price Channel Range;
08. RSI indicator Overbuying/Overselling strategy;
09. Exit from the Overbuying/Overselling Zones of CCI;
10. Exit from the Overbuying/Overselling Zones of Williams Percentage Range;
11. Bounce from the Borders of the Bollinger Channel;
12. ADX Adaptive Channel Breakthrough (Uses a custom Indicator);
13. Bounce from the Borders of the Standard Deviation Channel (Uses a custom Indicator);
14. NRTR Change of Trend (Uses a custom Indicator);
15. Detect Change of Trend using the Adaptative Moving Average (AMA) indicator.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/347.zip "Download all attachments in the single ZIP archive")

[simple-ma-ea-molanis.mq5](https://www.mql5.com/en/articles/download/347/simple-ma-ea-molanis.mq5 "Download simple-ma-ea-molanis.mq5")(68.98 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/5908)**
(36)


![Dina Paches](https://c.mql5.com/avatar/2017/7/596F146E-95A1.png)

**[Dina Paches](https://www.mql5.com/en/users/dipach)**
\|
29 Jan 2014 at 17:20

Since earlier I wrote that my antivirus blocked the loading of the site, of course, I can not help but write that this blockade stopped.

It seems that after the next update of the antivirus.

I will look for time to get acquainted with the programme in detail.

To create something like this is not an ordinary mind is required.

![Yury Reshetov](https://c.mql5.com/avatar/2013/6/51B9C78D-95BF.png)

**[Yury Reshetov](https://www.mql5.com/en/users/reshetov)**
\|
29 Jan 2014 at 17:25

**komposter:**

Switch off your antivirus?

I switched it off once to access my own website. It took me almost 24 hours to get the virus off my computer. Even worse, my ISP blocked all my email ports because the virus was sending spam from me.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
30 Jan 2014 at 11:23

+


![Thiago Ferreira](https://c.mql5.com/avatar/2013/12/52A24514-B45C.JPG)

**[Thiago Ferreira](https://www.mql5.com/en/users/tcferreira)**
\|
4 Mar 2014 at 02:26

Good for making simple but limited EAs. If you want an EA with more options, you'll have to opt for another EA Buider.


![Aleksandr Brown](https://c.mql5.com/avatar/2013/5/51829572-CC9A.JPG)

**[Aleksandr Brown](https://www.mql5.com/en/users/brown-aleks)**
\|
20 Jul 2014 at 23:38

The idea of creating a visual constructor is definitely doomed to success! But I didn't see any usefulness in Expert Advisor Visual Wizard. It is a very simple programme. It is not even clear who it is designed for. The [MQL5 Wizard](https://www.mql5.com/en/articles/171 "Article: Creating an Expert Advisor without Programming Using MQL5 Wizard") can cope with such an elementary design.

It would be good if the visual constructor had the same variety and flexibility as MQL5. It seems to me that it would be easy to implement such an idea if the meaning of the "click, drag and drop" icon is not the blocks of technical analysis, but elementary operators. That is. 1 operator = 1 pictogram. And the entry and exit points of the pictogram are parameters of the operator. Throw a dozen or more pictograms (operators, pre-prepared functions and classes, etc.), unite them by lines of source to input parameters, and it's done. You can compile and test it.

![Analysis of the Main Characteristics of Time Series](https://c.mql5.com/2/0/Time_Series_Analysis_in_MQL5.png)[Analysis of the Main Characteristics of Time Series](https://www.mql5.com/en/articles/292)

This article introduces a class designed to give a quick preliminary estimate of characteristics of various time series. As this takes place, statistical parameters and autocorrelation function are estimated, a spectral estimation of time series is carried out and a histogram is built.

![Universal Regression Model for Market Price Prediction](https://c.mql5.com/2/0/universal_regression.png)[Universal Regression Model for Market Price Prediction](https://www.mql5.com/en/articles/250)

The market price is formed out of a stable balance between demand and supply which, in turn, depend on a variety of economic, political and psychological factors. Differences in nature as well as causes of influence of these factors make it difficult to directly consider all the components. This article sets forth an attempt to predict the market price on the basis of an elaborated regression model.

![Interview with Alexander Arashkevich (ATC 2011)](https://c.mql5.com/2/0/AAA777_avatar.png)[Interview with Alexander Arashkevich (ATC 2011)](https://www.mql5.com/en/articles/556)

The Championship fervour has finally subsided and we can take a breath and start rethinking its results again. And we have another winner Alexander Arashkevich (AAA777) from Belarus, who has won a special prize from the major sponsor of Automated Trading Championship 2011 - a 3 day trip to one of the Formula One races of the 2012 season. We could not miss the opportunity to talk with him.

![The Role of Statistical Distributions in Trader's Work](https://c.mql5.com/2/0/statistic_measument.png)[The Role of Statistical Distributions in Trader's Work](https://www.mql5.com/en/articles/257)

This article is a logical continuation of my article Statistical Probability Distributions in MQL5 which set forth the classes for working with some theoretical statistical distributions. Now that we have a theoretical base, I suggest that we should directly proceed to real data sets and try to make some informational use of this base.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=wanzypcrfoghgmarfjlcqkvgmcebikds&ssn=1769158300609871190&ssn_dr=0&ssn_sr=0&fv_date=1769158300&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F347&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20Expert%20Advisors%20Using%20Expert%20Advisor%20Visual%20Wizard%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915830070991202&fz_uniq=5062772561130989724&sv=2552)

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