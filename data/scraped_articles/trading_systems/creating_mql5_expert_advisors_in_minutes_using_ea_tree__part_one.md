---
title: Creating MQL5 Expert Advisors in minutes using EA Tree: Part One
url: https://www.mql5.com/en/articles/337
categories: Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:39:17.994006
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/337&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049281690897066132)

MetaTrader 5 / Trading systems


### Introduction

[EA Tree](https://www.mql5.com/go?link=http://www.eatree.com/ "http://www.eatree.com/") is the first drag and drop MetaTrader MQL5 Expert Advisor Builder. It is a windows application that runs locally on your computer. You can create complex MQL5 code using a very easy to use graphical user interface.

In EA Tree, Expert Advisors are created by connecting boxes together. Boxes may contain MQL5 functions, technical indicators, custom indicators, or values. Outputs of one box may be connected to inputs of another box to form a "tree of boxes". EA Tree generates MQL5 code from the "tree of boxes" and then uses the [MetaTrader 5](https://www.metatrader5.com/ "MetaTrader 5 trading platform") platform installed on your computer to convert the MQL5 code into an executable Expert Advisor.

![Figure 1. EA Tree Graphical User Interface](https://c.mql5.com/2/3/Image1_MACDsampleW-MM.png)

Figure 1. EA Tree Graphical User Interface

In the next few section, we will go over the basic concepts of EA Tree.

### 1\. Basic Concepts

The concept of the "box" and "buttons".

Boxes are a representation of information in EA Tree. They may contain technical indicator, mathematical function, etc. Each box has input buttons and output buttons. In this example, the Add box has three input buttons labeled A, B, and С and one output button labeled OUT.

Each box has a unique label using its name followed by an index. In this example the label _Add\_1_ indicates that the box name is _Add_ and its index is 1, which means that it was the first box that was created in a layout of boxes.

![Figure 2. Add box](https://c.mql5.com/2/3/Image2_add-box-1.png)

Figure 2. Add box

**Connecting buttons:**

Output buttons of one box can be connected to input buttons of another box if they have the same data types, e.g. they are both [double](https://www.mql5.com/en/docs/basis/types/double "Double data type") data type.

In this example, we connect the OUT button of the Add\_2 box to the A button of the If\_Else\_1 box.

![Figure 3. "Add" box + "if else" box](https://c.mql5.com/2/3/Image3_ifelse-add.jpg)

Figure 3. "Add" box + "If-Else" box

### **2\. The "Trade" box**

The Trade box is the main box in EA Tree. It has many input buttons. The most important buttons are the openLong, openShort, closeLong, and closeShort buttons. There are also many trading parameters.

![Figure 4. "Trade" box](https://c.mql5.com/2/3/Image4_tradeBox.jpg)

Figure 4. "Trade" box

### **3\. The "MM" (MoneyManagement) box**

The MM box handles MoneyManagement in EA Tree. It has several input buttons. You need to connect its output button to the MM button of the Trade box.

![Figure 5. "Trade" box + MM box](https://c.mql5.com/2/3/Image5_TradePlusMM.jpg)

Figure 5. "Trade" box + MM box

### **4\. Logic Boxes**

Logic boxes are important to connect between trade box condition buttons: openLong, openShort, closeLong, and closeShort and the rest of the boxes in the layout.

**If-Else logic box**

The If-Else box has this logic:

If A operator B then T(output button) is true else F(output button) is true.

Where operator could be equal to, not equal to, less than, greater than, etc.

![Figure 6. "If-Else" logic box](https://c.mql5.com/2/3/Image6_ifElseBox.jpg)

Figure 6. "If-Else" logic box

**Crossover logic box**

The Crossoverbox has this logic:

If A operator1 B AND C operator2 D then T(output button) is true else F(output button) is true.

Where operator1 and operator2 could be equal to, not equal to, less than, greater than, etc. The shifts sets indexes of the indicators to connect to.

In this example we have this logic:

If current [Stochastic](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so") main > current [Stochastic](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so")signal and

If previous [Stochastic](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so")main > previous [Stochastic](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so")signal

Then T is true and F is false

![Figure 7. Crossover + Stochastic Logic boxes](https://c.mql5.com/2/3/Image7_CrossOverPlusStoch.jpg)

Figure 7. "Crossover" + "Stochastic" logic boxes

**AND logic box**

The AND box has this logic:

If (A AND B AND C AND D) then OUT(output button) is true

![](https://c.mql5.com/2/3/and-box.png)

Figure 8. "And" box

**OR logic box**

The OR box has this logic:

If (A OR B OR C OR D) then OUT (output button) is true

![Figure 9. "Or" box](https://c.mql5.com/2/3/Image9_or-box.png)

Figure 9. "Or" box

### **5\. Technical Indicators Boxes**

There are many [technical indicators](https://www.metatrader5.com/en/terminal/help/charts_analysis/indicators "Technical indicators") listed under the Technical Indicators menu such as the [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "MACD technical indicator") and [Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "Moving Averge technical indicator") (MA) technical indicator boxes

**The MACD technical indicator box**

![Figure 10. MACD technical indicator box](https://c.mql5.com/2/3/Image10_MACDBox.jpg)

Figure 10. MACD technical indicator box

**The MA (Moving Average) technical indicator box**

![Figure 11. MA (Moving Average) technical indicator box](https://c.mql5.com/2/3/Image11_MABox.jpg)

Figure 11. MA (Moving Average) technical indicator box

### **6\. Custom Indicators**

Custom Indicators tools are available at the Custom Indicators Menu and the Toolbar. EA Tree uses only MQL5 custom indicators.

### **7\. MQL5 Functions**

EA Tree has many boxes for MQL5 functions including:

- Time Series Boxes such as iOpen, iClose, HighestHigh, LowestLow, etc.
- Conversion Functions;
- Math Functions;
- String Functions.

### **8\. Tutorial: Creating a simple Expert Advisor**

In this section, let us use some of the tools covered in this article to create a simple EA.

First, let us list the Expert Advisor trade rules.

**Entry Rules**:

Open Long:

1. Current [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main > current [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") signal and
2. Previous [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main < previous [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") signal and
3. Current [EMA(20)](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") \> previous [EMA(20)](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma")

Open Short:

1. Current [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main < current [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") signal and
2. Previous [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main > previous MACD signal and
3. Current [EMA(20)](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") < previous [EMA(20)](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma")

**Exit Rules:**

Close Long: same rules as Open Short

Close Short: same rules as Open Long

We will use default trade box and no money management settings.

Let us get started:

1\. Create a Trade box and create an AND box and connect its output to the openLong button of the Trade box:

![Figure 12. Trade box + And Box](https://c.mql5.com/2/3/Image12_.png)

Figure 12. Trade box + And Box

2\. Create a Crossover box using and connect its T(true) output button to the A button of the And box.

![Figure 13. Trade box + And box + Crossover box](https://c.mql5.com/2/3/Image13_.png)

Figure 13. Trade box + And box + Crossover box

3\. Create a MACD box using, connect its Main output button to the A and C buttons of the Crossover box, and connect its Signal output button to the B and D buttons of the Crossover box.

The logic here is:

If current [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main > current [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") signal and

previous [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main < previous [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") signal

![Figure 14. Trade box + And box + Crossover box + MACD box](https://c.mql5.com/2/3/Image14_.png)

Figure 14. Trade box + And box + Crossover box + MACD box

4. Create a MA and If-Else boxes. Connect the Main output button of the MA box to the A and B buttons of the If-Else box. Connect the OUTPUT of IF-Else box to the B button of the AND box.

The subtree for the openLong condition is now complete with these three conditions:

1. Current [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main > current [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") signal and
2. Previous [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") main < previous [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") signal and
3. Current [EMA(20)](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") \> previous [EMA(20)](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma")

![Figure 15. Trade box + And box + Crossover box + MACD box + MA box + If-else box](https://c.mql5.com/2/3/Image15_.png)

Figure 15. Trade box + And box + Crossover box + MACD box + MA box + If-else box

5. Create another _And_ box and connect its output button to the openShort button of the trade box.

6. Since the openShort logic is the opposite of that of the openLong, connect the F output button of the Crossover box to the A button of the new And box. Also, connect the F output button of the If-Else box to the B button of the new And box:

![ Figure 16. Trade box + And box + Crossover box + MACD box + MA box + If-else box + And box](https://c.mql5.com/2/3/Image16_.png)

Figure 16. Trade box + And box + Crossover box + MACD box + MA box + If-else box + And box

7\. To get exit signals, connect the OUT variable of the first AND box to the closeLong button of the Trade box and the OUT variable of the second AND box to the closeShort button of the Trade box:

![ Figure 17. Adding CloseShort to the Trade box](https://c.mql5.com/2/3/Image17.png)

Figure 17. Adding CloseShort to the Trade box

8\. Double click on both MACD and MA boxes and select a number of variables to be input variables in the MQL5 EA that we will generate.

![Figure 18. Input parameters](https://c.mql5.com/2/3/Image18_.png)

Figure 18. Input parameters

9\. The layout is now complete. Save the Layout and save the MQL5 file. Then open the new MQL5 Expert Advisor file in [MetaEditor 5](https://www.metatrader5.com/en/automated-trading/metaeditor "MetaEditor") and compile it.

![Figure 19. Generated Expert Advisor](https://c.mql5.com/2/3/Image19_.png)

Figure 19. Generated Expert Advisor

10\. Finally, we optimize selected input variables in [MetaTrader 5 Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "MetaTrader 5 Strategy Tester").

![Figure 20. Testing of the Expert Advisor](https://c.mql5.com/2/3/Image20_.png)

Figure 20. Testing of the Expert Advisor

### Conclusion

There are many benefits of using EA Tree:

- Easy to use and understand drag and drop graphical user interface;
- You do not need programming background;
- You can quickly learn [MQL5](https://www.mql5.com/en/docs/basis "MQL5 Programming Language");
- Privacy of your trading secrets;
- You decrease Expert Advisor development time to minutes instead of days or months;
- You can develop complex MQL5 Expert Advisors with multiple currencies and multiple timeframes;
- You can incorporate multiple trading strategies into one Expert Advisor;
- You easily reuse code by saving and loading diagrams (trees of boxes);
- You are still able to import MQL5 custom indicators;
- You create correct MetaTrader 5 MQL code every time.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/337.zip "Download all attachments in the single ZIP archive")

[eatree-sample.mq5](https://www.mql5.com/en/articles/download/337/eatree-sample.mq5 "Download eatree-sample.mq5")(33.7 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/5126)**
(77)


![Leonardo Ciaccio](https://c.mql5.com/avatar/2022/11/6366ba15-3064.png)

**[Leonardo Ciaccio](https://www.mql5.com/en/users/leonardociaccio)**
\|
11 Oct 2016 at 21:09

For MT4 use this, without installation for all browser 😜 Watch this video, the best http://j.mp/2dtVd5z

![Balut](https://c.mql5.com/avatar/avatar_na2.png)

**[Balut](https://www.mql5.com/en/users/balut)**
\|
16 Dec 2016 at 12:59

**PH2000:**

154

[Blaiserboy](https://www.mql5.com/en/users/blaiserboy/news "dave mackay")2014.02.09 20:57 [#](https://www.mql5.com/en/forum/5126/page4#comment_759302 "Permanent link") [PT](https://www.mql5.com/en/forum/5126/page4# "Translate")

The trailing stop function re MT5 has to be revised as it is not called

Dear

Dave,

First I owe you an apology for asking.

I´m having some problems with my EA since a start this program. To fit the boxes was so easy(demo version) but my standard version Eatree EAs aren´t running in my mt5 broker.

I don´t work with forex and so five digits too.(only two digits). I´m having "divided by zero" problems and i think it is because off different types parameters off trailing step and lot size

As you see i´m not a C++ or MT5 programmer but a I saw most off .mt5 files with lot and stops (double) parameters

With EATree all i could see was integer stops parameters in the "trade 1 box” and double lots parameters to fit it

I´ve  already seen the user guide, demos videos and faq´s and  in despite off there is a lot off box fitting examples there is almost nothing about MM, Lot and stops parameters

Do i have to convert the integer stops parameters to doubles types one? How can i do it?

Many thanks for considering my request

PH 2000

![Balut](https://c.mql5.com/avatar/avatar_na2.png)

**[Balut](https://www.mql5.com/en/users/balut)**
\|
16 Dec 2016 at 13:01

I bought EATree mt5

No one EA works with my broker

I have problems "shared by zero"

I asked the EATree support for explanation

There was no usable answer.

Have you solved the prolem and how?

Greetings Balut

![Charles Magno](https://c.mql5.com/avatar/2019/8/5D434B01-E524.jpg)

**[Charles Magno](https://www.mql5.com/en/users/magnosud)**
\|
11 Dec 2018 at 12:24

Are there some [functional](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") ea builder?

Is EATree updated? Is it working on mt5 yet?


![rayapureddy](https://c.mql5.com/avatar/avatar_na2.png)

**[rayapureddy](https://www.mql5.com/en/users/rayapureddy)**
\|
16 Feb 2019 at 13:13

i have a problem, that in eatree when i make file and save it, its not coming in meta editor 5 , where it is saving iam not getting, can any one explain clearly from layout saving to mt5 saving ,up to meta editor5.


![Interview with Igor Korepin (ATC 2011)](https://c.mql5.com/2/0/Xupypr_ava.png)[Interview with Igor Korepin (ATC 2011)](https://www.mql5.com/en/articles/547)

Appearance of the Expert Advisor cs2011 by Igor Korepin (Xupypr) at the very top of the Automated Trading Championship 2011 was really impressive - its balance was almost twice that of the EA featured on the second place. However, despite such a sound breakaway, the Expert Advisor could not stay long on the first line. Igor frankly said that he relied much on a lucky start of his trading robot in the competition. We'll see if luck helps this simple EA to take the lead in the ATC 2011 race again.

![Interview with Tim Fass (ATC 2011)](https://c.mql5.com/2/0/avatar_Tim.png)[Interview with Tim Fass (ATC 2011)](https://www.mql5.com/en/articles/546)

A student from Germany Tim Fass (Tim) is participating in the Automated Trading Championship for the first time. Nevertheless, his Expert Advisor The\_Wild\_13 already got featured at the very top of the Championship rating and seems to be holding his position in the top ten. Tim told us about his Expert Advisor, his faith in the success of simple strategies and his wildest dreams.

![Interview with Ilnur Khasanov (ATC 2011)](https://c.mql5.com/2/0/aharata.png)[Interview with Ilnur Khasanov (ATC 2011)](https://www.mql5.com/en/articles/548)

The Expert Advisor of Ilnur Khasanov (aharata) is holding its place in our TOP-10 chart of the Automated Trading Championship 2011 participants from the third week already, though Ilnur's acquaintance with Forex has started only a year ago. The idea that forms the basis of the Expert Advisor is simple but the trading robot contains self-optimization elements. Perhaps, that is the key to its survival? Besides, the author had to change the Expert Advisor planned to be submitted for the Championship...

![ATC Champions League: Interview with Olexandr Topchylo (ATC 2011)](https://c.mql5.com/2/0/avatar2__1.png)[ATC Champions League: Interview with Olexandr Topchylo (ATC 2011)](https://www.mql5.com/en/articles/545)

Interview with Olexandr Topchylo (Better) is the second publication within the "ATC Champions League" project. Having won the Automated Trading Championship 2007, this professional trader caught the attention of investors. Olexandr says that his first place in the ATC 2007 is one of the major events of his trading experience. However, later on this popularity helped him discover the biggest disappointment - it is so easy to lose investors after the first drawdown on an investor account.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=epelowjfzmpxlmnfzpwzzopvzpwhsojv&ssn=1769092756130595122&ssn_dr=0&ssn_sr=0&fv_date=1769092756&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F337&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20MQL5%20Expert%20Advisors%20in%20minutes%20using%20EA%20Tree%3A%20Part%20One%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909275694749336&fz_uniq=5049281690897066132&sv=2552)

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