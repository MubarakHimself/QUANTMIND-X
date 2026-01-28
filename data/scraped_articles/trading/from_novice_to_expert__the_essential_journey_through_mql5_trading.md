---
title: From Novice to Expert: The Essential Journey Through MQL5 Trading
url: https://www.mql5.com/en/articles/15320
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:01:08.331050
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=sfofcbznyvqofsqvyifeuhupkhwokgbq&ssn=1769180467115977522&ssn_dr=0&ssn_sr=0&fv_date=1769180467&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15320&back_ref=https%3A%2F%2Fwww.google.com%2F&title=From%20Novice%20to%20Expert%3A%20The%20Essential%20Journey%20Through%20MQL5%20Trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918046718373055&fz_uniq=5068927395295067954&sv=2552)

MetaTrader 5 / Tester


Key Sections:

> 1. [Introduction: A Brief Walk Through My Journey to MQL5](https://www.mql5.com/en/articles/15320#para1)
> 2. [Uncovering the Secrets to a Thriving MQL5 Trader](https://www.mql5.com/en/articles/15320#para2)
> 3. [Implementation of Artificial Intelligence in Algorithm Development](https://www.mql5.com/en/articles/15320#para3)
> 4. [Conclusion](https://www.mql5.com/en/articles/15320#para4)

### Introduction: A Brief Walk Through My Journey to MQL5

When I joined the MQL5 community in 2019, I was a beginner with only manual trading skills on my phone and limited knowledge of algorithmic trading, despite a solid Computer Science background. I wish I had grasped the concepts of algorithmic trading earlier. Initially, I dismissed it and continued manual trading. However, three years later, while working as an electronic technician, I sought to continue trading despite my demanding job. This desire prompted me to learn about algorithmic trading to develop systems that would automate my trading or at least provide actionable signals.

Returning to the MQL5 community felt like finding a home of possibilities. The articles I encountered motivated me to read more, each enriched with valuable skills. Many articles guide readers through algorithm projects, evolving from basic to complex stages. Over the past two years, I have sharpened my MQL5 skills through various custom projects. Learning has become faster with the rapid improvement of technology and resources. I recommend reading this article thoroughly to avoid missing any crucial information.

### Who Is This Article For?

> 1. Absolute beginners to get started.
> 2. Medium and advanced developers to unlock new potentials.

I will guide you through the creation of a working program customized to your needs and inspire you to get started. The goal of this article is to make the MQL5 community user-friendly by simplifying algorithm development with emerging technologies.

The algorithmic trading environment has become much friendlier compared to a few years ago, thanks to accessible educational resources. Here are some questions that guided my focus, and I hope they inspire you too:

- How can beginners efficiently learn and master the basics of MQL5?
- What resources are available for MQL5 learners to quickly overcome specific coding challenges?
- How can traders invent new strategies, optimize, and automate using MQL5?

I will discuss three areas to kickstart your MQL5 journey as an absolute beginner or to advance your skills as a senior developer. One piece of advice I want to give you is: “Fall in love with text; there are hidden gems in reading.” Many miss opportunities because they dislike reading. The internet, especially the MQL5 community, offers countless opportunities for careers, authorship, development, and freelancing. I visit [mql5.com](https://www.mql5.com/en/articles/15320) at least three times a day, and I have integrated it with my mobile terminal ID, called MetaQuotes ID, to receive crucial notifications on my mobile device. As [William Shakespeare](https://en.wikipedia.org/wiki/William_Shakespeare "https://en.wikipedia.org/wiki/William_Shakespeare") said, “All that glitters is not gold.” True wealth lies in reading text, which may not be as appealing as images, yet even algorithm is composed of valuable text.

> -  Secret 1: MQL5 Community
> -  Secret 2: Desktop MetaTrader 5 Platform
> -  Secret 3: Artificial Intelligence

These three elements can be used together to make you a better MQL5 algorithm developer. While the goal of this article is to motivate you to get started, I will also ensure we build a basic program to implement these three secrets.

### Uncovering the Secrets to a Thriving MQL5 Trader

### Secret 1: MQL5 Community

MQL5 is a comprehensive resource for everything related to algorithmic trading. It contains everything you need to get started and continue developing as a trader. Here are some key components of the MQL5 community that are essential for developers:

![mql5.com](https://c.mql5.com/2/86/rect1.png)

- [Documentation](https://www.mql5.com/en/docs): Comprehensive resource providing detailed information about MQL5 programming, including functions, classes, and development guidelines.
- ' [AlgoBook](https://www.mql5.com/en/book)': A collection of algorithmic books to help develop your own algorithms.
- ' [NeuroBook](https://www.mql5.com/en/neurobook)': Specialized books focusing on neural networks and machine learning techniques in algorithmic trading using MQL5.
- ' [CodeBase](https://www.mql5.com/en/code)': A repository of free MQL5 scripts, indicators, and expert advisors (EAs) contributed by the MQL5 community for traders to download, use, and modify.
- [Forum](https://www.mql5.com/en/forum): An interactive platform for discussing trading strategies, programming issues, and sharing experiences and solutions related to MQL5 development.
- [Articles](https://www.mql5.com/en/articles): A library of articles offering tutorials, trading strategies, programming tips, and insights into various aspects of algorithmic trading with MQL5.
- [Market:](https://www.mql5.com/en/market) An online store where traders can buy and sell MQL5 trading applications, including EAs, indicators, and scripts.
- [Signals:](https://www.mql5.com/en/signals) A service allowing traders to subscribe to trading signals provided by experienced traders, enabling automated copying of their trades.
- [VPS:](https://www.mql5.com/en/vps) Virtual Private Server service provided by MetaTrader, enabling traders to run their trading robots and signals 24/7 without interruptions.
- [Freelance:](https://www.mql5.com/en/job) A platform where traders can hire freelance developers to create custom trading applications, indicators, or scripts in MQL5.
- [Quotes](https://www.mql5.com/en/quotes/overview): Real-time quotes of financial instruments, providing traders with up-to-date market prices necessary for making informed trading decisions.
- [Calendar:](https://www.mql5.com/en/economic-calendar "https://www.mql5.com/en/economic-calendar") An economic calendar listing significant economic events and news that can impact financial markets and trading strategies.
- ' [WebTermina](https://www.mql5.com/go?link=https://web.metatrader.app/terminal "https://web.metatrader.app/terminal") l': An online version of the MetaTrader trading platform, allowing traders to access their accounts and trade directly from a web browser without installing any software.
- [Search Box](https://www.mql5.com/): A tool enabling users to quickly find specific information, resources, or content within the MQL5 community and its extensive library of documentation, articles, and forum discussions.

### Secret 2: Desktop MetaTrader 5 Platform

This platform is another rich resource essential for mastering algorithmic trading. It features many of the resources found in the MQL5 community and serves as the starting point for writing algorithms. When strategizing, I start by viewing charts to identify repeating patterns in past market data, then I develop the logic on paper before coding my algorithm. [MetaTrader 5](https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=mt5editor&utm_campaign=search "https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe?utm_source=mt5editor&utm_campaign=search") comes with the MetaQuotes Language Editor and the Strategy Tester, among other features, making it a fantastic tool set for development and studying.

Let me briefly show you an example:

I open the chart to analyze what has been happening. Here is an image of GBP/USD before any indicator is applied. I then add EMA 50 and EMA 21 on the chart to see how they interact with price changes. I observe potential crossovers and mark them with rectangle tools. From this image, I can start strategizing for automation.

![GBPUSD](https://c.mql5.com/2/86/GBPUSDM1.png)

![Setting MA21](https://c.mql5.com/2/86/ma21.PNG)

![Setting MA 50](https://c.mql5.com/2/86/ma50.PNG)

![GBPUSD MA crossover strategy](https://c.mql5.com/2/86/GBPUSDM1-2.png)

### MetaQuotes Language Editor (MetaEditor):

Templates: MetaEditor comes with pre-made templates for scripts, Expert Advisors (EAs), and indicators, providing a solid starting point for projects and helping users understand the structure and key components of MQL5 programs.

Learning by Doing: Educational theorist John Dewey emphasized the importance of "learning by doing" in the early 20th century. Start by creating simple projects using these templates. Modify and experiment with them to get a feel for how different elements interact. This practical approach solidifies your foundational knowledge and aligns with Dewey's philosophy that active involvement in learning enhances understanding and retention.

### Secret 3: Artificial Intelligence

There are many AI models being invented every day, but let's briefly talk about ChatGPT. It is a conversational AI model designed to assist users by providing answers, inspiration, and productivity help. It is free to use and can aid in writing, learning, brainstorming, and more. ChatGPT was created by OpenAI, a company founded in 2015.

AI models like ChatGPT are trained on vast amounts of data, including programming languages and documentation. This training process involves analyzing patterns, understanding syntax, and learning best practices from numerous code examples across various languages. As a result, ChatGPT can offer expert-level guidance, helping you understand complex programming concepts and solve coding problems efficiently.

### Implementation of Artificial Intelligence in Algorithm Development

Here, I will guide you through how the collaboration of tools can simplify your learning and break the limitations caused by not knowing these resources. To get started with ChatGPT, I will split activities into two:

> 1. Understanding the MetaEditor templates with ChatGPT
> 2. Developing a script with AI assistance, called "My First MQL5 Program"

### Understanding the MetaEditor templates with ChatGPT

In MetaTrader 5 desktop, go to **Tools** and select **MetaQuotes Language Editor**, or press **F4** on your keyboard. A new window for the MetaEditor will open. Go to **File** and select **New**. Below is an animated image showing the process:

![MetaEditor: MyFirstIndicator](https://c.mql5.com/2/86/MetaEditor64_bXbn5xiI9B.gif)

I have named the file "MyFirstIndicator," but this is optional; you can name it as you like. When you do this, you will get a code template shown in the snippet below.

```
//+------------------------------------------------------------------+
//|                                             MyFirstIndicator.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//---

//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

As a beginner, you might ask yourself, "What on earth is this?" Well, let me show you how to easily understand everything as part of our secret unlocking goal.

1. Go to [chat.openai.com](https://www.mql5.com/go?link=http://chat.openai.com/ "http://chat.openai.com/") and initiate a chat.
2. Copy the program code from MetaEditor.
3. Type a prompt in ChatGPT: "Explain this MQL5 code in parts:" (paste the code you copied) and click send.

ChatGPT will reply with a detailed explanation of what is going on. Continue reading to discover more.

In response, ChatGPT split the program into snippets and explained it in a more engaging way, making it easy to understand the template. Here is the program explained by ChatGPT in italic text:

```
//+------------------------------------------------------------------+
//|                                             MyFirstIndicator.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
```

- _This section is a header comment that includes the name of the file, the copyright information, and a link to the MetaQuotes website. This is typical in MQL5 code to give context and_ attribution.

```
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window
```

_These #property directives provide metadata about the indicator:_

- _#property copyright: Indicates the copyright holder and year._
- _#property link: Provides a link to a website (typically where more information can be found)._
- _#property version: Specifies the version of the indicator._
- _#property indicator\_chart\_window: Indicates that the indicator will be drawn in the main chart window (as opposed to a separate window)._

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping

//---
   return(INIT_SUCCEEDED);
  }
```

_This section defines the OnInit function, which is called when the indicator is initialized:_

- _OnInit(): The function is responsible for setting up the indicator, such as allocating buffers and initializing variables._
- _return'(INIT\_SUCCEEDED);': Indicates that the initialization was successful. INIT\_SUCCEEDED is a predefined constant in MQL5._
- _Currently, the function does not perform any actions because the main body between the comments (//---) is empty._

```
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//---

//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

_This section defines the OnCalculate function, which is called to calculate the indicator values on each tick:_

- _OnCalculate: This is where the main logic of the indicator is implemented. It processes the price data and computes the indicator values._
- _Rates\_total: Total number of bars in the chart._
- _'Prev\_calculated': Number of bars calculated during the previous call._
- _Time\[\], open\[\], high\[\], low\[\], close\[\], tick\_volume\[\], volume\[\], spread\[\]: Arrays containing the time, open, high, low, close prices, tick volume, real volume, and spread for each bar._

The function currently does not perform any calculations, as the main body is empty. It simply returns rates\_total, indicating the total number of bars that have been processed. This is important for keeping track of the bars processed in subsequent calls. In summary, this is a skeleton code for a custom indicator in MQL5. It sets up the basic structure and metadata but does not perform any specific actions or calculations yet.

### Developing a script with AI assistance, called "My First MQL5 Program"

First, we need to know the purpose of the program we are developing. In this case, I will suggest a script program to tell me whether the D1 candle is a Bull or a Bear. Next, go to ChatGPT and send a prompt. This time, we want to provide the script program template for storage in AI memory and then ask to modify it for the conditions.

![Create a Script](https://c.mql5.com/2/86/script.PNG)

![Set the name of the Script](https://c.mql5.com/2/86/script2.PNG)

First, we need to know the purpose of the program we are developing. In this case, I will suggest a script program to tell me whether the D1 candle is a Bull or a Bear. After pressing "Finish," we get this program template, and we copy it to [ChatGPT](https://www.mql5.com/go?link=http://chat.openai.com/ "http://chat.openai.com/") to accomplish our plan according to this context:

```
//+------------------------------------------------------------------+
//|                                          MyFirstMQL5 Program.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

  }
//+------------------------------------------------------------------+
```

Below is a code snippet as prompted it on ChatGPT:

```
Create an MQL5 script from this template that tells day candle status on screen: //+------------------------------------------------------------------+
//|                                          MyFirstMQL5 Program.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---

  }
//+------------------------------------------------------------------+
```

ChatGPT response:

```
//+------------------------------------------------------------------+
//|                                          MyFirstMQL5 Program.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

// Input parameters for customizing the text display
input color TextColor = clrRed;               // Color of the text
input int FontSize = 20;                      // Font size
input ENUM_ANCHOR_POINT AnchorCorner = ANCHOR_LEFT_UPPER; // Text anchor point
input int X_Offset = 10;                      // X offset
input int Y_Offset = 10;                      // Y offset

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   // Get the opening price of the current day
   double openPrice = iOpen(_Symbol, PERIOD_D1, 0);

   // Get the closing price of the current day
   double closePrice = iClose(_Symbol, PERIOD_D1, 0);

   // Determine the candle status
   string candleStatus = (closePrice >= openPrice) ? "Bullish" : "Bearish";

   // Display the candle status on the screen
   DisplayCandleStatus(candleStatus);
  }

//+------------------------------------------------------------------+
//| Function to display the candle status on the screen              |
//+------------------------------------------------------------------+
void DisplayCandleStatus(string status)
  {
   string objName = "CandleStatusText";

   if(ObjectFind(0, objName) < 0)
     {
      // Create the text object if it doesn't exist
      ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, objName, OBJPROP_CORNER, AnchorCorner);
      ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, X_Offset);
      ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, Y_Offset);
      ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, FontSize);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, TextColor);
     }

   // Update the text
   ObjectSetString(0, objName, OBJPROP_TEXT, "Day Candle: " + status);
  }
//+------------------------------------------------------------------+
```

Testing

Here is the procedure that I followed during testing.

- Copy the Code: After receiving the modified script from ChatGPT, copy the code.
- Open MetaEditor: Launch MetaEditor on your computer.
- Create a New Script:

  - Go to File > New > Script.
  - Enter a name for your script, then click Next and Finish.

- Paste the Code: Replace the auto-generated code in the new script file with the modified code you received from ChatGPT.
- Compile the Script:

  - Click the Compile button on the MetaEditor toolbar.
  - If there are no errors, your script will be compiled successfully.

- Check the Output:

  - In the Navigator tab, look under the Scripts section to find your compiled script file.
  - Drag and drop the script onto a chart in MetaTrader 5 to see the results.

- Review the Results: Monitor the chart to verify if the script correctly identifies whether the D1 candle is a Bull or a Bear.

![Compiling in MetaEditor](https://c.mql5.com/2/86/MetaEditor64_9py18tUwWI.gif)

The program compiled successfully without any further editing. The D1 candle was confirmed to be Bullish by the script.

![The script is now added in the scripts folder](https://c.mql5.com/2/86/Script3.PNG)![Boom 500 Index D1](https://c.mql5.com/2/86/Boom_500_IndexDaily.png)

### Conclusion

Mastering MQL5 is a journey that combines learning, experimentation, and leveraging advanced tools like AI. By mastering the basics, engaging with community resources, overcoming dynamic challenges with AI assistance, optimizing strategies, and embracing automation, you can unlock your full potential as a trader. These three secrets are just the beginning; continuous learning and adaptation will ensure you stay ahead in the dynamic world of algorithmic trading. Embrace the opportunities that MQL5 offers, and watch your trading success soar.

Much can be done with AI, and the limit is your mind and the skill of communicating your idea with it. Not all programs created by AI will execute efficiently every time; sometimes, you need to go the extra mile in debugging and revisiting community documentation and MQL5 Books for in-depth research. There is a lot to share, but let's save that for next time. At least you now have a companion. Happy developing further!

| File Name | Description |
| --- | --- |
| 'MyFirstIndicator.mq5' | MQL5 indicator Template. |
| MyFirstMQL5 Program.mq5 | Script algorithm. |
| MyFirstMQL5 Program.ex5 | Compile scrip program ready for use. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15320.zip "Download all attachments in the single ZIP archive")

[MyFirstIndicator.mq5](https://www.mql5.com/en/articles/download/15320/myfirstindicator.mq5 "Download MyFirstIndicator.mq5")(1.59 KB)

[MyFirstMQL5\_Program.mq5](https://www.mql5.com/en/articles/download/15320/myfirstmql5_program.mq5 "Download MyFirstMQL5_Program.mq5")(2.46 KB)

[MyFirstMQL5\_Program.ex5](https://www.mql5.com/en/articles/download/15320/myfirstmql5_program.ex5 "Download MyFirstMQL5_Program.ex5")(9.24 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/470700)**
(1)


![Yuriy Scherbakov](https://c.mql5.com/avatar/2019/2/5C6822DD-B96A.jpg)

**[Yuriy Scherbakov](https://www.mql5.com/en/users/yura.scherbakov)**
\|
29 Apr 2025 at 18:46

**MetaQuotes:**

Published the article [From Beginner to Expert: Essentials on the Way to Trade in MQL5](https://www.mql5.com/ru/articles/15320):

Author: [Clemence Benjamin](https://www.mql5.com/ru/users/Billionaire2024 "Billionaire2024")

Chat. Openai. Com is not working


![Developing a multi-currency Expert Advisor (Part 5): Variable position sizes](https://c.mql5.com/2/73/Developing_a_multi-currency_advisor_Part_1___LOGO__4.png)[Developing a multi-currency Expert Advisor (Part 5): Variable position sizes](https://www.mql5.com/en/articles/14336)

In the previous parts, the Expert Advisor (EA) under development was able to use only a fixed position size for trading. This is acceptable for testing, but is not advisable when trading on a real account. Let's make it possible to trade using variable position sizes.

![Creating a Dynamic Multi-Symbol, Multi-Period Relative Strength Indicator (RSI) Indicator Dashboard in MQL5](https://c.mql5.com/2/86/Creating_a_Dynamic_Multi-Symbol_Indicator_Dashboard_in_MQL5____LOGO__1.png)[Creating a Dynamic Multi-Symbol, Multi-Period Relative Strength Indicator (RSI) Indicator Dashboard in MQL5](https://www.mql5.com/en/articles/15356)

In this article, we develop a dynamic multi-symbol, multi-period RSI indicator dashboard in MQL5, providing traders real-time RSI values across various symbols and timeframes. The dashboard features interactive buttons, real-time updates, and color-coded indicators to help traders make informed decisions.

![Reimagining Classic Strategies (Part III): Forecasting Higher Highs And Lower Lows](https://c.mql5.com/2/86/Reimagining_Classic_Strategies__Forecasting_Higher_Highs_And_Lower_Lows___LOGO.png)[Reimagining Classic Strategies (Part III): Forecasting Higher Highs And Lower Lows](https://www.mql5.com/en/articles/15388)

In this series article, we will empirically analyze classic trading strategies to see if we can improve them using AI. In today's discussion, we tried to predict higher highs and lower lows using the Linear Discriminant Analysis model.

![Practicing the development of trading strategies](https://c.mql5.com/2/73/Experience_in_developing_a_trading_strategy___LOGO.png)[Practicing the development of trading strategies](https://www.mql5.com/en/articles/14494)

In this article, we will make an attempt to develop our own trading strategy. Any trading strategy must be based on some kind of statistical advantage. Moreover, this advantage should exist for a long time.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/15320&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068927395295067954)

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