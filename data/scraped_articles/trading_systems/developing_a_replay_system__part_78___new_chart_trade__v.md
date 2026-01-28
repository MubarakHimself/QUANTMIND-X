---
title: Developing a Replay System (Part 78): New Chart Trade (V)
url: https://www.mql5.com/en/articles/12492
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:32:16.903604
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/12492&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069526349959333503)

MetaTrader 5 / Examples


### Introduction

In the previous article " [Developing a Replay System (Part 77): New Chart Trade (IV)](https://www.mql5.com/en/articles/12476), I explained the subject as thoroughly as possible, while trying not to overcomplicate matters, even though this is often a rather complex topic. The question at hand is: how do you develop a communication protocol? This is essential if you want to transfer information between different applications or programs, whether they are running in the same environment or not. In our specific case, we want the Chart Trade indicator to be able to instruct the Expert Advisor on what actions to take. In other words, when the user tells Chart Trade they want to sell, the Expert Advisor will execute a sell order. When the user indicates they want to buy, the Expert Advisor will place a market buy order.

In recent articles, I have focused mainly on explaining why we should create the Chart Trade indicator, and, most importantly,on how to design a message protocol before attempting to code anything else. Up until now, however, we have concentrated solely on the indicator.

That explanation, however, is incomplete without first examining the receiving side, which resides within the Expert Advisor. This is because MetaTrader 5 explicitly forbids indicators from sending orders, managing positions, or directly interacting with the system responsible for communicating with the trading server. In other words, indicators cannot initiate, close, or modify positions or orders.

Some might argue that a loophole exists here. In reality, it is something else entirely. Something we will explore in future articles when we develop another tool that will prove indispensable. For now, however, we remain at the early stages of building the order system.

At this moment, our main concern is different: understanding how the Expert Advisor knows what is happening. This is critical, because the user will never interact with the Expert Advisor directly. Instead, the user will interact exclusively with the Chart Trade indicator. Chart Trade in turn must communicate instructions to the Expert Advisor. After all, the indicator alone is useless if the component responsible for actually sending requests to the market - whether to open or close a position - does not know what is happening. Remember: only the Expert Advisor has permission to execute these operations in MetaTrader 5. No other program can do this.

So, the central question of this article is:

### How does the Expert Advisor understand the Chart Trade?

Let's start with this very question. If you are not yet sure how this works, I recommend revisiting the previous article, where I explained how to design a message protocol. That background is essential, because here we will see how to make the "magic" happen. Neither the Chart Trade indicator nor the Expert Advisor knows of the other's existence. And they don't need to. It is enough for both to trust that the message will be delivered and understood.

Even without knowing that an Expert Advisor is on the chart, the Chart Trade indicator can still convey to it the user's intentions. This is a fascinating process to observe.

More importantly, once you understand this mechanism, you will be able to design much more versatile programs and applications for MetaTrader 5. And this knowledge goes far beyond MetaTrader 5 itself.

Yes, dear reader. This kind of message exchange between programs or processes is also widely used in operating systems such as Windows, Linux, and macOS. Every modern system relies on this principle: you develop several smaller programs that can communicate with one another. Thereby, a broad and sustainable ecosystem is created.

Each application can be optimized to perform a specific task. But when combined, they can accomplish virtually anything, at far lower cost in terms of implementation, maintenance, and improvement.

From this point on, I encourage you to move away from the idea of building large, complex, all-in-one applications that are cumbersome to maintain or update. Instead, think about creating smaller, simpler programs. This simplicity brings agility, making it easier to enhance or refine the application. For many, this might be new. But believe me, most of the industry works this way. No one builds fully monolithic systems anymore, because doing so is not only expensive but also highly impractical, especially when it comes time to modify or improve them.

A simpler application is much easier to debug, adjust, or optimize. For this reason, the Expert Advisor will be programmed only to handle what MetaTrader 5 requires it to handle: sending and modifying orders and positions. Everything else will be delegated to other programs and applications.

To truly understand how the Expert Advisor interprets Chart Trade, let's begin by creating a very simple piece of code. And I do mean simple. At this stage, we will not send orders, close positions, or reverse trades - not yet. Doing so now would only complicate matters unnecessarily.

What we need is the simplest possible example, so that you can understand how the receiving side works. This will give you a clearer understanding of how the message protocol allows the Chart Trade indicator (which is unaware of the Expert Advisor's existence) to trigger actions in it.

At the same time, the Expert Advisor, which likewise has no knowledge of the Chart Trade, can still respond to user requests. Remember: the user never interacts directly with the Expert Advisor.

The simplest code to achieve this can be found below. It is presented in full.

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Demo version between interaction"
04. #property description "of Chart Trade and Expert Advisor"
05. #property version   "1.78"
06. #property link "https://www.mql5.com/pt/articles/11760"
07. //+------------------------------------------------------------------+
08. #include <Market Replay\Defines.mqh>
09. //+------------------------------------------------------------------+
10. class C_Decode
11. {
12.    private   :
13.       struct stInfoEvent
14.       {
15.          EnumEvents ev;
16.          string     szSymbol;
17.          bool       IsDayTrade;
18.          ushort     Leverange;
19.          double     PointsTake,
20.                     PointsStop;
21.       }info[1];
22.    public   :
23. //+------------------------------------------------------------------+
24.       C_Decode()
25.          {
26.             info[0].szSymbol = _Symbol;
27.          }
28. //+------------------------------------------------------------------+
29.       bool Decode(const int id, const string sparam)
30.       {
31.          string Res[];
32.
33.          if (StringSplit(sparam, '?', Res) != 6) return false;
34.          stInfoEvent loc = {(EnumEvents) StringToInteger(Res[0]), Res[1], (bool)(Res[2] == "D"), (ushort) StringToInteger(Res[3]), StringToDouble(Res[4]), StringToDouble(Res[5])};
35.          if ((id == loc.ev) && (loc.szSymbol == info[0].szSymbol)) info[0] = loc;
36.
37.          ArrayPrint(info, 2);
38.
39.          return true;
40.       }
41. //+------------------------------------------------------------------+
42. }*GL_Decode;
43. //+------------------------------------------------------------------+
44. int OnInit()
45. {
46.    GL_Decode = new C_Decode;
47.
48.    return INIT_SUCCEEDED;
49. }
50. //+------------------------------------------------------------------+
51. void OnTick() {}
52. //+------------------------------------------------------------------+
53. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
54. {
55.    switch (id)
56.    {
57.       case CHARTEVENT_CUSTOM + evChartTradeBuy     :
58.       case CHARTEVENT_CUSTOM + evChartTradeSell    :
59.       case CHARTEVENT_CUSTOM + evChartTradeCloseAll:
60.          (*GL_Decode).Decode(id - CHARTEVENT_CUSTOM, sparam);
61.          break;
62.    }
63. }
64. //+------------------------------------------------------------------+
65. void OnDeinit(const int reason)
66. {
67.    delete GL_Decode;
68. }
69. //+------------------------------------------------------------------+
```

Expert Advisor source code

Come on, you're telling me that the code above is simple? Honestly, I don't get it. To me, it looks extremely complicated. I can hardly understand a thing. I feel completely lost here. If this is what you call simple, then I can only imagine what something complicated must look like! Yes, this code really is quite simple. But it does use certain elements that many of you don't normally encounter. Or, more precisely, things that many of you don't even realize are possible in MQL5. So if that's your case, and if you truly intend to become a professional programmer in the future, stay with me. Follow the explanation I'm about to give, because things are in fact much simpler than they might appear.The code is straightforward and very direct in achieving exactly what it needs to do.

And the knowledge I'm about to share will help you slow down, think more clearly, and approach problems more calmly. It also serves as a reminder that programming can - and should - be fun. If you see it only as boring, difficult, or overly complicated work, then perhaps you should consider doing something else entirely. Forget about programming or development. Because when you are coding, you should truly feel like a child in a candy store - never quite sure which candy you'll pick first.

But enough digression. Let's look at how this code actually works. The first seven lines shouldn't be an issue for anyone to understand. Even beginners should follow them easily. Line eight, however, is a familiar one. Here we are including a file in our code. This file is a header containing definitions that we will soon need. For now, let's skip over the code between lines 10 and 42. This is because we'll go through it carefully later on. First, we need to cover some small but important details.

That brings us to the first truly essential function in our code. It is OnInit, which begins on line 44. Because of what happens in line 42, things might work a little differently than many would expect. In line 46, we use the new operator to allocate memory and initialize the class C\_Decode. Why allocate memory this way? Wouldn't it be simpler to just let the compiler handle it? Yes, it would be easier. But as a programmer, you must learn to control when and where a class is used. If you leave it up to the compiler, you may sometimes end up using the class with values you did not expect.

Believe me, it's not uncommon. Especially in large codebases, where classes are declared in multiple places, confusion is inevitable. At some point, things will get tangled. By explicitly using new and delete, you ensure full control over where your code begins and ends. Many beginner programmers struggle with this concept. Odd as it may sound, some cannot clearly identify where their code ends. or even where it begins. But it happens.

So when line 46 runs, memory will be allocated for the C\_Decode class. and at the same time, line 24 will be called. This line is the constructor of the C\_Decode classс In this constructor, only one variable is initialized. It is the symbol name, or in other words, the name of the asset where the Expert Advisor is running. Now, pay close attention: in this model, cross-order permissions are not enabled. At this point, this might not be entirely clear to you. But keep reading - you'll soon understand why cross-order operations are not permitted here. With a few small modifications (and they really are simple), cross orders could be enabled. But for now, don't worry about that. Remember: this code exists only to demonstrate how the message protocol works. It is not meant to actually send any requests to the trading server.

Back to the OnInit function. After executing line 46, line 48 returns a value that indicates initialization was successful. This is important. Because if the return value is not **INIT\_SUCCEEDED**, MetaTrader 5 will automatically take steps to deactivate the Expert Advisor. One of these steps is triggering the Deinit event, which calls the OnDeinit procedure in your Expert Advisor code.

If your code does not contain this procedure, MetaTrader 5 will fall back on default measures. Either way, your application will no longer receive CPU time. In other words, it won't be scheduled for execution.

And any information left on the chart that belongs to the application will be ignored. It's common for applications to create and place objects on the chart. But if MetaTrader 5 triggers Deinit and your code doesn't remove those objects. They'll remain on the chart. Often this will be misleadingly showing false or invalid data.

For this reason, it's good practice to always implement the OnDeinit procedure. Often it will do very little. But in our case, it contains one important line: line 67. Here we use the delete operator to release the memory we allocated. At this point, the destructor of C\_Decode is called. Since we haven't explicitly declared a destructor, the compiler automatically provides one when **delete** is used. Because the **delete** operator requires a destructor. You don't need to worry about this. But it's worth knowing how it works under the hood.

Now that we've covered where the Expert Advisor code begins and ends, let's move on. At line 51, we encounter the procedure that is called on every tick received by the asset. This procedure is mandatory in every Expert Advisor. However, you should avoid placing too much logic inside it. The reasons for this are explained in the series of articles on building a fully automated Expert Advisor. If you're interested in creating an EA with some level of automation, I strongly recommend reading that series.

It consists of 15 articles, beginning with [Creating an EA that works automatically (Part 01): Concepts and structures](https://www.mql5.com/en/articles/11216). Going through that series will help you tremendously. In fact, in this replay/simulation series, we'll be reusing some concepts from the automated EA series. Either way, don't skip it.

Now, moving forward: the other important procedure appears in line 53. It is OnChartEvent. And this is where things start to get truly interesting. But to keep things organized, we'll examine this in the next section.

### How to Decode a Message from an Event

In previous articles in this replay/simulator series, I explained how EventChartCustom and OnChartEvent are interconnected. We made heavy use of this connection to enable the replay/simulator service to send data to the control indicator. This is required so that the indicator could know exactly where we were in the simulation or replay. If you haven’t read those articles (and have just arrived at this point in the series), I suggest you go back and check them out.

One example is [Developing a Replay System (Part 60): Playing the Service (I)](https://www.mql5.com/en/articles/12086). There are seven articles on the subject. Reading them all will help you understand not only the technical details but also why this approach makes sense.

Back then, the service communicated with the control indicator using numeric parameters. That was the easy part, since the information was readily available in parameters like lparam or dparam. But here, things change. This time we'll use the sparam parameter. In other words, the message will now be a string. And inside that string is the information we need. The catch is that the information is encoded. The way it is encoded is defined by the message protocol. I explained this concept in the previous article. Here, we'll focus on decoding that message according to the protocol.

Recall that in the last article I mentioned the message would contain the event itself. This is important because it allows the receiver - in our case, the Expert Advisor - to verify whether the message has been corrupted or not.

Since the same protocol is used across the three recognized event types, they can all be handled consistently. In lines 57–59, you'll see these three events, which the Expert Advisor must intercept. And because they can all go through the same decoding process, the only truly relevant line is line 60. That line calls the code starting on line 29. Now pay very close attention. If you don't understand this explanation right away, read it again, because although it's uncommon, it's extremely important.

At line 31, we declare a local variable: a dynamic array. Since it's dynamic, the program itself handles memory allocation. So we don't need to worry about it. Then, at line 33, we call [StringSplit](https://www.mql5.com/en/docs/strings/stringsplit). It fills the array with substrings separated by a given character. In our case, that character is a question mark (?). Why this specific choice?

I explained this in the last article. The question mark was chosen as the delimiter for grouping values. During splitting, we expect to obtain a fixed number of substrings. If the message is valid (or at least minimally valid) we should get six strings. If StringSplit returns more or fewer, the test fails and we return false. If exactly six are returned, we proceed.

Now, line 34 is where things get really interesting. It's not common code, but it'’s not alien either. To understand it, you need to see it alongside the structure declared on line 13. **Pay close attention here**. Each element on line 34 directly corresponds to a field in the stInfoEvent structure. What I did here was intentional. Pay attention to the order of the variables declared in stInfoEvent. It is very important. Now look at the order of the values in line 34. Wow! They match! The result is equivalent to writing this more explicit code:

```
      stInfoEvent loc;

      loc.ev = (EnumEvents) StringToInteger(Res[0]);
      loc.szSymbol = Res[1];
      loc.IsDayTrade = (Res[2] == "D");
      loc.Leverange = (ushort) StringToInteger(Res[3]);
      loc.PointsTake = StringToDouble(Res[4]);
      loc.PointsStop = StringToDouble(Res[5]);
```

Code snippet - Model 01

The two approaches are functionally identical. However, there's an important caveat: line 34 carries a risk. This risk is avoided in the above code version. That risk arises if you change the structure stInfoEvent. Why? Changing the stInfoEvent structure is a problem when writing code as shown in line 34, but not when writing code as shown in the snippet.

The reason is quite simple. This may seem silly, but it will nonetheless cause a lot of headaches and hours of trying to figure out why everything is going wrong. Let's imagine you decide to swap the variables declared at lines 19 and 20. With the explicit version above, nothing breaks - the assignments still map correctly.

But with the compact assignment of line 34, everything shifts out of place. What was supposed to be assigned to PointsTake ends up in PointsStop, and vice versa. Sounds like nonsense. This isn't a joke. It really happens.

It has to do with how the compiler organizes variables in memory. It's the same principle sometimes used intentionally in legacy C programming. Developers manipulate memory layouts directly to force certain behaviors. But that's far beyond the scope of this article.

The takeaway is simple: only use the compact assignment style once your structure is fully defined and finalized. And once it's finalized, **do not change it**. If you do, you must revisit and adjust every assignment written in the compact style (like line 34). To see this, use this code to test the system, simply changing line 34 or the order of variable declarations in the stInfoEvent structure.

Let's get back to the code. At line 35, we check whether the data matches expected values - both the asset name and the event type. If so, we assign the values to a static array. **NOTE**: This assignment is only for testing and demonstration purposes. For now, we just need stInfoEvent to be placed in an array. The reason is line 37. In this line, we use the MQL5 function [ArrayPrint](https://www.mql5.com/en/docs/array/arrayprint). This function displays the array contents in the terminal. One such result is shown in the image below.

![Image](https://c.mql5.com/2/147/Image_01__2.png)

### Conclusion

As explained in this article, the Expert Advisor is still not capable of sending requests to the trading server. That's intentional. Before doing so, you must fully understand how the code works - and, more importantly, how the communication protocol between Chart Trade and the Expert Advisor functions. Without this foundation, you risk running into serious issues.

Still, presenting it this way has its benefits. Along the way, I was able to explain a concept you may well encounter in the future and not immediately understand why it does (or doesn't) work. In any case, we now have the building blocks in place for what's coming next. In the video below, you can see how the system behaves in practice. So, until the next article, where things will get even more interesting.

YouTube

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/12492](https://www.mql5.com/pt/articles/12492)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12492.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/12492/anexo.zip "Download Anexo.zip")(420.65 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)
- [Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

**[Go to discussion](https://www.mql5.com/en/forum/494427)**

![Black Hole Algorithm (BHA)](https://c.mql5.com/2/107/Black_Hole_Algorithm_LOGO.png)[Black Hole Algorithm (BHA)](https://www.mql5.com/en/articles/16655)

The Black Hole Algorithm (BHA) uses the principles of black hole gravity to optimize solutions. In this article, we will look at how BHA attracts the best solutions while avoiding local extremes, and why this algorithm has become a powerful tool for solving complex problems. Learn how simple ideas can lead to impressive results in the world of optimization.

![Neural Networks in Trading: A Multi-Agent Self-Adaptive Model (Final Part)](https://c.mql5.com/2/104/Multi-agent_adaptive_model_MASA___LOGO__1.png)[Neural Networks in Trading: A Multi-Agent Self-Adaptive Model (Final Part)](https://www.mql5.com/en/articles/16570)

In the previous article, we introduced the multi-agent self-adaptive framework MASA, which combines reinforcement learning approaches and self-adaptive strategies, providing a harmonious balance between profitability and risk in turbulent market conditions. We have built the functionality of individual agents within this framework. In this article, we will continue the work we started, bringing it to its logical conclusion.

![Price Action Analysis Toolkit Development (Part 38): Tick Buffer VWAP and Short-Window Imbalance Engine](https://c.mql5.com/2/166/19290-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 38): Tick Buffer VWAP and Short-Window Imbalance Engine](https://www.mql5.com/en/articles/19290)

In Part 38, we build a production-grade MT5 monitoring panel that converts raw ticks into actionable signals. The EA buffers tick data to compute tick-level VWAP, a short-window imbalance (flow) metric, and ATR-based position sizing. It then visualizes spread, ATR, and flow with low-flicker bars. The system calculates a suggested lot size and a 1R stop, and issues configurable alerts for tight spreads, strong flow, and edge conditions. Auto-trading is intentionally disabled; the focus remains on robust signal generation and a clean user experience.

![Automating Trading Strategies in MQL5 (Part 29): Creating a price action Gartley Harmonic Pattern system](https://c.mql5.com/2/165/19111-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 29): Creating a price action Gartley Harmonic Pattern system](https://www.mql5.com/en/articles/19111)

In this article, we develop a Gartley Pattern system in MQL5 that identifies bullish and bearish Gartley harmonic patterns using pivot points and Fibonacci ratios, executing trades with precise entry, stop loss, and take-profit levels. We enhance trader insight with visual feedback through chart objects like triangles, trendlines, and labels to clearly display the XABCD pattern structure.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qkqlwjfqrhwpqgqqrhxnbuoildsfqiyn&ssn=1769182335279632507&ssn_dr=0&ssn_sr=0&fv_date=1769182335&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12492&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2078)%3A%20New%20Chart%20Trade%20(V)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918233580528701&fz_uniq=5069526349959333503&sv=2552)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).