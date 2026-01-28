---
title: Developing a trading Expert Advisor from scratch (Part 16): Accessing data on the web (II)
url: https://www.mql5.com/en/articles/10442
categories: Integration, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:24:27.656658
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/10442&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068187806221661850)

MetaTrader 5 / Integration


### Introduction

In the previous article " [Developing a trading Expert Advisor from scratch (Part 15): Accessing data on the web (I)](https://www.mql5.com/en/articles/10430)", we presented the entire logic and ideas behind the methods of using the MetaTrader 5 platform to access marked data from specialized websites.

In that article, we considered how to access these sites and how to find and retrieve information from them in order to use it in the platform. But it doesn't end there, as simply capturing data doesn't make much sense. The most interesting part is to learn how to take this data from the platform and use it in an Expert Advisor. The method to do this is not so obvious and so it is hard to implement without knowing and understanding all the functions available in MetaTrader 5.

### Planning and implementation

If you have not read and understood the previous article, I recommend that you do so and try to understand all the concepts present there, because here we will continue that topic - we will study a huge number of things, solve a series of problems and in in the end we will come to a beautiful solution, since we will use MetaTrader 5 in a way that has not yet been explored. I say this because it was difficult to find links to use some of the features present in the platform, but here I will try to explain how to use one of these resources.

So, let's get ready and get to work.

### 1\. Access to Internet data through an Expert Advisor

This is the most interesting part that can be implemented in this system. Although it is a simple thing, it is by far the most dangerous if poorly planned. Dangerous because it can leave the EA waiting for a response from the server, even if only for a moment.

This logic is shown in the figure below:

![](https://c.mql5.com/2/45/001__1.png)

Let's see how the EA interacts directly with the web server that contains the information we want to capture. Below you can see a complete code example that works exactly like this.

```
#property copyright "Daniel Jose"
#property version "1.00"
//+------------------------------------------------------------------+
int OnInit()
{
        EventSetTimer(1);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        EventKillTimer();
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
void OnTimer()
{
        Print(GetDataURL("https://tradingeconomics.com/stocks", 100, "<!doctype html>", 2, "INDU:IND", 172783, 173474, 0x0D));
}
//+------------------------------------------------------------------+
string GetDataURL(const string url, const int timeout, const string szTest, int iTest, const string szFind, int iPos, int iInfo, char cLimit)
{
        string  headers, szInfo = "";
        char    post[], charResultPage[];
        int     counter;

        if (WebRequest("GET", url, NULL, NULL, timeout, post, 0, charResultPage, headers) == -1) return "Bad";
        for (int c0 = 0, c1 = StringLen(szTest); c0 < c1; c0++) if (szTest[c0] != charResultPage[iTest + c0]) return "Failed";
        for (int c0 = 0, c1 = StringLen(szFind); c0 < c1; c0++) if (szFind[c0] != charResultPage[iPos + c0]) return "Error";
        for (counter = 0; charResultPage[counter + iInfo] == 0x20; counter++);
        for (;charResultPage[counter + iInfo] != cLimit; counter++) szInfo += CharToString(charResultPage[counter + iInfo]);

        return szInfo;
}
//+------------------------------------------------------------------+
```

If you look closely, we can see that the code is exactly the same that was created in the previous article, but now this code is part of the EA, and has been adapted for this, i.e. if something already worked there, it will work here. The difference is that the EA contains a new condition, which implies that the code will be executed every second, that is, the EA will make a request to the desired web server every second and wait for a response. Then it will provide the captured data and return to other internal functions. This loop will be repeated throughout the lifetime of the EA. The result of the execution can be seen below.

![](https://c.mql5.com/2/45/ScreenRecorderProject65.gif)

Although this is done exactly this way, I do not recommend this practice because the EA will get stuck waiting for the server response even for a few moments - this can endanger the trading system of the platform and the EA itself. On the other hand, if you are interested in learning the method, you can learn a lot through this system.

But if you have a local server that will route information between the Internet and the platform, perhaps this method will be enough. In this case, if the system makes a request, then the following will happen: the local server will not yet have any information and will quickly respond, which will save you the next steps.

Now let's consider another way to perform this type of task, which is a little more secure. Since we will use the MetaTrader 5 threading system to achieve at least some security and to prevent the EA from being subject to the conditions of the remote web server, we can hang for a few moments waiting for the remote server to respond. We will create additional conditions for the EA to know what is happening, being able to collect information from the Web.

### **2\. Creating a communication channel**

A better yet simpler way to get data collected online and to use it in an EA is a channel. Although it works, in some cases it is not very suitable for us, since there are limitations on the use of such channels. But at least the EA will be able to access information collected on the web without having to waiting for a response from a remote server.

It was already mentioned above that the easiest way to solve the problem is data routing: creating a local server that would download the data and provide it to the MetaTrader 5 platform. But this requires certain knowledge and computing power and complicates almost all cases. However, we can use the MetaTrader 5 features to create a very similar channel, which would be much simpler than routing through a local server.

The figure below shows how we will promote such a channel.

![](https://c.mql5.com/2/45/002__1.png)

It is created using an object. Note that the EA will look inside the object for the information that the script has placed there. To understand how this actually works, let us take a look at three codes, which are shown in full below. One will be an EA, the other one will be a header that contains the object, and the third will be a script.

```
#property copyright "Daniel Jose"
#property description "Testing Inner Channel"
#property version "1.00"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
int OnInit()
{

        EventSetTimer(1);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        EventKillTimer();
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
void OnTimer()
{
        Print(GetInfoInnerChannel());
}
//+------------------------------------------------------------------+
```

The following code is the header we need to use. Please note that the object here is declared to be shared between the EA and the script.

```
//+------------------------------------------------------------------+
//|                                          Canal Intra Process.mqh |
//|                                                      Daniel Jose |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_NameObjectChannel   "Inner Channel Info WEB"
//+------------------------------------------------------------------+
void CreateInnerChannel(void)
{
        long id;

        ObjectCreate(id = ChartID(), def_NameObjectChannel, OBJ_LABEL, 0, 0, 0);
        ObjectSetInteger(id, def_NameObjectChannel, OBJPROP_COLOR, clrNONE);
}
//+------------------------------------------------------------------+
void RemoveInnerChannel(void)
{
        ObjectDelete(ChartID(), def_NameObjectChannel);
}
//+------------------------------------------------------------------+
inline void SetInfoInnerChannel(string szArg)
{
        ObjectSetString(ChartID(), def_NameObjectChannel, OBJPROP_TEXT, szArg);
}
//+------------------------------------------------------------------+
inline string GetInfoInnerChannel(void)
{
        return ObjectGetString(ChartID(), def_NameObjectChannel, OBJPROP_TEXT);
}
//+------------------------------------------------------------------+
```

And finally, here is a script. It will replace the creation of a local server, and will actually do the job of the server.

```
#property copyright "Daniel Jose"
#property version   "1.00"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
void OnStart()
{
        CreateInnerChannel();
        while (!IsStopped())
        {
                SetInfoInnerChannel(GetDataURL("https://tradingeconomics.com/stocks", 100, "<!doctype html>", 2, "INDU:IND", 172783, 173474, 0x0D));
                Sleep(200);
        }
        RemoveInnerChannel();
}
//+------------------------------------------------------------------+
string GetDataURL(const string url, const int timeout, const string szTest, int iTest, const string szFind, int iPos, int iInfo, char cLimit)
{
        string  headers, szInfo = "";
        char    post[], charResultPage[];
        int     counter;

        if (WebRequest("GET", url, NULL, NULL, timeout, post, 0, charResultPage, headers) == -1) return "Bad";
        for (int c0 = 0, c1 = StringLen(szTest); (!_StopFlag) && (c0 < c1); c0++) if (szTest[c0] != charResultPage[iTest + c0]) return "Failed";
        for (int c0 = 0, c1 = StringLen(szFind); (!_StopFlag) && (c0 < c1); c0++) if (szFind[c0] != charResultPage[iPos + c0]) return "Error";
        for (counter = 0; (!_StopFlag) && (charResultPage[counter + iInfo] == 0x20); counter++);
        for (;(!_StopFlag) && (charResultPage[counter + iInfo] != cLimit); counter++) szInfo += CharToString(charResultPage[counter + iInfo]);

        return (_StopFlag ? "" : szInfo);
}
//+------------------------------------------------------------------+
```

Here we have an object that the EA can see and that is created by the script. The idea is that the EA and the script can coexist in the same chart, then this object will be the communication channel between the EA and the script. The EA will be a client, the script will be a server, and the object will be a communication channel between them. Thus, the script will capture the values on the remote web server and will put the desired value into the object. The EA will see from time to time what value is in the object (if the object exists), because if the script is not running, the object should not be available. Anyway, the time when the EA looks at the value in the object does not violate the script in any way. If the script is blocked because it is waiting for a response from a remote server, this will not affect the EA, as it will continue operation regardless of what the script does.

While this is a great solution, it's not perfect: the problem is with the script.

To understand this, watch the following video, paying attention to every detail.

YouTube

Everything works great. It was expected, since this kind of solution is widely used in programming when developing a client-server program, where we do not want one to block the other. In other words, we use a channel to communicate between processes. Often, when they are in the same environment, the channel will be created using memory — an isolated area is specifically allocated for this, which is however shared and visible to both client and server. The server adds data there, and the client visits the same area to grab the existing data. So, one does not depend on the other, while they both are connected.

The idea is to use the same principle. But the way the script operates generates a problem. When we switch the timeframe, the script closes, and even when we use an infinite loop, it is closed by MetaTrader 5. Since this happens, we would have to reinitialize the script and put it back to the chart. But if we need to constantly switch timeframes, this will be a problem, not to mention the need to launch the script back on the chart every time.

Furthermore, we may forget to check whether the script is on the chart or not and hence you will end up using the wrong information as due to the way the EA is coded we cannot know whether the script is on the chart. This can be solved by checking whether a script is on the chart or not. This task is not difficult: simply write a check of the last script publication time in the object. This would solve the problem.

However, it is possible to create a much better solution (at least in some cases), and to be honest, this is almost the ideal solution, and we will use the same concept presented above, only instead of a script we will use a service.

### 3\. Creating a service

This is an extreme solution, but as the script has problem of being terminated with each timeframe change, we have to use another method. But by solving one problem, we create another. Anyway, it is good to know which solutions are possible and how they can be used. But the main thing is to know the limitations that each solution presents and thus to try to find something in the middle, which allows solving the problem in the best possible way.

Programming is such a thing that when we try to solve one problem, we create a new one.

Our goal is to create something similar to the image below:

![](https://c.mql5.com/2/45/003__1.png)

While this may seem like a simple matter, the resources involved are generally very little explored. So, I will try to get into the details to help anyone who wants to learn more about working with these resources.

### 3.1. Access to global variables

This part is so little studied that at first I even thought about creating a dll just to support this function, but after looking into the MQL5 documentation, I found it. The problem is that we need to access or create a common point between the service and the EA. When we used a script, this point was an object, but when we use a service, we cannot do the same. The solution would be to use an external variable, but when I tried to do that, the performance was not as expected. For further details, you can read documentation related to [external variables](https://www.mql5.com/en/docs/basis/variables/externalvariables). It explains what to do.

So, this idea was not good, so I decided to use the dll. However, I still wanted to learn MetaTrader 5 and MQL5, so looking into the terminal, I found what you can see in the image below:

![](https://c.mql5.com/2/45/005__1.png)![](https://c.mql5.com/2/45/04__1.png)![](https://c.mql5.com/2/45/004__1.png)

This is what we were looking for: we added a variable to be able to check how this procedure can be configured. However, we can only use [double](https://www.mql5.com/en/docs/basis/types/double) values. You could think that this is the problem (although this is really a limitation), but this is enough when we want to transmit short messages, which is our case. Actually, the double type is a short 8-character string, so we can transmit values or short messages between programs.

So, the first part of the problem is solved. MetaTrader 5 provides methods for creating a channel without having to create a dll, but now we have another problem: how to access these variables through the program? Is it possible to create global variables inside the program — inside an Expert Advisor, a script, an indicator or a service? Or do should we use only those declared in the terminal?

These questions are very important if we really want to use this solution. If it were not possible to use them through programs, we would have to use dlls. But it is possible. For more information, see [Global Variables of the Terminal](https://www.mql5.com/en/docs/globals).

### 3.2. Using a terminal variable to exchange information

Now that we've considered the basics, let's create something simple so that we can test and understand how the process of using terminal variables will work in practice.

For this purpose, I created the following codes. The first one is the header file:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_GlobalNameChannel   "InnerChannel"
//+------------------------------------------------------------------+
```

Here we simply define the name of the global terminal variable which will be the same for the two processes that will run on the graphical terminal.

Below is the code that represents the service to run.

```
#property service
#property copyright "Daniel Jose"
#property version   "1.00"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
void OnStart()
{
        double count = 0;
        while (!IsStopped())
        {
                if (!GlobalVariableCheck(def_GlobalNameChannel)) GlobalVariableTemp(def_GlobalNameChannel);
                GlobalVariableSet(def_GlobalNameChannel, count);
                count += 1.0;
                Sleep(1000);
        }
}
//+------------------------------------------------------------------+
```

Its operation is simple: it checks whether the variable has already been declared and what the [GlobalVariableCheck](https://www.mql5.com/en/docs/globals/globalvariablecheck) is doing. If the variable does not exist, it will be temporarily created by the [GlobalVariableTemp](https://www.mql5.com/en/docs/globals/globalvariabletemp) function and will then receive a value from the [GlobalVariableSet](https://www.mql5.com/en/docs/globals/globalvariableset) function. In other words, we are testing, creating and writing information, the service is acting as a server, just like a script, only we are not accessing the website yet. First, we should understand how the system works.

The next step is to create a client which is an Expert Advisor in our case:

```
#property copyright "Daniel Jose"
#property description "Testing internal channel\nvia terminal global variable"
#property version "1.03"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
int OnInit()
{
        EventSetTimer(1);

        return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
        EventKillTimer();
}
//+------------------------------------------------------------------+
void OnTick()
{
}
//+------------------------------------------------------------------+
void OnTimer()
{
        double value;
        if (GlobalVariableCheck(def_GlobalNameChannel))
        {
                GlobalVariableGet(def_GlobalNameChannel, value);
                Print(value);
        }
}
//+------------------------------------------------------------------+
```

The code is simple: every second the EA will check whether the variable exists. If it exists, the EA will read the value using [GlobalVariableGet](https://www.mql5.com/en/docs/globals/globalvariableget) and will output this value into the terminal.

Let's see how this process can be implemented. First we run the service. It is done as follows:

![](https://c.mql5.com/2/45/Init_Service_212.gif)

But another scenario is possible, when the service has stopped and we restart it. In this case we will proceed as follows:

![](https://c.mql5.com/2/45/Init_Service___2.gif)

After that, we check the terminal variables and get the following result:

![](https://c.mql5.com/2/45/006__1.png)

Note that the system is obviously working, but now we have to place the EA on the chart, get the values, and thus confirm the connection over the channel. So, after placing the EA on the chart, we get the following result:

![](https://c.mql5.com/2/45/Comunica7wo___01.gif)

That's all, the system works the way we want. We actually have a model that is shown below. It is a typical client-server format, and this is exactly what we want to do. We are trying to implement exactly this format because of the advantages that I mentioned before.

![](https://c.mql5.com/2/45/007__1.png)

Now we only need to add a system to read and get values from the web. Then we will have the final model to test. This part is pretty easy: take the code we've been using since the beginning and add it to the service. To perform the test, we just need to modify the server file to read the value from the website and to publish that value for the client to read. The new service code is as follows.

```
#property service
#property copyright "Daniel Jose"
#property version   "1.00"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
void OnStart()
{
        string szRet;

        while (!IsStopped())
        {
                if (!GlobalVariableCheck(def_GlobalNameChannel)) GlobalVariableTemp(def_GlobalNameChannel);
                szRet = GetDataURL("https://tradingeconomics.com/stocks", 100, "<!doctype html>", 2, "INDU:IND", 172783, 173474, 0x0D);
                GlobalVariableSet(def_GlobalNameChannel, StringToDouble(szRet));
                Sleep(1000);
        }
}
//+------------------------------------------------------------------+
string GetDataURL(const string url, const int timeout, const string szTest, int iTest, const string szFind, int iPos, int iInfo, char cLimit)
{
        string  headers, szInfo = "";
        char    post[], charResultPage[];
        int     counter;

        if (WebRequest("GET", url, NULL, NULL, timeout, post, 0, charResultPage, headers) == -1) return "Bad";
        for (int c0 = 0, c1 = StringLen(szTest); c0 < c1; c0++) if (szTest[c0] != charResultPage[iTest + c0]) return "Failed";
        for (int c0 = 0, c1 = StringLen(szFind); c0 < c1; c0++) if (szFind[c0] != charResultPage[iPos + c0]) return "Error";
        for (counter = 0; charResultPage[counter + iInfo] == 0x20; counter++);
        for (;charResultPage[counter + iInfo] != cLimit; counter++) szInfo += CharToString(charResultPage[counter + iInfo]);

        return szInfo;
}
//+------------------------------------------------------------------+
```

We now have a system that works as shown in the image below:

![](https://c.mql5.com/2/45/003__3.png)

So, it is ready, and we get the following results. Furthermore, changing the timeframe will no longer be a problem.

![](https://c.mql5.com/2/45/Primeira_Captura__1.gif)

### Conclusion

Today we have considered several MetaTrader 5 features that have been little explored. One of them is communication channels. However, we are still not taking full advantage of this feature. But we can go even further — we will do this in the next article. Everything we have seen so far in this series shows us how much we can do in the MetaTrader 5 platform. Just choose a path and keep going until you get the desired results, although it is useful to know the limitations, benefits, and risks associated with each of the possible paths.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10442](https://www.mql5.com/pt/articles/10442)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10442.zip "Download all attachments in the single ZIP archive")

[Script\_e\_EA.zip](https://www.mql5.com/en/articles/download/10442/script_e_ea.zip "Download Script_e_EA.zip")(3.03 KB)

[Serviso\_e\_EA.zip](https://www.mql5.com/en/articles/download/10442/serviso_e_ea.zip "Download Serviso_e_EA.zip")(2.39 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/429218)**
(2)


![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
24 May 2022 at 19:31

Very good article.

It opens up possibilities for exchanging information on markets not fully covered by MT5, such as cryptos.

Or even data from [financial indicators](https://www.mql5.com/en/economic-calendar "Economic calendar - economic indicators and events"), which can influence decision-making in strategies.

![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
25 May 2022 at 18:59

**Guilherme Mendonca financial indicators, which can influence decision-making in strategies.**

And that's not all... in the next article coming out next week, things get a lot more personal .... 😁👍

![Neural networks made easy (Part 16): Practical use of clustering](https://c.mql5.com/2/48/Neural_networks_made_easy_016.png)[Neural networks made easy (Part 16): Practical use of clustering](https://www.mql5.com/en/articles/10943)

In the previous article, we have created a class for data clustering. In this article, I want to share variants of the possible application of obtained results in solving practical trading tasks.

![The price movement model and its main provisions (Part 1): The simplest model version and its applications](https://c.mql5.com/2/47/price-motion.png)[The price movement model and its main provisions (Part 1): The simplest model version and its applications](https://www.mql5.com/en/articles/10955)

The article provides the foundations of a mathematically rigorous price movement and market functioning theory. Up to the present, we have not had any mathematically rigorous price movement theory. Instead, we have had to deal with experience-based assumptions stating that the price moves in a certain way after a certain pattern. Of course, these assumptions have been supported neither by statistics, nor by theory.

![Automated grid trading using limit orders on Moscow Exchange (MOEX)](https://c.mql5.com/2/47/moex-trading.png)[Automated grid trading using limit orders on Moscow Exchange (MOEX)](https://www.mql5.com/en/articles/10672)

The article considers the development of an MQL5 Expert Advisor (EA) for MetaTrader 5 aimed at working on MOEX. The EA is to follow a grid strategy while trading on MOEX using MetaTrader 5 terminal. The EA involves closing positions by stop loss and take profit, as well as removing pending orders in case of certain market conditions.

![Neural networks made easy (Part 15): Data clustering using MQL5](https://c.mql5.com/2/48/Neural_networks_made_easy_015.png)[Neural networks made easy (Part 15): Data clustering using MQL5](https://www.mql5.com/en/articles/10947)

We continue to consider the clustering method. In this article, we will create a new CKmeans class to implement one of the most common k-means clustering methods. During tests, the model managed to identify about 500 patterns.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/10442&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068187806221661850)

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