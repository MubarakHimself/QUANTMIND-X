---
title: Developing a trading Expert Advisor from scratch (Part 17): Accessing data on the web (III)
url: https://www.mql5.com/en/articles/10447
categories: Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:24:18.336613
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/10447&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068184138319591056)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a trading Expert Advisor from scratch (Part 16): Accessing data on the web (II)](https://www.mql5.com/en/articles/10442), we talked about the problems and consequences of data capturing from the web. We also considered how to use it in an Expert Advisor and discussed three possible solutions each having their pros and cons.

In the first solution, which implied data capturing directly via the Expert Advisor, we considered a possible problem, connected with the slow server response. We also mentioned the consequences this can have on a trading system.

In the second solution, we implemented a channel based on the client-server model, in which the EA acted as the client, a script was the server, and an object served as the channel. This model performs well up to the point where you decide to change the timeframe, where it becomes inconvenient. Despite this fact, it is the best system presented, because the use of the client-server model ensures that the EA will not wait for a remote server — it will simply read the data contained in the object, regardless of where this information has come from.

In the third and final solution, we improved the client-server system by using a service. Thus, we started using a MetaTrader 5 platform resource, which is quite little studied: the global variables of the terminal. This solution fixed the issue with the timeframe change, which was the biggest drawback of the model utilizing scripts. However, we have a new problem: the system of terminal's global variables only allows the use of the double type. Many do not know how to avoid this, and therefore they pass various information, such as a piece of text, through the channel provided by MetaTrader 5.

In this article, we will discuss how to get around this limitation. But do not expect miracles, because it will take a lot of work to have the system work the way you want it.

This time we will proceed to developing an alternative system.

### 1\. Planning

As we know, we can only use [double](https://www.mql5.com/en/docs/basis/types/double) type variables in the channel system provided by MetaTrader 5. This type consists of 8 bytes. You might think that it is not very useful information. But let's figure out the following moment:

Computer systems work with bytes, although many people have forgotten about this concept. It is very important to understand this system. Each byte consists of 8 bits. 1 bit is the smallest possible number in a computing system. The smallest and simplest type present in the language is the Boolean type, which consists of one single bit. This is the simplest of the basics.

So, any information, no matter how complex it may be, will be contained within 1 byte. Again, no matter how complicated information is, it will always be within one byte, which is made up of 8 bits. When we join 2 bytes we get the first compound set in the system. This first set is known as WORD, the second set as DWORD which will be 2 WORD and the third set will be QWORD which will be 2 DWORD. This is the nomenclature used in [assembly](https://en.wikipedia.org/wiki/Assembly_language "https://en.wikipedia.org/wiki/Assembly_language"), which is the mother language of all modern languages, so most systems use the same types. The only difference is in how these types are named.

I hope you have been able to follow the reasoning up to this point. To make things easier for those who are just beginning, take a look at the figures below:

![](https://c.mql5.com/2/45/001.1.png)![](https://c.mql5.com/2/45/001.2.png)

![](https://c.mql5.com/2/45/001.3.png)![](https://c.mql5.com/2/45/001.4.png)

The above images show the main types currently available, they cover from 1 to 64 bits. You might be thinking, "Why do I need this explanation for?". It is important to know this information in order to understand what we will be doing throughout this article, since we are going to manipulate these types in order to be able to pass information with different internal properties.

Each of these types can get different names depending on the language used, in the case of MQL5 they are shown in the following table:

| Name | Number of bytes | Name based on assembly language (images above) |
| --- | --- | --- |
| bool | Uses only 1 bit; a byte can have 8 bool values. | Uses only 1 bit; a byte can have 8 bool values. |
| char | 1 | Byte |
| short | 2 | Word |
| int | 4 | DWord |
| long | 8 | QWord |

This table covers signed integers, for more details in MQL5 see [integer types](https://www.mql5.com/en/docs/basis/types/integer), other names are defined there. Next, [real types](https://www.mql5.com/en/docs/basis/types/double) have some similarities to integer types, but have their own internal formatting and styling. An example of formatting and modeling can be seen at a [double precision number](https://en.wikipedia.org/wiki/Double-precision_floating-point_format "https://en.wikipedia.org/wiki/Double-precision_floating-point_format"), but basically it will match the table below:

| Name | Number of bytes | Name based on assembly language (images above) |
| --- | --- | --- |
| Float | 4 | DWord |
| Doble | 8 | QWord |

One interesting thing to note is that both the floating point and integer models use the same database but with different lengths. Now we've come to the point we're really interested in. If you understand the logic, you can eventually come to the following conclusion, which can be seen in the image below:

![](https://c.mql5.com/2/45/001.5.png)

QWORD is 8 bytes, and thus 'double' allows putting 8 information bytes. For example, you can pass 8 printable characters into a terminal global variable, and you will obtain the result of connection between the service and the EA, as shown below.

![](https://c.mql5.com/2/45/002.gif)

The data is ok, I think the idea itself is understandable. The big detail is that if the message has more than 8 printable characters, then it will have to be fragmented into more parts. But if the information is to be delivered very quickly, that is, in 1 cycle, you will have to use as many global terminal variables as necessary to transmit the messages in a cycle. Then they need to be glued together to restore the original message. But if it can be delivered in packets, we will have to create a form for the server so that service knows that the client, which is in this case the EA, will read the message and will wait for the next block.

This type of problem has multiple solutions. If you want to understand or implement these solutions, you will not need to create everything from scratch — you can use the same modeling as in network communication protocols such as [TCP/IP](https://ru.wikipedia.org/wiki/TCP/IP "https://ru.wikipedia.org/wiki/TCP/IP") or [UDP](https://www.mql5.com/go?link=https://nordvpn.com/blog/tcp-versus-udp/ "https://nordvpn.com/blog/tcp-versus-udp/"), and adapt the idea to the information transfer system using global terminal variables. Once you understand how the protocols work, this task is no longer complicated and becomes a matter of skill and knowledge of the language you are using. This is a very broad topic that deserves separate study for each type of situation and problem.

### 2\. Implementation

Now that we understand the idea we are going to use, we can make an initial implementation to test how the system will transfer information between the service and the EA. But we will only pass printable characters.

### 2.1. Basic model

We will use a system from the previous article and will modify the files, starting with the header file. Its new content is shown in full in the code below:

```
//+------------------------------------------------------------------+
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_GlobalNameChannel   "InnerChannel"
//+------------------------------------------------------------------+
union uDataServer
{
        double  value;
        char    Info[sizeof(double)];
};
//+------------------------------------------------------------------+
```

This header is basic. It contains a declaration of the global terminal value. It also has a new structure, a [union](https://www.mql5.com/en/docs/basis/types/classes#union). The union differs from the [structure](https://www.mql5.com/en/docs/basis/types/classes) in that a structure is a combination of data without interleaving, while the union always uses it, when smaller data is inside the bigger. In the previous case, we have a double value as the basis, which has 8 bytes inside it. But pay attention that I use a system to capture the length [sizeof](https://www.mql5.com/en/docs/basis/operations/other#sizeof), so if we have a larger double in the future, which is unlikely, this code will automatically adapt to it.

As a result, we get the following:

![](https://c.mql5.com/2/45/002__2.png)

Note that this is similar to the picture above, but that's what the union does.

The next code to modify is the EA that corresponds to the client. The full code can be seen below:

```
#property copyright "Daniel Jose"
#property description "Testing internal channel\nvia terminal global variable"
#property version "1.04"
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
        uDataServer loc;

        if (GlobalVariableCheck(def_GlobalNameChannel))
        {
                GlobalVariableGet(def_GlobalNameChannel, loc.value);
                Print(CharArrayToString(loc.Info, 0, sizeof(uDataServer)));
        }
}
//+------------------------------------------------------------------+
```

Note that here we use function [CharArrayToString](https://www.mql5.com/en/docs/convert/chararraytostring) to convert a uchar array into a string. However, pay attention that we still get a double value because it is the only one which can be received from a global variable of the terminal. In contrast to that, the sting in MQL5 follows the principle of C/C++ and thus we cannot use any character, but we can only create our own. But that's another story. Here, we will not go into detail regarding how to do it: you might want to use modeling data compression to surpass the 8 byte limit.

But we still need a program that acts as a server. In our case the server is a service. Below is the code to test the system:

```
//+------------------------------------------------------------------+
#property service
#property copyright "Daniel Jose"
#property version   "1.03"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
void OnStart()
{
        uDataServer loc;
        char car = 33;

        while (!IsStopped())
        {
                if (!GlobalVariableCheck(def_GlobalNameChannel)) GlobalVariableTemp(def_GlobalNameChannel);
                for (char c0 = 0; c0 < sizeof(uDataServer); c0++)
                {
                        loc.Info[c0] = car;
                        car = (car >= 127 ? 33 : car + 1);
                }
                GlobalVariableSet(def_GlobalNameChannel, loc.value);
                Sleep(1000);
        }
}
//+------------------------------------------------------------------+
```

This is something simple but extremely effective and functional.

By launching the program in the platform, we will obtain the following result

![](https://c.mql5.com/2/45/002__1.gif)

It may seem silly and pointless, but with a little creativity someone can make this system useful enough and make it do the things that others can't even imagine.

To demonstrate this, let's modify the system and show a very simple thing, just to arouse curiosity and interest. Think about which very exotic functionality can be found for such a communication system.

### 2.2. Exchange stickers

Exchanging stickers is the exchange of information between the client and the server during which the server knows what information the client wants to receive, and so the server can start producing or looking for that information.

The concept is quite simple to understand. But its implementation can be quite a challenge, especially when it comes to data modeling where we only have 8 bytes available while the channel is used for transferring data.

### 2.2.1. Client-server communication test

Take a look at the code of the service which is shown in full below:

```
#property service
#property copyright "Daniel Jose"
#property version   "1.03"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
void OnStart()
{
        uDataServer loc, loc1, loc2;
        char car = 33;

        while (!IsStopped())
        {
                if (!GlobalVariableCheck(def_GlobalValueInChannel))
                {
                        GlobalVariableTemp(def_GlobalValueInChannel);
                        GlobalVariableTemp(def_GlobalMaskInfo);
                        GlobalVariableTemp(def_GlobalPositionInfos);
                }
                for (char c0 = 0; c0 < sizeof(uDataServer); c0++)
                {
                        loc.Info[c0] = car;
                        car = (car >= 127 ? 33 : car + 1);
                }
                GlobalVariableSet(def_GlobalValueInChannel, loc.value);
                GlobalVariableGet(def_GlobalMaskInfo, loc1.value);
                GlobalVariableGet(def_GlobalPositionInfos, loc2.value);
                Print(CharArrayToString(loc1.Info, 0, sizeof(uDataServer)), "   ",loc2.Position[0], "    ", loc2.Position[1]);
                Sleep(1000);
        }
}
//+------------------------------------------------------------------+
```

Pay attention to some especially interesting part in the new service code (which acts as a server). Now we have three variables instead of one. They work to create a channel large enough to enable communication between the client (which is an EA in or case) and the server (our service). Pay attention to the following line:

```
Print(CharArrayToString(loc1.Info, 0, sizeof(uDataServer)), "   ",loc2.Position[0], "    ", loc2.Position[1]);
```

These are the data published by the client. Note that we are using 2 variables to pass 3 different pieces of information. But how is that possible? To understand this, we need to see the header code, which is shown in full below.

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_GlobalValueInChannel        "Inner Channel"
#define def_GlobalMaskInfo                      "Mask Info"
#define def_GlobalPositionInfos         "Positions Infos"
//+------------------------------------------------------------------+
union uDataServer
{
        double  value;
        uint    Position[2];
        char    Info[sizeof(double)];
};
//+------------------------------------------------------------------+
```

You may think that each variable within this union is isolated from the other. I advise you to look at the beginning of this article, because although we have variables with different names, here they are treated as one variable, which is 8 bytes wide. To make it clearer, have a look at the image below, which accurately reflects what is happening:

![](https://c.mql5.com/2/45/003__4.png)

This scheme shows what's inside uDataServer.

If it looks too complicated to you, you should try experimenting with unions to understand how they actually work, as they are very useful in programming.

But let's get back to the system. The next thing to do is to create the code for the client—the EA. It can be seen in full below.

```
#property copyright "Daniel Jose"
#property description "Testing internal channel\nvia terminal global variable"
#property version "1.04"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
enum eWhat {DOW_JONES, SP500};
input eWhat     user01 = DOW_JONES;     //Search
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
        uDataServer loc;

        SetFind();
        if (GlobalVariableCheck(def_GlobalValueInChannel))
        {
                GlobalVariableGet(def_GlobalMaskInfo, loc.value);
                Print(CharArrayToString(loc.Info, 0, sizeof(uDataServer)), "  ", GlobalVariableGet(def_GlobalValueInChannel));
        }
}
//+------------------------------------------------------------------+
inline void SetFind(void)
{
        static int b = -1;
        uDataServer loc1, loc2;

        if ((GlobalVariableCheck(def_GlobalValueInChannel)) && (b != user01))
        {
                b = user01;
                switch (user01)
                {
                        case DOW_JONES  :
                                StringToCharArray("INDU:IND", loc1.Info, 0, sizeof(uDataServer));
                                loc2.Position[0] = 172783;
                                loc2.Position[1] = 173474;
                                break;
                        case SP500              :
                                StringToCharArray("SPX:IND", loc1.Info, 0, sizeof(uDataServer));
                                loc2.Position[0] = 175484;
                                loc2.Position[1] = 176156;
                                break;
                }
                GlobalVariableSet(def_GlobalMaskInfo, loc1.value);
                GlobalVariableSet(def_GlobalPositionInfos, loc2.value);
        }
};
//+------------------------------------------------------------------+
```

Note that in this EA we transmit and receive information, that is, we can control how the service should work. In one variable, we pass a small string that will indicate what the service should look for, and in the other one we pass 2 address points.

In response, the service will return information. But to understand this first point, look at the result in the video below:

### 3.1.2.2 - Creating a practical version

Now that we've seen how the system works, we can make something really functional. This time we will collect information from the web server. This requires a number of modifications that ensure a perfect understanding of what is happening. Sometimes we might imagine that we are receiving updated data although in fact we are using garbage in analysis. You have to be very careful during the programming phase not to expose yourself to this risk. What you can do is add as many tests as you can and try to have the system report any weird activity it might detect while running.

Remember: _**Information will only be useful to you if you trust it.**_

First, let's edit the header file to make it look like this:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#define def_GlobalValueInChannel        "Inner Channel"
#define def_GlobalMaskInfo              "Mask Info"
#define def_GlobalPositionInfos         "Positions Infos"
//+------------------------------------------------------------------+
#define def_MSG_FailedConnection        "BAD"
#define def_MSG_FailedReturn            "FAILED"
#define def_MSG_FailedMask              "ERROR"
#define def_MSG_FinishServer            "FINISH"
//+------------------------------------------------------------------+
union uDataServer
{
        double  value;
        uint            Position[2];
        char            Info[sizeof(double)];
};
//+------------------------------------------------------------------+
```

The highlighted parts represent the codes we will use to report some strange activity. You should use a maximum of 8 characters, but you also need to create a sequence that is unlikely to be created by the market, which is not an easy thing to do. Even if everything seems fine, there is always a risk that the market will generate a value corresponding to the sequence that you will use as server error messages. Anyway, you can also use a global variable of the terminal for this purpose, which will increase the number of possible combinations, thus allowing you to create more things. But I wanted to use as few global terminal variables as possible. However, in a real case, I would think about it and possibly use a variable just for indication and reporting of errors or abnormal activity.

The next part is the full code of the EA.

```
#property copyright "Daniel Jose"
#property description "Testing internal channel\nvia terminal global variable"
#property version "1.04"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
enum eWhat {DOW_JONES, SP500};
input eWhat     user01 = DOW_JONES;             //Search
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
        ClientServer();
}
//+------------------------------------------------------------------+
inline void ClientServer(void)
{
        uDataServer loc1, loc2;
        string          sz0;

        SetFind();
        if (GlobalVariableCheck(def_GlobalValueInChannel))
        {
                GlobalVariableGet(def_GlobalMaskInfo, loc1.value);
                loc2.value = GlobalVariableGet(def_GlobalValueInChannel);
                sz0 = CharArrayToString(loc2.Info, 0, sizeof(uDataServer));
                if (sz0 == def_MSG_FailedConnection) Print("Failed in connection."); else
                if (sz0 == def_MSG_FailedReturn) Print("Error in Server Web."); else
                if (sz0 == def_MSG_FailedMask) Print("Bad Mask or position."); else
                if (sz0 == def_MSG_FinishServer) Print("Service Stop."); else
                Print(CharArrayToString(loc1.Info, 0, sizeof(uDataServer)), "  ", loc2.value);
        }
}
//+------------------------------------------------------------------+
inline void SetFind(void)
{
        static int b = -1;
        uDataServer loc1, loc2;

        if ((GlobalVariableCheck(def_GlobalValueInChannel)) && (b != user01))
        {
                b = user01;
                switch (user01)
                {
                        case DOW_JONES  :
                                StringToCharArray("INDU:IND", loc1.Info, 0, sizeof(uDataServer));
                                loc2.Position[0] = 172783;
                                loc2.Position[1] = 173474;
                                break;
                        case SP500              :
                                StringToCharArray("SPX:IND", loc1.Info, 0, sizeof(uDataServer));
                                loc2.Position[0] = 175487;
                                loc2.Position[1] = 176159;
                                break;
                }
                GlobalVariableSet(def_GlobalMaskInfo, loc1.value);
                GlobalVariableSet(def_GlobalPositionInfos, loc2.value);
        }
};
//+------------------------------------------------------------------+
```

The highlighted lines are very important and should be well thought out as we really want to know what is going on. As you can see, we can tell the user something more details than offered by sequences created in the header file, so that it becomes easier to program and maintain the solution. The rest of the code has not changed much. Look at the service code below.

```
#property service
#property copyright "Daniel Jose"
#property version   "1.03"
//+------------------------------------------------------------------+
#include <Inner Channel.mqh>
//+------------------------------------------------------------------+
void OnStart()
{
        uDataServer loc1, loc2;

        while (!IsStopped())
        {
                if (!GlobalVariableCheck(def_GlobalValueInChannel))
                {
                        GlobalVariableTemp(def_GlobalValueInChannel);
                        GlobalVariableTemp(def_GlobalMaskInfo);
                        GlobalVariableTemp(def_GlobalPositionInfos);
                }
                GlobalVariableGet(def_GlobalMaskInfo, loc1.value);
                GlobalVariableGet(def_GlobalPositionInfos, loc2.value);
                if (!_StopFlag)
                {
                        GlobalVariableSet(def_GlobalValueInChannel, GetDataURL(
                                                                                "https://tradingeconomics.com/stocks",
                                                                                100,
                                                                                "<!doctype html>",
                                                                                2,
                                                                                CharArrayToString(loc1.Info, 0, sizeof(uDataServer)),
                                                                                loc2.Position[0],
                                                                                loc2.Position[1],
                                                                                0x0D
                                                                               )
                                        );
                        Sleep(1000);
                }
        }
        GlobalVariableSet(def_GlobalValueInChannel, Codification(def_MSG_FinishServer));
}
//+------------------------------------------------------------------+
double GetDataURL(const string url, const int timeout, const string szTest, int iTest, const string szFind, int iPos, int iInfo, char cLimit)
{
        string          headers, szInfo = "";
        char                    post[], charResultPage[];
        int                     counter;

        if (WebRequest("GET", url, NULL, NULL, timeout, post, 0, charResultPage, headers) == -1) return Codification(def_MSG_FailedConnection);
        for (int c0 = 0, c1 = StringLen(szTest); (c0 < c1) && (!_StopFlag); c0++) if (szTest[c0] != charResultPage[iTest + c0]) return Codification(def_MSG_FailedReturn);
        for (int c0 = 0, c1 = StringLen(szFind); (c0 < c1) && (!_StopFlag); c0++) if (szFind[c0] != charResultPage[iPos + c0]) return Codification(def_MSG_FailedMask);
        if (_StopFlag) return Codification(def_MSG_FinishServer);
        for (counter = 0; charResultPage[counter + iInfo] == 0x20; counter++);
        for (;charResultPage[counter + iInfo] != cLimit; counter++) szInfo += CharToString(charResultPage[counter + iInfo]);

        return StringToDouble(szInfo);
}
//+------------------------------------------------------------------+
inline double Codification(const string arg)
{
        uDataServer loc;
        StringToCharArray(arg, loc.Info, 0, sizeof(uDataServer));

        return loc.value;
}
//+------------------------------------------------------------------+
```

The highlighted line is also important — the service will alert that it is no longer running.

So, when you execute this system, you will get the following result:

### Conclusion

I hope I have explained the idea related to researching, searching and using web data on the MetaTrader 5 platform. I understand that this might not be very clear at first, especially for those who do not have very extensive knowledge in programming, but over time, through discipline and learning, you will eventually master most of this material. Here I tried to share at least a little of what I know.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10447](https://www.mql5.com/pt/articles/10447)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10447.zip "Download all attachments in the single ZIP archive")

[Servi0o\_-\_EA.zip](https://www.mql5.com/en/articles/download/10447/servi0o_-_ea.zip "Download Servi0o_-_EA.zip")(10.71 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/429559)**
(2)


![Vadim Zotov](https://c.mql5.com/avatar/2015/4/55241719-F38E.JPG)

**[Vadim Zotov](https://www.mql5.com/en/users/dosent)**
\|
8 Jul 2022 at 19:17

This article shows that MetaQuotes needs to extend the possibilities of [global variables of](https://www.mql5.com/en/docs/basis/variables/global "MQL5 Documentation: Global Variables") the terminal. Restriction of double type creates many problems, which can be avoided if you have an opportunity to use other types. The most important thing is to be able to use string type. Then the tambourine dances brilliantly demonstrated by the author of the article will become unnecessary. And all users will be very grateful to MQL developers for making their lives much easier. I hope that the author of the article will also be happy that he made it happen.


![yaxi wu](https://c.mql5.com/avatar/avatar_na2.png)

**[yaxi wu](https://www.mql5.com/en/users/wuyaxi)**
\|
23 Aug 2022 at 17:04

The several system programs are basically not passable by file compilation!


![Learn how to design a trading system by Chaikin Oscillator](https://c.mql5.com/2/48/why-and-how__1.png)[Learn how to design a trading system by Chaikin Oscillator](https://www.mql5.com/en/articles/11242)

Welcome to our new article from our series about learning how to design a trading system by the most popular technical indicator. Through this new article, we will learn how to design a trading system by the Chaikin Oscillator indicator.

![Data Science and Machine Learning (Part 06): Gradient Descent](https://c.mql5.com/2/47/data_science_articles_series__1.png)[Data Science and Machine Learning (Part 06): Gradient Descent](https://www.mql5.com/en/articles/11200)

The gradient descent plays a significant role in training neural networks and many machine learning algorithms. It is a quick and intelligent algorithm despite its impressive work it is still misunderstood by a lot of data scientists let's see what it is all about.

![Developing a trading Expert Advisor from scratch (Part 18): New order system (I)](https://c.mql5.com/2/47/development__1.png)[Developing a trading Expert Advisor from scratch (Part 18): New order system (I)](https://www.mql5.com/en/articles/10462)

This is the first part of the new order system. Since we started documenting this EA in our articles, it has undergone various changes and improvements while maintaining the same on-chart order system model.

![Automated grid trading using limit orders on Moscow Exchange (MOEX)](https://c.mql5.com/2/47/moex-trading.png)[Automated grid trading using limit orders on Moscow Exchange (MOEX)](https://www.mql5.com/en/articles/10672)

The article considers the development of an MQL5 Expert Advisor (EA) for MetaTrader 5 aimed at working on MOEX. The EA is to follow a grid strategy while trading on MOEX using MetaTrader 5 terminal. The EA involves closing positions by stop loss and take profit, as well as removing pending orders in case of certain market conditions.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nyjzwvatjjrnkvpslpwlznfbluvnbomh&ssn=1769178256438233772&ssn_dr=1&ssn_sr=0&fv_date=1769178256&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10447&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20(Part%2017)%3A%20Accessing%20data%20on%20the%20web%20(III)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917825703164018&fz_uniq=5068184138319591056&sv=2552)

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