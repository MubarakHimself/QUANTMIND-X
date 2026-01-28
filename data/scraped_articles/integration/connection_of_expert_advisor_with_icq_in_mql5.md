---
title: Connection of Expert Advisor with ICQ in MQL5
url: https://www.mql5.com/en/articles/64
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:27:29.310268
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/64&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068257328857282384)

MetaTrader 5 / Integration


### Introduction

ICQ is a centralized service of instant exchange of text messages, with an offline mode, which uses the OSCAR protocol. For a trader, ICQ can serve as a terminal, which displays timely information, as well as a control panel. This article will demonstrate an example of how to implement an ICQ client, with a minimum set of functions, within an Expert Advisor.

The draft of the IcqMod project, containing an open original code, was used and processed as a basis for the article.  Protocol of exchange with the ICQ server is implemented in the DLL module **icq\_mql5.dl** **l**. It is written in C++ and makes uses the only Windows library winsock2. A compiled modules and a source code for Visual Studio 2005 are attached to this article.

Distinguishing features and limitations of the implementation of this client:

- the maximum number of simultaneously working clients is theoretically unlimited.
- the maximum size of an incoming messages - 150 characters. The reception of longer messages is not supported.
- Unicode support.
- supports only a direct connection. Connection made through a proxy server (HTTP / SOCK4 / SOCK5) is not supported.
- offline messages do not get processed.

**Description of the library functions**

Descriptions of constants and functions of the dll module are located in the executable file **icq\_mql5.mqh**.

The function ICQConnect is used to connect to the server:

```
uint  ICQConnect (ICQ_CLIENT & cl,  // Variable for storing data about the connection
string  host,  // Server name, such as login.icq.com
ushort  port,  // Server port, eg 5190
string  login, // Account Number (UIN)
string  pass   // Account password for)
```

Description of the return value through ICQConnect:

| Constant's Name | Value | Description |
| --- | --- | --- |
| ICQ\_CONNECT\_STATUS\_OK | 0xFFFFFFFF | Connection established |
| ICQ\_CONNECT\_STATUS\_RECV\_ERROR | 0xFFFFFFFE | Reading data error |
| ICQ\_CONNECT\_STATUS\_SEND\_ERR | 0xFFFFFFFD | Sending data error |
| ICQ\_CONNECT\_STATUS\_CONNECT\_ERROR | 0xFFFFFFFC | server connection error |
| ICQ\_CONNECT\_STATUS\_AUTH\_ERROR | 0xFFFFFFFB | Authorization error: incorrect password or exceeded the limit of connections |

Structure for storing data about the connection:

```
 struct  ICQ_CLIENT (
uchar  status;     // connection status code
ushort  sequence;  // sequence meter
uint  sock;        // socket number  )
```

In practice, in order to perform an analysis of the connection status in this structure, we use the variable status, which can assume the following values:

| Constant's Name | Value | Description |
| --- | --- | --- |
| ICQ\_CLIENT\_STATUS\_CONNECTED | 0x01 | A connection to the server is established |
| ICQ\_CLIENT\_STATUS\_DISCONNECTED | 0x02 | Connection to server failed |

Frequent attempts to connect to the server can lead to a temporary block of access to your account. Thus it is necessary to wait out a time interval between connection attempts to the server.

Recommended timeout is 20-30 seconds.

The function ICQClose serves to terminate the connection to the server:

```
 void  ICQClose (
ICQ_CLIENT & cl  // Variable for storing connection data)
```

The ICQSendMsg function is used to send text messages:

```
 uint  ICQSendMsg (
ICQ_CLIENT & cl,  // Variable to store data about the connection.
string  uin,        // Account number of the recipient
string  msg         // Message)
```

The return value equals to 0x01, if the message is sent successful, and equals 0x00 if there was a sending error.

The ICQReadMsg function checks for incoming messages:

```
 uint  ICQReadMsg (
ICQ_CLIENT & cl,  // Variable for storing connection data
string  & Uin,     // Account number of the sender
string  & Msg,     // Message
uint  & Len        // Number of received symbols in the message)
```

The return value equals to 0x01 if there is an incoming message and equals to 0x00 if there is no message.

### COscarClient class

For the convenience of working with ICQ in an object-oriented environment of MQL5, the COscarClient class was developed. Aside from the basic functions described above, it contains a mechanism, which, after a specific interval of time, automatically reconnects to the server (when autocon = true). Description of the class is included in the attached file **icq\_mql5.mqh** and is given below:

```
//+------------------------------------------------------------------+
class COscarClient
//+------------------------------------------------------------------+
{
private:
  ICQ_CLIENT client;        // storing connection data
          uint connect;     // flag of status connection
      datetime timesave;     // the time of last connection to the server
      datetime time_in;      // the time of last reading of messages

public:
      string uin;            // buffer for the storage of the uin of the sender for a received message
      string msg;            // buffer for the storage of text for a received message
        uint len;            // the number of symbols in the received message

      string login;          // number of the sender's account (UIN)
      string password;       // password for UIN
      string server;         // name of the server
        uint port;           // network port
        uint timeout;        // timeout tasks (in seconds) between attempts to reconnect to the server
        bool autocon;        // automatic connection resume

           COscarClient();   // constructor for initialization of variable classes
      bool Connect(void);     // establishment of a connection with a server
      void Disconnect(void);  // breaking a connection with a server
      bool SendMessage(string  UIN, string  msg); // sending a message
      bool ReadMessage(string &UIN, string &msg, uint &len); // receiving a message
};
```

**Expert Advisor on the COscarClient bases**

The minimum Expert Advisor code needed to work with ICQ using the class COscarClient is located in the **icq\_demo.mq5** file and below:

```
#include <icq_mql5.mqh>

COscarClient client;

//+------------------------------------------------------------------+
int OnInit()
//+------------------------------------------------------------------+
  {
   printf("Start ICQ Client");

   client.login      = "641848065";     //<- login
   client.password   = "password";      //<- password
   client.server     = "login.icq.com";
   client.port       = 5190;
   client.Connect();

   return(0);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
//+------------------------------------------------------------------+
  {
   client.Disconnect();
   printf("Stop ICQ Client");
  }
//+------------------------------------------------------------------+
void OnTick()
//+------------------------------------------------------------------+
  {
   string text;
   static datetime time_out;
   MqlTick last_tick;

   // reading the messages
   while(client.ReadMessage(client.uin,client.msg,client.len))
     printf("Receive: %s, %s, %u", client.uin, client.msg, client.len);

   // transmission of quotes every 30 seconds
   if((TimeCurrent()-time_out)>=30)
     {
      time_out = TimeCurrent();
      SymbolInfoTick(Symbol(), last_tick);

      text = Symbol()+" BID:"+DoubleToString(last_tick.bid, Digits())+
                  " ASK:"+DoubleToString(last_tick.ask, Digits());

      if (client.SendMessage("266690424",        //<- number of the recipient
                                        text)) //<- message text
         printf("Send: " + text);
     }
  }
//+------------------------------------------------------------------+
```

Figure 1. serves as a demonstration of the work of the Expert Advisor, which allows for the exchange of text messages with the ICQ client.

> > > ![Figure 1. Text messaging between MetaTrader5 and ICQ2Go](https://c.mql5.com/2/1/figure1__3.png)
> > >
> > > Figure 1. Text messaging between MetaTrader 5 and ICQ2Go

### Capacity building

Let's complicate the task by bringing it closer to practical application. For example, we need to manage the work of our Expert Advisor, and obtain the necessary information remotely, using a mobile phone or another PC connected to the Internet. To do this, we describe a set of commands for controlling the future Expert Advisor. Also, let's complement the advisor with a parsing function to decode the incoming commands.

The format, common to all commands, will be as following:

**\[? \|!\] \[command\] \[parameter\] \[value\]** ,

where? - A symbol of command reading;  ! \- A symbol of a operation writing.

A list of commands given in the table below:

|  |  |  |
| --- | --- | --- |
| help | reading | Display of reference of the syntax and the list of commands |
| info | reading | Display of data of the account summary |
| symb | reading | Market price for given currency pair |
| ords | reading/writing | Managing of open orders |
| param | reading/writing | managing Expert Advisor's parameters |
| close | record | Termination of the Expert Advisor's work and the closure of the terminal |
| shdwn | record | Shutdown of PC |

The Expert Advisor, which implements the processing of this set of commands, is located in the file **icq\_power.mq5** .

Figure 2 shows a clear demonstration of the work of the Expert Advisor. Commands are received from the CCP with an installed ICQ client (Figure 2a), as well as through the WAP server http://wap.ebuddy.com which implements the work with ICQ (Figure 2b). The second option is preferable for those who do not wish to deal with searching for, installing, and configuring software for ICQ on their mobile phone.

![](https://c.mql5.com/2/1/pict-2a_en.png)![](https://c.mql5.com/2/1/pict-2b_en.png)

Figure 2. Working with an advisor via the ICQ client for Pocket PC (Figure 2a), as well as through the wap.ebuddy.com wap site (Figure 2b).

**Visual ICQ component**

This section will briefly consider an example of a script **icq\_visual.mq5** which implements a component, the visual appearance of which is shown in Figure 3.

![Figure 3. Visual ICQ component](https://c.mql5.com/2/1/figure3_.png)

Figure 3. Visual ICQ component

The form of the component resembles a Windows window and is built from arrays of control elements, such as buttons, text boxes, and text labels.

For convenience, an integrated control element for storing a list of accounts and contacts is implemented in the form. Values are selected from the list through the use of appropriate navigation buttons.

To create a window in the style of ICQ 6.5, we can replace the buttons with image tags. Figure 4 shows the visual appearance of the component, implemented in the **icq\_visual\_skin.mq5** script. For those who wish to create their own component design, it is sufficient enough to develop and replace the skin.bmp file, which is responsible for the appearance of the window.

![Figure 4. Color design of the visual component of ICQ](https://c.mql5.com/2/1/figure4_.png)

Figure 4. Color
design of the visual ICQ component

### Conclusion

This article demonstrates one of the easiest ways to implement an ICQ client for MetaTrader 5 by using the means of an embedded programming language.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/64](https://www.mql5.com/ru/articles/64)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/64.zip "Download all attachments in the single ZIP archive")

[icq\_mql5\_doc.zip](https://www.mql5.com/en/articles/download/64/icq_mql5_doc.zip "Download icq_mql5_doc.zip")(406.44 KB)

[icq\_mql5.zip](https://www.mql5.com/en/articles/download/64/icq_mql5.zip "Download icq_mql5.zip")(98.31 KB)

[dll\_source\_icq\_mql5.zip](https://www.mql5.com/en/articles/download/64/dll_source_icq_mql5.zip "Download dll_source_icq_mql5.zip")(87.02 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to create bots for Telegram in MQL5](https://www.mql5.com/en/articles/2355)
- [Using the TesterWithdrawal() Function for Modeling the Withdrawals of Profit](https://www.mql5.com/en/articles/131)
- [Guide to writing a DLL for MQL5 in Delphi](https://www.mql5.com/en/articles/96)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1127)**
(21)


![Mikhail Vdovin](https://c.mql5.com/avatar/2013/9/5230EA4E-019A.JPG)

**[Mikhail Vdovin](https://www.mql5.com/en/users/micle)**
\|
30 Apr 2014 at 13:00

Andei, are there any plans for a similar library for skype?


![Artem Temnikov](https://c.mql5.com/avatar/2013/12/52B0ADCE-BEAB.jpg)

**[Artem Temnikov](https://www.mql5.com/en/users/fleder)**
\|
30 Apr 2014 at 13:03

**micle:**

Andrei, are there any plans for a similar library for Skype?

Isn't [this one](https://www.mql5.com/ru/code/1537) suitable?


![Mikhail Vdovin](https://c.mql5.com/avatar/2013/9/5230EA4E-019A.JPG)

**[Mikhail Vdovin](https://www.mql5.com/en/users/micle)**
\|
30 Apr 2014 at 13:07

**Fleder:**

Doesn't [this one](https://www.mql5.com/ru/code/1537) fit?

Oh! Of course it does. But I want to be compatible with x64. Probably need to fix the function declaration...


![---](https://c.mql5.com/avatar/avatar_na2.png)

**[\-\-\-](https://www.mql5.com/en/users/sergeev)**
\|
30 Apr 2014 at 13:30

**micle:**

Andrei, are there any plans for a similar library for Skype?

unfortunately the shop has been closed.

Skype doesn't support [sending messages](https://www.mql5.com/en/articles/8586 "Article: Use MQL5.community channels and group chats") with dll anymore

![Mikhail Vdovin](https://c.mql5.com/avatar/2013/9/5230EA4E-019A.JPG)

**[Mikhail Vdovin](https://www.mql5.com/en/users/micle)**
\|
30 Apr 2014 at 13:42

**sergeev:**

Unfortunately, the shop has been shut down.

Skype no longer supports dll messaging.

hmmm. littlesoft. be damned.


![An Example of a Trading Strategy Based on Timezone Differences on Different Continents](https://c.mql5.com/2/0/5g6ovfni.png)[An Example of a Trading Strategy Based on Timezone Differences on Different Continents](https://www.mql5.com/en/articles/59)

Surfing the Internet, it is easy to find many strategies, which will give you a number of various recommendations. Let’s take an insider’s approach and look into the process of strategy creation, based on the differences in timezones on different continents.

![The Magic of Filtration](https://c.mql5.com/2/17/893_81.jpg)[The Magic of Filtration](https://www.mql5.com/en/articles/1577)

Most of the automated trading systems developers use some form of trading signals filtration. In this article, we explore the creation and implementation of bandpass and discrete filters for Expert Advisors, to improve the characteristics of the automated trading system.

![Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners](https://c.mql5.com/2/0/create_EA_step_by_step_MQL5.png)[Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners](https://www.mql5.com/en/articles/100)

The Expert Advisors programming in MQL5 is simple, and you can learn it easy. In this step by step guide, you will see the basic steps required in writing a simple Expert Advisor based on a developed trading strategy. The structure of an Expert Advisor, the use of built-in technical indicators and trading functions, the details of the Debug mode and use of the Strategy Tester are presented.

![New Opportunities with MetaTrader 5](https://c.mql5.com/2/0/new_opportunities_MQL5__1.png)[New Opportunities with MetaTrader 5](https://www.mql5.com/en/articles/84)

MetaTrader 4 gained its popularity with traders from all over the world, and it seemed like nothing more could be wished for. With its high processing speed, stability, wide array of possibilities for writing indicators, Expert Advisors, and informatory-trading systems, and the ability to chose from over a hundred different brokers, - the terminal greatly distinguished itself from the rest. But time doesn’t stand still, and we find ourselves facing a choice of MetaTrade 4 or MetaTrade 5. In this article, we will describe the main differences of the 5th generation terminal from our current favor.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fricnidtoswvmpjqqkqcrrvnifwftaxi&ssn=1769178448002634058&ssn_dr=0&ssn_sr=0&fv_date=1769178448&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F64&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Connection%20of%20Expert%20Advisor%20with%20ICQ%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917844841515651&fz_uniq=5068257328857282384&sv=2552)

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