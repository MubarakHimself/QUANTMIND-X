---
title: Using WinInet in MQL5.  Part 2:  POST Requests and Files
url: https://www.mql5.com/en/articles/276
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:21:08.927610
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/276&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071767506844069395)

MetaTrader 5 / Examples


### Introduction

In the previous lesson " [Using WinInet.dll for Data Exchange between Terminals via the Internet](https://www.mql5.com/en/articles/73)", we've learned how to work with the library, open web pages, send and receive information using GET requests.

In this lesson, we're going to learn how to:

- create and send simple POST requests to a server;
- send files to a server using the representation method **multipart/form-data**;
- work with Cookies and read information from websites using your login.

As previously, I strongly recommend setting up a local proxy server [Charles](https://www.mql5.com/go?link=https://www.charlesproxy.com/ "http://www.charlesproxy.com/"); it will be necessary for your studying and further experiments.

### POST Requests

To send information, we'll need those wininet.dll functions and the created class **CMqlNet** that were described in details in the previous article.

Due to a big number of fields in the **CMqlNet::Request** methods, we had to create a separate structure **tagRequest** that contains all the required fields for a request.

```
//------------------------------------------------------------------ struct tagRequest
struct tagRequest
{
  string stVerb;   // method of the request GET/POST/…
  string stObject; // path to an instance of request, for example "/index.htm" или "/get.php?a=1"
  string stHead;   // request header
  string stData;   // addition string of data
  bool fromFile;   // if =true, then stData designates the name of a data file
  string stOut;    // string for receiving an answer
  bool toFile;     // if =true, then stOut designates the name of a file for receiving an answer

  void Init(string aVerb, string aObject, string aHead,
            string aData, bool from, string aOut, bool to); // function of initialization of all fields
};
//------------------------------------------------------------------ Init
void tagRequest::Init(string aVerb, string aObject, string aHead,
                      string aData, bool from, string aOut, bool to)
{
  stVerb=aVerb;     // method of the request GET/POST/…
  stObject=aObject; // path to the page "/get.php?a=1" or "/index.htm"
  stHead=aHead;     // request header, for example "Content-Type: application/x-www-form-urlencoded"
  stData=aData;     // addition string of data
  fromFile=from;    // if =true, the stData designates the name of a data file
  stOut=aOut;       // field for receiving an answer
  toFile=to;        // if =true, then stOut designates the name of a file for receiving an answer
}
```

In addition, we need to replace the header of the **CMqlNet::Request** method with a shorter one:

```
//+------------------------------------------------------------------+
bool MqlNet::Request(tagRequest &req)
  {
   if(!TerminalInfoInteger(TERMINAL_DLLS_ALLOWED))
     {
      Print("-DLL not allowed"); return(false);
     }
//--- checking whether DLLs are allowed in the terminal
   if(!MQL5InfoInteger(MQL5_DLLS_ALLOWED))
     {
      Print("-DLL not allowed");
      return(false);
     }
//--- checking whether DLLs are allowed in the terminal
   if(req.toFile && req.stOut=="")
     {
      Print("-File not specified ");
      return(false);
     }
   uchar data[];
    int hRequest,hSend;
   string Vers="HTTP/1.1";
    string nill="";

//--- read file to array
   if(req.fromFile)
     {
      if(FileToArray(req.stData,data)<0)
        {
         Print("-Err reading file "+req.stData);
         return(false);
        }
     }
   else StringToCharArray(req.stData,data);

   if(hSession<=0 || hConnect<=0)
     {
      Close();
      if(!Open(Host,Port,User,Pass,Service))
        {
         Print("-Err Connect");
         Close();
         return(false);
        }
     }
//--- creating descriptor of the request
   hRequest=HttpOpenRequestW(hConnect,req.stVerb,req.stObject,Vers,nill,0,
   INTERNET_FLAG_KEEP_CONNECTION|INTERNET_FLAG_RELOAD|INTERNET_FLAG_PRAGMA_NOCACHE,0);
   if(hRequest<=0)
     {
      Print("-Err OpenRequest");
      InternetCloseHandle(hConnect);
      return(false);
     }
//--- sending the request
   hSend=HttpSendRequestW(hRequest,req.stHead,StringLen(req.stHead),data,ArraySize(data));
//--- sending the file
   if(hSend<=0)
     {
      int err=0;
      err=GetLastError(err);
      Print("-Err SendRequest= ",err);
     }
//--- reading the page
   if(hSend>0) ReadPage(hRequest,req.stOut,req.toFile);
//--- closing all handles
   InternetCloseHandle(hRequest); InternetCloseHandle(hSend);

   if(hSend<=0)
     {
      Close();
      return(false);
     }
   return(true);
  }
```

Now let's start working.

### Sending Data to a Website of the "application/x-www-form-urlencoded" Type

In the previous lesson, we've analyzed the [MetaArbitrage](https://www.mql5.com/en/articles/73) example (monitoring of quotes).

Let's remember, that the EA sends Bid prices of its symbol using a GET request; and as an answer, it receives prices of other brokers that are sent in the same way to the server from other terminals.

To change a GET request to a POST request, it is sufficient to "hide" the request line itself in the body of the request that comes after its header.

BOOL [**HttpSendRequest**](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384247(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa384247(v=VS.85).aspx")(

\_\_in  HINTERNET hRequest,

\_\_in  LPCTSTR lpszHeaders,

\_\_in  DWORD dwHeadersLength,

\_\_in  LPVOID lpOptional,

\_\_in  DWORD dwOptionalLength

);

- **hRequest**\[in\]

Handle returned by [**HttpOpenRequest**](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384233(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa384233(v=VS.85).aspx").
- **lpszHeaders**\[in\]

Pointer to a line containing headers to be added to the request. This parameter may be empty.
- **dwHeadersLength**\[in\]

Size of the header in bytes.
- **lpOptional**\[in\]

Pointer to an array with uchar data that is sent right after the header. Generally, this parameter is used for POST and PUT operations.
- **dwOptionalLength**\[in\]

Size of data in bytes. The parameter can be =0; it means that no additional information is sent.

From the description of the function, we can understand that data is sent as a byte **uchar**-array (the fourth parameter of the function). That is all what we need to know at this stage.

In the MetaArbitrage example, the GET request looks as following:

www.fxmaster.de/metaarbitr.php?server=Metaquotes&pair=EURUSD&bid=1.4512&time=13286794

The request itself is highlighted with the red color. Thus if we need to make a **POST** request, we should move its text to the **lpOptional** array of data.

Let's create a script called MetaSwap, which will send and receive information about swaps of a symbol.

```
#include <InternetLib.mqh>

string Server[];        // array of server names
double Long[], Short[]; // array for swap information
MqlNet INet;           // class instance for working

//------------------------------------------------------------------ OnStart
void OnStart()
{
//--- opening a session
  if (!INet.Open("www.fxmaster.de", 80, "", "", INTERNET_SERVICE_HTTP)) return;

//--- zeroizing arrays
  ArrayResize(Server, 0); ArrayResize(Long, 0); ArrayResize(Short, 0);
//--- the file for writing an example of swap information
  string file=Symbol()+"_swap.csv";
//--- sending swaps
  if (!SendData(file, "GET"))
  {
    Print("-err RecieveSwap");
    return;
  }
//--- read data from the received file
  if (!ReadSwap(file)) return;
//--- refresh information about swaps on the chart
  UpdateInfo();
}
```

Operation of the script is very simple.

First of all, the Internet session **INet.Open** is opened. Then the **SendData** function sends information about swaps of the current symbol. Then, if it is successfully sent, received swaps are read using **ReadSwap** and displayed on the chart using **UpdateInfo**.

At this moment, we are interested only in the **SendData** function.

```
//------------------------------------------------------------------ SendData
bool SendData(string file, string mode)
{
  string smb=Symbol();
  string Head="Content-Type: application/x-www-form-urlencoded"; // header
  string Path="/mt5swap/metaswap.php"; // path to the page
  string Data="server="+AccountInfoString(ACCOUNT_SERVER)+
              "&pair="+smb+
              "&long="+DTS(SymbolInfoDouble(smb, SYMBOL_SWAP_LONG))+
              "&short="+DTS(SymbolInfoDouble(smb, SYMBOL_SWAP_SHORT));

  tagRequest req; // initialization of parameters
  if (mode=="GET")  req.Init(mode, Path+"?"+Data, Head, "",   false, file, true);
  if (mode=="POST") req.Init(mode, Path,          Head, Data, false, file, true);

  return(INet.Request(req)); // sending request to the server
}
```

In this script, two methods of sending information are demonstrated - using GET and POST, for you to feel the difference between them.

Let's describe the variables of the function one by one:

- **Head** \- header of the request describing type of its contents. Actually, this is not the entire header of the request. The other fields of the header are created by the wininet.dll library. However, they can be modified using the [**HttpAddRequestHeaders**](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384227(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa384227(v=VS.85).aspx") function.
- **Path** \- this is the path to the request instance relatively to the initial domain www.fxmaster.de. In other words, it is the path to a **php script** that will process the request. By the way, it is not necessary to request only a php script, it can be an ordinary html page (we've even tried to request a mq5 file during our first lesson).
- **Data** \- this is the information that is sent to the server. Data is written according to the rules of passing of **parameter name=value**. The " **&**" sign is used as a data separator.

And the main thing - pay attention to the difference between making GET and POST requests in **tagRequest::Init**.

**In the GET method**, the path is sent together with the request body (united with the " **?**" sign), and the data field **lpOptional**(named **stData** in the structure) is left empty. **In the POST method**, the path exists on its own, and the request body is moved to **lpOptional**.

As you can see, the difference is not significant. The server script **metaswap.php** that receives the request is attached to the article.

### Sending "multipart/form-data" Data

Actually, POST requests are not analogous to GET requests (otherwise they wouldn't be needed). POST requests have a significant advantage - using them, you can send files with binary content.

The matter is a request of the urlencoded type is allowed to send a limited set of symbols. Otherwise, the "unallowed" symbols will be replaces with codes. Thus when sending binary data, they will be distorted. So you're no able to send even a small gif file using a GET request.

To solve this problem, special rules of describing a request are worked out; they allow exchanging with binary files in addition to text ones.

To reach this goal, the body of the request is divided into **sections**. The main thing is each section can have its own type of data. For example, the first one is text, the next one is an image/jpeg, etc. In other words, one request that is sent to the server can contain several types of data at once.

Let's take a look at the structure of such description by the example of data passed by the MetaSwap script.

Header of the request **Head** will look as following:

Content-Type: multipart/form-data; boundary=SEPARATOR\\r\\n

The keyword **SEPARATOR** – is a random set of symbols. However, you should watch this to be outside of the request data. In other words, this line should be unique - some abracadabra like hdsJK263shxaDFHLsdhsDdjf9 or anything else that comes to your mind :). In PHP, such a line is formed using the MD5 code of a current time.

The POST request itself looks as following (for easier understanding the fields are highlighted according to the general meaning):

\\r\\n

--SEPARATOR\\r\\n

Content-Disposition: form-data; name="Server"\\r\\n

\\r\\n

MetaQuotes-Demo

\\r\\n

--SEPARATOR\\r\\n

Content-Disposition: form-data; name="Pair"\\r\\n

\\r\\n

EURUSD

\\r\\n

--SEPARATOR\\r\\n

Content-Disposition: form-data; name="Long"\\r\\n

\\r\\n

1.02

\\r\\n

--SEPARATOR\\r\\n

Content-Disposition: form-data; name="Short"\\r\\n

\\r\\n

-0.05

\\r\\n

--SEPARATOR--\\r\\n

We explicitly specify the places for line feeds " **\\r\\n**", because they are obligatory symbols in a request. As you see, the same four fields are passed in the request, at that it is done in the usual text way.

The important peculiarities of placing separators:

- Two symbols " **--**" are placed before the separator.
- For the closing separator two additional symbols " **--**" are added after it.


In the next example, you can see a correct method of passing files in a request.

Imagine that an Expert Advisor makes a chart snapshot and a detailed report on the account in a text file when closing a position.

\\r\\n

--SEPARATOR\\r\\n

Content-Disposition: form-data; name="ExpertName"\\r\\n

\\r\\n

MACD\_Sample

\\r\\n

--SEPARATOR\\r\\n

Content-Disposition: file; name="screen"; filename="screen.gif"\\r\\n

Content-Type: image/gif\\r\\n

Content-Transfer-Encoding: binary\\r\\n

\\r\\n

......content of the gif file.....

\\r\\n

--SEPARATOR\\r\\n

Content-Disposition: form-data; name="statement"; filename="statement.csv"\\r\\n

Content-Type: application/octet-stream\\r\\n

Content-Transfer-Encoding: binary\\r\\n

\\r\\n

......content of the csv file.....

\\r\\n

--SEPARATOR--\\r\\n

Two new headers appear in the request:

**Content-Type** \- describes the type of content. All the possible types are accurately described in the [RFC\[2046\]](https://www.mql5.com/go?link=http://www.ietf.org/rfc/rfc2046.txt "http://www.ietf.org/rfc/rfc2046.txt") standard. We used two types - **image/gif** and **application/octet-stream**.

Two variants of writing **Content-Disposition** \- fileand form-data are equivalent and are correctly processed by PHP in both cases. So you can use file or form-data at your option. You can better see the difference between their representations in Charles.

**Content-Transfer-Encoding** \- it describes the encoding of content. It may be absent for text data.

To consolidate the material, let's write the script ScreenPost, which sends screenshots to the server:

```
#include <InternetLib.mqh>

MqlNet INet; // class instance for working

//------------------------------------------------------------------ OnStart
void OnStart()
{
  // opening session
  if (!INet.Open("www.fxmaster.de", 80, "", "", INTERNET_SERVICE_HTTP)) return;

  string giffile=Symbol()+"_"+TimeToString(TimeCurrent(), TIME_DATE)+".gif"; // name of file to be sent

  // creating screenshot 800х600px
  if (!ChartScreenShot(0, giffile, 800, 600)) { Print("-err ScreenShot "); return; }

  // reading gif file to the array
  int h=FileOpen(giffile, FILE_ANSI|FILE_BIN|FILE_READ); if (h<0) { Print("-err Open gif-file "+giffile); return; }
  FileSeek(h, 0, SEEK_SET);
  ulong n=FileSize(h); // determining the size of file
  uchar gif[]; ArrayResize(gif, (int)n); // creating uichar array according to the size of data
  FileReadArray(h, gif); // reading file to the array
  FileClose(h); // closing the file

  // creating file to be sent
  string sendfile="sendfile.txt";
  h=FileOpen(sendfile, FILE_ANSI|FILE_BIN|FILE_WRITE); if (h<0) { Print("-err Open send-file "+sendfile); return; }
  FileSeek(h, 0, SEEK_SET);

  // forming a request
  string bound="++1BEF0A57BE110FD467A++"; // separator of data in the request
  string Head="Content-Type: multipart/form-data; boundary="+bound+"\r\n"; // header
  string Path="/mt5screen/screen.php"; // path to the page

  // writing data
  FileWriteString(h, "\r\n--"+bound+"\r\n");
  FileWriteString(h, "Content-Disposition: form-data; name=\"EA\"\r\n"); // the "name of EA" field
  FileWriteString(h, "\r\n");
  FileWriteString(h, "NAME_EA");
  FileWriteString(h, "\r\n--"+bound+"\r\n");
  FileWriteString(h, "Content-Disposition: file; name=\"data\"; filename=\""+giffile+"\"\r\n"); // field of the gif file
  FileWriteString(h, "Content-Type: image/gif\r\n");
  FileWriteString(h, "Content-Transfer-Encoding: binary\r\n");
  FileWriteString(h, "\r\n");
  FileWriteArray(h, gif); // writing gif data
  FileWriteString(h, "\r\n--"+bound+"--\r\n");
  FileClose(h); // closing the file

  tagRequest req; // initialization of parameters
  req.Init("POST", Path, Head, sendfile, true, "answer.htm", true);

  if (INet.Request(req)) Print("-err Request"); // sending the request to the server
  else Print("+ok Request");
}
```

The server script that receives information:

```
<?php
$ea=$_POST['EA'];
$data=file_get_contents($_FILES['data']['tmp_name']); // information in the file
$file=$_FILES['data']['name'];
$h=fopen(dirname(__FILE__)."/$ea/$file", 'wb'); // creating a file in the EA folder
fwrite($h, $data); fclose($h); // saving data
?>
```

**It is highly recommended to get acquainted with the rules of receiving files by the server to avoid security problems!**

### Working with Cookies

This subject will be described briefly as an addition to the previous lesson and food for thought about its features.

As you know, Cookies are intended to avoid continuous requesting of personal details by servers. Once a server receives personal details required for the current work session from a user, it leaves a text file with that information on the user computer. Further, when the user moves between pages, the server doesn't request that information again from the user; it automatically takes the information from the browser cache.

For example, when you enable the "Remember me" option while authorizing at the [www.mql5.com](https://www.mql5.com/) server, you save a Cookie with you details on your computer. At the next visit of the website, the browser will pass the Cookie to the server without asking you.

If you are interested, you can open the folder (WinXP) **C:\\Documents and Settings\\<User>\\Cookies** and view the contents of different web sites that you've visited.

With regard to our needs, Cookies can be used for reading your pages of the MQL5 forum. In other words, you will read the information as if you are authorized at the website under your login, and then you will analyze obtained pages. The optimal variant is to analyze Cookies using a local proxy server Charles. It shows detailed information about all received/sent requests, including Cookies.

For example:

- An Expert Advisor (or an external application) that requests the [https://www.mql5.com/en/job](https://www.mql5.com/en/job) page once an hour and receives the list of new job offers.
- Also it requests a branch, for example [https://www.mql5.com/en/forum/53](https://www.mql5.com/en/forum/53), and checks if there are new messages.
- In addition, it may check whether there are new "private messages" at forums.


To set a Cookie in a request, the [**InternetSetCookie**](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa385107(v=VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa385107(v=VS.85).aspx") function is used.

BOOL **InternetSetCookie**(

\_\_in  LPCTSTR lpszUrl,

\_\_in  LPCTSTR lpszCookieName,

\_\_in  LPCTSTR lpszCookieData

);

- **lpszUrl**\[in\] - Name of a server, for example, www.mql5.com
- **lpszCookieName**\[in\]- Name of a Cookie
- **lpszCookieData**\[in\] - Data for the Cookie

To set several Cookies, call this function for each of them.

An interesting feature: a call of **InternetSetCookie** can be made at any time, even when you are not connected to the server.

### Conclusion

We, we have got acquainted with another type of HTTP requests, obtained the possibility of sending binary files, what allows extending the facilities of working with your servers; and we have learned the methods of working with Cookies.

We can make the following list of directions of further developments:

- Organization of remote storage of reports;
- Exchange of files between users, updating of versions of Expert Advisors/indicators;
- Creation of custom scanners that work under your account and monitor activity at a website.

### Useful Links

01. A proxy server for viewing headers sent - [http://www.charlesproxy.com/](https://www.mql5.com/go?link=https://www.charlesproxy.com/ "http://www.charlesproxy.com/")
02. Description of WinHTTP - [http://msdn.microsoft.com/en-us/library/aa385331%28VS.85%29.aspx](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa385331(VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa385331(VS.85).aspx")
03. Description of HTTP Session - [http://msdn.microsoft.com/en-us/library/aa384322%28VS.85%29.aspx](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384322(VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa384322(VS.85).aspx")
04. The Denwer toolkit for a local installation of Apache+PHP - [http://www.denwer.ru/](https://www.mql5.com/go?link=http://www.denwer.ru/ "http://www.denwer.ru/")
05. Types of request headers - [http://www.codenet.ru/webmast/php/HTTP-POST.php#part\_3\_2](https://www.mql5.com/go?link=http://www.codenet.ru/webmast/php/HTTP-POST.php "http://www.codenet.ru/webmast/php/HTTP-POST.php")
06. Types of requests - [http://www.w3.org/TR/REC-html40/interact/forms.html#form-content-type](https://www.mql5.com/go?link=http://www.w3.org/TR/REC-html40/interact/forms.html "http://www.w3.org/TR/REC-html40/interact/forms.html")
07. Types of requests - ftp://ftp.isi.edu/in-notes/iana/assignments/media-types/media-types.
08. Structure of using HINTERNET - [http://msdn.microsoft.com/en-us/library/aa383766%28VS.85%29.aspx](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa383766(VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa383766(VS.85).aspx")
09. Working with files - [http://msdn.microsoft.com/en-us/library/aa364232%28VS.85%29.aspx](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa364232(VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa364232(VS.85).aspx")
10. Types of data to pass to MQL - [http://msdn.microsoft.com/en-us/library/aa383751%28VS.85%29.aspx](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa383751(VS.85).aspx "http://msdn.microsoft.com/en-us/library/aa383751(VS.85).aspx")

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/276](https://www.mql5.com/ru/articles/276)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/276.zip "Download all attachments in the single ZIP archive")

[metaswap.zip](https://www.mql5.com/en/articles/download/276/metaswap.zip "Download metaswap.zip")(0.66 KB)

[screenpost.zip](https://www.mql5.com/en/articles/download/276/screenpost.zip "Download screenpost.zip")(0.33 KB)

[internetlib.mqh](https://www.mql5.com/en/articles/download/276/internetlib.mqh "Download internetlib.mqh")(12.82 KB)

[screenpost.mq5](https://www.mql5.com/en/articles/download/276/screenpost.mq5 "Download screenpost.mq5")(2.63 KB)

[metaswap.mq5](https://www.mql5.com/en/articles/download/276/metaswap.mq5 "Download metaswap.mq5")(4.84 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)
- [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)
- [Getting Rid of Self-Made DLLs](https://www.mql5.com/en/articles/364)
- [Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)
- [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)
- [The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3927)**
(16)


![Juer](https://c.mql5.com/avatar/avatar_na2.png)

**[Juer](https://www.mql5.com/en/users/juer)**
\|
16 Sep 2018 at 11:06

```
#property tester_library "wininet.dll"
#property tester_library "Kernel32.dll"
```

It seemed to help.


![Juer](https://c.mql5.com/avatar/avatar_na2.png)

**[Juer](https://www.mql5.com/en/users/juer)**
\|
16 Sep 2018 at 11:12

**Juer:**

It seemed to help

No, it didn't.

![Juer](https://c.mql5.com/avatar/avatar_na2.png)

**[Juer](https://www.mql5.com/en/users/juer)**
\|
16 Sep 2018 at 11:21

```
#property tester_library "wininet.dll"
#property tester_library "Kernel32.dll"
```

Dope... Testing works after compilation. When debugging in [testing mode](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation "Tick generation modes in MetaTrader 5 Client Terminal") it does not work. And then it doesn't work when testing.


![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
16 Sep 2018 at 20:16

**Juer:**

Dope... Testing works after compilation. When debugging in [testing mode](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation "Tick generation modes in MetaTrader 5 Client Terminal") it does not work. And then it doesn't work just when testing.

After running in debug mode ex-file remains from this mode. Apparently, only the release version will work.

![Yu Pang Chan](https://c.mql5.com/avatar/2017/10/59DA7A74-25D4.JPG)

**[Yu Pang Chan](https://www.mql5.com/en/users/freezemusic)**
\|
22 Mar 2021 at 11:03

Thanks anyway for the great library.

I finally managed to POST data through wininet, so let me also contribute something here:

for access violation,

1\. Changing "nill" to [NULL](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants "MQL5 documentation: Other Constants") does not help

2\. Changing "nill" to 0 does not help

what I did is actually changed the following:

```
--- string nill = "";
+++ string nill = "\0";
```

and everything will be fine after that. Enjoy :)

![Using Self-Organizing Feature Maps (Kohonen Maps) in MetaTrader 5](https://c.mql5.com/2/0/Self_organizing_maps_in_MQL5.png)[Using Self-Organizing Feature Maps (Kohonen Maps) in MetaTrader 5](https://www.mql5.com/en/articles/283)

One of the most interesting aspects of Self-Organizing Feature Maps (Kohonen maps) is that they learn to classify data without supervision. In its basic form it produces a similarity map of input data (clustering). The SOM maps can be used for classification and visualizing of high-dimensional data. In this article we will consider several simple applications of Kohonen maps.

![Decreasing Memory Consumption by Auxiliary Indicators](https://c.mql5.com/2/0/MQL5_indicator_memory_optimization.png)[Decreasing Memory Consumption by Auxiliary Indicators](https://www.mql5.com/en/articles/259)

If an indicator uses values of many other indicators for its calculations, it consumes a lot of memory. The article describes several methods of decreasing the memory consumption when using auxiliary indicators. Saved memory allows increasing the number of simultaneously used currency pairs, indicators and strategies in the client terminal. It increases the reliability of trade portfolio. Such a simple care about technical resources of your computer can turn into money resources at your deposit.

![Payments and payment methods](https://c.mql5.com/2/0/mql5_payment__1.png)[Payments and payment methods](https://www.mql5.com/en/articles/302)

MQL5.community Services offer great opportunities for traders as well as for the developers of applications for the MetaTrader terminal. In this article, we explain how payments for MQL5 services are performed, how the earned money can be withdraw, and how the operation security is ensured.

![Advanced Adaptive Indicators Theory and Implementation in MQL5](https://c.mql5.com/2/0/Advanced_adaptive_indicators_MQL5.png)[Advanced Adaptive Indicators Theory and Implementation in MQL5](https://www.mql5.com/en/articles/288)

This article will describe advanced adaptive indicators and their implementation in MQL5: Adaptive Cyber Cycle, Adaptive Center of Gravity and Adaptive RVI. All indicators were originally presented in "Cybernetic Analysis for Stocks and Futures" by John F. Ehlers.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=twryufucmwyonqbulqhpexitzuusmevx&ssn=1769192467574591428&ssn_dr=0&ssn_sr=0&fv_date=1769192467&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F276&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Using%20WinInet%20in%20MQL5.%20Part%202%3A%20POST%20Requests%20and%20Files%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919246791098483&fz_uniq=5071767506844069395&sv=2552)

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