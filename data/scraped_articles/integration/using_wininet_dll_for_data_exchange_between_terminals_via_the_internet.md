---
title: Using WinInet.dll for Data Exchange between Terminals via the Internet
url: https://www.mql5.com/en/articles/73
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:08:57.046724
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=abmirrrmwjwgdxmvfifgxrqmvpsbvjit&ssn=1769252935294020323&ssn_dr=0&ssn_sr=0&fv_date=1769252935&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F73&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Using%20WinInet.dll%20for%20Data%20Exchange%20between%20Terminals%20via%20the%20Internet%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925293593321544&fz_uniq=5083366285070309959&sv=2552)

MetaTrader 5 / Integration


MetaTrader 5 opens up unique opportunities for users, using a number of new user interface elements in its arsenal. Because of this, the functions that were previously unavailable can now be used to the maximum.

In this lesson we will learn to:

- use basic Internet technologies;
- exchange data between terminals via the server;
- create a generic library class to work with the Internet in the MQL5 environment.

The MQL5 [CodeBase](https://www.mql5.com/en/code) contains an example of [script](https://www.mql5.com/en/code/82), which works with the wininet.dll library and shows an example of the server page request. But today we'll go much further, and make the server, not only give us the page, but also to send and store this data for subsequent transfers to other requesting terminals.

Note: for those who don't have access to a server, configured with PHP, we suggest downloading the [Denwer](https://www.mql5.com/go?link=http://www.denwer.ru/ "http://www.denwer.ru/") kit, and using it as a working platform. Also, we recommend using the Apache server and PHP on your localhost for testing.

To send any request to the server, we will need the 7 major functions of the library.

| [InternetAttemptConnect](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384331(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384331(vs.85).aspx") | Attempt to locate an Internet connection and establish it |
| [InternetOpen](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa385096(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa385096(vs.85).aspx") | Initializes the structure for the work of the WinInet library functions. This function must be activated before activating any other functions of the library. |
| [InternetConnect](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa385098(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa385098(vs.85).aspx") | Opens the resource specified by the address HTTP URL or FTP. Returns the descriptor to an open connection |
| [HttpOpenRequest](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384233(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384233(vs.85).aspx") | Creates a descriptor for HTTP requests for setting up a connection |
| [HttpSendRequest](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384247(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384247(vs.85).aspx") | Sends a query, using the created descriptor |
| [InternetReadFile](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa385103(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa385103(vs.85).aspx") | Reads data, received from the server after the query has been processed |
| [InternetCloseHandle](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa384350(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384350(vs.85).aspx") | Releases the transferred descriptor |

A detailed description of all of the functions and their parameters can be found in the [MSDN](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa385331(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa385331(vs.85).aspx") Help system.

The declaration of functions remained the same as in MQL4, with the exception of the use of Unicode
calls and line transfers by the link.

```
#import "wininet.dll"
int InternetAttemptConnect(int x);
int InternetOpenW(string &sAgent,int lAccessType,string &sProxyName,string &sProxyBypass,int lFlags);
int InternetConnectW(int hInternet,string &szServerName,int nServerPort,string &lpszUsername,string &lpszPassword,int dwService,int dwFlags,int dwContext);
int HttpOpenRequestW(int hConnect,string &Verb,string &ObjectName,string &Version,string &Referer,string &AcceptTypes,uint dwFlags,int dwContext);
int HttpSendRequestW(int hRequest,string &lpszHeaders,int dwHeadersLength,uchar &lpOptional[],int dwOptionalLength);
int HttpQueryInfoW(int hRequest,int dwInfoLevel,int &lpvBuffer[],int &lpdwBufferLength,int &lpdwIndex);
int InternetReadFile(int hFile,uchar &sBuffer[],int lNumBytesToRead,int &lNumberOfBytesRead);
int InternetCloseHandle(int hInet);
#import

//To make it clear, we will use the constant names from wininet.h.
#define OPEN_TYPE_PRECONFIG     0           // use the configuration by default
#define FLAG_KEEP_CONNECTION    0x00400000  // do not terminate the connection
#define FLAG_PRAGMA_NOCACHE     0x00000100  // no cashing of the page
#define FLAG_RELOAD             0x80000000  // receive the page from the server when accessing it
#define SERVICE_HTTP            3           // the required protocol
```

A detailed description of the flags is located in the same section [MSDN](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa385331(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa385331(vs.85).aspx") for each of the functions. If you wish to see the declaration of other constants and functions, then you can download the original wininet.h file, located in the attachments to the article.

**1\. Guides for creating and deleting Internet session**

The first thing we should do is create a session and open a connection to the host. A session should preferably be created only once during the program initialization (eg, in a function [OnInit)](https://www.mql5.com/en/docs/basis/function/events#oninit). Or it can be done at the very beginning of launching the Expert Advisor, but it is important to make sure that its successful creation was done only once before the closing of the session. And it shouldn't be invoked repeated and without need, with every new iteration of the implementation [OnStart](https://www.mql5.com/en/docs/basis/function/events#onstart) or [OnTimer](https://www.mql5.com/en/docs/basis/function/events#ontimer). It's important to avoid frequent calls and creation of the structures required for each call.

Therefore, we will use only one global class instance for describing the session and connection descriptiors.

```
   string            Host;       // host name
   int               Port;       // port
   int               Session;    // session descriptor
   int               Connect;    // connection descriptor

bool MqlNet::Open(string aHost,int aPort)
  {
   if(aHost=="")
     {
      Print("-Host is not specified");
      return(false);
     }
   // checking the DLL resolution in the terminal
   if(!TerminalInfoInteger(TERMINAL_DLLS_ALLOWED))
     {
      Print("-DLL is not allowed");
      return(false);
     }
   // if the session was identifies, then we close
   if(Session>0 || Connect>0) Close();
   // record of attempting to open into the journal
   Print("+Open Inet...");
   // if we were not able to check for the presence of an Internet connection, then we exit
   if(InternetAttemptConnect(0)!=0)
     {
      Print("-Err AttemptConnect");
      return(false);
     }
   string UserAgent="Mozilla"; string nill="";
   // open a session
   Session=InternetOpenW(UserAgent,OPEN_TYPE_PRECONFIG,nill,nill,0);
   // if we were not able to open a session, then exit
   if(Session<=0)
     {
      Print("-Err create Session");
      Close();
      return(false);
     }
   Connect=InternetConnectW(Session,aHost,aPort,nill,nill,SERVICE_HTTP,0,0);
   if(Connect<=0)
     {
      Print("-Err create Connect");
      Close();
      return(false);
     }
   Host=aHost; Port=aPort;
   // otherwise all attempts were successful
   return(true);
  }
```

After initialization the descriptors _Session_ and _Connect_ can be used in all of the following functions. Once all work is completed and MQL-programs are uninstalled, they must be removed. This is done by using the [InternetCloseHandle](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa384350(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384350(vs.85).aspx") function.

```
void MqlNet::CloseInet()
  {
   Print("-Close Inet...");
   if(Session>0) InternetCloseHandle(Session); Session=-1;
   if(Connect>0) InternetCloseHandle(Connect); Connect=-1;
  }
```

**Attention!** When working with Internet functions, it is necessary to free up all of the descriptors, derived from them, using [InternetCloseHandle](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa384350(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384350(vs.85).aspx").

**2\. Sending a request to the server and receiving the page**

For sending a request and receiving a page in response to this request, we will need the remaining three functions [HttpOpenRequest](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384233(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384233(vs.85).aspx"), [HttpSendRequest](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa384247(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384247(vs.85).aspx") и [InternetReadFile](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa385103(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa385103(vs.85).aspx"). The essence of the receiving the page in response to a request is basically the simple process of saving its contents into a local file.

![](https://c.mql5.com/2/1/bw.png)

For the convenience of working with requests and contents we will create two universal functions.

_Sending a request:_

```
bool MqlNet::Request(string Verb,string Object,string &Out,bool toFile=false,string addData="",bool fromFile=false)
  {
   if(toFile && Out=="")
     {
      Print("-File is not specified ");
      return(false);
     }
   uchar data[];
   int hRequest,hSend,h;
   string Vers="HTTP/1.1";
   string nill="";
   if(fromFile)
     {
      if(FileToArray(addData,data)<0)
        {
         Print("-Err reading file "+addData);

         return(false);
        }
     } // read file in the array
   else StringToCharArray(addData,data);

   if(Session<=0 || Connect<=0)
     {
      Close();
      if(!Open(Host,Port))
        {
         Print("-Err Connect");
         Close();
         return(false);
        }
     }
   // create a request descriptor
   hRequest=HttpOpenRequestW(Connect,Verb,Object,Vers,nill,nill,FLAG_KEEP_CONNECTION|FLAG_RELOAD|FLAG_PRAGMA_NOCACHE,0);
   if(hRequest<=0)
     {
      Print("-Err OpenRequest");
      InternetCloseHandle(Connect);
      return(false);
     }
   // send request
   // headline for request
   string head="Content-Type: application/x-www-form-urlencoded";
   // sent file
   hSend=HttpSendRequestW(hRequest,head,StringLen(head),data,ArraySize(data)-1);
   if(hSend<=0)
     {
      Print("-Err SendRequest");
      InternetCloseHandle(hRequest);
      Close();
     }
   // read the page
   ReadPage(hRequest,Out,toFile);
   // close all handles
   InternetCloseHandle(hRequest);
   InternetCloseHandle(hSend);
   return(true);
  }
```

Function parameters of MqlNet:: Request:

- string **Verb**– request type “GET” or “POST”;
- string **Object**– name of the page with its passed on parameters;
- string & **Out**– line to which the answer is received;
- bool **toFile** – if toFile=true, then Out indicates the name of the file where the answer should be received;
- string **addData** \- Additional data;
- bool **fromFile** \- If fromFile = true, then addData is the name of the file that needs to be sent.

_Reading the contents of the received descriptor_

```
void MqlNet::ReadPage(int hRequest,string &Out,bool toFile)
  {
   // read the page
   uchar ch[100];
   string toStr="";
   int dwBytes,h;
   while(InternetReadFile(hRequest,ch,100,dwBytes))
     {
      if(dwBytes<=0) break;
      toStr=toStr+CharArrayToString(ch,0,dwBytes);
     }
   if(toFile)
     {
      h=FileOpen(Out,FILE_BIN|FILE_WRITE);
      FileWriteString(h,toStr);
      FileClose(h);
     }
   else Out=toStr;
  }
```

Function parameters of MqlNet:: ReadPage:

- Int **hRequest** \- request descriptor, from which the data is read;
- string & **Out**– line to which the answer is received;
- bool **toFile** \- If toFile = true, then Out is the name of the file where the answer will be received.

And by gathering all of this into one, we will obtain an MqlNet library class for working with the Internet.

```
class MqlNet
  {
   string            Host;     // host name
   int               Port;     // port
   int               Session; // session descriptor
   int               Connect; // connection descriptor
public:
                     MqlNet(); // class constructor
                    ~MqlNet(); // destructor
   bool              Open(string aHost,int aPort); // create a session and open a connection
   void              Close(); // close session and connection
   bool              Request(string Verb,string Request,string &Out,bool toFile=false,string addData="",bool fromFile=false); // send request
   bool              OpenURL(string URL,string &Out,bool toFile); // somply read the page into the file or the variable
   void              ReadPage(int hRequest,string &Out,bool toFile); // read the page
   int               FileToArray(string FileName,uchar &data[]); // copy the file into the array for sending
  };
```

That's basically all of the required functions that are likely to satisfy the diversified needs for working with the Internet. Consider the examples of their use.

**Example 1. Automatic download of MQL-programs into the terminal's folders. MetaGrabber script**

To begin our testing of the class's work, let us start with the easiest tasks - reading the page and saving its contents to the specified folder. But a simple reading of the pages is unlikely to be very interesting, so in order to gain something from the work of the script, let's assign it a functional of grabber of mql programs from sites. The task of the MetaGrabber script will be:

- URL analysis and separation of it into the host, the request, and the file name;
- sending a request to the host, receiving and saving the file into the terminal folder \\\ Files;
- moving it from the Files to one of the required data folders:

\\Experts, \\Indicators, \\Scripts, \\Include, \\Libraries, \\Tester(set), \\Templates.


To solve the second problem we use the MqlNet class. For the third task we use the function [MoveFileEx](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa365240(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa365240(vs.85).aspx") from Kernel32.dll

```
#import "Kernel32.dll"
bool MoveFileExW(string &lpExistingFileName, string &lpNewFileName, int dwFlags);
#import "Kernel32.dll"
```

For the first problem, let's make a small service function of parsing the URL line.

We need to allocate three lines from the address: the host, the path to the site, and the file name.

For example, in line http://www.mysite.com/folder/page.html

> \- Host = www.mysite.com
>
>  \- Request = / folder / page.html
>
>  \- File name = page.html

In the case of CodeBase on the MQL5 site, the path ways have the same structure. For example, the path to the library [ErrorDescription.mq5](https://www.mql5.com/en/code/79) on page https://www.mql5.com/ru/code/79 looks like http://p.mql5.com/data/18/79/ErrorDescription.mqh. This path is easily obtained by right clicking on the link and selecting "Copy Link". Thus, the URL is split into two parts, one for the request and one for the file name for convenience of file storage.

> \- Host = p.mql5.com
>
>  \- Request = / data/18/79/5/ErrorDescription.mqh
>
>  \- File name = ErrorDescription.mqh

This is the kind of line parsing that the following ParseURL function will be dealing with.

```
void ParseURL(string path,string &host,string &request,string &filename)
  {
   host=StringSubstr(URL,7);
   // removed
   int i=StringFind(host,"/");
   request=StringSubstr(host,i);
   host=StringSubstr(host,0,i);
   string file="";
   for(i=StringLen(URL)-1; i>=0; i--)
      if(StringSubstr(URL,i,1)=="/")
        {
         file=StringSubstr(URL,i+1);
         break;
        }
   if(file!="") filename=file;
  }
```

In the outer parameters of the script we will make only two parameters - URL (path of the mql5 file) and the type of folder of subsequent placement - that is, into which terminal folder you wish to place it.

As a result, we obtain a short but very useful script.

```
//+------------------------------------------------------------------+
//|                                                  MetaGrabber.mq5 |
//|                                 Copyright © 2010 www.fxmaster.de |
//|                                         Coding by Sergeev Alexey |
//+------------------------------------------------------------------+
#property copyright "www.fxmaster.de  © 2010"
#property link      "www.fxmaster.de"
#property version               "1.00"
#property description  "Download files from internet"

#property script_show_inputs

#include <InternetLib.mqh>

#import "Kernel32.dll"
bool MoveFileExW(string &lpExistingFileName,string &lpNewFileName,int dwFlags);
#import
#define MOVEFILE_REPLACE_EXISTING 0x1

enum _FolderType
  {
   Experts=0,
   Indicators=1,
   Scripts=2,
   Include=3,
   Libraries=4,
   Files=5,
   Templates=6,
   TesterSet=7
  };

input string URL="";
input _FolderType FolderType=0;
//------------------------------------------------------------------ OnStart
int OnStart()
  {
   MqlNet INet; // variable for working in the Internet
   string Host,Request,FileName="Recieve_"+TimeToString(TimeCurrent())+".mq5";

   // parse url
   ParseURL(URL,Host,Request,FileName);

   // open session
   if(!INet.Open(Host,80)) return(0);
   Print("+Copy "+FileName+" from  http://"+Host+" to "+GetFolder(FolderType));

   // obtained file
   if(!INet.Request("GET",Request,FileName,true))
     {
      Print("-Err download "+URL);
      return(0);
     }
   Print("+Ok download "+FileName);

   // move to the target folder
   string to,from,dir;
   // if there is no need to move it elsewhere
   if(FolderType==Files) return(0);

   // from
   from=TerminalInfoString(TERMINAL_DATA_PATH)+"\\MQL5\\Files\\"+FileName;

   // to
   to=TerminalInfoString(TERMINAL_DATA_PATH)+"\\";
   if(FolderType!=Templates && FolderType!=TesterSet) to+="MQL5\\";
   to+=GetFolder(FolderType)+"\\"+FileName;

   // move file
   if(!MoveFileExW(from,to,MOVEFILE_REPLACE_EXISTING))
     {
      Print("-Err move to "+to);
      return(0);
     }
   Print("+Ok move "+FileName+" to "+GetFolder(FolderType));

   return(0);
  }
//------------------------------------------------------------------ GetFolder
string GetFolder(_FolderType foldertype)
  {
   if(foldertype==Experts) return("Experts");
   if(foldertype==Indicators) return("Indicators");
   if(foldertype==Scripts) return("Scripts");
   if(foldertype==Include) return("Include");
   if(foldertype==Libraries) return("Libraries");
   if(foldertype==Files) return("Files");
   if(foldertype==Templates) return("Profiles\\Templates");
   if(foldertype==TesterSet) return("Tester");
   return("");
  }
//------------------------------------------------------------------ ParseURL
void ParseURL(string path,string &host,string &request,string &filename)
  {
   host=StringSubstr(URL,7);
   // removed
   int i=StringFind(host,"/");
   request=StringSubstr(host,i);
   host=StringSubstr(host,0,i);
   string file="";
   for(i=StringLen(URL)-1; i>=0; i--)
      if(StringSubstr(URL,i,1)=="/")
        {
         file=StringSubstr(URL,i+1);
         break;
        }
   if(file!="") filename=file;
  }
//+------------------------------------------------------------------+
```

![](https://c.mql5.com/2/1/grabopti.gif)

Let's conduct the experiments on our favorite section [https://www.mql5.com/en/code](https://www.mql5.com/en/code). The downloaded files will immediately appear in the editor's navigator, and they can be compiled without restarting the terminal or the editor. And they will not wander through the long paths of the file system in the search for the desired folder to move the files into.

Attention! Many sites put up a security against massive content downloading, and your IP-address, in case of such mass download, may be blocked by this resource. So pay careful attention to the use of "machine" downloading of files from resources, which you access often and do not want to be banned from using.

Those who wish to go even further, improving the proposed service, can use the [Clipboard](https://www.mql5.com/en/code/104) script with the interception of the clipboard's contents and the further automatic downloading.

**Example 2. Monitoring quotes from multiple brokers on a single chart**

So we have learned to obtain files from the Internet. Now let's consider a more interesting question - how to send and store this data on the server. For this we need a small additional PHP-script, which will be located on the server. Using the written MqlNet class, we create an Expert Advisor for monitoring - **MetaArbitrage** . The task of the expert in conjunction with the PHP-script will be:

- Sending an Expert Advisor request to the server;
- formation of the response page (PHP) on the server;
- reception of this page by the Expert Advisor;
- its analysis and the delivery of results to the screen.

The schematic diagram of the interaction between the MQL-module and the PHP-script is as follows:

![](https://c.mql5.com/2/1/11__1.jpg)

We will use the MqlNet class to solve these tasks.

To avoid the duplication of data, as well as to weed out outdated quotes - we will send 4 main parameters: the name of the broker's server (the source of current prices), the currency, the price and time of quotes in UTC. For example, a request to access the script from the resources of our company is as follows:

```
www.fxmaster.de/metaarbitr.php?server=Metaquotes&pair=EURUSD&bid=1.4512&time=13286794
```

These parameters and the actual quote are stored on the server and will be issued on the response page, along with all other stored quotes of this currency.

The "collateral" convenience of this exchange is that quotes can be sent from MT5, as well as from MT4!

The page, which is formed by the server, is a regular CSV file. In this script it looks like this:

> ServerName1; Bid1; Time1
>
> ServerName 2; Bid2; Time2
>
> ServerName 3; Bid3; Time3
>
> …
>
> ServerName N; BidN; TimeN

But you can add to it your own additional parameters (eg, server type - demo or real). We store this CSV-file and parse it line by line, with the output of a table of value and line prices displayed on the screen.

Processing this file can be in be accomplished in many different ways, choosing the once that are required in each particular case. For example, filter out the quotes, received from the MetaTrader 4 demo server, etc.

![](https://c.mql5.com/2/1/Arbitrage1.gif)

The advantages of using the Internet server are obvious - you are sending your quotes, which can be received and viewed by another trader. Likewise, you will receive quotes that are sent to other traders. That is, the interaction between the terminals is bilateral, the exchange of data is accomplished as shown in the following scheme:

![](https://c.mql5.com/2/1/Arb.png)

This scheme serve as the basis for the principle of information exchange between any number of terminals. A full MetaArbitrage Expert Advisor and the PHP-script with commentaries can be downloaded from the link in the attachments. More about PHP-used functions can be read on the following site [php.su](https://www.mql5.com/go?link=http://www.php.su/functions/?page=abc "http://www.php.su/functions/?page=abc")

**Example 3. Exchange of messages (mini chat) within the terminal. MetaChat Expert Advisor**

Let's take a step away from trading and numbers, and create an application, which will allow us to chat with several people at once, without exiting the terminal. To do this we will need one more PHP script, which in general is very similar to the previous one. Except for the fact that in this script, instead of analyzing time quotations, we will be analyzing the number of lines in a file. The task of the Expert Advisor will be:

- Sending a text line to the server;
- addition of this line into the shared file, controlling the size of the file, issuing the response file (php);
- receiving the current chat and displaying it on the screen.


The work of MetaChat will not differ from that of the previous Expert Advisor. The same principle, and the same simple CSV file for output.

**![](https://c.mql5.com/2/1/Untitled.gif)**

MetaChat and MetaArbitrage are maintained on the site of its developers. The PHP-scripts for their work are also located there.

Therefore, if you want to test a work or use this service, you can access it through the following link:

MetaСhat - www.fxmaster.de/metachat.php

MetaArbitrage - www.fxmaster.de/metaarbitr.php

### **Conclusion**

And so we have familiarized ourselves with the HTTP-requests. We gained the ability to send and receive data through the Internet, and to organize the working process more comfortably. But any capabilities can always be improves. The following can be considered the new potential directions of their improvements:

- reading the news or receiving other information directly into the terminal for the analysis of Expert Advisors;
- remote management of Expert Advisors;
- Automatic updates of Expert Advisors / indicators;
- copiers / translators of trades, sending signals;
- downloading templates along with lights and set-files for Expert Advisors:
- And much, much more ...


In this article we used the GET type of requests. They sufficiently fulfill the task when you need to obtain a file, or send a request, with a small number of parameters, for a server analyses.

In our next lesson, we will take a careful look at POST requests - sending files to the server or file sharing between terminals, and we will consider the examples of their use.

**Useful Resources**

- Denver Set for installing the Apache server + PHP [http://www.denwer.ru/](https://www.mql5.com/go?link=http://www.denwer.ru/ "http://www.denwer.ru/")
- A proxy for viewing sent headlines [http://www.charlesproxy.com/](https://www.mql5.com/go?link=https://www.charlesproxy.com/ "http://www.charlesproxy.com/")
- Types of headline requests [http://www.codenet.ru/webmast/php/HTTP-POST.php#part\_3\_2](https://www.mql5.com/go?link=http://www.codenet.ru/webmast/php/http-post.php "http://www.codenet.ru/webmast/php/http-post.php")
- Типы запросов [http://www.w3.org/TR/REC-html40/interact/forms.html#form-content-type](https://www.mql5.com/go?link=https://www.w3.org/TR/REC-html40/interact/forms.html "http://www.w3.org/tr/rec-html40/interact/forms.html")
- Request types ftp://ftp.isi.edu/in-notes/iana/assignments/media-types/media-types
- Description of WinHTTP [http://msdn.microsoft.com/en-us/library/aa385331%28VS.85%29.aspx](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa385331(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa385331(vs.85).aspx")
- Description of HTTP Session [http://msdn.microsoft.com/en-us/library/aa384322%28VS.85%29.aspx](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa384322(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa384322(vs.85).aspx")
- Usage structure of HINTERNET [http://msdn.microsoft.com/en-us/library/aa383766%28VS.85%29.aspx](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa383766(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa383766(vs.85).aspx")
- Working with files [http://msdn.microsoft.com/en-us/library/aa364232%28VS.85%29.aspx](https://www.mql5.com/go?link=http://msdn.microsoft.com/en-us/library/aa364232(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa364232(vs.85).aspx")
- Types of data for transfers to MQL [http://msdn.microsoft.com/en-us/library/aa383751%28VS.85%29.aspx](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/aa383751(vs.85).aspx "http://msdn.microsoft.com/en-us/library/aa383751(vs.85).aspx")

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/73](https://www.mql5.com/ru/articles/73)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/73.zip "Download all attachments in the single ZIP archive")

[internetlib.mqh](https://www.mql5.com/en/articles/download/73/internetlib.mqh "Download internetlib.mqh")(8.35 KB)

[metaarbitrage.mq5](https://www.mql5.com/en/articles/download/73/metaarbitrage.mq5 "Download metaarbitrage.mq5")(9.27 KB)

[metachat.mq5](https://www.mql5.com/en/articles/download/73/metachat.mq5 "Download metachat.mq5")(5.68 KB)

[metagrabber.mq5](https://www.mql5.com/en/articles/download/73/metagrabber.mq5 "Download metagrabber.mq5")(3.24 KB)

[metaarbitr.zip](https://www.mql5.com/en/articles/download/73/metaarbitr.zip "Download metaarbitr.zip")(0.72 KB)

[wininet.zip](https://www.mql5.com/en/articles/download/73/wininet.zip "Download wininet.zip")(12.81 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)
- [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)
- [Getting Rid of Self-Made DLLs](https://www.mql5.com/en/articles/364)
- [Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)
- [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276)
- [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)
- [The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1210)**
(53)


![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
1 Oct 2019 at 16:13

**Sergey Naumov:**

And how to call the script from the indicator? As I know it is also impossible by normal methods.

[https://www.mql5.com/ru/articles/5337](https://www.mql5.com/ru/articles/5337)

![Sergey Naumov](https://c.mql5.com/avatar/2015/10/561D484F-DDFD.jpg)

**[Sergey Naumov](https://www.mql5.com/en/users/wisess)**
\|
2 Oct 2019 at 09:44

Thank you.


![Oriol Carbonell Sole](https://c.mql5.com/avatar/2018/6/5B2F50FF-E92B.jpg)

**[Oriol Carbonell Sole](https://www.mql5.com/en/users/scalper-forex)**
\|
7 Jan 2020 at 12:48

Why don't my signals replicate?

![Alexandr Gavrilin](https://c.mql5.com/avatar/2025/12/694aad80-f58e.png)

**[Alexandr Gavrilin](https://www.mql5.com/en/users/dken)**
\|
25 Dec 2021 at 21:07

```
2021.12.26 01:06:05.840 metaarbitrage (EURUSD,H1)       +Open Inet...
2021.12.26 01:06:08.537 metaarbitrage (EURUSD,H1)       Access violation at 0x00007FF93AAE2740 read to 0xFFFFFFFFFFFFFFFF in 'wininet.dll'
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)          crash -->  00007FF93AAE2740 66837C410200      cmp        word [rcx+rax*2+0x2], 0x0
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)                     00007FF93AAE2746 488D4001          lea        rax, [rax+0x1]
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)                     00007FF93AAE274A 75F4              jnz        0x7ff93aae2740
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)                     00007FF93AAE274C 488B0D8D5D3D00    mov        rcx, [rip+0x3d5d8d]
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)                     00007FF93AAE2753 8D044502000000    lea        eax, [rax*2+0x2]
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)                     00007FF93AAE275A 448BC0            mov        r8d, eax
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)                     00007FF93AAE275D 33D2              xor        edx, edx
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)       00: 0x00007FF93AAE2740
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)       01: 0x000002C546534171
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)       02: 0x0000000000CC0038
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)       03: 0x000002C500000000
2021.12.26 01:06:08.539 metaarbitrage (EURUSD,H1)
```

This is what started coming up when I tried to check it.

![Alex FXPIP](https://c.mql5.com/avatar/2020/10/5F9DC28E-F0D0.PNG)

**[Alex FXPIP](https://www.mql5.com/en/users/rufxstrategy)**
\|
4 Oct 2023 at 15:45

This is the coolest thing ever! Especially for licensing advisors ... scrap [webrequest](https://www.mql5.com/en/docs/network/webrequest "MQL5 Documentation: WebRequest Function")... this thing I understand even in the tester will work! Wow


![An Example of a Trading System Based on a Heiken-Ashi Indicator](https://c.mql5.com/2/0/Heikin_Ashi_MQL5.png)[An Example of a Trading System Based on a Heiken-Ashi Indicator](https://www.mql5.com/en/articles/91)

In this article we look into the question of using a Heiken-Ashi indicator in trading. Based on this indicator, a simple trading system is considered and an MQL5 Expert Advisor is written. Trading operations are implemented on the bases of classes of the Standard class library. The testing results of the reviewed trading strategy, are based on the history, and obtained using the built-in MetaTrader 5 strategy tester, are provided in the article.

![Research of Statistical Recurrences of Candle Directions](https://c.mql5.com/2/17/890_32.gif)[Research of Statistical Recurrences of Candle Directions](https://www.mql5.com/en/articles/1576)

Is it possible to predict the behavior of the market for a short upcoming interval of time, based on the recurring tendencies of candle directions, at specific times throughout the day? That is, If such an occurrence is found in the first place. This question has probably arisen in the mind of every trader. The purpose of this article is to attempt to predict the behavior of the market, based on the statistical recurrences of candle directions during specific intervals of time.

![Guide to writing a DLL for MQL5 in Delphi](https://c.mql5.com/2/0/delphi_DLL_MQL5__1.png)[Guide to writing a DLL for MQL5 in Delphi](https://www.mql5.com/en/articles/96)

The article examines the mechanism of creating a DLL module, using the popular programming language of ObjectPascal, within a Delphi programming environment. The materials, provided in this article, are designed to primarily target beginner programmers, who are working with problems, which breach the boundaries of the embedded programming language of MQL5, by connecting the outside DLL modules.

![Creating a Multi-Currency Indicator, Using a Number of Intermediate Indicator Buffers](https://c.mql5.com/2/0/Multicurrency_Indicator_MQL5.png)[Creating a Multi-Currency Indicator, Using a Number of Intermediate Indicator Buffers](https://www.mql5.com/en/articles/83)

There has been a recent rise of interest in the cluster analyses of the FOREX market. MQL5 opens up new possibilities of researching the trends of the movement of currency pairs. A key feature of MQL5, differentiating it from MQL4, is the possibility of using an unlimited amount of indicator buffers. This article describes an example of the creation of a multi-currency indicator.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dlzexzazlsvsdxvevgimnczrgytrkuuy&ssn=1769252935294020323&ssn_dr=0&ssn_sr=0&fv_date=1769252935&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F73&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Using%20WinInet.dll%20for%20Data%20Exchange%20between%20Terminals%20via%20the%20Internet%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925293593238989&fz_uniq=5083366285070309959&sv=2552)

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